# Model card: gpu-log

## Task
Universal log line parser: for every line of a text log, tag each token with one of the roles
TS, LEVEL, SOURCE, THREAD, KEY, VALUE, MSG, FN, FILE, LINE, COL (BIO scheme, 23 labels) and
classify the line as entry / continuation / frame. A deterministic compiler turns the tags into
`{ timestamp, level, source, thread, kv, message, frame, kind }`. JSON lines and pure logfmt
lines bypass the model.

## Architecture
- Tokenizer: shared gpu-utils character-class runs; 9 sparse hashed features per token
  (word hash 1024, 3-char prefix hash 512, shape 8, first char 129, last char 129, length 16,
  position bucket 20, position-from-end bucket 12, column bucket 10; digits collapsed to `0`
  in the hashes and the character features). Single flat embedding table, 1,860 rows × 32.
- Embedding: 32 dims summed over the 9 features, linear to 64.
- Sequence mixing: 5 residual blocks `x + W2·relu(conv3(x, dilation d) + b1) + b2`,
  d = 1, 2, 4, 8, 16, hidden 64 (receptive field ±31 tokens). Neighbours outside the line
  read as zero, which lets many lines share one flat sequence per dispatch.
- Heads: tag head 64 → 64 → 23 (per token, Viterbi with BIO constraints on the CPU); kind
  head 64 → 3 per token, mean-pooled over the line.
- Parameters: 150,042 (embedding 59,520; blocks 82,240; heads 5,975; projection 2,112).
- Quantization: int6 symmetric per-tensor, quantization-aware from epoch 2 (straight-through).

## Training data
Synthetic only, `training/gpu_log/{gen,formats,traces,data}.py`, 160,000 lines, seed 1
(~7.6M tokens). ~57% entry lines drawn from 26 format builders (syslog RFC 3164/5424,
journald, nginx/apache access and error, log4j/logback in 10 layouts incl. Spring, Kafka,
Elasticsearch, ZooKeeper, HDFS, Tomcat, Python logging in 9 layouts incl. gunicorn, uvicorn,
celery, Go std log/glog/klog/zap/logrus/slog, Rust env_logger/tracing, Node pino/winston/
bunyan/nest/debug/npm, containerd CRI, kubectl --timestamps, docker compose, Heroku router
and app, CloudWatch/Lambda, PostgreSQL, MySQL, Redis, Serilog, PHP error log, logcat, dmesg,
.NET console, java.util.logging, ad-hoc "ts level source msg" layouts, plain lines, mixed
logfmt), the rest stack traces and continuations for Java, Python, JavaScript/Node (V8 and
Firefox shapes), Go, Rust, C#, Ruby, PHP plus generic dumps (JSON bodies, SQL, wrapped text,
separators, `Caused by:`). 46 timestamp spellings, 6 level classes in ~70 spellings/cases,
~300 message templates with slot fillers (ips, paths, urls, ids, durations, SQL, unicode),
kv tails in `=`, `: `, `[k=v]` and quoted styles, noise (trailing whitespace, indentation,
truncation, very long tails). Real logs are never used for training.

Held-out set: 12,000 lines from the same generator with seed 2.
Unfamiliar set: `training/data/unfamiliar.txt`, __UNF_N__ hand-written lines from formats the
generator cannot produce (Loghub BGL/HPC/Thunderbird/Windows/HealthApp/Proxifier/OpenStack
layouts, Envoy, HAProxy, Varnish, Rails, pm2, CEF, Cisco, IIS, Squid, Postfix, BIND, Ansible,
Terraform, git, ffmpeg, cargo, pytest, jest, Jenkins, GitHub Actions, macOS unified log,
Chrome, Minecraft, Unity, fail2ban, UFW, auditd, OpenVPN, Dovecot, Exim, dnsmasq, ntpd,
MongoDB, Cassandra, ClickHouse, RabbitMQ, Jetty, WildFly, Traefik, Nomad, Vault-in-syslog,
CoreDNS, PowerShell, Windows Event text, and frames from gdb/C++, Dart, Elixir, Perl, Lua,
Julia, OCaml, bash, Ruby 3.4, Kotlin). Not used for any tuning.
Real logs: Loghub 2k samples (16 systems, research license, evaluation only, downloaded at
run time) scored against the `*_structured.csv` Level / Component / Content columns.

## Evaluation
Span F1 is exact-match over role spans (micro-averaged); "line exact" requires every token
tag and the kind to be right.

| Set | Lines | Token acc | Span F1 | Kind acc | Line exact |
|---|---|---|---|---|---|
| held-out (generated, seed 2) | 12,000 | __HO_TOK__ | __HO_F1__ | __HO_KIND__ | __HO_LINE__ |
| unfamiliar (hand-written) | __UNF_N__ | __UNF_TOK__ | __UNF_F1__ | __UNF_KIND__ | __UNF_LINE__ |

Per-role span F1 on the unfamiliar set:

__UNF_ROLES__

Loghub 2k samples, first 300 lines per system, weak labels (level normalised to the enum;
source counts if the predicted SOURCE contains or is contained in the CSV Component; message
counts if the predicted MSG equals or prefixes the CSV Content):

__LOGHUB__

Systems whose layout is in the generator: HDFS, Hadoop, Spark, Zookeeper, OpenSSH, Linux, Mac
(syslog), Apache, Android. The others (BGL, HPC, Thunderbird, Windows, HealthApp, Proxifier,
OpenStack) are unseen layouts.

## Size and latency
| Measure | Value |
|---|---|
| Package (min + Brotli, incl. weights) | __SIZE__ (budget 120,000 B) |
| CPU path, 10 MB / 113K lines, memoisation off | __CPU_RAW__ |
| CPU path, 10 MB / 113K lines, memoisation on | __CPU_MEMO__ |
| Fast paths only (JSON) | __FAST__ |
| WebGPU, design estimate | __GPU__ |
| Cold start (device + 7 pipelines + 600 KB weight upload), estimate | ~40–80 ms on an integrated GPU |
| Warm call, 1 KB input (~15 lines, 600 tokens) | CPU ~4 ms; GPU ~2–3 ms, dominated by readback, so "auto" uses the CPU below 256 tokens |

WebGPU could not be executed on this machine (no adapter under Node; Dawn's Node bindings
fail to load), so the GPU column is a design estimate: per token the model costs ~90K MACs
(embedding 2K, blocks 82K, heads 6K); a 10 MB file is ~4.5M model tokens → ~0.4 TMAC, which
an integrated GPU sustaining 0.5–1 TFLOPS on these small kernels finishes in ~1–2 s plus
~35 dispatches of 131,072 tokens each with a 13.6 MB readback per dispatch. The
tokenizer/featurizer and Viterbi stay on the CPU (~__CPU_PRE__ of the CPU-path time), so the
end-to-end GPU estimate is ~__GPU__. The shader runs seven passes per dispatch (embed, five
blocks ping-ponging two state buffers, heads) with one workgroup of 64 threads per token and
the three tap vectors staged in workgroup memory; `bench/gpu-parity.mjs` builds a browser page
that checks it against the CPU path on the fixtures.

## Limitations and intended use
Structuring logs for viewers, search and grouping. Not a substitute for a format-specific
parser when the format is known: a JSON or logfmt pipeline should keep its own parser (those
lines take the deterministic paths anyway). Outputs are probabilistic for everything else;
per-line context only; hostnames are not extracted; `iso` requires a year in the text; kv
pairs inside prose are left in the message; frame `language` is a heuristic over the frame's
shape and file extension.

## Checkpoint
- Promoted: 2026-09-18, seed 1, 5 epochs over 160,000 lines, QAT from epoch 2, batch 48,
  AdamW lr 2e-3 one-cycle, 2 CPU threads, __TRAIN_TIME__.
- Training command: `pnpm train` then `pnpm export` (`training/runs/latest.pt`, gitignored).
- Fixtures: `model/fixtures.json`, __FIX_N__ cases with feature rows and logits from the
  dequantized int6 tensors; `test/parity.test.ts` asserts the CPU path matches at 1e-4.
