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
Unfamiliar set: `training/data/unfamiliar.txt`, 104 hand-written lines from formats the
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
| held-out (generated, seed 2) | 12,000 | 0.9587 | 0.9955 | 0.9977 | 0.9782 |
| unfamiliar (hand-written) | 104 | 0.7082 | 0.5095 | 0.8077 | 0.1346 |

Per-role span F1 on the unfamiliar set:

| Role | P | R | F1 | Spans |
|---|---|---|---|---|
| TS | 0.691 | 0.918 | 0.789 | 61 |
| LEVEL | 0.604 | 0.707 | 0.652 | 41 |
| SOURCE | 0.333 | 0.444 | 0.381 | 54 |
| THREAD | 0.625 | 0.694 | 0.658 | 36 |
| KEY | 0.400 | 0.389 | 0.394 | 36 |
| VALUE | 0.324 | 0.324 | 0.324 | 37 |
| MSG | 0.225 | 0.403 | 0.289 | 72 |
| FN | 0.667 | 0.500 | 0.571 | 16 |
| FILE | 0.800 | 0.500 | 0.615 | 24 |
| LINE | 0.938 | 0.625 | 0.750 | 24 |
| COL | 1.000 | 0.800 | 0.889 | 5 |

Loghub 2k samples, first 300 lines per system, weak labels (level normalised to the enum;
source counts if the predicted SOURCE contains or is contained in the CSV Component; message
counts if the predicted MSG equals or prefixes the CSV Content):

| System | Level | Source | Message |
|---|---|---|---|
| HDFS | 300/300 (100%) | 300/300 (100%) | 262/300 (87%) |
| Hadoop | 300/300 (100%) | 300/300 (100%) | 57/300 (19%) |
| Spark | 300/300 (100%) | 299/300 (100%) | 292/300 (97%) |
| Zookeeper | 300/300 (100%) | 300/300 (100%) | 300/300 (100%) |
| OpenSSH | n/a | 0/300 (0%) | 197/300 (66%) |
| Linux | 0/300 (0%) | 300/300 (100%) | 299/300 (100%) |
| Mac | n/a | 292/300 (97%) | 105/300 (35%) |
| Apache | 300/300 (100%) | n/a | 181/300 (60%) |
| Android | 300/300 (100%) | 273/300 (91%) | 230/300 (77%) |
| BGL | 0/300 (0%) | 2/300 (1%) | 2/300 (1%) |
| HPC | n/a | 71/300 (24%) | 20/300 (7%) |
| Thunderbird | n/a | 147/300 (49%) | 61/300 (20%) |
| Windows | 300/300 (100%) | 85/300 (28%) | 4/300 (1%) |
| HealthApp | n/a | 300/300 (100%) | 133/300 (44%) |
| Proxifier | n/a | n/a | 0/300 (0%) |
| OpenStack | 300/300 (100%) | 300/300 (100%) | 71/300 (24%) |

Systems whose layout is in the generator: HDFS, Hadoop, Spark, Zookeeper, OpenSSH, Linux, Mac
(syslog), Apache, Android. The others (BGL, HPC, Thunderbird, Windows, HealthApp, Proxifier,
OpenStack) are unseen layouts.

## Size and latency
| Measure | Value |
|---|---|
| Package (min + Brotli, incl. weights) | 97.5 KiB (99,840 B) (budget 120,000 B) |
| CPU path, 10 MB / 113K lines, memoisation off | 0.018 MB/s (208 lines/s; 54.39 s per MB, ~9 min per 10 MB) |
| CPU path, 10 MB / 113K lines, memoisation on | 0.43 MB/s (23.09 s; 5K lines/s; the bench file has 20 distinct templates, so nearly every model line is memoised) |
| Fast paths only (JSON) | 20.8 MB/s |
| WebGPU, design estimate | ~0.4 MB/s end to end (~25 s per 10 MB): the model itself drops from ~9 min to ~1–2 s, after which the CPU tokenizer/featurizer/Viterbi pass (~23 s per 10 MB on one core) dominates |
| Cold start (device + 7 pipelines + 600 KB weight upload), estimate | ~40–80 ms on an integrated GPU |
| Warm call, 1 KB input (~15 lines, 600 tokens) | CPU ~4 ms; GPU ~2–3 ms, dominated by readback, so "auto" uses the CPU below 256 tokens |

WebGPU could not be executed through the browser runtime on this machine (Node has no adapter
and Dawn's Node bindings fail to load), so the GPU column is a design estimate; the WGSL
itself is executed and checked against the CPU reference on a lavapipe (Mesa) adapter through
wgpu-py in `training/tests/test_wgsl.py`. Per token the model costs ~90K MACs (embedding 2K,
blocks 82K, heads 6K); a 10 MB file is ~4.5M model tokens → ~0.4 TMAC, which an integrated
GPU sustaining 0.5–1 TFLOPS on these small kernels finishes in ~1–2 s across ~35 dispatches of
131,072 tokens (13.6 MB readback each). The rest of the pipeline stays on the CPU: the
memoised benchmark run, where the model is almost free, still needs 23 s per 10 MB for
tokenising, hashing, packing and Viterbi on one core, so that is the end-to-end bound for the
GPU path until the pre-pass is optimised (it is ~4% of the un-memoised CPU-path time). The
shader runs seven passes per dispatch (embed, five blocks ping-ponging two state buffers,
heads) with one workgroup of 64 threads per token and the three tap vectors staged in
workgroup memory; `bench/gpu-parity.mjs` builds a browser page that checks it against the CPU
path on the fixtures.

## Limitations and intended use
Structuring logs for viewers, search and grouping. Not a substitute for a format-specific
parser when the format is known: a JSON or logfmt pipeline should keep its own parser (those
lines take the deterministic paths anyway). Outputs are probabilistic for everything else;
per-line context only; hostnames are not extracted; `iso` requires a year in the text; kv
pairs inside prose are left in the message; frame `language` is a heuristic over the frame's
shape and file extension.

## Checkpoint
- Promoted: 2026-09-18, seed 1, 5 epochs over 160,000 lines, QAT from epoch 2, batch 48,
  AdamW lr 2e-3 one-cycle, 2 CPU threads, 658 s.
- Training command: `pnpm train` then `pnpm export` (`training/runs/latest.pt`, gitignored).
- Fixtures: `model/fixtures.json`, 27 cases with feature rows and logits from the
  dequantized int6 tensors; `test/parity.test.ts` asserts the CPU path matches at 1e-4.
