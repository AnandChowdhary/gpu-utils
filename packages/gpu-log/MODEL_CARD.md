# Model card: gpu-log

## Task
Universal log line parser: for every line of a text log, tag each token with one of the roles
TS, LEVEL, SOURCE, HOST, THREAD, KEY, VALUE, MSG, FN, FILE, LINE, COL (BIO scheme, 25 labels)
and classify the line as entry / continuation / frame. A deterministic compiler turns the tags
into `{ timestamp, level, source, host, thread, kv, message, frame, kind }`. JSON lines and
pure logfmt lines bypass the model.

## Architecture
- Model family: shared `ConvTagger` (`gpu_utils_training.models`), i.e. the runtime's
  `convTaggerForward` and `wgsl/conv_tagger.wgsl`; no package-specific kernel.
- Tokenizer: shared gpu-utils character-class runs; 9 sparse hashed features per token
  (word hash 1024, 3-char prefix hash 512, shape 8, first char 129, last char 129, length 16,
  position bucket 20, position-from-end bucket 12, column bucket 10; digits collapsed to `0`
  in the hashes and the character features). Single flat embedding table, 1,860 rows × 32.
- Embedding: 32 dims summed over the 9 slots, linear to 64.
- Sequence mixing: 5 residual blocks `x + relu(conv3(x, dilation d)) W2 + b2`,
  d = 1, 2, 4, 8, 16, hidden 64 (receptive field ±31 tokens). Each line is its own sequence
  (padded per batch), so context never crosses lines.
- Heads: tag head 64 → 64 → 25 (per token, constrained Viterbi on the CPU); pooled kind head
  mean(x) → 64 → 3.
- Parameters: 154,332 (embedding 59,520; projection 2,112; blocks 82,240; heads 10,460).
- Quantization: int6 symmetric per-tensor, quantization-aware from epoch 1 (straight-through),
  best epoch by held-out span F1.

## Training data
Synthetic only, `training/gpu_log/{gen,formats,traces,data}.py`, 160,000 lines, seed 1
(~7.6M tokens). ~57% entry lines drawn from 33 format builders (syslog RFC 3164/5424,
journald, nginx/apache access and error, log4j/logback in 10 layouts incl. Spring, Kafka,
Elasticsearch, ZooKeeper, HDFS, Tomcat, Python logging in 9 layouts incl. gunicorn, uvicorn,
celery, Go std log/glog/klog/zap/logrus/slog, Rust env_logger/tracing, Node pino/winston/
bunyan/nest/debug/npm, containerd CRI, kubectl --timestamps, docker compose, Heroku router
and app, CloudWatch/Lambda, PostgreSQL, MySQL, Redis, Serilog, PHP error log, logcat, dmesg,
.NET console, java.util.logging, ad-hoc "ts level source msg" layouts, plain lines, mixed
logfmt; **v2 adds** cluster/supercomputer RAS layouts (BGL-, HPC-, Thunderbird- and
Slurm-like), Windows CBS/CSI lines and Event Viewer text blocks, Proxifier, syslog wrapping a
structured app line, ten bracket/parenthesis wrappers for source and thread, multi-word
sources, kv tails with quoted and nested-brace values, and key=value prose inside messages),
the rest stack traces and continuations for Java, Python, JavaScript/Node (V8 and
Firefox shapes), Go, Rust, C#, Ruby, PHP plus generic dumps (JSON bodies, SQL, wrapped text,
separators, `Caused by:`). 46 timestamp spellings, 6 level classes in ~70 spellings/cases,
~300 message templates with slot fillers (ips, paths, urls, ids, durations, SQL, unicode),
kv tails in `=`, `: `, `[k=v]` and quoted styles, noise (trailing whitespace, indentation,
truncation, very long tails). Real logs are never used for training.

## Evaluation sets
1. **Held-out**: 12,000 lines from the same generator, seed 2.
2. **Unfamiliar v1** (frozen): `eval/unfamiliar-v1.jsonl`, the 104 hand-written v1 lines
   (Envoy, HAProxy, Varnish, Rails, CEF, Cisco, IIS, Squid, Postfix, BIND, Ansible, Terraform,
   git, cargo, pytest, Jenkins, macOS unified log, fail2ban, auditd, OpenVPN, Dovecot, ntpd,
   MongoDB, Cassandra, ClickHouse, RabbitMQ, Traefik, Nomad, CoreDNS, PowerShell, Windows Event
   text, and frames from gdb/C++, Dart, Elixir, Perl, Lua, Julia, OCaml, bash, Kotlin, plus
   BGL/HPC/Thunderbird/Windows/Proxifier-shaped cluster and Windows lines). **This set drove
   the v2 generator widening, so for v2 it is contaminated** and is kept only for continuity.
3. **Unfamiliar v2** (fresh): `training/data/unfamiliar_v2.txt`, 148 hand-written lines from
   layouts the v2 generator does not produce (SQL Server, Neo4j, memcached, telegraf, Nagios,
   Zabbix, Chef, Salt, VirtualBox, libvirt, Xorg, wpa_supplicant, CUPS, Samba, apt history,
   and more). Written before any v2 evaluation, never tuned on.
4. **Real-world** (Loghub): 16,000 real log lines, 1,000 per system for 16 systems, whose gold
   spans are derived mechanically from Loghub's own `*_structured.csv` columns
   (`training/gpu_log/loghub.py`), not from our generator. Where Loghub labels no field — the
   `sshd` between host and pid in OpenSSH, BGL's leading epoch, OpenStack's composite request
   blob — the region is *unlabelled*: its tokens are excluded from token accuracy and
   predictions inside it are dropped rather than counted as false positives. Downloaded at
   evaluation time; research license; evaluation only, never training (THIRD_PARTY_NOTICES.md).
   Nine of the 16 layouts are imitated by the generator; HealthApp and OpenStack are not, and
   Loghub's own field split is idiosyncratic for a few systems (Zookeeper splits the thread
   bracket oddly, HPC calls a bare epoch the timestamp), which bounds the achievable score.

## Evaluation
Span F1 is exact match over role spans (micro-averaged); "line exact" requires every token
tag *and* the line kind to be right. All numbers are the int6 (quantized) model.

### v2, all 12 roles

| Set | Lines | Token acc | Span F1 | Kind acc | Line exact |
|---|---|---|---|---|---|
| held-out (generated, seed 2) | 12,000 | 0.9623 | 0.9957 | 0.9955 | 0.9775 |
| unfamiliar v1 (frozen, **contaminated**) | 104 | 0.7241 | 0.5730 | 0.8846 | 0.2115 |
| unfamiliar v2 (hand-written, fresh) | 148 | 0.7614 | 0.5732 | 0.8986 | 0.1824 |
| real-world (Loghub, real lines) | 16,000 | 0.8879 | 0.7657 | 1.0000 | 0.4691 |

### v1 vs v2 on the same sets

v1 has no HOST role, so both models are scored with HOST folded into O ("common roles"), by
the same code (`evaluate.py --baseline`, v1 rebuilt from its int6 artefacts on `main`). The
held-out set is the v2 generator's, which is why v1 scores lower there than in its own card.

| Set | Lines | Token acc v1 → v2 | Span F1 v1 → v2 | Kind acc v1 → v2 | Line exact v1 → v2 |
|---|---|---|---|---|---|
| held-out (generated, seed 2) | 12,000 | 0.9343 → 0.9623 | 0.9394 → 0.9956 | 0.9837 → 0.9955 | 0.8390 → 0.9775 |
| unfamiliar v1 (frozen, contaminated) | 104 | 0.7082 → 0.7552 | 0.5095 → 0.5894 | 0.8077 → 0.8846 | 0.1346 → 0.2885 |
| unfamiliar v2 (hand-written, fresh) | 148 | 0.7719 → 0.7716 | 0.5634 → 0.5762 | 0.9257 → 0.8986 | 0.1892 → 0.1824 |
| real-world (Loghub, real lines) | 16,000 | 0.8079 → 0.8888 | 0.5899 → 0.7468 | 0.9978 → 1.0000 | 0.2119 → 0.4696 |

The frozen v1 set is contaminated for v2 (it is what the widening was aimed at) and is listed
only for continuity. The fresh hand-written set is flat: adversarial one-off layouts are not
what more generator coverage buys. The real-world set is where v2 pays off.

### Per role, real-world set (v2, all roles)

| Role | P | R | F1 | Gold spans |
|---|---|---|---|---|
| TS | 0.923 | 0.998 | 0.959 | 16,000 |
| LEVEL | 0.942 | 1.000 | 0.970 | 9,000 |
| SOURCE | 0.634 | 0.837 | 0.721 | 14,000 |
| HOST | 0.968 | 0.918 | 0.943 | 8,000 |
| THREAD | 0.752 | 0.735 | 0.743 | 10,936 |
| MSG | 0.582 | 0.616 | 0.598 | 16,000 |

(The real-world gold has no KEY/VALUE/FN/FILE/LINE/COL spans: Loghub's columns do not label
them, so those roles are only measured on the generated and hand-written sets.)

### Per system, real-world set (common roles)

| System | Lines | Layout | Span F1 v1 → v2 | Line exact v1 → v2 |
|---|---|---|---|---|
| **layouts the generator imitates** | 14,000 | — | 0.5991 → 0.7789 | 0.2421 → 0.5336 |
| **unseen layouts** | 2,000 | — | 0.5287 → 0.5569 | 0.0000 → 0.0215 |
| Android | 1,000 | imitated | 0.6366 → 0.7772 | 0.3090 → 0.5860 |
| Apache | 1,000 | imitated | 0.7256 → 0.7256 | 0.4020 → 0.4020 |
| BGL | 1,000 | imitated | 0.4536 → 0.9538 | 0.0000 → 0.8730 |
| HDFS | 1,000 | imitated | 0.9435 → 0.9723 | 0.8520 → 0.8620 |
| HPC | 1,000 | imitated | 0.1657 → 0.6366 | 0.0020 → 0.3000 |
| Hadoop | 1,000 | imitated | 0.6481 → 0.6732 | 0.0310 → 0.0630 |
| HealthApp | 1,000 | **unseen** | 0.2870 → 0.3948 | 0.0000 → 0.0000 |
| Linux | 1,000 | imitated | 0.7834 → 0.7858 | 0.4580 → 0.4290 |
| Mac | 1,000 | imitated | 0.6023 → 0.6062 | 0.1420 → 0.1480 |
| OpenSSH | 1,000 | imitated | 0.7160 → 0.7618 | 0.4060 → 0.5710 |
| OpenStack | 1,000 | **unseen** | 0.6802 → 0.6821 | 0.0000 → 0.0430 |
| Proxifier | 1,000 | imitated | 0.2680 → 0.9085 | 0.0000 → 0.7420 |
| Spark | 1,000 | imitated | 0.8275 → 0.8205 | 0.7860 → 0.7530 |
| Thunderbird | 1,000 | imitated | 0.5083 → 0.9128 | 0.0000 → 0.8540 |
| Windows | 1,000 | imitated | 0.3960 → 0.9486 | 0.0000 → 0.8820 |
| Zookeeper | 1,000 | imitated | 0.5122 → 0.5252 | 0.0020 → 0.0060 |

**Read this honestly.** The lines are real and neither model has seen them, but the v2
generator *was* widened toward the shapes of BGL, HPC, Thunderbird, Windows and Proxifier
(via the v1 hand-written set), so the big jumps on those five systems measure layout coverage,
not generalisation. The generalisation signal is the unseen-layout row: +2.8 points of span F1
and +2 points of line exact, small but real. Zookeeper and Hadoop stay low partly because
Loghub's own column split there disagrees with any consistent convention (it splits the thread
bracket mid-token, and Hadoop's Content keeps a leading id).

## Size and latency
| Measure | Value |
|---|---|
| Package (min + Brotli, incl. weights) | 101.8 KiB (104,244 B) of the 120,000 B budget |
| Parameters | 154,332 int6 |
| CPU path, 10 MB / 113K lines, memoisation on | 28.09 s → 0.36 MB/s, 4K lines/s |
| CPU path, 1 MB, memoisation off (raw model throughput) | 49.29 s → 0.020 MB/s, 229 lines/s |
| Fast paths only (JSON) | 21.3 MB/s |
| WebGPU, design estimate | ~0.4 MB/s end to end (~25 s per 10 MB): the model itself drops from ~8 min to ~1–2 s, after which the CPU tokenizer/featurizer/Viterbi pass dominates |
| Cold start (device + pipelines + 600 KB weight upload), estimate | ~40–80 ms on an integrated GPU |
| Warm call, 1 KB input (~15 lines, 600 tokens) | CPU ~4 ms; GPU ~2–3 ms, dominated by readback, so "auto" uses the CPU below 256 tokens |

Numbers from `node packages/gpu-log/bench/throughput.mjs 10` on one core of this box (Node 24).
The benchmark file has 20 distinct templates, so with memoisation on nearly every model line is
a cache hit and the 28 s is the per-line CPU work: tokenizing, hashing, the memo key, Viterbi
and the compiler. v1 did that in 23 s; the difference is the Viterbi pass, which is quadratic
in the label count (25 labels now, 23 before).

WebGPU cannot run through the browser runtime on this machine, so the GPU column is a design
estimate; the canonical `conv_tagger.wgsl` is executed on a Mesa lavapipe adapter through
wgpu-py in `training/tests/test_wgsl.py` and matches the fixtures to 3.8e-05. Per token the
model costs ~92K MACs; a 10 MB file is ~4.5M model tokens → ~0.4 TMAC, ~1–2 s on an integrated
GPU across dispatches of up to 131,072 padded tokens. **Known bottleneck for the shared-infra
pass:** the runtime tokenizer (regex per character) and `hashToken` (`TextEncoder` per token,
twice per token here) dominate the CPU pre-pass that then bounds the GPU path.

## Limitations and intended use
Structuring logs for viewers, search and grouping. Not a substitute for a format-specific
parser when the format is known: a JSON or logfmt pipeline should keep its own parser (those
lines take the deterministic paths anyway).

What the model still cannot do, measured on the real-world set:

- **Message boundaries are the weakest role.** MSG precision/recall trail every other role:
  the model often starts the message one token early or late when an unfamiliar prefix (an id,
  a `[client x]` bracket, a duplicated node name) sits between the header and the text.
- **SOURCE vs HOST vs THREAD in unfamiliar headers.** When a layout puts an unseen field in
  the position the generator uses for a logger, the field is usually labelled SOURCE.
- **One line at a time.** Nothing carries across lines, so an unindented continuation of a
  previous entry is tagged as its own entry.
- **Unseen layouts cost about 5 points of span F1** relative to layouts the generator imitates
  (see the seen/unseen split above), and hand-written unfamiliar layouts cost far more: those
  sets are adversarial by construction, one line per format, with no repeated templates.
- `iso` requires a year in the text; kv pairs inside prose are left in the message; frame
  `language` is a heuristic over the frame's shape and file extension.

## Checkpoint
- Promoted: 2026-09-18, `training/runs/default/best.pt`, seed 1, epoch 4 of 5 over 160,000
  generated lines (best held-out span F1), QAT from epoch 1, batch 48, AdamW lr 2e-3 with
  warm-up + cosine, 2 CPU threads, 644 s (10.7 min) — inside the 30-minute CPU budget.
- Training command: `pnpm train` then `pnpm export` (checkpoints are gitignored).
- Fixtures: `model/fixtures.json`, 31 canonical cases (`input`, `rows`, `logits`,
  `pooled`) from the decoded int6 weights; `test/parity.test.ts` asserts the CPU path matches at
  1e-4 and `training/tests/test_wgsl.py` the WGSL kernel.
