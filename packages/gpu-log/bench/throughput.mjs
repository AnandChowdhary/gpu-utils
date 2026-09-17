/**
 * Throughput benchmark: synthesises a ~10 MB log file from a fixed mix of real-looking lines and
 * times `parse()` on the CPU reference path (and on WebGPU when a device exists).
 *
 *   pnpm --filter gpu-log build && node packages/gpu-log/bench/throughput.mjs [mb]
 */
import { parse } from "../dist/index.js";

const MB = Number(process.argv[2] ?? 10);
const lines = [
  "2024-01-15 10:30:00,123 [http-nio-8080-exec-7] INFO  com.example.api.UserController - Handled GET /api/v1/users/123 from 10.0.0.9 in 12ms",
  "Jan 15 10:30:01 web-01 sshd[1234]: Accepted publickey for alice from 10.0.0.9 port 51234 ssh2: RSA SHA256:abcdef",
  '10.0.0.9 - - [15/Jan/2024:10:30:01 +0000] "GET /index.html HTTP/1.1" 200 5123 "https://example.com/" "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"',
  '2024/01/15 10:30:02 [error] 1234#1234: *5 connect() failed (111: Connection refused) while connecting to upstream, client: 10.0.0.9, server: example.com, request: "GET /api HTTP/1.1", upstream: "http://10.0.0.5:8080/api"',
  '{"level":30,"time":1705314602123,"pid":42,"hostname":"web-01","reqId":"req-1","msg":"request completed","responseTime":12}',
  'time=2024-01-15T10:30:02.456Z level=INFO msg="cache warmed" keys=1234 duration=45ms',
  "I0115 10:30:02.789012    1 controller.go:123] Reconciling default/api-7d9f8b6c4-x2k9p",
  '2024-01-15T10:30:03.000000Z  WARN request{id=7}: app::server: slow query took 1.2s query="SELECT * FROM users"',
  "[2024-01-15T10:30:03.100Z] ERROR (api/42 on web-01): Unhandled exception while processing request req-2",
  "TypeError: Cannot read properties of undefined (reading 'id')",
  "    at fetchUser (/app/src/users.js:10:15)",
  "    at processTicksAndRejections (node:internal/process/task_queues:95:5)",
  '2024-01-15T10:30:03.200Z\tINFO\tserver/http.go:88\tlistening\t{"addr": ":8080"}',
  "ERROR:django.request:Internal Server Error: /api/orders",
  "Traceback (most recent call last):",
  '  File "/app/main.py", line 12, in <module>',
  "    main()",
  "ValueError: invalid literal for int() with base 10: 'x'",
  "\tat com.example.service.UserService.save(UserService.java:88)",
  '2024-01-15T10:30:04.123456+00:00 heroku[router]: at=info method=GET path="/" host=app.herokuapp.com request_id=abc fwd="10.0.0.9" dyno=web.1 connect=1ms service=12ms status=200 bytes=1234 protocol=https',
];
let text = "";
while (text.length < MB * 1024 * 1024) text += `${lines.join("\n")}\n`;
const bytes = Buffer.byteLength(text);
console.log(`input: ${(bytes / 1048576).toFixed(1)} MB, ${text.split("\n").length - 1} lines`);

for (const backend of ["cpu", "webgpu"]) {
  try {
    const t0 = performance.now();
    const r = await parse(text, { backend });
    const ms = performance.now() - t0;
    const tokens = r.lines.length;
    console.log(
      `${backend}: ${(ms / 1000).toFixed(2)} s -> ${(bytes / 1048576 / (ms / 1000)).toFixed(2)} MB/s, ${(tokens / (ms / 1000) / 1000).toFixed(0)}K lines/s; paths: json ${r.stats.json}, logfmt ${r.stats.logfmt}, model ${r.stats.model}`,
    );
  } catch (e) {
    console.log(`${backend}: unavailable (${e.name}: ${e.message})`);
  }
}

// Where does CPU time go? Time the model-free part by disabling the model lines.
const t1 = performance.now();
const jsonOnly = lines
  .filter((l) => l.startsWith("{"))
  .join("\n")
  .repeat(Math.ceil((MB * 1048576) / 800));
const r2 = await parse(jsonOnly, { backend: "cpu" });
console.log(
  `json fast path only: ${(Buffer.byteLength(jsonOnly) / 1048576 / ((performance.now() - t1) / 1000)).toFixed(1)} MB/s over ${r2.lines.length} lines`,
);
