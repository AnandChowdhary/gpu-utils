"""Building blocks for the synthetic generator: line builder, timestamps, levels, messages, kv."""

from __future__ import annotations

import random
import string
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from . import vocab as V

ROLES = ["TS", "LEVEL", "SOURCE", "THREAD", "KEY", "VALUE", "MSG", "FN", "FILE", "LINE", "COL"]
KINDS = ["entry", "continuation", "frame"]


@dataclass
class Line:
    kind: str
    parts: list[tuple[str, str | None]] = field(default_factory=list)

    def add(self, text: str, role: str | None = None) -> Line:
        if text:
            self.parts.append((text, role))
        return self

    def text(self) -> str:
        return "".join(t for t, _ in self.parts)

    def spans(self) -> list[tuple[int, int, str]]:
        out: list[tuple[int, int, str]] = []
        pos = 0
        for t, role in self.parts:
            if role is not None:
                # merge with an adjacent span of the same role only when explicitly continued
                out.append((pos, pos + len(t), role))
            pos += len(t)
        return out


def markup(line: Line) -> str:
    """Inline annotation format, also used for the hand-written eval sets: ⟦ROLE|text⟧."""
    return "".join(f"⟦{r}|{t}⟧" if r else t for t, r in line.parts) + f" ##{line.kind}"


MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
TZ_NAMES = ["UTC", "GMT", "CET", "CEST", "EST", "PST", "PDT", "JST", "IST", "Z"]


def rand_dt(rng: random.Random) -> datetime:
    base = datetime(2015, 1, 1)
    return base + timedelta(seconds=rng.randint(0, 11 * 365 * 86400), microseconds=rng.randint(0, 999999))


def tz_offset(rng: random.Random, colon: bool = True) -> str:
    if rng.random() < 0.5:
        return "+00:00" if colon else "+0000"
    sign = rng.choice("+-")
    h = rng.choice([1, 2, 3, 4, 5, 5, 8, 9, 10])
    m = rng.choice(["00", "00", "00", "30"])
    return f"{sign}{h:02d}:{m}" if colon else f"{sign}{h:02d}{m}"


def timestamp(rng: random.Random, style: int | None = None) -> str:
    """One of ~45 real-world timestamp spellings."""
    d = rand_dt(rng)
    ms = f"{d.microsecond // 1000:03d}"
    us = f"{d.microsecond:06d}"
    ns = us + f"{rng.randint(0, 999):03d}"
    hms = d.strftime("%H:%M:%S")
    ymd = d.strftime("%Y-%m-%d")
    if style is None:
        style = rng.randrange(46)
    mon = MONTHS[d.month - 1]
    dow = DAYS[d.weekday()]
    ampm = d.strftime("%I:%M:%S %p")
    match style:
        case 0:
            return f"{ymd}T{hms}Z"
        case 1:
            return f"{ymd}T{hms}.{ms}Z"
        case 2:
            return f"{ymd}T{hms}.{us}Z"
        case 3:
            return f"{ymd}T{hms}.{ns}Z"
        case 4:
            return f"{ymd}T{hms}{tz_offset(rng)}"
        case 5:
            return f"{ymd}T{hms}.{ms}{tz_offset(rng)}"
        case 6:
            return f"{ymd}T{hms}.{us}{tz_offset(rng, rng.random() < 0.5)}"
        case 7:
            return f"{ymd} {hms}"
        case 8:
            return f"{ymd} {hms},{ms}"
        case 9:
            return f"{ymd} {hms}.{ms}"
        case 10:
            return f"{ymd} {hms}.{us}"
        case 11:
            return f"{ymd} {hms} {tz_offset(rng, False)}"
        case 12:
            return f"{ymd} {hms}.{ms} {rng.choice(TZ_NAMES[:5])}"
        case 13:
            return d.strftime("%Y/%m/%d ") + hms
        case 14:
            return d.strftime("%Y/%m/%d ") + f"{hms}.{us}"
        case 15:
            return f"{mon} {d.day:2d} {hms}"
        case 16:
            return f"{mon} {d.day} {hms}"
        case 17:
            return f"{d.day:02d}/{mon}/{d.year}:{hms} {tz_offset(rng, False)}"
        case 18:
            return f"{dow} {mon} {d.day:02d} {hms} {d.year}"
        case 19:
            return f"{dow} {mon} {d.day:02d} {hms}.{us} {d.year}"
        case 20:
            return f"{dow} {ymd} {hms} {rng.choice(['UTC', 'CET', 'EST'])}"
        case 21:
            return f"{d.day:02d}-{mon}-{d.year} {hms}.{ms}"
        case 22:
            return d.strftime("%m/%d/%Y ") + hms
        case 23:
            return d.strftime("%m/%d/%Y ") + ampm
        case 24:
            return d.strftime("%d.%m.%Y ") + hms
        case 25:
            return d.strftime("%y/%m/%d ") + hms
        case 26:
            return d.strftime("%m%d ") + f"{hms}.{us}"
        case 27:
            return d.strftime("%m-%d ") + f"{hms}.{ms}"
        case 28:
            return hms
        case 29:
            return f"{hms}.{ms}"
        case 30:
            return str(int(d.timestamp()))
        case 31:
            return f"{int(d.timestamp())}.{ms}"
        case 32:
            return str(int(d.timestamp()) * 1000 + int(ms))
        case 33:
            return d.strftime("%Y%m%d ") + hms
        case 34:
            return d.strftime("%Y%m%dT%H%M%SZ")
        case 35:
            return f"{d.day:02d} {mon} {d.year} {hms}.{ms}"
        case 36:
            return f"{dow}, {d.day:02d} {mon} {d.year} {hms} GMT"
        case 37:
            return f"{ymd}T{hms}.{ms}{tz_offset(rng)}"
        case 38:
            return f"{mon} {d.day}, {d.year} {ampm}"
        case 39:
            return f"{ymd} {hms}.{ms} {tz_offset(rng)}"
        case 40:
            return d.strftime("%y%m%d ") + hms
        case 41:
            return f"{rng.randint(0, 99999):5d}.{us}"  # kernel uptime
        case 42:
            return f"{ymd}T{hms}.{us}"
        case 43:
            return f"{ymd} {hms}.{ms}{tz_offset(rng)}"
        case 44:
            return f"{d.strftime('%Y%m%d')}-{hms}:{ms}"
        case _:
            return f"{ymd}-{hms.replace(':', '.')}.{us}"


LEVELS = {
    "trace": ["TRACE", "trace", "Trace", "VERBOSE", "verbose", "FINEST", "V", "T", "VRB", "vrb"],
    "debug": ["DEBUG", "debug", "Debug", "DBG", "dbg", "FINE", "FINER", "CONFIG", "D", "DEBU"],
    "info": ["INFO", "INFO", "INFO", "info", "info", "Info", "INF", "inf", "NOTICE", "notice", "INFORMATION", "Information", "I", "N"],
    "warn": ["WARN", "WARN", "WARNING", "WARNING", "warn", "warning", "Warn", "Warning", "WRN", "wrn", "W"],
    "error": ["ERROR", "ERROR", "error", "error", "Error", "ERR", "err", "SEVERE", "E", "ERRO", "CRITICAL", "critical", "Critical", "CRIT", "crit"],
    "fatal": ["FATAL", "fatal", "Fatal", "FTL", "EMERG", "emerg", "EMERGENCY", "ALERT", "alert", "PANIC", "panic", "F"],
}
LEVEL_NAMES = list(LEVELS)


def level_word(rng: random.Random) -> tuple[str, str]:
    name = rng.choices(LEVEL_NAMES, weights=[3, 10, 40, 18, 22, 4])[0]
    return rng.choice(LEVELS[name]), name


def pad_level(rng: random.Random, word: str) -> tuple[str, str]:
    """Returns (word, trailing padding) for %-5level-style alignment."""
    if rng.random() < 0.35:
        return word, " " * max(0, 5 - len(word))
    return word, ""


def ip(rng: random.Random) -> str:
    if rng.random() < 0.08:
        return rng.choice(["::1", "2001:db8::1", "fe80::1%eth0", "0:0:0:0:0:0:0:1", "2001:0db8:85a3:0000:0000:8a2e:0370:7334"])
    return ".".join(str(rng.randint(0, 255)) for _ in range(4))


def hexid(rng: random.Random, n: int = 8) -> str:
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def uuid(rng: random.Random) -> str:
    h = hexid(rng, 32)
    return f"{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"


def duration(rng: random.Random) -> str:
    v = rng.random()
    if v < 0.3:
        return f"{rng.randint(1, 999)}ms"
    if v < 0.5:
        return f"{rng.randint(1, 9999) / 100:.2f}s"
    if v < 0.65:
        return f"{rng.randint(1, 999)}µs"
    if v < 0.8:
        return f"{rng.randint(1, 5000)}.{rng.randint(0, 999):03d}ms"
    if v < 0.9:
        return f"{rng.randint(1, 120)}m{rng.randint(0, 59)}s"
    return f"{rng.randint(1, 999)} ms"


def size(rng: random.Random) -> str:
    return rng.choice([f"{rng.randint(1, 999)}B", f"{rng.randint(1, 999)}KB", f"{rng.randint(1, 99)}.{rng.randint(0, 9)}MB",
                       f"{rng.randint(1, 9999)} bytes", f"{rng.randint(1, 64)}GiB", f"{rng.randint(100, 99999)}"])


def word(rng: random.Random, n: int | None = None) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n or rng.randint(3, 9)))


def path(rng: random.Random) -> str:
    return rng.choice([
        f"/var/log/{word(rng)}.log", f"/etc/{word(rng)}/{word(rng)}.conf", f"/tmp/{word(rng)}-{hexid(rng, 6)}.tmp",
        f"/home/{rng.choice(V.USERS)}/{word(rng)}/{word(rng)}.txt", f"/data/{word(rng)}/{rng.randint(1, 99)}/part-{rng.randint(0, 9999):05d}.parquet",
        f"C:\\Users\\{rng.choice(V.USERS)}\\AppData\\Local\\{word(rng)}.dat", f"./{word(rng)}/{word(rng)}.json",
        f"s3://{word(rng)}-bucket/{word(rng)}/{rng.randint(2015, 2026)}/{word(rng)}.csv.gz", f"/dev/sd{rng.choice('abc')}{rng.randint(1, 4)}",
        f"/usr/lib/{word(rng)}/{word(rng)}.so.{rng.randint(1, 9)}", f"/opt/app/{word(rng)}/bin/{word(rng)}",
    ])


def url(rng: random.Random) -> str:
    scheme = rng.choice(["http", "https", "https", "grpc", "redis", "postgres", "amqp", "ws"])
    host = rng.choice(["localhost", "api.example.com", "db.internal", "10.0.0.5", "cache.prod.svc.cluster.local", "example.org", "[::1]"])
    port = rng.choice(["", "", f":{rng.randint(80, 65535)}"])
    return f"{scheme}://{host}{port}{rng.choice(V.PATHS) if scheme in ('http', 'https', 'ws') else ''}"


def number(rng: random.Random) -> str:
    return rng.choice([str(rng.randint(0, 9)), str(rng.randint(10, 999)), str(rng.randint(1000, 999999)),
                       f"{rng.random() * 100:.2f}", f"{rng.randint(1, 99)}%", f"-{rng.randint(1, 99)}", f"0x{hexid(rng, rng.choice([2, 4, 8, 16]))}",
                       f"{rng.randint(1, 999)},{rng.randint(0, 999):03d}", f"{rng.random():.4f}", f"{rng.randint(1, 9)}e{rng.randint(1, 9)}"])


def identifier(rng: random.Random) -> str:
    return rng.choice([uuid(rng), hexid(rng, 16), hexid(rng, 12), f"req-{hexid(rng, 8)}", f"{word(rng)}-{rng.randint(1, 99)}",
                       f"blk_{rng.choice(['', '-'])}{rng.randint(10**15, 10**19)}", f"job_{rng.randint(1000, 99999)}",
                       f"{word(rng, 3).upper()}-{rng.randint(1, 9999)}", f"sess_{hexid(rng, 10)}", f"0x{hexid(rng, 8)}",
                       f"{rng.choice(V.SERVICES)}-{hexid(rng, 5)}", f"trace-{hexid(rng, 32)}", f"span-{hexid(rng, 16)}"])


def sql(rng: random.Random) -> str:
    t = rng.choice(["users", "orders", "sessions", "products", "events", "audit_log"])
    return rng.choice([
        f"SELECT * FROM {t} WHERE id = {rng.randint(1, 9999)}",
        f"SELECT {t}.id, {t}.name FROM {t} INNER JOIN accounts ON accounts.id = {t}.account_id LIMIT {rng.randint(1, 100)}",
        f"UPDATE {t} SET updated_at = NOW() WHERE id = ${rng.randint(1, 5)}",
        f"INSERT INTO {t} (id, name, created_at) VALUES (?, ?, ?)",
        f"DELETE FROM {t} WHERE created_at < '{rand_dt(rng).date()}'",
        f'SELECT COUNT(*) FROM "{t}" WHERE "status" = \'active\'',
    ])


def http_request(rng: random.Random) -> str:
    return f"{rng.choice(V.METHODS)} {rng.choice(V.PATHS)} HTTP/{rng.choice(['1.0', '1.1', '1.1', '2.0', '2'])}"


def exception_class(rng: random.Random) -> str:
    return rng.choice(V.JAVA_EXC + V.PY_EXC + V.JS_EXC + V.CS_EXC + V.RB_EXC + V.PHP_EXC)


MESSAGE_TEMPLATES: list[str] = [
    "Started {service} on port {port}",
    "Starting {service} version {ver} (commit {hex})",
    "Server listening on {url}",
    "Listening at: {url} ({num})",
    "{method} {path} {status} {dur}",
    "{method} {path} -> {status} in {dur}",
    "Request completed: {method} {path} status={status} duration={dur}",
    "Handled {method} {path} from {ip} in {dur}",
    '"{req}" {status} {num}',
    "Connection to {host}:{port} refused",
    "connect to {host}:{port} failed: {error}",
    "Connected to {url}",
    "Failed to connect to {url}: {error}",
    "User {user} logged in from {ip}",
    "User '{user}' authenticated successfully",
    "Login failed for user {user}: {error}",
    "Accepted publickey for {user} from {ip} port {port} ssh2: RSA SHA256:{hex}",
    "Invalid user {user} from {ip} port {port}",
    "pam_unix(sshd:session): session opened for user {user} by (uid={num})",
    "Received disconnect from {ip} port {port}:11: disconnected by user",
    "Failed password for invalid user {user} from {ip} port {port} ssh2",
    "Retrying request ({num}/{num}) after {dur}",
    "Retry {num} of {num} for {ident} failed: {error}",
    "Cache miss for key {ident}",
    "Cache hit ratio: {num}% ({num} hits, {num} misses)",
    "Evicted {num} entries from cache {word}",
    "Processed {num} records in {dur}",
    "Processed batch {ident} ({num} items, {size})",
    "Received SIGTERM, shutting down gracefully",
    "Shutting down {service}...",
    "Shutdown complete",
    "Pod {pod} is not ready: {error}",
    "Scaled deployment {service} from {num} to {num} replicas",
    "Successfully pulled image \"{image}\" in {dur}",
    "Back-off restarting failed container {word} in pod {pod}_{ns}({uuid})",
    "Readiness probe failed: {method} {path}: {error}",
    "Liveness probe failed: HTTP probe failed with statuscode: {status}",
    "GC pause {dur} (young gen, {size} -> {size})",
    "Full GC (Allocation Failure) {size}->{size}({size}), {num} secs",
    "Heap usage {num}% of {size}",
    "Query took {dur}: {sql}",
    "Executing query: {sql}",
    "Slow query ({dur}): {sql}",
    "Executed {sql} [took {dur}]",
    "duration: {num} ms  statement: {sql}",
    "Transaction {ident} rolled back: {error}",
    "Deadlock detected between transactions {num} and {num}",
    "Connection pool exhausted (active={num}, idle={num}, max={num})",
    "HikariPool-1 - Start completed.",
    "Acquired connection to {host} in {dur}",
    "Publishing message to topic {word} (partition {num}, offset {num})",
    "Consumed {num} messages from {word} in {dur}",
    "Consumer group {word} rebalancing",
    "Message {ident} delivered to {num} subscribers",
    "Job {ident} enqueued (queue={word})",
    "Job {ident} completed in {dur}",
    "Job {ident} failed after {num} attempts: {error}",
    "Task {ident} succeeded in {dur}: {word}",
    "Scheduled task '{word}' next run at {ts}",
    "Loading configuration from {path}",
    "Config file {path} not found, using defaults",
    "Watching {path} for changes",
    "Wrote {size} to {path}",
    "Failed to open {path}: {error}",
    "File {path} changed, reloading",
    "Rotating log file {path}",
    "Disk usage on {path} at {num}%",
    "Reading {num} lines from {path}",
    "Certificate for {host} expires in {num} days",
    "TLS handshake completed with {host} ({word})",
    "DNS lookup for {host} failed: {error}",
    "Resolved {host} to {ip}",
    "Upstream {host}:{port} marked as down",
    "Health check for {service} passed ({dur})",
    "Health check failed: {error}",
    "Timeout after {dur} waiting for {service}",
    "Circuit breaker for {service} is OPEN",
    "Rate limit exceeded for client {ip} ({num} req/s)",
    "Too many requests from {ip}, blocking for {dur}",
    "Metrics flushed: {num} series in {dur}",
    "Exporting {num} spans to {url}",
    "Registered signal handlers for [TERM, HUP, INT]",
    "Changing view acls to: {user},{user}",
    "Registered {num} routes",
    "Mapped \"{{{path}}}\" onto {word}.{word}()",
    "Initializing {word} module",
    "Initialized {word} in {dur}",
    "Tomcat started on port(s): {port} (http) with context path ''",
    "Started Application in {num} seconds (JVM running for {num})",
    "Using {word} as the {word} provider",
    "No handler found for {method} {path}",
    "Unhandled exception while processing request {ident}",
    "Exception occurred: {exc}: {error}",
    "{exc}: {error}",
    "error: {error}",
    "Error: {error}",
    "unexpected error: {error} (code {num})",
    "Caught {exc} while handling {method} {path}",
    "Panic recovered: {error}",
    "Retrying in {dur}...",
    "Waiting for {service} to become ready ({num}/{num})",
    "{service} is ready",
    "Synced {num} objects ({num} added, {num} updated, {num} deleted)",
    "Reconciling {word}/{word}",
    "Successfully reconciled {ident} in {dur}",
    "Unable to reconcile {ident}: {error}",
    "Deleting orphaned {word} {ident}",
    "Node {host} became NotReady",
    "Node {host} has condition MemoryPressure=True",
    "Allocating {size} for buffer {ident}",
    "Memory usage: {size} RSS, {size} heap",
    "CPU throttled for {dur} ({num}% of period)",
    "Thread pool saturated: {num} queued tasks",
    "Worker {num} started (pid {num})",
    "Worker {num} exited with code {num}",
    "Spawned {num} workers",
    "Booting worker with pid: {num}",
    "Reaped child process {num}",
    "Received request id={ident} from {ip}",
    "Sending {size} response to {ip}",
    "Response {status} sent in {dur}",
    "Uploaded {path} to {url} ({size})",
    "Downloaded {size} from {url} in {dur}",
    "Checksum mismatch for {path}: expected {hex}, got {hex}",
    "Verified signature for {ident}",
    "Token for {user} expired at {ts}",
    "Refreshing access token for {user}",
    "Session {ident} created for user {user}",
    "Session {ident} expired",
    "Permission denied for {user} on {path}",
    "Audit: {user} deleted {word} {ident}",
    "Sent email to {user}@{host} (subject: \"{word} {word}\")",
    "Webhook {url} responded with {status}",
    "Webhook delivery {ident} failed: {error}",
    "Payment {ident} for {num} {word} authorized",
    "Order {ident} placed by {user} (total {num})",
    "Inventory for SKU {ident} below threshold ({num} left)",
    "Migrated database schema to version {num}",
    "Applying migration {num}_{word}",
    "Migration {num}_{word} applied in {dur}",
    "Index {word}_idx created on {word}({word})",
    "Compacting {num} segments in {word}",
    "Snapshot {ident} completed ({size})",
    "Replication lag: {dur} behind {host}",
    "Leader elected: {host}",
    "Lost leadership, stepping down",
    "Term {num}: voted for {host}",
    "Applied {num} raft entries",
    "Block {ident} replicated to {ip}:{port}",
    "PacketResponder {num} for block {ident} terminating",
    "Receiving block {ident} src: /{ip}:{port} dest: /{ip}:{port}",
    "BLOCK* NameSystem.addStoredBlock: blockMap updated: {ip}:{port} is added to {ident} size {num}",
    "Notification time out: {num}",
    "Received connection request /{ip}:{port}",
    "Send worker leaving thread",
    "Created MRAppMaster for application {ident}",
    "Registered signal handlers for [TERM, HUP, INT]",
    "instruction cache parity error corrected",
    "Thermal pressure state: {num} Memory pressure state: {num}",
    "session closed for user {user}",
    "(root) CMD (run-parts /etc/cron.hourly)",
    "Loaded Servicing Stack v{ver} with Core: {path}",
    "workerEnv.init() ok {path}",
    "mod_jk child workerEnv in error state {num}",
    "Component State Change: Component \\042{word}\\042 is in the unavailable state (HWID={num})",
    "authentication failure; logname= uid={num} euid={num} tty=NODEVssh ruser= rhost={ip}",
    "check pass; user unknown",
    "Sync 'cache' with {num} items",
    "Received unknown command {word} from {ip}",
    "Dropped {num} packets on {word}",
    "Link {word} is up ({num} Mbps, full duplex)",
    "eth{num}: link becomes ready",
    "Out of memory: Killed process {num} ({word}) total-vm:{num}kB, anon-rss:{num}kB",
    "Started Session {num} of user {user}.",
    "Stopped target {word}.",
    "Reached target Multi-User System.",
    "Starting {word} daemon...",
    "{word}.service: Main process exited, code=exited, status={num}/FAILURE",
    "{word}.service: Failed with result 'exit-code'.",
    "Removed session {num}.",
    "New session {num} of user {user}.",
    "Time has been changed",
    "Clock skew detected: {num}s",
    "Selected source {ip}",
    "Deleting {num} expired keys",
    "DB saved on disk",
    "Background saving started by pid {num}",
    "Ready to accept connections",
    "Accepted connection from {ip}:{port}",
    "Client {ip}:{port} disconnected",
    "Closing connection {ident} (idle for {dur})",
    "Keepalive timeout on connection {ident}",
    "Received {size} on socket {num}",
    "epoll wait returned {num} events",
    "Parsed {num} tokens in {dur}",
    "Compiled {num} files in {dur}",
    "Build finished with {num} warnings",
    "webpack compiled successfully in {dur}",
    "ready - started server on 0.0.0.0:{port}, url: http://localhost:{port}",
    "compiling {path}...",
    "Hot reload: {path} updated",
    "Test suite passed ({num} tests, {dur})",
    "{num} passing ({dur})",
    "Coverage: {num}% lines, {num}% branches",
    "Deploy {ident} to {word} succeeded",
    "Rolling back release {ident}",
    "Feature flag {word} enabled for {num}% of users",
    "Experiment {word}: variant {word} assigned to {user}",
    "Model {word} loaded in {dur} ({size})",
    "Inference batch {num}: {num} samples, {dur}",
    "Epoch {num}/{num} loss={num} acc={num}",
    "Checkpoint saved to {path}",
    "CUDA out of memory. Tried to allocate {size}",
    "Vault sealed; refusing request",
    "Renewed lease {ident} for {dur}",
    "Consul: member {host} joined",
    "Serf: EventMemberFailed: {host} {ip}",
    "gRPC call {word}/{word} failed: {error}",
    "grpc: addrConn.createTransport failed to connect to {{Addr: \"{ip}:{port}\"}}. Err: {error}",
    "transport: Error while dialing {error}",
    "http: TLS handshake error from {ip}:{port}: {error}",
    "http: proxy error: {error}",
    "reverse proxy error: {error}",
    "upstream timed out ({num}: {error}) while reading response header from upstream",
    "connect() failed ({num}: {error}) while connecting to upstream",
    "open() \"{path}\" failed ({num}: {error})",
    "*{num} no live upstreams while connecting to upstream",
    "client intended to send too large body: {num} bytes",
    "SSL_do_handshake() failed (SSL: error:{hex}:SSL routines:{word}:{word})",
    "AH00558: apache2: Could not reliably determine the server's fully qualified domain name, using {ip}. Set the 'ServerName' directive globally to suppress this message",
    "AH01630: client denied by server configuration: {path}",
    "script '{path}' not found or unable to stat",
    "PHP Notice:  Undefined variable: {word} in {path} on line {num}",
    "Undefined index: {word}",
    "Deprecated: {word}(): Passing null to parameter #{num} is deprecated",
    "Maximum execution time of {num} seconds exceeded",
    "Allowed memory size of {num} bytes exhausted (tried to allocate {num} bytes)",
    "Uncaught {exc}: {error}",
    "Traceback follows",
    "Unhandled promise rejection: {error}",
    "DeprecationWarning: {word} is deprecated, use {word} instead",
    "(node:{num}) UnhandledPromiseRejectionWarning: {exc}: {error}",
    "ExperimentalWarning: {word} is an experimental feature",
    "MaxListenersExceededWarning: Possible EventEmitter memory leak detected. {num} {word} listeners added",
    "Compiled with warnings.",
    "Module not found: Error: Can't resolve '{word}' in '{path}'",
    "npm ERR! code {word}",
    "npm WARN deprecated {word}@{ver}: {error}",
    "warning: unused variable `{word}`",
    "error[E{num}]: cannot borrow `{word}` as mutable because it is also borrowed as immutable",
    "Compiling {word} v{ver} ({path})",
    "Finished dev [unoptimized + debuginfo] target(s) in {dur}",
    "Running `target/debug/{word}`",
    "thread '{word}' panicked at '{error}', {path}",
    "goroutine {num} [running]:",
    "runtime error: index out of range [{num}] with length {num}",
    "runtime error: invalid memory address or nil pointer dereference",
    "[signal SIGSEGV: segmentation violation code=0x1 addr=0x0 pc=0x{hex}]",
    "context canceled",
    "dial tcp {ip}:{port}: connect: {error}",
    "read tcp {ip}:{port}->{ip}:{port}: {error}",
    "Get \"{url}\": {error}",
    "Post \"{url}\": context deadline exceeded (Client.Timeout exceeded while awaiting headers)",
    "level mismatch: expected {word}, got {word}",
    "Got {num} results for query '{word} {word}'",
    "Indexed {num} documents in {dur}",
    "[gc][{num}] overhead, spent [{dur}] collecting in the last [{dur}]",
    "[{word}] started",
    "[{word}] publish_address {{{ip}:{port}}}, bound_addresses {{[::]:{port}}}",
    "flush of [{word}][{num}] took {dur}",
    "recovered [{num}] indices into cluster_state",
    "Cluster health status changed from [YELLOW] to [GREEN] (reason: [shards started [[{word}][{num}]]])",
    "[Controller id={num}] Processing automatic preferred replica leader election",
    "[Partition {word}-{num} broker={num}] Log loaded for partition {word}-{num} with initial high watermark {num}",
    "[Consumer clientId={word}, groupId={word}] Discovered group coordinator {host}:{port} (id: {num} rack: null)",
    "[KafkaServer id={num}] started",
    "Created log for partition [{word},{num}] in {path}",
    "ZooKeeper client session {ident} established",
    "Processing ruok command from /{ip}:{port}",
    "Accepted socket connection from /{ip}:{port}",
    "Closed socket connection for client /{ip}:{port} which had sessionid 0x{hex}",
    "Established session 0x{hex} with negotiated timeout {num} for client /{ip}:{port}",
    "Cannot open channel to {num} at election address /{ip}:{port}",
    "Received connection request /{ip}:{port}",
    "Notification: {num} (message format version), {num} (n.leader), 0x{hex} (n.zxid), 0x{hex} (n.round), LOOKING (n.state), {num} (n.sid), 0x{hex} (n.peerEpoch) LOOKING (my state)",
    "Container {ident} transitioned from RUNNING to COMPLETED",
    "Killing container {ident}",
    "Task attempt {ident} is done. And is in the process of committing",
    "Progress of TaskAttempt {ident} is : {num}",
    "Adding task '{ident}' to tip {ident}, for tracker '{host}:{port}'",
    "ApplicationMaster launched with {num} containers",
    "Diagnostics report from {ident}: Container killed by the ApplicationMaster.",
    "Memory usage of ProcessTree {num} for container-id {ident}: {size} of {size} physical memory used",
    "Registered executor NettyRpcEndpointRef({url}) ({ip}:{port}) with ID {num}",
    "Block {ident} stored as values in memory (estimated size {size}, free {size})",
    "Finished task {num}.{num} in stage {num}.{num} (TID {num}) in {dur} on {host} ({num}/{num})",
    "Removed broadcast_{num}_piece{num} on {host}:{port} in memory (size: {size}, free: {size})",
    "Starting task {num}.{num} in stage {num}.{num} (TID {num}, {host}, partition {num},PROCESS_LOCAL, {num} bytes)",
    "Lost executor {num} on {host}: {error}",
    "Stage {num} ({word} at {word}.scala:{num}) finished in {dur}",
    "acquire lock={num}, flags=0x{hex}, tag=\"{word} {word}\", name={word}, ws=null, uid={num}, pid={num}",
    "printFreezingDisplayLogsopening app wtoken = AppWindowToken{{{hex} token=Token{{{hex} ActivityRecord{{{hex} u0 {word}/.{word} t{num}}}}}}}, allDrawn= false",
    "Displayed {word}/.{word}: +{dur}",
    "Force stopping {word} appid={num} user={num}: {word}",
    "Start proc {num}:{word}/u0a{num} for activity {word}/.{word}",
    "Killing {num}:{word}/u0a{num} (adj {num}): empty #{num}",
    "onStandStepChanged {num}",
    "onReceive action: android.intent.action.SCREEN_ON",
    "calculateCaloriesWithCache totalCalories={num}",
    "getTodayTotalDetailSteps = {num}##{num}##{num}##{num}",
    "IOThunderboltSwitch<0>(0x0)::listenerCallback - Thunderbolt HPD packet for route = 0x0 port = {num} unplug = {num}",
    "ARPT: {num}.{num}: {word}: {word} {num}",
    "{word}.exe - {host}:{port} open through proxy {host}:{port} HTTPS",
    "{word}.exe - {host}:{port} close, {size} sent, {size} received, lifetime {dur}",
    "state_change.unavailable",
    "Interface {word}, changed state to down",
    "Ignoring {word}: {error}",
    "Skipping {word} ({error})",
    "Nothing to do",
    "Done.",
    "OK",
    "done",
    "...",
    "Starting",
    "Hello, world!",
    "{word} {word} {word}",
    "{word} {word} {word} {word} {word} {word}",
    "{Word} {word} {word} {word}.",
    "{word}: {word} {word} {num}",
    "{Word}: {word} '{word}' {word} {num} {word}",
    "{word}={num} {word}={word} inside the message text",
    "using {word}={word}, {word}={num}",
    "Value for {word} is {num}",
    "Set {word} to {num} (was {num})",
    "Metrics: rps={num} p50={dur} p99={dur}",
    "Stats: {word}={num}, {word}={num}, {word}={num}",
    "Total: {num}, failed: {num}, skipped: {num}",
    "status: {status} len: {num} time: {num}",
    "{ip} \"{req}\" status: {status} len: {num} time: {num}",
    "{ip} - - [{ts}] \"{req}\" {status} {num}",
    "\"{req}\" {status} {num} \"{referer}\" \"{ua}\"",
    "Caller: {word}.{word}() at {path}:{num}",
    "See {url} for details",
    "Documentation: {url}",
    "Visit {url}",
    "Unicode: café résumé naïve 日本語 emoji 🚀 done",
    "Größe überschritten für Datei {path}",
    "Erreur de connexion à {host}",
    "Ошибка: {error}",
    "接続に失敗しました: {host}",
    "Tab\tseparated\tmessage part",
    "Message with trailing spaces",
    ("Very long message: " + "lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor incididunt ut labore et dolore magna aliqua " * 2).rstrip(),
    "{{\"event\": \"{word}\", \"id\": {num}, \"ok\": true}}",
    "payload={{\"user\":\"{user}\",\"action\":\"{word}\"}}",
    "response: {{\"status\":\"{word}\",\"items\":[{num},{num},{num}]}}",
    "body: [{num}, {num}, \"{word}\"]",
    "args: ['{word}', '{word}', {num}]",
    "<{word}> {word} {word}",
    "[{word}] {word} {word} ({num})",
    "({word}) {word} {word}",
    "{word}: {word} {word}: {word}",
    "{Word} {word}: {error}",
    "{Word} {word} ({word} {word})",
    "!!! {word} {word} !!!",
    "*** {word} {word} ***",
    "=== {word} {word} ===",
    "--- {word} ---",
    "--> {word} {word}",
    "<-- {status} {method} {path} ({dur})",
    "==> {word} {path} <==",
    "#{num} {word} {word}",
    "% {word} {word}",
    "{word}@{host}: {word} {word}",
    "{user}@{host}:{path}$ {word} {word}",
    "$ {word} --{word}={num}",
    "> {word}@{ver} {word}",
    "PING {host} ({ip}) {num}({num}) bytes of data.",
    "{num} bytes from {host} ({ip}): icmp_seq={num} ttl={num} time={num} ms",
    "{num} packets transmitted, {num} received, {num}% packet loss, time {num}ms",
    "iptables: DROP IN=eth0 OUT= MAC={hex} SRC={ip} DST={ip} LEN={num} PROTO=TCP SPT={port} DPT={port}",
    "audit: type={num} audit({num}.{num}:{num}): pid={num} uid={num} auid={num} ses={num} msg='op={word} acct=\"{user}\" exe=\"{path}\" hostname=? addr=? terminal=? res=success'",
]

WORDS = ["cache", "worker", "queue", "session", "request", "response", "handler", "router", "config", "metrics",
         "buffer", "socket", "stream", "channel", "cluster", "node", "shard", "replica", "index", "table", "batch",
         "token", "auth", "user", "order", "payment", "invoice", "job", "task", "event", "timer", "pool", "client",
         "server", "proxy", "upstream", "backend", "frontend", "module", "plugin", "driver", "device", "volume",
         "snapshot", "backup", "restore", "sync", "flush", "compact", "rotate", "reload", "restart", "stop", "start",
         "default", "main", "primary", "secondary", "eth0", "wlan0", "docker0", "lo", "sda", "nvme0n1", "tmpfs"]


def fill(rng: random.Random, template: str) -> str:
    out = []
    i = 0
    while i < len(template):
        if template.startswith("{{", i):
            out.append("{")
            i += 2
            continue
        if template.startswith("}}", i):
            out.append("}")
            i += 2
            continue
        if template[i] == "{":
            j = template.index("}", i)
            key = template[i + 1:j]
            out.append(slot(rng, key))
            i = j + 1
        else:
            out.append(template[i])
            i += 1
    return "".join(out)


def slot(rng: random.Random, key: str) -> str:
    match key:
        case "service":
            return rng.choice(V.SERVICES)
        case "port":
            return str(rng.choice([80, 443, 8080, 8443, 3000, 5000, 5432, 6379, 9090, 9200, 27017, rng.randint(1024, 65535)]))
        case "ver":
            return f"{rng.randint(0, 12)}.{rng.randint(0, 30)}.{rng.randint(0, 99)}"
        case "hex":
            return hexid(rng, rng.choice([7, 8, 12, 16, 40]))
        case "url":
            return url(rng)
        case "method":
            return rng.choice(V.METHODS)
        case "path":
            return rng.choice(V.PATHS) if rng.random() < 0.5 else path(rng)
        case "status":
            return rng.choice(V.STATUSES)
        case "dur":
            return duration(rng)
        case "req":
            return http_request(rng)
        case "num":
            return number(rng)
        case "host":
            return rng.choice(V.HOSTS)
        case "ip":
            return ip(rng)
        case "error":
            return rng.choice(V.ERRORS)
        case "user":
            return rng.choice(V.USERS)
        case "ident":
            return identifier(rng)
        case "uuid":
            return uuid(rng)
        case "word":
            return rng.choice(WORDS) if rng.random() < 0.7 else word(rng)
        case "Word":
            return slot(rng, "word").capitalize()
        case "size":
            return size(rng)
        case "pod":
            return rng.choice(V.PODS)
        case "ns":
            return rng.choice(V.K8S_NS)
        case "image":
            return f"{rng.choice(['docker.io/library', 'ghcr.io/acme', 'gcr.io/proj', 'registry.example.com/team'])}/{rng.choice(WORDS)}:{rng.choice(['latest', 'v1.2.3', 'sha256-' + hexid(rng, 12), '3.19'])}"
        case "sql":
            return sql(rng)
        case "ts":
            return timestamp(rng, rng.choice([0, 1, 7, 8, 9]))
        case "exc":
            return exception_class(rng)
        case "referer":
            return rng.choice(V.REFERERS)
        case "ua":
            return rng.choice(V.UAS)
        case _:
            raise KeyError(key)


def message(rng: random.Random) -> str:
    return fill(rng, rng.choice(MESSAGE_TEMPLATES))


KV_KEYS = ["user", "user_id", "userId", "request_id", "requestId", "req_id", "trace_id", "traceID", "span_id", "duration",
           "duration_ms", "latency", "elapsed", "took", "status", "status_code", "method", "path", "url", "host", "port",
           "ip", "client_ip", "remote_addr", "bytes", "size", "count", "n", "attempt", "retries", "err", "error", "reason",
           "msg", "component", "module", "service", "env", "region", "az", "version", "pid", "tid", "thread", "id",
           "job", "queue", "topic", "partition", "offset", "key", "value", "name", "type", "kind", "namespace", "pod",
           "container", "node", "cluster", "table", "rows", "query", "db", "conn", "peer", "addr", "fd", "code", "op",
           "source", "target", "dest", "src", "dst", "proto", "ttl", "len", "seq", "uid", "gid", "exe", "terminal",
           "caller", "func", "file", "line", "level", "lvl", "time", "ts", "logger", "app", "dyno", "connect", "fwd",
           "protocol", "request", "upstream", "server", "referrer", "agent", "tag", "label", "state", "phase", "result",
           "ok", "success", "enabled", "dry_run", "force", "timeout", "interval", "limit", "page", "total", "mem", "cpu",
           "heap", "gc", "x-request-id", "http.method", "http.status_code", "net.peer.ip", "k8s.pod.name", "aws.region",
           "Duration", "Billed Duration", "Memory Size", "Max Memory Used", "Init Duration", "RequestId", "Version"]


def kv_value(rng: random.Random, key: str) -> str:
    k = key.lower()
    if "duration" in k or k in ("latency", "elapsed", "took", "timeout", "interval", "connect", "service"):
        return duration(rng) if rng.random() < 0.8 else str(rng.randint(1, 5000))
    if k in ("status", "status_code", "http.status_code", "code"):
        return rng.choice(V.STATUSES)
    if k == "method" or k == "http.method":
        return rng.choice(V.METHODS)
    if k in ("path", "url", "request", "upstream"):
        return rng.choice(V.PATHS) if k == "path" else url(rng)
    if k in ("host", "server", "node", "peer", "addr", "dest", "target"):
        return rng.choice(V.HOSTS) if rng.random() < 0.5 else f"{ip(rng)}:{rng.randint(1000, 65535)}"
    if "ip" in k or k in ("remote_addr", "src", "dst", "fwd"):
        return ip(rng)
    if k in ("user", "user_id", "userid", "uid"):
        return rng.choice(V.USERS) if rng.random() < 0.6 else str(rng.randint(1, 99999))
    if k in ("err", "error", "reason", "msg"):
        return rng.choice(V.ERRORS) if rng.random() < 0.7 else message(rng)
    if "id" in k or k in ("key", "job", "trace", "span", "tag", "label"):
        return identifier(rng)
    if k in ("level", "lvl"):
        return rng.choice(["info", "warn", "error", "debug", "INFO", "WARN", "ERROR", "30", "50"])
    if k in ("time", "ts"):
        return timestamp(rng, rng.choice([0, 1, 5, 30, 32]))
    if k in ("ok", "success", "enabled", "dry_run", "force"):
        return rng.choice(["true", "false", "1", "0", "yes", "no"])
    if k in ("bytes", "size", "mem", "heap", "memory size", "max memory used", "len"):
        return size(rng)
    if k in ("env", "region", "az", "phase", "state", "result", "kind", "type", "proto", "protocol", "dyno", "queue", "topic"):
        return rng.choice(["prod", "staging", "dev", "us-east-1", "eu-west-1b", "Running", "Pending", "ok", "failed", "tcp",
                           "https", "web.1", "worker.3", "default", "high", "events", "audit"])
    if k in ("caller", "file", "func"):
        return rng.choice([f"{word(rng)}.go:{rng.randint(1, 999)}", f"{word(rng)}/{word(rng)}.go:{rng.randint(1, 999)}", f"{word(rng)}.{word(rng)}"])
    if k in ("version",):
        return f"v{rng.randint(0, 9)}.{rng.randint(0, 20)}.{rng.randint(0, 50)}"
    if k in ("component", "module", "logger", "app", "service", "container", "pod", "namespace", "cluster", "table", "db", "source", "name", "op"):
        return rng.choice(V.SERVICES + WORDS + V.PODS[:3] + V.K8S_NS[:3])
    if k in ("query",):
        return sql(rng)
    if k in ("agent",):
        return rng.choice(V.UAS)
    if k in ("referrer",):
        return rng.choice(V.REFERERS)
    if k == "exe":
        return path(rng)
    return number(rng) if rng.random() < 0.6 else word(rng)


def add_kv(line: Line, rng: random.Random, n: int | None = None, style: str | None = None, sep: str | None = None) -> None:
    """Appends n key=value pairs (with a leading separator) to the line."""
    n = n if n is not None else rng.choice([1, 1, 2, 2, 3, 4, 5, 7])
    style = style or rng.choices(["eq", "colon", "bracket", "eqspace"], weights=[70, 18, 8, 4])[0]
    sep = sep if sep is not None else rng.choice([" ", " ", " ", "  ", ", ", ",", "\t", " | "])
    quote_p = rng.random()
    for i in range(n):
        key = rng.choice(KV_KEYS)
        val = kv_value(rng, key)
        if i > 0 or line.parts:
            line.add(sep if i > 0 else (" " if sep in (", ", ",", " | ") else sep))
        needs_quote = " " in val or "," in val or (val == "" and rng.random() < 0.3)
        q = '"' if rng.random() < 0.85 else "'"
        quoted = needs_quote or quote_p < 0.15
        if style == "bracket":
            line.add("[")
        if " " in key and style != "colon":
            key = key.replace(" ", "_")
        line.add(key, "KEY")
        if style == "colon":
            line.add(": " if rng.random() < 0.8 else ":")
        elif style == "eqspace":
            line.add(" = ")
        else:
            line.add("=")
        if rng.random() < 0.05:
            val = ""
            quoted = rng.random() < 0.5
        if quoted:
            line.add(q)
            if val:
                line.add(val, "VALUE")
            line.add(q)
        else:
            if val:
                line.add(val, "VALUE")
        if style == "bracket":
            line.add("]")


PKG_SEGMENTS = ["com", "org", "io", "net", "de", "co", "example", "acme", "app", "api", "core", "util", "service",
                "web", "db", "http", "auth", "internal", "impl", "client", "server", "config", "model", "data", "cache",
                "spring", "apache", "google", "netty", "kafka", "hibernate", "boot", "cloud", "aws", "sdk", "v1", "v2"]
CLASS_WORDS = ["Foo", "Bar", "Main", "App", "Server", "Client", "Service", "Controller", "Handler", "Manager", "Repository",
               "Factory", "Builder", "Filter", "Listener", "Worker", "Job", "Task", "Runner", "Loader", "Cache", "Pool",
               "Config", "Router", "Mapper", "Reader", "Writer", "Parser", "Store", "Engine", "Scheduler", "Monitor", "Gateway"]


def java_logger(rng: random.Random) -> str:
    """Random dotted logger names: com.example.Foo, org.acme.api.UserService, c.e.d.App, Foo."""
    n = rng.choice([1, 2, 2, 3, 3, 3, 4, 5])
    segs = [rng.choice(PKG_SEGMENTS) if rng.random() < 0.8 else word(rng, rng.randint(2, 6)) for _ in range(n - 1)]
    cls = rng.choice(CLASS_WORDS) if rng.random() < 0.5 else rng.choice(CLASS_WORDS) + rng.choice(CLASS_WORDS)
    if rng.random() < 0.15:
        cls = cls + "$" + rng.choice(["1", "Inner", "Builder", "Worker", str(rng.randint(1, 9))])
    if rng.random() < 0.15 and segs:
        segs = [s[0] for s in segs]  # %c{1.} abbreviation: c.e.d.App
    return ".".join(segs + [cls])


def py_logger(rng: random.Random) -> str:
    n = rng.choice([1, 2, 2, 3, 4])
    return ".".join(rng.choice(PKG_SEGMENTS + WORDS[:20]).lower() if rng.random() < 0.8 else word(rng) for _ in range(n))


def source_name(rng: random.Random) -> str:
    v = rng.random()
    if v < 0.25:
        return java_logger(rng)
    if v < 0.35:
        return py_logger(rng)
    pool = rng.choice([V.JAVA_LOGGERS, V.PY_LOGGERS, V.GO_PKGS, V.RUST_TARGETS, V.NODE_MODULES, V.SERVICES, V.PROCS])
    return rng.choice(pool)


def thread_name(rng: random.Random) -> str:
    return rng.choice([
        "main", "main", "Thread-1", f"Thread-{rng.randint(0, 99)}", f"pool-{rng.randint(1, 9)}-thread-{rng.randint(1, 32)}",
        f"http-nio-{rng.choice([8080, 8443])}-exec-{rng.randint(1, 50)}", "scheduling-1", "reactor-http-nio-2", "nioEventLoopGroup-3-1",
        f"worker-{rng.randint(0, 15)}", "MainThread", "Dummy-2", "ThreadPoolExecutor-0_1", "asyncio_0", "kafka-producer-network-thread | producer-1",
        f"grpc-default-executor-{rng.randint(0, 9)}", "ForkJoinPool.commonPool-worker-3", "Timer-0", "DestroyJavaVM", "Finalizer",
        f"{rng.randint(1, 65535)}", f"{rng.randint(100, 65535)}#{rng.randint(100, 65535)}", "MainProcess", "ForkPoolWorker-3", "SpawnProcess-1",
        f"tid={rng.randint(1, 99999)}", "background-preinit", "RMI TCP Connection(2)-127.0.0.1", "AsyncAppender-Worker-Thread-0",
        "QuorumPeer[myid=1]/0:0:0:0:0:0:0:0:2181", f"SendWorker:{rng.randint(10**10, 10**12)}", "NIOServerCxn.Factory:0.0.0.0/0.0.0.0:2181",
        f"req-{uuid(rng)}", "vert.x-eventloop-thread-0", "Catalina-utility-1", "puma srv tp 001", f"ForkPoolWorker-{rng.randint(1, 8)}",
    ])
