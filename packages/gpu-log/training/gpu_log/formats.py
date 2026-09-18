"""Entry-line formats. Each builder returns a Line of kind "entry" (unless noted)."""

from __future__ import annotations

import random
from collections.abc import Callable

from . import vocab as V
from .gen import (
    Line, add_kv, duration, hexid, hostname, http_request, identifier, ip, java_logger, level_word, message,
    pad_level, py_logger, source_name, thread_name, timestamp, uuid, word,
)

Builder = Callable[[random.Random], Line]


def msg_tail(line: Line, rng: random.Random, kv_p: float = 0.25, sep: str | None = None, prefix: str = "") -> None:
    """Message, optionally followed by a kv tail.

    `prefix` (the RFC 5424 "BOM" marker) is part of the MSG span: it is glued to the first
    word of the message, so a span boundary inside that token would be unlearnable.
    """
    m = prefix + message(rng)
    line.add(m, "MSG")
    if rng.random() < kv_p:
        add_kv(line, rng, sep=sep)


def bracket(rng: random.Random) -> tuple[str, str]:
    return rng.choice([("[", "]"), ("[", "]"), ("[", "]"), ("(", ")"), ("<", ">"), ("", ""), ("{", "}")])


def maybe_pri(line: Line, rng: random.Random) -> None:
    if rng.random() < 0.15:
        line.add(f"<{rng.randint(0, 191)}>", "LEVEL")


def pid_suffix(line: Line, rng: random.Random, p: float = 0.8) -> None:
    if rng.random() < p:
        line.add("[").add(str(rng.randint(1, 99999)), "THREAD").add("]")


# --- syslog / journal -------------------------------------------------------------------

def syslog_3164(rng: random.Random) -> Line:
    L = Line("entry")
    maybe_pri(L, rng)
    L.add(timestamp(rng, rng.choice([15, 15, 16])), "TS").add(" ")
    L.add(hostname(rng), "HOST").add(" ")
    proc = rng.choice(V.PROCS)
    L.add(proc, "SOURCE")
    pid_suffix(L, rng)
    L.add(": " if rng.random() < 0.9 else " ")
    msg_tail(L, rng, 0.15)
    return L


def syslog_5424(rng: random.Random) -> Line:
    L = Line("entry")
    L.add(f"<{rng.randint(0, 191)}>", "LEVEL").add("1 ")
    L.add(timestamp(rng, rng.choice([1, 2, 5, 6])), "TS").add(" ")
    hh = hostname(rng)
    L.add(hh, "HOST") if hh != "-" else L.add("-")
    L.add(" ")
    L.add(rng.choice(V.PROCS + V.SERVICES + ["-"]), "SOURCE").add(" ")
    L.add(rng.choice([str(rng.randint(1, 99999)), "-"]), "THREAD").add(" ")
    L.add(rng.choice(["-", "ID47", "MSGID", word(rng).upper()])).add(" ")
    if rng.random() < 0.6:
        L.add(f"[{rng.choice(['exampleSDID@32473', 'timeQuality', 'meta', 'origin', 'app@12345'])}")
        for _ in range(rng.randint(1, 3)):
            L.add(" ").add(rng.choice(["iut", "eventSource", "eventID", "tzKnown", "isSynced", "ip", "software", "swVersion", "sequenceId"]), "KEY")
            L.add('="').add(rng.choice(["3", "Application", "1011", "1", "0", ip(rng), "rsyslogd", "8.2001.0", str(rng.randint(1, 9999))]), "VALUE").add('"')
        L.add("]")
        if rng.random() < 0.3:
            L.add("[").add("other@1").add(" ").add("k", "KEY").add('="').add("v", "VALUE").add('"]')
    else:
        L.add("-")
    if rng.random() < 0.9:
        L.add(" ")
        msg_tail(L, rng, 0.15, prefix="BOM" if rng.random() < 0.1 else "")
    return L


def journal(rng: random.Random) -> Line:
    L = Line("entry")
    style = rng.choice([15, 4, 6, 20, 42])
    L.add(timestamp(rng, style), "TS").add(" ")
    L.add(hostname(rng), "HOST").add(" ")
    L.add(rng.choice(V.PROCS + ["systemd", "systemd", "kernel", "sshd"]), "SOURCE")
    pid_suffix(L, rng)
    L.add(": ")
    msg_tail(L, rng, 0.1)
    return L


def journal_marker(rng: random.Random) -> Line:
    L = Line("continuation")
    L.add(rng.choice(["-- Logs begin at ", "-- Journal begins at ", "-- Boot ", "-- Reboot --", "-- No entries --"]))
    if L.text().endswith("at "):
        L.add(timestamp(rng, 20), "TS").add(", end at ").add(timestamp(rng, 20), "TS").add(". --")
    elif L.text() == "-- Boot ":
        L.add(hexid(rng, 32)).add(" --")
    return L


# --- web servers -------------------------------------------------------------------------

def access_log(rng: random.Random) -> Line:
    L = Line("entry")
    if rng.random() < 0.2:
        L.add(rng.choice(V.HOSTS + ["example.com", "api.example.com"]), "HOST").add(" ")
    L.add(ip(rng), "SOURCE").add(" - ")
    L.add(rng.choice(["-", "-", "-", rng.choice(V.USERS)])).add(" [")
    L.add(timestamp(rng, 17), "TS").add("] \"")
    L.add(http_request(rng), "MSG").add("\" ")
    L.add(rng.choice(V.STATUSES), "VALUE").add(" ")
    L.add(rng.choice(["-", str(rng.randint(0, 999999))]), "VALUE")
    if rng.random() < 0.7:
        L.add(' "').add(rng.choice(V.REFERERS), "VALUE").add('" "').add(rng.choice(V.UAS), "VALUE").add('"')
        if rng.random() < 0.3:
            L.add(" ")
            add_kv(L, rng, style="eq", sep=" ")
        elif rng.random() < 0.3:
            L.add(' "').add(rng.choice(["-", ip(rng)]), "VALUE").add('"')
    return L


def nginx_error(rng: random.Random) -> Line:
    L = Line("entry")
    L.add(timestamp(rng, 13), "TS").add(" [")
    L.add(rng.choice(["error", "warn", "crit", "notice", "info", "alert", "emerg", "error", "error"]), "LEVEL").add("] ")
    L.add(f"{rng.randint(1, 99999)}#{rng.randint(1, 99999)}", "THREAD").add(": ")
    if rng.random() < 0.8:
        L.add(f"*{rng.randint(1, 99999)} ")
    L.add(message(rng), "MSG")
    n = rng.randint(0, 5)
    for k in ["client", "server", "request", "upstream", "host", "referrer"][:n]:
        L.add(", ").add(k, "KEY").add(": ")
        val = {"client": ip(rng), "server": rng.choice(V.HOSTS + ["_", "localhost"]), "request": http_request(rng),
               "upstream": f"http://{ip(rng)}:{rng.randint(80, 9999)}{rng.choice(V.PATHS)}", "host": rng.choice(V.HOSTS),
               "referrer": rng.choice(V.REFERERS)}[k]
        if k in ("request", "upstream", "host", "referrer"):
            L.add('"').add(val, "VALUE").add('"')
        else:
            L.add(val, "VALUE")
    return L


def apache_error(rng: random.Random) -> Line:
    L = Line("entry")
    L.add("[").add(timestamp(rng, rng.choice([18, 19])), "TS").add("] [")
    v = rng.random()
    if v < 0.5:
        L.add(rng.choice(["core", "mpm_prefork", "ssl", "proxy", "php7", "authz_core", "mpm_event", "rewrite"]), "SOURCE").add(":")
    L.add(rng.choice(["error", "warn", "notice", "info", "crit", "debug", "error", "notice"]), "LEVEL").add("] ")
    if v < 0.5:
        L.add("[pid ").add(str(rng.randint(1, 99999)), "THREAD")
        if rng.random() < 0.5:
            L.add(":tid ").add(str(rng.randint(1, 10**12)), "THREAD")
        L.add("] ")
    if rng.random() < 0.6:
        L.add("[").add("client", "KEY").add(" ").add(f"{ip(rng)}:{rng.randint(1000, 65535)}", "VALUE").add("] ")
    if rng.random() < 0.4:
        L.add(f"AH{rng.randint(0, 9999):05d}: ")
    L.add(message(rng), "MSG")
    if rng.random() < 0.3:
        L.add(", referer: ").add(rng.choice(V.REFERERS[2:]), "VALUE")
    return L


# --- JVM ---------------------------------------------------------------------------------

def log4j(rng: random.Random) -> Line:
    L = Line("entry")
    variant = rng.randrange(10)
    ts = timestamp(rng, rng.choice([8, 8, 8, 9, 29, 7, 1, 5, 21]))
    lw, _ = level_word(rng)
    lw, pad = pad_level(rng, lw)
    thread = thread_name(rng)
    logger = rng.choice(V.JAVA_LOGGERS) if rng.random() < 0.5 else java_logger(rng)
    if variant == 0:  # %d [%t] %-5p %c - %m
        L.add(ts, "TS").add(" [").add(thread, "THREAD").add("] ").add(lw, "LEVEL").add(pad).add(" ").add(logger, "SOURCE").add(" - ")
    elif variant == 1:  # %d %-5p [%t] %c{1}: %m
        L.add(ts, "TS").add(" ").add(lw, "LEVEL").add(pad).add(" [").add(thread, "THREAD").add("] ").add(logger.split(".")[-1], "SOURCE").add(": ")
    elif variant == 2:  # %d %p %c: %m  (hadoop)
        L.add(ts, "TS").add(" ").add(lw, "LEVEL").add(" ").add(logger, "SOURCE").add(": ")
    elif variant == 3:  # spark yy/MM/dd
        L.add(timestamp(rng, 25), "TS").add(" ").add(lw, "LEVEL").add(" ").add(logger, "SOURCE").add(": ")
    elif variant == 4:  # zookeeper: %d - %-5p [%t:%C{1}@%L] - %m
        L.add(ts, "TS").add(" - ").add(lw, "LEVEL").add(pad).add(" [").add(thread, "THREAD").add(":").add(logger.split(".")[-1], "SOURCE").add("@").add(str(rng.randint(1, 999)), "LINE").add("] - ")
    elif variant == 5:  # kafka: [%d] %p %m (%c)
        L.add("[").add(ts, "TS").add("] ").add(lw, "LEVEL").add(" ")
        msg_tail(L, rng, 0.1)
        L.add(" (").add(logger, "SOURCE").add(")")
        return L
    elif variant == 6:  # elasticsearch: [%d][%-5p][%c{1.}] [node] %m
        L.add("[").add(ts, "TS").add("][").add(lw, "LEVEL").add(pad).add("][").add(logger, "SOURCE").add(" " * rng.randint(0, 12)).add("] [").add(hostname(rng), "HOST").add("] ")
    elif variant == 7:  # spring boot: %d  %5p %pid --- [%15.15t] %-40.40logger{39} : %m
        L.add(ts, "TS").add("  ").add(" " * max(0, 5 - len(lw))).add(lw, "LEVEL").add(" ").add(str(rng.randint(1, 99999)), "THREAD").add(" --- [")
        L.add(" " * max(0, 15 - len(thread[:15]))).add(thread[:15], "THREAD").add("] ").add(logger[:40], "SOURCE").add(" " * max(0, 40 - len(logger[:40]))).add(" : ")
    elif variant == 8:  # hdfs (loghub): yymmdd hhmmss pid LEVEL comp: msg
        L.add(f"{rng.randint(10, 24):02d}{rng.randint(1, 12):02d}{rng.randint(1, 28):02d} {rng.randint(0, 23):02d}{rng.randint(0, 59):02d}{rng.randint(0, 59):02d}", "TS")
        L.add(" ").add(str(rng.randint(1, 9999)), "THREAD").add(" ").add(lw, "LEVEL").add(" ").add(logger, "SOURCE").add(": ")
    else:  # tomcat / j.u.l one-line: dd-MMM-yyyy HH:mm:ss.SSS LEVEL [thread] class.method msg
        L.add(timestamp(rng, 21), "TS").add(" ").add(lw, "LEVEL").add(" [").add(thread, "THREAD").add("] ").add(logger, "SOURCE").add(".").add(rng.choice(V.JAVA_METHODS)).add(" ")
    if rng.random() < 0.15:  # MDC
        L.add("[")
        add_kv(L, rng, n=rng.randint(1, 2), style="eq", sep=", ")
        L.add("] ")
    msg_tail(L, rng, 0.15)
    return L


def jul_two_line(rng: random.Random) -> list[Line]:
    """java.util.logging SimpleFormatter: header line then 'LEVEL: msg'."""
    a = Line("entry")
    a.add(timestamp(rng, 38), "TS").add(" ").add(rng.choice(V.JAVA_CLASSES).replace("java.base/", ""), "SOURCE").add(" ").add(rng.choice(V.JAVA_METHODS))
    b = Line("continuation")
    b.add(rng.choice(["INFO", "SEVERE", "WARNING", "FINE", "CONFIG"]), "LEVEL").add(": ").add(message(rng), "MSG")
    return [a, b]


# --- Python -------------------------------------------------------------------------------

def python_logging(rng: random.Random) -> Line:
    L = Line("entry")
    v = rng.randrange(9)
    lw = rng.choice(["DEBUG", "INFO", "INFO", "WARNING", "ERROR", "CRITICAL"])
    name = rng.choice(V.PY_LOGGERS) if rng.random() < 0.6 else py_logger(rng)
    if v == 0:  # basicConfig default: LEVEL:name:msg
        L.add(lw, "LEVEL").add(":").add(name, "SOURCE").add(":")
    elif v == 1:  # %(asctime)s - %(name)s - %(levelname)s - %(message)s
        L.add(timestamp(rng, 8), "TS").add(" - ").add(name, "SOURCE").add(" - ").add(lw, "LEVEL").add(" - ")
    elif v == 2:  # %(asctime)s %(levelname)s %(message)s
        L.add(timestamp(rng, rng.choice([8, 7])), "TS").add(" ").add(lw, "LEVEL").add(" ")
    elif v == 3:  # airflow: [%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s
        L.add("[").add(timestamp(rng, 8), "TS").add("] {").add(rng.choice(V.PY_FILES).split("/")[-1].split("\\")[-1], "FILE").add(":").add(str(rng.randint(1, 999)), "LINE").add("} ").add(lw, "LEVEL").add(" - ")
    elif v == 4:  # %(asctime)s [%(levelname)s] %(name)s: %(message)s
        L.add(timestamp(rng, rng.choice([8, 9, 1])), "TS").add(" [").add(lw, "LEVEL").add("] ").add(name, "SOURCE").add(": ")
    elif v == 5:  # gunicorn: [ts] [pid] [LEVEL] msg
        L.add("[").add(timestamp(rng, 11), "TS").add("] [").add(str(rng.randint(1, 99999)), "THREAD").add("] [").add(lw, "LEVEL").add("] ")
    elif v == 6:  # uvicorn: LEVEL:     msg
        L.add(lw, "LEVEL").add(":").add(" " * (9 - len(lw)))
        if rng.random() < 0.5:
            L.add(f"{ip(rng)}:{rng.randint(1000, 65535)} - \"").add(http_request(rng), "MSG").add("\" ").add(rng.choice(V.STATUSES), "VALUE").add(" ").add(rng.choice(["OK", "Not Found", "Internal Server Error", "Created"]))
            return L
    elif v == 7:  # celery: [ts: LEVEL/MainProcess] msg
        L.add("[").add(timestamp(rng, 8), "TS").add(": ").add(lw, "LEVEL").add("/").add(thread_name(rng), "THREAD").add("] ")
    else:  # %(asctime)s %(process)d %(levelname)s %(name)s [%(threadName)s] %(message)s  (openstack-ish)
        L.add(timestamp(rng, 9), "TS").add(" ").add(str(rng.randint(1, 99999)), "THREAD").add(" ").add(lw, "LEVEL").add(" ").add(name, "SOURCE").add(" [").add(rng.choice(["-", f"req-{uuid(rng)}", thread_name(rng)]), "THREAD").add("] ")
    msg_tail(L, rng, 0.1)
    return L


# --- Go ------------------------------------------------------------------------------------

def go_log(rng: random.Random) -> Line:
    L = Line("entry")
    v = rng.randrange(7)
    if v == 0:  # std log
        L.add(timestamp(rng, rng.choice([13, 14])), "TS").add(" ")
        if rng.random() < 0.4:
            L.add(rng.choice(V.GO_FILES).split("/")[-1], "FILE").add(":").add(str(rng.randint(1, 999)), "LINE").add(": ")
        if rng.random() < 0.3:
            L.add("[").add(rng.choice(V.GO_PKGS), "SOURCE").add("] ")
    elif v == 1:  # std log with prefix
        L.add(rng.choice(V.GO_PKGS + V.SERVICES), "SOURCE").add(rng.choice([": ", " ", "| "])).add(timestamp(rng, 13), "TS").add(" ")
    elif v == 2:  # glog / klog
        L.add(rng.choice("IIIWEF"), "LEVEL").add(timestamp(rng, 26), "TS").add(" " * rng.randint(1, 7)).add(str(rng.randint(1, 99999)), "THREAD").add(" ")
        L.add(rng.choice(V.GO_FILES).split("/")[-1], "FILE").add(":").add(str(rng.randint(1, 999)), "LINE").add("] ")
        if rng.random() < 0.4:  # klog structured
            L.add('"').add(message(rng), "MSG").add('"')
            if rng.random() < 0.8:
                L.add(" ")
                add_kv(L, rng, style="eq")
            return L
    elif v == 3:  # zap console: ts \t LEVEL \t caller \t msg \t {json}
        L.add(timestamp(rng, rng.choice([1, 5, 31])), "TS").add("\t").add(rng.choice(["INFO", "DEBUG", "WARN", "ERROR", "info", "warn", "error", "DPANIC", "FATAL"]), "LEVEL").add("\t")
        if rng.random() < 0.5:
            L.add(rng.choice(V.GO_PKGS), "SOURCE").add("\t")
        L.add(rng.choice(V.GO_PKGS) + "/" + rng.choice(V.GO_FILES).split("/")[-1], "FILE").add(":").add(str(rng.randint(1, 999)), "LINE").add("\t")
        L.add(message(rng), "MSG")
        if rng.random() < 0.6:
            L.add("\t{")
            for i in range(rng.randint(1, 4)):
                if i:
                    L.add(", ")
                k = rng.choice(["user", "id", "duration", "err", "path", "status", "count"])
                L.add('"').add(k, "KEY").add('": ')
                if k in ("id", "status", "count"):
                    L.add(str(rng.randint(0, 9999)), "VALUE")
                else:
                    L.add('"').add(rng.choice(V.USERS + V.ERRORS + V.PATHS + [duration(rng)]), "VALUE").add('"')
            L.add("}")
        return L
    elif v == 4:  # logrus text (non-pure logfmt because of padding/prefix)
        L.add(rng.choice(["time", "ts"]), "KEY").add('="').add(timestamp(rng, rng.choice([4, 1])), "VALUE").add('" ')
        L.add("level", "KEY").add("=").add(rng.choice(["info", "warning", "error", "debug", "fatal"]), "VALUE").add(" ")
        L.add("msg", "KEY").add('="').add(message(rng), "VALUE").add('"')
        if rng.random() < 0.8:
            L.add(" ")
            add_kv(L, rng, style="eq", sep=" ")
        if rng.random() < 0.5:
            return L
        L.parts.insert(0, (rng.choice(["app | ", "[worker] ", "2024/01/15 ", ">> "]), None))
        return L
    elif v == 5:  # logrus colored-ish text: LEVEL[0000] msg   key=value
        L.add(rng.choice(["INFO", "WARN", "ERRO", "DEBU", "FATA"]), "LEVEL").add("[").add(rng.choice([f"{rng.randint(0, 9999):04d}", timestamp(rng, 4)]), "TS").add("] ").add(message(rng), "MSG")
        if rng.random() < 0.7:
            L.add(" " * rng.randint(1, 20))
            add_kv(L, rng, style="eq", sep=" ")
        return L
    else:  # slog text with a non-kv prefix (pure logfmt is a fast path)
        L.add(rng.choice(["[app] ", "app: ", "", ""]))
        L.add("time", "KEY").add("=").add(timestamp(rng, 42), "VALUE").add(" ").add("level", "KEY").add("=").add(rng.choice(["INFO", "WARN", "ERROR", "DEBUG"]), "VALUE").add(" ")
        L.add("msg", "KEY").add("=")
        m = message(rng)
        if " " in m:
            L.add('"').add(m, "VALUE").add('"')
        else:
            L.add(m, "VALUE")
        if rng.random() < 0.8:
            L.add(" ")
            add_kv(L, rng, style="eq", sep=" ")
        return L
    if rng.random() < 0.3:
        lw, _ = level_word(rng)
        L.add("[").add(lw, "LEVEL").add("] ")
    msg_tail(L, rng, 0.2)
    return L


# --- Rust ---------------------------------------------------------------------------------

def rust_log(rng: random.Random) -> Line:
    L = Line("entry")
    v = rng.randrange(5)
    lw = rng.choice(["INFO", "INFO", "DEBUG", "WARN", "ERROR", "TRACE"])
    target = rng.choice(V.RUST_TARGETS)
    if v == 0:  # env_logger: [ts LEVEL target] msg
        L.add("[").add(timestamp(rng, rng.choice([0, 1])), "TS").add(" ").add(lw, "LEVEL").add(" " * max(1, 6 - len(lw))).add(target, "SOURCE").add("] ")
    elif v == 1:  # env_logger without ts: [LEVEL target] msg
        L.add("[").add(lw, "LEVEL").add(" ").add(target, "SOURCE").add("] ")
    elif v == 2:  # tracing fmt: ts  LEVEL target: msg k=v
        L.add(timestamp(rng, rng.choice([2, 3])), "TS").add(" ").add(" " * max(1, 6 - len(lw))).add(lw, "LEVEL").add(" ")
        if rng.random() < 0.3:
            L.add("ThreadId(").add(f"{rng.randint(1, 20):02d}", "THREAD").add(") ")
        if rng.random() < 0.4:  # span
            L.add(rng.choice(["request", "handle", "conn", "worker"])).add("{")
            add_kv(L, rng, n=rng.randint(1, 2), style="eq", sep=" ")
            L.add("}: ")
        L.add(target, "SOURCE").add(": ")
        L.add(message(rng), "MSG")
        if rng.random() < 0.5:
            L.add(" ")
            add_kv(L, rng, style="eq", sep=" ")
        return L
    elif v == 3:  # tracing compact/pretty-ish
        L.add(timestamp(rng, 2), "TS").add(" ").add(lw, "LEVEL").add(" ").add(target, "SOURCE").add(": ")
    else:  # pretty_env_logger / simple: LEVEL - msg  or  LEVEL target > msg
        L.add(lw, "LEVEL").add(rng.choice([" - ", ": ", " "])).add(target, "SOURCE").add(rng.choice([" > ", ": ", " - "]))
    msg_tail(L, rng, 0.2)
    return L


# --- Node -----------------------------------------------------------------------------------

def node_log(rng: random.Random) -> Line:
    L = Line("entry")
    v = rng.randrange(9)
    lw = rng.choice(["INFO", "WARN", "ERROR", "DEBUG", "info", "warn", "error", "debug", "verbose", "silly", "http", "LOG", "FATAL", "TRACE"])
    if v == 0:  # pino-pretty: [ts] LEVEL (pid on host): msg
        L.add("[").add(timestamp(rng, rng.choice([29, 1, 39])), "TS").add("] ").add(lw.upper(), "LEVEL").add(" (")
        if rng.random() < 0.5:
            L.add(rng.choice(V.NODE_MODULES), "SOURCE").add("/")
        L.add(str(rng.randint(1, 99999)), "THREAD").add(" on ").add(hostname(rng), "HOST").add("): ")
    elif v == 1:  # winston: ts [level]: msg  / level: msg
        if rng.random() < 0.6:
            L.add(timestamp(rng, rng.choice([1, 7])), "TS").add(" ")
        L.add(rng.choice(["[", ""]))
        L.add(lw.lower(), "LEVEL").add(rng.choice(["]: ", ": "]))
        L.add(message(rng), "MSG")
        if rng.random() < 0.3:
            L.add(" {")
            for i in range(rng.randint(1, 3)):
                if i:
                    L.add(",")
                L.add('"').add(rng.choice(["service", "requestId", "user", "durationMs"]), "KEY").add('":"').add(identifier(rng), "VALUE").add('"')
            L.add("}")
        return L
    elif v == 2:  # bunyan pretty: [ts]  INFO: app/pid on host: msg
        L.add("[").add(timestamp(rng, 1), "TS").add("] ").add(" " * max(0, 5 - len(lw))).add(lw.upper(), "LEVEL").add(": ").add(rng.choice(V.NODE_MODULES), "SOURCE").add("/").add(str(rng.randint(1, 99999)), "THREAD").add(" on ").add(hostname(rng), "HOST").add(": ")
    elif v == 3:  # nest: [Nest] pid  - date, time     LOG [Context] msg
        L.add("[Nest] ").add(str(rng.randint(1, 99999)), "THREAD").add("  - ").add(timestamp(rng, 23).replace(" ", ", ", 1), "TS").add("     ").add(rng.choice(["LOG", "ERROR", "WARN", "DEBUG", "VERBOSE"]), "LEVEL").add(" [").add(rng.choice(V.NODE_MODULES), "SOURCE").add("] ")
    elif v == 4:  # debug module: namespace msg +Nms
        L.add(rng.choice(V.NODE_MODULES), "SOURCE").add(" ").add(message(rng), "MSG").add(" +").add(duration(rng))
        return L
    elif v == 5:  # npm
        L.add("npm ").add(rng.choice(["WARN", "ERR!", "verb", "info", "notice", "http", "sill"]), "LEVEL").add(" ")
        L.add(rng.choice(["deprecated", "config", "fetch", "audit", "cli", "lifecycle", ""]), "SOURCE").add(" ")
    elif v == 6:  # next.js / vite style
        L.add(rng.choice(["✓ ", "○ ", "⚠ ", "✗ ", "- ", "> ", "▲ ", "  ➜  ", "[vite] ", "event - ", "wait  - ", "warn  - ", "error - ", "info  - "]))
        L.add(message(rng), "MSG")
        return L
    elif v == 7:  # console.log with prefix
        L.add(rng.choice(["[server] ", "[app] ", "[worker] ", "[api] ", "server: ", "app> "])).add(message(rng), "MSG")
        if rng.random() < 0.3:
            L.add(" ")
            add_kv(L, rng, style="eq")
        return L
    else:  # ts level  msg
        L.add(timestamp(rng, rng.choice([1, 9])), "TS").add(" ").add(lw, "LEVEL").add(" " * rng.randint(1, 3))
    msg_tail(L, rng, 0.2)
    return L


# --- containers / cloud --------------------------------------------------------------------

INNER: list[Builder] = []  # filled at bottom


def inner_line(rng: random.Random) -> Line:
    return rng.choice(INNER)(rng)


def cri_log(rng: random.Random) -> Line:
    """containerd/CRI: ts stream flag inner."""
    L = Line("entry")
    L.add(timestamp(rng, rng.choice([3, 3, 2])), "TS").add(" ").add(rng.choice(["stdout", "stderr"])).add(" ").add(rng.choice(["F", "F", "P"])).add(" ")
    inner = inner_line(rng)
    L.parts.extend(inner.parts)
    return L


def kubectl_ts(rng: random.Random) -> Line:
    L = Line("entry")
    L.add(timestamp(rng, 3), "TS").add(" ")
    inner = inner_line(rng)
    L.parts.extend(inner.parts)
    return L


def compose_prefix(rng: random.Random) -> Line:
    L = Line("entry")
    name = rng.choice([f"{rng.choice(['web', 'db', 'api', 'redis', 'worker', 'nginx'])}_{rng.randint(1, 3)}", f"{rng.choice(V.SERVICES)}-{rng.randint(1, 3)}",
                       f"{rng.randint(0, 9)}|{rng.choice(V.SERVICES)}", f"[pod/{rng.choice(V.PODS)}/{rng.choice(['app', 'sidecar', 'istio-proxy'])}]"])
    L.add(name, "SOURCE").add(" " * rng.randint(0, 6)).add("| ")
    inner = inner_line(rng)
    L.parts.extend(inner.parts)
    return L


def heroku(rng: random.Random) -> Line:
    L = Line("entry")
    L.add(timestamp(rng, 6), "TS").add(" ")
    if rng.random() < 0.5:
        L.add("heroku[router]", "SOURCE").add(": ")
        L.add("at", "KEY").add("=").add(rng.choice(["info", "error", "warning"]), "VALUE").add(" ")
        if rng.random() < 0.2:
            L.add("code", "KEY").add("=").add(rng.choice(["H12", "H10", "H13", "H18"]), "VALUE").add(" ").add("desc", "KEY").add('="').add(rng.choice(["Request timeout", "App crashed", "Connection closed without response"]), "VALUE").add('" ')
        L.add("method", "KEY").add("=").add(rng.choice(V.METHODS), "VALUE").add(" ").add("path", "KEY").add('="').add(rng.choice(V.PATHS), "VALUE").add('" ')
        L.add("host", "KEY").add("=").add(f"{word(rng)}.herokuapp.com", "VALUE").add(" ").add("request_id", "KEY").add("=").add(uuid(rng), "VALUE").add(" ")
        L.add("fwd", "KEY").add('="').add(ip(rng), "VALUE").add('" ').add("dyno", "KEY").add("=").add(f"web.{rng.randint(1, 8)}", "VALUE").add(" ")
        L.add("connect", "KEY").add("=").add(f"{rng.randint(0, 9)}ms", "VALUE").add(" ").add("service", "KEY").add("=").add(f"{rng.randint(1, 30000)}ms", "VALUE").add(" ")
        L.add("status", "KEY").add("=").add(rng.choice(V.STATUSES), "VALUE").add(" ").add("bytes", "KEY").add("=").add(str(rng.randint(0, 99999)), "VALUE").add(" ").add("protocol", "KEY").add("=").add(rng.choice(["https", "http"]), "VALUE")
        return L
    L.add(rng.choice([f"app[web.{rng.randint(1, 8)}]", f"app[worker.{rng.randint(1, 4)}]", "app[api]", "heroku[web.1]", "app[scheduler.1234]", "heroku[run.1234]"]), "SOURCE").add(": ")
    inner = inner_line(rng) if rng.random() < 0.5 else None
    if inner:
        L.parts.extend(inner.parts)
    else:
        msg_tail(L, rng, 0.2)
    return L


def cloudwatch(rng: random.Random) -> Line:
    L = Line("entry")
    v = rng.randrange(6)
    rid = uuid(rng)
    if v == 0:
        L.add(timestamp(rng, 1), "TS").add("\t").add(rid, "THREAD").add("\t").add(rng.choice(["INFO", "ERROR", "WARN", "DEBUG"]), "LEVEL").add("\t")
        msg_tail(L, rng, 0.2, sep=" ")
    elif v == 1:
        L.add("START RequestId: ").add(rid, "THREAD").add(" Version: ").add(rng.choice(["$LATEST", "1", "42"]))
    elif v == 2:
        L.add("END RequestId: ").add(rid, "THREAD")
    elif v == 3:
        L.add("REPORT RequestId: ").add(rid, "THREAD").add("\t")
        for k, val in [("Duration", f"{rng.randint(1, 99999) / 100:.2f} ms"), ("Billed Duration", f"{rng.randint(1, 9999)} ms"), ("Memory Size", f"{rng.choice([128, 256, 512, 1024])} MB"), ("Max Memory Used", f"{rng.randint(20, 900)} MB")]:
            L.add(k, "KEY").add(": ").add(val, "VALUE").add("\t")
        if rng.random() < 0.5:
            L.add("Init Duration", "KEY").add(": ").add(f"{rng.randint(100, 3000) / 100:.2f} ms", "VALUE").add("\t")
        L.parts.pop()
    elif v == 4:  # console export prefix
        L.add(timestamp(rng, 5), "TS").add(" ")
        L.parts.extend(inner_line(rng).parts)
    else:  # log stream prefix
        L.add(f"{rng.randint(2018, 2026)}/{rng.randint(1, 12):02d}/{rng.randint(1, 28):02d}/[$LATEST]{hexid(rng, 32)}").add(" ")
        L.add(timestamp(rng, 1), "TS").add(" ")
        L.parts.extend(inner_line(rng).parts)
    return L


# --- assorted services -------------------------------------------------------------------

def postgres(rng: random.Random) -> Line:
    L = Line("entry")
    L.add(timestamp(rng, 12), "TS").add(" [").add(str(rng.randint(1, 99999)), "THREAD").add("] ")
    if rng.random() < 0.5:
        L.add(rng.choice(V.USERS)).add("@").add(rng.choice(["app", "postgres", "mydb"])).add(" ")
    L.add(rng.choice(["LOG", "LOG", "ERROR", "FATAL", "WARNING", "STATEMENT", "DETAIL", "HINT", "PANIC", "NOTICE"]), "LEVEL").add(":  ")
    msg_tail(L, rng, 0.05)
    return L


def mysql(rng: random.Random) -> Line:
    L = Line("entry")
    L.add(timestamp(rng, 2), "TS").add(" ").add(str(rng.randint(0, 99)), "THREAD").add(" [").add(rng.choice(["Note", "Warning", "ERROR", "System"]), "LEVEL").add("] ")
    if rng.random() < 0.7:
        L.add(f"[MY-{rng.randint(10000, 15000):06d}] [").add(rng.choice(["Server", "InnoDB", "Repl"]), "SOURCE").add("] ")
    msg_tail(L, rng, 0.05)
    return L


def redis(rng: random.Random) -> Line:
    L = Line("entry")
    L.add(str(rng.randint(1, 99999)), "THREAD").add(":").add(rng.choice("MCSX")).add(" ").add(timestamp(rng, 35), "TS").add(" ").add(rng.choice([".", "-", "*", "#"]), "LEVEL").add(" ")
    msg_tail(L, rng, 0.05)
    return L


def dotnet(rng: random.Random) -> list[Line]:
    a = Line("entry")
    a.add(rng.choice(["info", "warn", "fail", "dbug", "trce", "crit"]), "LEVEL").add(": ").add(rng.choice(["Microsoft.Hosting.Lifetime", "Microsoft.AspNetCore.Hosting.Diagnostics", "App.Services.Worker", "Microsoft.EntityFrameworkCore.Database.Command", "System.Net.Http.HttpClient.Default.LogicalHandler"]), "SOURCE").add("[").add(str(rng.randint(0, 99)), "THREAD").add("]")
    b = Line("continuation")
    b.add("      ").add(message(rng), "MSG")
    return [a, b]


def serilog(rng: random.Random) -> Line:
    L = Line("entry")
    L.add(timestamp(rng, 39), "TS").add(" [").add(rng.choice(["INF", "WRN", "ERR", "DBG", "VRB", "FTL"]), "LEVEL").add("] ")
    if rng.random() < 0.3:
        L.add("[").add(rng.choice(["App.Program", "Microsoft.Hosting.Lifetime", "App.Api.Controllers.UsersController"]), "SOURCE").add("] ")
    msg_tail(L, rng, 0.1)
    return L


def php_error(rng: random.Random) -> Line:
    L = Line("entry")
    if rng.random() < 0.7:
        L.add("[").add(timestamp(rng, rng.choice([21, 18])).replace(" ", "-", 1) if rng.random() < 0.5 else timestamp(rng, 18), "TS").add("] ")
    L.add("PHP ").add(rng.choice(["Warning", "Notice", "Fatal error", "Parse error", "Deprecated"]), "LEVEL").add(":  ")
    L.add(message(rng), "MSG").add(" in ").add(rng.choice(V.PHP_FILES), "FILE").add(" on line ").add(str(rng.randint(1, 999)), "LINE")
    return L


def logcat(rng: random.Random) -> Line:
    L = Line("entry")
    v = rng.randrange(3)
    tag = rng.choice(["ActivityManager", "WindowManager", "PowerManagerService", "dalvikvm", "art", "System.err", "Zygote", "chromium", "OkHttp", "ReactNativeJS", "Unity", "AndroidRuntime", "libc", "audit"])
    if v == 0:  # threadtime
        L.add(timestamp(rng, 27), "TS").add(" " * rng.randint(1, 2)).add(str(rng.randint(1, 32000)), "THREAD").add(" " * rng.randint(1, 2)).add(str(rng.randint(1, 32000)), "THREAD").add(" ").add(rng.choice("VDIWEF"), "LEVEL").add(" ").add(tag, "SOURCE").add(": ")
    elif v == 1:  # brief
        L.add(rng.choice("VDIWEF"), "LEVEL").add("/").add(tag, "SOURCE").add("(").add(f"{rng.randint(1, 32000):5d}", "THREAD").add("): ")
    else:  # time
        L.add(timestamp(rng, 27), "TS").add(" ").add(rng.choice("VDIWEF"), "LEVEL").add("/").add(tag, "SOURCE").add("(").add(f"{rng.randint(1, 32000):5d}", "THREAD").add("): ")
    msg_tail(L, rng, 0.05)
    return L


def dmesg(rng: random.Random) -> Line:
    L = Line("entry")
    if rng.random() < 0.7:
        L.add("[").add(timestamp(rng, 41), "TS").add("] ")
    else:
        L.add("[").add(timestamp(rng, 18), "TS").add("] ")
    if rng.random() < 0.6:
        L.add(rng.choice(["usb 1-1", "eth0", "EXT4-fs (sda1)", "systemd", "kernel", "audit", "NVRM", "nvme nvme0", "Bluetooth", "wlan0", "iwlwifi 0000:00:14.3", "docker0"]), "SOURCE").add(": ")
    msg_tail(L, rng, 0.05)
    return L


def generic_entry(rng: random.Random) -> Line:
    """Ad-hoc formats: ts level msg with random separators/brackets."""
    L = Line("entry")
    seps = [" ", " ", "  ", " | ", " - ", "\t", " :: ", " » "]
    sep = rng.choice(seps)
    order = rng.choice(["ts level src msg", "ts level msg", "level ts msg", "ts msg", "level msg", "ts src level msg", "src level msg",
                        "ts thread level src msg", "ts level src thread msg", "src ts level msg", "level src ts msg", "ts level thread msg"])
    for part in order.split():
        if part == "ts":
            o, c = bracket(rng)
            L.add(o).add(timestamp(rng), "TS").add(c).add(sep)
        elif part == "level":
            lw, _ = level_word(rng)
            o, c = bracket(rng)
            lw, pad = pad_level(rng, lw)
            L.add(o).add(lw, "LEVEL").add(c).add(pad).add(rng.choice([sep, ":" + sep, sep]))
        elif part == "src":
            o, c = bracket(rng)
            L.add(o).add(source_name(rng), "SOURCE").add(c).add(rng.choice([sep, ": ", " - "]))
        elif part == "thread":
            o, c = bracket(rng) if rng.random() < 0.8 else ("", "")
            L.add(o).add(thread_name(rng), "THREAD").add(c).add(sep)
        else:
            msg_tail(L, rng, 0.3)
    return L


def plain_line(rng: random.Random) -> Line:
    """No prefix at all: still an entry with a message."""
    L = Line("entry")
    L.add(message(rng), "MSG")
    if rng.random() < 0.2:
        L.add(" ")
        add_kv(L, rng)
    return L


def logfmt_mixed(rng: random.Random) -> Line:
    """Mostly-logfmt lines that are NOT pure logfmt (prefix or free text)."""
    L = Line("entry")
    if rng.random() < 0.5:
        L.add(timestamp(rng), "TS").add(" ")
    if rng.random() < 0.4:
        lw, _ = level_word(rng)
        L.add(lw, "LEVEL").add(" ")
    if rng.random() < 0.6:
        L.add(message(rng), "MSG").add(" ")
    add_kv(L, rng, n=rng.randint(2, 8))
    return L


ENTRY_BUILDERS: list[tuple[Builder, float]] = [
    (syslog_3164, 7), (syslog_5424, 3), (journal, 3), (access_log, 5), (nginx_error, 3), (apache_error, 3),
    (log4j, 9), (python_logging, 6), (go_log, 6), (rust_log, 4), (node_log, 5), (cri_log, 2), (kubectl_ts, 1),
    (compose_prefix, 2), (heroku, 2), (cloudwatch, 3), (postgres, 2), (mysql, 1), (redis, 1), (serilog, 1),
    (php_error, 1.5), (logcat, 2), (dmesg, 1.5), (generic_entry, 8), (plain_line, 3), (logfmt_mixed, 4),
]
MULTI_BUILDERS: list[tuple[Callable[[random.Random], list[Line]], float]] = [(jul_two_line, 1), (dotnet, 1)]
INNER.extend([log4j, python_logging, go_log, rust_log, node_log, generic_entry, plain_line, logfmt_mixed, syslog_3164])


# --- v2 coverage: supercomputer, Windows, Proxifier, bracket variants, nested kv, prose kv ---

def cluster_node(rng: random.Random) -> str:
    return rng.choice([f"R{rng.randint(0, 99):02d}-M{rng.randint(0, 1)}-N{rng.randint(0, 15)}-C:J{rng.randint(0, 17):02d}-U{rng.randint(1, 11):02d}",
                       f"node-{rng.randint(0, 1023)}", f"dn{rng.randint(1, 999)}", f"cn{rng.randint(1, 9999)}", f"tbird-admin{rng.randint(1, 9)}",
                       f"sn{rng.randint(1, 999)}", f"an{rng.randint(1, 99)}", f"nid{rng.randint(1000, 99999):05d}", f"c{rng.randint(0, 9)}-{rng.randint(0, 9)}c{rng.randint(0, 2)}s{rng.randint(0, 15)}"])


def supercomputer(rng: random.Random) -> Line:
    """Cluster / supercomputer RAS logs: alert flag, epoch, dotted date, node, timestamp, node, facility, component, level, msg."""
    L = Line("entry")
    v = rng.randrange(4)
    node = cluster_node(rng)
    if v == 0:  # BGL-like
        L.add(rng.choice(["- ", "- ", "KERNDTLB ", "APPREAD ", "KERNRTSP "])).add(str(rng.randint(10**9, 2 * 10**9))).add(" ")
        L.add(f"{rng.randint(2003, 2026)}.{rng.randint(1, 12):02d}.{rng.randint(1, 28):02d}").add(" ").add(node, "HOST").add(" ")
        L.add(timestamp(rng, 45), "TS").add(" ").add(node, "HOST").add(" ")
        L.add(rng.choice(["RAS", "RAS", "RAS", "NULL", "BGLMASTER"])).add(" ").add(rng.choice(["KERNEL", "APP", "DISCOVERY", "HARDWARE", "LINKCARD", "MMCS", "MONITOR", "CMCS"]), "SOURCE").add(" ")
        L.add(rng.choice(["INFO", "FATAL", "WARNING", "ERROR", "SEVERE", "FAILURE"]), "LEVEL").add(" ")
    elif v == 1:  # HPC-like: id node component state epoch flag msg
        L.add(str(rng.randint(1, 9999999))).add(" ").add(node, "HOST").add(" ")
        L.add(rng.choice(["unix.hw", "node", "action", "boot_cmd", "unix.sw", "cpu.hw", "mem.hw", "net.sw"]), "SOURCE").add(" ")
        L.add(rng.choice(["state_change.unavailable", "start", "end", "cmd", "node", "running", "unavailable", "state_change.available"])).add(" ")
        L.add(timestamp(rng, 30), "TS").add(" ").add(str(rng.randint(1, 9))).add(" ")
    elif v == 2:  # Thunderbird-like: - epoch yyyy.mm.dd host <syslog line>
        L.add("- ").add(str(rng.randint(10**9, 2 * 10**9))).add(" ").add(f"{rng.randint(2003, 2026)}.{rng.randint(1, 12):02d}.{rng.randint(1, 28):02d}").add(" ")
        L.add(node, "HOST").add(" ").add(timestamp(rng, 16), "TS").add(" ")
        L.add(rng.choice([f"{node}/{node}", f"local@{node}", node, f"src@{node}"]), "HOST").add(" ")
        L.add(rng.choice(V.PROCS + ["crond(pam_unix)", "/apps/x86_64/system/ganglia-3.0.1/sbin/gmetad", "sshd(pam_unix)", "ntpd", "kernel", "pbs_mom"]), "SOURCE")
        pid_suffix(L, rng, 0.7)
        L.add(": ")
    else:  # Slurm / PBS-like
        L.add("[").add(timestamp(rng, 2), "TS").add("] ")
        if rng.random() < 0.5:
            L.add(rng.choice(["error", "debug", "info", "fatal", "verbose", "debug2"]), "LEVEL").add(": ")
        L.add(rng.choice(["slurmctld", "slurmd", "slurmstepd", "_slurm_rpc_node_registration", "pbs_server", "job_scheduler"]), "SOURCE").add(": ")
    msg_tail(L, rng, 0.1)
    return L


def windows_cbs(rng: random.Random) -> Line:
    """Windows CBS/CSI/setupapi style: 'yyyy-mm-dd hh:mm:ss, Level   Component   msg'."""
    L = Line("entry")
    L.add(timestamp(rng, 7), "TS").add(", ")
    lw = rng.choice(["Info", "Info", "Info", "Error", "Warning", "Verbose"])
    L.add(lw, "LEVEL").add(" " * max(1, 22 - len(lw)))
    L.add(rng.choice(["CBS", "CSI", "CBS", "DPX", "DISM", "WcpInitialize", "SQM", "TI", "SetupAPI", "Session"]), "SOURCE").add(" " * rng.randint(1, 6))
    if rng.random() < 0.3:
        L.add(f"{rng.randint(1, 999999):08d}@{rng.randint(2010, 2026)}/{rng.randint(1, 12)}/{rng.randint(1, 28)}:{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}.{rng.randint(0, 999):03d} ")
    msg_tail(L, rng, 0.05)
    return L


def windows_event(rng: random.Random) -> list[Line]:
    """Windows Event Viewer text export: one field per line."""
    out: list[Line] = []
    out.append(Line("entry").add("Log Name:      ").add(rng.choice(["System", "Application", "Security", "Setup", "Microsoft-Windows-Sysmon/Operational"]), "VALUE"))
    out.append(Line("continuation").add("Source:        ").add(rng.choice(["Service Control Manager", "Microsoft-Windows-Kernel-General", "Windows Error Reporting", "Application Error", "Microsoft-Windows-Security-Auditing", "EventLog", "Windows PowerShell", "MSSQLSERVER", "DCOM", "Disk"]), "SOURCE"))
    out.append(Line("continuation").add("Date:          ").add(timestamp(rng, 23), "TS"))
    out.append(Line("continuation").add("Event ID:      ").add(str(rng.randint(1, 9999)), "VALUE"))
    out.append(Line("continuation").add("Task Category: ").add(rng.choice(["None", "Logon", "(1)", "Process Creation"]), "VALUE"))
    out.append(Line("continuation").add("Level:         ").add(rng.choice(["Information", "Error", "Warning", "Critical", "Verbose"]), "LEVEL"))
    out.append(Line("continuation").add("Keywords:      ").add(rng.choice(["Classic", "Audit Success", "Audit Failure", "(70368744177664)"]), "VALUE"))
    out.append(Line("continuation").add("User:          ").add(rng.choice(["N/A", "SYSTEM", "NT AUTHORITY\\SYSTEM", f"DOMAIN\\{rng.choice(V.USERS)}"]), "VALUE"))
    out.append(Line("continuation").add("Computer:      ").add(rng.choice([f"{word(rng).upper()}-PC", f"DESKTOP-{hexid(rng, 7).upper()}", f"srv{rng.randint(1, 99)}.corp.local"]), "HOST"))
    out.append(Line("continuation").add("Description:"))
    out.append(Line("continuation").add(message(rng), "MSG"))
    return out


def proxifier(rng: random.Random) -> Line:
    L = Line("entry")
    L.add("[").add(f"{rng.randint(1, 12):02d}.{rng.randint(1, 31):02d} {rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}", "TS").add("] ")
    L.add(rng.choice(["chrome.exe", "firefox.exe", "python.exe", "ssh.exe", "Dropbox.exe", "Skype.exe", "outlook.exe", "curl.exe", "java.exe", "Code.exe"]), "SOURCE")
    if rng.random() < 0.5:
        L.add(" *").add(str(rng.randint(1, 999)))
    L.add(" - ")
    host = rng.choice(V.HOSTS + ["proxy.cse.cuhk.edu.hk", "update.microsoft.com", "api.dropbox.com"])
    port = rng.randint(80, 65535)
    v = rng.random()
    if v < 0.4:
        L.add(f"{host}:{port} open through proxy {rng.choice(V.HOSTS)}:{rng.randint(1000, 9999)} {rng.choice(['HTTPS', 'SOCKS5', 'HTTP'])}", "MSG")
    elif v < 0.8:
        L.add(f"{host}:{port} close, {rng.randint(0, 99999)} bytes sent, {rng.randint(0, 999999)} bytes received, lifetime {rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}", "MSG")
    else:
        L.add(f"{host}:{port} error : Could not connect through proxy {rng.choice(V.HOSTS)}:{rng.randint(1000, 9999)} - {rng.choice(V.ERRORS)}", "MSG")
    return L


MULTI_WORD_SOURCES = ["Service Control Manager", "Windows PowerShell", "Application Error", "Group Policy", "Task Scheduler", "Windows Update Agent",
                      "User Profile Service", "Disk Manager", "Print Spooler", "Remote Desktop Services", "Audit Service", "Cluster Service",
                      "SQL Server", "Volume Shadow Copy", "Network Manager", "Bluetooth Daemon", "Power Manager", "Step Counter", "Health Kit"]


def bracket_variants(rng: random.Random) -> Line:
    """Sources and threads in every wrapper: (src), <thread>, {thread}, [src:thread], [thread|src], src(pid), src/pid, [pid/tid]."""
    L = Line("entry")
    if rng.random() < 0.85:
        o, c = bracket(rng)
        L.add(o).add(timestamp(rng), "TS").add(c).add(rng.choice([" ", "  ", " | ", "\t"]))
    src = rng.choice(MULTI_WORD_SOURCES) if rng.random() < 0.35 else source_name(rng)
    thr = thread_name(rng)
    lw, _ = level_word(rng)
    v = rng.randrange(10)
    if v == 0:
        L.add("(").add(src, "SOURCE").add(") ").add(lw, "LEVEL").add(" ")
    elif v == 1:
        L.add(lw, "LEVEL").add(" <").add(thr, "THREAD").add("> ").add(src, "SOURCE").add(": ")
    elif v == 2:
        L.add("{").add(thr, "THREAD").add("} [").add(lw, "LEVEL").add("] ").add(src, "SOURCE").add(" - ")
    elif v == 3:
        L.add("[").add(src, "SOURCE").add(":").add(thr, "THREAD").add("] ").add(lw, "LEVEL").add(": ")
    elif v == 4:
        L.add("[").add(thr, "THREAD").add("|").add(src, "SOURCE").add("] ").add(lw, "LEVEL").add(" ")
    elif v == 5:
        L.add(lw, "LEVEL").add(" ").add(src, "SOURCE").add("(").add(str(rng.randint(1, 99999)), "THREAD").add("): ")
    elif v == 6:
        L.add(src, "SOURCE").add("/").add(str(rng.randint(1, 99999)), "THREAD").add(" ").add(lw, "LEVEL").add(": ")
    elif v == 7:
        L.add("[").add(str(rng.randint(1, 99999)), "THREAD").add("/").add(str(rng.randint(1, 99999)), "THREAD").add("] ").add(lw, "LEVEL").add(" ").add(src, "SOURCE").add(": ")
    elif v == 8:
        L.add("<").add(lw, "LEVEL").add("> ").add("[").add(src, "SOURCE").add("] ").add("(").add(thr, "THREAD").add(") ")
    else:
        L.add(src, "SOURCE").add(" [").add(thr, "THREAD").add("] <").add(lw, "LEVEL").add("> ")
    msg_tail(L, rng, 0.3)
    return L


def nested_value(rng: random.Random) -> str:
    """Values with braces, brackets and quotes inside (JSON blobs, maps, lists)."""
    v = rng.random()
    if v < 0.3:
        return "{" + ", ".join(f'"{rng.choice(gen_words())}": {rng.choice([str(rng.randint(0, 999)), "true", "null", f"\"{word(rng)}\"", "{\"x\": 1}"])}' for _ in range(rng.randint(1, 3))) + "}"
    if v < 0.5:
        return "{" + ", ".join(f"{rng.choice(gen_words())}={rng.choice([str(rng.randint(0, 999)), word(rng)])}" for _ in range(rng.randint(1, 3))) + "}"
    if v < 0.7:
        return "[" + ", ".join(rng.choice([word(rng), str(rng.randint(0, 999)), f'"{word(rng)}"']) for _ in range(rng.randint(1, 4))) + "]"
    if v < 0.85:
        return f"({rng.choice(gen_words())}={rng.randint(0, 99)}, {rng.choice(gen_words())}={word(rng)})"
    return f"<{word(rng)} {rng.choice(gen_words())}={rng.randint(0, 99)}>"


def gen_words() -> list[str]:
    from .gen import WORDS
    return WORDS


def nested_kv(rng: random.Random) -> Line:
    """Entry with a kv tail whose values are quoted strings with spaces or nested structures."""
    L = Line("entry")
    if rng.random() < 0.7:
        L.add(timestamp(rng), "TS").add(" ")
    if rng.random() < 0.6:
        lw, _ = level_word(rng)
        L.add(lw, "LEVEL").add(" ")
    if rng.random() < 0.5:
        L.add(source_name(rng), "SOURCE").add(rng.choice([": ", " - ", " "]))
    L.add(message(rng), "MSG")
    sep = rng.choice([" ", " ", ", ", "  ", "\t"])
    for i in range(rng.randint(1, 4)):
        L.add(sep if i else " ")
        L.add(rng.choice(["data", "payload", "opts", "tags", "labels", "meta", "ctx", "args", "params", "headers", "body", "extra", "fields", "attrs"]), "KEY")
        L.add(rng.choice(["=", "=", ": ", "="]))
        val = nested_value(rng)
        if rng.random() < 0.3:
            q = rng.choice(['"', "'"])
            L.add(q).add(val.replace(q, ""), "VALUE").add(q)
        else:
            L.add(val, "VALUE")
    return L


PROSE_KV_TEMPLATES = [
    "Set {word}={num} for client {ip}",
    "Using timeout={num}s and retries={num}",
    "Connected with user={user} host={host} in {dur}",
    "Rejected: status={status} reason={word}",
    "Values: min={num} max={num} avg={num}",
    "Starting with config={path} and port={port}",
    "Ignoring option {word}={word} (deprecated)",
    "{Word} finished: ok={num} failed={num} skipped={num}",
    "request_id={ident} not found in cache",
    "Node {host} reported load={num} mem={size}",
    "Setting {word} = {num} (was {num})",
    "watch: {word}={word}, {word}={num}",
    "{word}: {num} -> {num} ({word}={dur})",
    "Applied {word}={word} to {num} items",
]


def prose_kv(rng: random.Random) -> Line:
    """key=value inside the sentence is message text, not a kv tail (sometimes followed by a real tail)."""
    from .gen import fill
    L = Line("entry")
    if rng.random() < 0.8:
        L.add(timestamp(rng), "TS").add(" ")
    if rng.random() < 0.7:
        lw, _ = level_word(rng)
        o, c = bracket(rng)
        L.add(o).add(lw, "LEVEL").add(c).add(" ")
    if rng.random() < 0.5:
        L.add(source_name(rng), "SOURCE").add(rng.choice([": ", " - ", " "]))
    L.add(fill(rng, rng.choice(PROSE_KV_TEMPLATES)), "MSG")
    if rng.random() < 0.35:
        L.add(rng.choice(["  ", " | ", " -- ", "\t"]))
        add_kv(L, rng, n=rng.randint(1, 3), style="eq", sep=" ")
    return L


def syslog_wrapped(rng: random.Random) -> Line:
    """syslog prefix wrapping a structured application line (vault/consul/docker via journald)."""
    L = Line("entry")
    L.add(timestamp(rng, rng.choice([15, 16, 4])), "TS").add(" ").add(hostname(rng), "HOST").add(" ")
    L.add(rng.choice(V.PROCS + V.SERVICES), "SOURCE")
    pid_suffix(L, rng, 0.8)
    L.add(": ")
    inner = rng.choice([log4j, python_logging, go_log, rust_log, node_log, generic_entry, bracket_variants, nested_kv])(rng)
    L.parts.extend(inner.parts)
    return L


ENTRY_BUILDERS.extend([(supercomputer, 4), (windows_cbs, 2), (proxifier, 1.5), (bracket_variants, 5), (nested_kv, 3), (prose_kv, 3), (syslog_wrapped, 2)])
MULTI_BUILDERS.append((windows_event, 0.5))
INNER.extend([bracket_variants, nested_kv])
