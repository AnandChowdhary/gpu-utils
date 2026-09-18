"""Stack traces and multi-line continuations, one Line per physical line."""

from __future__ import annotations

import random
from collections.abc import Callable

from . import vocab as V
from .gen import Line, add_kv, hexid, message, sql, timestamp, word


def indent(rng: random.Random, default: str) -> str:
    return rng.choice([default, default, default, "\t", "    ", "  ", "        "])


def java_trace(rng: random.Random) -> list[Line]:
    out: list[Line] = []
    exc = rng.choice(V.JAVA_EXC)
    head = Line("continuation")
    v = rng.random()
    if v < 0.4:
        head.add("Exception in thread \"").add(rng.choice(["main", "pool-1-thread-2", "http-nio-8080-exec-3"]), "THREAD").add("\" ")
    elif v < 0.6:
        head.add(rng.choice(["Caused by: ", "Suppressed: ", "Wrapped by: "]))
    head.add(exc, "MSG") if rng.random() < 0.5 else head.add(exc).add(": ").add(message(rng), "MSG")
    out.append(head)
    ind = indent(rng, "\t")
    for _ in range(rng.randint(1, 8)):
        f = Line("frame")
        f.add(ind).add("at ")
        cls = rng.choice(V.JAVA_CLASSES)
        f.add(f"{cls}.{rng.choice(V.JAVA_METHODS)}", "FN").add("(")
        r = rng.random()
        if r < 0.75:
            f.add(rng.choice(V.FILES_JAVA), "FILE").add(":").add(str(rng.randint(1, 3000)), "LINE")
        elif r < 0.9:
            f.add("Native Method")
        else:
            f.add("Unknown Source")
        f.add(")")
        if rng.random() < 0.2:
            f.add(" ~[").add(rng.choice(["spring-web-6.1.2.jar", "?", "app.jar", "na"])).add(":").add(rng.choice(["6.1.2", "?", "na"])).add("]")
        out.append(f)
    if rng.random() < 0.5:
        m = Line("continuation")
        m.add(ind).add("... ").add(str(rng.randint(1, 99))).add(rng.choice([" more", " common frames omitted"]))
        out.append(m)
    return out


def python_trace(rng: random.Random) -> list[Line]:
    out: list[Line] = []
    h = Line("continuation")
    h.add(rng.choice(["Traceback (most recent call last):", "Traceback (most recent call last):", "The above exception was the direct cause of the following exception:",
                      "During handling of the above exception, another exception occurred:"]))
    out.append(h)
    ind = indent(rng, "  ")
    for _ in range(rng.randint(1, 6)):
        f = Line("frame")
        f.add(ind).add("File \"").add(rng.choice(V.PY_FILES), "FILE").add("\", line ").add(str(rng.randint(1, 2000)), "LINE")
        if rng.random() < 0.9:
            f.add(", in ").add(rng.choice(V.PY_FUNCS), "FN")
        out.append(f)
        if rng.random() < 0.8:
            src = Line("continuation")
            src.add(ind + "  ").add(rng.choice(["result = func(*args, **kwargs)", "return self._send(request)", "raise ValueError(f\"bad value: {x}\")",
                                                "conn = await pool.acquire()", "main()", "data = json.loads(body)", "x = items[idx]",
                                                "^^^^^^^^^^^^^^^^^^^^^^^^^^^^", "~~~~~^^^^^^^^^", "response.raise_for_status()", "sys.exit(main())"]))
            out.append(src)
    tail = Line("continuation")
    exc = rng.choice(V.PY_EXC)
    if rng.random() < 0.7:
        tail.add(exc).add(": ").add(message(rng), "MSG")
    else:
        tail.add(exc, "MSG")
    out.append(tail)
    return out


def js_trace(rng: random.Random) -> list[Line]:
    out: list[Line] = []
    h = Line("continuation")
    exc = rng.choice(V.JS_EXC)
    if rng.random() < 0.3:
        h.add(rng.choice(["Uncaught ", "UnhandledPromiseRejection: ", "[UnhandledPromiseRejection: ", "Error: "]))
    h.add(exc).add(": ").add(message(rng), "MSG")
    out.append(h)
    ind = indent(rng, "    ")
    for _ in range(rng.randint(1, 8)):
        f = Line("frame")
        f.add(ind).add("at ")
        fn = rng.choice(V.JS_FUNCS)
        file = rng.choice(V.JS_FILES)
        line, col = str(rng.randint(1, 5000)), str(rng.randint(1, 200))
        r = rng.random()
        if r < 0.6:
            f.add(fn, "FN").add(" (").add(file, "FILE").add(":").add(line, "LINE").add(":").add(col, "COL").add(")")
        elif r < 0.8:
            f.add(file, "FILE").add(":").add(line, "LINE").add(":").add(col, "COL")
        elif r < 0.9:
            f.add("async ").add(fn, "FN").add(" (").add(file, "FILE").add(":").add(line, "LINE").add(":").add(col, "COL").add(")")
        else:
            f.add(fn, "FN").add(" (<anonymous>)")
        if rng.random() < 0.1:
            f.add(" {")
        out.append(f)
    if rng.random() < 0.3:
        for _ in range(rng.randint(1, 3)):
            c = Line("continuation")
            c.add(ind + "  ").add(rng.choice(["code: 'ECONNREFUSED',", "errno: -111,", "syscall: 'connect',", "address: '127.0.0.1',", "port: 5432", "}", "[cause]: Error: connect ECONNREFUSED"]))
            out.append(c)
    if rng.random() < 0.3:
        c = Line("continuation")
        c.add(ind).add("at ").add("processTicksAndRejections", "FN").add(" (").add("node:internal/process/task_queues", "FILE").add(":").add("95", "LINE").add(":").add("5", "COL").add(")")
        c.kind = "frame"
        out.append(c)
    if rng.random() < 0.2:
        c = Line("continuation")
        c.add(rng.choice(["Node.js v20.10.0", "Node.js v18.19.0", "Node.js v22.3.0"]))
        out.append(c)
    return out


def firefox_trace(rng: random.Random) -> list[Line]:
    out: list[Line] = []
    for _ in range(rng.randint(1, 5)):
        f = Line("frame")
        f.add(rng.choice(V.JS_FUNCS).split(" ")[0].replace("<anonymous>", "anon"), "FN").add("@").add(rng.choice(V.JS_FILES), "FILE").add(":").add(str(rng.randint(1, 9999)), "LINE").add(":").add(str(rng.randint(1, 300)), "COL")
        out.append(f)
    return out


def go_trace(rng: random.Random) -> list[Line]:
    out: list[Line] = []
    h = Line("continuation")
    r = rng.random()
    if r < 0.4:
        h.add("panic: ").add(message(rng), "MSG")
        if rng.random() < 0.3:
            h.add(" [recovered]")
    elif r < 0.6:
        h.add("goroutine ").add(str(rng.randint(1, 9999)), "THREAD").add(" [").add(rng.choice(["running", "select", "chan receive", "IO wait", "semacquire", "sleep", "runnable"])).add(rng.choice(["", ", 5 minutes", ", locked to thread"])).add("]:")
    elif r < 0.8:
        h.add("fatal error: ").add(message(rng), "MSG")
    else:
        h.add("[signal SIGSEGV: segmentation violation code=0x1 addr=0x").add(hexid(rng, 2)).add(" pc=0x").add(hexid(rng, 6)).add("]")
    out.append(h)
    for _ in range(rng.randint(1, 6)):
        a = Line("frame")
        fn = rng.choice(V.GO_FUNCS)
        a.add(fn, "FN").add("(")
        if rng.random() < 0.7:
            a.add(", ".join(f"0x{hexid(rng, rng.choice([1, 2, 8, 12]))}" for _ in range(rng.randint(1, 4))))
        elif rng.random() < 0.5:
            a.add("...")
        a.add(")")
        if rng.random() < 0.1:
            a.parts.insert(0, ("created by ", None))
            a.parts = [a.parts[0], a.parts[1]]
            if rng.random() < 0.5:
                a.add(" in goroutine ").add(str(rng.randint(1, 99)), "THREAD")
        out.append(a)
        b = Line("frame")
        b.add("\t").add(rng.choice(V.GO_FILES), "FILE").add(":").add(str(rng.randint(1, 3000)), "LINE")
        if rng.random() < 0.8:
            b.add(" +0x").add(hexid(rng, rng.choice([2, 3, 4])))
        if rng.random() < 0.1:
            b.add(" fp=0x").add(hexid(rng, 12)).add(" sp=0x").add(hexid(rng, 12)).add(" pc=0x").add(hexid(rng, 6))
        out.append(b)
    if rng.random() < 0.2:
        e = Line("continuation")
        e.add("exit status ").add(str(rng.randint(1, 2)))
        out.append(e)
    return out


def rust_trace(rng: random.Random) -> list[Line]:
    out: list[Line] = []
    h = Line("continuation")
    v = rng.random()
    if v < 0.5:
        h.add("thread '").add(rng.choice(["main", "tokio-runtime-worker", "<unnamed>", "worker-3"]), "THREAD").add("' panicked at ")
        if rng.random() < 0.5:
            h.add(rng.choice(V.RUST_FILES), "FILE").add(":").add(str(rng.randint(1, 999)), "LINE").add(":").add(str(rng.randint(1, 99)), "COL").add(":")
            out.append(h)
            m = Line("continuation")
            m.add(message(rng), "MSG")
            out.append(m)
        else:
            h.add("'").add(message(rng), "MSG").add("', ").add(rng.choice(V.RUST_FILES), "FILE").add(":").add(str(rng.randint(1, 999)), "LINE").add(":").add(str(rng.randint(1, 99)), "COL")
            out.append(h)
    elif v < 0.7:
        h.add("stack backtrace:")
        out.append(h)
    else:
        h.add("note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace")
        out.append(h)
        return out
    for i in range(rng.randint(1, 7)):
        a = Line("frame")
        a.add(f"{i:4d}: " if rng.random() < 0.7 else f"  {i}: ").add(rng.choice(V.RUST_FUNCS), "FN")
        out.append(a)
        if rng.random() < 0.8:
            b = Line("frame")
            b.add(" " * rng.choice([13, 18, 8])).add("at ").add(rng.choice(V.RUST_FILES), "FILE").add(":").add(str(rng.randint(1, 999)), "LINE")
            if rng.random() < 0.8:
                b.add(":").add(str(rng.randint(1, 99)), "COL")
            out.append(b)
    if rng.random() < 0.3:
        n = Line("continuation")
        n.add("note: Some details are omitted, run with `RUST_BACKTRACE=full` for a verbose backtrace.")
        out.append(n)
    return out


def csharp_trace(rng: random.Random) -> list[Line]:
    out: list[Line] = []
    h = Line("continuation")
    if rng.random() < 0.5:
        h.add("Unhandled exception. ")
    h.add(rng.choice(V.CS_EXC)).add(": ").add(message(rng), "MSG")
    out.append(h)
    ind = indent(rng, "   ")
    for _ in range(rng.randint(1, 6)):
        f = Line("frame")
        f.add(ind).add("at ")
        fn = rng.choice(V.CS_FUNCS)
        name, _, args = fn.partition("(")
        f.add(name, "FN").add("(" + args)
        if rng.random() < 0.6:
            f.add(" in ").add(rng.choice(V.CS_FILES), "FILE").add(":line ").add(str(rng.randint(1, 999)), "LINE")
        out.append(f)
        if rng.random() < 0.15:
            c = Line("continuation")
            c.add(rng.choice(["--- End of stack trace from previous location ---", "--- End of inner exception stack trace ---", " ---> System.Net.Sockets.SocketException (111): Connection refused"]))
            out.append(c)
    return out


def ruby_trace(rng: random.Random) -> list[Line]:
    out: list[Line] = []
    h = Line("continuation")
    if rng.random() < 0.5:
        h.add(rng.choice(V.RB_FILES), "FILE").add(":").add(str(rng.randint(1, 999)), "LINE").add(":in `").add(rng.choice(V.RB_FUNCS), "FN").add("': ").add(message(rng), "MSG").add(" (").add(rng.choice(V.RB_EXC)).add(")")
        h.kind = "frame"
    else:
        h.add(rng.choice(V.RB_EXC)).add(": ").add(message(rng), "MSG")
    out.append(h)
    quote = rng.choice(["`", "'"])
    for _ in range(rng.randint(1, 7)):
        f = Line("frame")
        if rng.random() < 0.7:
            f.add(indent(rng, "\t")).add("from ")
        f.add(rng.choice(V.RB_FILES), "FILE").add(":").add(str(rng.randint(1, 999)), "LINE").add(":in " + quote).add(rng.choice(V.RB_FUNCS), "FN").add("'")
        out.append(f)
    return out


def php_trace(rng: random.Random) -> list[Line]:
    out: list[Line] = []
    h = Line("continuation")
    v = rng.random()
    if v < 0.5:
        h = Line("entry")
        h.add("PHP ").add("Fatal error", "LEVEL").add(":  Uncaught ").add(rng.choice(V.PHP_EXC)).add(": ").add(message(rng), "MSG")
        h.add(" in ").add(rng.choice(V.PHP_FILES), "FILE").add(":").add(str(rng.randint(1, 999)), "LINE")
        out.append(h)
        s = Line("continuation")
        s.add("Stack trace:")
        out.append(s)
    else:
        h.add("Stack trace:")
        out.append(h)
    for i in range(rng.randint(1, 6)):
        f = Line("frame")
        f.add(f"#{i} ")
        if rng.random() < 0.9:
            f.add(rng.choice(V.PHP_FILES), "FILE").add("(").add(str(rng.randint(1, 999)), "LINE").add("): ")
        else:
            f.add("[internal function]: ")
        fn = rng.choice(V.PHP_FUNCS)
        name, _, args = fn.partition("(")
        f.add(name, "FN").add("(" + args)
        out.append(f)
    m = Line("frame")
    m.add(f"#{rng.randint(2, 9)} {{main}}")
    m.kind = "continuation"
    out.append(m)
    if rng.random() < 0.6:
        t = Line("continuation")
        t.add("  thrown in ").add(rng.choice(V.PHP_FILES), "FILE").add(" on line ").add(str(rng.randint(1, 999)), "LINE")
        out.append(t)
    return out


def generic_continuation(rng: random.Random) -> list[Line]:
    """Indented dumps, wrapped text, JSON/SQL/config bodies that follow an entry."""
    out: list[Line] = []
    v = rng.randrange(8)
    ind = rng.choice(["  ", "    ", "\t", "        ", " ", ""])
    if v == 0:  # json body
        for s in ["{", f'  "id": {rng.randint(1, 999)},', f'  "user": "{word(rng)}",', '  "ok": false,', '  "tags": ["a", "b"]', "}"]:
            out.append(Line("continuation").add(ind).add(s))
    elif v == 1:  # sql body
        q = sql(rng)
        for part in [q[: len(q) // 2], q[len(q) // 2:]]:
            out.append(Line("continuation").add(ind).add(part))
    elif v == 2:  # key: value dump (still tagged as kv)
        for _ in range(rng.randint(1, 4)):
            L = Line("continuation").add(ind)
            add_kv(L, rng, n=1, style="colon")
            out.append(L)
    elif v == 3:  # wrapped message
        m = message(rng)
        out.append(Line("continuation").add(ind).add(m, "MSG"))
    elif v == 4:  # list items
        for _ in range(rng.randint(1, 3)):
            out.append(Line("continuation").add(ind).add(rng.choice(["- ", "* ", "• ", "> "])).add(message(rng), "MSG"))
    elif v == 5:  # bare table rows / hex dump
        out.append(Line("continuation").add(ind).add(" ".join(hexid(rng, 2) for _ in range(rng.randint(4, 16)))))
        out.append(Line("continuation").add(ind).add("\t".join(word(rng) for _ in range(rng.randint(2, 5)))))
    elif v == 6:  # separators
        out.append(Line("continuation").add(rng.choice(["-" * rng.randint(10, 80), "=" * rng.randint(10, 80), "*" * 20, "#" * 30, "~~~~", "...", "^^^^^^^^", "----------------------------------------"])))
    else:  # exception-ish header without frames, or 'Caused by' chains
        L = Line("continuation")
        L.add(rng.choice(["Caused by: ", "Error: ", "error: ", "warning: ", "Details: ", "Reason: ", "Message: ", "Exception: ", "Inner exception: ", "--> ", "==> "]))
        L.add(message(rng), "MSG")
        out.append(L)
    return out


TRACE_BUILDERS: list[tuple[Callable[[random.Random], list[Line]], float]] = [
    (java_trace, 5), (python_trace, 5), (js_trace, 5), (firefox_trace, 0.5), (go_trace, 3), (rust_trace, 3),
    (csharp_trace, 2.5), (ruby_trace, 2.5), (php_trace, 2.5), (generic_continuation, 6),
]
