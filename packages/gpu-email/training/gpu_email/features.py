"""Tokenizer and sparse featurizer for gpu-email. Must match src/features.ts exactly.

Every token gets a fixed-width row of NUM_SLOTS feature ids into one shared embedding
table. Slots are token-level (word hash, shape, ...), line-level (what the token's line
looks like: first/last word, punctuation, keyword hits) and document-level (what the
lines above and below look like: quote prefixes, attribution markers, blank lines).
Everything here is deterministic and computed on the CPU before the model runs.

Parity with TypeScript is enforced by ../../test/features.test.ts against
../../model/fixtures.json and test/fixtures/features-hashes.json, both written from
this module.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

from gpu_utils_training.features import (
    CLASS_DIGIT,
    CLASS_LETTER,
    CLASS_NEWLINE,
    CLASS_SPACE,
    SHAPE_TITLE,
    SHAPE_UPPER,
    Token,
    hash_token,
    tokenize,
)

# --- label sets ---------------------------------------------------------------------

LINE_KINDS = [
    "reply",
    "attribution",
    "quote",
    "signature",
    "disclaimer",
    "forward_header",
    "greeting",
    "closing",
]
FIELDS = ["NAME", "TITLE", "COMPANY", "PHONE", "EMAIL", "URL", "ADDRESS"]
BIO_LABELS = ["O"] + [f"{p}-{f}" for f in FIELDS for p in ("B", "I")]

# --- keyword groups (lowercased substring / whole-token matches) --------------------
# Keep these lists byte-identical to src/features.ts.

KW_WROTE = [
    "wrote",
    "écrit",
    "ecrit",
    "schrieb",
    "escribió",
    "escribio",
    "escreveu",
    "skrev",
    "scritto",
    "schreef",
    "napisał",
    "napsal",
    "写道",
    "書きました",
]
KW_ORIGINAL = [
    "original message",
    "ursprüngliche nachricht",
    "message d'origine",
    "mensaje original",
    "messaggio originale",
    "oorspronkelijk bericht",
    "原始邮件",
    "元のメッセージ",
    "original appointment",
]
KW_FORWARD = [
    "forwarded message",
    "begin forwarded",
    "weitergeleitete nachricht",
    "message transféré",
    "mensaje reenviado",
    "messaggio inoltrato",
    "doorgestuurd bericht",
    "转发",
    "転送",
    "fwd:",
]
KW_MOBILE = [
    "sent from my",
    "sent from a",
    "envoyé de mon",
    "von meinem",
    "enviado desde mi",
    "inviato da",
    "verzonden vanaf",
    "から送信",
    "get outlook for",
    "sent via",
    "sent with",
    "sent from mail",
    "sent from outlook",
]
KW_HEADER = [
    "from:",
    "sent:",
    "to:",
    "subject:",
    "date:",
    "cc:",
    "reply-to:",
    "von:",
    "gesendet:",
    "an:",
    "betreff:",
    "datum:",
    "de :",
    "de:",
    "envoyé :",
    "envoyé:",
    "à :",
    "à:",
    "objet :",
    "objet:",
    "para:",
    "enviado el:",
    "enviado:",
    "asunto:",
    "fecha:",
    "da:",
    "inviato:",
    "a:",
    "oggetto:",
    "差出人:",
    "送信日時:",
    "宛先:",
    "件名:",
    "发件人:",
    "发送时间:",
    "收件人:",
    "主题:",
    "差出人：",
    "宛先：",
    "件名：",
    "发件人：",
    "收件人：",
    "主题：",
]
KW_DISCLAIMER = [
    "confidential",
    "intended recipient",
    "disclaimer",
    "privileged",
    "unauthorized",
    "unauthorised",
    "vertraulich",
    "confidentiel",
    "confidencial",
    "unsubscribe",
    "mailing list",
    "this email",
    "this e-mail",
    "this message",
    "diese e-mail",
    "ce message",
    "este mensaje",
    "please consider the environment",
    "virus",
    "legally",
    "liability",
    "the sender",
    "notify",
    "registered in",
    "registered office",
]
KW_CONTACT_WORDS = [
    "tel",
    "phone",
    "mobile",
    "mob",
    "cell",
    "fax",
    "office",
    "direct",
    "e-mail",
    "email",
    "mail",
    "web",
    "www",
    "skype",
    "linkedin",
    "twitter",
    "whatsapp",
    "téléphone",
    "telefon",
    "teléfono",
    "telefono",
    "portable",
    "handy",
    "móvil",
    "movil",
    "手机",
    "電話",
    "电话",
    "tél",
    "ph",
    "m",
    "t",
    "p",
    "f",
    "e",
    "w",
    "d",
    "o",
    "c",
]
KW_CLOSING_WORDS = [
    "regards",
    "thanks",
    "cheers",
    "best",
    "sincerely",
    "cordialement",
    "grüßen",
    "grüße",
    "gruß",
    "saludos",
    "atentamente",
    "merci",
    "danke",
    "gracias",
    "thx",
    "ty",
    "kind",
    "warm",
    "warmly",
    "yours",
    "truly",
    "care",
    "soon",
    "bye",
    "ciao",
    "wishes",
    "cordiali",
    "cumprimentos",
    "obrigado",
    "hälsningar",
    "groeten",
    "abrazo",
    "bien",
    "amicalement",
    "bises",
    "liebe",
    "viele",
    "beste",
    "mfg",
    "vg",
    "lg",
    "br",
    "rgds",
    "thank",
    "appreciate",
    "talk",
    "speak",
    "よろしく",
    "谢谢",
    "敬上",
    "祝好",
    "敬具",
]
KW_GREETING_WORDS = [
    "hi",
    "hello",
    "hey",
    "dear",
    "morning",
    "afternoon",
    "evening",
    "bonjour",
    "hallo",
    "hola",
    "salut",
    "guten",
    "geehrte",
    "geehrter",
    "liebe",
    "lieber",
    "こんにちは",
    "你好",
    "您好",
    "greetings",
    "team",
    "all",
    "folks",
    "everyone",
    "everybody",
    "hiya",
    "yo",
    "buongiorno",
    "olá",
    "ola",
    "hej",
    "hoi",
    "estimado",
    "estimada",
    "cher",
    "chère",
    "sir",
    "madam",
    "mr",
    "ms",
    "mrs",
    "dr",
    "herr",
    "frau",
    "monsieur",
    "madame",
    "señor",
    "señora",
]
KW_DATE_WORDS = [
    "jan",
    "feb",
    "mar",
    "apr",
    "may",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
    "january",
    "february",
    "march",
    "april",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
    "janv",
    "févr",
    "fév",
    "mars",
    "avr",
    "avril",
    "mai",
    "juin",
    "juil",
    "juillet",
    "août",
    "déc",
    "janvier",
    "février",
    "octobre",
    "novembre",
    "décembre",
    "septembre",
    "januar",
    "februar",
    "märz",
    "juni",
    "juli",
    "oktober",
    "dezember",
    "ene",
    "abr",
    "ago",
    "dic",
    "enero",
    "febrero",
    "marzo",
    "mayo",
    "junio",
    "julio",
    "agosto",
    "septiembre",
    "octubre",
    "noviembre",
    "diciembre",
    "mon",
    "tue",
    "wed",
    "thu",
    "fri",
    "sat",
    "sun",
    "monday",
    "tuesday",
    "wednesday",
    "thursday",
    "friday",
    "saturday",
    "sunday",
    "lundi",
    "mardi",
    "mercredi",
    "jeudi",
    "vendredi",
    "samedi",
    "dimanche",
    "montag",
    "dienstag",
    "mittwoch",
    "donnerstag",
    "freitag",
    "samstag",
    "sonntag",
    "lunes",
    "martes",
    "miércoles",
    "jueves",
    "viernes",
    "sábado",
    "domingo",
    "am",
    "pm",
    "uhr",
    "at",
    "um",
    "às",
    "kl",
    "年",
    "月",
    "日",
    "午前",
    "午後",
]

KW_CLOSING_PHRASES = [
    "thank you",
    "take care",
    "talk soon",
    "speak soon",
    "all the best",
    "best wishes",
    "mit freundlichen",
    "bien à vous",
    "un saludo",
    "kind regards",
    "warm regards",
    "many thanks",
]
KW_GREETING_PHRASES = [
    "good morning",
    "good afternoon",
    "good evening",
    "sehr geehrte",
    "お世話になっております",
    "dear all",
    "hi all",
    "hi there",
    "hello there",
]

KW_CONTACT_SET = set(KW_CONTACT_WORDS)
KW_CLOSING_SET = set(KW_CLOSING_WORDS)
KW_GREETING_SET = set(KW_GREETING_WORDS)
KW_DATE_SET = set(KW_DATE_WORDS)

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
URL_RE = re.compile(r"(?:https?://[^\s<>\"')\]]+|www\.[A-Za-z0-9-]+\.[^\s<>\"')\]]+)")

# --- slot layout --------------------------------------------------------------------

WORD_BUCKETS = 1024
SUFFIX_BUCKETS = 512
FIRST_WORD_BUCKETS = 256
LAST_WORD_BUCKETS = 256

# (name, size) in row order. Keep identical to src/features.ts.
SLOTS: list[tuple[str, int]] = [
    ("word", WORD_BUCKETS),
    ("shape", 8),
    ("len", 16),
    ("cls", 5),
    ("suffix", SUFFIX_BUCKETS),
    ("pos_in_line", 9),
    ("line_flags", 8),
    ("line_from_top", 9),
    ("line_from_bottom", 9),
    ("line_len", 9),
    ("first_word", FIRST_WORD_BUCKETS),
    ("first_shape", 9),
    ("last_word", LAST_WORD_BUCKETS),
    ("last_char", 9),
    ("content_flags", 8),
    ("word_count", 9),
    ("title_frac", 6),
    ("digit_count", 6),
    ("neighbours", 16),
    ("kw_wrote", 2),
    ("kw_original", 2),
    ("kw_forward", 2),
    ("kw_mobile", 2),
    ("kw_header", 2),
    ("kw_disclaimer", 2),
    ("kw_contact", 2),
    ("kw_closing", 2),
    ("kw_greeting", 2),
    ("kw_date", 2),
    ("ctx_above", 16),
    ("since_attrib", 8),
    ("since_blank", 7),
    ("until_blank", 7),
    ("until_quoteish", 8),
    ("header_run", 5),
    ("para_from_top", 6),
    ("para_from_bottom", 6),
]
NUM_SLOTS = len(SLOTS)
SLOT_BASE: list[int] = []
_acc = 0
for _name, _size in SLOTS:
    SLOT_BASE.append(_acc)
    _acc += _size
NUM_ROWS = _acc


def _bucket(v: int, edges: list[int]) -> int:
    """Index of the first edge >= v, or len(edges) when v exceeds all edges."""
    for i, e in enumerate(edges):
        if v <= e:
            return i
    return len(edges)


# bucket edges (value <= edge → index)
EDGES_LINE_IDX = [0, 1, 2, 3, 4, 7, 11, 19]  # 9 buckets
EDGES_LINE_LEN = [0, 3, 10, 20, 40, 60, 80, 100]  # 9 buckets
EDGES_WORD_COUNT = [0, 1, 2, 3, 4, 6, 9, 14]  # 9 buckets
EDGES_DIGITS = [0, 2, 5, 8, 12]  # 6 buckets
EDGES_SINCE_ATTRIB = [0, 1, 2, 3, 5, 9]  # +1 for "none" → 8 buckets (0 = none)
EDGES_PARA = [0, 1, 2, 3, 5]  # 6 buckets
EDGES_PARA_POS = [0, 1, 2, 3, 4, 7]  # 7 buckets
EDGES_UNTIL_QUOTEISH = [1, 2, 3, 4, 6, 10]  # +1 for "none" → 8 buckets (0 = none)
EDGES_HEADER_RUN = [0, 1, 2, 3]  # 5 buckets


@dataclass
class LineInfo:
    start: int  # token index (inclusive)
    end: int  # token index (exclusive), includes the newline token if any
    text: str  # line text without the newline, original case
    lower: str
    blank: bool
    quote_prefixed: bool
    delimiter: bool
    is_header: bool
    is_attrib_marker: bool  # kw_wrote (ending with ':' not required) or kw_original
    is_forward_marker: bool


def split_lines(tokens: list[Token]) -> list[tuple[int, int]]:
    """[start, end) token ranges, one per line; the newline token belongs to its line."""
    lines: list[tuple[int, int]] = []
    start = 0
    for i, t in enumerate(tokens):
        if t.cls == CLASS_NEWLINE:
            lines.append((start, i + 1))
            start = i + 1
    if start < len(tokens):
        lines.append((start, len(tokens)))
    return lines


def _strip_prefix(lower: str) -> str:
    """Strip leading spaces, quote markers and Outlook-style '*' before header keys."""
    i = 0
    n = len(lower)
    while i < n and lower[i] in " \t>*|":
        i += 1
    return lower[i:]


def _contains_any(hay: str, needles: list[str]) -> bool:
    for n in needles:
        if n in hay:
            return True
    return False


def _starts_with_any(hay: str, needles: list[str]) -> bool:
    for n in needles:
        if hay.startswith(n):
            return True
    return False


def _line_info(tokens: list[Token], start: int, end: int) -> LineInfo:
    parts = []
    for i in range(start, end):
        if tokens[i].cls != CLASS_NEWLINE:
            parts.append(tokens[i].text)
    text = "".join(parts)
    lower = text.lower()
    blank = True
    for i in range(start, end):
        if tokens[i].cls != CLASS_NEWLINE and tokens[i].cls != CLASS_SPACE:
            blank = False
            break
    stripped = lower.strip(" \t")
    quote_prefixed = stripped.startswith(">")
    delimiter = stripped == "--"
    body = _strip_prefix(lower)
    is_header = _starts_with_any(body, KW_HEADER)
    is_attrib_marker = _contains_any(lower, KW_WROTE) or _contains_any(lower, KW_ORIGINAL)
    is_forward_marker = _contains_any(lower, KW_FORWARD)
    return LineInfo(
        start,
        end,
        text,
        lower,
        blank,
        quote_prefixed,
        delimiter,
        is_header,
        is_attrib_marker,
        is_forward_marker,
    )


def _last_char_class(text: str) -> int:
    t = text.rstrip(" \t")
    if not t:
        return 0
    c = t[-1]
    if c == ":" or c == "：":
        return 1
    if c == ",":
        return 2
    if c == ".":
        return 3
    if c == "?" or c == "!":
        return 4
    if c == ";":
        return 5
    cat = unicodedata.category(c)
    if cat[0] == "L":
        return 6
    if cat == "Nd":
        return 7
    return 8


def featurize(text: str) -> list[list[int]]:
    return featurize_tokens(tokenize(text))


def featurize_tokens(tokens: list[Token]) -> list[list[int]]:  # noqa: C901
    n = len(tokens)
    rows: list[list[int]] = [[0] * NUM_SLOTS for _ in range(n)]
    if n == 0:
        return rows
    ranges = split_lines(tokens)
    lines = [_line_info(tokens, s, e) for s, e in ranges]
    num_lines = len(lines)

    # --- document-level context -------------------------------------------------
    nonblank_below = [0] * num_lines
    acc = 0
    for li in range(num_lines - 1, -1, -1):
        nonblank_below[li] = acc
        if not lines[li].blank:
            acc += 1

    # paragraphs: runs of non-blank lines
    para_index = [0] * num_lines
    pos_in_para = [0] * num_lines
    para_count = 0
    in_para = False
    for li, line in enumerate(lines):
        if line.blank:
            in_para = False
            para_index[li] = para_count
            pos_in_para[li] = 0
        else:
            if not in_para:
                para_count += 1
                in_para = True
                pos_in_para[li] = 0
            else:
                pos_in_para[li] = pos_in_para[li - 1] + 1
            para_index[li] = para_count - 1
    until_blank = [0] * num_lines
    run = 0
    for li in range(num_lines - 1, -1, -1):
        if lines[li].blank:
            run = 0
            until_blank[li] = 0
        else:
            until_blank[li] = run
            run += 1

    until_quoteish = [0] * num_lines  # 0 = none
    nxt = -1
    for li in range(num_lines - 1, -1, -1):
        line = lines[li]
        if nxt >= 0:
            until_quoteish[li] = _bucket(nxt - li, EDGES_UNTIL_QUOTEISH) + 1
        if line.quote_prefixed or line.is_attrib_marker or line.is_forward_marker or line.is_header:
            nxt = li

    quote_above = False
    delim_above = False
    attrib_above = False
    header_above = False
    last_attrib = -1
    header_run = 0
    row_line = [0] * n
    for li, line in enumerate(lines):
        # per-line values
        flags = (1 if line.quote_prefixed else 0) | (2 if line.delimiter else 0) | (4 if line.blank else 0)
        from_top = _bucket(li, EDGES_LINE_IDX)
        from_bottom = _bucket(nonblank_below[li], EDGES_LINE_IDX)
        line_len = _bucket(len(line.text.encode("utf-16-le")) // 2, EDGES_LINE_LEN)

        first_word = 0
        first_shape = 8
        last_word = 0
        word_count = 0
        letter_tokens = 0
        title_tokens = 0
        digit_chars = 0
        letter_chars = 0
        kw_contact = 0
        kw_closing = 0
        kw_greeting = 0
        kw_date = 0
        first_nonspace_seen = False
        for i in range(line.start, line.end):
            t = tokens[i]
            if t.cls == CLASS_NEWLINE:
                continue
            if t.cls != CLASS_SPACE and not first_nonspace_seen:
                first_nonspace_seen = True
                first_shape = t.shape
            if t.cls == CLASS_LETTER or t.cls == CLASS_DIGIT:
                word_count += 1
                h = hash_token(t.text, FIRST_WORD_BUCKETS - 1) + 1
                if first_word == 0:
                    first_word = h
                last_word = hash_token(t.text, LAST_WORD_BUCKETS - 1) + 1
            if t.cls == CLASS_LETTER:
                letter_tokens += 1
                letter_chars += len(t.text)
                if t.shape == SHAPE_TITLE or t.shape == SHAPE_UPPER:
                    title_tokens += 1
                w = t.text.lower()
                if w in KW_CONTACT_SET:
                    kw_contact = 1
                if w in KW_CLOSING_SET:
                    kw_closing = 1
                if w in KW_GREETING_SET:
                    kw_greeting = 1
                if w in KW_DATE_SET:
                    kw_date = 1
            elif t.cls == CLASS_DIGIT:
                digit_chars += len(t.text)
        lower = line.lower
        if _contains_any(lower, KW_CLOSING_PHRASES):
            kw_closing = 1
        if _contains_any(lower, KW_GREETING_PHRASES):
            kw_greeting = 1
        has_email = 1 if EMAIL_RE.search(line.text) else 0
        has_url = 2 if URL_RE.search(line.text) else 0
        phone_like = 4 if (7 <= digit_chars <= 20 and letter_chars <= 12) else 0
        content_flags = has_email | has_url | phone_like
        wc = _bucket(word_count, EDGES_WORD_COUNT)
        if letter_tokens == 0:
            title_frac = 0
        elif title_tokens == 0:
            title_frac = 1
        elif title_tokens * 3 < letter_tokens:
            title_frac = 2
        elif title_tokens * 3 < letter_tokens * 2:
            title_frac = 3
        elif title_tokens < letter_tokens:
            title_frac = 4
        else:
            title_frac = 5
        digits_b = _bucket(digit_chars, EDGES_DIGITS)
        prev_blank = li > 0 and lines[li - 1].blank
        next_blank = li + 1 < num_lines and lines[li + 1].blank
        neighbours = (1 if prev_blank else 0) | (2 if next_blank else 0) | (4 if li > 0 else 0) | (8 if li + 1 < num_lines else 0)
        kw_wrote = 1 if _contains_any(lower, KW_WROTE) else 0
        kw_original = 1 if _contains_any(lower, KW_ORIGINAL) else 0
        kw_forward = 1 if _contains_any(lower, KW_FORWARD) else 0
        kw_mobile = 1 if _contains_any(lower, KW_MOBILE) else 0
        kw_header = 1 if line.is_header else 0
        kw_disclaimer = 1 if _contains_any(lower, KW_DISCLAIMER) else 0

        # context from lines above (exclusive of this line, except since_attrib which
        # is 1 when this line itself is a marker)
        ctx_above = (1 if quote_above else 0) | (2 if delim_above else 0) | (4 if attrib_above else 0) | (8 if header_above else 0)
        if line.is_attrib_marker or line.is_forward_marker:
            last_attrib = li
        since_attrib = 0 if last_attrib < 0 else _bucket(li - last_attrib, EDGES_SINCE_ATTRIB) + 1
        since_blank = _bucket(pos_in_para[li], EDGES_PARA_POS)
        until_b = _bucket(until_blank[li], EDGES_PARA_POS)
        header_run_b = _bucket(header_run, EDGES_HEADER_RUN)
        para_top = _bucket(para_index[li], EDGES_PARA)
        para_bottom = _bucket(max(0, para_count - 1 - para_index[li]), EDGES_PARA)

        line_vals = [
            flags,
            from_top,
            from_bottom,
            line_len,
            first_word,
            first_shape,
            last_word,
            _last_char_class(line.text),
            content_flags,
            wc,
            title_frac,
            digits_b,
            neighbours,
            kw_wrote,
            kw_original,
            kw_forward,
            kw_mobile,
            kw_header,
            kw_disclaimer,
            kw_contact,
            kw_closing,
            kw_greeting,
            kw_date,
            ctx_above,
            since_attrib,
            since_blank,
            until_b,
            until_quoteish[li],
            header_run_b,
            para_top,
            para_bottom,
        ]
        # --- token-level -------------------------------------------------------
        count = line.end - line.start
        for k, i in enumerate(range(line.start, line.end)):
            t = tokens[i]
            row = rows[i]
            row_line[i] = li
            row[0] = hash_token(t.text, WORD_BUCKETS)
            row[1] = t.shape
            row[2] = min(len(t.text.encode("utf-16-le")) // 2, 15)
            row[3] = t.cls
            row[4] = hash_token(t.text[-3:], SUFFIX_BUCKETS)
            if t.cls == CLASS_NEWLINE:
                pos = 8
            elif k == count - 1 or (k == count - 2 and tokens[line.end - 1].cls == CLASS_NEWLINE):
                pos = 7
            else:
                pos = _bucket(k, [0, 1, 2, 3, 4, 7])
            row[5] = pos
            for s, v in enumerate(line_vals):
                row[6 + s] = v
        # update "above" state after this line
        if line.quote_prefixed:
            quote_above = True
        if line.delimiter:
            delim_above = True
        if line.is_attrib_marker:
            attrib_above = True
        if line.is_header:
            header_above = True
            header_run += 1
        elif not line.blank:
            header_run = 0

    # add slot bases
    for row in rows:
        for s in range(NUM_SLOTS):
            row[s] += SLOT_BASE[s]
    return rows
