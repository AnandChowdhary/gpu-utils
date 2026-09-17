"""Tokenizer and hashing. Mirrors packages/runtime/src/tokenize.ts and hash.ts exactly.

Parity is enforced by packages/runtime/test/fixtures/tokenize.json, which both the
TypeScript and Python test suites read.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

# Character classes (must match tokenize.ts)
CLASS_LETTER, CLASS_DIGIT, CLASS_SPACE, CLASS_NEWLINE, CLASS_OTHER = 0, 1, 2, 3, 4

# Shape classes (must match tokenize.ts)
SHAPE_LOWER, SHAPE_UPPER, SHAPE_TITLE, SHAPE_MIXED, SHAPE_DIGITS, SHAPE_SPACE, SHAPE_NEWLINE, SHAPE_OTHER = range(8)


@dataclass(frozen=True)
class Token:
    text: str
    start: int  # UTF-16 code unit offset, to match JS
    end: int
    cls: int
    shape: int


def _char_class(ch: str) -> int:
    if ch == "\n" or ch == "\r":
        return CLASS_NEWLINE
    if ch == " " or ch == "\t":
        return CLASS_SPACE
    cat = unicodedata.category(ch)
    if cat[0] == "L" or cat[0] == "M":
        return CLASS_LETTER
    if cat == "Nd":
        return CLASS_DIGIT
    return CLASS_OTHER


def _shape(text: str, cls: int) -> int:
    if cls == CLASS_SPACE:
        return SHAPE_SPACE
    if cls == CLASS_NEWLINE:
        return SHAPE_NEWLINE
    if cls == CLASS_DIGIT:
        return SHAPE_DIGITS
    if cls == CLASS_OTHER:
        return SHAPE_OTHER
    if text.islower():
        return SHAPE_LOWER
    if text.isupper():
        return SHAPE_UPPER
    if text[0].isupper() and text[1:].islower():
        return SHAPE_TITLE
    return SHAPE_MIXED


def _utf16_len(s: str) -> int:
    return len(s.encode("utf-16-le")) // 2


def tokenize(text: str) -> list[Token]:
    """Split into runs of the same character class. Each 'other' char is its own token.
    A CRLF pair is a single newline token; otherwise newline chars are single tokens."""
    tokens: list[Token] = []
    i = 0
    offset = 0
    n = len(text)
    while i < n:
        ch = text[i]
        cls = _char_class(ch)
        j = i + 1
        if cls == CLASS_NEWLINE:
            if ch == "\r" and j < n and text[j] == "\n":
                j += 1
        elif cls != CLASS_OTHER:
            while j < n and _char_class(text[j]) == cls:
                j += 1
        chunk = text[i:j]
        width = _utf16_len(chunk)
        tokens.append(Token(chunk, offset, offset + width, cls, _shape(chunk, cls)))
        offset += width
        i = j
    return tokens


def hash_token(text: str, buckets: int) -> int:
    """32-bit FNV-1a over UTF-8 of the lowercased token with digits collapsed to '0'."""
    normalized = "".join("0" if c.isdigit() else c for c in text.lower())
    h = 0x811C9DC5
    for b in normalized.encode("utf-8"):
        h ^= b
        h = (h * 0x01000193) & 0xFFFFFFFF
    return h % buckets
