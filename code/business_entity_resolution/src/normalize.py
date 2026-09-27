"""Unicode-aware normalisation for business names and addresses.

Handles the noise patterns called out in the problem statement:

* abbreviations (Corp vs Corporation, Pvt vs Private, Rd vs Road)
* legal suffix inconsistencies
* punctuation differences (& vs "and")
* word-order transpositions
* typos / transliterations (accent folding)
* address component reordering and missing components

Nothing here performs any external lookup -- it is purely local string work,
which the challenge's fair-play rules require.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

from .config import (
    ADDRESS_ABBREVS,
    NAME_ABBREVS,
    NAME_SUFFIXES,
    REGION_ABBREVS,
    STOP_TOKENS,
    SYMBOL_WORDS,
)

_WS_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")
_DIGITS_RE = re.compile(r"\d+")
_POSTAL_RE = re.compile(r"\b\d{5}(?:-\d{4})?\b|\b\d{6}\b")
_PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d\-\s().]{7,}\d)(?!\d)")


# --------------------------------------------------------------------------
# low-level helpers
# --------------------------------------------------------------------------

def strip_accents(text: str) -> str:
    """NFD-decompose and drop combining marks.

    ``Café`` -> ``cafe``, ``München`` -> ``munchen``.  This makes the
    pipeline robust to transliteration variants mentioned in the brief.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


def _symbol_pass(text: str) -> str:
    for sym, word in SYMBOL_WORDS.items():
        if sym in text:
            text = text.replace(sym, word)
    return text


def basic_clean(text: str) -> str:
    """Lowercase, fold accents, expand symbols, strip punctuation."""
    if not text:
        return ""
    text = str(text)
    text = strip_accents(text).lower()
    text = _symbol_pass(text)
    # "st." -> "st", "r.d." -> "r d"; punctuation becomes separation so we
    # never merge "A.B" into "ab" while "AB" stays "ab".
    text = re.sub(r"[^0-9a-z]+", " ", text)
    return _WS_RE.sub(" ", text).strip()


def tokenize(text: str, *, drop_stops: bool = False) -> list[str]:
    """Whitespace tokenise already-cleaned text."""
    if not text:
        return []
    toks = text.split()
    if drop_stops:
        toks = [t for t in toks if t not in STOP_TOKENS]
    return toks


def expand_tokens(tokens: Iterable[str], table: dict[str, str]) -> list[str]:
    """Expand known abbreviations, preserving order."""
    return [table.get(t, t) for t in tokens]


def strip_suffixes(tokens: list[str]) -> list[str]:
    """Drop leading/trailing legal suffixes to expose the distinctive core.

    ``["acme", "corporation", "inc"]`` -> ``["acme"]``
    Interior suffixes are left alone: "Standard Oil Company of California"
    keeps "company" because it is not a trailing tag.
    """
    toks = list(tokens)
    while len(toks) > 1 and toks[-1] in NAME_SUFFIXES:
        toks.pop()
    while len(toks) > 1 and toks[0] in NAME_SUFFIXES:
        toks.pop(0)
    return toks


def dedupe_preserve_order(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# --------------------------------------------------------------------------
# name normalisation
# --------------------------------------------------------------------------

def normalize_name(raw: str) -> "NameFeatures":
    """Return the several graded views of a business name we match on."""
    full = basic_clean(raw)
    full_toks = tokenize(full)

    expanded = expand_tokens(full_toks, NAME_ABBREVS)
    expanded = dedupe_preserve_order(expanded)

    core_toks = strip_suffixes(expanded)
    core = " ".join(core_toks)

    # Word-order transposition invariant: sort the core tokens.
    sorted_core = " ".join(sorted(core_toks))

    # Content-only view (stopwords dropped) -- helps "The Coffee Shop"
    # line up with "Coffee Shop".
    content_toks = [t for t in core_toks if t not in STOP_TOKENS]
    content = " ".join(content_toks) if content_toks else core
    sorted_content = " ".join(sorted(content_toks)) if content_toks else sorted_core

    return NameFeatures(
        raw=str(raw or ""),
        full=full,
        full_tokens=full_toks,
        core=core,
        core_tokens=core_toks,
        sorted_core=sorted_core,
        content=content,
        content_tokens=content_toks,
        sorted_content=sorted_content,
        initial_prefix=_initialism(content_toks),
    )


def _initialism(tokens: list[str]) -> str:
    """``J P Morgan Chase`` -> ``jpmc``-style key from first letters."""
    return "".join(t[0] for t in tokens if t)


# --------------------------------------------------------------------------
# address normalisation
# --------------------------------------------------------------------------

def normalize_address(raw: str) -> "AddressFeatures":
    """Return tokenised + key views of an address."""
    full = basic_clean(raw)
    toks = tokenize(full)

    # Two-stage expansion: address vocabulary first, then region table.
    toks = expand_tokens(toks, ADDRESS_ABBREVS)
    toks = expand_tokens(toks, REGION_ABBREVS)
    toks = dedupe_preserve_order(toks)

    content_toks = [t for t in toks if t not in STOP_TOKENS]

    digits = tuple(_DIGITS_RE.findall(full))
    postals = tuple(dict.fromkeys(_POSTAL_RE.findall(str(raw or ""))))
    # Fall back to digit runs of length 5/6 when the regex missed (e.g.
    # "PIN : 110001" or French "75001 Cedex").
    if not postals:
        postals = tuple(d for d in digits if len(d) in (5, 6))

    phone = _PHONE_RE.search(re.sub(r"\s+", " ", str(raw or "")))

    return AddressFeatures(
        raw=str(raw or ""),
        full=full,
        tokens=toks,
        content=content_toks,
        sorted_tokens=" ".join(sorted(content_toks)),
        digit_set=frozenset(digits),
        postcodes=postals,
        phone=phone.group(0) if phone else "",
    )


# --------------------------------------------------------------------------
# result containers
# --------------------------------------------------------------------------

class NameFeatures:
    __slots__ = (
        "raw", "full", "full_tokens", "core", "core_tokens",
        "sorted_core", "content", "content_tokens", "sorted_content",
        "initial_prefix",
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])

    @property
    def is_empty(self) -> bool:
        return not self.full_tokens

    def __repr__(self) -> str:  # pragma: no cover
        return f"NameFeatures(core={self.core!r})"


class AddressFeatures:
    __slots__ = (
        "raw", "full", "tokens", "content", "sorted_tokens",
        "digit_set", "postcodes", "phone",
    )

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw[k])

    @property
    def is_empty(self) -> bool:
        return not self.tokens

    def __repr__(self) -> str:  # pragma: no cover
        return f"AddressFeatures(full={self.full[:40]!r})"


# --------------------------------------------------------------------------
# blocking keys (cheap, exact-match oriented)
# --------------------------------------------------------------------------

def blocking_keys(name: NameFeatures, addr: AddressFeatures) -> set[str]:
    """Exact structural keys for the packed-key blocking index.

    These are the precision-grade half of candidate generation -- word-order
    invariant name forms, initialisms, first|last token pairs, postcodes,
    house-number|street. The fuzzy half comes from the sampled character
    4-gram shingles built alongside them in ``blocking.py``.
    """
    keys: set[str] = set()

    core = name.core
    if core:
        keys.add("N:" + core)
        keys.add("N:" + name.sorted_core)
        if name.initial_prefix and len(name.initial_prefix) >= 3:
            keys.add("I:" + name.initial_prefix)

    # First + last distinctive token is a strong, very selective key.
    ct = name.content_tokens
    if len(ct) >= 2:
        keys.add("F:" + ct[0] + "|" + ct[-1])
    elif len(ct) == 1:
        keys.add("F:" + ct[0])

    # Longest address token pair (street name) when present.
    at = addr.content
    if len(at) >= 2:
        keys.add("A:" + at[0] + "|" + at[1])

    # Postal codes are highly selective within a country.
    for p in addr.postcodes:
        keys.add("P:" + p)

    # Normalised phone digits, if the address carries one.
    if addr.phone:
        digits = re.sub(r"\D", "", addr.phone)
        if 7 <= len(digits) <= 15:
            keys.add("T:" + digits)

    # Full-name character 3-gram shingle over the core, for typo tolerance
    # at the exact-key layer (cheap and bounded).
    if len(core) >= 6:
        keys.add("G:" + core[:4])

    return keys
