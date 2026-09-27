"""Memory-lean record storage and blocking-key emission.

Why this exists
---------------
The real dataset is large (test: 1.73M / 4.89M / 5.08M rows for S1/S2/S3;
train: 2.21M / 5.03M / 5.29M) on a machine with 7.6 GB of RAM.  Holding the
data as Python objects -- one ``Record`` plus several normalised strings per
row -- costs well over 10 GB and will not fit.

So records are stored as **one contiguous byte buffer plus int32 offsets per
field**.  A test-split target set (S2+S3, ~9.97M rows) costs roughly 1 GB of
blob plus 120 MB of offsets instead of 6-8 GB of Python objects.

Blocking keys are emitted as ``(crc32(key), row_index)`` packed into a single
``uint64``.  Sorting that array groups identical keys together with no
per-key Python objects at all: 60M keys cost 480 MB instead of several GB of
``dict[str, list[int]]``.

``zlib.crc32`` is used rather than ``hash()`` so results are reproducible
across runs (``hash()`` is randomised per process by ``PYTHONHASHSEED``).
A crc32 collision can only *add* a candidate pair, never remove one, so
recall is unaffected and the classifier filters the extra false positives.
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import Sequence

import numpy as np

from .normalize import (
    AddressFeatures,
    NameFeatures,
    basic_clean,
    dedupe_preserve_order,
    expand_tokens,
    strip_suffixes,
    tokenize,
)
from .config import ADDRESS_ABBREVS, NAME_ABBREVS, REGION_ABBREVS, STOP_TOKENS

__all__ = [
    "RecordStore",
    "load_store",
    "save_store",
    "load_store_mmap",
    "country_string",
    "derive_name_features",
    "derive_addr_features",
    "record_keys",
    "pack_keys",
    "crc32",
]

# Field ids used when unpacking offsets.
FIELD_NAME = 0
FIELD_ADDR = 1

POSTCODE_LEN = {5, 6}


# --------------------------------------------------------------------------
# key hashing
# --------------------------------------------------------------------------

def crc32(data: bytes) -> int:
    """Stable 32-bit hash (unlike the per-process randomised ``hash()``)."""
    return zlib.crc32(data) & 0xFFFFFFFF


def pack_keys(keys, row_index: int, out: list[int]) -> int:
    """Append ``crc32(k) << 32 | row_index`` for each key. Returns count."""
    base = row_index
    for k in keys:
        out.append((crc32(k) << 32) | base)
    return len(keys)


# --------------------------------------------------------------------------
# feature reconstruction from a single cleaned string
# --------------------------------------------------------------------------

def derive_name_features(raw_cleaned: str) -> NameFeatures:
    """Rebuild a ``NameFeatures`` from one already-cleaned string.

    ``raw_cleaned`` has passed ``basic_clean`` (lowercased, accent-folded,
    punctuation -> space).  Everything else -- abbreviation expansion,
    suffix stripping, stopword removal, sorting -- is cheap pure-Python work
    over a handful of tokens, so it is done per pair rather than stored
    (storing it would multiply memory by ~6x).
    """
    full_toks = tokenize(raw_cleaned)
    expanded = dedupe_preserve_order(expand_tokens(full_toks, NAME_ABBREVS))
    core_toks = strip_suffixes(expanded)
    content_toks = [t for t in core_toks if t not in STOP_TOKENS]
    if not content_toks:
        content_toks = core_toks
    return NameFeatures(
        raw=raw_cleaned,
        full=raw_cleaned,
        full_tokens=full_toks,
        core=" ".join(core_toks),
        core_tokens=core_toks,
        sorted_core=" ".join(sorted(core_toks)),
        content=" ".join(content_toks),
        content_tokens=content_toks,
        sorted_content=" ".join(sorted(content_toks)),
        initial_prefix="".join(t[0] for t in content_toks if t),
    )


def derive_addr_features(raw_cleaned: str) -> AddressFeatures:
    """Rebuild an ``AddressFeatures`` from one already-cleaned string."""
    import re
    toks = tokenize(raw_cleaned)
    toks = dedupe_preserve_order(
        expand_tokens(expand_tokens(toks, ADDRESS_ABBREVS), REGION_ABBREVS)
    )
    content = [t for t in toks if t not in STOP_TOKENS]
    digits = tuple(re.findall(r"\d+", raw_cleaned))
    postals = tuple(d for d in digits if len(d) in POSTCODE_LEN)
    m = re.search(r"(?<!\d)(?:\+?\d[\d\-\s().]{7,}\d)(?!\d)", raw_cleaned)
    return AddressFeatures(
        raw=raw_cleaned,
        full=raw_cleaned,
        tokens=toks,
        content=content,
        sorted_tokens=" ".join(sorted(content)),
        digit_set=frozenset(digits),
        postcodes=postals,
        phone=m.group(0) if m else "",
    )


# --------------------------------------------------------------------------
# blocking keys
# --------------------------------------------------------------------------

MAX_KEYS = 8


def record_keys(name_cleaned: str, addr_cleaned: str) -> list[bytes]:
    """Selective blocking keys for one record (as bytes for crc32).

    Every key is *exact* -- its value is designed to be identical for two
    records that describe the same business, so a shared key means a
    candidate pair.  Keys whose group grows beyond the caller's DF cap are
    dropped at query time, which keeps fan-out bounded without needing a
    document-frequency pass over the corpus.
    """
    keys: list[bytes] = []

    n_toks = tokenize(name_cleaned)
    n_exp = dedupe_preserve_order(expand_tokens(n_toks, NAME_ABBREVS))
    n_core = strip_suffixes(n_exp)
    n_content = [t for t in n_core if t not in STOP_TOKENS] or n_core
    sorted_core = " ".join(sorted(n_core))

    # 1. word-order-invariant exact name
    if sorted_core:
        keys.append(b"n:" + sorted_core.encode())

    # 2. first + last distinctive token (survives a changed middle)
    if n_content:
        if len(n_content) >= 2:
            keys.append(b"f:" + (n_content[0] + "|" + n_content[-1]).encode())
        else:
            keys.append(b"f:" + n_content[0].encode())

    # 3. initialism  ("j p morgan chase" -> "jpmc")
    if len(n_content) >= 3:
        ini = "".join(t[0] for t in n_content)
        if len(ini) >= 3:
            keys.append(b"i:" + ini.encode())

    # 4. longest token -- the most distinctive single word in the name
    if n_content:
        longest = max(n_content, key=len)
        if len(longest) >= 5:
            keys.append(b"l:" + longest.encode())

    # 5. sorted-name prefix: tolerates typos inside a long name
    if len(sorted_core) >= 6:
        keys.append(b"g:" + sorted_core[:5].encode())

    # --- address -------------------------------------------------------
    a_toks = tokenize(addr_cleaned)
    a_exp = dedupe_preserve_order(
        expand_tokens(expand_tokens(a_toks, ADDRESS_ABBREVS), REGION_ABBREVS)
    )
    a_content = [t for t in a_exp if t not in STOP_TOKENS]

    # 6. postcode (highly selective within a country)
    import re
    digits = re.findall(r"\d+", addr_cleaned)
    for d in digits:
        if len(d) in POSTCODE_LEN:
            keys.append(b"p:" + d.encode())
            break

    # 7. house number + first street token
    if len(a_content) >= 2 and a_content[0].isdigit():
        keys.append(b"a:" + (a_content[0] + "|" + a_content[1]).encode())

    return keys[:MAX_KEYS]


# --------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------

class RecordStore:
    """Column-oriented storage: one byte buffer + int32 offsets per field.

    ``country`` is dict-encoded to a ``uint8`` code rather than kept as a
    list of Python strings -- 11.7M records would otherwise cost ~700 MB of
    str objects for four distinct values, versus 11 MB here.
    """

    __slots__ = (
        "id_bytes", "id_off",
        "name_bytes", "name_off",
        "addr_bytes", "addr_off",
        "country_codes", "country_vocab",
        "n", "_ids", "_names", "_addrs",
        "mmap_files",
    )

    def __init__(self):
        self.id_bytes = b""
        self.id_off = np.zeros(0, dtype=np.int64)
        self.name_bytes = b""
        self.name_off = np.zeros(0, dtype=np.int32)
        self.addr_bytes = b""
        self.addr_off = np.zeros(0, dtype=np.int32)
        self.country_codes = np.zeros(0, dtype=np.uint8)
        self.country_vocab: list[str] = []
        self.n = 0
        self._ids: list[str] | None = None
        self._names: list[str] | None = None
        self._addrs: list[str] | None = None
        # Keeps the backing file handles alive for mmap-backed stores.
        self.mmap_files: list = []

    # -- element access ----------------------------------------------------

    @property
    def ids(self) -> list[str]:
        if self._ids is None:
            self._ids = _split_all(self.id_bytes, self.id_off)
        return self._ids

    @property
    def names(self) -> list[str]:
        if self._names is None:
            self._names = _split_all(self.name_bytes, self.name_off)
        return self._names

    @property
    def addrs(self) -> list[str]:
        if self._addrs is None:
            self._addrs = _split_all(self.addr_bytes, self.addr_off)
        return self._addrs

    def name_of(self, i: int) -> str:
        if self._names is not None:
            return self._names[i]
        a, b = self.name_off[i], self.name_off[i + 1]
        return self.name_bytes[a:b].decode("utf-8")

    def addr_of(self, i: int) -> str:
        if self._addrs is not None:
            return self._addrs[i]
        a, b = self.addr_off[i], self.addr_off[i + 1]
        return self.addr_bytes[a:b].decode("utf-8")

    def id_of(self, i: int) -> str:
        if self._ids is not None:
            return self._ids[i]
        a, b = self.id_off[i], self.id_off[i + 1]
        return self.id_bytes[a:b].decode("utf-8")

    def drop_caches(self) -> None:
        """Release the materialised list views, keeping only the blobs."""
        self._ids = None
        self._names = None
        self._addrs = None

    def __repr__(self) -> str:  # pragma: no cover
        mb = (len(self.id_bytes) + len(self.name_bytes) + len(self.addr_bytes)) / 1e6
        return f"<RecordStore n={self.n:,} blob={mb:.0f}MB>"


def _split_all(buf: bytes, off: np.ndarray) -> list[str]:
    """Decode every field from a blob using its offset array."""
    n = len(off) - 1
    out: list[str] = [""] * n
    # Bound the work: slicing a bytes object is O(len).
    for i in range(n):
        out[i] = buf[off[i]:off[i + 1]].decode("utf-8")
    return out


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------

def load_store(
    path: str | Path,
    *,
    limit: int | None = None,
    keep_ids: bool = True,
) -> RecordStore:
    """Stream a source TSV into a ``RecordStore``.

    Only the columns we actually need are kept: ``entity_id`` (to emit
    output), ``business_name`` / ``business_address`` run through
    ``basic_clean`` so downstream work never repeats regexes over raw text,
    and ``country`` dict-encoded to a uint8 (four distinct labels for ~11.7M
    rows would otherwise cost ~700 MB of Python str objects).

    ``limit`` truncates the file, which is how the smoke tests stay fast.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    store = RecordStore()
    id_b = bytearray()
    name_b = bytearray()
    addr_b = bytearray()
    id_o = [0]
    name_o = [0]
    addr_o = [0]
    codes: list[int] = []
    vocab: dict[str, int] = {}

    import csv as _csv
    with path.open("r", encoding="utf-8", newline="") as fh:
        first = fh.readline()
        if "\t" not in first:
            raise ValueError(
                f"{path} is not tab-separated (header: {first[:120]!r}); "
                "read it with sep='\\t'."
            )
        fh.seek(0)
        reader = _csv.DictReader(fh, delimiter="\t", quoting=_csv.QUOTE_MINIMAL)
        if reader.fieldnames and "entity_id" not in reader.fieldnames:
            raise ValueError(f"{path}: missing entity_id; got {reader.fieldnames}")
        for row in reader:
            eid = (row.get("entity_id") or "").strip()
            if not eid:
                continue
            id_b += eid.encode("utf-8")
            id_o.append(len(id_b))
            nm = basic_clean(row.get("business_name") or "")
            name_b += nm.encode("utf-8")
            name_o.append(len(name_b))
            ad = basic_clean(row.get("business_address") or "")
            addr_b += ad.encode("utf-8")
            addr_o.append(len(addr_b))
            # Dict-encode country.  The open set (US / India / France / ...)
            # is never hard-coded: unseen values are appended to the vocab.
            c = (row.get("country") or "").strip()
            code = vocab.get(c)
            if code is None:
                code = min(len(vocab), 255)
                vocab[c] = code
            codes.append(code)
            if limit is not None and len(id_o) - 1 >= limit:
                break

    store.id_bytes = bytes(id_b)
    store.id_off = np.asarray(id_o, dtype=np.int64)
    store.name_bytes = bytes(name_b)
    store.name_off = np.asarray(name_o, dtype=np.int32)
    store.addr_bytes = bytes(addr_b)
    store.addr_off = np.asarray(addr_o, dtype=np.int32)
    store.country_vocab = [k for k, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    store.country_codes = (
        np.asarray(codes, dtype=np.uint8) if codes
        else np.zeros(0, dtype=np.uint8)
    )
    store.n = len(id_o) - 1
    return store


def country_string(store: RecordStore, i: int) -> str:
    """Country label of row ``i`` (open set -- never filtered)."""
    if i < len(store.country_codes):
        code = int(store.country_codes[i])
        if code < len(store.country_vocab):
            return store.country_vocab[code]
    return ""


def load_countries(path: str | Path, limit: int | None = None) -> list[str]:
    """Read just the ``country`` column (used for country-match features)."""
    path = Path(path)
    out: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        header = fh.readline()
        if "\t" not in header:
            raise ValueError(f"{path}: not tab-separated")
        cols = header.rstrip("\n").split("\t")
        if "country" not in cols:
            return [""]
        ci = cols.index("country")
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            parts = line.rstrip("\n").split("\t")
            out.append(parts[ci] if ci < len(parts) else "")
    return out


# --------------------------------------------------------------------------
# spill / reload for multiprocessing
# --------------------------------------------------------------------------

def save_store(store: RecordStore, out_dir: str | Path) -> None:
    """Write a store to disk so worker processes can memory-map it.

    Blobs are written as raw bytes and columns as ``.npy``.  This exists so
    N worker processes share **one** copy of ~1 GB of target data through the
    page cache instead of each holding their own -- on a 7.6 GB machine that
    is the difference between 8 workers fitting and not fitting.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "id_bytes.bin").write_bytes(store.id_bytes)
    (out_dir / "name_bytes.bin").write_bytes(store.name_bytes)
    (out_dir / "addr_bytes.bin").write_bytes(store.addr_bytes)
    np.save(out_dir / "id_off.npy", store.id_off)
    np.save(out_dir / "name_off.npy", store.name_off)
    np.save(out_dir / "addr_off.npy", store.addr_off)
    np.save(out_dir / "country_codes.npy", store.country_codes)
    import json as _json
    (out_dir / "country_vocab.json").write_text(
        _json.dumps(store.country_vocab), encoding="utf-8"
    )
    (out_dir / "n.txt").write_text(str(store.n), encoding="utf-8")


def _open_blob(path: Path):
    """Memory-map a raw byte file; ``bytes``-like and sliceable."""
    import mmap as _mmap
    if not path.exists() or path.stat().st_size == 0:
        return b""
    fh = open(path, "rb")
    mm = _mmap.mmap(fh.fileno(), 0, access=_mmap.ACCESS_READ)
    # mmap dups the fd internally, but we keep the handle on the store too so
    # the mapping cannot outlive its file unexpectedly.
    return mm, fh


def load_store_mmap(in_dir: str | Path) -> RecordStore:
    """Reload a store written by :func:`save_store`, memory-mapping the blobs.

    ``name_bytes[a:b].decode()`` behaves identically to the in-memory bytes
    case, so the rest of the pipeline is unaware of which mode it is in.
    """
    in_dir = Path(in_dir)
    store = RecordStore()
    for attr, fname in (
        ("id_bytes", "id_bytes.bin"),
        ("name_bytes", "name_bytes.bin"),
        ("addr_bytes", "addr_bytes.bin"),
    ):
        res = _open_blob(in_dir / fname)
        if isinstance(res, tuple):
            mm, fh = res
            store.mmap_files.append(fh)
            setattr(store, attr, mm)
        else:
            setattr(store, attr, res)

    store.id_off = np.load(in_dir / "id_off.npy", mmap_mode="r")
    store.name_off = np.load(in_dir / "name_off.npy", mmap_mode="r")
    store.addr_off = np.load(in_dir / "addr_off.npy", mmap_mode="r")
    store.country_codes = np.load(in_dir / "country_codes.npy", mmap_mode="r")
    import json as _json
    store.country_vocab = _json.loads(
        (in_dir / "country_vocab.json").read_text(encoding="utf-8")
    )
    store.n = int((in_dir / "n.txt").read_text(encoding="utf-8").strip())
    return store


def store_from_rows(
    rows: Sequence[tuple[str, str, str, str]],
) -> RecordStore:
    """Build a small in-memory store from ``(id, name, addr, country)`` rows.

    Worker processes receive a Source 1 chunk as plain pickled tuples -- far
    cheaper to ship than a store object -- and reconstitute it here.  Inputs
    are already ``basic_clean``-ed, matching what :func:`load_store` produces.
    """
    store = RecordStore()
    id_b = bytearray()
    name_b = bytearray()
    addr_b = bytearray()
    id_o = [0]
    name_o = [0]
    addr_o = [0]
    codes: list[int] = []
    vocab: dict[str, int] = {}
    for eid, nm, ad, country in rows:
        id_b += eid.encode("utf-8")
        id_o.append(len(id_b))
        name_b += nm.encode("utf-8")
        name_o.append(len(name_b))
        addr_b += ad.encode("utf-8")
        addr_o.append(len(addr_b))
        code = vocab.get(country)
        if code is None:
            code = len(vocab)
            vocab[country] = code
        codes.append(code)
    store.id_bytes = bytes(id_b)
    store.id_off = np.asarray(id_o, dtype=np.int64)
    store.name_bytes = bytes(name_b)
    store.name_off = np.asarray(name_o, dtype=np.int32)
    store.addr_bytes = bytes(addr_b)
    store.addr_off = np.asarray(addr_o, dtype=np.int32)
    store.country_vocab = [k for k, _ in sorted(vocab.items(), key=lambda kv: kv[1])]
    store.country_codes = (
        np.asarray(codes, dtype=np.uint8) if codes
        else np.zeros(0, dtype=np.uint8)
    )
    store.n = len(id_o) - 1
    return store
