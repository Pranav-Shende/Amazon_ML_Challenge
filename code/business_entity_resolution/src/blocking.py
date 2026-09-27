"""Candidate generation / blocking, sized for ~10M-record corpora.

Why not TF-IDF cosine?
----------------------
The obvious approach -- a sparse TF-IDF similarity matrix between Source 1
(1.73M rows) and Source 2 (4.89M rows) -- is infeasible here: one 2,000-row
chunk against a single source needs ~39 GB of dense float32, and this machine
has 7.6 GB total.  Brute-force nearest neighbours is out for the same reason.

Instead blocking is done with **exact keys packed into a sorted uint64 array**:

    packed = crc32(key) << 32 | row_index

Sorting that single array groups all records sharing a key into a contiguous
run, so candidate lookup is two ``np.searchsorted`` calls per key -- no
``dict[str, list[int]]`` (which would cost several GB for 60M postings) and
no Python object per posting.

Two properties make this practical:

* **DF cap.**  A key shared by more than ``df_cap`` records is skipped
  entirely at query time, which bounds fan-out adaptively: ubiquitous tokens
  like "street" drop out on their own while rare ones are kept.  The cap is
  applied at query time, so it costs no extra pass over the data.
* **crc32 collisions only add pairs.**  A hash collision unions two key
  groups, producing extra candidates -- never fewer -- so recall is
  unaffected and the classifier simply filters the extras.

Keys are a mix of *exact* structure (sorted core name, postcode, initialism,
house-number+street) and *fuzzy* structure (sampled character 4-grams of the
sorted name), so a single typo invalidates at most one shingle while the rest
still connect the pair.
"""

from __future__ import annotations

import logging

import numpy as np

from .config import NAME_ABBREVS, PipelineConfig
from .normalize import dedupe_preserve_order, expand_tokens, strip_suffixes, tokenize
from .store import RecordStore, crc32, record_keys

log = logging.getLogger(__name__)

# Sample every Nth character 4-gram of the sorted core name.  Sampling keeps
# the build array small; a typo still only destroys its own shingle, because
# the remaining sampled shingles are unaffected.
GRAM_STRIDE = 3
GRAM_LEN = 4
MAX_GRAMS = 6


def _name_grams(sorted_core: str) -> list[bytes]:
    """Order-invariant, typo-tolerant shingles of a business name."""
    if len(sorted_core) < GRAM_LEN:
        return []
    grams: list[bytes] = []
    for i in range(0, len(sorted_core) - GRAM_LEN + 1, GRAM_STRIDE):
        grams.append(b"c:" + sorted_core[i:i + GRAM_LEN].encode())
        if len(grams) >= MAX_GRAMS:
            break
    return grams


def keys_for(name_cleaned: str, addr_cleaned: str) -> list[bytes]:
    """All blocking keys for one record: exact structure + sampled shingles."""
    keys = record_keys(name_cleaned, addr_cleaned)
    core = strip_suffixes(
        dedupe_preserve_order(expand_tokens(tokenize(name_cleaned), NAME_ABBREVS))
    )
    sorted_core = " ".join(sorted(core))
    keys.extend(_name_grams(sorted_core))
    return keys


# Batch size for the streaming build.  ~4M postings is ~32 MB of packed
# values -- small enough to hold twice, big enough that the file I/O is a
# single sequential write per ~3M records.
FLUSH_POSTINGS = 4_000_000


class KeyIndex:
    """Sorted packed-key array enabling DF-capped candidate lookup."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.packed: np.ndarray = np.zeros(0, dtype=np.uint64)
        self.n_keys = 0

    # -- build -------------------------------------------------------------

    @classmethod
    def build(cls, store: RecordStore, cfg: PipelineConfig) -> "KeyIndex":
        idx = cls(cfg)
        n = store.n
        if n == 0:
            return idx

        # Read straight from the byte blobs rather than materialising
        # `store.names` / `store.addrs`: for the 10M-row target files those
        # lists cost ~1.1 GB of Python str objects transiently, which this
        # machine cannot afford.  A blob slice + decode is ~0.3 us.
        name_of = store.name_of
        addr_of = store.addr_of

        # Streaming build -- the accumulator is the reason this cannot simply
        # be `buf: list[int]`.  A 10.3M-record store emits ~134M postings, and
        # a Python list of that many non-small ints is ~6.4 GB of objects on a
        # machine with 7.6 GB total.  Flushing batches to a scratch file and
        # reading them straight back with `np.fromfile` keeps the peak at the
        # array itself (~1.1 GB) plus one batch, independent of corpus size.
        import os
        import tempfile

        fd, tmp_path = tempfile.mkstemp(prefix="ber_keys_", suffix=".u64")
        try:
            with os.fdopen(fd, "wb") as fh:
                pending: list[int] = []
                append = pending.append
                for i in range(n):
                    for k in keys_for(name_of(i), addr_of(i)):
                        append((crc32(k) << 32) | i)
                    if len(pending) >= FLUSH_POSTINGS:
                        fh.write(np.asarray(pending, dtype=np.uint64).tobytes())
                        pending.clear()
                if pending:
                    fh.write(np.asarray(pending, dtype=np.uint64).tobytes())
                nbytes = fh.tell()

            if nbytes == 0:
                log.warning("No blocking keys emitted for %d records.", n)
                idx.packed = np.zeros(0, dtype=np.uint64)
                idx.n_keys = 0
                return idx

            arr = np.fromfile(tmp_path, dtype=np.uint64)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        arr.sort(kind="quicksort")
        idx.packed = arr
        idx.n_keys = int(arr.size)
        log.info(
            "Key index: %s records -> %s keys (%.0f MB, mean %.1f keys/record)",
            f"{n:,}", f"{idx.n_keys:,}", arr.nbytes / 1e6, idx.n_keys / n,
        )
        return idx

    # -- query -------------------------------------------------------------

    def lookup(self, key: bytes) -> np.ndarray:
        """Return row indices sharing ``key``, empty if over the DF cap."""
        if self.packed.size == 0:
            return _EMPTY_IDX
        c = crc32(key)
        base = np.uint64(c) << np.uint64(32)
        lo = int(np.searchsorted(self.packed, base))
        if c == 0xFFFFFFFF:
            hi = self.packed.size
        else:
            hi = int(np.searchsorted(self.packed, np.uint64(c + 1) << np.uint64(32)))
        size = hi - lo
        if size == 0:
            return _EMPTY_IDX
        if size > self.cfg.df_cap:
            # Too generic to be informative; skipping keeps fan-out bounded.
            return _EMPTY_IDX
        return (self.packed[lo:hi] & np.uint64(0xFFFFFFFF)).astype(np.int64)

    def candidates_for(
        self,
        name_cleaned: str,
        addr_cleaned: str,
        *,
        max_candidates: int,
        rarest_first: bool = True,
    ) -> np.ndarray:
        """Candidate row indices for one query record.

        When the budget is exceeded, keys are consumed in ascending group-size
        order (``rarest_first``), because the rarest shared key is the most
        selective evidence available and therefore the least likely to flood
        the budget with noise.
        """
        keys = keys_for(name_cleaned, addr_cleaned)
        if not keys:
            return _EMPTY_IDX

        groups = []
        for k in keys:
            hits = self.lookup(k)
            if hits.size:
                groups.append(hits)
        if not groups:
            return _EMPTY_IDX

        if rarest_first and len(groups) > 1:
            groups.sort(key=lambda a: a.size)

        # Concatenate then dedupe, preferring rarer groups first.
        merged = np.concatenate(groups)
        uniq = np.unique(merged)
        if uniq.size > max_candidates:
            # Keep rows reachable via the rarest keys; np.unique sorts by id,
            # so re-derive membership in the order groups were prioritised.
            allowed = np.unique(np.concatenate(groups[:1]))
            for g in groups[1:]:
                if allowed.size >= max_candidates:
                    break
                allowed = np.unique(np.concatenate([allowed, g]))
            uniq = allowed[:max_candidates]
            uniq.sort()
        return uniq

    def memory_mb(self) -> float:
        return self.packed.nbytes / 1e6

    # -- spill / reload -----------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the packed array for workers to memory-map."""
        import os
        from pathlib import Path as _P
        path = _P(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save to a temporary name then rename: a worker that arrives early
        # must never observe a half-written array.
        tmp = path.parent / (path.stem + ".writing.npy")
        np.save(tmp, self.packed)
        os.replace(tmp, path)

    @classmethod
    def load(cls, cfg: PipelineConfig, path: str | Path) -> "KeyIndex":
        """Memory-map an index saved by :meth:`save` (read-only).

        Shared through the page cache, so N workers cost one copy.
        """
        idx = cls(cfg)
        arr = np.load(path, mmap_mode="r")
        idx.packed = arr.view(np.uint64)
        idx.n_keys = int(arr.size)
        return idx


_EMPTY_IDX = np.zeros(0, dtype=np.int64)
