"""End-to-end orchestration, sized for the real challenge data.

The dataset is much larger than a laptop-friendly pipeline can hold as
Python objects:

    train  S1 2,206,821   S2 5,034,616   S3 5,285,603   (7.64M ground-truth links)
    test   S1 1,732,544   S2 4,887,273   S3 5,082,316

on a machine with 7.6 GB of RAM.  So this module:

* stores each split in **column blobs** (``store.py``) instead of objects,
* blocks with a **sorted packed-key array** instead of a TF-IDF matrix,
* learns from **mini-batches** (``SGDClassifier``) instead of one giant
  feature matrix,
* **frees the training data before loading the test data**, so the two
  splits are never resident at once,
* streams Source 1 and **writes output rows incrementally** in file order,
  which guarantees exactly one row per entity without holding results.

Modes
-----
``test``        produce the two submission files (trains first if labels exist)
``train-eval``  score a held-out slice of train; reports macro F_0.5
``predict``     inference from a previously saved model
``validate``    check output files against the challenge rules
"""

from __future__ import annotations

import csv
import gc
import logging
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .blocking import KeyIndex
from .config import PipelineConfig
from .features import N_FEATURES, TokenStats, extract_features
from .model import MatchingModel, f05_at_threshold, macro_f05
from .normalize import normalize_address, normalize_name
from .store import (
    RecordStore,
    country_string,
    derive_addr_features,
    derive_name_features,
    load_store,
    save_store,
)

log = logging.getLogger(__name__)

MATCHING_HEADER = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_HEADER = ["source1_entity_id", "candidate_entity_ids"]


# --------------------------------------------------------------------------
# containers
# --------------------------------------------------------------------------

@dataclass
class TargetSide:
    """One or both of Source 2 / Source 3, each with its own key index.

    Keeping them separate avoids concatenating ~1 GB of blobs (which would
    transiently double peak memory); row indices are just offset by the
    preceding store's length.
    """

    stores: list[RecordStore] = field(default_factory=list)
    indexes: list[KeyIndex] = field(default_factory=list)

    @property
    def n_total(self) -> int:
        return sum(s.n for s in self.stores)

    @property
    def offsets(self) -> list[int]:
        out, acc = [], 0
        for s in self.stores:
            out.append(acc)
            acc += s.n
        return out

    def id_of(self, store_i: int, row: int) -> str:
        return self.stores[store_i].id_of(row)

    def name_of(self, store_i: int, row: int) -> str:
        return self.stores[store_i].name_of(row)

    def addr_of(self, store_i: int, row: int) -> str:
        return self.stores[store_i].addr_of(row)

    def country_of(self, store_i: int, row: int) -> str:
        return country_string(self.stores[store_i], row)

    def memory_mb(self) -> float:
        return sum(i.memory_mb() for i in self.indexes)


def load_split_targets(
    split_dir: Path, split: str, cfg: PipelineConfig, limit: int | None = None,
) -> TargetSide:
    """Load Source 2 + Source 3 for a split and build a key index over each."""
    side = TargetSide()
    for suffix in ("2", "3"):
        path = split_dir / f"{split}_source{suffix}.tsv"
        if not path.exists():
            log.warning("Missing %s -- skipping that source.", path)
            continue
        t0 = time.time()
        store = load_store(path, limit=limit)
        log.info("Loaded %s: %s rows in %.1fs", path.name, f"{store.n:,}", time.time() - t0)
        t0 = time.time()
        idx = KeyIndex.build(store, cfg)
        log.info("  index built in %.1fs (%.0f MB)", time.time() - t0, idx.memory_mb())
        side.stores.append(store)
        side.indexes.append(idx)
        # Names/addresses are read from the blob on demand during scoring, so
        # any list materialised by earlier stages can be released here.
        store.drop_caches()
        gc.collect()
    return side


# --------------------------------------------------------------------------
# ground truth
# --------------------------------------------------------------------------

def load_truth(
    path: Path,
    limit: int | None = None,
    keep_ids: set[str] | None = None,
) -> dict[str, set[str]]:
    """``{source1_entity_id: {matched ids}}``.

    Stored as id strings rather than resolved row indices: a 10.3M-entry
    ``str -> int`` map costs ~1.4 GB, while the ground truth itself is
    ~900 MB, and membership checks against it are O(1) either way.

    ``keep_ids`` retains only those Source 1 rows -- the single largest
    memory saving available.  Only the fit slice plus the calibration slice
    are ever consulted, but the file covers all 2.2M rows, so an unfiltered
    load spends ~1.4 GB to answer lookups for ~150k of them.  On a 7.6 GB
    machine that difference is the one that keeps the process resident
    instead of paging.
    """
    if not path.exists():
        log.warning("Ground truth not found at %s", path)
        return {}
    truth: dict[str, set[str]] = {}
    kept = 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        header = fh.readline()
        if "\t" not in header:
            raise ValueError(f"{path}: not tab-separated")
        cols = header.rstrip("\n").split("\t")
        if "source1_entity_id" not in cols:
            raise ValueError(f"{path}: missing source1_entity_id; got {cols}")
        si = cols.index("source1_entity_id")
        mi = cols.index("matched_entity_ids") if "matched_entity_ids" in cols else si + 1
        for i, line in enumerate(fh):
            if limit is not None and i >= limit:
                break
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= si:
                continue
            s1 = parts[si].strip()
            if not s1:
                continue
            if keep_ids is not None and s1 not in keep_ids:
                continue
            kept += 1
            raw = parts[mi] if mi < len(parts) else ""
            truth[s1] = {t.strip() for t in raw.split(",") if t.strip()}
    log.info("Loaded %s ground-truth rows from %s", f"{len(truth):,}", path.name)
    return truth


# --------------------------------------------------------------------------
# shared feature helper
# --------------------------------------------------------------------------

def pair_features(
    qn, qa, q_country: str,
    tn, ta, t_country: str,
    stats: TokenStats | None,
) -> np.ndarray:
    """One candidate pair -> feature vector.

    ``qn``/``qa`` are already-derived ``NameFeatures``/``AddressFeatures``
    for the query, so callers derive those once per row rather than once per
    pair.
    """
    return extract_features(qn, qa, q_country, tn, ta, t_country, stats)


def _derive_query(store: RecordStore, i: int):
    return derive_name_features(store.name_of(i)), derive_addr_features(store.addr_of(i))


def _derive_target(store: RecordStore, i: int):
    return derive_name_features(store.name_of(i)), derive_addr_features(store.addr_of(i))


# --------------------------------------------------------------------------
# chunked Source-1 traversal
# --------------------------------------------------------------------------

def iter_s1_chunks(n: int, chunk: int):
    for start in range(0, n, chunk):
        yield start, min(start + chunk, n)


# --------------------------------------------------------------------------
# parallel scoring
# --------------------------------------------------------------------------
#
# Windows uses `spawn`, not `fork`, so workers cannot inherit the parent's
# memory.  Instead the shared state (~2 GB of target blobs + packed key
# index + the fitted model) is spilled once to a temp directory and each
# worker memory-maps it.  The page cache then holds **one** copy across all
# workers -- the only reason 8 processes fit inside 7.6 GB of RAM.

_WORKER: dict = {}


def _init_worker(shared_dir: str, cfg: PipelineConfig, model_path: str) -> None:
    """Attach this process to the shared blobs, key index, and model."""
    from .blocking import KeyIndex
    from .model import MatchingModel
    from .store import load_store_mmap

    shared = Path(shared_dir)
    side = TargetSide()
    for tag in ("source2", "source3"):
        d = shared / tag
        if (d / "n.txt").exists():
            side.stores.append(load_store_mmap(d))
            side.indexes.append(KeyIndex.load(cfg, d / "packed.npy"))
    _WORKER["targets"] = side
    _WORKER["cfg"] = cfg
    _WORKER["model"] = MatchingModel.load(model_path, cfg)
    _WORKER["chunks"] = shared / "chunks"


def _run_chunk(
    task: tuple[int, list[tuple[str, str, str, str]]],
) -> tuple[int, int, int, int]:
    """Score one Source 1 chunk; write rows to per-chunk files.

    Returns ``(chunk_id, matched, singletons, candidate_pairs)`` so the
    parent can tally totals without re-reading 1.7M lines.
    """
    cid, rows = task
    from .store import store_from_rows

    model = _WORKER["model"]
    cfg = _WORKER["cfg"]
    targets = _WORKER["targets"]
    out = Path(_WORKER["chunks"])

    s1 = store_from_rows(rows)
    preds, cands = score_chunk(model, s1, targets, 0, s1.n, None, cfg)

    n_matched = n_singletons = n_candidates = 0
    with (out / f"m{cid:06d}.bin").open("wb") as mf, \
         (out / f"c{cid:06d}.bin").open("wb") as cf:
        tab_nl = "\t"
        for i in range(s1.n):
            eid = s1.id_of(i)
            p = preds.get(i, [])
            c = cands.get(i, [])
            mf.write(f"{eid}{tab_nl}{','.join(p)}\n".encode("utf-8"))
            cf.write(f"{eid}{tab_nl}{','.join(c)}\n".encode("utf-8"))
            n_candidates += len(c)
            if p:
                n_matched += 1
            else:
                n_singletons += 1
    return cid, n_matched, n_singletons, n_candidates


def _close_worker() -> None:
    """Release mmap handles this process opened (the in-process path only).

    Worker processes die with their mappings, but when ``--workers 1`` runs
    everything here, the handles must be closed *before* the temporary
    directory is deleted -- Windows refuses to unlink a file that is still
    mapped.
    """
    side = _WORKER.pop("targets", None)
    if side is not None:
        for st in side.stores:
            for attr in ("name_bytes", "addr_bytes", "id_bytes"):
                blob = getattr(st, attr, None)
                if blob is not None and hasattr(blob, "close"):
                    try:
                        blob.close()
                    except Exception:
                        pass
            for fh in st.mmap_files:
                try:
                    fh.close()
                except Exception:
                    pass
    _WORKER.pop("model", None)
    _WORKER.pop("cfg", None)
    _WORKER.pop("chunks", None)
    gc.collect()


def _s1_task_iter(s1: RecordStore, chunk: int):
    """Yield ``(chunk_id, rows)`` lazily so only one chunk is ever resident."""
    for cid, (start, end) in enumerate(iter_s1_chunks(s1.n, chunk)):
        rows = [
            (s1.id_of(i), s1.name_of(i), s1.addr_of(i), country_string(s1, i))
            for i in range(start, end)
        ]
        yield cid, rows


# --------------------------------------------------------------------------
# mode: test
# --------------------------------------------------------------------------

def _recover_model(
    cfg: PipelineConfig, report: dict, warning: str
) -> MatchingModel:
    """Prefer the Pass A checkpoint over discarding it for the heuristic.

    Calibration is the second half of training; if it raises, the fitted
    weights from the first half are still valid and worth ~20 minutes of
    refitting, so load them when they exist.
    """
    try:
        import tempfile as _tf
        ckpt = Path(_tf.gettempdir()) / "ber_fit_checkpoint.json"
        if ckpt.exists():
            m = MatchingModel.load(ckpt, cfg)
            if getattr(m, "_stream_started", False) or m._clf is not None:
                log.info("Recovered fitted model from %s", ckpt)
                report["train"] = {"mode": m.kind, "recovered_after": warning}
                return m
    except Exception as exc:                       # noqa: BLE001
        log.warning("Fitted-model checkpoint unusable: %s", exc)
    m = MatchingModel(cfg)
    m._init_heuristic()
    report["train"] = {"mode": "heuristic (training failed)"}
    report["train_warning"] = warning
    return m


def run_test(
    data_dir: Path,
    output_dir: Path,
    cfg: PipelineConfig,
    *,
    limit: int | None = None,
) -> dict:
    """Produce ``matching_results.tsv`` + ``candidate_pairs.tsv``."""
    import shutil
    import tempfile
    import os as _os

    t0 = time.time()
    report: dict = {"mode": "test", "config": cfg.tag()}

    model = MatchingModel(cfg)

    # ---- phase 1: supervised training (entirely freed before test) ------
    train_dir = data_dir / "dataset" / "train"
    if train_dir.exists() and (train_dir / "train_ground_truth.tsv").exists():
        try:
            train_report, model = train_on_train_split(train_dir, cfg, model, limit)
            report["train"] = train_report
        except MemoryError as exc:
            log.warning("Training ran out of memory (%s).", exc)
            model = _recover_model(cfg, report, f"MemoryError: {exc}")
        except Exception as exc:                   # noqa: BLE001
            log.warning("Training failed (%s: %s).", type(exc).__name__, exc)
            model = _recover_model(
                cfg, report, f"{type(exc).__name__}: {exc}"
            )
        # Drop everything the training phase allocated before touching test.
        gc.collect()
    else:
        log.info("No usable train split; using the heuristic scorer.")
        model._init_heuristic()
        report["train"] = {"mode": "heuristic (no train split)"}

    # ---- phase 2: blocking + scoring on the test split -------------------
    test_dir = data_dir / "dataset" / "test"
    s1_path = test_dir / "test_source1.tsv"
    if not s1_path.exists():
        raise FileNotFoundError(s1_path)

    t0_load = time.time()
    s1 = load_store(s1_path, limit=limit)
    log.info(
        "Loaded %s: %s rows in %.1fs", s1_path.name, f"{s1.n:,}",
        time.time() - t0_load,
    )
    n_chunks = max(1, -(-s1.n // cfg.query_chunk)) if s1.n else 0

    output_dir.mkdir(parents=True, exist_ok=True)
    match_path = output_dir / "matching_results.tsv"
    cand_path = output_dir / "candidate_pairs.tsv"

    workers = cfg.workers or (_os.cpu_count() or 2)
    workers = max(1, min(workers, n_chunks)) if n_chunks else 1

    n_matched = n_singletons = n_candidates = 0

    with tempfile.TemporaryDirectory(prefix="ber_shared_") as tmp:
        shared = Path(tmp)
        chunks_dir = shared / "chunks"
        chunks_dir.mkdir()

        # -- spill the shared state, then free it from this process --------
        t0_spill = time.time()
        targets = load_split_targets(test_dir, "test", cfg, limit=limit)
        for tag, st, ix in zip(("source2", "source3"),
                               targets.stores, targets.indexes):
            d = shared / tag
            save_store(st, d)
            ix.save(d / "packed.npy")
        model_path = shared / "model.json"
        model.save(model_path)
        log.info(
            "Spilled shared state for %d worker(s) in %.1fs",
            workers, time.time() - t0_spill,
        )
        del targets
        gc.collect()

        # -- dispatch -------------------------------------------------------
        tasks = _s1_task_iter(s1, cfg.query_chunk)
        if workers == 1:
            _init_worker(str(shared), cfg, str(model_path))
            done = 0
            for t in tasks:
                _, m, s, c = _run_chunk(t)
                n_matched += m
                n_singletons += s
                n_candidates += c
                done += 1
                if cfg.verbose and done % 25 == 0:
                    log.info("  finished %d / %d chunks", done, n_chunks)
            _close_worker()
        else:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(
                max_workers=workers,
                initializer=_init_worker,
                initargs=(str(shared), cfg, str(model_path)),
            ) as ex:
                done = 0
                for _cid, m, s, c in ex.map(_run_chunk, tasks, chunksize=1):
                    n_matched += m
                    n_singletons += s
                    n_candidates += c
                    done += 1
                    if cfg.verbose and done % 25 == 0:
                        log.info("  finished %d / %d chunks", done, n_chunks)

        # -- stitch the per-chunk files back together, in Source 1 order ----
        with match_path.open("wb") as mf, cand_path.open("wb") as cf:
            mf.write(("\t".join(MATCHING_HEADER) + "\n").encode("utf-8"))
            cf.write(("\t".join(CANDIDATE_HEADER) + "\n").encode("utf-8"))
            for cid in range(n_chunks):
                with (chunks_dir / f"m{cid:06d}.bin").open("rb") as f:
                    shutil.copyfileobj(f, mf, 1 << 20)
                with (chunks_dir / f"c{cid:06d}.bin").open("rb") as f:
                    shutil.copyfileobj(f, cf, 1 << 20)

    report.update(
        {
            "threshold": model.threshold,
            "model_kind": model.kind,
            "workers": workers,
            "n_source1": s1.n,
            "n_with_matches": n_matched,
            "n_singletons": n_singletons,
            "n_candidate_pairs": n_candidates,
            "candidates_per_row": round(n_candidates / max(s1.n, 1), 2),
            "matching_results": str(match_path),
            "candidate_pairs": str(cand_path),
            "seconds": round(time.time() - t0, 2),
        }
    )
    log.info(
        "Done in %.0fs | %s S1 rows | %s matched | %s singletons | %s candidates",
        report["seconds"], f"{s1.n:,}", f"{n_matched:,}",
        f"{n_singletons:,}", f"{n_candidates:,}",
    )
    return report


def score_chunk(
    model: MatchingModel,
    s1: RecordStore,
    targets: TargetSide,
    start: int,
    end: int,
    stats: TokenStats | None,
    cfg: PipelineConfig,
) -> tuple[dict[int, list[str]], dict[int, list[str]]]:
    """Score one chunk of Source 1. Returns (predictions, candidates) by row."""
    preds: dict[int, list[str]] = {}
    cands: dict[int, list[str]] = {}
    if not targets.stores:
        return preds, cands

    per_store_cap = max(4, cfg.max_candidates_per_row // max(len(targets.stores), 1))

    # Feature rows for the whole chunk, allocated once.
    feat_rows: list[np.ndarray] = []
    feat_owner: list[tuple[int, int, int]] = []   # (local_row, store_i, row)

    for local, qi in enumerate(range(start, end)):
        qn, qa = _derive_query(s1, qi)
        q_country = country_string(s1, qi)

        hit_ids: list[str] = []
        seen: set[str] = set()
        n_feat_before = len(feat_rows)

        for si, (store, index) in enumerate(zip(targets.stores, targets.indexes)):
            rows = index.candidates_for(
                s1.name_of(qi), s1.addr_of(qi), max_candidates=per_store_cap,
            )
            for r in rows:
                tid = store.id_of(int(r))
                if tid in seen:
                    continue
                seen.add(tid)
                hit_ids.append(tid)
                tn, ta = _derive_target(store, int(r))
                feat_rows.append(
                    pair_features(qn, qa, q_country, tn, ta,
                                  targets.country_of(si, int(r)), stats)
                )
                feat_owner.append((local, si, int(r)))

        if not hit_ids:
            continue
        cands[qi] = hit_ids

        # Score just this row's candidates so predictions stay grouped.
        X = np.vstack(feat_rows[n_feat_before:])
        probs = model.predict_proba(X)
        order = sorted(
            range(len(hit_ids)),
            key=lambda j: (-probs[j], hit_ids[j]),
        )
        keep = [hit_ids[j] for j in order if probs[j] >= model.threshold]
        if keep:
            preds[qi] = keep

        # Only the current row's rows remain pending; drop them eagerly so
        # the chunk never accumulates every candidate it scored.
        del feat_rows[n_feat_before:]
        del feat_owner[n_feat_before:]

    return preds, cands


# --------------------------------------------------------------------------
# training on the labelled split
# --------------------------------------------------------------------------

def train_on_train_split(
    train_dir: Path,
    cfg: PipelineConfig,
    model: MatchingModel,
    limit: int | None = None,
) -> tuple[dict, MatchingModel]:
    """Learn from ``dataset/train``, then release everything it allocated.

    Returns ``(report, model)``.  Rows are split deterministically: the first
    80% train, the last 20% are held out purely for threshold calibration so
    the chosen threshold never sees data the model learned from.
    """
    t0 = time.time()
    s1_path = train_dir / "train_source1.tsv"
    s1 = load_store(s1_path, limit=limit)

    # Cap both passes.  Fitting a 37-feature linear model saturates well
    # before every Source 1 row has been seen, and cost is linear in rows --
    # so `train_rows`/`calib_rows` are what keep a full run to ~2 hours
    # instead of ~6 on a 7.6 GB machine.  Calibration always starts *after*
    # the fit range, so the threshold never sees data the model learned on.
    fit_cap = cfg.train_rows if cfg.train_rows and cfg.train_rows > 0 else 10 ** 18
    split_at = min(int(s1.n * 0.8), fit_cap)
    calib_cap = cfg.calib_rows if cfg.calib_rows and cfg.calib_rows > 0 else 10 ** 18
    calib_end = min(s1.n, split_at + calib_cap)
    rng = random.Random(cfg.random_state)

    # Compute the row bounds *before* reading ground truth so only the ids we
    # will actually look up are retained.  An unfiltered load costs ~1.4 GB to
    # answer lookups for `calib_end` rows -- on 7.6 GB that is what pushes the
    # process out of physical memory and into paging.
    keep = {s1.id_of(qi) for qi in range(calib_end)}
    truth = load_truth(
        train_dir / "train_ground_truth.tsv", limit=limit, keep_ids=keep,
    )
    if not truth:
        return {"mode": "skipped (no ground truth)"}, model
    del keep

    targets = load_split_targets(train_dir, "train", cfg, limit=limit)
    log.info(
        "train: S1=%s targets=%s truth=%s rows (of 2.2M)",
        f"{s1.n:,}", f"{targets.n_total:,}", f"{len(truth):,}",
    )

    model.begin_streaming()
    pos_seen = neg_seen = pairs_seen = 0
    X_batch: list[np.ndarray] = []
    y_batch: list[int] = []

    def flush() -> None:
        nonlocal X_batch, y_batch
        if not y_batch:
            return
        X = np.vstack(X_batch)
        y = np.asarray(y_batch, dtype=np.int64)
        model.partial_fit(X, y)
        X_batch, y_batch = [], []

    per_store_cap = max(4, cfg.max_candidates_per_row // max(len(targets.stores), 1))
    pos_pairs: list[tuple[int, int, int]] = []
    neg_pairs: list[tuple[int, int, int]] = []

    # ---- pass A: fit on the first 80% of Source 1 ------------------------
    for qi in range(split_at):
        truth_ids = truth.get(s1.id_of(qi), set())
        if not truth_ids:
            continue
        for si, (store, index) in enumerate(zip(targets.stores, targets.indexes)):
            rows = index.candidates_for(
                s1.name_of(qi), s1.addr_of(qi), max_candidates=per_store_cap,
            )
            for r in rows:
                r = int(r)
                tid = store.id_of(r)
                (pos_pairs if tid in truth_ids else neg_pairs).append((qi, si, r))

        # Bound the working set: flush whenever the buffers get large.
        if len(pos_pairs) + len(neg_pairs) >= 40_000:
            _flush_training_batch(
                model, s1, targets, pos_pairs, neg_pairs, cfg, rng,
                X_batch, y_batch,
            )
            pos_seen += len(pos_pairs)
            neg_seen += len(neg_pairs)
            pos_pairs, neg_pairs = [], []
            flush()

    if pos_pairs or neg_pairs:
        _flush_training_batch(
            model, s1, targets, pos_pairs, neg_pairs, cfg, rng, X_batch, y_batch,
        )
        pos_seen += len(pos_pairs)
        neg_seen += len(neg_pairs)
        pos_pairs, neg_pairs = [], []
    flush()

    # Persist the fitted weights before calibration.  Pass B is the phase that
    # has already been lost once to a killed run, and re-fitting costs ~20
    # minutes -- if calibration later raises, run_test picks this up instead
    # of discarding Pass A and dropping to the heuristic.
    try:
        import tempfile as _tf
        ckpt = Path(_tf.gettempdir()) / "ber_fit_checkpoint.json"
        model.save(ckpt)
        log.info("Fitted model checkpointed to %s", ckpt)
    except Exception as exc:                       # noqa: BLE001
        log.warning("Could not checkpoint fitted model: %s", exc)

    report = {
        "mode": model.kind,
        "n_train_rows": split_at,
        "n_positive_pairs": pos_seen,
        "n_negative_pairs": neg_seen,
        "n_total_pairs": pairs_seen,
    }
    log.info(
        "Pass A done: %s positive / %s negative pairs",
        f"{pos_seen:,}", f"{neg_seen:,}",
    )

    # ---- pass B: calibrate the threshold on the held-out 20% -------------
    if cfg.calibrate_threshold and calib_end > split_at:
        val_scores: dict[str, list[tuple[str, float]]] = {}
        y_true_val: dict[str, set[str]] = {}
        for qi in range(split_at, calib_end):
            eid = s1.id_of(qi)
            y_true_val[eid] = truth.get(eid, set())
            qn, qa = _derive_query(s1, qi)
            q_country = country_string(s1, qi)
            scored: list[tuple[str, float]] = []
            for si, (store, index) in enumerate(zip(targets.stores, targets.indexes)):
                rows = index.candidates_for(
                    s1.name_of(qi), s1.addr_of(qi), max_candidates=per_store_cap,
                )
                if rows.size == 0:
                    continue
                # Derive each target exactly once.  The previous form called
                # `_derive_target(...)` twice inside the comprehension --
                # ~39 us wasted on every pair, which dominated this pass.
                xs: list[list[float]] = []
                for r in rows:
                    r = int(r)
                    tn, ta = _derive_target(store, r)
                    xs.append(
                        pair_features(qn, qa, q_country, tn, ta,
                                      targets.country_of(si, r), None)
                    )
                probs = model.predict_proba(np.asarray(xs, dtype=np.float32))
                scored.extend(
                    (store.id_of(int(r)), float(p))
                    for r, p in zip(rows, probs)
                )
            # De-duplicate across the two sources, keep the best score.
            best: dict[str, float] = {}
            for cid, p in scored:
                if cid not in best or p > best[cid]:
                    best[cid] = p
            val_scores[eid] = sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))

        report["n_val_rows"] = calib_end - split_at
        report["calibration"] = calibrate(val_scores, y_true_val, cfg)
        model.threshold = float(report["calibration"]["threshold"])
        report["val_macro_f05"] = report["calibration"]["val_f05"]
        log.info(
            "Calibrated threshold=%.2f (val macro F_0.5=%s)",
            model.threshold, report["val_macro_f05"],
        )
    else:
        model.threshold = model.cfg.heuristic_threshold
        report["threshold"] = model.threshold

    # ---- release everything ----------------------------------------------
    report["seconds"] = round(time.time() - t0, 2)
    return report, model


def _flush_training_batch(
    model: MatchingModel,
    s1: RecordStore,
    targets: TargetSide,
    pos_pairs: list[tuple[int, int, int]],
    neg_pairs: list[tuple[int, int, int]],
    cfg: PipelineConfig,
    rng: random.Random,
    X_batch: list[np.ndarray],
    y_batch: list[int],
) -> None:
    """Vectorise buffered pairs and append them to the pending mini-batch."""
    if pos_pairs:
        n_neg = min(len(neg_pairs), int(len(pos_pairs) * cfg.negative_ratio))
        negs = rng.sample(neg_pairs, n_neg) if n_neg else []
    else:
        negs = []
    # Deriving a row's query name/address costs ~40 us; a row contributes
    # ~50 candidates, so caching it per row removes ~40 us x 49 from every
    # row's worth of work.
    qcache: dict[int, tuple] = {}
    for label, group in ((1, pos_pairs), (0, negs)):
        for qi, si, r in group:
            cached = qcache.get(qi)
            if cached is None:
                cached = _derive_query(s1, qi), country_string(s1, qi)
                qcache[qi] = cached
            (qn, qa), q_country = cached
            tn, ta = _derive_target(targets.stores[si], r)
            X_batch.append(
                pair_features(qn, qa, q_country, tn, ta,
                              targets.country_of(si, r), None)
            )
            y_batch.append(label)


def calibrate(
    scores: dict[str, list[tuple[str, float]]],
    truth: dict[str, set[str]],
    cfg: PipelineConfig,
) -> dict:
    """Sweep thresholds; return the macro-F_0.5-optimal one."""
    if not scores or not truth:
        return {"threshold": cfg.threshold, "val_f05": None, "curve": []}
    best_thr, best_score = cfg.threshold, -1.0
    curve = []
    for thr in cfg.threshold_grid:
        score, _ = f05_at_threshold(scores, truth, float(thr))
        curve.append({"threshold": float(thr), "f05": round(score, 5)})
        if score > best_score + 1e-12:
            best_score, best_thr = score, float(thr)
    return {"threshold": best_thr, "val_f05": round(best_score, 5), "curve": curve}


# --------------------------------------------------------------------------
# mode: train-eval
# --------------------------------------------------------------------------

def run_train_eval(
    data_dir: Path,
    output_dir: Path,
    cfg: PipelineConfig,
    *,
    limit: int | None = None,
) -> dict:
    """Report macro F_0.5 on a held-out slice of the training split."""
    train_dir = data_dir / "dataset" / "train"
    model = MatchingModel(cfg)
    report, model = train_on_train_split(train_dir, cfg, model, limit)
    report["mode"] = "train-eval"
    report["threshold"] = model.threshold
    report["model_kind"] = model.kind
    report.setdefault("val_macro_f05", None)
    log.info("train-eval | val macro F_0.5 = %s", report.get("val_macro_f05"))
    return report


# --------------------------------------------------------------------------
# mode: predict
# --------------------------------------------------------------------------

def run_predict(
    data_dir: Path,
    output_dir: Path,
    cfg: PipelineConfig,
    *,
    limit: int | None = None,
) -> dict:
    """Inference only, reusing a saved model file."""
    t0 = time.time()
    model_path = cfg.extra.get("model_path")
    if model_path and Path(model_path).exists():
        model = MatchingModel.load(model_path, cfg)
        log.info(
            "Loaded model from %s (kind=%s, thr=%.2f)",
            model_path, model.kind, model.threshold,
        )
    else:
        model = MatchingModel(cfg)
        model._init_heuristic()
        log.warning("No saved model found; using heuristic scorer.")

    test_dir = data_dir / "dataset" / "test"
    s1 = load_store(test_dir / "test_source1.tsv", limit=limit)
    targets = load_split_targets(test_dir, "test", cfg, limit=limit)

    output_dir.mkdir(parents=True, exist_ok=True)
    match_path = output_dir / "matching_results.tsv"
    cand_path = output_dir / "candidate_pairs.tsv"
    with match_path.open("w", encoding="utf-8", newline="") as mf, \
         cand_path.open("w", encoding="utf-8", newline="") as cf:
        mf.write("\t".join(MATCHING_HEADER) + "\n")
        cf.write("\t".join(CANDIDATE_HEADER) + "\n")
        for start, end in iter_s1_chunks(s1.n, cfg.query_chunk):
            preds, cands = score_chunk(model, s1, targets, start, end, None, cfg)
            for k in range(start, end):
                eid = s1.id_of(k)
                mf.write(f"{eid}\t{','.join(preds.get(k, []))}\n")
                cf.write(f"{eid}\t{','.join(cands.get(k, []))}\n")

    return {
        "mode": "predict",
        "model_kind": model.kind,
        "threshold": model.threshold,
        "n_source1": s1.n,
        "seconds": round(time.time() - t0, 2),
    }
