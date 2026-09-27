# Business Entity Resolution — Amazon ML Challenge 2026

Deterministic, local-only pipeline that resolves business entities across
three independent sources and emits the two submission files required by the
challenge.

Given `Source 1` (the deduplicated reference source), the pipeline finds all
matching records from `Source 2` and `Source 3`. A Source 1 entity may match
zero, one, or many records — zero-match entities are written with an empty
list (singletons), which is worth a full `1.0` on that row when predicted
correctly.

---

## 1. Requirements

| | |
|---|---|
| Python | 3.10 or newer |
| Dependencies | see `requirements.txt` (numpy, scipy, scikit-learn, rapidfuzz) |
| Network | **none** — the pipeline performs no external lookups |
| Disk | challenge TSVs (~2.5 GB, read only) plus ~1.5 GB of scratch in the system temp directory for the key index and per-chunk output staged for workers; both are cleaned up on exit |

```bash
pip install -r code/business_entity_resolution/requirements.txt
```

---

## 2. Expected data layout

Run from the directory that contains `dataset/` (the `student_resource/`
folder shipped with the challenge):

```
student_resource/
├── dataset/
│   ├── train/
│   │   ├── train_source1.tsv
│   │   ├── train_source2.tsv
│   │   ├── train_source3.tsv
│   │   └── train_ground_truth.tsv
│   └── test/
│       ├── test_source1.tsv
│       ├── test_source2.tsv
│       └── test_source3.tsv
├── utils/
│   └── validate_submission.py
└── output/                        # created by this pipeline
```

All inputs are tab-separated. Source files carry the columns
`entity_id`, `business_name`, `business_address`, `country`; the ground-truth
file carries `source1_entity_id`, `matched_entity_ids`.

> **Always read with `sep="\t"`.** Omitting it silently produces a single
> column containing the whole line. The loader here enforces this and raises
> a clear error rather than failing silently.

---

## 3. Reproduce end-to-end

### 3.1 Generate the submission files

```bash
python code/business_entity_resolution/src/run_pipeline.py \
  --data-dir . \
  --mode test \
  --output-dir output
```

This writes:

* `output/matching_results.tsv` — the leaderboard file
* `output/candidate_pairs.tsv` — the blocking output fed to the model

### 3.2 Validate before submitting

Using the official challenge validator:

```bash
python utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

Or the bundled equivalent (same rules, ships with this code):

```bash
python code/business_entity_resolution/src/run_pipeline.py \
  --mode validate \
  --data-dir . \
  --output-dir output
```

Both print `PASS` and exit `0` when the files are safe to submit, or a
numbered list of issues and exit `1`.

### 3.3 Measure your own score

There is no ground truth for the test set. Hold out a slice of training data
and score it yourself with the challenge's macro `F_0.5`:

```bash
python code/business_entity_resolution/src/run_pipeline.py \
  --data-dir . \
  --mode train-eval \
  --output-dir output
```

This reports `val_macro_f05` and writes `output/val_matching_results.tsv` for
error inspection. **It does not overwrite the two submission files.**

### 3.4 Inference with a previously fitted model

```bash
python code/business_entity_resolution/src/run_pipeline.py \
  --data-dir . \
  --mode predict \
  --model-path output/matching_model.json \
  --output-dir output
```

---

## 4. Pipeline architecture

```
 TSV inputs
    │
    ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 1. Normalisation  (src/normalize.py)                     │
 │    Unicode NFKD accent folding · lowercase · punctuation │
 │    & → "and" · abbreviation expansion (Corp→Corporation, │
 │    Rd→Road, Pvt→Private, CA→California…) · legal-suffix  │
 │    stripping · word-order sorting · postcode/phone/digit │
 │    extraction · country kept as an open string set       │
 └──────────────────────────────────────────────────────────┘
    │
    ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 2. Candidate generation / blocking (src/blocking.py)     │
 │    Packed-key index: crc32(key) << 32 | row_index,       │
 │    sorted into one uint64 array; lookup = 2 × searchsort │
 │    Keys — exact: sorted core name, first|last token,     │
 │               initialism, longest token, name prefix,    │
 │               postcode, house-number|street              │
 │           fuzzy: sampled char 4-grams of sorted name     │
 │    Keys shared by > df_cap rows are skipped (fan-out     │
 │    bound; drops "street"-style tokens on its own)        │
 │    → written to output/candidate_pairs.tsv               │
 └──────────────────────────────────────────────────────────┘
    │
    ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 3. Feature engineering (src/features.py)                 │
 │    37 features: RapidFuzz ratio/token-sort/token-set/    │
 │    partial over name + ratio/Jaccard over address,       │
 │    token Jaccard, length ratios, initialism match,       │
 │    suffix-only difference, postcode/phone/digit          │
 │    agreement, country match, rare-token share, and       │
 │    interaction terms.  Cheap address scorers only —      │
 │    WRatio measured ~60us on real addresses vs ~2us for   │
 │    ratio, and sorted_tokens makes ratio ≡ token_sort.    │
 └──────────────────────────────────────────────────────────┘
    │
    ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 4. Matching model (src/model.py)                         │
 │    Mini-batch logistic regression (SGDClassifier, class  │
 │    balanced) when labels are available; calibrated       │
 │    weighted heuristic otherwise.  Learned from sampled   │
 │    positives/negatives streamed in batches, so the       │
 │    7.6M ground-truth links are never all held in RAM.    │
 │    Decision threshold selected by sweeping for the       │
 │    highest macro F_0.5 on a held-out train slice.        │
 └──────────────────────────────────────────────────────────┘
    │
    ▼
 ┌──────────────────────────────────────────────────────────┐
 │ 5. Output assembly (src/pipeline.py)                     │
 │    One row per Source 1 entity (input order) · empty     │
 │    list for singletons · matches ⊆ candidates · S2/S3    │
 │    IDs only · no duplicates in any list →                │
 │    output/matching_results.tsv                           │
 └──────────────────────────────────────────────────────────┘
```

### Why precision-heavy thresholding

The metric is macro `F_0.5`, which weights precision **2×** over recall:

```
F_0.5 = (1.25 × P × R) / (0.25 × P + R)
```

Two consequences shape the design:

* **False merges are punished harder than missed links.** The threshold is
  never chosen for raw F1 — it is swept specifically for `F_0.5`.
* **Singletons are worth a full 1.0.** Predicting an empty list for an entity
  with no true matches scores `1.0`; predicting any match scores `0.0`.
  Correctly identifying singletons therefore earns credit, so the pipeline
  keeps a conservative floor rather than emitting hopeful matches.

### Handling France (open-set country)

Training covers `US` and `India`; the test set adds `France`. Country is
treated as an **open set of string labels**: it is never hard-coded, filtered,
or one-hot encoded against a fixed vocabulary. It enters the model only as a
binary `country_match` feature (do the two records agree?), which transfers
trivially to unseen countries. Every test entity — France included — is
emitted, because rows are driven by `test_source1.tsv` rather than by any
country filter.

---

## 5. Key tunables

| Flag | Default | Effect |
|---|---|---|
| `--df-cap` | 60 | Skip a blocking key shared by more than this many rows (bounds fan-out) |
| `--max-candidates` | 80 | Hard cap on candidates per Source 1 row (↑ recall, ↓ precision) |
| `--workers` | all cores | Scoring processes; they memory-map one shared copy of the data |
| `--train-rows` | 300,000 | Source 1 rows used to fit the model (cost is linear in rows) |
| `--calib-rows` | 50,000 | Held-out rows used to pick the threshold |
| `--query-chunk` | 5000 | Source 1 rows held in flight per scoring chunk |
| `--threshold` | calibrated | Decision threshold; setting it disables calibration |
| `--no-calibrate` | off | Skip the `F_0.5` threshold sweep |
| `--limit` | off | Read only N rows per input file (smoke tests) |
| `--seed` | 42 | Reproducibility seed |

**Blocking determines the recall ceiling.** If your validation `F_0.5` is
capped, raise `--max-candidates` first — no classifier can recover a pair that
never reached `candidate_pairs.tsv`. `--df-cap` is the second lever: a higher
value lets noisier keys through and widens the candidate set.

---

## 6. Scale, determinism & performance

The real split is far larger than a laptop pipeline can hold as Python
objects, so the implementation is built around memory rather than around
convenience:

| | train | test |
|---|---:|---:|
| Source 1 | 2,206,821 | 1,732,544 |
| Source 2 | 5,034,616 | 4,887,273 |
| Source 3 | 5,285,603 | 5,082,316 |
| ground-truth links | 7,638,365 | — |

* **Column blobs, not objects.** `store.py` holds each field as one contiguous
  byte buffer plus `int32` offsets. A 9.97M-row target set costs ~1 GB of blob
  instead of 6–8 GB of `str` objects. `country` is dict-encoded to `uint8`
  (four labels × 11.7M rows ≈ 11 MB instead of ~700 MB of Python strings).
* **Sorted packed keys, not a TF-IDF matrix.** The obvious approach — a sparse
  cosine similarity between 1.73M and 4.89M rows — needs ~39 GB *per chunk* of
  dense float32, on a machine with 7.6 GB. Packed keys need 8 bytes per
  posting, and candidate lookup is two `searchsorted` calls.
* **`crc32`, not `hash()`.** `hash()` is randomised per process, so blocking
  would differ between runs. `zlib.crc32` is stable. A collision can only *add*
  a candidate (it unions two key groups), never remove one — so recall is
  unaffected and the classifier filters the extras.
* **Spilled to disk, shared by workers.** Windows uses `spawn`, so workers
  cannot inherit memory. The target blobs and key index are written once and
  memory-mapped by every worker: N processes cost one copy of the data through
  the page cache, not N.
* **Streams Source 1 and writes incrementally**, so exactly one row per entity
  is produced without ever holding the result set. Per-chunk output files are
  stitched back together in Source 1 order.
* Everything is seeded (`--seed`, default 42), and worker count does not change
  the result — outputs were verified byte-identical at `--workers 1` and
  `--workers 4`.

Measured on the full test split (8 workers, i5-12450H): blocking index build
is a few minutes; scoring runs at roughly 52 candidates/row.

---

## 7. Fair-play compliance

The challenge **strictly prohibits** external data lookup — no commercial ER
APIs, no government registration lookups, no geocoding APIs, no internet
augmentation of any kind. This pipeline performs **only local string and
matrix computation** on the supplied files. It opens no network connections,
resolves nothing against outside data, and enriches records with nothing
beyond what is already in the TSVs.

All dependencies are permissively licensed (BSD-3-Clause / MIT), and the model
itself is a linear logistic regression with 38 parameters (37 coefficients plus
the intercept) — orders of magnitude below the 8B-parameter ceiling.

---

## 8. Repository layout

```
code/business_entity_resolution/
├── src/
│   ├── __init__.py
│   ├── config.py               # all tunables + abbreviation vocabulary
│   ├── normalize.py            # name/address normalisation, blocking keys
│   ├── store.py                # column-blob storage, mmap spill/reload
│   ├── blocking.py             # packed sorted-key index, DF-capped lookup
│   ├── features.py             # 37 pairwise features, token IDF stats
│   ├── model.py                # scorer, F_0.5 metrics, threshold sweep
│   ├── pipeline.py             # orchestration, worker pool, output writers
│   ├── validate_submission.py  # local reimplementation of the challenge rules
│   └── run_pipeline.py         # CLI entry point
├── README.md
└── requirements.txt
```

# Team Codex:
Pranav Shende
Swyam Gupta
