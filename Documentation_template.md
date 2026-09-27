# Business Entity Resolution --- Methodology Documentation

## 1. Problem Statement

The objective of this project is to perform **business entity
resolution** across three independent data sources.

-   **Source 1** is the reference/deduplicated source.
-   **Source 2** and **Source 3** contain business records that may
    refer to the same real-world entities.
-   For every Source 1 entity, the system predicts all matching entity
    IDs from Source 2 and Source 3.
-   A Source 1 entity may have:
    -   no matches (singleton),
    -   one match, or
    -   multiple matches.

The final prediction is written in the challenge-required
`matching_results.tsv` format. A separate `candidate_pairs.tsv` file
records the candidate matches generated during blocking.

The solution is designed to work locally without external APIs or web
lookups.

------------------------------------------------------------------------

## 2. Dataset and Data Characteristics

The pipeline expects the following directory structure:

``` text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
│
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

All source files are tab-separated (`TSV`).

The source records contain:

-   `entity_id`
-   `business_name`
-   `business_address`
-   `country`

The ground-truth file contains:

-   `source1_entity_id`
-   `matched_entity_ids`

The dataset is large, so the implementation is explicitly designed to
avoid loading the complete data as ordinary Python string objects.

The approximate dataset sizes documented by the project are:

  Dataset      Source 1    Source 2    Source 3
  --------- ----------- ----------- -----------
  Train       2,206,821   5,034,616   5,285,603
  Test        1,732,544   4,887,273   5,082,316

The training ground truth contains approximately 7.64 million matching
links.

------------------------------------------------------------------------

## 3. Overall Methodology

The solution follows a five-stage entity-resolution pipeline:

``` text
Raw TSV data
     |
     v
1. Text normalization
     |
     v
2. Candidate generation / blocking
     |
     v
3. Pairwise feature engineering
     |
     v
4. Matching model + threshold selection
     |
     v
5. Prediction assembly and validation
```

The key design principle is to avoid comparing every Source 1 record
against every Source 2/3 record. Instead, the system first generates a
small set of plausible candidates and then applies a more detailed
similarity model only to those candidates.

------------------------------------------------------------------------

# 4. Data Preprocessing and Normalization

Business names and addresses can refer to the same entity while
differing in punctuation, capitalization, abbreviations, word order, or
legal suffixes.

The normalization stage therefore creates comparable representations
before candidate generation and feature extraction.

## 4.1 Basic text cleaning

The normalizer performs operations including:

-   Unicode NFKD normalization
-   accent/diacritic folding
-   lowercasing
-   punctuation normalization
-   whitespace normalization
-   replacement of symbols such as `&` with `and`
-   tokenization

For example:

``` text
"J.P. Morgan & Co."
```

can be converted into a cleaner token representation similar to:

``` text
j p morgan and co
```

This reduces superficial differences between records.

## 4.2 Business-name abbreviation expansion

Common business-name abbreviations are expanded.

Examples include:

``` text
corp -> corporation
inc  -> incorporated
ltd  -> limited
pvt  -> private
co   -> company
intl -> international
tech -> technologies
mfg  -> manufacturing
```

Name abbreviations are kept separate from address abbreviations so that
the same token can be interpreted differently depending on its context.

## 4.3 Address normalization

Address-specific abbreviations are expanded.

Examples include:

``` text
rd   -> road
st   -> street
ave  -> avenue
blvd -> boulevard
ln   -> lane
dr   -> drive
apt  -> apartment
ste  -> suite
```

Regional/state abbreviations are also expanded inside addresses. The
configuration includes mappings for the United States, India, and
France.

## 4.4 Legal suffix removal

Common corporate/legal suffixes are removed when constructing the **core
business name**.

Examples include:

``` text
inc
corp
company
ltd
llc
llp
gmbh
plc
pvt
holdings
group
enterprises
services
solutions
```

The original cleaned representation is still retained so that suffix
differences can be used as a weak feature rather than being completely
discarded.

## 4.5 Word-order normalization

Sorted token representations are generated for business names.

This allows records such as:

``` text
ABC Retail Solutions
```

and

``` text
Solutions ABC Retail
```

to receive a strong similarity signal even though their token order
differs.

## 4.6 Structured information extraction

The normalization layer also extracts structured signals from addresses
and identifiers, including:

-   numeric tokens
-   postal codes
-   phone-number-like sequences
-   house numbers
-   initialisms

These signals are later used in both blocking and pairwise feature
engineering.

------------------------------------------------------------------------

# 5. Candidate Generation / Blocking

## 5.1 Motivation

A naive all-pairs comparison would require comparing millions of Source
1 records against millions of Source 2/3 records.

This is computationally infeasible.

The solution therefore uses **blocking** to generate only plausible
candidate pairs.

Blocking determines the maximum possible recall of the downstream model:
if a true pair is never generated as a candidate, the classifier cannot
recover it later.

## 5.2 Blocking keys

Multiple exact blocking keys are generated for each record.

### Name-based keys

The system uses keys based on:

1.  Word-order-invariant core business name
2.  First and last distinctive name token
3.  Name initialism
4.  Longest distinctive token
5.  Prefix of the sorted core name

### Address-based keys

The system also uses:

6.  Postal code
7.  House number + first street token

These complementary keys allow a match to survive different types of
variation.

For example, a business whose name has changed slightly may still be
retrieved through its postal code or address structure.

## 5.3 Stable hashing

Blocking keys are represented using CRC32.

The implementation uses:

``` text
packed_key = crc32(key) << 32 | row_index
```

The packed keys are stored in a sorted `uint64` array.

This provides a memory-efficient representation compared with
maintaining large Python dictionaries containing lists of row IDs.

CRC32 is used instead of Python's built-in `hash()` because Python
hashes can vary between processes. Stable hashing improves
reproducibility.

A CRC32 collision can only add an extra candidate; it does not remove
the true candidate, after which the matching model can reject the false
positive.

## 5.4 Frequency cap

Very common blocking keys can create an excessive number of candidate
pairs.

The pipeline therefore applies:

``` text
df_cap = 60
```

by default.

If a blocking key is shared by more than 60 records, that key is
skipped.

This prevents generic terms such as common street words from producing
huge candidate groups.

## 5.5 Candidate budget

Each Source 1 entity has a default maximum of:

``` text
max_candidates_per_row = 80
```

This limits scoring cost and prevents a small number of highly ambiguous
records from dominating computation.

------------------------------------------------------------------------

# 6. Feature Engineering

After blocking, every candidate pair is represented by a numerical
feature vector.

The implementation uses **37 pairwise features**.

These features compare the Source 1 record with a candidate Source 2/3
record.

## 6.1 Business-name similarity

Name features include:

-   RapidFuzz token-sort similarity
-   token-set similarity
-   partial similarity
-   WRatio
-   standard string ratio
-   core-name ratio
-   content-name ratio
-   sorted-core ratio
-   token Jaccard similarity
-   content-token Jaccard similarity
-   name-length difference
-   name-length ratio
-   prefix agreement
-   initialism agreement
-   suffix-only difference

These features capture both exact-like and fuzzy similarity.

## 6.2 Address similarity

Address features include:

-   fuzzy address similarity
-   token-based address similarity
-   address Jaccard similarity
-   sorted-token comparison
-   address-length statistics
-   numeric agreement
-   postal-code agreement
-   phone-number agreement
-   digit-set overlap

Addresses are particularly useful because business names can be similar
across unrelated entities, while a matching postal code or phone number
provides stronger evidence.

## 6.3 Country information

Country is represented as a simple agreement feature:

``` text
country_match = 1
```

when both records have the same country and:

``` text
country_match = 0
```

otherwise.

The implementation does not require a fixed one-hot vocabulary for
countries. This allows the approach to handle an unseen country in the
test set without filtering it out.

## 6.4 Rare-token and interaction features

The feature set also includes signals such as:

-   rare-token overlap/share
-   combinations of name and address similarities
-   interactions between structured and fuzzy signals

The objective is to allow the model to distinguish between superficially
similar businesses and strong multi-field matches.

------------------------------------------------------------------------

# 7. Matching Model

## 7.1 Supervised model

When training labels are available, the pipeline uses a **mini-batch
logistic regression classifier implemented with `SGDClassifier`**.

The classifier uses class balancing because positive entity matches are
much rarer than non-matching candidate pairs.

Training is performed in batches rather than constructing one enormous
in-memory training matrix.

## 7.2 Positive and negative examples

Positive examples come from the provided ground-truth links.

Negative examples are sampled from candidate pairs that are not labelled
as true matches.

The configured negative-to-positive sampling ratio is:

``` text
negative_ratio = 4.0
```

This means approximately four negative examples are sampled for each
positive example during model fitting.

## 7.3 Training limits

The default configuration limits training to:

``` text
train_rows = 300,000
```

Source 1 rows.

This keeps training computationally manageable while still providing a
large training sample.

A separate calibration set of:

``` text
calib_rows = 50,000
```

rows is used for threshold selection.

------------------------------------------------------------------------

# 8. Decision Threshold and Evaluation Metric

The challenge metric is **macro F0.5**.

The implementation uses:

``` text
F0.5 = (1.25 × Precision × Recall)
       / (0.25 × Precision + Recall)
```

F0.5 gives twice as much weight to precision as recall.

Therefore, the pipeline does not simply use a conventional 0.5
probability threshold or optimize F1.

## 8.1 Threshold calibration

The system sweeps candidate thresholds over a predefined grid and
selects the threshold that produces the highest macro F0.5 on the
held-out calibration data.

The configured threshold grid ranges from:

``` text
0.20 to 0.80
```

in increments of:

``` text
0.02
```

The default configuration records a threshold of approximately:

``` text
0.52
```

but calibration can replace this with the best threshold found on the
validation slice.

## 8.2 Why precision is emphasized

False matches can create incorrect entity merges.

The precision-heavy F0.5 metric therefore encourages conservative
matching.

This is especially important for singleton entities. If a Source 1
entity truly has no matches, predicting an empty list is preferable to
adding an unsupported candidate.

------------------------------------------------------------------------

# 9. Handling Singleton Entities

The output is driven by the Source 1 records.

For every Source 1 entity, the pipeline emits exactly one output row.

If no candidate passes the matching threshold, the predicted match list
is empty.

Conceptually:

``` text
Source 1 entity -> []
```

represents a singleton.

This is important because the solution must not simply output only
entities for which a match was found.

------------------------------------------------------------------------

# 10. Memory and Scalability Design

The dataset is too large to safely represent every field as normal
Python string objects.

The project therefore uses a memory-conscious storage layer.

## 10.1 Column-oriented byte storage

The `RecordStore` stores fields using:

-   contiguous byte buffers
-   integer offset arrays

instead of one Python object per string.

This significantly reduces memory overhead.

## 10.2 Country dictionary encoding

Country values are dictionary encoded into small integer codes stored as
`uint8`.

This avoids storing millions of repeated Python string objects.

## 10.3 Memory-mapped target data

During parallel scoring, workers memory-map shared target data rather
than creating an independent full copy in each process.

This is especially useful on Windows, where multiprocessing uses
`spawn`.

## 10.4 Chunked processing

Source 1 is processed in chunks.

The default query chunk is:

``` text
5,000 rows
```

Only a limited amount of data is kept in flight at one time.

## 10.5 Temporary files

Intermediate shared data and per-chunk outputs are written to temporary
storage and cleaned up after the pipeline finishes.

------------------------------------------------------------------------

# 11. Parallel Processing

The scoring stage is CPU-bound because of RapidFuzz similarity
calculations and feature construction.

The implementation therefore uses multiple processes rather than
threads.

The number of workers can be controlled with:

``` text
--workers
```

or automatically set to the available logical CPU count.

For lower-memory machines, fewer workers should be used.

For example:

``` bash
python code/business_entity_resolution/src/run_pipeline.py \
    --data-dir . \
    --mode test \
    --output-dir output \
    --workers 2
```

------------------------------------------------------------------------

# 12. Output Generation

The pipeline produces two main files.

## 12.1 `matching_results.tsv`

This is the primary prediction/submission file.

It contains one row for each Source 1 entity and the predicted matching
entity IDs.

The pipeline guarantees that:

-   every Source 1 entity is represented,
-   output IDs come from Source 2/3,
-   predicted IDs are drawn from generated candidates,
-   duplicate IDs are not emitted within a prediction list,
-   singleton entities receive an empty list.

## 12.2 `candidate_pairs.tsv`

This file contains the candidate relationships generated during
blocking.

It is useful for:

-   debugging,
-   checking whether true matches are being retrieved,
-   understanding the recall ceiling of blocking,
-   inspecting candidate volume.

------------------------------------------------------------------------

# 13. Validation

Before submission, the generated files can be validated using the
bundled validation command:

``` bash
python code/business_entity_resolution/src/run_pipeline.py \
    --mode validate \
    --data-dir . \
    --output-dir output
```

The validator checks the structural correctness of the submission.

The project also supports the challenge's official validation script
when it is available in the challenge package.

A successful validation should report:

``` text
PASS
```

------------------------------------------------------------------------

# 14. Training Evaluation

Because the test set has no public ground truth, model quality is
estimated using a held-out portion of the training data.

Run:

``` bash
python code/business_entity_resolution/src/run_pipeline.py \
    --data-dir . \
    --mode train-eval \
    --output-dir output
```

This produces:

``` text
val_macro_f05
```

and writes validation predictions to:

``` text
output/val_matching_results.tsv
```

The validation results should be used to identify:

-   false matches,
-   missed matches,
-   singleton errors,
-   blocking failures,
-   threshold problems.

**Important:** The exact validation score should be reported only after
actually running the evaluation. It should not be estimated or
fabricated from the methodology.

------------------------------------------------------------------------

# 15. Reproducibility

The pipeline uses a fixed random seed:

``` text
random_state = 42
```

Stable CRC32 hashing is used for blocking keys instead of Python's
randomized `hash()`.

The main tunable parameters are centralized in `config.py` and can also
be overridden through command-line arguments.

Important parameters include:

  Parameter                    Default
  -------------------------- ---------
  `df_cap`                          60
  `max_candidates_per_row`          80
  `query_chunk`                  5,000
  `train_rows`                 300,000
  `calib_rows`                  50,000
  `negative_ratio`                 4.0
  `random_state`                    42
  `heuristic_threshold`           0.65

------------------------------------------------------------------------

# 16. End-to-End Execution

After installing the dependencies:

``` bash
pip install -r code/business_entity_resolution/requirements.txt
```

the test pipeline can be executed with:

``` bash
python code/business_entity_resolution/src/run_pipeline.py \
    --data-dir . \
    --mode test \
    --output-dir output
```

For an initial smoke test, use a small limit:

``` bash
python code/business_entity_resolution/src/run_pipeline.py \
    --data-dir . \
    --mode test \
    --output-dir output \
    --limit 1000 \
    --workers 2
```

After verifying the small run, the full test set can be processed.

------------------------------------------------------------------------

# 17. Design Rationale

The main methodological choices are driven by the characteristics of the
problem.

### Why normalize aggressively?

Business records often differ because of formatting rather than
identity. Normalization reduces these superficial differences.

### Why use blocking?

Full pairwise comparison is computationally infeasible at the dataset
scale. Blocking reduces millions of possible comparisons to a manageable
candidate set.

### Why use several blocking keys?

No single key is reliable for every record. Multiple complementary keys
improve the probability that genuine matches are retrieved.

### Why use fuzzy features after blocking?

Blocking provides efficiency but is intentionally coarse. Fuzzy and
structured features provide the detailed evidence needed to distinguish
true matches from false candidates.

### Why use a precision-heavy threshold?

The evaluation metric is F0.5, and incorrect merges can be particularly
damaging. Threshold selection is therefore based on the actual challenge
metric rather than F1 or accuracy.

### Why use a memory-efficient architecture?

The input data contains millions of records. Conventional Python object
storage and all-pairs similarity calculations would require excessive
memory and computation.

------------------------------------------------------------------------

# 18. Limitations and Future Improvements

The current approach has several limitations.

## 18.1 Blocking recall

A true match that does not share any retained blocking key will never
reach the classifier.

Potential improvements include:

-   additional phonetic keys,
-   transliteration-aware keys,
-   character n-gram indexing,
-   adaptive blocking,
-   multiple-stage blocking.

## 18.2 Fixed candidate limits

The default candidate limit of 80 provides a computational budget but
can potentially remove a true match in highly ambiguous groups.

An adaptive candidate budget could allocate more candidates to uncertain
records.

## 18.3 Linear model capacity

Logistic regression provides an efficient and interpretable baseline,
but more expressive models could potentially capture nonlinear
interactions.

Potential alternatives include:

-   gradient-boosted trees,
-   LightGBM/XGBoost-style models,
-   pairwise neural models,
-   cross-encoder approaches for selected candidate pairs.

Any replacement should still respect the memory and runtime constraints
of the full dataset.

## 18.4 Limited external context

The pipeline intentionally performs no external enrichment.

Additional trusted business registries or geographic information could
potentially improve entity resolution, but this would introduce
dependency, availability, privacy, and reproducibility considerations.

------------------------------------------------------------------------

# 19. Summary

The final methodology is a **hybrid deterministic + supervised
entity-resolution system**:

1.  Normalize names and addresses.
2.  Generate candidates using multiple exact blocking keys.
3.  Extract 37 similarity and structured features for each candidate
    pair.
4.  Train a class-balanced mini-batch logistic regression model.
5.  Calibrate the decision threshold using macro F0.5.
6.  Score test candidates in memory-efficient chunks.
7.  Produce one prediction row per Source 1 entity.
8.  Preserve singleton predictions.
9.  Validate the generated submission files.

The architecture balances **matching quality, precision, computational
efficiency, memory usage, and reproducibility** for a
multi-million-record entity-resolution problem.
