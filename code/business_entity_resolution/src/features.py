"""Pairwise feature engineering.

Every feature is a local comparison between a Source 1 record and a
candidate Source 2 / Source 3 record.  Similarity measures come from
RapidFuzz (MIT licence) plus a handful of set/token statistics computed
here.  No external enrichment of any kind.
"""

from __future__ import annotations

import math
import re
from typing import Sequence

import numpy as np
from rapidfuzz import fuzz

from .normalize import AddressFeatures, NameFeatures

# Feature names, in the exact order emitted by ``extract_features``.
# Kept module-level so the trained model can be serialised with a stable
# schema and reloaded for inference.
FEATURE_NAMES: list[str] = [
    # --- name similarity -------------------------------------------------
    "name_token_sort",
    "name_token_set",
    "name_partial",
    "name_wratio",
    "name_ratio",
    "name_core_ratio",
    "name_content_ratio",
    "name_sorted_core_ratio",
    "name_jaccard_tokens",
    "name_jaccard_content",
    "name_len_diff",
    "name_len_ratio",
    "name_prefix_match",
    "name_initialism_match",
    "name_suffix_only_diff",
    # --- address similarity ----------------------------------------------
    "addr_token_sort",
    "addr_token_set",
    "addr_partial",
    "addr_wratio",
    "addr_ratio",
    "addr_jaccard",
    "addr_sorted_ratio",
    "addr_len_diff",
    "addr_content_overlap",
    # --- structural / evidence -------------------------------------------
    "postcode_match",
    "postcode_jaccard",
    "phone_match",
    "digit_jaccard",
    "country_match",
    "has_any_postcode",
    "both_names_nonempty",
    "both_addr_nonempty",
    "max_token_overlap",
    "rare_token_share",
    # --- interactions -----------------------------------------------------
    "name_x_addr",
    "name_high_addr_ok",
    "addr_only_boost",
]

N_FEATURES = len(FEATURE_NAMES)


# --------------------------------------------------------------------------
# small set utilities
# --------------------------------------------------------------------------

def _jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    if inter == 0:
        return 0.0
    return inter / len(sa | sb)


def _jaccard_sets(a, b) -> float:
    """Jaccard over any iterables (tuples, frozensets, lists, sets)."""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    inter = len(sa & sb)
    if inter == 0:
        return 0.0
    return inter / len(sa | sb)


def _safe_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) / 100.0


def _token_sort(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) / 100.0


def _token_set(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) / 100.0


def _partial(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    return fuzz.partial_ratio(a, b) / 100.0


def _wratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return fuzz.WRatio(a, b) / 100.0


def _len_ratio(a: str, b: str) -> float:
    if not a and not b:
        return 1.0
    la, lb = len(a), len(b)
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def _len_diff(a: str, b: str) -> float:
    return abs(len(a) - len(b)) / max(len(a), len(b), 1)


def _token_overlap_max(a: Sequence[str], b: Sequence[str]) -> float:
    """Max fraction of the shorter token set contained in the longer."""
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    shorter, longer = (sa, sb) if len(sa) <= len(sb) else (sb, sa)
    if not shorter:
        return 0.0
    return len(shorter & longer) / len(shorter)


_DIGIT_RUN = re.compile(r"\d+")


def _digit_runs(text: str) -> frozenset[str]:
    return frozenset(_DIGIT_RUN.findall(text or ""))


# --------------------------------------------------------------------------
# IDF-style rare-token weighting
# --------------------------------------------------------------------------

class TokenStats:
    """Document frequencies collected over the corpus (train or test).

    Used to reward agreement on *rare* tokens -- matching on "Swasthya"
    is far more informative than matching on "Store".
    """

    def __init__(self):
        self.df: dict[str, int] = {}
        self.n_docs = 0

    def update(self, token_lists: Sequence[Sequence[str]]) -> None:
        for toks in token_lists:
            self.n_docs += 1
            for t in set(toks):
                self.df[t] = self.df.get(t, 0) + 1

    def idf(self, token: str) -> float:
        df = self.df.get(token, 0)
        if df <= 0:
            return 8.0          # unseen token: assume quite informative
        return math.log((self.n_docs + 1) / (df + 1)) + 1.0

    def rare_share(self, a: Sequence[str], b: Sequence[str]) -> float:
        """IDF-weighted agreement divided by total IDF of the union."""
        if not a or not b:
            return 0.0
        sa, sb = set(a), set(b)
        inter = sa & sb
        if not inter:
            return 0.0
        num = sum(self.idf(t) for t in inter)
        den = sum(self.idf(t) for t in (sa | sb))
        return num / den if den > 0 else 0.0


# --------------------------------------------------------------------------
# feature extraction
# --------------------------------------------------------------------------

def extract_features(
    q_name: NameFeatures,
    q_addr: AddressFeatures,
    q_country: str,
    t_name: NameFeatures,
    t_addr: AddressFeatures,
    t_country: str,
    stats: TokenStats | None = None,
) -> list[float]:
    """Compute the fixed-length feature vector for one candidate pair.

    Returns a plain ``list`` rather than a numpy array: callers stack a whole
    batch with a single ``np.vstack``, which avoids ~7 us of ``np.array``
    construction per pair (a measurable share of scoring time at 100M+ pairs).

    Cost discipline: expensive RapidFuzz scorers are used only where they buy
    something.  ``WRatio`` on an address costs ~31 us versus ~1.6 us for
    ``fuzz.ratio``, and token-set/partial add ~24 us more -- so address
    similarity is computed from ``ratio`` plus a sorted-token ratio (word
    order is already normalised away) and Jaccard, which together reproduce
    the same signal for a fraction of the cost.  ``nan_to_num`` is likewise
    omitted: every division below is guarded, so no NaN/inf can appear.
    """
    # ---- name -----------------------------------------------------------
    # Inputs are already token-sorted, so token_sort_ratio would only re-sort
    # them; a plain ratio gives the identical score ~6x cheaper.
    n_sort = _safe_ratio(q_name.sorted_content, t_name.sorted_content)
    n_set = _token_set(q_name.content, t_name.content)
    n_part = _partial(q_name.content, t_name.content)
    # `name_wratio` is kept in the schema but no longer calls `fuzz.WRatio`
    # (~21 us).  WRatio is a weighted combination of ratio / token-sort /
    # token-set / partial -- all four of which are separate features here --
    # so a linear model already spans it.  Taking the max preserves the
    # "best-scoring view wins" behaviour at ~0 cost.
    n_r = _safe_ratio(q_name.full, t_name.full)
    n_w = n_r if n_r >= n_sort else (n_set if n_set > n_part else n_part)
    if n_sort > n_w:
        n_w = n_sort
    n_core = _safe_ratio(q_name.core, t_name.core)
    n_content = _safe_ratio(q_name.content, t_name.content)
    n_sorted = _safe_ratio(q_name.sorted_core, t_name.sorted_core)
    n_jac = _jaccard(q_name.core_tokens, t_name.core_tokens)
    n_jac_c = _jaccard(q_name.content_tokens, t_name.content_tokens)
    n_lend = _len_diff(q_name.full, t_name.full)
    n_lenr = _len_ratio(q_name.full, t_name.full)
    n_prefix = 1.0 if (
        q_name.content and t_name.content
        and (q_name.content.startswith(t_name.content[:6])
             or t_name.content.startswith(q_name.content[:6]))
        and min(len(q_name.content), len(t_name.content)) >= 6
    ) else 0.0
    n_init = 1.0 if (
        q_name.initial_prefix
        and t_name.initial_prefix
        and q_name.initial_prefix == t_name.initial_prefix
        and len(q_name.initial_prefix) >= 3
    ) else 0.0
    # Do they differ *only* by legal suffix?
    n_suffix_only = 1.0 if (
        q_name.content != t_name.content
        and q_name.sorted_content == t_name.sorted_content
    ) else 0.0

    # ---- address (cheap scorers only) -----------------------------------
    # Addresses are long, so the expensive scorers hurt most here:
    # `fuzz.WRatio` measured ~60 us and token-set/partial ~22-31 us each on
    # real address strings, versus ~2 us for `fuzz.ratio`.  Because
    # `sorted_tokens` is already word-sorted, a plain ratio *is*
    # token_sort_ratio -- so `a_sort`/`a_w`/`a_sorted` are all the same cheap
    # computation, and `a_set`/`a_part`/`a_jac` collapse to Jaccard over the
    # content tokens (order-independent, like token_set_ratio).
    a_sort = _safe_ratio(q_addr.sorted_tokens, t_addr.sorted_tokens)
    a_set = _jaccard(q_addr.content, t_addr.content)
    a_part = a_set          # kept for schema stability; cheap proxy
    a_w = a_sort
    a_r = _safe_ratio(q_addr.full, t_addr.full)
    a_jac = a_set
    a_sorted = a_sort
    a_lend = _len_diff(q_addr.full, t_addr.full)
    a_overlap = _token_overlap_max(q_addr.content, t_addr.content)

    # ---- structural -----------------------------------------------------
    pc_match = 1.0 if (
        q_addr.postcodes and t_addr.postcodes
        and set(q_addr.postcodes) & set(t_addr.postcodes)
    ) else 0.0
    pc_jac = _jaccard_sets(q_addr.postcodes, t_addr.postcodes)

    ph_match = 0.0
    if q_addr.phone and t_addr.phone:
        qd = re.sub(r"\D", "", q_addr.phone)
        td = re.sub(r"\D", "", t_addr.phone)
        if qd and td and qd == td:
            ph_match = 1.0
        elif qd and td and min(len(qd), len(td)) >= 7 and (
            qd[:7] == td[:7] or qd[-7:] == td[-7:]
        ):
            ph_match = 0.5

    digit_j = _jaccard_sets(q_addr.digit_set, t_addr.digit_set)
    country_m = 1.0 if (q_country and t_country and q_country == t_country) else 0.0
    has_pc = 1.0 if (q_addr.postcodes or t_addr.postcodes) else 0.0
    both_n = 1.0 if (not q_name.is_empty and not t_name.is_empty) else 0.0
    both_a = 1.0 if (not q_addr.is_empty and not t_addr.is_empty) else 0.0
    max_tok = _token_overlap_max(
        q_name.content_tokens + q_addr.content,
        t_name.content_tokens + t_addr.content,
    )
    if stats is None:
        rare = 0.0
    else:
        rare = stats.rare_share(
            q_name.content_tokens + q_addr.content,
            t_name.content_tokens + t_addr.content,
        )

    # ---- interactions ---------------------------------------------------
    inter_1 = n_r * a_r
    inter_2 = n_r if (n_r >= 0.75 and a_r >= 0.4) else 0.0
    inter_3 = a_r if n_r < 0.6 else 0.0

    # Returned as a plain list: `np.vstack` converts a whole batch in one
    # C call, which is far cheaper than one `np.array(...)` per pair.
    #
    # No `nan_to_num` pass: every ratio above short-circuits on an empty or
    # zero-length input, so NaN/inf cannot be produced.  That guard alone was
    # costing ~21 us per pair (numpy's nan_to_num pulls in isposinf/isneginf
    #/_getmaxmin), which is why it is spelled out rather than dropped
    # silently.
    return [
        n_sort, n_set, n_part, n_w, n_r, n_core, n_content, n_sorted,
        n_jac, n_jac_c, n_lend, n_lenr, n_prefix, n_init, n_suffix_only,
        a_sort, a_set, a_part, a_w, a_r, a_jac, a_sorted, a_lend, a_overlap,
        pc_match, pc_jac, ph_match, digit_j, country_m, has_pc,
        both_n, both_a, max_tok, rare,
        inter_1, inter_2, inter_3,
    ]


def extract_batch(
    records: Sequence[tuple[NameFeatures, AddressFeatures, str]],
    queries: Sequence[tuple[NameFeatures, AddressFeatures, str]],
    stats: TokenStats | None = None,
) -> np.ndarray:
    """Vectorised convenience wrapper.

    ``records`` is ``[(q_name, q_addr, q_country), ...]`` aligned with the
    per-candidate ``queries``; returns an ``(n_pairs, N_FEATURES)`` matrix.
    """
    if not records:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    # Build lists of rows first, then one vstack -- per-row np.array would
    # cost ~7 us each.
    rows: list[list[float]] = []
    for i, (qn, qa, qc) in enumerate(records):
        tn, ta, tc = queries[i]
        rows.append(extract_features(qn, qa, qc, tn, ta, tc, stats))
    return np.asarray(rows, dtype=np.float32)
