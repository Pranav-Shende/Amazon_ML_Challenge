"""Local submission validator.

Mirrors the rules of the challenge's ``utils/validate_submission.py`` so you
can catch a rejection locally instead of spending a submission on it:

  1. Output format exactly as specified (TSV, exact column names).
  2. ``matched_entity_ids`` / ``candidate_entity_ids`` reference only
     Source 2/Source 3 entities that exist in the test set.  No self-matches
     to Source 1, no phantom IDs.
  3. Every Source 1 entity in ``test_source1.tsv`` appears exactly once.
  4. No duplicate entity IDs within any ID list, and no duplicate
     ``source1_entity_id`` rows.
  5. ``matching_results`` must be a subset of ``candidate_pairs`` (a matched
     ID that was never a candidate signals a pipeline bug).

Run it from the ``student_resource/`` directory:

    python utils/validate_submission.py \\
        --matching output/matching_results.tsv \\
        --candidate output/candidate_pairs.tsv \\
        --test-dir dataset/test

Prints ``PASS`` and exits 0 when safe to submit, otherwise a numbered list
of issues and exits 1.  stdlib only -- no third-party dependencies.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

MATCHING_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]


def _read_tsv(path: Path) -> tuple[list[str], list[list[str]]]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    # Plain utf-8, matching the official validator -- not utf-8-sig. Our own
    # writer never emits a BOM, and silently swallowing one would hide a
    # real problem in files we did not write.
    with path.open("r", encoding="utf-8", newline="") as fh:
        rows = [row for row in csv.reader(fh, delimiter="\t")]
    if not rows:
        return [], []
    return rows[0], rows[1:]


def _norm_header(header: list[str]) -> list[str]:
    """Official comparison is case- and whitespace-insensitive."""
    return [h.strip().lower() for h in header]


def _load_pairs(path: Path, expected_cols: list[str]) -> tuple[list[tuple[str, list[str]]], list[str]]:
    """Return (rows, issues). Normalises to (id, [ids]) with header checked."""
    issues: list[str] = []
    header, raw = _read_tsv(path)

    if _norm_header(header) != _norm_header(expected_cols):
        if len(header) == 1 and "\t" in (header[0] if header else ""):
            issues.append(
                f"{path.name}: header looks unsplit -- read with sep='\\t'."
            )
        else:
            issues.append(
                f"{path.name}: header must be exactly {expected_cols}, got {header}"
            )

    pairs: list[tuple[str, list[str]]] = []
    for i, row in enumerate(raw, start=2):     # 1-based, +1 for header
        if not row or all(not c.strip() for c in row):
            continue
        if len(row) == 1:
            # An empty second column may come back as a single-field row
            # depending on the writer; treat as singleton.
            row = [row[0], ""]
        if len(row) > 2:
            issues.append(f"{path.name} line {i}: expected 2 columns, got {len(row)}")
            continue
        s1_id = row[0].strip()
        rest = row[1].strip()
        ids = [t.strip() for t in rest.split(",") if t.strip()]
        pairs.append((s1_id, ids))
    return pairs, issues


def validate(
    matching_path: Path,
    candidate_path: Path,
    test_dir: Path,
    *,
    check_ids: bool = False,
) -> tuple[bool, list[str]]:
    """Run every rule. Returns ``(passed, issues)``.

    Entries prefixed ``WARNING:`` do not fail the run -- the official
    validator treats them the same way. ``check_ids`` gates the ID-existence
    rule, which needs both 500 MB target files resident as sets; that costs
    several GB for the full test set, so it is opt-in exactly as in the
    official tool.
    """
    issues: list[str] = []

    # ---- ground truth: test entity inventory ----------------------------
    s1_path = test_dir / "test_source1.tsv"
    s2_path = test_dir / "test_source2.tsv"
    s3_path = test_dir / "test_source3.tsv"
    if not s1_path.exists():
        issues.append(f"Missing test file: {s1_path}")
        return False, issues

    _, s1_rows = _read_tsv(s1_path)
    s1_ids = [r[0] for r in s1_rows if r and r[0]]
    s1_set = set(s1_ids)

    valid_targets: set[str] = set()
    if check_ids:
        for p in (s2_path, s3_path):
            if not p.exists():
                issues.append(f"Missing test file: {p}")
                return False, issues
            _, rows = _read_tsv(p)
            valid_targets.update(r[0] for r in rows if r and r[0])
    else:
        # Cost note mirrors the official tool's.
        issues.append(
            "WARNING: ID-existence check is OFF (the default) -- not checking "
            "that matched/candidate IDs exist in the test set. Every other "
            "rule is still checked. Re-run with --check-ids to enable it."
        )

    # ---- parse outputs ---------------------------------------------------
    match_pairs, match_hdr_issues = _load_pairs(matching_path, MATCHING_COLUMNS)
    issues.extend(match_hdr_issues)

    # The official validator treats candidate_pairs.tsv as optional: a missing
    # file is a warning, not an error.
    cand_pairs: list[tuple[str, list[str]]] = []
    cand_present = candidate_path.exists()
    if not cand_present:
        issues.append(
            "WARNING: candidate_pairs.tsv not found -- treated as a warning, "
            "as in the official rules."
        )
    else:
        cand_pairs, cand_hdr_issues = _load_pairs(candidate_path, CANDIDATE_COLUMNS)
        issues.extend(cand_hdr_issues)

    if match_hdr_issues or (cand_present and cand_hdr_issues):
        # Header problems make everything else unreliable; report early.
        return False, issues

    # ---- rule 3: every Source 1 entity present exactly once -------------
    seen_rows: dict[str, int] = {}
    for s1_id, _ in match_pairs:
        seen_rows[s1_id] = seen_rows.get(s1_id, 0) + 1
    dup_rows = sorted(k for k, v in seen_rows.items() if v > 1)
    if dup_rows:
        issues.append(
            f"matching_results: duplicate source1_entity_id rows "
            f"({len(dup_rows)}): {dup_rows[:5]}"
        )

    missing = [e for e in s1_ids if e not in seen_rows]
    if missing:
        issues.append(
            f"matching_results: {len(missing)} Source 1 entities missing "
            f"(e.g. {missing[:5]})"
        )
    extra = [e for e in seen_rows if e not in s1_set]
    if extra:
        issues.append(
            f"matching_results: {len(extra)} source1_entity_id values not in "
            f"the test set (e.g. {extra[:5]})"
        )

    # ---- rule 4: candidate row coverage + duplicates ---------------------
    cand_seen: dict[str, int] = {}
    for s1_id, _ in cand_pairs:
        cand_seen[s1_id] = cand_seen.get(s1_id, 0) + 1
    cand_dups = sorted(k for k, v in cand_seen.items() if v > 1)
    if cand_dups:
        issues.append(
            f"candidate_pairs: duplicate source1_entity_id rows "
            f"({len(cand_dups)}): {cand_dups[:5]}"
        )
    if cand_present:
        cand_missing = [e for e in s1_ids if e not in cand_seen]
        if cand_missing:
            issues.append(
                f"candidate_pairs: {len(cand_missing)} Source 1 entities missing "
                f"(e.g. {cand_missing[:5]})"
            )

    # ---- rule 2: ID validity + no duplicates within lists ---------------
    for label, pairs in (
        ("matching_results", match_pairs),
        ("candidate_pairs", cand_pairs),
    ):
        bad_prefix: list[str] = []
        unknown: list[str] = []
        self_ref: list[str] = []
        for s1_id, ids in pairs:
            if len(ids) != len(set(ids)):
                issues.append(
                    f"{label}: duplicate IDs within list for {s1_id}"
                )
            for cid in ids:
                if cid.startswith("S1-"):
                    self_ref.append(f"{s1_id}->{cid}")
                elif not (cid.startswith("S2-") or cid.startswith("S3-")):
                    bad_prefix.append(f"{s1_id}->{cid}")
                elif check_ids and cid not in valid_targets:
                    unknown.append(f"{s1_id}->{cid}")
        if self_ref:
            issues.append(
                f"{label}: {len(self_ref)} self-matches to Source 1 "
                f"(e.g. {self_ref[:3]})"
            )
        if bad_prefix:
            issues.append(
                f"{label}: {len(bad_prefix)} IDs not prefixed S2-/S3- "
                f"(e.g. {bad_prefix[:3]})"
            )
        if unknown:
            issues.append(
                f"{label}: {len(unknown)} IDs not present in the test set "
                f"(e.g. {unknown[:3]})"
            )

    # ---- rule 5: matches must be a subset of candidates -------------------
    # Officially a warning: a matched ID that never appeared as a candidate
    # only lowers your score, it never rejects the submission.
    if cand_present:
        cand_map = dict(cand_pairs)
        not_cand: list[str] = []
        for s1_id, ids in match_pairs:
            cset = set(cand_map.get(s1_id, []))
            for cid in ids:
                if cid not in cset:
                    not_cand.append(f"{s1_id}->{cid}")
        if not_cand:
            issues.append(
                f"WARNING: matching_results: {len(not_cand)} matched IDs never "
                f"appeared in candidate_pairs (e.g. {not_cand[:3]}) -- this "
                f"lowers your score but does not reject the submission."
            )

    return (not any(not i.startswith("WARNING:") for i in issues)), issues


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Validate challenge submission files.")
    ap.add_argument("--matching", required=True, type=Path)
    ap.add_argument("--candidate", type=Path, default=Path("output/candidate_pairs.tsv"),
                    help="Optional file; absence is a warning, not an error.")
    ap.add_argument("--test-dir", required=True, type=Path)
    ap.add_argument("--check-ids", action="store_true",
                    help="Also verify every referenced ID exists in the test "
                         "set. Needs both target files resident as sets -- "
                         "several GB for the full test split.")
    args = ap.parse_args(argv)

    try:
        passed, issues = validate(
            args.matching, args.candidate, args.test_dir,
            check_ids=args.check_ids,
        )
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    warnings = [m for m in issues if m.startswith("WARNING:")]
    errors = [m for m in issues if not m.startswith("WARNING:")]

    for msg in warnings:
        print(f"  {msg}")
    if passed:
        print("PASS")
        return 0

    print(f"FAIL - {len(errors)} issue(s) to fix:")
    for i, msg in enumerate(errors, start=1):
        print(f"  {i}. {msg}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
