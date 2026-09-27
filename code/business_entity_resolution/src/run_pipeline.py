#!/usr/bin/env python3
"""CLI entry point for the Business Entity Resolution pipeline.

Examples
--------
Run on the test split (produces the two submission files)::

    python code/business_entity_resolution/src/run_pipeline.py \\
        --data-dir . \\
        --mode test \\
        --output-dir output

Evaluate on a held-out slice of the training split::

    python code/business_entity_resolution/src/run_pipeline.py \\
        --data-dir . \\
        --mode train-eval \\
        --output-dir output

Validate the generated files against the challenge rules::

    python code/business_entity_resolution/src/run_pipeline.py \\
        --mode validate \\
        --data-dir . \\
        --output-dir output

Every mode is deterministic: rerunning with the same inputs and the same
``--seed`` reproduces byte-identical outputs.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Allow running as a plain script (``python src/run_pipeline.py``) as well
# as an importable package.
if __package__ in (None, ""):
    # .../code/business_entity_resolution/src/run_pipeline.py -> .../code
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from business_entity_resolution.src.config import PipelineConfig
    from business_entity_resolution.src.pipeline import run_predict, run_test, run_train_eval
    from business_entity_resolution.src.validate_submission import validate
else:
    from .config import PipelineConfig
    from .pipeline import run_predict, run_test, run_train_eval
    from .validate_submission import validate


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Business Entity Resolution pipeline (Amazon ML Challenge 2026).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument(
        "--data-dir", type=Path, default=Path("."),
        help="Directory containing dataset/ (and optionally utils/).",
    )
    ap.add_argument(
        "--mode", choices=["test", "train-eval", "predict", "validate"],
        default="test",
        help=(
            "test = produce submission files; "
            "train-eval = score on a held-out train slice; "
            "predict = inference with a saved model; "
            "validate = check output files only."
        ),
    )
    ap.add_argument(
        "--output-dir", type=Path, default=Path("output"),
        help="Where the TSV outputs are written.",
    )
    ap.add_argument("--model-path", type=Path, default=None,
                    help="Saved model JSON (used by --mode predict).")
    ap.add_argument("--save-model", type=Path, default=None,
                    help="If set, write the fitted model here as JSON.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only read this many rows per input file (smoke tests).")

    # --- tunables -------------------------------------------------------
    ap.add_argument("--df-cap", type=int, default=60,
                    help="Skip a blocking key shared by more than this many "
                         "records; bounds candidate fan-out.")
    ap.add_argument("--max-candidates", type=int, default=80,
                    help="Hard cap on candidates per Source 1 row. Blocking "
                         "sets the recall ceiling -- raise this first if "
                         "validation F_0.5 looks recall-capped.")
    ap.add_argument("--query-chunk", type=int, default=5000,
                    help="Source 1 rows held in flight per scoring chunk.")
    ap.add_argument("--workers", type=int, default=0,
                    help="Scoring processes (0 = every logical core).")
    ap.add_argument("--train-rows", type=int, default=300_000,
                    help="Source 1 rows used to fit the model (0 = all). "
                         "Cost is linear in rows.")
    ap.add_argument("--calib-rows", type=int, default=50_000,
                    help="Held-out rows used to pick the threshold (0 = all).")
    ap.add_argument("--threshold", type=float, default=None,
                    help="Decision threshold (default: calibrated or 0.65).")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="Skip threshold calibration.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--check-ids", action="store_true",
                    help="validate mode: also verify every referenced ID "
                         "exists in the test set (several GB of RAM).")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--report", type=Path, default=None,
                    help="Write a JSON run report here.")
    return ap


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.df_cap = args.df_cap
    cfg.max_candidates_per_row = args.max_candidates
    cfg.query_chunk = args.query_chunk
    if args.workers:
        cfg.workers = args.workers
    cfg.train_rows = args.train_rows
    cfg.calib_rows = args.calib_rows
    cfg.random_state = args.seed
    cfg.verbose = not args.quiet
    cfg.calibrate_threshold = not args.no_calibrate
    if args.threshold is not None:
        cfg.threshold = args.threshold
        cfg.heuristic_threshold = args.threshold
        cfg.calibrate_threshold = False
    if args.model_path is not None:
        cfg.extra["model_path"] = str(args.model_path)
    if args.save_model is not None:
        cfg.extra["save_model"] = str(args.save_model)
    return cfg


def setup_logging(quiet: bool) -> None:
    logging.basicConfig(
        level=logging.WARNING if quiet else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )


def cmd_validate(args: argparse.Namespace) -> int:
    matching = args.output_dir / "matching_results.tsv"
    candidate = args.output_dir / "candidate_pairs.tsv"
    test_dir = args.data_dir / "dataset" / "test"
    passed, issues = validate(
        matching, candidate, test_dir,
        check_ids=getattr(args, "check_ids", False),
    )
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.quiet)
    log = logging.getLogger("run_pipeline")
    cfg = config_from_args(args)

    if args.mode == "validate":
        return cmd_validate(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "test":
        report = run_test(args.data_dir, args.output_dir, cfg, limit=args.limit)
    elif args.mode == "train-eval":
        report = run_train_eval(args.data_dir, args.output_dir, cfg, limit=args.limit)
    elif args.mode == "predict":
        report = run_predict(args.data_dir, args.output_dir, cfg, limit=args.limit)
    else:                                        # pragma: no cover
        raise SystemExit(f"Unknown mode: {args.mode}")

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        log.info("Wrote report to %s", args.report)

    print(json.dumps(report, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
