"""Matching model: supervised scoring + F_0.5 threshold calibration.

Two operating modes:

* **Trained** -- when ``train_ground_truth.tsv`` is available, candidate
  pairs are labelled and a logistic regression is fitted.  The decision
  threshold is then selected by maximising macro F_0.5 on a held-out
  validation split (or on the training pairs if the data is too small to
  split).
* **Heuristic** -- a weighted, calibrated score over the same features, used
  when no labels are present.  Tuned to sit on the precision-heavy side of
  the decision boundary, because F_0.5 weights precision 2x over recall.

Licence note: scikit-learn is BSD-3-Clause and the model here is a linear
logistic regression (~40 parameters), comfortably inside the challenge's
"MIT/Apache 2.0 licensed, <= 8B parameters" rule for the final model.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .features import FEATURE_NAMES, N_FEATURES

log = logging.getLogger(__name__)

MODEL_FILENAME = "matching_model.json"


# --------------------------------------------------------------------------
# F_0.5 metric
# --------------------------------------------------------------------------

def f_beta(precision: float, recall: float, beta: float = 0.5) -> float:
    if precision <= 0.0 and recall <= 0.0:
        return 0.0
    b2 = beta * beta
    denom = (b2 * precision) + recall
    if denom == 0:
        return 0.0
    return (1.0 + b2) * precision * recall / denom


def macro_f05(
    y_true: dict[str, set[str]],
    y_pred: dict[str, set[str]],
) -> float:
    """Macro-averaged F_0.5 over Source 1 entities.

    Mirrors the challenge's definition: computed per Source 1 entity, then
    averaged, with singletons included (empty vs empty = 1.0).
    """
    if not y_true:
        return 0.0
    scores: list[float] = []
    for s1_id, truth in y_true.items():
        pred = y_pred.get(s1_id, set())
        if not truth and not pred:
            scores.append(1.0)
            continue
        tp = len(truth & pred)
        fp = len(pred - truth)
        fn = len(truth - pred)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append(f_beta(precision, recall, 0.5))
    return float(np.mean(scores)) if scores else 0.0


def f05_at_threshold(
    scores_by_row: dict[str, list[tuple[str, float]]],
    y_true: dict[str, set[str]],
    threshold: float,
) -> tuple[float, dict[str, set[str]]]:
    """Evaluate macro F_0.5 for one threshold, returning pred and score."""
    y_pred: dict[str, set[str]] = {}
    for s1_id, cands in scores_by_row.items():
        chosen = {cid for cid, sc in cands if sc >= threshold}
        y_pred[s1_id] = chosen
    return macro_f05(y_true, y_pred), y_pred


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------

class MatchingModel:
    """Logistic-regression scorer with a heuristic fallback."""

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        self.threshold = cfg.threshold
        self.kind = "heuristic"
        self._clf = None
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self._heuristic_w: np.ndarray | None = None
        self._heuristic_norm: float = 1.0
        self._heuristic_center: float = 0.5
        self._heuristic_slope: float = 10.0
        self._feature_names = list(FEATURE_NAMES)

    # -- heuristic fallback ------------------------------------------------

    def _init_heuristic(self) -> None:
        """Hand-tuned weights emphasising precision.

        A weighted sum of the features, *normalised* by the total weight
        mass so the result stays in a meaningful range, then squashed by a
        sigmoid.  Normalising matters: an unnormalised sum of ~30 weights
        saturates the sigmoid and destroys all discrimination.
        """
        w = np.zeros(N_FEATURES, dtype=np.float64)
        name = {n: i for i, n in enumerate(FEATURE_NAMES)}

        for k, weight in {
            "name_token_sort": 1.7,
            "name_token_set": 1.3,
            "name_ratio": 1.5,
            "name_core_ratio": 1.4,
            "name_content_ratio": 1.1,
            "name_wratio": 1.2,
            "name_jaccard_tokens": 1.8,
            "name_jaccard_content": 1.2,
            "name_initialism_match": 0.8,
            "name_len_ratio": 0.6,
            "addr_token_sort": 1.1,
            "addr_ratio": 0.9,
            "addr_wratio": 0.8,
            "addr_jaccard": 1.3,
            "addr_content_overlap": 1.0,
            "addr_sorted_ratio": 0.7,
            "postcode_match": 2.4,
            "phone_match": 2.6,
            "digit_jaccard": 1.0,
            "country_match": 0.5,
            "max_token_overlap": 1.4,
            "rare_token_share": 2.0,
            "name_x_addr": 1.6,
            "name_high_addr_ok": 1.1,
            "both_addr_nonempty": 0.3,
        }.items():
            w[name[k]] = weight

        # Mild penalties: length mismatch and address-only agreement are
        # weak evidence for a *business identity* match.
        for k, weight in {
            "name_len_diff": -0.9,
            "addr_len_diff": -0.4,
            "addr_only_boost": -0.6,
        }.items():
            w[name[k]] = weight

        self._heuristic_w = w
        # Total weight mass, used to normalise the sum into ~[0, 1].
        self._heuristic_norm = float(np.abs(w).sum()) or 1.0
        # Sigmoid midpoint: pairs scoring above this read as matches.
        self._heuristic_center = 0.50
        # Slope. 10 gives ~0.80 for a strong match and ~0.08 for a weak one.
        self._heuristic_slope = 10.0
        # Without labels there is nothing to calibrate against, so fall back
        # to the tuned precision-heavy default unless the user set --threshold.
        self.threshold = self.cfg.heuristic_threshold
        self.kind = "heuristic"

    # -- scaling helpers ---------------------------------------------------

    def _fit_scaler(self, X: np.ndarray) -> None:
        self._mu = X.mean(axis=0)
        self._sd = X.std(axis=0)
        self._sd[self._sd < 1e-8] = 1.0

    def _scale(self, X: np.ndarray) -> np.ndarray:
        if self._mu is None or self._sd is None:
            return X
        return (X - self._mu) / self._sd

    # -- training ----------------------------------------------------------

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        *,
        calibration: dict[str, list[tuple[str, float]]] | None = None,
        y_true: dict[str, set[str]] | None = None,
    ) -> dict:
        """Fit the scorer; optionally calibrate the threshold on held-out data.

        Parameters
        ----------
        X, y:
            Candidate-pair features and binary labels (1 = true match).
        calibration:
            ``{s1_id: [(cand_id, score), ...]}`` produced on a validation
            split, used with ``y_true`` to pick the F_0.5-optimal threshold.
        y_true:
            Ground-truth sets keyed by Source 1 id for that validation split.
        """
        report: dict = {"n_samples": int(len(y)), "n_positive": int(y.sum())}

        # Need both classes, else sklearn refuses to fit.
        if len(np.unique(y)) < 2 or len(y) < 50:
            self._init_heuristic()
            report["mode"] = "heuristic (insufficient class balance)"
            report["threshold"] = self.threshold
            return report

        try:
            from sklearn.linear_model import LogisticRegression
        except ImportError:                      # pragma: no cover
            self._init_heuristic()
            report["mode"] = "heuristic (sklearn unavailable)"
            report["threshold"] = self.threshold
            return report

        self._fit_scaler(X)
        Xs = self._scale(X.astype(np.float64))

        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            C=1.0,
            class_weight="balanced",
            random_state=self.cfg.random_state,
        )
        clf.fit(Xs, y)
        self._clf = clf
        self.kind = "logistic_regression"
        report["mode"] = "logistic_regression"
        report["n_features"] = int(X.shape[1])

        coef = clf.coef_.ravel()
        top = sorted(
            zip(FEATURE_NAMES, coef.tolist()),
            key=lambda kv: abs(kv[1]),
            reverse=True,
        )[:12]
        report["top_features"] = [{k: round(v, 4) for k, v in top}]

        # -- threshold calibration ----------------------------------------
        if (
            self.cfg.calibrate_threshold
            and calibration is not None
            and y_true is not None
            and y_true
        ):
            best_thr, best_score = self.threshold, -1.0
            grid = []
            for thr in self.cfg.threshold_grid:
                score, _ = f05_at_threshold(calibration, y_true, thr)
                grid.append({"threshold": thr, "f05": round(score, 5)})
                if score > best_score + 1e-12:
                    best_score, best_thr = score, thr
            self.threshold = float(best_thr)
            report["threshold"] = self.threshold
            report["val_f05"] = round(best_score, 5)
            report["calibration_curve"] = grid
            log.info(
                "Calibrated threshold=%.2f (val macro F_0.5=%.4f)",
                self.threshold, best_score,
            )
        else:
            report["threshold"] = self.threshold
            if calibration is not None and y_true:
                score, _ = f05_at_threshold(calibration, y_true, self.threshold)
                report["val_f05"] = round(score, 5)

        return report

    # -- streaming training -------------------------------------------------

    def begin_streaming(self) -> None:
        """Start a mini-batch training session.

        The real training set has 7.6M positive labels; materialising all of
        them as a feature matrix would need several GB.  ``SGDClassifier``
        lets us learn from batches and discard them immediately.  Our
        features are already bounded ratios in [0,1], so no scaler is
        needed (``_mu``/``_sd`` stay ``None``, which makes ``_scale`` a
        no-op).
        """
        try:
            from sklearn.linear_model import SGDClassifier
        except ImportError:                       # pragma: no cover
            self._init_heuristic()
            return
        self._clf = SGDClassifier(
            loss="log_loss",
            alpha=1e-4,
            penalty="l2",
            max_iter=1,
            tol=None,
            random_state=self.cfg.random_state,
            average=True,        # averaged weights are much more stable here
        )
        self._mu = None
        self._sd = None
        self._stream_started = False
        self._stream_n = 0
        self.kind = "sgd_logistic_regression"

    def partial_fit(self, X: np.ndarray, y: np.ndarray) -> int:
        """Fold one mini-batch into the running model. Returns samples seen."""
        if X.size == 0:
            return 0
        if self._clf is None or self.kind != "sgd_logistic_regression":
            self.begin_streaming()
        if not hasattr(self, "_stream_started") or not self._stream_started:
            self._clf.partial_fit(X, y, classes=np.array([0, 1], dtype=np.int64))
            self._stream_started = True
        else:
            self._clf.partial_fit(X, y)
        self._stream_n = getattr(self, "_stream_n", 0) + int(len(y))
        return int(len(y))

    # -- inference ---------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if X.size == 0:
            return np.zeros(0, dtype=np.float64)
        if self._clf is not None:
            if (self.kind == "sgd_logistic_regression"
                    and not getattr(self, "_stream_started", False)):
                # A streaming session was opened but never received a batch
                # -- which happens when blocking retrieves no positives at
                # all (e.g. a truncated run).  Fall back to the heuristic
                # rather than letting sklearn raise NotFittedError.
                self._clf = None
            else:
                return self._clf.predict_proba(
                    self._scale(X.astype(np.float64))
                )[:, 1]
        if self._heuristic_w is None:
            self._init_heuristic()
        # Normalised weighted average of the features, in roughly [0, 1],
        # then a sigmoid so the value reads as a probability.
        raw = X.astype(np.float64) @ self._heuristic_w
        adjusted = raw / self._heuristic_norm
        logit = self._heuristic_slope * (adjusted - self._heuristic_center)
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -40, 40)))

    def score_pairs(
        self,
        X: np.ndarray,
        pair_ids: list[tuple[str, str]],
    ) -> dict[str, list[tuple[str, float]]]:
        """Score aligned pairs -> ``{s1_id: [(cand_id, prob), ...]}``."""
        out: dict[str, list[tuple[str, float]]] = {}
        probs = self.predict_proba(X)
        for (s1_id, cand_id), p in zip(pair_ids, probs):
            out.setdefault(s1_id, []).append((cand_id, float(p)))
        return out

    def select(
        self,
        scores: dict[str, list[tuple[str, float]]],
    ) -> dict[str, list[str]]:
        """Apply the threshold, preserving deterministic ordering."""
        thr = self.threshold
        chosen: dict[str, list[str]] = {}
        for s1_id, cands in scores.items():
            # Highest confidence first; id breaks ties deterministically.
            hits = sorted(
                ((cid, sc) for cid, sc in cands if sc >= thr),
                key=lambda kv: (-kv[1], kv[0]),
            )
            chosen[s1_id] = [cid for cid, _ in hits]
        return chosen

    # -- persistence -------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload: dict = {
            "kind": self.kind,
            "threshold": self.threshold,
            "feature_names": self._feature_names,
            "heuristic_norm": float(getattr(self, "_heuristic_norm", 1.0)),
            "heuristic_center": float(getattr(self, "_heuristic_center", 0.5)),
            "heuristic_slope": float(getattr(self, "_heuristic_slope", 10.0)),
            "heuristic_weights": (
                None if self._heuristic_w is None
                else [float(v) for v in self._heuristic_w]
            ),
            "mean": None if self._mu is None else [float(v) for v in self._mu],
            "std": None if self._sd is None else [float(v) for v in self._sd],
        }
        if self._clf is not None:
            payload["logreg"] = {
                "classes": self._clf.classes_.tolist(),
                "coef": self._clf.coef_.tolist(),
                "intercept": self._clf.intercept_.tolist(),
            }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        log.info("Saved model (%s) to %s", self.kind, path)

    @classmethod
    def load(cls, path: str | Path, cfg: PipelineConfig) -> "MatchingModel":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(cfg)
        model.kind = payload.get("kind", "heuristic")
        model.threshold = float(payload.get("threshold", cfg.threshold))
        model._feature_names = list(payload.get("feature_names", FEATURE_NAMES))
        model._heuristic_norm = float(payload.get("heuristic_norm", 1.0))
        model._heuristic_center = float(payload.get("heuristic_center", 0.5))
        model._heuristic_slope = float(payload.get("heuristic_slope", 10.0))
        hw = payload.get("heuristic_weights")
        model._heuristic_w = None if hw is None else np.asarray(hw, dtype=np.float64)
        model._mu = None if payload.get("mean") is None else np.asarray(payload["mean"])
        model._sd = None if payload.get("std") is None else np.asarray(payload["std"])

        lr = payload.get("logreg")
        if lr:
            try:
                from sklearn.linear_model import LogisticRegression
                clf = LogisticRegression(
                    solver="lbfgs", max_iter=2000,
                    random_state=cfg.random_state,
                )
                coef = np.asarray(lr["coef"], dtype=np.float64)
                intercept = np.asarray(lr["intercept"], dtype=np.float64)
                # Reconstruct without refitting.
                clf.classes_ = np.asarray(lr["classes"])
                clf.coef_ = coef
                clf.intercept_ = intercept
                clf.n_features_in_ = coef.shape[1]
                model._clf = clf
                model.kind = "logistic_regression"
            except Exception as exc:            # pragma: no cover
                log.warning("Could not restore logistic model (%s); using heuristic.", exc)
                model._init_heuristic()
        elif model._heuristic_w is None:
            model._init_heuristic()
        return model
