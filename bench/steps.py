"""Which extra pipeline steps help? A free check with no LLM.

Roles:

* Candidates — the steps tried and the missing-rate cut.
* Candidate steps — the sklearn steps each candidate adds.
* Comparison — fit each candidate, compare it to baseline.

Usage:

    python -m bench.steps --data local/sample.csv --target died
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from automl_agent.scoring.intervals import describe_paired, paired_delta
from automl_agent.scoring.ranking import best_cut_ceiling, ks_statistic
from automl_agent.scoring.splits import split_three_way
from automl_agent.scripts.train import (
    LogBuffer,
    _proba,
    build_estimator,
    fit_estimator,
    load_data,
    scorers,
)

# --- Role: candidates -----------------------------------------------------------------

# Losers stay listed: they show why the registry is short.
CANDIDATES = (
    "baseline",
    "per_column_impute",
    "interactions",
    "both",
    "quantile",
    "power",
    "select_k",
    "variance",
    "pca",
)
MODELS = ("logreg", "hist_gbdt")
# Taken from real plans, not tuned on these rows.
HIGH_MISSING = 0.3


# --- Role: candidate steps ------------------------------------------------------------


def _extra(name: str, high: list[int], rest: list[int], n_columns: int) -> list[tuple[str, Any]]:
    """_extra | Candidate steps: steps put before the estimator."""
    from sklearn.compose import ColumnTransformer
    from sklearn.decomposition import PCA
    from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import (
        PolynomialFeatures,
        PowerTransformer,
        QuantileTransformer,
        StandardScaler,
    )

    median = [("impute0", SimpleImputer(strategy="median"))]
    per_column = [
        (
            "per_column",
            ColumnTransformer(
                [
                    ("high", SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True), high),
                    ("rest", SimpleImputer(strategy="median"), rest),
                ]
            ),
        )
    ]
    interactions = [
        ("interactions", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False))
    ]
    return {
        "baseline": [],
        "per_column_impute": per_column,
        "interactions": median + interactions,
        "both": per_column + interactions,
        "quantile": median
        + [("quantile", QuantileTransformer(output_distribution="normal", random_state=42))],
        "power": median + [("power", PowerTransformer())],
        "select_k": median + [("select", SelectKBest(f_classif, k=max(3, n_columns // 2)))],
        "variance": median + [("variance", VarianceThreshold(0.0))],
        "pca": median + [("scale0", StandardScaler()), ("pca", PCA(n_components=0.95, random_state=42))],
    }[name]


# --- Role: comparison -----------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """Fit each candidate per model and print paired deltas."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="원본 CSV (저장소 밖 또는 local/)")
    parser.add_argument("--target", required=True, help="정답 컬럼 이름")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args(argv)

    import numpy as np
    from sklearn.pipeline import Pipeline

    # Keep the trainer's log quiet; it is not our output.
    log = LogBuffer(echo=False)
    cfg = {
        "model": "logreg",
        "hyperparams": {},
        "task": "classification",
        "seed": args.seed,
        "metric": "roc_auc",
        "data": {"path": str(args.data), "target_column": args.target},
    }
    x, y, n_classes, groups, task, schema = load_data(cfg, log)
    splits = split_three_way(x, y, args.seed, groups=groups, stratify=True)
    columns = list((schema or {}).get("columns") or [])
    n_columns = len(columns) or int(x.shape[1])
    high = [
        position
        for position, rate in enumerate(np.isnan(splits.x_train).mean(axis=0))
        if rate > HIGH_MISSING
    ]
    rest = [position for position in range(n_columns) if position not in high]
    print(
        f"encoded columns {n_columns}, over {HIGH_MISSING:.0%} missing in train: {len(high)}\n"
        f"train {len(splits.y_train)} / val {len(splits.y_val)} / test {len(splits.y_test)}"
    )

    def fit(model_key: str, candidate: str) -> tuple[Any, Any, int | None]:
        pipe, _applied, _dropped = build_estimator(
            model_key, {}, args.seed, log, {}, labels=[0, 1], task=task
        )
        extra = _extra(candidate, high, rest, n_columns)
        steps = extra + [(name, step) for name, step in pipe.steps if not extra or name != "impute"]
        pipe = Pipeline(steps)
        fit_estimator(
            pipe, splits.x_train, splits.y_train, args.seed, log,
            stratify=True, groups=splits.groups_train,
        )
        return (
            pipe.predict(splits.x_val),
            _proba(pipe, splits.x_val, 2, log),
            getattr(pipe.steps[-1][1], "n_features_in_", None),
        )

    for model_key in MODELS:
        print(f"\n=== {model_key} ===")
        base = fit(model_key, "baseline")
        for candidate in CANDIDATES:
            if candidate == "baseline":
                continue
            pred, proba, width = fit(model_key, candidate)
            ceiling = best_cut_ceiling(ks_statistic(splits.y_val, proba))
            print(f"  {candidate:18} width={width}  ceiling={ceiling}")
            for metric in ("roc_auc", "balanced_accuracy"):
                def one(ys: Any, ps: Any, pr: Any, _metric: str = metric) -> float:
                    return float(scorers(ys, ps, pr, "binary", task)[_metric]())

                delta = paired_delta(
                    one, splits.y_val, (pred, proba), base[:2],
                    direction="maximize", groups=splits.groups_val,
                    seed=args.seed, resamples=args.resamples,
                )
                if delta is None:
                    print(f"    {metric:18} (구간 없음)")
                    continue
                print(f"    {metric:18} {describe_paired({**delta.flatten(), 'metric': metric})}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
