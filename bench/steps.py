"""후보 파이프라인 step 중에 값을 하는 건 어느 것인가? 공짜 대조 — LLM도, 새 실행도 없다.

``docs/REGISTRY-GAP.md``가 쓴 것과 같은 방법이다: 구성을 손으로 짜고, 실행기 자신의 코드로
실행기 자신의 split에서 학습시키고, 같은 행에서 짝 Δ와 95% CI로 비교한다. 여기서는 LLM을 부르지
않고 bench 팔을 쓰지도 않으므로 판정 대상 묶음을 건드리지 않는다.

새 스크립트가 아니라 실행기의 함수를 쓰는 이유: ``load_data``가 정답 정책과 인코딩을 적용하고,
``split_three_way``가 이 저장소의 모든 점수가 측정되는 프로토콜이고, ``scorers``가 각 지표의 단
하나뿐인 정의다. 셋 중 어느 것이든 두 번째 구현이 생기면 숫자가 ``result.json``의 것과 비교
불가능해진다.

**행은 출력하지 않는다.** 원본 파일은 고정 스크립트들이 그러듯 이 프로세스 안에서만 열고, 출력은
열 개수 집계와 점수다. 이걸 통과한 후보 step이 :mod:`automl_agent.dataset.pipeline`가 registry에
넣는 것들이고, 통과하지 못한 것은 그 모듈 docstring에 누락이 아니라 측정 결과로 적혀 있다.

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

# 검토한 변환 전부. 진 것까지 남긴 이유는 registry가 짧은 것이 출발점이 아니라 결과임을
# 이 목록이 보여주기 때문이다.
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
# 이만큼 결측인 열은 상수+지시자로 채우고 나머지는 중앙값으로 채운다. 이 기준값은 계획들이
# 열 이름을 댈 때("cr_diff, gcs and paco2") 실제로 쓴 값이고 훑어서 고른 매개변수가 아니다 —
# 여기서 훑으면 step의 값을 매기는 바로 그 행에서 step의 구성을 고르는 셈이 된다.
HIGH_MISSING = 0.3


def _extra(name: str, high: list[int], rest: list[int], n_columns: int) -> list[tuple[str, Any]]:
    """후보마다 estimator 앞에 끼워 넣는 step."""
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="원본 CSV (저장소 밖 또는 local/)")
    parser.add_argument("--target", required=True, help="정답 컬럼 이름")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resamples", type=int, default=1000)
    args = parser.parse_args(argv)

    import numpy as np
    from sklearn.pipeline import Pipeline

    # 실행기의 로그는 이 측정의 출력이 아니므로 출력 없이 모으기만 한다.
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
