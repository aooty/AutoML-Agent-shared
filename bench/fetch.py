"""Rebuild the benchmark CSVs from OpenML. Run this first.

Roles:

* Fetch — download one dataset and write its CSV.
* Describe — print shape, missing share, and class balance.
* CLI — pick datasets by name and run.

Usage:

    python -m bench.fetch
    python -m bench.fetch adult
"""

from __future__ import annotations

import sys
import warnings

from bench.datasets import ALL, BY_NAME, DATA_DIR, Dataset

# --- Role: fetch ----------------------------------------------------------------------


def fetch(dataset: Dataset) -> int:
    from sklearn.datasets import fetch_openml

    with warnings.catch_warnings():
        # data_id is pinned, so version warnings do not matter.
        warnings.simplefilter("ignore")
        bunch = fetch_openml(data_id=dataset.data_id, as_frame=True, parser="auto")

    frame = bunch.frame
    if dataset.target not in frame.columns:
        print(
            f"{dataset.name}: 정답 열 '{dataset.target}'이 없습니다 — "
            f"이 data_id가 내는 이름은 {bunch.target.name!r}입니다"
        )
        return 1

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(dataset.csv_path, index=False, encoding="utf-8")
    describe(dataset)
    return 0


# --- Role: describe ------------------------------------------------------------------


def describe(dataset: Dataset) -> None:
    """Describe the CSV as re-read, not the fetched frame.

    A CSV round trip can turn category codes into numbers.
    """
    import pandas as pd

    from automl_agent.dataset.features import (
        MAX_ONEHOT_CARDINALITY,
        is_numeric_column,
        is_text_like_column,
    )

    frame = pd.read_csv(dataset.csv_path, low_memory=False)
    features = frame.drop(columns=[dataset.target])
    numeric = [name for name in features.columns if is_numeric_column(features[name])]
    text = [name for name in features.columns if is_text_like_column(features[name])]
    # Show columns the encoder will drop, before any arm runs.
    too_wide = {
        str(name): int(features[name].dropna().nunique())
        for name in text
        if features[name].dropna().nunique() > MAX_ONEHOT_CARDINALITY
    }
    missing = float(features.isna().to_numpy().mean())
    above = int((features.isna().mean() > 0.01).sum())

    print(f"{dataset.name:16s} id={dataset.data_id} → {dataset.csv_path.name}")
    print(
        f"{'':16s} {len(frame)}행 × {features.shape[1]}특성 "
        f"(수치 {len(numeric)} · 범주형 {len(text)}) "
        f"결측 {missing:.4f} (1% 넘는 열 {above}개)"
    )
    if too_wide:
        print(f"{'':16s} 고유값 {MAX_ONEHOT_CARDINALITY}개 초과로 버려질 열: {too_wide}")
    if dataset.task == "regression":
        target = frame[dataset.target]
        print(f"{'':16s} 목표 {dataset.target}: {target.min():.4g} ~ {target.max():.4g}")
    else:
        shares = frame[dataset.target].value_counts(normalize=True, dropna=False)
        print(f"{'':16s} 클래스 {dict(shares.round(4))}")


# --- Role: CLI -----------------------------------------------------------------------


def main(argv: list[str]) -> int:
    wanted = argv or [item.name for item in ALL]
    unknown = [name for name in wanted if name not in BY_NAME]
    if unknown:
        print(f"모르는 데이터셋: {unknown} — 아는 것은 {sorted(BY_NAME)}")
        return 2
    return max(fetch(BY_NAME[name]) for name in wanted)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
