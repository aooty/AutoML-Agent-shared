"""벤치마크의 CSV를 OpenML에서 다시 만든다. 무엇보다 먼저 한 번 돌린다.

    python -m bench.fetch            # 다섯 개 전부
    python -m bench.fetch adult      # 이름으로 하나

``bench/data/<name>.csv``에 정답 열을 OpenML 이름 그대로 포함해 쓴다 —
``run --data ... --target ...``가 기대하는 모양이다. 행은 커밋하지 않는다(``*.csv``는 저장소
전체에서 무시된다). 대신 이 스크립트와 ``datasets.py``의 ``data_id``가 그 행들의 사본 역할을
한다.

데이터셋마다 출력하는 것이 ``RESULTS.md``가 데이터셋의 신분으로 인용하는 값이다 — 행, 열,
결측 비율, 클래스 균형. CSV의 체크섬은 일부러 쓰지 않는다: 바이트는 그 파일을 쓴 pandas
버전에 달려 있어서 행이 같은 독자에게도 해시가 어긋나고, 이 벤치마크는 그 경우와 행이 실제로
다른 경우를 구분할 방법이 없다. 중요한 실패는 모양과 균형이 어긋나는 쪽이고, 그건 사람이 읽을
수 있다.
"""

from __future__ import annotations

import sys
import warnings

from bench.datasets import ALL, BY_NAME, DATA_DIR, Dataset


def fetch(dataset: Dataset) -> int:
    from sklearn.datasets import fetch_openml

    with warnings.catch_warnings():
        # OpenML 자체의 메타데이터 경고("multiple active versions")는 ``data_id``를 못박아
        # 둔 상황에서 남는 의문이 없다.
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


def describe(dataset: Dataset) -> None:
    """CSV를 설명한다. 거기에 써 넣은 frame이 아니다.

    둘은 같은 표가 아니다. OpenML은 ``category`` dtype으로 넘겨주는데, CSV를 한 번 왕복하면
    모든 열을 다시 추론하므로 정수 범주 코드 열이 ``int64``로 돌아와 레벨이 아니라 수치로
    estimator에 닿는다. ``house_sales``가 정확히 이렇다. 팔 세 개가 소비하는 것은 파일이므로
    설명하는 대상도 파일이다 — 다시 읽는 데 1초 들고, "노트는 범주형이라는데 카드는 수치라고
    한다" 부류의 혼란이 전부 사라진다.
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
    # 인코더 자신의 상한. 어느 팔이든 돌기 전에 버려질 열이 보이도록 여기서 적용한다 —
    # 먼저 돈 팔의 카드에만 나타나면 늦다.
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


def main(argv: list[str]) -> int:
    wanted = argv or [item.name for item in ALL]
    unknown = [name for name in wanted if name not in BY_NAME]
    if unknown:
        print(f"모르는 데이터셋: {unknown} — 아는 것은 {sorted(BY_NAME)}")
        return 2
    return max(fetch(BY_NAME[name]) for name in wanted)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
