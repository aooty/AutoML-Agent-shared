"""짝 판정을 계산한 배열을 저장한다. 모델도 원본 데이터도 없이 판정을 다시 계산할 수 있게.

``bench/runs/paired/*.json``의 델타는 모두 :func:`bench.paired.paired_delta`가 냈고, 그 함수는
``Scored``에서 딱 네 가지만 읽는다: 예측, test 정답, 양성 클래스 확률, 지표의 스칼라. 그걸
만들려면 :func:`bench.paired.reproduce`가 필요하고 그건 ``model.joblib``과 데이터셋을 요구하는데,
이 저장소는 ``bench/runs/random/``을 포함해 ``model.joblib``을 한 번도 추적한 적이 없다. 즉
독자는 판정 기준과 모든 실행의 모양까지 다시 읽을 수 있어도 숫자 자체는 이 저장소의 말을
믿어야 했다.

번들이 그 구멍을 막는다. leg마다 위의 네 가지에 :func:`bench.paired.score`가 지표를 재현하는 데
필요한 메타데이터를 더한 것이고, 그 뒤에 있는 모델 ~400MB에 비해 수백 KB다.
``bench/recheck.py``가 이것으로 leg마다의 점수와 짝마다의 구간을 다시 계산해 커밋된 판정과
비교한다.

번들이 못 하는 일 — 놓치기 쉬워서 여기 적어 둔다: **예측이 그 모델에서 나왔다는 것은 보여줄 수
없다.** test 지문을 기록하고 교차 검사하지만 지문은 test 행에 대한 것이고 모델에 대한 것이
아니다. 번들이 제3자에게 검증 가능하게 만드는 것은 *산술*이다. 출처는 여전히 커밋된 config,
실행 보관본, 그리고 이 문서 흔적에 달려 있다.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from bench.paired import Scored, Winner

# 납작한 npz 이름공간의 구분자. npz 키는 zip 안의 멤버 이름이 되고 거기서 슬래시는 일부
# 도구에 디렉터리로 읽히므로 ``/``가 아니라 ``::``를 쓴다.
SEP = "::"
META_KEY = "__meta__"
BUNDLE_SUFFIX = ".predictions.npz"


def bundle_path(verdict_path: Path) -> Path:
    """판정 파일 옆에 있어야 하는 번들.

    인자로 받지 않고 유도하는 이유는 명령줄에서 판정과 번들이 어긋날 수 없게 하려는 것이다:
    ``replan-seed42.json``은 항상 ``replan-seed42.predictions.npz``와 짝이다.
    """
    return verdict_path.with_suffix("").with_suffix(BUNDLE_SUFFIX)


def _leg_meta(dataset: str, arm: str, metric: str, scored: Scored) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "arm": arm,
        "metric": metric,
        "task": scored.task,
        "average": scored.average,
        "n_classes": scored.n_classes,
        "n_rows": scored.n_rows,
        "label": scored.winner.label,
        "fits": scored.winner.fits,
        "test_score": scored.test_score,
        "fingerprint": scored.fingerprint,
        "has_proba": scored.proba is not None,
    }


def write_bundle(
    path: Path, legs: dict[str, dict[str, Scored]], metrics: dict[str, str]
) -> dict[str, Any]:
    """데이터셋마다의 leg를 압축 npz 하나에 쓴다. 안에 심은 manifest를 돌려준다.

    ``legs``의 키는 데이터셋 → 팔 → ``Scored``이고, 팔 키는 판정 파일의 짝이 ``a``/``b``에 쓰는
    문자열과 같아야 한다 — 그래야 ``recheck``가 맞춰 둘 두 번째 작명 규칙 없이 저장된 leg와
    기록된 델타를 맞출 수 있다.

    정답은 회귀일 때 ``float64``로, 나머지는 그대로 저장하고 확률은 전체 폭으로 저장한다. 바이트를
    아끼려고 어느 쪽이든 좁히면 정확한 재검사가 불가능해지는데, 그게 이 파일이 존재하는 이유다.
    """
    arrays: dict[str, Any] = {}
    manifest: list[dict[str, Any]] = []
    for dataset in sorted(legs):
        metric = metrics[dataset]
        for arm in sorted(legs[dataset]):
            scored = legs[dataset][arm]
            stem = f"{dataset}{SEP}{arm}{SEP}"
            arrays[stem + "pred"] = np.asarray(scored.pred)
            arrays[stem + "y"] = np.asarray(scored.y_test)
            if scored.proba is not None:
                arrays[stem + "proba"] = np.asarray(scored.proba, dtype=np.float64)
            manifest.append(_leg_meta(dataset, arm, metric, scored))
    meta = {"format": 1, "generated_by": "bench/predictions.py", "legs": manifest}
    arrays[META_KEY] = np.asarray(json.dumps(meta, ensure_ascii=False, sort_keys=True))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return meta


def read_bundle(path: Path) -> tuple[dict[str, dict[str, Scored]], dict[str, Any]]:
    """판정을 계산한 ``Scored`` leg를 되살린다. ``(legs, meta)``를 돌려준다.

    leg 안의 ``Winner``는 껍데기다: 번들을 넘어 살아남는 것은 ``arm``, ``label``, ``val_score``,
    ``fits``뿐이고 재검사가 읽는 필드가 그것들이다. 디렉터리와 config는 살아남지 않고, 그래야
    한다 — 경로를 실어 보내는 번들은 독자에게 그 경로가 자기 기계에서도 풀린다고 믿게 만든다.
    """
    with np.load(path, allow_pickle=False) as data:
        if META_KEY not in data:
            raise ValueError(f"{path}: {META_KEY}가 없습니다 — bench/predictions.py가 쓴 파일이 아닙니다")
        meta = json.loads(str(data[META_KEY].item()))
        legs: dict[str, dict[str, Scored]] = {}
        for entry in meta["legs"]:
            dataset, arm = entry["dataset"], entry["arm"]
            stem = f"{dataset}{SEP}{arm}{SEP}"
            proba = data[stem + "proba"] if entry["has_proba"] else None
            legs.setdefault(dataset, {})[arm] = Scored(
                winner=Winner(
                    arm=arm,
                    label=entry["label"],
                    directory=Path("(번들에는 경로가 없습니다)"),
                    config={},
                    val_score=float("nan"),
                    recorded_test=float("nan"),
                    fits=int(entry["fits"]),
                ),
                pred=data[stem + "pred"],
                y_test=data[stem + "y"],
                test_score=float(entry["test_score"]),
                fingerprint=str(entry["fingerprint"]),
                task=str(entry["task"]),
                n_classes=int(entry["n_classes"]),
                proba=proba,
                average=str(entry["average"]),
            )
    return legs, meta
