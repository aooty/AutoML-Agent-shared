"""3-arm 벤치마크를 판정한다: 팔 사이 *test* 차이의 짝지은 부트스트랩.

구간 둘이 아니라 차이를 재는 이유. 한 팔의 95% 구간은 *주변부*다 — 폭의 대부분이 행 추출 잡음이고, 두
팔은 바로 그 같은 test 행에서 채점되므로 그 잡음은 짝에 공통이고 차이에서 상쇄된다. 그래서 한 팔의
점추정을 다른 팔의 주변부 구간에 대고 보려면 구간 폭만큼 큰 효과가 필요한데, adult에서는
balanced_accuracy 0.017쯤이고 이 벤치마크가 찾는 어떤 효과보다 크다. MIMIC 실행이 늘 드는 예다:
*같은* 구성을 아홉 번 반복한 폭(0.0110)이 측정 하나의 구간 폭(0.028) 안에 들어갔다. 차이를 재추출하면
공통항이 사라지고 실제로 다른 것, 즉 계획이 측정된다.

기록된 점수를 믿지 않고 분할을 다시 계산하는 이유. 두 팔의 수는 같은 행에서 나왔을 때만 뺄 수 있고,
"같은 시드"는 그게 아니다 — 시드는 자기에게 건네진 행렬을 나누고, 거기 닿는 것은 정답 결측 정책이나
다시 추출한 파일이나 정답이 이제 회귀로 읽히는 카드에 따라 움직인다. 그래서 이 스크립트는 팔마다 자기
``train_config.json``에서 분할을 다시 유도하고 test 행의 지문을 뜨고, 서로 다른 두 분할 사이의 델타를
출력하는 대신 **팔끼리 어긋나는 데이터셋은 판정을 거부한다**. 우승자도 다시 채점해 실행이 기록한 수와
맞는지 본다. 거기서 어긋나면 지문은 맞아도 행이 움직인 것이다.

계약은 어느 팔도 돌기 전에 ``docs/RESULTS.md``에 사전 등록된 것이다: 예산 맞춘 primary 비교(루프 팔의
학습 횟수 k를 random 팔의 앞 k회 뽑기에 대고), 보조로 best-of-5, 재추출 4000회, 95% CI, 팔마다
``selection_gap``.

팔들이 돌아간 것과 같이 ``OMP_NUM_THREADS=1``로 돌린다. 그 값은 출력에 기록되어 나중 독자가 그랬는지
볼 수 있다.

사용법:
    python -m bench.paired                      # 판정 시드에서 데이터셋 전부
    python -m bench.paired adult --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from automl_agent.config import MODEL_FILENAME, SCHEMA_FILENAME, read_json_object
from automl_agent.scoring.metrics import TASK_REGRESSION
from automl_agent.scoring.splits import split_three_way, val_fingerprint
from automl_agent.scripts.train import (
    LogBuffer,
    load_data,
    load_schema,
    scorers,
)
from automl_agent.scripts.train import (
    _proba as positive_proba,  # 다시 구현하지 않고 그대로 쓴다: ``reproduce`` 참고
)
from automl_agent.threads import describe_thread_state, thread_state
from bench.datasets import ALL, BY_NAME, JUDGED_NAMES, JUDGED_SEED, Dataset
from bench.random_search import RUNS_DIR
from bench.random_search import run_dir as random_run_dir

RESAMPLES = 4000
# 1e-9이 아니라 1e-6. 트리의 예측은 정확한 순회지만 ``logreg``의 예측은 ``X @ coef_``다 —
# 합산 순서가 스레드 수에 달린 BLAS matmul이고, 그 수는 학습 subprocess와 이 프로세스에서 다르다.
# 그 경로에서 실측된 흔들림은 5.7e-08이다. 그걸 걸러 내는 허용 오차는 아무도 안 읽는 허용 오차다.
# A2에서 잰 xgboost 0.0026 흔들림은 예측이 아니라 *학습*에서였고, 이 스크립트는 아무것도 학습하지
# 않는다.
SCORE_TOLERANCE = 1e-6
ARTIFACTS_DIR = RUNS_DIR / "artifacts"
OUT_DIR = RUNS_DIR / "paired"
# 스레드 수는 기록에 들어가야 한다: A2가 스레드 수에 걸친 balanced_accuracy 폭 0.0077을 쟀고,
# 짝 판정 하나의 반폭의 0.99배다. 어느 스레드 상태에서 돌았는지 말하지 않는 판정은 다른 판정과
# 비교할 수 없다. 변수 목록과 기록의 모양은 ``automl_agent.threads``에서 온다. 실행들이 이제
# ``result.json``마다 써 넣는 것과 같은 출처다 — 여기 사본을 하나 더 두면 둘이 어긋날 수 있고,
# 그러면 판정과 그것이 판정하는 시도들이 서로 다른 환경을 서술하게 된다.


class Refusal(Exception):
    """이 데이터셋은 판정할 수 없다. 아무도 쓸 수 없는 델타를 돌려주는 대신 올린다."""


# --------------------------------------------------------------------------- #
# 팔마다 우승자 찾기
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Winner:
    """팔 하나가 고른 모델, 그리고 사전 등록이 요구하는 두 수."""

    arm: str
    label: str
    directory: Path
    config: dict[str, Any]
    val_score: float
    recorded_test: float
    fits: int


@dataclass
class Scored:
    """이 스크립트가 직접 유도한 분할에서 다시 채점한 우승자."""

    winner: Winner
    pred: Any
    y_test: Any
    test_score: float
    fingerprint: str
    task: str
    n_classes: int
    proba: Any = None
    average: str = "binary"

    @property
    def n_rows(self) -> int:
        return len(self.y_test)


def loop_winner(arm: str, directory: Path, metric: str) -> Winner | None:
    """전체 루프 실행(``llm`` 또는 ``no_llm``)의 우승자. ``history.json``에서 읽는다.

    ``holdout.json``이 아니라 ``history.json``인 이유: holdout 파일은 점수와 분할 라벨과 적용된 설정을
    기록하지만 *그것을 낸 iteration이 어느 것인지*는 적지 않는다. 그건 ``best.iteration``만 말하고,
    기본값 ``--keep-models best``에서는 진 iteration의 ``model.joblib``이 이미 지워져 있다 — 그래서 그
    필드는 모델로 가는 편한 경로가 아니라 풀리는 유일한 경로다.
    """
    history = read_json_object(directory / "history.json")
    if history is None:
        return None
    holdout = read_json_object(directory / "holdout.json")
    if holdout is None:
        raise Refusal(f"{arm}: history.json은 있는데 holdout.json이 없습니다 ({directory})")
    best = dict(history.get("best") or {})
    iteration = best.get("iteration")
    if not isinstance(iteration, int):
        raise Refusal(f"{arm}: history.json의 best.iteration이 정수가 아닙니다 ({iteration!r})")
    iter_dir = directory / "train" / f"iter_{iteration:02d}"
    recorded = dict(holdout.get("metrics") or {}).get(metric)
    if not isinstance(recorded, (int, float)):
        raise Refusal(f"{arm}: holdout.json에 {metric}이 없습니다 ({directory})")
    config = read_json_object(iter_dir / "train_config.json")
    if config is None:
        raise Refusal(f"{arm}: iteration {iteration}의 train_config.json을 읽을 수 없습니다")
    if not (iter_dir / MODEL_FILENAME).exists():
        raise Refusal(
            f"{arm}: iteration {iteration}의 {MODEL_FILENAME}이 없습니다 — "
            "--keep-models가 우승자까지 지웠거나 잘못된 iteration을 가리킵니다"
        )
    return Winner(
        arm=arm,
        label=f"iteration {iteration}",
        directory=iter_dir,
        config=config,
        val_score=float(best.get("score", float("nan"))),
        recorded_test=float(recorded),
        fits=int(history.get("iterations") or 0),
    )


def random_winner(directory: Path, k: int, metric: str) -> Winner:
    """random 팔의 **앞 k회 뽑기** 중 우승자. ``summary.json``에서 읽는다.

    뽑기 순서는 시드가 고정하고 test 행을 건드리기 전에 쓰였으므로 "앞 k회"는 결과를 보고 한 선택이
    아니다 — 팔을 다시 돌리지 않고 예산을 맞출 수 있는 이유가 그것이다.
    """
    summary = read_json_object(directory / "summary.json")
    if summary is None:
        raise Refusal(f"random: summary.json을 읽을 수 없습니다 ({directory})")
    chosen = dict(summary.get("best_by_val_for_k") or {}).get(str(k))
    if not isinstance(chosen, int):
        raise Refusal(f"random: k={k}에 대한 best_by_val_for_k 항목이 없습니다")
    records = {int(r["draw"]): r for r in summary.get("records") or []}
    record = records.get(chosen)
    if record is None:
        raise Refusal(f"random: draw {chosen}의 기록이 summary.json에 없습니다")
    if str(summary.get("metric")) != metric:
        raise Refusal(
            f"random: summary.json의 지표가 {summary.get('metric')!r}인데 판정 지표는 {metric!r}입니다"
        )
    test = dict(record.get("test") or {})
    recorded = test.get("test_score")
    if not isinstance(recorded, (int, float)):
        raise Refusal(
            f"random: draw {chosen}에 test 점수가 없습니다 — "
            "앞 k회 우승자만 test를 봅니다, 이 뽑기가 그중 하나가 아닙니다"
        )
    draw_dir = Path(str(record["dir"]))
    config = read_json_object(draw_dir / "train_config.json")
    if config is None:
        raise Refusal(f"random: draw {chosen}의 train_config.json을 읽을 수 없습니다")
    if not (draw_dir / MODEL_FILENAME).exists():
        raise Refusal(f"random: draw {chosen}의 {MODEL_FILENAME}이 없습니다 ({draw_dir})")
    return Winner(
        arm=f"random@k={k}",
        label=f"draw {chosen}",
        directory=draw_dir,
        config=config,
        val_score=float(record.get("val_score", float("nan"))),
        recorded_test=float(recorded),
        fits=k,
    )


# --------------------------------------------------------------------------- #
# 분할 다시 유도하고 다시 채점하기
# --------------------------------------------------------------------------- #


def reproduce(winner: Winner, metric: str) -> Scored:
    """우승자 자신의 config에서 이 스크립트가 유도한 test 행에서 ``winner``를 채점한다.

    모든 단계가 저장소 자신의 것이다: 모델 옆에 저장된 schema로 ``load_data``, config의 시드로
    ``split_three_way``, 양성 열은 ``_proba``, 지표는 ``scorers``. 어느 하나라도 다시 구현하면 이 수와
    기록된 수가 어긋날 때 해석이 불가능해진다 — 분할 탓일 수도 있고 이 파일 탓일 수도 있다. 코드를
    공유하면 남는 원인이 분할뿐이다.
    """
    import joblib

    # ``echo=False``: ``load_data``는 모델마다 열 줄 넘게 늘어놓고, 이 스크립트는 데이터셋마다 팔마다
    # 하나씩 올린다. 출력은 표이고 그 줄들이 표를 묻는다. 모으기는 계속 하므로 실패는 버퍼에서 읽어
    # 낼 수 있다.
    log = LogBuffer(echo=False)
    schema = load_schema(winner.directory / SCHEMA_FILENAME, log)
    x_arr, y_arr, n_classes, groups, task, _schema = load_data(winner.config, log, schema)
    seed = int(winner.config.get("seed", 42))
    splits = split_three_way(
        x_arr, y_arr, seed, groups=groups, stratify=task != TASK_REGRESSION
    )
    model = joblib.load(winner.directory / MODEL_FILENAME)
    y_test = np.asarray(splits.y_test)
    pred = model.predict(splits.x_test)
    proba = positive_proba(model, splits.x_test, n_classes, log)
    average = "binary" if n_classes == 2 else "macro"
    return Scored(
        winner=winner,
        pred=pred,
        y_test=y_test,
        proba=proba,
        test_score=score(metric, y_test, pred, proba, average, task),
        fingerprint=val_fingerprint(splits.x_test, y_test),
        task=task,
        n_classes=n_classes,
        average=average,
    )


def score(metric: str, y_true: Any, pred: Any, proba: Any, average: str, task: str) -> float:
    """실행들이 채점된 것과 같은 thunk 표로 계산한 ``metric``."""
    thunk = scorers(y_true, pred, proba, average, task=task).get(metric)
    if thunk is None:
        raise Refusal(f"{metric}은 이 task({task})의 지표 목록에 없습니다")
    return float(thunk())


SPLIT_FIELDS = ("data", "target_missing", "seed", "task", "metric")


def split_identity(config: dict[str, Any]) -> dict[str, Any]:
    """test 분할에 *어느 행이* 들어가는지를 정하는 config 필드.

    config 전체가 아니다: 모델과 하이퍼파라미터는 설계상 팔마다 다르고, 그걸 섞어 넣으면 모든 비교가
    불일치로 보인다. 남는 것은 ``split_three_way``와 로더가 실제로 읽는 것들이다.
    """
    return {key: config.get(key) for key in SPLIT_FIELDS if key in config}


# --------------------------------------------------------------------------- #
# 짝지은 부트스트랩
# --------------------------------------------------------------------------- #


@dataclass
class Delta:
    a: str
    b: str
    kind: str
    delta: float
    ci_low: float
    ci_high: float
    p_better: float
    resamples_used: int
    identical_predictions: bool = False


def paired_delta(a: Scored, b: Scored, metric: str, resamples: int, seed: int, kind: str) -> Delta:
    """공유된 test 행의 재추출에서 차이 ``a - b``를 부트스트랩한다.

    행 인덱스는 짝마다 같은 시드로 다시 seed한 rng에서 나오므로, 한 데이터셋의 모든 짝이 동일한 재추출
    수열을 본다 — 대안인 공유 인덱스 행렬 하나는 adult에서 4000 x 9769 int64이고 아무 쓸모 없이 실행
    내내 312MB를 붙잡는다. 구간과 ``P(delta > 0)``은 한 번의 통과에서 나온다. 따로 계산하면 두 번
    재추출하고 둘이 어긋날 수 있다.

    퇴화한 재추출은 0으로 세지 않고 버린다: 한 클래스만 뽑힌 재추출에는 balanced_accuracy가 없고, 정답이
    상수로 뽑힌 재추출에는 r2가 없다. 그 개수는 보고한다 — "4000 중 3960"과 "4000 중 40"은 다른
    상황이기 때문이다.
    """
    base = Delta(
        a=a.winner.arm, b=b.winner.arm, kind=kind, delta=a.test_score - b.test_score,
        ci_low=float("nan"), ci_high=float("nan"), p_better=float("nan"), resamples_used=0,
    )
    if np.array_equal(np.asarray(a.pred), np.asarray(b.pred)):
        base.identical_predictions = True
        return base
    y_test = a.y_test
    rng = np.random.default_rng(seed)
    n = len(y_test)
    deltas: list[float] = []
    for _ in range(resamples):
        rows = rng.integers(0, n, size=n)
        y_boot = y_test[rows]
        if a.task == TASK_REGRESSION:
            if float(np.var(y_boot)) == 0.0:
                continue
        elif len(np.unique(y_boot)) < 2:
            continue
        a_proba = None if a.proba is None else a.proba[rows]
        b_proba = None if b.proba is None else b.proba[rows]
        deltas.append(
            score(metric, y_boot, a.pred[rows], a_proba, a.average, a.task)
            - score(metric, y_boot, b.pred[rows], b_proba, b.average, b.task)
        )
    if not deltas:
        raise Refusal(f"{a.winner.arm} vs {b.winner.arm}: 쓸 수 있는 재추출이 없습니다")
    arr = np.asarray(deltas)
    base.ci_low, base.ci_high = (float(v) for v in np.quantile(arr, [0.025, 0.975]))
    base.p_better = float((arr > 0).mean())
    base.resamples_used = len(arr)
    return base


# --------------------------------------------------------------------------- #
# 데이터셋 하나
# --------------------------------------------------------------------------- #


@dataclass
class DatasetVerdict:
    dataset: str
    metric: str
    task: str
    verdict: str = "ok"
    refusal: str | None = None
    test_rows: int = 0
    test_fingerprint: str | None = None
    arms: dict[str, dict[str, Any]] = field(default_factory=dict)
    pairs: list[Delta] = field(default_factory=list)
    missing_arms: list[str] = field(default_factory=list)


def adjudicate(
    dataset: Dataset, seed: int, resamples: int, legs: dict[str, Scored] | None = None
) -> DatasetVerdict:
    out = DatasetVerdict(dataset=dataset.name, metric=dataset.metric, task=dataset.task)
    try:
        _adjudicate(dataset, seed, resamples, out, legs)
    except Refusal as exc:
        out.verdict = "refused"
        out.refusal = str(exc)
        out.pairs = []
    return out


def _adjudicate(
    dataset: Dataset,
    seed: int,
    resamples: int,
    out: DatasetVerdict,
    legs: dict[str, Scored] | None = None,
) -> None:
    metric = dataset.metric
    loops: dict[str, Winner] = {}
    for arm, prefix in (("llm", "llm"), ("no_llm", "nollm")):
        winner = loop_winner(arm, ARTIFACTS_DIR / f"{prefix}-{dataset.name}-seed{seed}", metric)
        if winner is None:
            out.missing_arms.append(arm)
        else:
            loops[arm] = winner

    random_dir = random_run_dir(dataset, seed)
    has_random = (random_dir / "summary.json").exists()
    if not has_random:
        out.missing_arms.append("random")
    if not loops and not has_random:
        raise Refusal(f"{dataset.name}: 시드 {seed}의 실행을 하나도 찾지 못했습니다")

    # 필요한 random 앞자리: 루프 팔마다 자기 학습 횟수 하나(예산 맞춘 primary)와 5(best-of-5 보조).
    # 두 팔의 k가 우연히 같을 때 두 번 재현하지 않도록 dict를 공유한다.
    winners: dict[str, Winner] = dict(loops)
    if has_random:
        needed = {5, *(w.fits for w in loops.values())}
        for k in sorted(needed):
            if k < 1:
                raise Refusal(f"{dataset.name}: 루프 팔의 학습 횟수가 {k}회로 기록됐습니다")
            winner = random_winner(random_dir, k, metric)
            winners[winner.arm] = winner

    scored: dict[str, Scored] = {}
    for name, winner in winners.items():
        result = reproduce(winner, metric)
        gap = abs(result.test_score - winner.recorded_test)
        if gap > SCORE_TOLERANCE:
            raise Refusal(
                f"{name}: 다시 유도한 test 점수 {result.test_score:.6f}가 기록된 "
                f"{winner.recorded_test:.6f}와 {gap:.2e} 다릅니다 — 같은 행이 아닙니다"
            )
        scored[name] = result

    fingerprints = {name: s.fingerprint for name, s in scored.items()}
    if len(set(fingerprints.values())) > 1:
        listing = ", ".join(f"{name}={fp[:12]}" for name, fp in sorted(fingerprints.items()))
        raise Refusal(f"{dataset.name}: 팔마다 test 지문이 다릅니다 ({listing}) — 판정하지 않습니다")
    identities = {name: json.dumps(split_identity(w.config), sort_keys=True) for name, w in winners.items()}
    if len(set(identities.values())) > 1:
        raise Refusal(
            f"{dataset.name}: 분할을 정하는 config 필드가 팔마다 다릅니다 "
            f"({sorted(set(identities.values()))})"
        )

    any_scored = next(iter(scored.values()))
    out.test_rows = any_scored.n_rows
    out.test_fingerprint = any_scored.fingerprint
    for name, s in scored.items():
        out.arms[name] = {
            "label": s.winner.label,
            "fits": s.winner.fits,
            "val_score": s.winner.val_score,
            "test_recorded": s.winner.recorded_test,
            "test_reproduced": s.test_score,
            # val 최고에서 test를 뺀 값: 같은 validation 행에서 여러 시도 중 최고를 고르는 값.
            # 양수면 validation이 우승자를 실제보다 좋게 보여 준 것이다.
            "selection_gap": s.winner.val_score - s.test_score,
            "dir": s.winner.directory.as_posix(),
        }

    pairs: list[tuple[str, str, str]] = []
    for arm in ("llm", "no_llm"):
        if arm in scored and has_random:
            pairs.append((arm, f"random@k={scored[arm].winner.fits}", "primary"))
            if scored[arm].winner.fits != 5:
                pairs.append((arm, "random@k=5", "best_of_5"))
    if "llm" in scored and "no_llm" in scored:
        pairs.append(("llm", "no_llm", "primary"))

    for a_name, b_name, kind in pairs:
        out.pairs.append(
            paired_delta(scored[a_name], scored[b_name], metric, resamples, seed, kind)
        )

    # 중간에 거부되면 leg가 남지 않도록 마지막에 한다. 짝이 지워진 데이터셋의 leg를 담은 번들은
    # 판정이 일부러 공표하지 않은 수를 ``recheck``에 검증거리로 내주게 된다.
    if legs is not None:
        legs.update(scored)


# --------------------------------------------------------------------------- #
# 사전 등록된 기준
# --------------------------------------------------------------------------- #


def criterion(
    results: list[DatasetVerdict], challenger: str = "llm", seed: int = JUDGED_SEED
) -> dict[str, Any]:
    """어느 실행보다도 먼저 ``docs/RESULTS.md``에 커밋된 성공 기준을 평가한다.

    > 이진 4개 중 3개 이상에서, `llm` 팔이 `random` 팔을 짝지은 Δ의 95% CI가 0을 걸치지 않게
    > 이긴다.

    이진만, primary(예산 맞춘) 비교만, 그리고 판정 시드에서만 평가한다. 시드 제한을 독자에게 맡기지 않고
    여기서 강제한다: 시드 43/44는 잡음 바닥을 설명하러 있고, 사전 등록이 "짝지은 비교가 존재하는 시드는
    42뿐"이라고 못박은 것은 나중 실행이 답이 제일 예쁘게 나온 시드를 고를 수 없게 하려는 것이다. 다른
    시드의 실행은 같은 표를 내고 판정은 내지 않는다.

    "4개 중 3개"는 *고정된* 넷에 대한 셈이므로, 그보다 적게 덮은 실행은 기준에 닿을 수조차 없고 그
    미달을 "구분되지 않음"으로 보고해서는 안 된다 — 그 표현은 답처럼 읽히는데 실제로 일어난 일은 질문을
    안 한 것이다. 부분 범위는 부분이라 이름 붙이고 어느 데이터셋이 없는지 적는다.
    """
    # 넷은 회귀를 제외해서 찾는 대신 ``datasets.JUDGED_NAMES``에 이름으로 적혀 있다. 그 제외는
    # ``ALL``에 task 타입이 둘일 때는 같은 결과였고 셋이 된 순간 틀렸다: 다중분류가 이 분모에 끼어들어
    # "4개 중 3개"로 커밋된 기준을 아무도 문서를 고치지 않은 채 "5개 중 3개"로 만들었을 것이다.
    expected = set(JUDGED_NAMES)
    binary = [r for r in results if r.dataset in expected]
    wins, losses, ties, absent = [], [], [], []
    for result in binary:
        pair = next(
            (
                p for p in result.pairs
                if p.a == challenger and p.kind == "primary" and p.b.startswith("random")
            ),
            None,
        )
        if result.verdict != "ok" or pair is None:
            absent.append(result.dataset)
        elif pair.ci_low > 0:
            wins.append(result.dataset)
        elif pair.ci_high < 0:
            losses.append(result.dataset)
        else:
            ties.append(result.dataset)
    missing = sorted(expected - {r.dataset for r in binary} - set(absent)) + sorted(absent)
    tally = f"이김 {len(wins)} / 걸침 {len(ties)} / 짐 {len(losses)}"
    if seed != JUDGED_SEED:
        complete = False
        status = (
            f"기준 판정 안 함 — 사전 등록은 시드 {JUDGED_SEED}만 판정에 씁니다. "
            f"시드 {seed}는 잡음 바닥용입니다 ({tally})"
        )
    elif missing:
        complete = False
        status = (
            f"부분 판정 — 기준은 이진 {len(expected)}개에 대한 것인데 "
            f"{len(wins) + len(ties) + len(losses)}개만 판정했습니다 "
            f"(없음: {', '.join(missing)}). 지금까지 {tally}"
        )
    else:
        complete = True
        if len(wins) >= 3:
            status = (
                f"기준 충족 — 이 {len(expected)}개 중 {len(wins)}개에서 "
                f"{challenger} 팔이 random search를 이겼습니다"
            )
        elif len(losses) >= 3:
            status = f"random search가 이겼습니다 — {len(losses)}개에서 CI가 0 아래입니다 ({tally})"
        else:
            status = f"구분되지 않음 — 이 데이터셋들과 이 예산으로는 답이 나오지 않습니다 ({tally})"
    return {
        "challenger": challenger,
        "seed": seed,
        "judged_seed": JUDGED_SEED,
        "expected_binary": sorted(expected),
        "judged": [r.dataset for r in binary if r.dataset not in absent],
        "wins": wins,
        "losses": losses,
        "indistinguishable": ties,
        "not_evaluated": missing,
        "complete": complete,
        "status": status,
    }


# --------------------------------------------------------------------------- #
# 출력
# --------------------------------------------------------------------------- #


def report(results: list[DatasetVerdict]) -> None:
    for result in results:
        print()
        if result.verdict != "ok":
            print(f"== {result.dataset} — 판정 거부")
            print(f"   {result.refusal}")
            continue
        print(
            f"== {result.dataset} ({result.metric}) · test {result.test_rows}행 "
            f"· 지문 {result.test_fingerprint}"
        )
        if result.missing_arms:
            print(f"   없는 팔: {', '.join(result.missing_arms)}")
        print(f"   {'팔':<14} {'우승':<14} {'학습':>4} {'val':>9} {'test':>9} {'선택편향':>9}")
        for name, arm in result.arms.items():
            print(
                f"   {name:<14} {arm['label']:<14} {arm['fits']:>4} "
                f"{arm['val_score']:>9.4f} {arm['test_reproduced']:>9.4f} "
                f"{arm['selection_gap']:>+9.4f}"
            )
        if not result.pairs:
            print("   짝지은 비교 없음 (한 팔뿐)")
            continue
        print()
        print(f"   {'비교':<26} {'종류':<10} {'Δ':>9}  {'95% CI of Δ':<22} {'P(Δ>0)':>7} {'재추출':>7}")
        for pair in result.pairs:
            label = f"{pair.a} − {pair.b}"
            if pair.identical_predictions:
                print(
                    f"   {label:<26} {pair.kind:<10} {pair.delta:>+9.4f}  "
                    f"{'예측이 동일':<22} {'—':>7} {'—':>7}"
                )
                continue
            ci = f"[{pair.ci_low:+.4f}, {pair.ci_high:+.4f}]"
            crosses = "" if (pair.ci_low > 0 or pair.ci_high < 0) else "  (0을 걸침)"
            print(
                f"   {label:<26} {pair.kind:<10} {pair.delta:>+9.4f}  {ci:<22} "
                f"{pair.p_better:>7.3f} {pair.resamples_used:>7}{crosses}"
            )


def payload(results: list[DatasetVerdict], seed: int, resamples: int) -> dict[str, Any]:
    return {
        "generated_by": "bench/paired.py",
        "seed": seed,
        "resamples": resamples,
        "score_tolerance": SCORE_TOLERANCE,
        # 환경 변수만으로는 정해지지 않는 ``cpu_count``까지 포함한다: 셋 다 안 잡힌 상태가
        # 4코어와 64코어에서 같은 산술이 아니다.
        "threads": thread_state(),
        "datasets": [
            {
                "dataset": r.dataset,
                "metric": r.metric,
                "task": r.task,
                "verdict": r.verdict,
                "refusal": r.refusal,
                "test_rows": r.test_rows,
                "test_fingerprint": r.test_fingerprint,
                "missing_arms": r.missing_arms,
                "arms": r.arms,
                "pairs": [vars(p) for p in r.pairs],
            }
            for r in results
        ],
        "criterion": criterion(results, "llm", seed),
        "criterion_no_llm": criterion(results, "no_llm", seed),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="3-arm 벤치마크의 짝지은 판정 (docs/RESULTS.md의 사전 등록 기준)"
    )
    parser.add_argument("datasets", nargs="*", default=[], help="이름 (기본: 전부)")
    parser.add_argument("--seed", type=int, default=JUDGED_SEED)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--out", type=Path, default=None, help="기본: bench/runs/paired/seed<seed>.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    # 미룬 import: bench/predictions.py가 ``Scored``/``Winner``를 만드느라 이 모듈을 import한다.
    # 모듈 수준에서 되돌려 import하면 순환이 된다. 여기 두는 것은 사실 하나도 말해 준다 — 번들은
    # 판정기를 돌린 산물이고 채점 함수들의 산물이 아니다.
    from bench.predictions import bundle_path, write_bundle

    args = parse_args(argv)
    names = args.datasets or [d.name for d in ALL]
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        print(f"모르는 데이터셋: {', '.join(unknown)}", file=sys.stderr)
        return 2

    print(
        f"짝지은 판정 · 시드 {args.seed} · 재추출 {args.resamples}회 · "
        f"{describe_thread_state(thread_state())}"
    )

    legs: dict[str, dict[str, Scored]] = {}
    results = []
    for name in names:
        collected: dict[str, Scored] = {}
        results.append(adjudicate(BY_NAME[name], args.seed, args.resamples, collected))
        if collected:
            legs[name] = collected
    report(results)

    out_path = args.out or OUT_DIR / f"seed{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload(results, args.seed, args.resamples), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if legs:
        target = bundle_path(out_path)
        write_bundle(target, legs, {r.dataset: r.metric for r in results})
        print(f"예측 번들 → {target.as_posix()} ({target.stat().st_size / 1024:.0f} KB)")
    print()
    print(f"기준 (llm): {criterion(results, 'llm', args.seed)['status']}")
    print(f"기준 (no_llm): {criterion(results, 'no_llm', args.seed)['status']}")
    print(f"판정 → {out_path.as_posix()}")
    refused = [r.dataset for r in results if r.verdict != "ok"]
    if refused:
        print(f"거부된 데이터셋: {', '.join(refused)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
