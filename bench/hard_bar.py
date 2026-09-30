"""올린 바가 루프를 실제로 돌게 만들면, 루프 팔이 규칙 폴백을 이기는가?

``docs/RESULTS.md``와 ``docs/PROMPT-REDO.md``가 같은 2차 판정에서 함께 떨어졌다 — 서로 다른 두
프롬프트에서 ``llm − no_llm``이 모자랐다. 두 측정 다 학습 횟수가 1/1/2/1/1/1이었다: 여섯 실행 중
다섯이 첫 시도에 ``goal_reached``에 닿았으니 ``critic``과 재계획 경로는 돈 적이 없다. 그 수들이 잰
것은 LLM이 *첫* 하이퍼파라미터 세트를 더 잘 고르는가이고, 진단과 재계획이 무언가를 사 오는가가
아니다.

``docs/REPLAN-SEEDS.md``는 이 저장소에서 루프 팔의 이득이 세 시드와 크기 바를 견딘 유일한 곳이고,
거기 닿은 방법은 목표를 올려 재계획이 일어나야 하게 만든 것이다. 데이터셋 하나에서 했다. 이 모듈은
그 조건을 판정 분모인 이진 4개로 넓히고 같은 세 시드에서 반복한다.

사전 등록은 ``docs/HARD-BAR.md``이고, 이 실행들이 존재하기 전에 커밋됐다.

레버 하나. ``--margin``만 0.25에서 0.65로 가고 나머지 플래그는 전부
``bench/scripts/run_prompt_redo.sh``가 쓴 것과 같다. 그것이 각 바를 무엇으로 만들 예정인지는
:data:`PREDICTED_BARS`에 적혀 있다 — ``goal.py``의 식과 기록된 실행들 자신의 문턱에서 유도했으므로,
바가 다른 곳에서 나온 실행은 흡수되지 않고 걸린다.

``REPLAN-SEEDS``가 쓴 0.5가 아니라 0.65인 이유: ``12fa74b``가 이 바들 중 둘을 순위 바닥에 눌러
앉히므로 ``--margin 0.5``는 bank-marketing에서 무력하고(``floored_margin``이 0.526), 0.6은 그 바닥을
넘지만 기록된 팔이 이미 닿은 첫 시도 검증 점수 아래에 앉는다. 0.65는 네 바를 전부 기록된 모든 첫
시도 위로 올린다.

random 팔은 다시 뽑지 않는다. ``bench/random_search.py``는 목표를 읽지 않으므로 바를 올려도 뽑기
하나조차 움직일 수 없고, (데이터셋, 시드)마다 기록된 서른 번의 뽑기를 재사용한다. 맞춘 예산은
가정하지 않고 각 실행 자신의 ``fits``에서 읽는다: 닿을 수 없는 바는 루프를 *돌게* 만들지만, 다섯
번을 다 쓰게 만들지는 않는다. 이 저장소에서 닿을 수 없는 바를 앞에 두고 5회 예산을 받은 실행은
``REPLAN-SEEDS.md``의 셋뿐이고, ``STALL_LIMIT``이 2라서 전부 ``critic_runs`` 3에서 ``stalled``로
끝났다. 그래서 두 팔이 서로 다른 학습 횟수에서 멈출 수 있고, 그러면 두 ``matched_level`` 쌍이 다른
k에 앉는다. 예산 표가 그것을 감추지 않고 적는다. 주 쌍은 팔 대 팔이라 영향을 받지 않는다.

루프 팔 둘은 ``--keep-models all``로 돌려야 한다: 관찰이 각 실행의 iteration 1을 다시 채점하는데,
기본값은 우승하지 않은 iteration의 ``model.joblib``을 전부 지운다.

팔들이 돌았던 것처럼 ``OMP_NUM_THREADS=1``로 돌린다. 값은 출력에 기록된다.

사용법:
    python -m bench.hard_bar                        # 네 데이터셋 × 세 시드
    python -m bench.hard_bar adult --seeds 42
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

from automl_agent.threads import describe_thread_state, thread_state
from bench.datasets import BY_NAME, JUDGED, JUDGED_SEED
from bench.paired import (
    ARTIFACTS_DIR,
    OUT_DIR,
    RESAMPLES,
    SCORE_TOLERANCE,
    DatasetVerdict,
    Delta,
    Refusal,
    Scored,
    Winner,
    loop_winner,
    paired_delta,
    random_winner,
    report,
    reproduce,
    split_identity,
)
from bench.predictions import bundle_path, write_bundle
from bench.random_search import run_dir as random_run_dir
from bench.replan import first_iteration_winner, run_shape

# 판정 분모인 이진 4개, ``bench/datasets.py`` 순서. 다시 나열하지 않고 ``JUDGED``에서 받는다 —
# "이진 4개 중 3개"의 분모는 그 모듈이 못박고 있고, 자기 사본을 적은 기준은 다른 문서들이 세는
# 분모와 어긋날 수 있다.
DATASETS: tuple[str, ...] = tuple(item.name for item in JUDGED)

# 무료 팔들이 늘 반복돼 온 세 시드. 범위가 사후에 넓어질 수 없게 명령줄이 아니라 여기서
# 못박는다.
SEEDS: tuple[int, ...] = (42, 43, 44)

# ``bench/scripts/run_prompt_redo.sh``와 다른 유일한 플래그.
MARGIN = 0.65

# 이 실험이 돌리는 팔. 숫자 접미사가 아니라 ``bar65``인 것은 조건이 요점이기 때문이다.
# ``llm2-adult-seed42``는 ``PROMPT-REDO.md``의 기록된 산출물이라 여기서 아무것도 써 넣으면 안 되고,
# 아카이브 디렉터리는 그 실행이 어떤 바를 마주했는지 말해야 한다.
ARMS: tuple[tuple[str, str], ...] = (("bar65llm", "bar65llm"), ("bar65nollm", "bar65nollm"))

# ``PROMPT-REDO.md``의 팔. 판정 시드에서만 다리로 다시 채점한다 — 그 팔들이 존재하는 시드가
# 그것뿐이다. 각각이 ``bar_shift`` 쌍 하나를 주고, 그 두 다리는 ``--margin``만 다르다: 같은
# 프롬프트, 같은 코드, 같은 카드, 같은 시드. 그 두 쌍이 재는 비대칭이, 주 Δ를 "LLM의 기여"로 혼자
# 읽을 수 없는 이유다.
RECORDED: tuple[tuple[str, str], ...] = (("llm2", "llm2"), ("no_llm2", "nollm2"))
RECORDED_SEEDS: tuple[int, ...] = (JUDGED_SEED,)

# 기준이 한 데이터셋의 세 시드 중 가장 작은 이김에 대고 재는 크기 바. ``REPEATS.md``에서 왔다 —
# ``llm`` 팔의 같은 설정 반복이 낸 반폭의 중앙값이다.
#
# *수입한* 상수이고 판정 파일도 그렇게 적는다. 그 측정은 낡은 프롬프트·낡은 바에서 쟀고, 이 실험은
# 바닥을 다시 재지 않는다 — 그건 그것 자체로 하나의 실험이다. 대신 판정마다 함께 내놓는 것은 이
# 바에서 규칙 팔 자신의 시드 산포이고, 그건 이 실험이 실제로 잰다. ``REPLAN-SEEDS.md``가
# ``RULES_SEED_SPREAD``를 인용한 것과 같은 조건이다.
NOISE_FLOOR = 0.0122
NOISE_FLOOR_SOURCE = "docs/REPEATS.md (낡은 프롬프트·낡은 바에서 잰 수입니다)"

# :data:`MARGIN`에서 커밋된 카드에 ``goal.derive_threshold``를 돌리면 나오는 값. (데이터셋, 시드)
# 마다 박는 이유는 기준선이 시드마다 자기 분할에서 계산되고 바가 그것과 함께 움직이기 때문이다 —
# bank-marketing의 ``floored_margin``은 세 시드에서 0.526 / 0.531 / 0.523이고, speeddating의 바닥은
# 시드 43에서는 아예 적용되지 않는다. 여기 값들은 모두 모든 ``floored_margin`` 위이므로 0.65에서는
# ``12fa74b``의 순위 바닥이 열두 카드 전부에서 무력하고, 바를 유도하는 것은 그냥 식이다.
#
# import 때 유도하지 않고 박아 둔다 — 오늘의 카드에서 자기 예측을 다시 계산하는 사전 등록은
# 아무것도 예측하지 않는다. 커밋된 카드와 ``goal.py``가 여전히 이 값을 내는지는
# ``tests/test_bench_hard_bar.py``가 확인하므로, 카드를 고치거나 유도를 바꾸면 바가 조용히 움직이는
# 대신 테스트가 깨진다. 실행들이 실제로 그 바를 마주했는지는 :func:`bar_check`가 본다.
PREDICTED_BARS: dict[tuple[str, int], float] = {
    ("adult", 42): 0.9182,
    ("adult", 43): 0.9194,
    ("adult", 44): 0.9189,
    ("bank-marketing", 42): 0.8817,
    ("bank-marketing", 43): 0.8816,
    ("bank-marketing", 44): 0.8811,
    ("speeddating", 42): 0.8840,
    ("speeddating", 43): 0.8879,
    ("speeddating", 44): 0.8882,
    ("spambase", 42): 0.9730,
    ("spambase", 43): 0.9711,
    ("spambase", 44): 0.9670,
}

# 기록된 ``llm2`` 팔이 첫 시도에 닿은 검증 점수. 그 팔이 존재하는 한 시드에서다. 0.6이 아니라
# 0.65인 이유가 이것이다 — 실행이 반복할 이유를 가지려면 바가 이 수 위에 앉아야 하는데, 0.6에서
# bank-marketing의 바는 0.8648이고 이미 닿은 점수가 0.8705다. 시드 43·44에는 대응하는 것이 없다
# (거기서는 기록된 팔이 돈 적이 없다). 그래서 전제의 이 절반은 시드 42에서만 확인하고 다른 어디에서도
# 주장하지 않는다.
RECORDED_FIRST_VAL: dict[str, float] = {
    "adult": 0.8460,
    "bank-marketing": 0.8705,
    "speeddating": 0.7878,
    "spambase": 0.9534,
}

# 실측 바가 예측에서 얼마나 떨어지면 어긋남이라고 부르는가. 예측은 두 문서에서 소수 네 자리로
# 반올림된 수에 대한 산술이라 정확할 수 없다. 이보다 큰 불일치는 반올림이 아니라 다른 기준선이다.
BAR_TOLERANCE = 5e-4

# ``bench/replan_seeds.py`` 참고: 재추출 시드는 실행 시드가 아니다. 파일 안의 모든 쌍이 한
# 수열에서 계산되게, 그리고 판정 파일마다 ``seed`` 하나를 읽는 ``bench/recheck.py``가 전부 다시
# 계산할 수 있게 박아 둔다.
RESAMPLE_SEED = JUDGED_SEED

PRIMARY: tuple[str, str] = ("bar65llm", "bar65nollm")


def _key(dataset: str, seed: int) -> str:
    """판정 항목의 이름. 양쪽이 다 움직이므로 둘 다 들어간다."""
    return f"{dataset}-seed{seed}"


def _arm_legs(
    dataset: str, seed: int, metric: str, out: DatasetVerdict
) -> tuple[dict[str, Winner], dict[str, Any]]:
    """새 팔 둘의 우승자, 각각의 iteration 1 다리, 그리고 각 실행의 모양."""
    winners: dict[str, Winner] = {}
    shapes: dict[str, dict[str, Any]] = {}
    for arm, prefix in ARMS:
        directory = ARTIFACTS_DIR / f"{prefix}-{dataset}-seed{seed}"
        winner = loop_winner(arm, directory, metric)
        if winner is None:
            out.missing_arms.append(arm)
            continue
        winners[arm] = winner
        winners[f"{arm}@it1"] = first_iteration_winner(arm, directory, metric)
        shapes[arm] = run_shape(directory)
    return winners, shapes


def _recorded_legs(
    dataset: str, seed: int, metric: str, out: DatasetVerdict
) -> tuple[dict[str, Winner], dict[str, Any]]:
    """판정 시드에서의 ``PROMPT-REDO.md`` 팔. ``bar_shift`` 쌍 둘에 쓴다."""
    winners: dict[str, Winner] = {}
    shapes: dict[str, dict[str, Any]] = {}
    if seed not in RECORDED_SEEDS:
        return winners, shapes
    for arm, prefix in RECORDED:
        directory = ARTIFACTS_DIR / f"{prefix}-{dataset}-seed{seed}"
        winner = loop_winner(arm, directory, metric)
        if winner is None:
            out.missing_arms.append(arm)
            continue
        winners[arm] = winner
        shapes[arm] = run_shape(directory)
    return winners, shapes


def _pairs_to_take(scored: dict[str, Scored], fits: int) -> list[tuple[str, str, str]]:
    """이 (데이터셋, 시드)가 받칠 수 있는 비교. ``HARD-BAR.md``가 나열한 순서다."""
    pairs: list[tuple[str, str, str]] = []
    if PRIMARY[1] in scored:
        pairs.append((*PRIMARY, "primary"))
    matched = f"random@k={fits}"
    if matched in scored:
        pairs.append(("bar65llm", matched, "matched_level"))
        if PRIMARY[1] in scored:
            pairs.append(("bar65nollm", matched, "rules_matched_level"))
    for arm, _ in ARMS:
        if arm in scored and f"{arm}@it1" in scored:
            # 반복이 무언가를 사 왔는가? ``REPLAN.md``의 질문을, 하나가 아니라 네
            # 데이터셋에서 두 팔 모두에 묻는다.
            pairs.append((arm, f"{arm}@it1", f"{arm}_replan_gain"))
    # 각각 레버 하나: 같은 프롬프트, 같은 코드, 같은 카드, 같은 시드, ``--margin``만 다르다.
    if "llm2" in scored:
        pairs.append(("bar65llm", "llm2", "bar_shift"))
    if "no_llm2" in scored:
        pairs.append(("bar65nollm", "no_llm2", "rules_bar_shift"))
    return pairs


def _adjudicate(
    dataset: str, seed: int, resamples: int, out: DatasetVerdict, legs: dict[str, Scored] | None
) -> None:
    entry = BY_NAME[dataset]
    metric = entry.metric
    winners, shapes = _arm_legs(dataset, seed, metric, out)
    if "bar65llm" not in winners:
        # 데이터에 대한 거부가 아니다. 이 기준이 말하는 실행이 아직 일어나지 않았다.
        raise Refusal(
            f"{dataset} 시드 {seed}: bar65llm 실행이 없습니다 — "
            "bench/scripts/run_hard_bar.sh를 먼저 돌려야 판정할 수 있습니다"
        )
    recorded, recorded_shapes = _recorded_legs(dataset, seed, metric, out)
    winners.update(recorded)
    shapes.update(recorded_shapes)

    random_dir = random_run_dir(entry, seed)
    if (random_dir / "summary.json").exists():
        fits = winners["bar65llm"].fits
        if fits < 1:
            raise Refusal(f"{dataset} 시드 {seed}: bar65llm의 학습 횟수가 {fits}회로 기록됐습니다")
        winners[f"random@k={fits}"] = random_winner(random_dir, fits, metric)
    else:
        out.missing_arms.append("random")

    scored: dict[str, Scored] = {}
    for name, winner in winners.items():
        result = reproduce(winner, metric)
        # iteration 1 다리는 test 점수가 기록된 적이 없어 nan이다 — holdout은 루프가 끝난 뒤
        # 우승자에 대해 한 번 돈다. 이유는 ``bench/replan.py``에 있다.
        if not math.isnan(winner.recorded_test):
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
        raise Refusal(
            f"{dataset} 시드 {seed}: 팔마다 test 지문이 다릅니다 ({listing}) — 판정하지 않습니다"
        )
    identities = {
        name: json.dumps(split_identity(w.config), sort_keys=True) for name, w in winners.items()
    }
    if len(set(identities.values())) > 1:
        raise Refusal(
            f"{dataset} 시드 {seed}: 분할을 정하는 config 필드가 팔마다 다릅니다 "
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
            "selection_gap": s.winner.val_score - s.test_score,
            "dir": s.winner.directory.as_posix(),
            **shapes.get(name, {}),
        }

    for a_name, b_name, kind in _pairs_to_take(scored, winners["bar65llm"].fits):
        out.pairs.append(
            paired_delta(scored[a_name], scored[b_name], metric, resamples, RESAMPLE_SEED, kind)
        )

    # 마지막에 둔다. 중간에 거부되면 다리가 남지 않는다.
    if legs is not None:
        legs.update(scored)


def adjudicate(
    dataset: str, seed: int, resamples: int, legs: dict[str, Scored] | None = None
) -> DatasetVerdict:
    """(데이터셋, 시드) 하나. 거부는 ``paired.py``처럼 쌍을 비운다."""
    entry = BY_NAME[dataset]
    out = DatasetVerdict(dataset=_key(dataset, seed), metric=entry.metric, task=entry.task)
    try:
        _adjudicate(dataset, seed, resamples, out, legs)
    except Refusal as exc:
        out.verdict = "refused"
        out.refusal = str(exc)
        out.pairs = []
    return out


# --------------------------------------------------------------------------- #
# 바가 사전 등록이 말한 곳에 앉았는가
# --------------------------------------------------------------------------- #


def bar_check(results: dict[tuple[str, int], DatasetVerdict]) -> list[dict[str, Any]]:
    """각 실행의 실제 ``goal.threshold``를 :data:`PREDICTED_BARS`와 대조한다.

    0.65에 대한 사전 등록의 논거 전체가 두 문서의 반올림된 수에 대한 산술이고, 거기 적어 둔 중단
    조건이 "바가 예측 표와 다르면 유료 절반을 돌리기 전에 멈춘다"다. 이것이 그 확인이다. 판정 파일에
    계산해 넣으므로 읽는 사람이 손으로 다시 할 필요가 없다.
    """
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            result = results.get((dataset, seed))
            if result is None:
                continue
            predicted = PREDICTED_BARS[(dataset, seed)]
            # 대조할 기록된 첫 시도가 있는 것은 시드 42뿐이다.
            first_val = RECORDED_FIRST_VAL[dataset] if seed == JUDGED_SEED else None
            for arm, _ in ARMS:
                actual = result.arms.get(arm, {}).get("goal_threshold")
                # 실행이 문턱을 기록하지 않았으면 ``None``. 일치로 읽히면 안 된다.
                known: float | None = (
                    float(actual)
                    if isinstance(actual, (int, float)) and not isinstance(actual, bool)
                    else None
                )
                off = abs(known - predicted) if known is not None else None
                rows.append(
                    {
                        "dataset": dataset,
                        "seed": seed,
                        "arm": arm,
                        "predicted_bar": predicted,
                        "actual_bar": actual,
                        "off_by": off,
                        "matches": off is not None and off <= BAR_TOLERANCE,
                        # 이 아래면 실행이 반복할 이유가 없었다. 그게 전제다.
                        "above_recorded_first_val": (
                            None
                            if first_val is None or known is None
                            else known > first_val
                        ),
                    }
                )
    return rows


# --------------------------------------------------------------------------- #
# 사전 등록된 기준
# --------------------------------------------------------------------------- #


def _primary(result: DatasetVerdict) -> Delta | None:
    a, b = PRIMARY
    return next((p for p in result.pairs if p.a == a and p.b == b and p.kind == "primary"), None)


def _per_dataset(
    dataset: str, results: dict[tuple[str, int], DatasetVerdict]
) -> dict[str, Any]:
    """데이터셋 하나의 세 시드를, 기준이 세는 판정으로 줄인 것."""
    judged: list[str] = []
    above: list[str] = []
    below: list[str] = []
    spans: list[str] = []
    not_looped: list[str] = []
    absent: list[str] = []
    deltas: dict[str, float] = {}
    for seed in SEEDS:
        label = f"seed{seed}"
        result = results.get((dataset, seed))
        pair = None if result is None else _primary(result)
        if result is None or result.verdict != "ok" or pair is None:
            absent.append(label)
            continue
        critic_runs = result.arms.get("bar65llm", {}).get("critic_runs")
        if not isinstance(critic_runs, int) or critic_runs < 1:
            # 이김이 아닌 것이 아니라 전제다. 재계획한 적 없는 루프에 재계획의 값을 물리면
            # 깨진 조건이 증거로 읽히게 된다.
            not_looped.append(label)
            continue
        judged.append(label)
        if pair.identical_predictions:
            spans.append(label)
            continue
        deltas[label] = pair.delta
        if pair.ci_low > 0:
            above.append(label)
        elif pair.ci_high < 0:
            below.append(label)
        else:
            spans.append(label)

    win_deltas = [deltas[label] for label in above]
    smallest = min(win_deltas) if win_deltas else float("nan")
    unanimous = len(judged) == len(SEEDS) and len(above) == len(SEEDS)
    wins = unanimous and smallest > NOISE_FLOOR
    return {
        "dataset": dataset,
        "judged": judged,
        "above_zero": above,
        "below_zero": below,
        "spans_zero": spans,
        "not_looped": not_looped,
        "not_evaluated": absent,
        "deltas": deltas,
        "smallest_win": smallest,
        "unanimous": unanimous,
        "wins": wins,
        "loses": bool(below),
        "spread": (max(deltas.values()) - min(deltas.values())) if len(deltas) > 1 else float("nan"),
        "median": statistics.median(deltas.values()) if deltas else float("nan"),
    }


def rules_seed_spread(results: dict[tuple[str, int], DatasetVerdict]) -> dict[str, float]:
    """데이터셋마다, 세 시드에 걸친 규칙 팔 자신의 test 산포.

    기준이 쓰는 크기 바가 다른 조건에서 수입된 것이라(:data:`NOISE_FLOOR_SOURCE`) 판정마다 함께
    내놓는다. 이 수는 여기서 *직접* 잰다. 같은 카드에서 무료 팔의 시드 산포보다 작은 주 Δ는 구간이
    무엇을 말하든 증거가 아니다.
    """
    out: dict[str, float] = {}
    for dataset in DATASETS:
        scores = [
            result.arms["bar65nollm"]["test_reproduced"]
            for seed in SEEDS
            if (result := results.get((dataset, seed))) is not None
            and isinstance(result.arms.get("bar65nollm", {}).get("test_reproduced"), (int, float))
        ]
        out[dataset] = (max(scores) - min(scores)) if len(scores) > 1 else float("nan")
    return out


def criterion(results: dict[tuple[str, int], DatasetVerdict]) -> dict[str, Any]:
    """``docs/HARD-BAR.md``에 이 실행들보다 먼저 커밋된 기준을 판정한다.

    > **데이터셋 하나가 "이김"인 조건**: 시드 42·43·44 **전부**에서 ``bar65llm``의 우승자가
    > ``bar65nollm``의 우승자를 같은 test 행에 대고 짝지은 Δ의 95% CI가 0을 걸치지 않게 이기고,
    > 그 세 Δ의 **최솟값이 0.0122를 넘는다.**
    >
    > **기준 충족**: 그런 데이터셋이 **이진 4개 중 3개 이상.**

    과반이 아니라 시드 전원 일치를 요구하는 것은, 이 저장소가 바로 거기서 두 번 걸렸기 때문이다:
    ``REPEATS.md``의 R2가 *같은* 설정의 반복에서 0을 벗어난 짝지은 Δ를 12쌍 중 4쌍에서 냈다. 그래서
    한 시드의 구간은 처치의 증거가 아니다. n=3에서 셋 중 둘 규칙은 동전과 구분하기 어렵다.

    4개 중 3개라는 집계는 ``RESULTS.md``의 분모를 다시 고르지 않고 그대로 받은 것이다. 이 실험은
    자기 없이 정해진 바를 넓히거나 좁히지 않는다.

    어떤 결과도 자기에게 유리한 독법을 고를 수 없게 우선순위를 미리 정해 둔다: 없거나 거부된
    (데이터셋, 시드)가 있으면 부분 판정. 루프가 실제로 재계획한 칸이 여섯 미만이면 "판정 불가"인데,
    그건 잰 null이 아니라 깨진 전제다. 어느 시드가 0 아래인 데이터셋은 상태 문구에 이름을 적는다.
    그다음이 4개 중 3개 셈이고, 그 밖은 "구분되지 않음"이다.
    """
    per = {dataset: _per_dataset(dataset, results) for dataset in DATASETS}
    absent = [
        f"{dataset}-{label}" for dataset in DATASETS for label in per[dataset]["not_evaluated"]
    ]
    not_looped = [
        f"{dataset}-{label}" for dataset in DATASETS for label in per[dataset]["not_looped"]
    ]
    looped = sum(len(per[dataset]["judged"]) for dataset in DATASETS)
    wins = [dataset for dataset in DATASETS if per[dataset]["wins"]]
    losses = [dataset for dataset in DATASETS if per[dataset]["loses"]]
    others = [d for d in DATASETS if d not in wins and d not in losses]
    tally = f"이김 {len(wins)} / 그 밖 {len(others)} / 짐 {len(losses)}"
    cells = len(DATASETS) * len(SEEDS)

    if absent:
        complete = False
        status = (
            f"부분 판정 — 기준의 범위는 (데이터셋, 시드) {cells}칸인데 "
            f"{looped}칸만 판정했습니다 (없음: {', '.join(absent)}). 지금까지 {tally}"
        )
    elif looped < 6:
        complete = False
        status = (
            f"판정 불가 — 루프가 재계획한 칸이 {looped}개뿐입니다 "
            f"(안 돈 것: {', '.join(not_looped) or '없음'}). 올린 바에서도 재계획이 돌지 "
            "않았다면 이 실험의 전제가 깨진 것이고, 기준을 약하게 고치지 않습니다"
        )
    elif len(wins) >= 3:
        complete = True
        status = (
            f"기준 충족 — 이진 {len(DATASETS)}개 중 {len(wins)}개에서 bar65llm이 세 시드 전부 "
            f"bar65nollm을 이겼고 가장 작은 Δ가 잡음 바닥 {NOISE_FLOOR}를 넘습니다 "
            f"({', '.join(wins)}). {tally}"
        )
        if losses:
            status += f" — 다만 {', '.join(losses)}에서는 어느 시드의 CI가 0 아래입니다"
    else:
        complete = True
        status = f"구분되지 않음 — 이김이 {len(wins)}개로 3개에 미치지 못합니다 ({tally})"
        if losses:
            status += f". CI가 0 아래인 데이터셋: {', '.join(losses)}"
    if not_looped and complete:
        status += f" / 재계획이 안 돈 칸 {len(not_looped)}개는 분모 밖입니다"

    return {
        "challenger": PRIMARY[0],
        "baseline": PRIMARY[1],
        "margin": MARGIN,
        "scope_datasets": list(DATASETS),
        "scope_seeds": list(SEEDS),
        "resample_seed": RESAMPLE_SEED,
        "required_wins": 3,
        "per_dataset": per,
        "wins": wins,
        "losses": losses,
        "not_evaluated": absent,
        "not_looped": not_looped,
        "looped_cells": looped,
        "total_cells": cells,
        "noise_floor": NOISE_FLOOR,
        "noise_floor_source": NOISE_FLOOR_SOURCE,
        "rules_seed_spread": rules_seed_spread(results),
        "complete": complete,
        "status": status,
    }


def budget_used(results: dict[tuple[str, int], DatasetVerdict]) -> list[dict[str, Any]]:
    """(데이터셋, 시드)마다 각 팔이 쓴 것 — 학습 횟수, 종료 이유, Critic 실행 횟수, 마주한 바.

    정체 가드는 미달인 바에서도 ``max_iterations`` 전에 실행을 끝낼 수 있으므로 처치의 실제 크기가
    달라진다. 그리고 칸을 기준의 범위 안에 넣는 것은 ``critic_runs``다.
    """
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            result = results.get((dataset, seed))
            if result is None:
                continue
            row: dict[str, Any] = {"dataset": dataset, "seed": seed}
            for arm, _ in ARMS:
                arm_row = result.arms.get(arm, {})
                row[arm] = {
                    "fits": arm_row.get("fits"),
                    "stop_reason": arm_row.get("stop_reason"),
                    "critic_runs": arm_row.get("critic_runs"),
                    "goal_threshold": arm_row.get("goal_threshold"),
                    "winner": arm_row.get("label"),
                }
            rows.append(row)
    return rows


def payload(
    results: dict[tuple[str, int], DatasetVerdict], resamples: int
) -> dict[str, Any]:
    ordered = [
        (dataset, seed)
        for dataset in DATASETS
        for seed in SEEDS
        if (dataset, seed) in results
    ]
    return {
        "generated_by": "bench/hard_bar.py",
        "preregistration": "docs/HARD-BAR.md",
        "widens": "docs/REPLAN-SEEDS.md",
        "margin": MARGIN,
        "run_seeds": sorted({seed for _, seed in ordered}),
        # 파일 전체에 하나인 부트스트랩 시드: ``bench/recheck.py``는 판정마다 ``seed`` 하나를
        # 읽고 그것으로 모든 쌍을 다시 계산한다.
        "seed": RESAMPLE_SEED,
        "resamples": resamples,
        "score_tolerance": SCORE_TOLERANCE,
        "threads": thread_state(),
        "datasets": [
            {
                "dataset": results[cell].dataset,
                "run_seed": cell[1],
                "metric": results[cell].metric,
                "task": results[cell].task,
                "verdict": results[cell].verdict,
                "refusal": results[cell].refusal,
                "test_rows": results[cell].test_rows,
                "test_fingerprint": results[cell].test_fingerprint,
                "missing_arms": results[cell].missing_arms,
                "arms": results[cell].arms,
                "pairs": [vars(p) for p in results[cell].pairs],
            }
            for cell in ordered
        ],
        "bar_check": bar_check(results),
        "budget": budget_used(results),
        "criterion": criterion(results),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "올린 바에서 LLM 팔이 규칙 폴백을 이기는가 (docs/HARD-BAR.md의 사전 등록 기준)"
        )
    )
    parser.add_argument(
        "names",
        nargs="*",
        help=f"데이터셋 이름 (생략하면 {' '.join(DATASETS)})",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEEDS),
        help=f"기본: {' '.join(str(s) for s in SEEDS)} (그보다 적으면 부분 판정입니다)",
    )
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="기본: bench/runs/paired/hard-bar.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = args.names or list(DATASETS)
    unknown = [name for name in names if name not in DATASETS] + [
        str(s) for s in args.seeds if s not in SEEDS
    ]
    if unknown:
        print(
            f"사전 등록의 범위 밖: {', '.join(unknown)} "
            f"(데이터셋 {', '.join(DATASETS)} · 시드 {', '.join(str(s) for s in SEEDS)})",
            file=sys.stderr,
        )
        return 2

    print(
        f"올린 바의 판정 · margin {MARGIN} · 재추출 {args.resamples}회 "
        f"(재추출 시드 {RESAMPLE_SEED}) · {describe_thread_state(thread_state())}"
    )

    legs: dict[str, dict[str, Scored]] = {}
    results: dict[tuple[str, int], DatasetVerdict] = {}
    for dataset in DATASETS:
        if dataset not in names:
            continue
        for seed in SEEDS:
            if seed not in args.seeds:
                continue
            collected: dict[str, Scored] = {}
            results[(dataset, seed)] = adjudicate(dataset, seed, args.resamples, collected)
            if collected:
                legs[results[(dataset, seed)].dataset] = collected
    report(list(results.values()))

    out_path = args.out or OUT_DIR / "hard-bar.json"
    if not legs:
        # 채점된 것이 없으니 내놓을 것도, 함께 내놓을 번들도 없다. 여기서 판정 파일을 쓰면 인수
        # 없이 돌린 ``python -m bench.recheck``가 읽는 디렉터리에 앉아서 그것을 실패시킨다.
        print(
            f"판정할 다리가 없습니다 — {out_path.as_posix()}를 쓰지 않았습니다. "
            "bench/scripts/run_hard_bar.sh를 먼저 돌리세요",
            file=sys.stderr,
        )
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload(results, args.resamples), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    target = bundle_path(out_path)
    write_bundle(target, legs, {r.dataset: r.metric for r in results.values()})
    print(f"예측 번들 → {target.as_posix()} ({target.stat().st_size / 1024:.0f} KB)")
    print()
    print(f"바 — 예측 대조 (허용 {BAR_TOLERANCE:g}):")
    for row in bar_check(results):
        mark = "일치" if row["matches"] else "어긋남"
        print(
            f"   {row['dataset']:16s} 시드 {row['seed']} {row['arm']:<12} "
            f"예측={row['predicted_bar']:.4f} 실측={row['actual_bar']} {mark} "
            f"첫시도val 위={row['above_recorded_first_val']}"
        )
    print()
    print("예산 — 실제로 쓴 것 (정체 가드가 미달인 바에서도 먼저 걸릴 수 있습니다):")
    for row in budget_used(results):
        for arm, _ in ARMS:
            spent = row[arm]
            print(
                f"   {row['dataset']:16s} 시드 {row['seed']} {arm:<12} 학습={spent['fits']} "
                f"종료={spent['stop_reason']} critic={spent['critic_runs']} "
                f"바={spent['goal_threshold']} 우승={spent['winner']}"
            )
    print()
    verdict = criterion(results)
    print(f"기준: {verdict['status']}")
    print(f"잡음 바닥 {NOISE_FLOOR} — {NOISE_FLOOR_SOURCE}")
    print(f"규칙 팔 시드 산포: {verdict['rules_seed_spread']}")
    print(f"판정 → {out_path.as_posix()}")
    refused = [r.dataset for r in results.values() if r.verdict != "ok"]
    if refused:
        print(f"거부된 칸: {', '.join(refused)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
