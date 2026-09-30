"""고친 가중치 사다리가 루프가 실제로 돌 때 값을 내는가, 그리고 헤드라인이 움직이는가?

``docs/WEIGHT-LEVER.md``는 기록된 규칙 우승자 자신의 config에 카드 비율을 주입해서,
``docs/HARD-BAR.md``가 LLM 팔에 청구했던 Δ의 67%·84%·114%를 회수했다. 그 문서는 자기가 못 한 일을
스스로 적었다: 루프를 돌리지 않았으므로 "그 격차가 가중치인가"를 쟀고 "제안한 코드 변경이 무엇을
낼까"를 재지 않았다. 커밋 ``d37217e``이 그 변경을 했고 — Critic의 첫 단이 ``WEIGHT_STEP``이 아니라
카드에서 온다 — 이 모듈이 그 뒤 열두 실행을 판정한다.

판정 둘. 어느 쪽도 다른 쪽을 구제할 수 없게 떼어 놓는다:

**A. 처방** — ``bar65nollmratio − bar65nollm``. ``WEIGHT-LEVER.md``가 쓴 것과 같은 조건 셋
(speeddating에서 회수, spambase에서 동일성)에 그 문서에는 필요 없던 하나를 더한다: 어느 칸도 0 아래로
나와서는 안 된다. 그 하나가 새로운 이유는 이번이 코드 변경이고 루프 전체가 다르게 돌기 때문이다.
기록된 config 하나에 수를 주입하는 방식으로는 다른 칸을 해칠 수 없었다.

**B. 헤드라인 다시 묻기** — ``HARD-BAR.md``의 기준을 *글자 그대로* 써서
``bar65llm − bar65nollmratio``를 본다. 글자 그대로를 약속이 아니라 강제한다:
:func:`as_hard_bar_board`가 ``headline`` 짝을 그 모듈의 ``primary``로 다시 이름 붙이고
:func:`headline_verdict`가 그 판을 ``bench.hard_bar.criterion``에 넘긴다. 그 축약의 두 번째 사본은
결국 어긋나고, 어긋난 사본이 듣기 좋은 쪽이 된다.

유료 팔은 다시 돌리지 않는다. ``critic()``은 LLM이 꺼져 있거나 그 답이 검증에 실패할 때만
``heuristic_verdict``에 닿고, 기록된 ``bar65llm`` 실행의 Critic verdict 37개 전부가
``source == "llm"``이다 — :func:`source_check`가 아카이브에서 그걸 다시 확인하고, 사전 등록이 어긋남을
정지 조건으로 만든다. 그래서 이 실험은 토큰이 0이다.

사전 등록은 ``docs/WEIGHT-START.md``이고, 이 실행이 하나도 생기기 전에 커밋됐다.

팔들이 돈 것처럼 ``OMP_NUM_THREADS=1``로 돌린다. 값은 출력에 기록된다.

사용법:
    python -m bench.weight_start                    # 데이터셋 넷을 시드 셋에서
    python -m bench.weight_start adult --seeds 42
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from automl_agent.config import read_json_object
from automl_agent.threads import describe_thread_state, thread_state
from bench import hard_bar
from bench.datasets import BY_NAME, JUDGED_SEED
from bench.hard_bar import BAR_TOLERANCE, DATASETS, MARGIN, PREDICTED_BARS, SEEDS
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

# 이 실험이 돌리는 팔과 재사용하는 팔 둘. 아카이브 접두사가 실행이 마주한 조건 둘을 다 적는다 —
# 바와 코드. ``bar65nollm-adult-seed42``는 ``HARD-BAR.md``의 기록된 아카이브이고 여기서 아무것도 그
# 안에 써서는 안 되기 때문이다.
FIXED = "bar65nollmratio"
RULES = "bar65nollm"
LOOP = "bar65llm"

# :data:`FIXED`와 :data:`RULES`를 가르는 커밋 하나. 판정 파일이 레버의 이름을 대고 독자가 거기에
# ``git show``를 돌릴 수 있도록 못박는다.
LEVER_COMMIT = "d37217e"

# 새 코드의 첫 ``class_weight`` 처방이 데이터셋마다 무엇이어야 하고 몇 번째 반복이어야 하는가.
# ``None``은 spambase다. 기록된 세 실행이 가중치를 아예 처방하지 않았다 — 관문 뒤 분기를 한 번도 타지
# 않았으므로 첫 단을 옮겨도 거기 닿지 못한다.
#
# 예측하는 것은 첫 단뿐이다. 그 뒤로는 궤적이 갈린다: 가중치 3.179는 다음 시도의 진단을 바꾸고 Critic이
# 아예 다른 분기를 탈 수 있다. 기록된 사다리(adult 1→1.5→2.25, bank-marketing
# 1→1.5→2.25→3.375→5.062, speeddating 1→1→1.5)가 다시 나타날 것으로 기대하지 않고 여기서 확인하지도
# 않는다.
#
# ``hard_bar.PREDICTED_BARS``처럼 import 시점에 유도하지 않고 손으로 적는다: 오늘 카드가 무엇을 말하든
# 거기서 자기 예측을 다시 계산하는 사전 등록은 아무것도 예측하지 않는 것이다. 커밋된 카드와
# ``critic.py``가 여전히 이 값을 내는지는 ``tests/test_bench_weight_start.py``가 단언하고, 실행이 실제로
# 그랬는지는 :func:`first_rung_check`가 단언한다.
EXPECTED_FIRST_RUNG: dict[str, tuple[int, float] | None] = {
    "adult": (1, 3.179),
    "bank-marketing": (1, 7.547),
    "speeddating": (2, 5.072),
    "spambase": None,
}

# 같은 자리에서 옛 코드가 한 것. 기록된 ``bar65nollm`` 아카이브에서 왔다. 실험이 *이* 수가 움직이는
# 것에 대한 것이라서 못박는다. 다른 데서 시작한 아카이브를 다시 만들면 시험하는 대상이 바뀌고, 그건
# 테스트가 잡는다.
RECORDED_FIRST_RUNG: dict[str, tuple[int, float] | None] = {
    "adult": (1, 1.5),
    "bank-marketing": (1, 1.5),
    "speeddating": (2, 1.5),
    "spambase": None,
}

# 기록된 팔들이 든 Critic verdict 개수와 각각의 출처. 재사용의 전제는 유료 팔이
# ``heuristic_verdict``에 한 번도 닿지 않았다는 것이다. :func:`source_check`가 이 개수를 다시 내므로,
# 다시 만든 아카이브가 verdict 0개로 조용히 통과할 수 없다.
EXPECTED_SOURCES: dict[str, tuple[str, int]] = {
    LOOP: ("llm", 37),
    RULES: ("heuristic", 43),
}

# 조건 1이 다루는 데이터셋에 대해 ``HARD-BAR.md``가 공표한 주 Δ. 소수점 넷째 자리까지.
# 판정은 이 값을 쓰지 않는다: 자기가 채점한 다리에서 ``bar65llm − bar65nollm``을 다시 계산하고 반으로
# 나눈다. 여기 있는 이유는 어긋남이 아무도 확인하지 않은 수가 아니라 실패가 되게 하려는 것이다.
# ``bench/weight_lever.py``와 같은 자릿수, 같은 허용오차.
PRIMARY_DELTAS: dict[tuple[str, int], float] = {
    ("speeddating", 42): 0.0819,
    ("speeddating", 43): 0.0777,
    ("speeddating", 44): 0.0761,
}
PUBLISHED_TOLERANCE = 5e-5

# 여기서 재지 않고 import한다: 낡은 프롬프트·낡은 바에서 잰 ``llm`` 팔 같은 설정 반복의 반폭
# 중앙값(``REPEATS.md``). 다시 적지 않고 ``hard_bar``에서 가져온다 — 판정 B가 그 모듈의 기준이므로
# 그 모듈의 바닥을 써야 한다.
NOISE_FLOOR = hard_bar.NOISE_FLOOR
NOISE_FLOOR_SOURCE = hard_bar.NOISE_FLOOR_SOURCE

# 실행 시드가 아니다. 파일의 모든 짝이 한 재추출 수열에서 나오도록, 그리고 판정 파일마다 ``seed``
# 하나를 읽는 ``bench/recheck.py``가 전부 다시 계산할 수 있도록 못박는다.
RESAMPLE_SEED = JUDGED_SEED

# 판정 A의 앞 두 조건이 각각 사는 데이터셋.
POSITIVE_ON = "speeddating"
IDENTITY_ON = "spambase"


def _key(dataset: str, seed: int) -> str:
    """판정 항목의 이름. 양쪽 다 움직이므로 둘 다 들어간다."""
    return f"{dataset}-seed{seed}"


def run_directory(arm: str, dataset: str, seed: int) -> Path:
    return ARTIFACTS_DIR / f"{arm}-{dataset}-seed{seed}"


# --------------------------------------------------------------------------- #
# Critic이 무엇을 처방했고 verdict가 어디서 왔는가
# --------------------------------------------------------------------------- #


def prescriptions(directory: Path) -> list[dict[str, Any]]:
    """실행 하나의 모든 Critic verdict. 출처와, 있다면 ``class_weight``까지.

    반복별 아카이브가 아니라 ``history.json``을 읽는다. 실행 자신이 원장으로 취급하는 파일이고,
    ``run_shape``이 ``critic_runs``를 세는 파일이기 때문이다.
    """
    history = read_json_object(directory / "history.json") or {}
    attempts = history.get("history") or []
    out: list[dict[str, Any]] = []
    for attempt in attempts:
        if not isinstance(attempt, dict):
            continue
        critic = attempt.get("critic")
        if not isinstance(critic, dict):
            continue
        changes = critic.get("concrete_changes")
        weight = changes.get("class_weight") if isinstance(changes, dict) else None
        out.append(
            {
                "iteration": attempt.get("iteration"),
                "source": critic.get("source"),
                "failure_type": critic.get("failure_type"),
                "class_weight": weight,
            }
        )
    return out


def positive_weight(weight: Any) -> float | None:
    """``class_weight`` map에서 양성 클래스의 수. 없으면 ``None``.

    map의 키는 config가 들고 있는 문자열 클래스 코드라서 ``1``이 아니라 ``"1"``을 읽는다. 애초에 map이
    아닌 것 — 규칙 Critic이 내지 않는 ``"balanced"`` — 은 ``None``을 돌려주므로, 판정을 죽이지 않고
    어긋남으로 보고된다.
    """
    if not isinstance(weight, dict):
        return None
    value = weight.get("1")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def first_rung(directory: Path) -> tuple[int, float] | None:
    """이 실행의 Critic이 처방한 첫 ``class_weight``. (반복, 양성 가중치) 형태."""
    for row in prescriptions(directory):
        value = positive_weight(row["class_weight"])
        if value is not None and isinstance(row["iteration"], int):
            return (row["iteration"], value)
    return None


def first_rung_check(arm: str = FIXED) -> list[dict[str, Any]]:
    """코드 변경이 :data:`EXPECTED_FIRST_RUNG`가 말한 자리에 착지했는가?

    사전 등록된 우선순위의 관문 1. 첫 단이 다른 데 있는 실행은 이 문서가 기술하는 처치가 아니고,
    기준을 약하게 고치는 대신 판정 불가로 적는다. 처방이 *없을* 것으로 예측된 데이터셋에 처방이 나타나면
    똑같이 크게 실패한다: 그게 동일성 예측의 기제이고, 점수만이 아니라 기제에서 확인해야 한다.
    """
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            directory = run_directory(arm, dataset, seed)
            expected = EXPECTED_FIRST_RUNG[dataset]
            present = (directory / "history.json").exists()
            actual = first_rung(directory) if present else None
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "arm": arm,
                    "expected_iteration": None if expected is None else expected[0],
                    "expected_weight": None if expected is None else expected[1],
                    "actual_iteration": None if actual is None else actual[0],
                    "actual_weight": None if actual is None else actual[1],
                    # 일어나지 않은 실행은 일치가 아니라 "없음"이다: 없는 history가 "처방 없음을
                    # 예측했고 없었다"로 읽혀서는 안 된다.
                    "run_present": present,
                    "matches": present and actual == expected,
                }
            )
    return rows


def source_check() -> list[dict[str, Any]]:
    """팔마다, 기록된 Critic verdict가 전부 어디서 왔는가.

    관문 2. 재사용의 전제는 ``d37217e``이 유료 팔에 닿을 수 없었다는 것이고, 그건 유료 팔이 한 번도 타지
    않은 폴백이 ``heuristic_verdict``이기 때문에만 성립한다. 새 팔은 반대 방향으로 확인한다: verdict
    전부가 ``heuristic``이어야 하고, 아니면 ``--no-llm``으로 돌린 것이 아니므로 레버가 이 문서가 기술하는
    레버가 아니다.
    """
    rows: list[dict[str, Any]] = []
    for arm, (expected_source, expected_count) in (
        *EXPECTED_SOURCES.items(),
        (FIXED, ("heuristic", 0)),
    ):
        counts: dict[str, int] = {}
        present = 0
        for dataset in DATASETS:
            for seed in SEEDS:
                directory = run_directory(arm, dataset, seed)
                if not (directory / "history.json").exists():
                    continue
                present += 1
                for row in prescriptions(directory):
                    key = str(row["source"])
                    counts[key] = counts.get(key, 0) + 1
        total = sum(counts.values())
        rows.append(
            {
                "arm": arm,
                "runs_present": present,
                "sources": counts,
                "expected_source": expected_source,
                # 0은 "못박지 않음"이다: 새 팔의 verdict 개수는 예측할 수 없다. 첫 단 뒤로 궤적이
                # 갈리기 때문이다. 예측할 수 있는 것은 *출처*뿐이다.
                "expected_count": expected_count or None,
                "matches": (
                    present == len(DATASETS) * len(SEEDS)
                    and set(counts) <= {expected_source}
                    and total > 0
                    and (expected_count == 0 or total == expected_count)
                ),
            }
        )
    return rows


def bar_check(results: dict[tuple[str, int], DatasetVerdict]) -> list[dict[str, Any]]:
    """새 열두 실행이 ``HARD-BAR.md``의 바를 마주했는가?

    관문 3. ``hard_bar.PREDICTED_BARS``가 못박은 것과 같은 표다 — 같은 카드, 같은 ``--margin``, 그래서
    같은 수. 다시 나열하지 않고 import한다: 바 표가 두 벌이면 독자가 실행이 아니라 그 둘을 비교하게 된다.
    """
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            result = results.get((dataset, seed))
            if result is None:
                continue
            predicted = PREDICTED_BARS[(dataset, seed)]
            actual = result.arms.get(FIXED, {}).get("goal_threshold")
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
                    "arm": FIXED,
                    "predicted_bar": predicted,
                    "actual_bar": actual,
                    "off_by": off,
                    "matches": off is not None and off <= BAR_TOLERANCE,
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# (데이터셋, 시드) 하나 채점하기
# --------------------------------------------------------------------------- #


def _legs(
    dataset: str, seed: int, metric: str, out: DatasetVerdict
) -> tuple[dict[str, Winner], dict[str, dict[str, Any]]]:
    """새 팔, 기록된 팔 둘, 새 팔의 반복 1, 그리고 각 실행의 모양."""
    winners: dict[str, Winner] = {}
    shapes: dict[str, dict[str, Any]] = {}
    for arm in (FIXED, RULES, LOOP):
        directory = run_directory(arm, dataset, seed)
        winner = loop_winner(arm, directory, metric)
        if winner is None:
            out.missing_arms.append(arm)
            continue
        winners[arm] = winner
        shapes[arm] = run_shape(directory)
    if FIXED in winners:
        # 반복이 고친 팔에 무엇을 사 줬는가? ``REPLAN.md``의 질문을 새 팔에 묻는다.
        winners[f"{FIXED}@it1"] = first_iteration_winner(
            FIXED, run_directory(FIXED, dataset, seed), metric
        )
    return winners, shapes


def _pairs_to_take(scored: dict[str, Scored], fits: int) -> list[tuple[str, str, str]]:
    """이 칸이 낼 수 있는 비교. ``WEIGHT-START.md``가 나열한 순서로."""
    pairs: list[tuple[str, str, str]] = []
    if FIXED in scored and RULES in scored:
        pairs.append((FIXED, RULES, "prescription"))
    if LOOP in scored and FIXED in scored:
        pairs.append((LOOP, FIXED, "headline"))
    if LOOP in scored and RULES in scored:
        # 다시 재지 않고 재사용한다: 조건 1의 절반 바가 나누는 분모이고, ``published_check``이
        # ``HARD-BAR.md``에 대고 비교하는 값이다.
        pairs.append((LOOP, RULES, "main"))
    if FIXED in scored and f"{FIXED}@it1" in scored:
        pairs.append((FIXED, f"{FIXED}@it1", "ratio_replan_gain"))
    matched = f"random@k={fits}"
    if FIXED in scored and matched in scored:
        pairs.append((FIXED, matched, "ratio_matched_level"))
    return pairs


def _adjudicate(
    dataset: str, seed: int, resamples: int, out: DatasetVerdict, legs: dict[str, Scored] | None
) -> None:
    entry = BY_NAME[dataset]
    metric = entry.metric
    winners, shapes = _legs(dataset, seed, metric, out)
    if FIXED not in winners:
        # 데이터에 대한 거부가 아니다: 이 판정이 다루는 실행이 아직 없다.
        raise Refusal(
            f"{dataset} 시드 {seed}: {FIXED} 실행이 없습니다 — "
            "bench/scripts/run_weight_start.sh를 먼저 돌려야 판정할 수 있습니다"
        )

    random_dir = random_run_dir(entry, seed)
    if (random_dir / "summary.json").exists():
        fits = winners[FIXED].fits
        if fits < 1:
            raise Refusal(f"{dataset} 시드 {seed}: {FIXED}의 학습 횟수가 {fits}회로 기록됐습니다")
        winners[f"random@k={fits}"] = random_winner(random_dir, fits, metric)
    else:
        out.missing_arms.append("random")

    scored: dict[str, Scored] = {}
    for name, winner in winners.items():
        result = reproduce(winner, metric)
        # 반복 1 다리는 nan이다. test 점수가 기록된 적이 없다 — holdout은 루프가 끝난 뒤 우승자에
        # 대해 한 번 돈다. 이유는 ``bench/replan.py``가 적는다.
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
        config_weight = dict(s.winner.config.get("hyperparams") or {}).get("class_weight")
        out.arms[name] = {
            "label": s.winner.label,
            "fits": s.winner.fits,
            "val_score": s.winner.val_score,
            "test_recorded": s.winner.recorded_test,
            "test_reproduced": s.test_score,
            "selection_gap": s.winner.val_score - s.test_score,
            "class_weight": config_weight,
            "model": s.winner.config.get("model"),
            "dir": s.winner.directory.as_posix(),
            **shapes.get(name, {}),
        }

    for a_name, b_name, kind in _pairs_to_take(scored, winners[FIXED].fits):
        out.pairs.append(
            paired_delta(scored[a_name], scored[b_name], metric, resamples, RESAMPLE_SEED, kind)
        )

    # 중간에 거부가 나면 다리를 남기지 않도록 맨 마지막에.
    if legs is not None:
        legs.update(scored)


def adjudicate(
    dataset: str, seed: int, resamples: int, legs: dict[str, Scored] | None = None
) -> DatasetVerdict:
    """(데이터셋, 시드) 하나. ``paired.py``처럼 거부는 짝을 비운다."""
    entry = BY_NAME[dataset]
    out = DatasetVerdict(dataset=_key(dataset, seed), metric=entry.metric, task=entry.task)
    try:
        _adjudicate(dataset, seed, resamples, out, legs)
    except Refusal as exc:
        out.verdict = "refused"
        out.refusal = str(exc)
        out.pairs = []
    return out


def _pair(result: DatasetVerdict, kind: str) -> Delta | None:
    return next((p for p in result.pairs if p.kind == kind), None)


def published_check(results: dict[tuple[str, int], DatasetVerdict]) -> list[dict[str, Any]]:
    """다시 계산한 주 Δ를 ``HARD-BAR.md``가 찍은 소수점 넷째 자리에 대고 비교한다."""
    rows: list[dict[str, Any]] = []
    for (dataset, seed), published in sorted(PRIMARY_DELTAS.items()):
        result = results.get((dataset, seed))
        pair = None if result is None else _pair(result, "main")
        recomputed = None if pair is None else pair.delta
        off = None if recomputed is None else abs(round(recomputed, 4) - published)
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "published": published,
                "recomputed": recomputed,
                "off_by": off,
                "matches": off is not None and off <= PUBLISHED_TOLERANCE,
            }
        )
    return rows


def identity_check(results: dict[tuple[str, int], DatasetVerdict]) -> list[dict[str, Any]]:
    """칸마다 조건 2의 기제: 고친 팔의 실행이 키 하나하나까지 기록된 그 실행인가?

    엄격한 시험은 예측이다 — ``WEIGHT-LEVER.md``의 ``refit`` 열두 칸이 ``hist_gbdt``와 ``logreg``의 같은
    설정 표류를 정확히 0으로 쟀으므로, 예측은 "비슷함"이 아니다. 실행 모양이 함께 가는 이유는, 다른 데서
    멈춘 실행은 다른 길로 동일성에 닿은 것이고 점수가 맞는 자리에서도 그건 알 만한 것이기 때문이다.
    """
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            result = results.get((dataset, seed))
            pair = None if result is None else _pair(result, "prescription")
            fixed = {} if result is None else result.arms.get(FIXED, {})
            rules = {} if result is None else result.arms.get(RULES, {})
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "identical_predictions": None if pair is None else pair.identical_predictions,
                    "delta": None if pair is None else pair.delta,
                    "same_fits": fixed.get("fits") == rules.get("fits"),
                    "same_stop_reason": fixed.get("stop_reason") == rules.get("stop_reason"),
                    "same_winner_label": fixed.get("label") == rules.get("label"),
                    "predicted_identical": EXPECTED_FIRST_RUNG[dataset] is None,
                }
            )
    return rows


# --------------------------------------------------------------------------- #
# 판정 A — 처방
# --------------------------------------------------------------------------- #


def _cell(result: DatasetVerdict | None) -> dict[str, Any]:
    """(데이터셋, 시드) 하나를 판정 A의 조건 셋이 읽는 것으로 줄인 것."""
    if result is None or result.verdict != "ok":
        return {"evaluated": False, "reason": "없음 또는 거부"}
    prescription = _pair(result, "prescription")
    main = _pair(result, "main")
    if prescription is None or main is None:
        return {"evaluated": False, "reason": "쌍이 모자랍니다"}
    half = main.delta / 2.0
    identical = prescription.identical_predictions
    return {
        "evaluated": True,
        "main": main.delta,
        "half_of_main": half,
        "prescription": prescription.delta,
        "prescription_ci": [prescription.ci_low, prescription.ci_high],
        "identical_predictions": identical,
        # 조건 1: 0을 벗어나고 *동시에* 설명해야 하는 격차의 절반 이상.
        "clears_zero": (not identical) and prescription.ci_low > 0,
        "reaches_half": prescription.delta >= half,
        # 조건 2, 차등 데이터셋에서. "바닥 아래"가 아니라 동일성이다.
        "identical_to_recorded": identical,
        # 조건 3, 모든 칸에서: 한 카드에서 버는 대가로 다른 카드를 잃는 처치는 처치가 아니다.
        "below_zero": (not identical) and prescription.ci_high < 0,
    }


def prescription_verdict(
    results: dict[tuple[str, int], DatasetVerdict], gates: dict[str, bool]
) -> dict[str, Any]:
    """판정 A. 이 실행이 하나도 생기기 전에 ``docs/WEIGHT-START.md``에 커밋됐다.

    > **처방이 산다는 조건** — 셋 **전부** 충족:
    >
    > 1. **회수**: speeddating 시드 42·43·44 **전부**에서 ``bar65nollmratio − bar65nollm``의 95%
    >    CI가 0을 걸치지 않게 양수이고, 그 Δ가 같은 시드의 주 Δ의 **절반 이상**이다.
    > 2. **동일성**: spambase 시드 42·43·44 **전부**에서 우승자가 ``bar65nollm``의 우승자와
    >    **예측까지 동일**하다.
    > 3. **짐 없음**: 열두 칸 어디에도 CI가 0 **아래**인 칸이 없다.
    >
    > **처방이 죽는 조건**: 1이 깨진다. 그 밖은 **부분 지지**.

    과반이 아니라 시드 전원 일치이고, 이유는 ``HARD-BAR.md``와 같다: ``REPEATS.md``의 R2가 *같은*
    설정의 반복에서 0을 벗어난 짝 Δ를 12쌍 중 4쌍에서 냈다. 그래서 시드 하나의 구간은 처치의 증거가
    아니고, n=3에서 셋 중 둘은 동전과 구분하기 어렵다.

    미리 못박은 우선순위: 관문 셋이 먼저다(첫 단, verdict 출처, 바) — 하나라도 어긋나면 판정 불가이고
    수를 건지려고 기준을 약하게 고치지 않는다. 그다음 없거나 거부된 칸이 있으면 부분. 그다음 조건 셋.
    """
    cells = {
        (dataset, seed): _cell(results.get((dataset, seed)))
        for dataset in DATASETS
        for seed in SEEDS
    }
    absent = [
        f"{dataset}-seed{seed}"
        for (dataset, seed), cell in cells.items()
        if not cell["evaluated"]
    ]
    failed_gates = [name for name, ok in gates.items() if not ok]

    def _all(dataset: str, field: str) -> bool:
        return all(
            cells[(dataset, seed)]["evaluated"] and cells[(dataset, seed)][field] for seed in SEEDS
        )

    recovered = _all(POSITIVE_ON, "clears_zero") and _all(POSITIVE_ON, "reaches_half")
    identical = _all(IDENTITY_ON, "identical_to_recorded")
    losses = [
        f"{dataset}-seed{seed}"
        for (dataset, seed), cell in cells.items()
        if cell["evaluated"] and cell["below_zero"]
    ]
    no_loss = not losses

    if failed_gates:
        verdict = "판정 불가"
        status = (
            f"판정 불가 — 전제 검사가 어긋났습니다 ({', '.join(failed_gates)}). "
            "코드 변경이 예상과 다른 자리에 착지했거나 실행 조건이 다릅니다 — "
            "기준을 약하게 고치지 않습니다"
        )
    elif absent:
        verdict = "부분 판정"
        status = (
            f"부분 판정 — (데이터셋, 시드) {len(DATASETS) * len(SEEDS)}칸 중 판정하지 못한 칸이 "
            f"있습니다 (없음/거부: {', '.join(absent)})"
        )
    elif recovered and identical and no_loss:
        verdict = "처방이 산다"
        status = (
            f"처방이 산다 — {POSITIVE_ON} 세 시드 전부에서 고친 사다리가 주 Δ의 절반 이상을 "
            f"회수했고, {IDENTITY_ON}는 예측까지 동일하고, 짐이 된 칸이 없습니다"
        )
    elif not recovered:
        verdict = "처방이 죽는다"
        status = (
            f"처방이 죽는다 — {POSITIVE_ON}에서 첫 단을 카드 비율로 옮긴 코드가 루프에서 주 Δ의 "
            "절반을 회수하지 못했습니다. WEIGHT-LEVER가 본 회수는 루프가 도달할 수 있는 설정이 "
            "아니었습니다"
        )
    else:
        broken = [
            name for name, ok in (("동일성", identical), ("짐 없음", no_loss)) if not ok
        ]
        verdict = "부분 지지"
        status = (
            f"부분 지지 — 회수는 세 시드 전부에서 됐지만 {', '.join(broken)} 조건이 깨졌습니다"
        )
    if losses and verdict not in ("판정 불가", "부분 판정"):
        status += f" / CI가 0 아래인 칸: {', '.join(losses)}"

    return {
        "treatment": FIXED,
        "baseline": RULES,
        "lever_commit": LEVER_COMMIT,
        "scope_datasets": list(DATASETS),
        "scope_seeds": list(SEEDS),
        "positive_prediction_on": POSITIVE_ON,
        "identity_prediction_on": IDENTITY_ON,
        "resample_seed": RESAMPLE_SEED,
        "expected_first_rung": {d: list(v) if v else None for d, v in EXPECTED_FIRST_RUNG.items()},
        "recorded_first_rung": {d: list(v) if v else None for d, v in RECORDED_FIRST_RUNG.items()},
        "gates": gates,
        "failed_gates": failed_gates,
        "cells": {f"{d}-seed{s}": cell for (d, s), cell in cells.items()},
        "condition_1_recovery": recovered,
        "condition_2_identical": identical,
        "condition_3_no_loss": no_loss,
        "below_zero_cells": losses,
        "not_evaluated": absent,
        "verdict": verdict,
        "status": status,
    }


# --------------------------------------------------------------------------- #
# 판정 B — HARD-BAR 자신의 기준으로 헤드라인 다시 묻기
# --------------------------------------------------------------------------- #


def as_hard_bar_board(
    results: dict[tuple[str, int], DatasetVerdict],
) -> dict[tuple[str, int], DatasetVerdict]:
    """이 칸들을 ``hard_bar.criterion``이 읽는 판으로 다시 쓴다. 상대만 바꿔서.

    ``HARD-BAR.md``의 축약 — 데이터셋 안에서 전원 일치, 가장 작은 시드에서 크기, 다시 계획하지 않은 칸은
    분모에서 제외, 그다음 4개 중 3개 — 을 여기 다시 구현하지 않는다. 두 번째 사본은 어긋나고, 어긋난 사본은
    어떤 결과가 오든 그걸 듣기 좋게 만드는 쪽이 된다. 그래서 ``headline`` 짝을 그 모듈의 ``primary``로
    다시 이름 붙여 넘긴다.

    함께 가는 것이 둘 있다. ``critic_runs``는 *기록된* ``bar65llm`` 실행에서 오고, 그건 ``HARD-BAR.md``가
    관문을 걸었던 것과 같은 실행이다 — 그래서 ``bank-marketing`` 시드 42는 새 이유가 아니라 전에 빠진 것과
    같은 이유로 분모 밖에 남는다. 그리고 판정 옆에 시드 산포가 공표되는 무료 팔은 고친 팔이다. 이 실험이
    측정한 팔이 그것이기 때문이다.
    """
    board: dict[tuple[str, int], DatasetVerdict] = {}
    for (dataset, seed), result in results.items():
        shim = DatasetVerdict(
            dataset=result.dataset,
            metric=result.metric,
            task=result.task,
            verdict=result.verdict,
            refusal=result.refusal,
            test_rows=result.test_rows,
            test_fingerprint=result.test_fingerprint,
        )
        loop_row = dict(result.arms.get(LOOP, {}))
        fixed_row = dict(result.arms.get(FIXED, {}))
        shim.arms[hard_bar.PRIMARY[0]] = loop_row
        shim.arms[hard_bar.PRIMARY[1]] = fixed_row
        headline = _pair(result, "headline")
        if headline is not None:
            shim.pairs.append(
                Delta(
                    a=hard_bar.PRIMARY[0],
                    b=hard_bar.PRIMARY[1],
                    kind="primary",
                    delta=headline.delta,
                    ci_low=headline.ci_low,
                    ci_high=headline.ci_high,
                    p_better=headline.p_better,
                    resamples_used=headline.resamples_used,
                    identical_predictions=headline.identical_predictions,
                )
            )
        board[(dataset, seed)] = shim
    return board


def headline_verdict(
    results: dict[tuple[str, int], DatasetVerdict], gates: dict[str, bool]
) -> dict[str, Any]:
    """판정 B: ``HARD-BAR.md``의 기준을 고치지 않고, 고친 규칙 팔에 대고."""
    outcome = hard_bar.criterion(as_hard_bar_board(results))
    failed_gates = [name for name, ok in gates.items() if not ok]
    if failed_gates:
        outcome["complete"] = False
        outcome["status"] = (
            f"판정 불가 — 전제 검사가 어긋났습니다 ({', '.join(failed_gates)}). "
            f"원래 상태: {outcome['status']}"
        )
    # ``hard_bar``가 자기 ``PRIMARY``로 채우는 키들을 실제로 비교한 팔로 바로잡는다.
    outcome["challenger"] = LOOP
    outcome["baseline"] = FIXED
    outcome["criterion_from"] = "bench/hard_bar.py criterion (글자 그대로 재사용)"
    outcome["compared_against"] = "docs/HARD-BAR.md의 구분되지 않음 (이김 1 / 그 밖 1 / 짐 2)"
    outcome["gates"] = gates
    outcome["failed_gates"] = failed_gates
    return outcome


# --------------------------------------------------------------------------- #
# 출력
# --------------------------------------------------------------------------- #


def gate_status(results: dict[tuple[str, int], DatasetVerdict]) -> dict[str, bool]:
    """정지 조건 셋을 판정 둘이 읽는 boolean으로 줄인 것."""
    return {
        "first_rung": all(row["matches"] for row in first_rung_check()),
        "verdict_sources": all(row["matches"] for row in source_check()),
        "bars": bool(results) and all(row["matches"] for row in bar_check(results)),
    }


def budget_used(results: dict[tuple[str, int], DatasetVerdict]) -> list[dict[str, Any]]:
    """칸마다 각 팔이 쓴 것 — 학습 횟수, 종료 이유, Critic 실행 횟수, 우승자의 가중치."""
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            result = results.get((dataset, seed))
            if result is None:
                continue
            row: dict[str, Any] = {"dataset": dataset, "seed": seed}
            for arm in (FIXED, RULES, LOOP):
                arm_row = result.arms.get(arm, {})
                row[arm] = {
                    "fits": arm_row.get("fits"),
                    "stop_reason": arm_row.get("stop_reason"),
                    "critic_runs": arm_row.get("critic_runs"),
                    "winner": arm_row.get("label"),
                    "model": arm_row.get("model"),
                    "class_weight": arm_row.get("class_weight"),
                    "val_score": arm_row.get("val_score"),
                    "test_reproduced": arm_row.get("test_reproduced"),
                }
            rows.append(row)
    return rows


def payload(results: dict[tuple[str, int], DatasetVerdict], resamples: int) -> dict[str, Any]:
    ordered = [
        (dataset, seed) for dataset in DATASETS for seed in SEEDS if (dataset, seed) in results
    ]
    gates = gate_status(results)
    return {
        "generated_by": "bench/weight_start.py",
        "preregistration": "docs/WEIGHT-START.md",
        "acts_on": "docs/WEIGHT-LEVER.md",
        "re_judges": "docs/HARD-BAR.md",
        "lever_commit": LEVER_COMMIT,
        "margin": MARGIN,
        "reuses_recorded_arms": [RULES, LOOP],
        "test_rows_shared_with": "docs/HARD-BAR.md",
        "run_seeds": sorted({seed for _, seed in ordered}),
        # 부트스트랩 시드. 파일 전체에 하나다: ``bench/recheck.py``가 판정마다 ``seed`` 하나를 읽고
        # 그걸로 모든 짝을 다시 계산한다.
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
        "gates": gates,
        "first_rung_check": first_rung_check(),
        "source_check": source_check(),
        "bar_check": bar_check(results),
        "identity_check": identity_check(results),
        "published_check": published_check(results),
        "budget": budget_used(results),
        "noise_floor": NOISE_FLOOR,
        "noise_floor_source": NOISE_FLOOR_SOURCE,
        "prescription": prescription_verdict(results, gates),
        "headline": headline_verdict(results, gates),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "고친 사다리가 루프에서 값을 내는가, 그리고 헤드라인 판정이 바뀌는가 "
            "(docs/WEIGHT-START.md의 사전 등록 기준)"
        )
    )
    parser.add_argument("names", nargs="*", help=f"데이터셋 이름 (생략하면 {' '.join(DATASETS)})")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(SEEDS),
        help=f"기본: {' '.join(str(s) for s in SEEDS)} (그보다 적으면 부분 판정입니다)",
    )
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument(
        "--out", type=Path, default=None, help="기본: bench/runs/paired/weight-start.json"
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
        f"고친 사다리의 판정 · 레버 {LEVER_COMMIT} · margin {MARGIN} · "
        f"재추출 {args.resamples}회 (재추출 시드 {RESAMPLE_SEED}) · "
        f"{describe_thread_state(thread_state())}"
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

    out_path = args.out or OUT_DIR / "weight-start.json"
    if not legs:
        # 채점된 것이 없으니 공표할 것도, 공표할 번들도 없다. 여기서 판정 파일을 쓰면 인수 없는
        # ``python -m bench.recheck``이 읽는 디렉터리에 놓여서 그걸 실패시킨다.
        print(
            f"판정할 다리가 없습니다 — {out_path.as_posix()}를 쓰지 않았습니다. "
            "bench/scripts/run_weight_start.sh를 먼저 돌리세요",
            file=sys.stderr,
        )
        return 1
    body = payload(results, args.resamples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    target = bundle_path(out_path)
    write_bundle(target, legs, {r.dataset: r.metric for r in results.values()})
    print(f"예측 번들 → {target.as_posix()} ({target.stat().st_size / 1024:.0f} KB)")

    print()
    print("첫 단 — 예측 대조:")
    for row in first_rung_check():
        mark = "일치" if row["matches"] else "어긋남"
        print(
            f"   {row['dataset']:16s} 시드 {row['seed']} "
            f"예측=(it{row['expected_iteration']}, {row['expected_weight']}) "
            f"실측=(it{row['actual_iteration']}, {row['actual_weight']}) {mark}"
        )
    print()
    print("verdict 출처:")
    for row in source_check():
        mark = "일치" if row["matches"] else "어긋남"
        print(
            f"   {row['arm']:<16} 실행={row['runs_present']} 출처={row['sources']} "
            f"기대={row['expected_source']} {mark}"
        )
    print()
    print(f"바 — 예측 대조 (허용 {BAR_TOLERANCE:g}):")
    for row in bar_check(results):
        mark = "일치" if row["matches"] else "어긋남"
        print(
            f"   {row['dataset']:16s} 시드 {row['seed']} 예측={row['predicted_bar']:.4f} "
            f"실측={row['actual_bar']} {mark}"
        )
    print()
    print(f"{IDENTITY_ON} 동일성 (예측: 예측까지 동일):")
    for row in identity_check(results):
        if not row["predicted_identical"]:
            continue
        print(
            f"   {row['dataset']:16s} 시드 {row['seed']} 동일예측={row['identical_predictions']} "
            f"Δ={row['delta']} 같은학습={row['same_fits']} 같은종료={row['same_stop_reason']} "
            f"같은우승={row['same_winner_label']}"
        )
    print()
    print("예산 — 실제로 쓴 것:")
    for row in budget_used(results):
        for arm in (FIXED, RULES, LOOP):
            spent = row[arm]
            print(
                f"   {row['dataset']:16s} 시드 {row['seed']} {arm:<16} 학습={spent['fits']} "
                f"종료={spent['stop_reason']} critic={spent['critic_runs']} "
                f"가중치={spent['class_weight']} 우승={spent['winner']}"
            )
    print()
    print(f"판정 A (처방): {body['prescription']['status']}")
    print(f"판정 B (헤드라인): {body['headline']['status']}")
    print(f"잡음 바닥 {NOISE_FLOOR} — {NOISE_FLOOR_SOURCE}")
    print(f"고친 규칙 팔 시드 산포: {body['headline']['rules_seed_spread']}")
    print(f"판정 → {out_path.as_posix()}")
    refused = [r.dataset for r in results.values() if r.verdict != "ok"]
    if refused:
        print(f"거부된 칸: {', '.join(refused)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
