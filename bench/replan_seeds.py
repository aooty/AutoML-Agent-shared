"""이 벤치마크가 측정한 단 하나의 재계획 이득이 시드 교체와 새 프롬프트를 견디는가?

``docs/REPLAN.md``는 이 모듈이 공격하려고 존재하는 그 한계로 끝난다: "시드 42 한 번의
``hard_llm``입니다. speeddating의 +0.0935가 LLM 비결정성 산포보다 큰지는 이 실험이 답하지
않습니다." 그 +0.0935 [+0.0654, +0.1219]는 이 저장소에서 재계획이 예산을 점수로 바꾸면서 구간이
0을 벗어난 유일한 자리이고, 실행 하나에 기대고 있다.

사전 등록은 ``docs/REPLAN-SEEDS.md``이고, 이 실행들이 존재하기 전에 커밋됐다. 기록된 팔 이후로 바뀐
것이 둘이라, 한쪽을 무시하지 않고 둘 다 시야에 둔다:

* **시드** — 셋(42, 43, 44). ``hard_nollm``이 이미 돌았던 그 셋이다;
* **프롬프트** — ``MODEL_REGISTRY``가 이제 실행기가 존중하는 레버를 알리고
  (``docs/REGISTRY-GAP.md``), 그 ``params``/``notes``가 계획 프롬프트에 렌더되므로, 그 커밋 이후의
  실행은 기록된 팔의 프롬프트가 아니다.

시드 42를 다시 돌리는 이유가 정확히 그것이다. 기록된 시드에서 새 프롬프트로 돈 실행이 없으면 모든
Δ가 프롬프트 *와* 시드를 함께 지게 되고, 그것이 이 저장소의 원장이 경고하는
"한 행에 레버가 둘인 전이"다.

규칙 팔도 세 시드 전부에서 다시 돌리고, 비용은 들지 않는다. 모델을 부르지 않으므로
``hard_nollm2 − hard_nollm``이 *코드* 변경을 *프롬프트* 변경에서 떼어 낸다 — 예측이 동일하게 나오면
루프 팔의 교란이 프롬프트로 좁혀지고, 동일하지 않으면 그 자체가 발견이다.

실행 시드 셋에 부트스트랩 시드가 왜 하나인가. 재추출 시드는 실행 시드가 아니다 — 행 인덱스를
고정하는 값이고, 각 실행의 시드를 쓰면 세 항목이 아무 이득 없이 서로 다른 재추출 수열을 갖게 된다.
``JUDGED_SEED``에 못박아 두어 시드 42 항목의 구간이 ``bench/replan.py``가 쓴 바로 그 수열에서
계산되고, 판정 파일마다 ``seed`` 하나를 읽는 ``bench/recheck.py``가 여기 모든 쌍을 다시 계산할 수
있게 한다.

새 팔 둘은 ``--keep-models all``로 돌려야 한다. 기준이 각 실행의 iteration 1을 다시 채점하고,
기본값은 우승하지 않은 iteration의 ``model.joblib``을 전부 지운다.

팔들이 돌았던 것과 같이 ``OMP_NUM_THREADS=1``에서 돌린다. 그 값은 출력에 기록된다.

사용법:
    python -m bench.replan_seeds                  # 범위에 든 시드 셋
    python -m bench.replan_seeds --seeds 42
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
from bench.datasets import BY_NAME, JUDGED_SEED
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

# 데이터셋 하나, 일부러 그렇게 했다. ``REPLAN.md``의 셋은 서로 다른 세 가지를 말했다 — adult는
# iteration 1을 우승자로 유지했고, spambase는 다중성의 값을 치렀고, speeddating은 얻었다 — 그리고
# 재현할 관측을 가진 것은 세 번째뿐이다. 나머지 둘로 넓히면 이것을 재현하는 것이 아니라 새 실행으로
# 새 질문을 묻는 것이고, 데이터셋을 더하려면 그 자체의 사전 등록이 필요하다.
DATASET = "speeddating"

# ``hard_nollm``이 이미 돌았던 셋이라, 무료 팔의 시드 산포가 같은 격자에서 측정된다. 명령줄에서
# 받지 않고 여기 못박아 두어 범위가 사후에 넓어질 수 없게 한다.
SEEDS: tuple[int, ...] = (42, 43, 44)

# 이 실험이 돌리는 팔들. 접미가 붙은 thread-id 접두사가 중요하다 — ``hardllm-speeddating-
# seed42``는 ``REPLAN.md``의 기록된 아카이브이고, 여기 있는 어느 것도 거기 쓸 수 없다.
ARMS: tuple[tuple[str, str], ...] = (("hard_llm2", "hardllm2"), ("hard_nollm2", "hardnollm2"))

# ``REPLAN.md``의 기록된 팔들. 다리를 더해 다시 채점한다. 규칙 팔은 세 시드 전부에 있고 루프 팔은
# 42에만 있다 — 프롬프트 비교가 시드 42의 관찰인 이유다.
RECORDED_RULES: tuple[str, str] = ("hard_nollm", "hardnollm")
RECORDED_LOOP: tuple[str, str] = ("hard_llm", "hardllm")
RECORDED_LOOP_SEEDS: tuple[int, ...] = (42,)

# 기준이 가장 작은 이김에 적용하는 크기 바. ``REPEATS.md``에서 왔다 — ``llm`` 팔의 같은 설정 반복의
# 반폭 중앙값이다. 이 벤치마크가 이미 측정한 잡음의 폭보다 작은 Δ는, 자기 CI가 무엇이라 하든
# +0.0935의 재현이 아니다.
NOISE_FLOOR = 0.0122

# 같은 세 시드에 걸친 규칙 팔의 test 산포. ``REPLAN.md``에 발표된 값이다. 그 문서의 사전 등록이
# 요구하는 대로 양의 판정 옆에 함께 인용한다.
RULES_SEED_SPREAD = 0.0017

# 재현 대상인 관측. ``REPLAN.md``의 쌍 표에서 왔다. 판정 파일이 무엇과 견주어지고 있는지를 함께
# 지도록 여기 적어 둔다.
RECORDED_PRIMARY = 0.0935

# 모듈 docstring을 보라 — 부트스트랩 시드는 실행 시드가 아니다.
RESAMPLE_SEED = JUDGED_SEED

PRIMARY = ("hard_llm2", "hard_llm2@it1")


def _arm_legs(seed: int, metric: str, out: DatasetVerdict) -> tuple[dict[str, Winner], dict[str, Any]]:
    """새 팔들의 우승자와 그 iteration 1 다리들, 그리고 각 실행의 모양."""
    winners: dict[str, Winner] = {}
    shapes: dict[str, dict[str, Any]] = {}
    for arm, prefix in ARMS:
        directory = ARTIFACTS_DIR / f"{prefix}-{DATASET}-seed{seed}"
        winner = loop_winner(arm, directory, metric)
        if winner is None:
            out.missing_arms.append(arm)
            continue
        winners[arm] = winner
        winners[f"{arm}@it1"] = first_iteration_winner(arm, directory, metric)
        shapes[arm] = run_shape(directory)
    return winners, shapes


def _recorded_legs(seed: int, metric: str, out: DatasetVerdict) -> tuple[dict[str, Winner], dict[str, Any]]:
    """이 시드에서의 ``REPLAN.md`` 자신의 팔들. 전이 관측 둘을 위한 다리로 쓴다."""
    winners: dict[str, Winner] = {}
    shapes: dict[str, dict[str, Any]] = {}
    wanted = [RECORDED_RULES]
    if seed in RECORDED_LOOP_SEEDS:
        wanted.append(RECORDED_LOOP)
    for arm, prefix in wanted:
        directory = ARTIFACTS_DIR / f"{prefix}-{DATASET}-seed{seed}"
        winner = loop_winner(arm, directory, metric)
        if winner is None:
            out.missing_arms.append(arm)
            continue
        winners[arm] = winner
        shapes[arm] = run_shape(directory)
    return winners, shapes


def _pairs_to_take(scored: dict[str, Scored], fits: int) -> list[tuple[str, str, str]]:
    """이 시드가 받칠 수 있는 비교들. ``REPLAN-SEEDS.md``가 나열한 순서대로."""
    pairs: list[tuple[str, str, str]] = [(*PRIMARY, "primary")]
    matched = f"random@k={fits}"
    if matched in scored and "random@k=1" in scored:
        # 같은 추가 예산이 아무것도 읽지 않는 검색에게 사 주는 것. 양의 주 판정 옆에 반드시
        # 있어야 하므로, 요청받을 때가 아니라 무조건 계산한다.
        pairs.append((matched, "random@k=1", "blind_gain"))
    if matched in scored:
        pairs.append(("hard_llm2", matched, "matched_level"))
    if "hard_nollm2" in scored:
        pairs.append(("hard_llm2", "hard_nollm2", "vs_rules"))
        pairs.append(("hard_nollm2", "hard_nollm2@it1", "rules_gain"))
        if "hard_nollm" in scored:
            # 규칙 팔은 모델을 부르지 않으므로, 이 쌍은 코드가 움직였을 때만 움직인다.
            pairs.append(("hard_nollm2", "hard_nollm", "code_shift"))
    if "hard_llm" in scored:
        # 한 행에 프롬프트 *와* 코드가 함께 있다 — 이것이 기준이 아니라 관찰인 이유다.
        pairs.append(("hard_llm2", "hard_llm", "prompt_shift"))
    return pairs


def _adjudicate(
    seed: int, resamples: int, out: DatasetVerdict, legs: dict[str, Scored] | None = None
) -> None:
    dataset = BY_NAME[DATASET]
    metric = dataset.metric
    winners, shapes = _arm_legs(seed, metric, out)
    if "hard_llm2" not in winners:
        # 데이터에 대한 거부가 아니다. 이 기준이 다루는 실행이 아직 일어나지 않았다는 뜻이다.
        raise Refusal(
            f"{DATASET} 시드 {seed}: hard_llm2 실행이 없습니다 — "
            "bench/scripts/run_replan_seeds.sh를 먼저 돌려야 판정할 수 있습니다"
        )
    recorded, recorded_shapes = _recorded_legs(seed, metric, out)
    winners.update(recorded)
    shapes.update(recorded_shapes)

    random_dir = random_run_dir(dataset, seed)
    if (random_dir / "summary.json").exists():
        for k in sorted({1, winners["hard_llm2"].fits}):
            if k < 1:
                raise Refusal(f"{DATASET} 시드 {seed}: hard_llm2의 학습 횟수가 {k}회로 기록됐습니다")
            leg = random_winner(random_dir, k, metric)
            winners[leg.arm] = leg
    else:
        out.missing_arms.append("random")

    scored: dict[str, Scored] = {}
    for name, winner in winners.items():
        result = reproduce(winner, metric)
        # iteration 1 다리들은 test 점수가 기록된 적이 없어 nan이다 — holdout은 루프가 끝난 뒤
        # 우승자에 대해 한 번만 돈다. 이유는 ``bench/replan.py``의 ``first_iteration_winner``에 있다.
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
        raise Refusal(f"{DATASET} 시드 {seed}: 팔마다 test 지문이 다릅니다 ({listing}) — 판정하지 않습니다")
    identities = {
        name: json.dumps(split_identity(w.config), sort_keys=True) for name, w in winners.items()
    }
    if len(set(identities.values())) > 1:
        raise Refusal(
            f"{DATASET} 시드 {seed}: 분할을 정하는 config 필드가 팔마다 다릅니다 "
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

    for a_name, b_name, kind in _pairs_to_take(scored, winners["hard_llm2"].fits):
        out.pairs.append(
            paired_delta(scored[a_name], scored[b_name], metric, resamples, RESAMPLE_SEED, kind)
        )

    # 마지막에 둔다. 중간에 거부되면 다리가 남지 않는다 — 쌍이 비워진 항목의 다리를 담은 번들은,
    # 이 판정이 내놓지 않는 수를 ``recheck``에게 내미는 셈이 된다.
    if legs is not None:
        legs.update(scored)


def adjudicate(seed: int, resamples: int, legs: dict[str, Scored] | None = None) -> DatasetVerdict:
    """시드 하나. 거부는 ``paired.py``처럼 쌍을 비운다."""
    dataset = BY_NAME[DATASET]
    out = DatasetVerdict(
        dataset=f"{DATASET}-seed{seed}", metric=dataset.metric, task=dataset.task
    )
    try:
        _adjudicate(seed, resamples, out, legs)
    except Refusal as exc:
        out.verdict = "refused"
        out.refusal = str(exc)
        out.pairs = []
    return out


# --------------------------------------------------------------------------- #
# 사전 등록된 기준
# --------------------------------------------------------------------------- #


def _primary(result: DatasetVerdict) -> Delta | None:
    a, b = PRIMARY
    return next(
        (p for p in result.pairs if p.a == a and p.b == b and p.kind == "primary"), None
    )


def criterion(results: dict[int, DatasetVerdict]) -> dict[str, Any]:
    """이 실행들 이전에 ``docs/REPLAN-SEEDS.md``에 커밋된 기준을 판정한다.

    > 판정 대상인 시드 **전부**에서 ``hard_llm2``의 우승자가 자기 실행의 iteration 1을 짝지은
    > Δ의 95% CI가 0을 걸치지 않게 이기고, 그 Δ들의 **최솟값이 0.0122를 넘는다.**

    과반이 아니라 만장일치다. n=3에서 셋 중 둘 규칙은 동전과 구별하기 어렵고, 시험 중인 주장은
    측정된 이득 하나가 재현된다는 것이지 때때로 재현된다는 것이 아니다.

    크기 바는 CI가 못 하는 일을 한다. 0을 벗어난 짝 구간은 두 다리가 이 행들에서 다르다고 말할 뿐,
    그 차이가 이 벤치마크가 이 팔에서 이미 측정한 실행 간 잡음보다 크다고는 말하지 않는다
    (``REPEATS.md``: 반폭 중앙값 0.0122). 아주 작은 양의 Δ 셋도 "CI 셋 다 0 위"는 만족시키지만
    +0.0935를 재현하지는 않는다.

    우선순위는 미리 못박아 두어 어떤 결과도 자기에게 유리한 읽기를 고를 수 없게 한다: 시드가 없거나
    거부되면 부분 판정. 루프가 실제로 돈 시드가 둘보다 적으면 "판정 불가"(측정된 무효가 아니라 깨진
    전제). CI가 0 아래인 시드가 있으면 부호가 시드에 뒤집힌다고 적는다. 그다음이 만장일치와 크기.
    그 밖은 전부 "구분되지 않음".

    루프가 재계획을 아예 하지 않은 시드(``critic_runs == 0``)는 이기지 못한 것으로 세지 않고
    분모에서 뺀다 — 돌지 않은 루프에 대해 재계획에 값을 물리면 깨진 조건이 증거로 읽힐 수 있다. 바는
    세 시드 전부에 대해 미리 계산해 두었고 셋 다 기록된 규칙 팔의 최고보다 훨씬 위이므로 이 경로는
    예상되지 않는다. 여기 있는 이유는 예상이 보장은 아니기 때문이다.
    """
    judged: list[str] = []
    wins: list[str] = []
    losses: list[str] = []
    ties: list[str] = []
    not_looped: list[str] = []
    absent: list[str] = []
    deltas: dict[str, float] = {}
    for seed in SEEDS:
        label = f"seed{seed}"
        result = results.get(seed)
        pair = None if result is None else _primary(result)
        critic_runs = None if result is None else result.arms.get("hard_llm2", {}).get("critic_runs")
        if result is None or result.verdict != "ok" or pair is None:
            absent.append(label)
            continue
        if not isinstance(critic_runs, int) or critic_runs < 1:
            not_looped.append(label)
            continue
        judged.append(label)
        if pair.identical_predictions:
            # 우승자가 iteration 1이다. 예산은 썼고 이긴 것은 없고 읽을 구간도 없다.
            ties.append(label)
            continue
        deltas[label] = pair.delta
        if pair.ci_low > 0:
            wins.append(label)
        elif pair.ci_high < 0:
            losses.append(label)
        else:
            ties.append(label)

    win_deltas = [deltas[label] for label in wins]
    smallest = min(win_deltas) if win_deltas else float("nan")
    unanimous = bool(judged) and len(wins) == len(judged)
    tally = f"이김 {len(wins)} / 걸침 {len(ties)} / 짐 {len(losses)}"
    if not_looped:
        tally += f" / 루프 안 돔 {len(not_looped)}"

    if absent:
        complete = False
        status = (
            f"부분 판정 — 기준의 범위는 시드 {len(SEEDS)}개인데 판정할 수 있는 것이 "
            f"{len(judged)}개입니다 (없음: {', '.join(absent)}). 지금까지 {tally}"
        )
    elif len(judged) < 2:
        complete = False
        status = (
            f"판정 불가 — 루프가 돈 시드가 {len(judged)}개뿐입니다 "
            f"(안 돈 것: {', '.join(not_looped) or '없음'}). 올린 바에서도 재계획이 돌지 "
            "않았다면 이 실험의 전제가 깨진 것이고, 기준을 약하게 고치지 않습니다"
        )
    elif losses:
        complete = True
        status = (
            f"부호가 시드에 뒤집힙니다 — {', '.join(losses)}에서 CI가 0 아래입니다 ({tally}). "
            "재계획 이득이 시드를 견딘다고 적을 수 없습니다"
        )
    elif unanimous and smallest > NOISE_FLOOR:
        complete = True
        status = (
            f"기준 충족 — 판정한 시드 {len(judged)}개 전부에서 재계획이 자기 iteration 1을 "
            f"이겼고, 가장 작은 Δ가 {smallest:+.4f}로 잡음 바닥 {NOISE_FLOOR}를 넘습니다. "
            f"blind_gain과 hard_nollm2 시드 산포를 반드시 함께 인용합니다 (기록된 관측 "
            f"{RECORDED_PRIMARY:+.4f}, 규칙 팔 시드 산포 {RULES_SEED_SPREAD})"
        )
    elif unanimous:
        complete = True
        status = (
            f"구분되지 않음 — 시드 {len(judged)}개 전부 양수이지만 가장 작은 Δ "
            f"{smallest:+.4f}가 잡음 바닥 {NOISE_FLOOR} 이하입니다. 부호는 견디고 크기는 "
            "견디지 않습니다"
        )
    else:
        complete = True
        status = f"구분되지 않음 — 만장일치가 아닙니다 ({tally})"

    return {
        "challenger": PRIMARY[0],
        "baseline": PRIMARY[1],
        "dataset": DATASET,
        "scope_seeds": list(SEEDS),
        "resample_seed": RESAMPLE_SEED,
        "judged": judged,
        "wins": wins,
        "losses": losses,
        "indistinguishable": ties,
        "not_looped": not_looped,
        "not_evaluated": absent,
        "unanimous": unanimous,
        "primary_deltas": deltas,
        "smallest_win": smallest,
        "noise_floor": NOISE_FLOOR,
        "recorded_primary": RECORDED_PRIMARY,
        "rules_seed_spread": RULES_SEED_SPREAD,
        "spread_of_primary": (max(deltas.values()) - min(deltas.values())) if deltas else float("nan"),
        "median_of_primary": statistics.median(deltas.values()) if deltas else float("nan"),
        "complete": complete,
        "status": status,
    }


def budget_used(results: dict[int, DatasetVerdict]) -> list[dict[str, Any]]:
    """시드마다 각 팔이 쓴 것 — 학습 횟수, 종료 이유, Critic 실행 횟수, 마주한 바.

    정체 가드가 미달인 바에서도 ``max_iterations`` 전에 실행을 끝낼 수 있어서 처치의 실제 크기가
    시드마다 다르고, ``critic_runs``가 한 시드를 기준의 범위에 넣는 값이다.
    """
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        result = results.get(seed)
        if result is None:
            continue
        row: dict[str, Any] = {"seed": seed}
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


def payload(results: dict[int, DatasetVerdict], resamples: int) -> dict[str, Any]:
    return {
        "generated_by": "bench/replan_seeds.py",
        "preregistration": "docs/REPLAN-SEEDS.md",
        "reproduces": "docs/REPLAN.md",
        "dataset": DATASET,
        "margin": 0.5,
        "run_seeds": [seed for seed in SEEDS if seed in results],
        # 부트스트랩 시드. 파일 전체에 하나다 — ``bench/recheck.py``가 판정마다 ``seed`` 하나를
        # 읽고 그것으로 모든 쌍을 다시 계산한다. 모듈 docstring을 보라.
        "seed": RESAMPLE_SEED,
        "resamples": resamples,
        "score_tolerance": SCORE_TOLERANCE,
        "threads": thread_state(),
        "datasets": [
            {
                "dataset": r.dataset,
                "run_seed": seed,
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
            for seed, r in ((s, results[s]) for s in SEEDS if s in results)
        ],
        "budget": budget_used(results),
        "criterion": criterion(results),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "speeddating의 재계획 이득이 시드와 새 프롬프트를 견디는가 "
            "(docs/REPLAN-SEEDS.md의 사전 등록 기준)"
        )
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
        help=f"기본: bench/runs/paired/replan-seeds-{DATASET}.json",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    unknown = [s for s in args.seeds if s not in SEEDS]
    if unknown:
        print(
            f"사전 등록의 범위 밖 시드: {', '.join(str(s) for s in unknown)} "
            f"(범위: {', '.join(str(s) for s in SEEDS)})",
            file=sys.stderr,
        )
        return 2

    print(
        f"재계획 이득의 시드 판정 · {DATASET} · margin 0.5 · 재추출 {args.resamples}회 "
        f"(재추출 시드 {RESAMPLE_SEED}) · {describe_thread_state(thread_state())}"
    )

    legs: dict[str, dict[str, Scored]] = {}
    results: dict[int, DatasetVerdict] = {}
    for seed in SEEDS:
        if seed not in args.seeds:
            continue
        collected: dict[str, Scored] = {}
        results[seed] = adjudicate(seed, args.resamples, collected)
        if collected:
            legs[results[seed].dataset] = collected
    report([results[seed] for seed in SEEDS if seed in results])

    out_path = args.out or OUT_DIR / f"replan-seeds-{DATASET}.json"
    if not legs:
        # 채점된 것이 없으므로 내놓을 것도, 그것을 담을 번들도 없다. 여기서 판정 파일을 쓰면
        # ``python -m bench.recheck``가 인수 없이 읽는 디렉터리에 앉아 그것을 실패시킨다 — 이
        # 저장소의 주장은 발표된 모든 Δ가 다시 계산된다는 것이고, 아무것도 발표하지 않는 파일은
        # 실패한 Δ가 아니라 아직 일어나지 않은 실행이다.
        print(
            f"판정할 다리가 없습니다 — {out_path.as_posix()}를 쓰지 않았습니다. "
            "bench/scripts/run_replan_seeds.sh를 먼저 돌리세요",
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
    print("예산 — 실제로 쓴 것 (정체 가드가 미달인 바에서도 먼저 걸릴 수 있습니다):")
    for row in budget_used(results):
        for arm, _ in ARMS:
            spent = row[arm]
            print(
                f"   시드 {row['seed']} {arm:<12} 학습={spent['fits']} "
                f"종료={spent['stop_reason']} critic={spent['critic_runs']} "
                f"바={spent['goal_threshold']} 우승={spent['winner']}"
            )
    print()
    print(f"기준: {criterion(results)['status']}")
    print(f"판정 → {out_path.as_posix()}")
    refused = [r.dataset for r in results.values() if r.verdict != "ok"]
    if refused:
        print(f"거부된 시드: {', '.join(refused)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
