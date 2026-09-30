"""재계획 팔의 판정: 미달인 바에서 루프가 예산을 점수로 바꾸는가?

사전 등록은 ``docs/REPLAN.md``이고, 이 실행들이 존재하기 전에 커밋됐다. 이 문서가 떼어 내는 질문은
앞선 측정 셋이 각각 놓친 그것이다. ``RESULTS.md``는 ``llm`` 팔의 예산을 정직하게 맞췄다 — 그런데 그
예산이 한두 번의 적합이어서 맞춰진 상대가 ``random@k=1``이었고, 비교는 정보를 읽은 계획 하나 대 눈
감은 추출 하나가 됐다. ``SPG.md``는 Critic 열네 번 분량으로 루프를 실제로 돌렸지만 바가 이미
넘겨진 상태였다 — 진단이 파고들 미달이 없었다. ``REPEATS.md``는 처치가 아니라 잡음을 쟀다.

그래서 이 실험은 바를 손이 닿지 않는 곳까지 올리고(``--margin 0.5``) 재계획이 *거기서* 값을 내는지
묻는다. 바를 일부러 올렸다는 사실은 사전 등록과 이 docstring에 밝혀 둔다. 그것이 iteration 1의
계획도 바꾸기 때문이다 — 문턱이 계획 프롬프트에 렌더된다 — 그래서 이 수들은 ``RESULTS.md``의 표에
절대 들어가지 않는다.

기준이 왜 실행 안쪽의 쌍인가. ``SPG.md``의 결과가 이 벤치마크에 단단한 발견 하나를 남겼다:
계획자는 결정적이지 않고, 어떤 처치도 일으킬 수 없었던 iteration 1의 val 차이(0.0018~0.0063)가
쫓고 있는 주 Δ(0.0010~0.0084)와 같은 자릿수였다. 실행마다 한 번씩 돈 두 팔을 서로 빼면 그 항이 Δ에
그대로 들어간다. 한 팔의 우승자를 *그 팔 자신의 iteration 1*과 견주면 — 같은 프롬프트, 같은 실행,
같은 test 행 — 그렇지 않다. 값은, 답하는 질문이 "루프가 검색을 이기는가"가 아니라 "루프가 자기 첫
시도를 이기는가"가 된다는 것이다. 그래서 ``REPLAN.md``는 양의 판정 옆에 눈 감은 검색의
이득(``random@k=fits − random@k=1``)을 반드시 함께 인용하라고 못박는다.

새 팔 둘은 ``--keep-models all``로 돌려야 한다. 기본값은 우승하지 않은 iteration의
``model.joblib``을 전부 지우고, 여기서 iteration 1은 보통 우승자가 아니다.

팔들이 돌았던 것과 같이 ``OMP_NUM_THREADS=1``에서 돌린다. 그 값은 출력에 기록된다.

사용법:
    python -m bench.replan                    # 범위에 든 데이터셋 셋
    python -m bench.replan adult --seed 42
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

from automl_agent.config import MODEL_FILENAME, read_json_object
from automl_agent.threads import describe_thread_state, thread_state
from bench.datasets import BY_NAME, JUDGED_SEED, Dataset
from bench.paired import (
    ARTIFACTS_DIR,
    OUT_DIR,
    RESAMPLES,
    SCORE_TOLERANCE,
    DatasetVerdict,
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

# 올린 바에서 돌린 팔 둘. ``hard_nollm``은 결정적 대조군이다 — 이쪽도 재계획을 하는데, 모델이
# 아니라 규칙에서 한다.
ARMS: tuple[tuple[str, str], ...] = (("hard_llm", "hardllm"), ("hard_nollm", "hardnollm"))

# 이진만, 그리고 넷 다도 아니다. ``bank-marketing``의 바는 margin 0.25와 0.5 사이에서 움직이지
# 않는다 — 양쪽 다 랭킹 하한 0.8400이 margin 바 0.8310을 누른다 — 그래서 거기서 돌리면 무엇을
# 시험하는 것이 아니라 ``llm-bank-marketing-seed42``를 반복하는 것이고, 반복은 ``REPEATS.md``의
# 주제다. 회귀는 ``SPG.md``와 같은 이유로 빠진다: 이 실험이 다루는 이진 진단 경로를 밟지 않는다
# (``balanced_accuracy_cut_headroom``, 랭킹 상한, ``class_weight``). ``bench.datasets``에서 끌어오지
# 않고 여기 못박아 두어 범위가 사후에 넓어질 수 없게 한다.
SCOPE: tuple[str, ...] = ("adult", "spambase", "speeddating")


def run_shape(directory: Path) -> dict[str, Any]:
    """루프가 스스로를 어떻게 썼는가: 왜 멈췄고, Critic이 몇 번 돌았고, 어떤 바를 마주했는가.

    학습 횟수는 일부러 여기 두지 않는다 — ``loop_winner``가 이미 같은 ``iterations`` 필드에서 읽어
    ``Winner.fits``에 담고, 커밋되는 판정 파일에 한 수의 이름이 둘이면 읽는 사람이 결국 그 둘을
    견주게 된다.

    ``critic_runs``는 한 데이터셋이 기준의 범위에 드는지를 아예 결정하는 값이다 — iteration 1에서
    멈춘 실행은 재계획을 하지 않았으므로 재계획에 대해 할 말이 없다. ``iterations - 1``이 아니라
    시도들에서 세는 이유는, 판정이 없는 시도는 iteration 수가 무엇이라 하든 진단이 아니기 때문이다.
    """
    history = read_json_object(directory / "history.json") or {}
    attempts = history.get("history") or []
    return {
        "stop_reason": history.get("stop_reason"),
        "critic_runs": sum(1 for a in attempts if isinstance(a, dict) and a.get("critic")),
        "goal_threshold": dict(history.get("goal") or {}).get("threshold"),
    }


def first_iteration_winner(arm: str, directory: Path, metric: str) -> Winner:
    """실행의 iteration 1을, 같은 짝짓기 장치를 통과할 수 있도록 ``Winner``로 감싼 것.

    ``recorded_test``가 ``nan``인 것은 빠뜨린 것이 아니다. iteration 1의 test 점수는 애초에 기록된
    적이 없다 — 실행 안의 어느 것도 그것을 볼 수 없었고, ``holdout``은 루프가 끝난 뒤 우승자에 대해
    한 번만 돈다. 그래서 이 다리는 다른 다리들이 지고 있는 재채점 교차확인을 지지 못하고,
    ``_adjudicate``가 이 다리에서만 그것을 건너뛴다. ``REPLAN.md``가 그 누락을 적어 두었다.

    살아남는 확인은 짝짓기에 중요한 쪽이다: config의 분할 필드를 다른 모든 다리와 견주고, test
    지문을 다른 모든 다리와 견주고, ``model.joblib``이 없으면 비교를 조용히 떨어뜨리지 않고 그
    데이터셋을 거부한다 — 기본값 ``--keep-models best``로 남겨 둔 실행이 만드는 상태가 그것이다.
    """
    history = read_json_object(directory / "history.json")
    if history is None:
        raise Refusal(f"{arm}: history.json을 읽을 수 없습니다 ({directory})")
    attempts = history.get("history") or []
    if not attempts:
        raise Refusal(f"{arm}: history가 비어 있습니다 ({directory})")
    first = attempts[0]
    val = dict(first.get("metrics") or {}).get(metric)
    if not isinstance(val, (int, float)):
        raise Refusal(f"{arm}: iteration 1의 {metric}이 history에 없습니다 — 실패한 시도입니다")
    iter_dir = directory / "train" / "iter_01"
    config = read_json_object(iter_dir / "train_config.json")
    if config is None:
        raise Refusal(f"{arm}: iteration 1의 train_config.json을 읽을 수 없습니다 ({iter_dir})")
    if not (iter_dir / MODEL_FILENAME).exists():
        raise Refusal(
            f"{arm}: iteration 1의 {MODEL_FILENAME}이 없습니다 — 이 실험은 "
            "--keep-models all 로 돌려야 합니다 (기본값은 우승자 아닌 iteration을 지웁니다)"
        )
    return Winner(
        arm=f"{arm}@it1",
        label="iteration 1",
        directory=iter_dir,
        config=config,
        val_score=float(val),
        recorded_test=float("nan"),
        fits=1,
    )


def _adjudicate(
    dataset: Dataset,
    seed: int,
    resamples: int,
    out: DatasetVerdict,
    legs: dict[str, Scored] | None = None,
) -> None:
    metric = dataset.metric
    winners: dict[str, Winner] = {}
    shapes: dict[str, dict[str, Any]] = {}
    for arm, prefix in ARMS:
        directory = ARTIFACTS_DIR / f"{prefix}-{dataset.name}-seed{seed}"
        winner = loop_winner(arm, directory, metric)
        if winner is None:
            out.missing_arms.append(arm)
            continue
        winners[arm] = winner
        winners[f"{arm}@it1"] = first_iteration_winner(arm, directory, metric)
        shapes[arm] = run_shape(directory)

    if "hard_llm" not in winners:
        # 데이터에 대한 거부가 아니다. 이 기준이 다루는 실행이 아직 일어나지 않았다는 뜻이다.
        # ``adjudicate``는 기록된 팔들이 실제로 낸 쌍은 그대로 남긴다.
        raise Refusal(
            f"{dataset.name}: 시드 {seed}의 hard_llm 실행이 없습니다 — "
            "bench/scripts/run_replan.sh를 먼저 돌려야 판정할 수 있습니다"
        )

    # 눈 감은 검색 다리들. ``k``는 LLM 팔 자신의 학습 횟수이므로 수준 비교가 예산이 맞춰진 것이
    # 되고, ``k=1``은 그 팔의 출발점이므로 이득 비교가 양쪽 다 적합 한 번에서 시작한다.
    random_dir = random_run_dir(dataset, seed)
    if (random_dir / "summary.json").exists():
        for k in sorted({1, winners["hard_llm"].fits}):
            if k < 1:
                raise Refusal(f"{dataset.name}: hard_llm의 학습 횟수가 {k}회로 기록됐습니다")
            leg = random_winner(random_dir, k, metric)
            winners[leg.arm] = leg
    else:
        out.missing_arms.append("random")

    scored: dict[str, Scored] = {}
    for name, winner in winners.items():
        result = reproduce(winner, metric)
        # iteration 1 다리들은 test 점수가 기록된 적이 없어 nan이다.
        # ``first_iteration_winner``를 보라.
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
        raise Refusal(f"{dataset.name}: 팔마다 test 지문이 다릅니다 ({listing}) — 판정하지 않습니다")
    identities = {
        name: json.dumps(split_identity(w.config), sort_keys=True) for name, w in winners.items()
    }
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
            # 검증 슬라이스 하나에서 여러 시도 중 최고를 고른 다중성의 값. 여기서는 부차
            # 지표가 아니라 핵심이다 — 기준의 Δ는 이 비용을 *치른 뒤*의 루프 이득이고, val의
            # 이득이 test에 나타나지 못하게 하는 것이 이것이다.
            "selection_gap": s.winner.val_score - s.test_score,
            "dir": s.winner.directory.as_posix(),
            **shapes.get(name, {}),
        }

    matched = f"random@k={winners['hard_llm'].fits}"
    pairs: list[tuple[str, str, str]] = [("hard_llm", "hard_llm@it1", "primary")]
    if matched in scored and "random@k=1" in scored:
        # 같은 추가 예산이 아무것도 읽지 않는 검색에게 사 주는 것. REPLAN.md가 양의 주 판정
        # 옆에 이 수를 반드시 두라고 못박았으므로, 요청받을 때가 아니라 무조건 계산한다.
        pairs.append((matched, "random@k=1", "blind_gain"))
    if matched in scored:
        pairs.append(("hard_llm", matched, "matched_level"))
    if "hard_nollm" in scored:
        pairs.append(("hard_llm", "hard_nollm", "vs_rules"))
        pairs.append(("hard_nollm", "hard_nollm@it1", "rules_gain"))

    for a_name, b_name, kind in pairs:
        out.pairs.append(
            paired_delta(scored[a_name], scored[b_name], metric, resamples, seed, kind)
        )

    # 마지막에 둔다. 중간에 거부되면 다리가 남지 않는다. 쌍이 비워진 데이터셋의 다리를 담은
    # 번들은, 이 판정이 일부러 내놓지 않는 수를 ``recheck``에게 검산하라고 내미는 셈이 된다.
    if legs is not None:
        legs.update(scored)


def adjudicate(
    dataset: Dataset, seed: int, resamples: int, legs: dict[str, Scored] | None = None
) -> DatasetVerdict:
    """거부는 ``paired.py``처럼 쌍을 비운다. 아무도 쓸 수 없는 Δ는 없는 것보다 나쁘다."""
    out = DatasetVerdict(dataset=dataset.name, metric=dataset.metric, task=dataset.task)
    try:
        _adjudicate(dataset, seed, resamples, out, legs)
    except Refusal as exc:
        out.verdict = "refused"
        out.refusal = str(exc)
        out.pairs = []
    return out


# --------------------------------------------------------------------------- #
# 사전 등록된 기준
# --------------------------------------------------------------------------- #


def _primary(result: DatasetVerdict) -> Any:
    return next(
        (
            p
            for p in result.pairs
            if p.a == "hard_llm" and p.b == "hard_llm@it1" and p.kind == "primary"
        ),
        None,
    )


def criterion(results: list[DatasetVerdict], seed: int = JUDGED_SEED) -> dict[str, Any]:
    """이 실행들 이전에 ``docs/REPLAN.md``에 커밋된 기준을 판정한다.

    > 판정 대상의 과반에서, ``hard_llm``의 우승자가 자기 실행의 iteration 1을 짝지은 Δ의
    > 95% CI가 0을 걸치지 않게 이긴다.

    여기서 규칙 둘이 단순한 이김 셈이 못 하는 일을 한다.

    ``판정 대상``은 루프가 *돈* 데이터셋이다 — ``critic_runs >= 1``. iteration 1에서 멈춘 실행은
    재계획을 하지 않았으므로, 그것을 이기지 못한 것으로 세면 하지도 않은 일에 대해 재계획에 값을
    물리는 셈이다. 대신 루프를 안 돈 것으로 적는다. 그런 데이터셋이 둘보다 적으면 판정은
    "판정 불가"다 — 이 실험의 전제가 깨진 것이고, 그건 기준을 느슨하게 고쳐 감쌀 결과가 아니라
    보고할 설계 실패다.

    우승자가 *바로* iteration 1인 것은 이김이 아니라 걸침이다 — ``paired_delta``가 동일한 예측으로
    표시하고 구간이 없다. 예산은 썼고 이긴 것은 없다.

    일부러 대칭이다: 반대 방향의 과반은 재계획이 점수를 *깎는다*로 적고, ``REPLAN.md``가 그것도
    결과로 사전 등록해 두었다. 그보다 약한 것은 전부 "구분되지 않음"이다.
    """
    in_scope = [r for r in results if r.dataset in SCOPE]
    wins: list[str] = []
    losses: list[str] = []
    ties: list[str] = []
    not_looped: list[str] = []
    absent: list[str] = []
    for result in in_scope:
        pair = _primary(result)
        critic_runs = result.arms.get("hard_llm", {}).get("critic_runs")
        if result.verdict != "ok" or pair is None:
            absent.append(result.dataset)
        elif not isinstance(critic_runs, int) or critic_runs < 1:
            not_looped.append(result.dataset)
        elif pair.identical_predictions:
            ties.append(result.dataset)
        elif pair.ci_low > 0:
            wins.append(result.dataset)
        elif pair.ci_high < 0:
            losses.append(result.dataset)
        else:
            ties.append(result.dataset)

    judged = wins + ties + losses
    missing = sorted(set(SCOPE) - {r.dataset for r in in_scope}) + sorted(absent)
    tally = f"이김 {len(wins)} / 걸침 {len(ties)} / 짐 {len(losses)}"
    if not_looped:
        tally += f" / 루프 안 돔 {len(not_looped)}"
    needed = len(judged) // 2 + 1

    if seed != JUDGED_SEED:
        complete = False
        status = f"기준 판정 안 함 — 사전 등록은 시드 {JUDGED_SEED}만 판정에 씁니다 ({tally})"
    elif missing:
        complete = False
        status = (
            f"부분 판정 — 기준의 범위는 {len(SCOPE)}개인데 판정할 수 있는 것이 "
            f"{len(judged)}개입니다 (없음: {', '.join(missing)}). 지금까지 {tally}"
        )
    elif len(judged) < 2:
        complete = False
        status = (
            f"판정 불가 — 루프가 돈 데이터셋이 {len(judged)}개뿐입니다 "
            f"(안 돈 것: {', '.join(not_looped) or '없음'}). 바를 올려도 미달이 되지 않았다면 "
            "이 실험의 전제가 깨진 것이고, 기준을 약하게 고치지 않습니다"
        )
    else:
        complete = True
        if len(wins) >= needed:
            status = (
                f"기준 충족 — 루프가 돈 {len(judged)}개 중 {len(wins)}개에서 재계획이 자기 "
                "iteration 1을 이겼습니다. random의 같은 예산 이득(blind_gain)과 hard_nollm "
                "시드 산포를 반드시 함께 인용합니다"
            )
        elif len(losses) >= needed:
            status = f"재계획이 점수를 내립니다 — {len(losses)}개에서 CI가 0 아래입니다 ({tally})"
        else:
            status = f"구분되지 않음 — 이 데이터셋들과 이 예산으로는 답이 나오지 않습니다 ({tally})"
    return {
        "challenger": "hard_llm",
        "baseline": "hard_llm@it1",
        "seed": seed,
        "judged_seed": JUDGED_SEED,
        "scope": list(SCOPE),
        "judged": judged,
        "majority_needed": needed,
        "wins": wins,
        "losses": losses,
        "indistinguishable": ties,
        "not_looped": not_looped,
        "not_evaluated": missing,
        "complete": complete,
        "status": status,
    }


def budget_used(results: list[DatasetVerdict]) -> list[dict[str, Any]]:
    """데이터셋마다 각 팔이 쓴 것 — 학습 횟수, 종료 이유, Critic 실행 횟수.

    ``REPLAN.md``가 이것을 Δ 옆에 두라고 요구하는 이유는 둘이다. 정체 가드가 미달인 바에서도
    ``max_iterations`` 전에 실행을 끝낼 수 있어서 처치의 실제 크기가 데이터셋마다 다르고,
    ``critic_runs``가 한 데이터셋을 기준의 범위에 넣는 값이기 때문이다.
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {"dataset": result.dataset}
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


def payload(results: list[DatasetVerdict], seed: int, resamples: int) -> dict[str, Any]:
    return {
        "generated_by": "bench/replan.py",
        "preregistration": "docs/REPLAN.md",
        "margin": 0.5,
        "seed": seed,
        "resamples": resamples,
        "score_tolerance": SCORE_TOLERANCE,
        # 실행들이 쓰는 것과 같은 기록기다. 판정과 그 아래 시도들이 같은 환경을 적게 된다 —
        # 환경 변수만으로는 정해지지 않는 ``cpu_count``까지.
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
        "budget": budget_used(results),
        "criterion": criterion(results, seed),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="미달인 바에서 재계획이 예산을 점수로 바꾸는가 (docs/REPLAN.md의 사전 등록 기준)"
    )
    parser.add_argument("datasets", nargs="*", default=[], help=f"이름 (기본: {', '.join(SCOPE)})")
    parser.add_argument("--seed", type=int, default=JUDGED_SEED)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument(
        "--out", type=Path, default=None, help="기본: bench/runs/paired/replan-seed<seed>.json"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = args.datasets or list(SCOPE)
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        print(f"모르는 데이터셋: {', '.join(unknown)}", file=sys.stderr)
        return 2

    print(
        f"재계획 판정 · margin 0.5 · 시드 {args.seed} · 재추출 {args.resamples}회 · "
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

    out_path = args.out or OUT_DIR / f"replan-seed{args.seed}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(payload(results, args.seed, args.resamples), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    # 이 수들이 나온 모델은 커밋되지 않고, 그랬던 적도 없다. 그래서 이것이 없으면 위의 Δ는
    # 그것을 만든 기계 말고는 아무도 검산할 수 없다. bench/recheck.py를 보라.
    if legs:
        target = bundle_path(out_path)
        write_bundle(target, legs, {r.dataset: r.metric for r in results})
        print(f"예측 번들 → {target.as_posix()} ({target.stat().st_size / 1024:.0f} KB)")
    print()
    print("예산 — 실제로 쓴 것 (정체 가드가 미달인 바에서도 먼저 걸릴 수 있습니다):")
    for row in budget_used(results):
        for arm, _ in ARMS:
            spent = row[arm]
            print(
                f"   {row['dataset']:<14} {arm:<11} 학습={spent['fits']} "
                f"종료={spent['stop_reason']} critic={spent['critic_runs']} "
                f"바={spent['goal_threshold']} 우승={spent['winner']}"
            )
    print()
    print(f"기준: {criterion(results, args.seed)['status']}")
    print(f"판정 → {out_path.as_posix()}")
    refused = [r.dataset for r in results if r.verdict != "ok"]
    if refused:
        print(f"거부된 데이터셋: {', '.join(refused)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
