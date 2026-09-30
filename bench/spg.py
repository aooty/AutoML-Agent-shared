"""``--search-past-goal`` 팔을 판정한다: 바를 넘긴 뒤에도 예산을 쓰는 것이 값을 하는가?

사전 등록은 ``docs/SPG.md``이고, 이 실행들이 하나도 없던 때 커밋됐다. 질문은 한 문장이다: 루프의 첫
시도가 바를 넘겨 iteration 1에서 실행이 끝나는 일이 잦다 — ``RESULTS.md``의 ``llm`` 실행 다섯 중 넷이
그랬고 Critic은 0번 돌았다 — 그러니 대신 루프를 계속 돌리면 *test* 점수가 어떻게 되는가.

``paired.py``와 다른 파일인 이유. 그 스크립트의 기준은 ``RESULTS.md``에 사전 등록된 것이고
(``llm`` 대 ``random``, 예산 맞춤, 이진 4개 중 3개) 그 뒤의 모든 수는 이 플래그를 끈 채 측정됐다.
거기에 팔을 더하면 ``bench/runs/paired/seed42.json``을 다시 쓰게 되는데, 그 산출물의 값은 미리 고정된
계약이 만들어 냈다는 점 하나다. 그래서 공용 기계는 import하고 판정은 다른 데 쓴다.

대조 팔을 ``RESULTS.md``에서 읽지 않고 다시 돌리는 이유. 둘인데 두 번째가 진짜다. 코드가 한 커밋
차이다(플래그를 끄면 동작이 보존된다고 논증했지만 논증은 측정이 아니다). 그리고 ``RESULTS.md``는 자기
한계 절에서 ``llm`` 팔의 재현성을 측정한 적이 없다고 말한다: 시드 하나, 실행 한 번, 비결정적 응답자.
기록된 실행을 빼면 델타에 *크기를 모르는* 항이 들어가는데, 이 벤치마크가 random 팔의 시드 폭
0.1472에서 이미 한 번 배운 실수다.

그 재실행이 공짜로 하나를 준다. 기준이 아니라 관찰로만 기록한다: ``spg_off``와 기록된 ``llm`` 실행의
비교는 ``llm`` 수 하나가 실행끼리 얼마나 움직이는지에 대한 첫 읽기다.

팔들이 돌아간 것과 같이 ``OMP_NUM_THREADS=1``로 돌린다. 그 값은 출력에 기록된다.

사용법:
    python -m bench.spg                       # 판정 시드에서 데이터셋 전부
    python -m bench.spg adult --seed 42
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from automl_agent.threads import describe_thread_state, thread_state
from bench.datasets import ALL, BY_NAME, JUDGED_NAMES, JUDGED_SEED, Dataset
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
    report,
    reproduce,
    split_identity,
)
from bench.predictions import bundle_path, write_bundle

# 기준이 대상으로 삼는 두 팔. 같은 코드, 같은 카드, 같은 시드, 플래그 하나 차이.
ARMS: tuple[tuple[str, str], ...] = (("spg_off", "spgoff"), ("spg_on", "spgon"))
# RESULTS.md의 팔. 관찰 짝에만 쓴다 — 다른 커밋이 만든 것이라 대조군이 아니고, 아래 기준은
# 이것을 보지 않는다.
RECORDED: tuple[str, str] = ("llm_recorded", "llm")


def _adjudicate(
    dataset: Dataset,
    seed: int,
    resamples: int,
    out: DatasetVerdict,
    legs: dict[str, Scored] | None = None,
) -> None:
    metric = dataset.metric
    winners: dict[str, Winner] = {}
    for arm, prefix in (*ARMS, RECORDED):
        winner = loop_winner(arm, ARTIFACTS_DIR / f"{prefix}-{dataset.name}-seed{seed}", metric)
        if winner is None:
            out.missing_arms.append(arm)
        else:
            winners[arm] = winner

    missing = [arm for arm, _ in ARMS if arm not in winners]
    if missing:
        raise Refusal(
            f"{dataset.name}: 시드 {seed}에서 {', '.join(missing)} 실행을 찾지 못했습니다 — "
            "이 판정은 두 팔이 다 있어야 성립합니다"
        )

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
            # validation 한 조각에서 여러 시도 중 최고를 고르는 다중성의 값. 여기서는 부수
            # 지표가 아니다 — ``spg_on``은 구조상 더 많은 시도에서 고르므로, 점수를 깎는
            # 플래그라면 여기서 드러난다.
            "selection_gap": s.winner.val_score - s.test_score,
            "dir": s.winner.directory.as_posix(),
        }

    # 기준의 짝. 델타가 양수면 플래그가 도운 것이 되게 ``spg_on``을 앞에 둔다.
    out.pairs.append(paired_delta(scored["spg_on"], scored["spg_off"], metric, resamples, seed, "primary"))
    if RECORDED[0] in scored:
        # 기준이 아니라 관찰: 거의 같은 코드의 두 실행이므로 대부분 응답자의 실행 간 잡음이다.
        # RESULTS.md는 이걸 측정한 적이 없고 거기의 모든 ``llm`` 델타는 이것 없이 인용된다.
        out.pairs.append(
            paired_delta(scored["spg_off"], scored[RECORDED[0]], metric, resamples, seed, "observation")
        )

    # 중간에 거부되면 leg가 남지 않도록 마지막에 한다. 짝이 지워진 데이터셋의 leg를 담은 번들은
    # 판정이 일부러 공표하지 않은 수를 ``recheck``에 검증거리로 내주게 된다.
    if legs is not None:
        legs.update(scored)


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


# --------------------------------------------------------------------------- #
# 사전 등록된 기준
# --------------------------------------------------------------------------- #


def criterion(results: list[DatasetVerdict], seed: int = JUDGED_SEED) -> dict[str, Any]:
    """이 실행들이 하나도 없던 때 ``docs/SPG.md``에 커밋된 기준을 평가한다.

    > 이진 4개 중 3개 이상에서, `spg_on`이 `spg_off`를 짝지은 Δ의 95% CI가 0을 걸치지 않게
    > 이긴다.

    일부러 대칭이다: 반대 방향으로 3개 이상이면 플래그가 점수를 *깎는다*고 보고하고, ``SPG.md``는
    그걸 실망이 아니라 결과로 사전 등록한다. 그보다 약하면 "구분되지 않음"이고, 고정된 넷보다 적게
    덮은 범위는 부분 판정이라 이름 붙인다 — 묻지 않은 질문은 차이 없음이라는 답이 아니다.

    이진만. 회귀 데이터셋은 Critic의 다른 절반을 지나가므로
    (``balanced_accuracy_cut_headroom`` 없음, class weight 없음, 랭킹 상한 없음) 집계에 섞지 않고
    따로 관찰한다.
    """
    expected = set(JUDGED_NAMES)
    binary = [r for r in results if r.dataset in expected]
    wins: list[str] = []
    losses: list[str] = []
    ties: list[str] = []
    absent: list[str] = []
    for result in binary:
        pair = next(
            (p for p in result.pairs if p.a == "spg_on" and p.b == "spg_off" and p.kind == "primary"),
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
            f"기준 판정 안 함 — 사전 등록은 시드 {JUDGED_SEED}만 판정에 씁니다 ({tally})"
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
                f"기준 충족 — 이 {len(expected)}개 중 {len(wins)}개에서 --search-past-goal이 "
                "점수를 올렸습니다"
            )
        elif len(losses) >= 3:
            status = (
                f"플래그가 점수를 내립니다 — {len(losses)}개에서 CI가 0 아래입니다 ({tally})"
            )
        else:
            status = f"구분되지 않음 — 이 데이터셋들과 이 예산으로는 답이 나오지 않습니다 ({tally})"
    return {
        "challenger": "spg_on",
        "baseline": "spg_off",
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


def budget_used(results: list[DatasetVerdict]) -> list[dict[str, Any]]:
    """데이터셋마다 각 팔이 실제로 쓴 예산.

    ``SPG.md``는 예산을 맞추지 않는다 — 쓴 양의 차이가 처치다 — 그래서 쓴 양이 델타와 함께 다녀야
    한다. 그러지 않으면 그 수가 공짜로 이긴 것처럼 읽힌다.
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        row: dict[str, Any] = {"dataset": result.dataset}
        for arm, _ in ARMS:
            row[arm] = result.arms.get(arm, {}).get("fits")
        rows.append(row)
    return rows


def payload(results: list[DatasetVerdict], seed: int, resamples: int) -> dict[str, Any]:
    return {
        "generated_by": "bench/spg.py",
        "preregistration": "docs/SPG.md",
        "seed": seed,
        "resamples": resamples,
        "score_tolerance": SCORE_TOLERANCE,
        # 실행들이 쓰는 것과 같은 기록기다. 판정과 그 아래 시도들이 같은 환경을 서술하게
        # 된다 — 환경 변수만으로는 정해지지 않는 ``cpu_count``까지.
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
        "fits": budget_used(results),
        "criterion": criterion(results, seed),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="--search-past-goal 팔의 짝지은 판정 (docs/SPG.md의 사전 등록 기준)"
    )
    parser.add_argument("datasets", nargs="*", default=[], help="이름 (기본: 전부)")
    parser.add_argument("--seed", type=int, default=JUDGED_SEED)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument(
        "--out", type=Path, default=None, help="기본: bench/runs/paired/spg-seed<seed>.json"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = args.datasets or [d.name for d in ALL]
    unknown = [n for n in names if n not in BY_NAME]
    if unknown:
        print(f"모르는 데이터셋: {', '.join(unknown)}", file=sys.stderr)
        return 2

    print(
        f"--search-past-goal 판정 · 시드 {args.seed} · 재추출 {args.resamples}회 · "
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

    out_path = args.out or OUT_DIR / f"spg-seed{args.seed}.json"
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
    print("학습 횟수 (예산은 맞추지 않습니다 — 그 차이가 처치입니다):")
    for row in budget_used(results):
        spent = " · ".join(f"{arm}={row.get(arm)}" for arm, _ in ARMS)
        print(f"   {row['dataset']:<16} {spent}")
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
