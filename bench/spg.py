"""Judge the --search-past-goal arm: is spending past the bar worth it?

Roles:

* Constants — the two judged arms and the recorded arm.
* Adjudication — score every leg and pair them per dataset.
* Criterion — judge the rule fixed in docs/SPG.md.
* Output — fits table, JSON payload, and the CLI.

Usage:
    python -m bench.spg
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

# --- Role: constants ------------------------------------------------------------------

# Same code, card, and seed; only the flag differs.
ARMS: tuple[tuple[str, str], ...] = (("spg_off", "spgoff"), ("spg_on", "spgon"))
# RESULTS.md arm from another commit; a note, not a control.
RECORDED: tuple[str, str] = ("llm_recorded", "llm")


# --- Role: adjudication ---------------------------------------------------------------


def _adjudicate(
    dataset: Dataset,
    seed: int,
    resamples: int,
    out: DatasetVerdict,
    legs: dict[str, Scored] | None = None,
) -> None:
    """_adjudicate | Adjudication: score legs, check same rows, and fill ``out``."""
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
            # spg_on picks from more tries, so harm shows here.
            "selection_gap": s.winner.val_score - s.test_score,
            "dir": s.winner.directory.as_posix(),
        }

    # spg_on first, so a positive delta means the flag helped.
    out.pairs.append(paired_delta(scored["spg_on"], scored["spg_off"], metric, resamples, seed, "primary"))
    if RECORDED[0] in scored:
        # A note, not the rule: mostly run-to-run LLM noise.
        out.pairs.append(
            paired_delta(scored["spg_off"], scored[RECORDED[0]], metric, resamples, seed, "observation")
        )

    # Last, so a refusal leaves no legs in the bundle.
    if legs is not None:
        legs.update(scored)


def adjudicate(
    dataset: Dataset, seed: int, resamples: int, legs: dict[str, Scored] | None = None
) -> DatasetVerdict:
    """Judge one dataset; a refusal empties the pairs."""
    out = DatasetVerdict(dataset=dataset.name, metric=dataset.metric, task=dataset.task)
    try:
        _adjudicate(dataset, seed, resamples, out, legs)
    except Refusal as exc:
        out.verdict = "refused"
        out.refusal = str(exc)
        out.pairs = []
    return out


# --- Role: criterion ------------------------------------------------------------------


def criterion(results: list[DatasetVerdict], seed: int = JUDGED_SEED) -> dict[str, Any]:
    """Judge the rule committed in docs/SPG.md before these runs.

    spg_on must win on 3 of 4 binary sets; 3 losses count too."""
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


# --- Role: output ---------------------------------------------------------------------


def budget_used(results: list[DatasetVerdict]) -> list[dict[str, Any]]:
    """Fits each arm spent per dataset; the budget gap is the treatment."""
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
        # Same recorder as the runs, including cpu_count.
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
