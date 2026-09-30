"""Check whether a higher bar lets the LLM loop beat the rule fallback.

Roles:

* Constants — datasets, seeds, margin, arms, predicted bars.
* Adjudication — score each arm per dataset and seed, pair them.
* Bar check — compare each run's real bar to the prediction.
* Criterion — apply the pre-registered win rule across datasets.
* Output — budget table and the verdict file payload.
* CLI — parse flags, run, print, write files.

Usage:
    python -m bench.hard_bar
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

# --- Role: constants ----------------------------------------------------------------------

# Taken from JUDGED so the denominator never drifts.
DATASETS: tuple[str, ...] = tuple(item.name for item in JUDGED)

# Fixed here so the scope cannot grow after the fact.
SEEDS: tuple[int, ...] = (42, 43, 44)

# The only flag that differs from run_prompt_redo.sh.
MARGIN = 0.65

# New names so the recorded llm2 runs stay untouched.
ARMS: tuple[tuple[str, str], ...] = (("bar65llm", "bar65llm"), ("bar65nollm", "bar65nollm"))

# PROMPT-REDO arms; they exist only at the judged seed.
RECORDED: tuple[tuple[str, str], ...] = (("llm2", "llm2"), ("no_llm2", "nollm2"))
RECORDED_SEEDS: tuple[int, ...] = (JUDGED_SEED,)

# Imported from REPEATS.md, measured under older prompts and bars.
NOISE_FLOOR = 0.0122
NOISE_FLOOR_SOURCE = "docs/REPEATS.md (낡은 프롬프트·낡은 바에서 잰 수입니다)"

# Hardcoded on purpose; a test checks they still match goal.py.
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

# First-try val scores of llm2, seed 42 only.
RECORDED_FIRST_VAL: dict[str, float] = {
    "adult": 0.8460,
    "bank-marketing": 0.8705,
    "speeddating": 0.7878,
    "spambase": 0.9534,
}

# Beyond rounding error means a different baseline.
BAR_TOLERANCE = 5e-4

# One bootstrap seed for the whole file, so recheck works.
RESAMPLE_SEED = JUDGED_SEED

PRIMARY: tuple[str, str] = ("bar65llm", "bar65nollm")

# --- Role: adjudication -------------------------------------------------------------------


def _key(dataset: str, seed: int) -> str:
    """_key | Role: verdict entry name from dataset and seed."""
    return f"{dataset}-seed{seed}"


def _arm_legs(
    dataset: str, seed: int, metric: str, out: DatasetVerdict
) -> tuple[dict[str, Winner], dict[str, Any]]:
    """_arm_legs | Role: winners, iteration-1 legs and shapes of new arms."""
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
    """_recorded_legs | Role: recorded PROMPT-REDO arms at the judged seed."""
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
    """_pairs_to_take | Role: comparisons this cell supports, in HARD-BAR.md order."""
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
            # Did the extra iterations beat the first try?
            pairs.append((arm, f"{arm}@it1", f"{arm}_replan_gain"))
    # Same everything except --margin.
    if "llm2" in scored:
        pairs.append(("bar65llm", "llm2", "bar_shift"))
    if "no_llm2" in scored:
        pairs.append(("bar65nollm", "no_llm2", "rules_bar_shift"))
    return pairs


def _adjudicate(
    dataset: str, seed: int, resamples: int, out: DatasetVerdict, legs: dict[str, Scored] | None
) -> None:
    """_adjudicate | Role: score all arms for one cell; raises Refusal."""
    entry = BY_NAME[dataset]
    metric = entry.metric
    winners, shapes = _arm_legs(dataset, seed, metric, out)
    if "bar65llm" not in winners:
        # The run has not happened yet; not a data problem.
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
        # Iteration-1 legs have no recorded test score (nan).
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

    # Last, so a refusal leaves no legs behind.
    if legs is not None:
        legs.update(scored)


def adjudicate(
    dataset: str, seed: int, resamples: int, legs: dict[str, Scored] | None = None
) -> DatasetVerdict:
    """Judge one (dataset, seed) cell; a refusal clears the pairs."""
    entry = BY_NAME[dataset]
    out = DatasetVerdict(dataset=_key(dataset, seed), metric=entry.metric, task=entry.task)
    try:
        _adjudicate(dataset, seed, resamples, out, legs)
    except Refusal as exc:
        out.verdict = "refused"
        out.refusal = str(exc)
        out.pairs = []
    return out


# --- Role: bar check ----------------------------------------------------------------------


def bar_check(results: dict[tuple[str, int], DatasetVerdict]) -> list[dict[str, Any]]:
    """Compare each run's real ``goal.threshold`` to :data:`PREDICTED_BARS`."""
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            result = results.get((dataset, seed))
            if result is None:
                continue
            predicted = PREDICTED_BARS[(dataset, seed)]
            first_val = RECORDED_FIRST_VAL[dataset] if seed == JUDGED_SEED else None
            for arm, _ in ARMS:
                actual = result.arms.get(arm, {}).get("goal_threshold")
                # A missing bar must not count as a match.
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
                        # Below this, the run had no reason to loop.
                        "above_recorded_first_val": (
                            None
                            if first_val is None or known is None
                            else known > first_val
                        ),
                    }
                )
    return rows


# --- Role: criterion ----------------------------------------------------------------------


def _primary(result: DatasetVerdict) -> Delta | None:
    """_primary | Role: the primary pair of a cell, if any."""
    a, b = PRIMARY
    return next((p for p in result.pairs if p.a == a and p.b == b and p.kind == "primary"), None)


def _per_dataset(
    dataset: str, results: dict[tuple[str, int], DatasetVerdict]
) -> dict[str, Any]:
    """_per_dataset | Role: reduce one dataset's seeds to criterion counts."""
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
            # No replan means a broken premise, not a loss.
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
    """Per dataset, the rule arm's test spread across seeds, measured here."""
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
    """Judge the rule in ``docs/HARD-BAR.md``: all three seeds win, above the floor.

    Met when at least 3 of the 4 binary datasets win."""
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


# --- Role: output -------------------------------------------------------------------------


def budget_used(results: dict[tuple[str, int], DatasetVerdict]) -> list[dict[str, Any]]:
    """Per cell and arm: fits, stop reason, critic runs, and bar faced."""
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
    """Build the verdict file contents."""
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
        # recheck.py reads this one seed for every pair.
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


# --- Role: CLI ----------------------------------------------------------------------------


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
        # An empty verdict file would break a bare recheck.
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
