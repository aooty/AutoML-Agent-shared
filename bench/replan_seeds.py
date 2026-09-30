"""Check if the speeddating replan gain survives new seeds and prompt.

Roles:

* Constants — dataset, seeds, arms, and the noise bars.
* Adjudication — score every leg and pair them per seed.
* Criterion — judge the rule fixed in docs/REPLAN-SEEDS.md.
* Output — budget table, JSON payload, and the CLI.

Usage:
    python -m bench.replan_seeds
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

# --- Role: constants ------------------------------------------------------------------

# One dataset on purpose: only speeddating showed a gain.
DATASET = "speeddating"

# Same seeds as hard_nollm; fixed here so scope cannot grow.
SEEDS: tuple[int, ...] = (42, 43, 44)

# New prefixes, so the recorded archive is never written.
ARMS: tuple[tuple[str, str], ...] = (("hard_llm2", "hardllm2"), ("hard_nollm2", "hardnollm2"))

# Recorded REPLAN.md arms; loop arm ran on seed 42 only.
RECORDED_RULES: tuple[str, str] = ("hard_nollm", "hardnollm")
RECORDED_LOOP: tuple[str, str] = ("hard_llm", "hardllm")
RECORDED_LOOP_SEEDS: tuple[int, ...] = (42,)

# Smallest win must beat this: median repeat half-width, REPEATS.md.
NOISE_FLOOR = 0.0122

# Rules arm test spread over the seeds, from REPLAN.md.
RULES_SEED_SPREAD = 0.0017

# The recorded gain being reproduced, from REPLAN.md.
RECORDED_PRIMARY = 0.0935

# One bootstrap seed for all runs, so recheck matches.
RESAMPLE_SEED = JUDGED_SEED

PRIMARY = ("hard_llm2", "hard_llm2@it1")


# --- Role: adjudication ---------------------------------------------------------------


def _arm_legs(seed: int, metric: str, out: DatasetVerdict) -> tuple[dict[str, Winner], dict[str, Any]]:
    """_arm_legs | Adjudication: new arms' winners, iteration 1 legs, and run shapes."""
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
    """_recorded_legs | Adjudication: the recorded REPLAN.md arms for this seed."""
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
    """_pairs_to_take | Adjudication: pairs this seed supports, in the registered order."""
    pairs: list[tuple[str, str, str]] = [(*PRIMARY, "primary")]
    matched = f"random@k={fits}"
    if matched in scored and "random@k=1" in scored:
        # Always computed: must sit beside any positive primary.
        pairs.append((matched, "random@k=1", "blind_gain"))
    if matched in scored:
        pairs.append(("hard_llm2", matched, "matched_level"))
    if "hard_nollm2" in scored:
        pairs.append(("hard_llm2", "hard_nollm2", "vs_rules"))
        pairs.append(("hard_nollm2", "hard_nollm2@it1", "rules_gain"))
        if "hard_nollm" in scored:
            # Rules arm calls no model, so only code moves this.
            pairs.append(("hard_nollm2", "hard_nollm", "code_shift"))
    if "hard_llm" in scored:
        # Prompt and code both changed, so only a note.
        pairs.append(("hard_llm2", "hard_llm", "prompt_shift"))
    return pairs


def _adjudicate(
    seed: int, resamples: int, out: DatasetVerdict, legs: dict[str, Scored] | None = None
) -> None:
    """_adjudicate | Adjudication: score legs, check same rows, and fill ``out``."""
    dataset = BY_NAME[DATASET]
    metric = dataset.metric
    winners, shapes = _arm_legs(seed, metric, out)
    if "hard_llm2" not in winners:
        # Not a data problem: the run has not happened yet.
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
        # Iteration 1 legs never had a test score (nan).
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

    # Last, so a refusal leaves no legs in the bundle.
    if legs is not None:
        legs.update(scored)


def adjudicate(seed: int, resamples: int, legs: dict[str, Scored] | None = None) -> DatasetVerdict:
    """Judge one seed; a refusal empties the pairs."""
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


# --- Role: criterion ------------------------------------------------------------------


def _primary(result: DatasetVerdict) -> Delta | None:
    """_primary | Criterion: the primary pair of one seed, or None."""
    a, b = PRIMARY
    return next(
        (p for p in result.pairs if p.a == a and p.b == b and p.kind == "primary"), None
    )


def criterion(results: dict[int, DatasetVerdict]) -> dict[str, Any]:
    """Judge the rule committed in docs/REPLAN-SEEDS.md before these runs.

    Every looped seed must win, and the smallest win must beat NOISE_FLOOR."""
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
            # Winner is iteration 1: budget spent, nothing won.
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


# --- Role: output ---------------------------------------------------------------------


def budget_used(results: dict[int, DatasetVerdict]) -> list[dict[str, Any]]:
    """Per seed, what each arm spent: fits, stop reason, critic runs, bar."""
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
        # One bootstrap seed per file; recheck reads only this.
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
        # Nothing scored; an empty verdict file would fail recheck.
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
