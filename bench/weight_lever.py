"""Was the speeddating win the replan, or one weight read from the card?

Roles:

* Settings — datasets, seeds, arms, and pinned numbers.
* Leg configs — paths and the config each leg trains on.
* Fitting — run ``train.py`` once per leg.
* Judging — score the legs and pair them per cell.
* Controls — refit check and published delta check.
* Criterion — the three pre-registered conditions.
* Output — JSON payload and command line.

Usage:

    python -m bench.weight_lever --fit
    python -m bench.weight_lever
    python -m bench.weight_lever speeddating --seeds 42
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from automl_agent.config import (
    MODEL_FILENAME,
    TRAIN_SCRIPT,
    read_json_object,
    utf8_env,
)
from automl_agent.nodes.model_selection import registry
from automl_agent.threads import describe_thread_state, thread_state
from bench.datasets import BY_NAME, JUDGED, JUDGED_SEED
from bench.paired import (
    ARTIFACTS_DIR,
    OUT_DIR,
    RESAMPLES,
    DatasetVerdict,
    Delta,
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
from bench.random_search import RUNS_DIR

# --- Role: settings ------------------------------------------------------------------

# Same order and set as JUDGED in bench/datasets.py.
DATASETS: tuple[str, ...] = tuple(item.name for item in JUDGED)

SEEDS: tuple[int, ...] = (42, 43, 44)

# Recorded arms; this file only reads them.
RULES = "bar65nollm"
LOOP = "bar65llm"

# The three legs; ratio is the treatment, others are controls.
RATIO = "ratio"
BALANCED = "balanced"
REFIT = "refit"
VARIANTS: tuple[str, ...] = (RATIO, BALANCED, REFIT)

# Pinned from the committed cards; a test checks them.
CARD_RATIOS: dict[str, float] = {
    "adult": 3.18,
    "bank-marketing": 7.55,
    "speeddating": 5.07,
    "spambase": 1.54,
}

# Positive weight the recorded rules winners held; pinned on purpose.
RECORDED_WEIGHTS: dict[str, float | None] = {
    "adult": 2.25,
    "bank-marketing": 5.062,
    "speeddating": 1.5,
    "spambase": None,
}

# Families whose menu offers class_weight; others are refused.
WEIGHT_FAMILIES: frozenset[str] = frozenset(
    entry["id"] for entry in registry() if "class_weight" in entry["params"]
)

# Published main deltas, only compared against the recomputed ones.
PRIMARY_DELTAS: dict[tuple[str, int], float] = {
    ("speeddating", 42): 0.0819,
    ("speeddating", 43): 0.0777,
    ("speeddating", 44): 0.0761,
}

# Allowed gap from the four-decimal published deltas.
PUBLISHED_TOLERANCE = 5e-5

# Size bar imported from REPEATS.md, not measured here.
NOISE_FLOOR = 0.0122
NOISE_FLOOR_SOURCE = "docs/REPEATS.md (낡은 프롬프트·낡은 바에서 잰 수입니다)"

# Refit floor from FINDINGS-mimic.md A2; bigger moves leave scope.
REFIT_TOLERANCE = 0.0026
REFIT_TOLERANCE_SOURCE = "docs/FINDINGS-mimic.md A2 (xgboost 재적합, 이 경로에서는 처음 잽니다)"

# Resample seed, not run seed; one seed per file.
RESAMPLE_SEED = JUDGED_SEED

WEIGHT_RUNS = RUNS_DIR / "weightlever"

# Per-fit timeout, same as bench/random_search.py.
TIMEOUT_SEC = 3600
OMP_THREADS = "1"


# --- Role: leg configs --------------------------------------------------------------


def _key(dataset: str, seed: int) -> str:
    """_key | Leg configs: verdict entry name for one dataset and seed."""
    return f"{dataset}-seed{seed}"


def run_dir(dataset: str, seed: int, variant: str) -> Path:
    """Folder of one fitted leg."""
    return WEIGHT_RUNS / f"{dataset}-seed{seed}" / variant


def recorded_winner_dir(dataset: str, seed: int) -> Path:
    """Iteration folder of the recorded rules winner, from ``history.json``."""
    directory = ARTIFACTS_DIR / f"{RULES}-{dataset}-seed{seed}"
    history = read_json_object(directory / "history.json")
    if history is None:
        raise Refusal(f"{dataset} 시드 {seed}: {RULES}의 history.json이 없습니다")
    iteration = dict(history.get("best") or {}).get("iteration")
    if not isinstance(iteration, int):
        raise Refusal(f"{dataset} 시드 {seed}: best.iteration이 정수가 아닙니다 ({iteration!r})")
    return directory / "train" / f"iter_{iteration:02d}"


def positive_weight(config: dict[str, Any]) -> float | None:
    """Positive class weight in a config, or ``None``.

    JSON map keys arrive as strings, so ``"1"`` is checked first."""
    weight = dict(config.get("hyperparams") or {}).get("class_weight")
    if isinstance(weight, dict):
        for key in ("1", 1):
            if key in weight:
                return float(weight[key])
        return None
    return None


def treated_config(dataset: str, seed: int, variant: str) -> dict[str, Any]:
    """The rules winner's config with only ``class_weight`` changed.

    Refuses families without ``class_weight`` and unexpected recorded weights."""
    source = read_json_object(recorded_winner_dir(dataset, seed) / "train_config.json")
    if source is None:
        raise Refusal(f"{dataset} 시드 {seed}: 규칙 팔 우승자의 train_config.json을 읽을 수 없습니다")
    family = str(source.get("model") or "")
    if family not in WEIGHT_FAMILIES:
        raise Refusal(
            f"{dataset} 시드 {seed}: 우승자 계열이 {family!r}인데 이 계열은 class_weight를 받지 "
            "않습니다 — 받지도 않는 키를 써 놓고 처치했다고 적지 않습니다"
        )
    recorded = positive_weight(source)
    expected = RECORDED_WEIGHTS[dataset]
    if recorded != expected:
        raise Refusal(
            f"{dataset} 시드 {seed}: 기록된 양성 가중치가 {recorded!r}인데 사전 등록은 "
            f"{expected!r}로 박혀 있습니다 — 이 실험이 재려던 것이 아닙니다"
        )

    config = json.loads(json.dumps(source))
    # One fit has no baseline; that file is gitignored.
    config.pop("paired_baseline", None)
    hyperparams = dict(config.get("hyperparams") or {})
    if variant == RATIO:
        hyperparams["class_weight"] = {"0": 1.0, "1": CARD_RATIOS[dataset]}
    elif variant == BALANCED:
        hyperparams["class_weight"] = "balanced"
    elif variant == REFIT:
        pass  # the control keeps the recorded weight
    else:
        raise Refusal(f"알 수 없는 처치: {variant!r}")
    config["hyperparams"] = hyperparams
    return config


# --- Role: fitting ------------------------------------------------------------------


def train_env() -> dict[str, str]:
    """Child env: the loop's own, with every thread pool pinned.

    Thread count moved balanced_accuracy by 0.0077 in A2."""
    env = dict(utf8_env())
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = OMP_THREADS
    return env


def fit_one(dataset: str, seed: int, variant: str) -> dict[str, Any]:
    """Write the config, run ``train.py`` once, return what it made."""
    directory = run_dir(dataset, seed, variant)
    directory.mkdir(parents=True, exist_ok=True)
    config = treated_config(dataset, seed, variant)
    config_path = directory / "train_config.json"
    result_path = directory / "result.json"
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    started = time.perf_counter()
    console = ""
    returncode = -1
    try:
        completed = subprocess.run(  # noqa: S603 - fixed script, no shell
            [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path), "--out", str(result_path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=train_env(),
            timeout=TIMEOUT_SEC,
            check=False,
        )
        console = (completed.stdout or "") + (completed.stderr or "")
        returncode = completed.returncode
    except subprocess.TimeoutExpired:
        returncode = -9
        console = f"timeout after {TIMEOUT_SEC}s"
    except OSError as exc:
        console = f"failed to spawn {TRAIN_SCRIPT}: {exc}"
    (directory / "train.log").write_text(console, encoding="utf-8")

    # train.py exits 0 on failure too; trust status.
    result = read_json_object(result_path) or {}
    metric = BY_NAME[dataset].metric
    return {
        "dataset": dataset,
        "seed": seed,
        "variant": variant,
        "class_weight": dict(config["hyperparams"]).get("class_weight"),
        "status": str(result.get("status") or "no_result"),
        "returncode": returncode,
        "val_score": dict(result.get("metrics") or {}).get(metric),
        "applied_class_weight": dict(result.get("applied_hyperparams") or {}).get("class_weight"),
        "dropped_hyperparams": result.get("dropped_hyperparams") or [],
        "elapsed_sec": round(time.perf_counter() - started, 2),
        "dir": directory.as_posix(),
    }


def fit_all(names: list[str], seeds: list[int]) -> list[dict[str, Any]]:
    """Fit every leg of the chosen datasets and seeds."""
    rows: list[dict[str, Any]] = []
    total = len(names) * len(seeds) * len(VARIANTS)
    index = 0
    for dataset in DATASETS:
        if dataset not in names:
            continue
        for seed in SEEDS:
            if seed not in seeds:
                continue
            for variant in VARIANTS:
                index += 1
                print(f"[{index}/{total}] {dataset}-seed{seed} {variant} ...", flush=True)
                try:
                    row = fit_one(dataset, seed, variant)
                except Refusal as exc:
                    row = {
                        "dataset": dataset, "seed": seed, "variant": variant,
                        "status": "refused", "refusal": str(exc),
                    }
                rows.append(row)
                weight = row.get("class_weight")
                print(
                    f"    {row['status']} · cw={weight!r} · val="
                    f"{row.get('val_score')} · {row.get('elapsed_sec')}s",
                    flush=True,
                )
    return rows


# --- Role: judging ------------------------------------------------------------------


def weight_leg(dataset: str, seed: int, variant: str) -> Winner:
    """One fitted leg as a :class:`Winner` with no recorded test score.

    ``recorded_test`` is ``nan``; the fingerprint check stands in."""
    directory = run_dir(dataset, seed, variant)
    result = read_json_object(directory / "result.json")
    if result is None:
        raise Refusal(
            f"{variant}: result.json이 없습니다 ({directory}) — "
            "python -m bench.weight_lever --fit을 먼저 돌려야 판정할 수 있습니다"
        )
    status = str(result.get("status") or "")
    if status != "ok":
        raise Refusal(f"{variant}: 학습이 {status!r}로 끝났습니다 ({directory})")
    config = read_json_object(directory / "train_config.json")
    if config is None:
        raise Refusal(f"{variant}: train_config.json을 읽을 수 없습니다 ({directory})")
    if not (directory / MODEL_FILENAME).exists():
        raise Refusal(f"{variant}: {MODEL_FILENAME}이 없습니다 ({directory})")
    metric = BY_NAME[dataset].metric
    val = dict(result.get("metrics") or {}).get(metric)
    if not isinstance(val, (int, float)):
        raise Refusal(f"{variant}: result.json에 {metric}이 없습니다 ({directory})")
    weight = dict(config.get("hyperparams") or {}).get("class_weight")
    return Winner(
        arm=variant,
        label=f"cw={weight}",
        directory=directory,
        config=config,
        val_score=float(val),
        recorded_test=float("nan"),
        fits=1,
    )


def _pairs_to_take(scored: dict[str, Scored]) -> list[tuple[str, str, str]]:
    """_pairs_to_take | Judging: pairs this cell supports, in doc order."""
    pairs: list[tuple[str, str, str]] = []
    if LOOP in scored and RULES in scored:
        # Recomputed here, then checked against the published value.
        pairs.append((LOOP, RULES, "main"))
    if RATIO in scored and RULES in scored:
        pairs.append((RATIO, RULES, "recovery"))
    if LOOP in scored and RATIO in scored:
        pairs.append((LOOP, RATIO, "remaining"))
    if BALANCED in scored and RATIO in scored:
        pairs.append((BALANCED, RATIO, "scale"))
    if BALANCED in scored and RULES in scored:
        pairs.append((BALANCED, RULES, "balanced_recovery"))
    if REFIT in scored and RULES in scored:
        pairs.append((REFIT, RULES, "refit_control"))
    return pairs


def _adjudicate(
    dataset: str, seed: int, resamples: int, out: DatasetVerdict, legs: dict[str, Scored] | None
) -> None:
    """_adjudicate | Judging: score all legs of one cell and pair them."""
    metric = BY_NAME[dataset].metric
    winners: dict[str, Winner] = {}
    for arm in (LOOP, RULES):
        directory = ARTIFACTS_DIR / f"{arm}-{dataset}-seed{seed}"
        winner = loop_winner(arm, directory, metric)
        if winner is None:
            raise Refusal(
                f"{dataset} 시드 {seed}: {arm} 실행이 없습니다 — "
                "bench/scripts/run_hard_bar.sh의 산출물이 있어야 판정할 수 있습니다"
            )
        winners[arm] = winner
    for variant in VARIANTS:
        winners[variant] = weight_leg(dataset, seed, variant)

    scored: dict[str, Scored] = {}
    for name, winner in winners.items():
        result = reproduce(winner, metric)
        if not math.isnan(winner.recorded_test):
            gap = abs(result.test_score - winner.recorded_test)
            if gap > 1e-6:
                raise Refusal(
                    f"{name}: 다시 유도한 test 점수 {result.test_score:.6f}가 기록된 "
                    f"{winner.recorded_test:.6f}와 {gap:.2e} 다릅니다 — 같은 행이 아닙니다"
                )
        scored[name] = result

    fingerprints = {name: s.fingerprint for name, s in scored.items()}
    if len(set(fingerprints.values())) > 1:
        listing = ", ".join(f"{name}={fp[:12]}" for name, fp in sorted(fingerprints.items()))
        raise Refusal(
            f"{dataset} 시드 {seed}: 다리마다 test 지문이 다릅니다 ({listing}) — 판정하지 않습니다"
        )
    identities = {
        name: json.dumps(split_identity(w.config), sort_keys=True) for name, w in winners.items()
    }
    if len(set(identities.values())) > 1:
        raise Refusal(
            f"{dataset} 시드 {seed}: 분할을 정하는 config 필드가 다리마다 다릅니다 "
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
            "class_weight": dict(s.winner.config.get("hyperparams") or {}).get("class_weight"),
            "model": s.winner.config.get("model"),
            "dir": s.winner.directory.as_posix(),
        }

    for a_name, b_name, kind in _pairs_to_take(scored):
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


def _pair(result: DatasetVerdict, kind: str) -> Delta | None:
    """_pair | Judging: the pair of one kind, or ``None``."""
    return next((p for p in result.pairs if p.kind == kind), None)


# --- Role: controls -----------------------------------------------------------------


def refit_check(results: dict[tuple[str, int], DatasetVerdict]) -> list[dict[str, Any]]:
    """Does refitting the recorded config reproduce it? Failing cells leave scope."""
    rows: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for seed in SEEDS:
            result = results.get((dataset, seed))
            if result is None:
                continue
            pair = _pair(result, "refit_control")
            if pair is None:
                rows.append(
                    {"dataset": dataset, "seed": seed, "identical": None, "delta": None, "ok": False}
                )
                continue
            rows.append(
                {
                    "dataset": dataset,
                    "seed": seed,
                    "identical": pair.identical_predictions,
                    "delta": pair.delta,
                    "tolerance": REFIT_TOLERANCE,
                    "ok": pair.identical_predictions or abs(pair.delta) <= REFIT_TOLERANCE,
                }
            )
    return rows


def published_check(results: dict[tuple[str, int], DatasetVerdict]) -> list[dict[str, Any]]:
    """Compare recomputed main deltas with the ``HARD-BAR.md`` values."""
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


# --- Role: criterion ----------------------------------------------------------------


def _cell(result: DatasetVerdict | None, refit_ok: bool) -> dict[str, Any]:
    """_cell | Criterion: one cell cut down to what the conditions read."""
    if result is None or result.verdict != "ok":
        return {"evaluated": False, "reason": "없음 또는 거부"}
    if not refit_ok:
        return {"evaluated": False, "reason": "재적합 대조 실패 — 판정 밖"}
    recovery = _pair(result, "recovery")
    remaining = _pair(result, "remaining")
    main = _pair(result, "main")
    if recovery is None or remaining is None or main is None:
        return {"evaluated": False, "reason": "쌍이 모자랍니다"}
    half = main.delta / 2.0
    return {
        "evaluated": True,
        "main": main.delta,
        "half_of_main": half,
        "recovery": recovery.delta,
        "recovery_ci": [recovery.ci_low, recovery.ci_high],
        "recovery_identical": recovery.identical_predictions,
        # Condition 1: above zero and at least half the gap.
        "recovery_clears_zero": (not recovery.identical_predictions) and recovery.ci_low > 0,
        "recovery_reaches_half": recovery.delta >= half,
        "remaining": remaining.delta,
        "remaining_ci": [remaining.ci_low, remaining.ci_high],
        # Condition 2: remaining gap is zero or under floor.
        "remaining_closed": (
            remaining.identical_predictions
            or (remaining.ci_low <= 0 <= remaining.ci_high)
            or abs(remaining.delta) < NOISE_FLOOR
        ),
        # Condition 3, on the differential dataset.
        "recovery_under_floor": recovery.identical_predictions or abs(recovery.delta) < NOISE_FLOOR,
    }


def criterion(results: dict[tuple[str, int], DatasetVerdict]) -> dict[str, Any]:
    """Judge the criterion committed in ``docs/WEIGHT-LEVER.md`` before these fits.

    All seeds must agree; missing cells make it partial."""
    refit_ok = {
        (row["dataset"], row["seed"]): bool(row["ok"]) for row in refit_check(results)
    }
    cells = {
        (dataset, seed): _cell(results.get((dataset, seed)), refit_ok.get((dataset, seed), False))
        for dataset in DATASETS
        for seed in SEEDS
    }

    absent = [
        f"{dataset}-seed{seed}"
        for (dataset, seed), cell in cells.items()
        if not cell["evaluated"] and cell.get("reason") != "재적합 대조 실패 — 판정 밖"
    ]
    out_of_scope = [
        f"{dataset}-seed{seed}"
        for (dataset, seed), cell in cells.items()
        if cell.get("reason") == "재적합 대조 실패 — 판정 밖"
    ]

    def _all(dataset: str, field: str) -> bool:
        return all(
            cells[(dataset, seed)]["evaluated"] and cells[(dataset, seed)][field]
            for seed in SEEDS
        )

    recovered = _all("speeddating", "recovery_clears_zero") and _all("speeddating", "recovery_reaches_half")
    closed = _all("speeddating", "remaining_closed")
    differential = _all("spambase", "recovery_under_floor")
    speeddating_scoped = all(cells[("speeddating", seed)]["evaluated"] for seed in SEEDS)

    if absent:
        verdict = "부분 판정"
        status = (
            f"부분 판정 — (데이터셋, 시드) {len(DATASETS) * len(SEEDS)}칸 중 판정하지 못한 칸이 "
            f"있습니다 (없음/거부: {', '.join(absent)})"
        )
    elif not speeddating_scoped:
        verdict = "판정 불가"
        status = (
            f"판정 불가 — speeddating 세 칸 중 판정 밖이 있습니다 ({', '.join(out_of_scope)}). "
            "주 예측이 그 세 칸에 걸려 있고, 기준을 약하게 고치지 않습니다"
        )
    elif recovered and closed and differential:
        verdict = "설명이 산다"
        status = (
            "설명이 산다 — speeddating 세 시드 전부에서 가중치 하나가 주 Δ의 절반 이상을 회수했고, "
            "남은 격차가 닫혔고, spambase에서는 예상대로 아무 일도 없었습니다"
        )
    elif not recovered:
        verdict = "설명이 죽는다"
        status = (
            "설명이 죽는다 — speeddating에서 카드 비율로 가중치를 옮긴 것이 주 Δ의 절반을 회수하지 "
            "못했습니다. 그 격차는 가중치의 출발점이 아닙니다"
        )
    else:
        missing = [
            name
            for name, ok in (("남은 격차", closed), ("차등", differential))
            if not ok
        ]
        verdict = "부분 지지"
        status = (
            f"부분 지지 — 회수는 세 시드 전부에서 됐지만 {', '.join(missing)} 조건이 깨졌습니다"
        )
    if out_of_scope and verdict not in ("판정 불가", "부분 판정"):
        status += f" / 재적합 대조가 어긋난 칸 {len(out_of_scope)}개는 판정 밖입니다"

    return {
        "explains": "docs/HARD-BAR.md의 speeddating 이김에 대한 사후 설명",
        "treatment": RATIO,
        "baseline": RULES,
        "loop_arm": LOOP,
        "scope_datasets": list(DATASETS),
        "scope_seeds": list(SEEDS),
        "positive_prediction_on": "speeddating",
        "differential_prediction_on": "spambase",
        "resample_seed": RESAMPLE_SEED,
        "card_ratios": CARD_RATIOS,
        "recorded_weights": RECORDED_WEIGHTS,
        "noise_floor": NOISE_FLOOR,
        "noise_floor_source": NOISE_FLOOR_SOURCE,
        "refit_tolerance": REFIT_TOLERANCE,
        "refit_tolerance_source": REFIT_TOLERANCE_SOURCE,
        "cells": {f"{d}-seed{s}": cell for (d, s), cell in cells.items()},
        "condition_1_recovery": recovered,
        "condition_2_remaining_closed": closed,
        "condition_3_differential": differential,
        "not_evaluated": absent,
        "out_of_scope": out_of_scope,
        "verdict": verdict,
        "status": status,
    }


# --- Role: output -------------------------------------------------------------------


def payload(results: dict[tuple[str, int], DatasetVerdict], resamples: int) -> dict[str, Any]:
    """The JSON written for this verdict."""
    ordered = [
        (dataset, seed) for dataset in DATASETS for seed in SEEDS if (dataset, seed) in results
    ]
    return {
        "generated_by": "bench/weight_lever.py",
        "preregistration": "docs/WEIGHT-LEVER.md",
        "checks": "docs/HARD-BAR.md",
        # Main delta and test rows are reused, not new.
        "reuses_recorded_arms": [LOOP, RULES],
        "test_rows_shared_with": "bench/runs/paired/hard-bar.json",
        "variants": list(VARIANTS),
        "run_seeds": sorted({seed for _, seed in ordered}),
        "seed": RESAMPLE_SEED,
        "resamples": resamples,
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
        "refit_check": refit_check(results),
        "published_check": published_check(results),
        "criterion": criterion(results),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(
        description=(
            "speeddating의 이김이 카드에서 읽은 가중치 하나였는가 "
            "(docs/WEIGHT-LEVER.md의 사전 등록 기준)"
        )
    )
    parser.add_argument("names", nargs="*", help=f"데이터셋 이름 (생략하면 {' '.join(DATASETS)})")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(SEEDS),
        help=f"기본: {' '.join(str(s) for s in SEEDS)} (그보다 적으면 부분 판정입니다)",
    )
    parser.add_argument(
        "--fit", action="store_true",
        help="판정하지 않고 세 처치를 학습합니다 (API 호출 없음, 36회)",
    )
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument(
        "--out", type=Path, default=None, help="기본: bench/runs/paired/weight-lever.json"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Fit the legs with ``--fit``, else judge and write the verdict."""
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

    if args.fit:
        print(
            f"가중치 레버 학습 · 처치 {', '.join(VARIANTS)} · "
            f"{describe_thread_state(thread_state())}"
        )
        rows = fit_all(names, list(args.seeds))
        failed = [r for r in rows if r.get("status") != "ok"]
        WEIGHT_RUNS.mkdir(parents=True, exist_ok=True)
        (WEIGHT_RUNS / "fits.json").write_text(
            json.dumps({"threads": thread_state(), "fits": rows}, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(f"\n{len(rows)}회 중 {len(rows) - len(failed)}회 ok")
        for row in failed:
            print(
                f"  실패: {row['dataset']}-seed{row['seed']} {row['variant']} — "
                f"{row.get('refusal') or row['status']}",
                file=sys.stderr,
            )
        return 1 if failed else 0

    print(
        f"가중치 레버의 판정 · 재추출 {args.resamples}회 (재추출 시드 {RESAMPLE_SEED}) · "
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

    out_path = args.out or OUT_DIR / "weight-lever.json"
    if not legs:
        print(
            f"판정할 다리가 없습니다 — {out_path.as_posix()}를 쓰지 않았습니다. "
            "python -m bench.weight_lever --fit을 먼저 돌리세요",
            file=sys.stderr,
        )
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    body = payload(results, args.resamples)
    out_path.write_text(
        json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    target = bundle_path(out_path)
    write_bundle(target, legs, {r.dataset: r.metric for r in results.values()})
    print()
    print(body["criterion"]["status"])
    print(f"{out_path.as_posix()} · {target.as_posix()}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
