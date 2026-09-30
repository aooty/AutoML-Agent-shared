"""Third arm: random search over the same menu the LLM sees.

Roles:

* Constants — budget, timeout, threads, choices, row limits.
* Drawing — pick a model, hyperparams and preprocessing at random.
* Training — write the config, run train.py, collect the result.
* Test scoring — score only the prefix winners on test rows.
* Search — run all draws for one dataset and seed.
* CLI — parse args and loop over datasets.

Usage:
    python -m bench.random_search adult
    python -m bench.random_search adult --seed 42
    python -m bench.random_search
"""

from __future__ import annotations

import argparse
import json
import math
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from automl_agent.config import (
    DEFAULT_TIME_BUDGET_SEC,
    MODEL_FILENAME,
    TRAIN_SCRIPT,
    read_json_object,
    utf8_env,
)
from automl_agent.nodes.model_selection import (
    LIMITS,
    available_models,
    sanitise_hyperparams,
)
from automl_agent.nodes.training import (
    DEFAULT_TARGET_MISSING_POLICY,
    TARGET_MISSING_POLICIES,
    preprocessing_block,
)
from automl_agent.scoring.metrics import card_task

# Alias table folds duplicate knobs, like max_iter and n_estimators.
from automl_agent.scripts.train import ALIASES as PARAM_ALIASES
from bench.datasets import ALL, BENCH_DIR, BY_NAME, CHEAP_SEEDS, Dataset

# --- Role: constants --------------------------------------------------------------------

RUNS_DIR = BENCH_DIR / "runs"

# Same as the other arms' --max-iterations.
DEFAULT_DRAWS = 5

# The repo's train timeout, so draws time out like LLM tries.
TIMEOUT_SEC = DEFAULT_TIME_BUDGET_SEC

# xgboost hist builds different models at different thread counts.
OMP_THREADS = "1"

# Registry note: svc and svr are too slow above ~20k rows.
UNUSABLE_ABOVE_ROWS: dict[str, int] = {"svc": 20_000, "svr": 20_000}

# sklearn's full documented choices; None is a real class_weight.
CHOICES: dict[str, tuple[Any, ...]] = {
    "class_weight": (None, "balanced"),
    "weights": ("uniform", "distance"),
    "kernel": ("rbf", "linear", "poly", "sigmoid"),
}

IMPUTE_STRATEGIES = ("median", "mean", "most_frequent")
# Only these keep NaN; others turn "none" into median.
NAN_NATIVE = ("hist_gbdt", "xgboost")

# mlp layer sizes are a list, so not in LIMITS.
HIDDEN_WIDTH = (8, 512)
MAX_HIDDEN_LAYERS = 2

# Ranges spanning two decades or more are drawn log-uniform.
LOG_UNIFORM_DECADES = 100.0


# --- Role: drawing ----------------------------------------------------------------------


def is_log_scale(low: float, high: float) -> bool:
    return low > 0 and high / low >= LOG_UNIFORM_DECADES


def is_integer_param(low: float, high: float) -> bool:
    """Whether ``LIMITS`` wrote both bounds as ints, meaning a count.

    Checks the type, not the value, so ``(0.0, 100.0)`` stays float."""
    return isinstance(low, int) and isinstance(high, int)


def draw_number(rng: random.Random, low: float, high: float) -> float | int:
    """One value in ``[low, high]`` by the log, uniform and integer rules."""
    if is_log_scale(low, high):
        value = math.exp(rng.uniform(math.log(low), math.log(high)))
    else:
        value = rng.uniform(low, high)
    if is_integer_param(low, high):
        return max(int(low), min(int(high), int(round(value))))
    return round(value, 6)


def canonical_params(model: str, params: list[Any]) -> list[str]:
    """The model's knobs with aliases folded, first spelling and order kept."""
    aliases = PARAM_ALIASES.get(model, {})
    seen: set[str] = set()
    kept: list[str] = []
    for raw in params:
        name = str(raw)
        canonical = aliases.get(name, name)
        if canonical in seen:
            continue
        seen.add(canonical)
        kept.append(canonical)
    return kept


def draw_plan(rng: random.Random, entry: dict[str, Any]) -> dict[str, Any]:
    """One random (model, hyperparams, preprocessing) draw; reads no scores or card."""
    model = str(entry["id"])
    raw: dict[str, Any] = {}
    untouched: list[str] = []
    for name in canonical_params(model, list(entry.get("params") or [])):
        if name in CHOICES:
            raw[name] = rng.choice(CHOICES[name])
        elif name == "hidden_layer_sizes":
            layers = rng.randint(1, MAX_HIDDEN_LAYERS)
            raw[name] = [int(draw_number(rng, *HIDDEN_WIDTH)) for _ in range(layers)]
        elif name in LIMITS:
            raw[name] = draw_number(rng, *LIMITS[name])
        else:
            # No known bounds; keep the estimator default and record it.
            untouched.append(name)

    impute = rng.choice(IMPUTE_STRATEGIES)
    if model in NAN_NATIVE and rng.random() < 0.5:
        impute = "none"
    drawn_preprocessing = {
        "impute": impute,
        "scale": rng.random() < 0.5,
        "missing_indicator": rng.random() < 0.5,
        "missing_count": rng.random() < 0.5,
    }
    return {
        "model": model,
        "hyperparams": sanitise_hyperparams(raw),
        # Same allowlist that checks the LLM's block.
        "preprocessing": preprocessing_block({"preprocessing": drawn_preprocessing}, {}),
        "untouched_params": untouched,
    }


def menu(card: dict[str, Any], n_rows: int) -> tuple[dict[str, Any], ...]:
    """Models this arm may draw, after the row-limit note."""
    offered = available_models(card_task(card))
    return tuple(
        entry
        for entry in offered
        if n_rows <= UNUSABLE_ABOVE_ROWS.get(str(entry["id"]), sys.maxsize)
    )


# --- Role: training ---------------------------------------------------------------------


def run_dir(dataset: Dataset, seed: int) -> Path:
    return RUNS_DIR / "random" / f"{dataset.name}-seed{seed}"


def repo_relative(path: Path) -> str:
    """``path`` relative to the repo root, or absolute if outside."""
    root = Path(__file__).resolve().parents[1]
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def train_env() -> dict[str, str]:
    """Child env: the loop's own env with all thread pools pinned."""
    env = dict(utf8_env())
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = OMP_THREADS
    return env


def train_config(dataset: Dataset, seed: int, card: dict[str, Any], plan: dict[str, Any]) -> dict:
    """Runner config for one draw, built like ``nodes/training.py`` does.

    Leaves out ``simulate`` and ``paired_baseline`` on purpose."""
    declared = (card.get("target_missing") or {}).get("policy")
    policy = declared if declared in TARGET_MISSING_POLICIES else DEFAULT_TARGET_MISSING_POLICY
    config: dict[str, Any] = {
        "model": plan["model"],
        "hyperparams": plan["hyperparams"],
        "preprocessing": plan["preprocessing"],
        "target_missing": {"policy": policy},
        "data": {"path": dataset.csv_arg, "target_column": dataset.target},
        "metric": dataset.metric,
        "task": card_task(card),
        "seed": seed,
    }
    # From the card, to match the LLM arm's memory limit.
    limit = (card.get("constraints") or {}).get("memory_limit_mb")
    if limit:
        config["memory_limit_mb"] = limit
    return config


def spawn(command: list[str], log_path: Path) -> tuple[int, bool]:
    """Run one train.py call; return ``(returncode, timed_out)`` and write the log."""
    console = ""
    returncode = -1
    timed_out = False
    try:
        completed = subprocess.run(  # noqa: S603 - fixed script, no shell
            command,
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
        timed_out = True
        returncode = -9
    except OSError as exc:
        console = f"failed to spawn {command[1]}: {exc}"
    log_path.write_text(console, encoding="utf-8")
    return returncode, timed_out


def one_draw(
    dataset: Dataset, seed: int, index: int, plan: dict[str, Any], card: dict[str, Any]
) -> dict[str, Any]:
    """Write the config, run ``train.py``, and return what the draw produced."""
    directory = run_dir(dataset, seed) / f"draw_{index:02d}"
    directory.mkdir(parents=True, exist_ok=True)
    config_path = directory / "train_config.json"
    result_path = directory / "result.json"
    config_path.write_text(
        json.dumps(train_config(dataset, seed, card, plan), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    started = time.perf_counter()
    returncode, timed_out = spawn(
        [sys.executable, str(TRAIN_SCRIPT), "--config", str(config_path), "--out", str(result_path)],
        directory / "train.log",
    )
    elapsed = time.perf_counter() - started

    # train.py exits 0 on failure too; trust ``status``.
    result = read_json_object(result_path) or {}
    status = "too_slow" if timed_out else str(result.get("status") or "no_result")
    metrics = dict(result.get("metrics") or {})
    return {
        "draw": index,
        "model": plan["model"],
        "hyperparams": plan["hyperparams"],
        "preprocessing": plan["preprocessing"],
        "untouched_params": plan["untouched_params"],
        "applied_hyperparams": result.get("applied_hyperparams"),
        "dropped_hyperparams": result.get("dropped_hyperparams") or [],
        "applied_preprocessing": result.get("applied_preprocessing"),
        "status": status,
        "returncode": returncode,
        "error_type": result.get("error_type"),
        # None, not 0.0: zero is a real score.
        "val_score": metrics.get(dataset.metric),
        "val_metrics": metrics,
        "val_split": result.get("split"),
        "model_file": (directory / MODEL_FILENAME).as_posix()
        if (directory / MODEL_FILENAME).exists()
        else None,
        "elapsed_sec": round(elapsed, 2),
        "dir": directory.as_posix(),
    }


# --- Role: test scoring -----------------------------------------------------------------


def best_by_val(records: list[dict[str, Any]], k: int) -> dict[str, Any] | None:
    """Winner of the first k draws, picked on validation only.

    Ties go to the earlier draw, like the loop's own rule."""
    scored = [item for item in records[:k] if item.get("val_score") is not None]
    if not scored:
        return None
    return max(scored, key=lambda item: (float(item["val_score"]), -int(item["draw"])))


def score_test(dataset: Dataset, record: dict[str, Any]) -> dict[str, Any] | None:
    """Score one draw's saved model on held-out test rows, like ``nodes/holdout.py``."""
    directory = Path(record["dir"])
    model_file = directory / MODEL_FILENAME
    if not model_file.exists():
        return None
    out_path = directory / "test.json"
    returncode, timed_out = spawn(
        [
            sys.executable,
            str(TRAIN_SCRIPT),
            "--config",
            str(directory / "train_config.json"),
            "--out",
            str(out_path),
            "--score-model",
            str(model_file),
        ],
        directory / "test.log",
    )
    result = read_json_object(out_path) or {}
    if timed_out or result.get("status") != "ok":
        return {"status": "too_slow" if timed_out else str(result.get("status") or "no_result"),
                "returncode": returncode}
    metrics = dict(result.get("metrics") or {})
    return {
        "status": "ok",
        "test_score": metrics.get(dataset.metric),
        "test_metrics": metrics,
        "split": result.get("split"),
    }


def test_candidates(records: list[dict[str, Any]], draws: int) -> list[int]:
    """Draw numbers the test split may see: prefix winners only."""
    chosen = {(best_by_val(records, k) or {}).get("draw") for k in range(1, draws + 1)}
    return sorted(draw for draw in chosen if isinstance(draw, int))


# --- Role: search -----------------------------------------------------------------------


def search(dataset: Dataset, seed: int, draws: int) -> int:
    card_path = dataset.card_path(seed)
    if not card_path.exists():
        print(f"{dataset.name}: 카드가 없습니다 ({card_path}) — 먼저 python -m bench.cards")
        return 1
    card = json.loads(card_path.read_text(encoding="utf-8"))
    n_rows = int(card.get("n_rows") or 0)

    offered = tuple(available_models(card_task(card)))
    candidates = menu(card, n_rows)
    excluded = sorted({str(e["id"]) for e in offered} - {str(e["id"]) for e in candidates})
    # Seed by dataset and seed, so datasets draw different sequences.
    rng = random.Random(f"{dataset.name}:{seed}")

    print(f"== {dataset.name} seed {seed} · {n_rows}행 · 뽑기 {draws}회 · 후보 {len(candidates)}개")
    if excluded:
        print(f"   행 수로 제외: {excluded} (레지스트리 notes)")

    records: list[dict[str, Any]] = []
    for index in range(1, draws + 1):
        plan = draw_plan(rng, rng.choice(candidates))
        record = one_draw(dataset, seed, index, plan, card)
        records.append(record)
        score = record["val_score"]
        shown = f"{float(score):.4f}" if score is not None else "—"
        print(
            f"   {index}/{draws} {record['model']:18s} {record['status']:9s} "
            f"val {dataset.metric}={shown} ({record['elapsed_sec']}s)"
        )

    for index in test_candidates(records, draws):
        record = records[index - 1]
        test = score_test(dataset, record)
        record["test"] = test
        if test and test.get("status") == "ok" and test.get("test_score") is not None:
            gap = float(record["val_score"]) - float(test["test_score"])
            record["selection_gap"] = round(gap, 6)
            print(
                f"   draw {index} test {dataset.metric}={float(test['test_score']):.4f} "
                f"(val−test {gap:+.4f})"
            )
        else:
            print(f"   draw {index} test 채점 실패: {test}")

    summary = {
        "arm": "random",
        "dataset": dataset.name,
        "data_id": dataset.data_id,
        "seed": seed,
        "metric": dataset.metric,
        "task": card_task(card),
        "draws": draws,
        "omp_num_threads": OMP_THREADS,
        "timeout_sec": TIMEOUT_SEC,
        "candidates": [str(entry["id"]) for entry in candidates],
        "excluded_by_rows": excluded,
        "card": card_path.as_posix(),
        # Repo-relative, so the committed file holds no user path.
        "train_script": repo_relative(TRAIN_SCRIPT),
        # Precomputed, so the judge just reads the LLM arm's k.
        "best_by_val_for_k": {
            str(k): (best_by_val(records, k) or {}).get("draw") for k in range(1, draws + 1)
        },
        "records": records,
    }
    out_path = run_dir(dataset, seed) / "summary.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    usable = [item for item in records if item.get("val_score") is not None]
    print(f"   요약 → {out_path.as_posix()}  (점수가 난 뽑기 {len(usable)}/{draws})")
    if not usable:
        print(f"   {dataset.name} seed {seed}: 점수가 난 뽑기가 없습니다 — 판정에 쓸 수 없습니다")
        return 1
    return 0


# --- Role: CLI --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.random_search", description=__doc__)
    parser.add_argument("names", nargs="*", help="데이터셋 이름 (생략하면 전부)")
    parser.add_argument(
        "--seed",
        type=int,
        action="append",
        dest="seeds",
        help=f"시드. 여러 번 쓸 수 있습니다 (생략하면 {list(CHEAP_SEEDS)})",
    )
    parser.add_argument(
        "--draws", type=int, default=DEFAULT_DRAWS, help=f"뽑기 횟수 (기본: {DEFAULT_DRAWS})"
    )
    args = parser.parse_args(argv)

    names = args.names or [item.name for item in ALL]
    unknown = [name for name in names if name not in BY_NAME]
    if unknown:
        print(f"모르는 데이터셋: {unknown} — 아는 것은 {sorted(BY_NAME)}")
        return 2
    if args.draws < 1:
        print(f"--draws는 1 이상이어야 합니다 (받은 값: {args.draws})")
        return 2

    failed = 0
    for name in names:
        for seed in args.seeds or list(CHEAP_SEEDS):
            failed += search(BY_NAME[name], seed, args.draws)
    if failed:
        print(f"\n{failed}개 (데이터셋, 시드)에서 쓸 만한 뽑기가 없었습니다")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
