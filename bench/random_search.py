"""세 번째 팔: LLM에게 보여 주는 것과 같은 메뉴에서 뽑는 random search.

    python -m bench.random_search adult              # 저렴한 시드 전부
    python -m bench.random_search adult --seed 42    # 시드 하나
    python -m bench.random_search                    # 데이터셋 다섯 개 전부

이 팔이 없으면 저장소의 핵심 주장에 바닥이 없다. 지금 비교할 수 있는 상대는 profiler의 기준선뿐이고
그건 *탐색*과의 비교가 아니다 — "루프가 진단하고 다시 계획한 것이 무작위 다섯 번보다 나았다"는 측정된
적이 없고, 그게 루프가 있는 이유다.

**모든 범위가 어디서 오는가, 그리고 왜 하나도 내가 정한 것이 아닌가.**

* 모델 메뉴는 import한 :func:`automl_agent.nodes.model_selection.available_models`다 — 선택 노드가
  부르는 것과 같은 호출이라, 이 팔은 LLM에게 제시되는 목록을 그대로 본다(``xgboost``가 설치됐는지
  보는 관문까지. LLM이 고를 수 없었던 모델을 뽑으면 다른 실험이 된다).
* 뽑기가 건드릴 수 있는 손잡이는 그 모델 자신의 registry ``params``다. 그래서 이 팔은 같은 모델에
  대해 LLM에게 제시되지 않은 손잡이를 조율할 수 없다.
* 수치 경계는 import한 ``model_selection.LIMITS``다. 그 표는 저장소가 이미 *LLM이 제안하는 무엇에든*
  적용하는 clamp라서, 두 팔이 사는 상자가 정확히 같다. 이 파일에는 손으로 쓴 범위가 없다 — 검사는
  import 한 줄이다.
* 범주형 값은 sklearn 자신의 문서화된 선택지 집합이고, 저장소 상수가 들고 있지 않아서
  :data:`CHOICES`에 적었다. 고른 부분집합이 아니라 전체 집합이다.
* 뽑은 dict는 전부 ``sanitise_hyperparams``를, 뽑은 preprocessing 블록은 전부
  ``nodes.training.preprocessing_block``을 지난다 — 계획이 지나는 것과 같은 두 관문이라, 어느 팔도
  표현할 수 없는 값은 양쪽에서 버려지고 어느 쪽이든 ``dropped_hyperparams``에 나타난다.
* 학습은 ``TRAIN_SCRIPT``의 ``scripts/train.py``, 학습 노드가 띄우는 것과 같은 파일이다. 같은 분할,
  같은 지표, 같은 결측 처리, 같은 결과 schema. 여기 따로 학습 경로를 두면 구현 둘의 차이를 재게 된다.

**값을 어떻게 뽑는가.** 기계적인 규칙 하나를 한 번 적고 매개변수별 판단 없이 적용한다. 매개변수별
분포야말로 벤치마크 저자가 이 팔을 조용히 돕거나 발목 잡는 자리이기 때문이다:

* 공표된 범위가 두 자릿수 이상 걸치면(``low > 0``이고 ``high / low >= 100``) **로그 균등** —
  ``n_estimators`` 1..2000, ``learning_rate`` 1e-4..1, ``C`` 1e-4..1e4, ``alpha`` 1e-8..1000,
  ``gamma``, ``max_iter``, ``max_leaf_nodes``, ``min_samples_leaf``, ``n_neighbors``, ``batch_size``;
* 아니면 **균등** — ``subsample`` 0.05..1, ``l1_ratio`` 0..1, ``l2_regularization``,
  ``reg_lambda``, ``max_depth`` 1..64;
* ``LIMITS``가 두 경계를 ``int``로 적었을 때만 **정수** — ``(1, 2000)``은 해당하고 ``(0.0, 100.0)``은
  아니다. 그래서 ``reg_lambda``와 ``l2_regularization``이 연속으로 남는다.

그 규칙의 결과 둘은 특수 처리로 없애기보다 적어 두는 쪽이 낫다. ``max_depth``가 균등으로 나오므로 트리
뽑기의 절반이 32보다 깊다 — 사실상 무제한이고, 그 estimator들에 대한 sklearn 자신의 기본값이지 망치려고
넣은 값이 아니다. ``epsilon``은 0..1e9로 공표되어 균등으로 뽑으면 말이 안 되는데, ``svr``에만 딸린
손잡이이고 여기 하나뿐인 회귀 데이터셋에서는 :data:`UNUSABLE_ABOVE_ROWS`가 ``svr``를 제외하므로 어떤
뽑기도 거기 닿지 않는다. 더 작은 회귀 데이터셋이 들어오면 다시 볼 줄이 그것이다.

**이 팔이 못 받는 것.** registry의 ``notes``, 데이터셋 카드의 집계, 앞 뽑기의 점수 — 하나도 못 받는다.
그게 요점이다. 이건 바닥이다. 예외는 :data:`UNUSABLE_ABOVE_ROWS` 하나이고, LLM 팔에 산문으로 건네진
정보를 이 팔에 청구하지 않기 위해 있다(``RESULTS.md`` 참고).

**뽑기 순서는 (데이터셋, 시드)의 순함수다.** 판정은 LLM 팔의 실제 학습 횟수 ``k``를 이 팔의 *앞 k회*
뽑기에 대고 비교하므로, 순서가 어떤 점수보다도 먼저 고정되어야 한다 — (데이터셋, 시드)마다
``random.Random`` 하나로 차례대로 뽑고, 다시 돌리면 재현된다. 다음에 무엇을 뽑을지 정하려고 점수를 읽는
코드는 여기 없다.
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

# 실행기가 하이퍼파라미터 이름을 풀어 내는 alias 표. 복사하지 않고 import하는 이유는 이 표가 뽑기에
# 손잡이가 실제로 몇 개인지를 정하기 때문이다: ``hist_gbdt``는 ``max_iter``와 ``n_estimators``를 둘 다
# 공표하는데 같은 손잡이다. 둘을 다 뽑으면 한 값을 넣고 조용히 덮어쓴다 — 버려진 것으로 기록되지도
# 않는다 — 그래서 뽑은 키를 먼저 정규 이름으로 접는다. sklearn을 이 프로세스로 끌어들이는 유일한
# import이고(``nodes/training.py``는 orchestrator가 sklearn에서 자유로워야 해서 그래서 거부한다),
# 곧 다섯 번 학습을 띄울 bench 스크립트는 상관없다.
from automl_agent.scripts.train import ALIASES as PARAM_ALIASES
from bench.datasets import ALL, BENCH_DIR, BY_NAME, CHEAP_SEEDS, Dataset

RUNS_DIR = BENCH_DIR / "runs"

# 다른 두 팔이 받는 학습 예산 ``--max-iterations``와 같게 맞춘다.
DEFAULT_DRAWS = 5

# 학습마다의 subprocess 타임아웃. 여기서 고른 수가 아니라 저장소 자신의 수다:
# ``nodes/training.py``가 ``config.train_timeout_sec``를 넘기고 그게 ``time_budget_sec``이고 그
# 기본값이 이것이다. 그래서 뽑기는 LLM 시도가 죽는 것과 같은 지점에서 죽고 기록에도 같은 모양으로
# 남는다 — ``too_slow``.
#
# (카드에는 profiler가 쓰고 아무도 읽지 않는 ``constraints.max_train_time_sec`` = 600도 있다. 그래서
# 여기서도 쓰지 않는다 — 팔들은 셋 모두에 실제로 강제되는 한계를 공유해야 한다.)
TIMEOUT_SEC = DEFAULT_TIME_BUDGET_SEC

# subprocess마다 못박는다. 위생이 아니라 측정된 이유다: xgboost의 ``tree_method="hist"``는 스레드
# 수가 고정되면 비트 단위로 같게 재실행되고 스레드 수가 다르면 완전히 다른 모델을 만든다
# (``max |Δproba|`` 0.34, 모든 행이 다름). 1..20 스레드에 걸친 폭은 balanced_accuracy 0.0077이고,
# 이 벤치마크가 판정에 쓰는 짝 판정 반폭의 0.99배다. harness 어디에도 스레드 수를 기록하는 곳이
# 없으므로 밖에서 못박고 ``RESULTS.md``에 적는다. ``FINDINGS-mimic.md`` 참고.
OMP_THREADS = "1"

# registry의 ``note``를 존중하는 단 한 곳. LLM 팔은 그 문장을 산문으로 받는데 균등 뽑기는 읽을 수
# 없기 때문이다: ``svc``/``svr``의 note는 "행에 대해 이차식이라 대략 20k행 위에서는 쓸 수 없다"다.
# 이게 없으면 다섯 중 셋에서 확정된 타임아웃에 뽑기 한 번을 태우게 되고, 그 차이는 계획 능력의 차이가
# 아니다 — 같은 문장을 한 팔에는 주고 다른 팔에는 안 준 것이다. 모델 id → 행 상한.
UNUSABLE_ABOVE_ROWS: dict[str, int] = {"svc": 20_000, "svr": 20_000}

# registry가 이름을 대는 문자열 매개변수에 대한 sklearn의 문서화된 선택지 집합. 계획이 요구할 수
# 있는 무엇에든 이 팔이 닿을 수 있게 전체 집합을 적는다. ``None``은 실재하는 ``class_weight``(클래스를
# 그대로 둔다)이고 대부분의 시도가 그 값으로 돈다. ``class_weight``가 ``ALLOWED_STRINGS``에 있어서
# ``sanitise_hyperparams``가 남겨 둔다.
CHOICES: dict[str, tuple[Any, ...]] = {
    "class_weight": (None, "balanced"),
    "weights": ("uniform", "distance"),
    "kernel": ("rbf", "linear", "poly", "sigmoid"),
}

# ``preprocessing``은 하이퍼파라미터가 아니라 계획 블록이고 LLM도 이걸 쓰므로 이 팔도 뽑는다.
# 전략은 ``nodes.training.preprocessing_block``이 받아 주는 것들이다.
IMPUTE_STRATEGIES = ("median", "mean", "most_frequent")
# ``impute: "none"``은 estimator에 NaN을 그대로 넘겨 거기서 쪼개게 한다. NaN을 못 받는 계열에는
# 실행기의 ``_wrap_preprocessing``이 요청을 낮추므로, 어디에나 제시하면 조용히 ``median``이 되는
# 뽑기만 나온다. 그래서 그걸 유지하는 두 계열에만 제시한다.
NAN_NATIVE = ("hist_gbdt", "xgboost")

# ``mlp``의 구조 손잡이. 스칼라가 아니라 목록이라 ``LIMITS``에 항목이 없고,
# ``sanitise_hyperparams``는 정수 목록을 통과시킨다. 층은 한둘, 너비는 나머지와 같은 로그 균등 규칙.
HIDDEN_WIDTH = (8, 512)
MAX_HIDDEN_LAYERS = 2

# 로그 균등 규칙이 갈리는 지점. 두 자릿수: 그 아래면 범위가 좁아서 균등이 왜곡하지 않고, 그 위면
# 균등이 뽑기의 절반을 맨 위 한 자릿수에 몰아넣고 그 결과를 탐색이라 부르게 된다.
LOG_UNIFORM_DECADES = 100.0


def is_log_scale(low: float, high: float) -> bool:
    return low > 0 and high / low >= LOG_UNIFORM_DECADES


def is_integer_param(low: float, high: float) -> bool:
    """``LIMITS``가 이 매개변수의 경계를 정수로 적었는가. 즉 크기가 아니라 개수인가.

    리터럴의 값이 아니라 *타입*을 본다. ``reg_lambda``와 ``l2_regularization``은 ``(0.0, 100.0)``으로
    공표되어 있어서, 값으로 검사하면 둘 다 정수라고 판정하고 모든 뽑기를 정수로 반올림한다 —
    정규화 강도가 보통 사는 ``(0, 1)`` 구간을 버리는 것이다. 표는 이미 적는 방식으로 두 종류를 구분하고
    있고(``n_estimators``는 ``(1, 2000)``, ``learning_rate``는 ``(1e-4, 1.0)``) 모든 항목이 제대로
    적혀 있으므로, 그 표기를 읽는 것이 기계적이면서 맞다.
    """
    return isinstance(low, int) and isinstance(high, int)


def draw_number(rng: random.Random, low: float, high: float) -> float | int:
    """모듈 docstring이 적은 규칙 하나에 따라 ``[low, high]``에서 값 하나."""
    if is_log_scale(low, high):
        value = math.exp(rng.uniform(math.log(low), math.log(high)))
    else:
        value = rng.uniform(low, high)
    if is_integer_param(low, high):
        return max(int(low), min(int(high), int(round(value))))
    return round(value, 6)


def canonical_params(model: str, params: list[Any]) -> list[str]:
    """모델이 공표한 손잡이. alias를 접고, 먼저 나온 표기를 남기고, 순서를 지킨다.

    ``hist_gbdt``는 ``max_iter``와 ``n_estimators``를 공표하고 실행기는 뒤를 앞에 맞추므로 둘은 한
    손잡이다. 둘을 다 설정하는 뽑기는 자기 값 하나를 버리는 것이다.
    """
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
    """(모델, hyperparams, preprocessing) 한 번 뽑기. 점수도 카드도 이력도 읽지 않는다."""
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
            # registry는 공표하는데 LIMITS도 CHOICES도 경계를 주지 않는 손잡이. 여기서 지어낸
            # 범위로 채우는 대신 estimator의 기본값에 두고 기록한다.
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
        # 노드 자신의 allowlist를 통과시켜, 이 블록이 LLM의 것을 검증하는 코드와 같은 코드로
        # 검증되게 한다. 그대로 통과해야 한다 — "해야 한다"가 통과시키는 이유다.
        "preprocessing": preprocessing_block({"preprocessing": drawn_preprocessing}, {}),
        "untouched_params": untouched,
    }


def menu(card: dict[str, Any], n_rows: int) -> tuple[dict[str, Any], ...]:
    """이 팔이 존중해도 되는 note 하나를 적용한 뒤, 뽑을 수 있는 모델들."""
    offered = available_models(card_task(card))
    return tuple(
        entry
        for entry in offered
        if n_rows <= UNUSABLE_ABOVE_ROWS.get(str(entry["id"]), sys.maxsize)
    )


def run_dir(dataset: Dataset, seed: int) -> Path:
    return RUNS_DIR / "random" / f"{dataset.name}-seed{seed}"


def repo_relative(path: Path) -> str:
    """저장소 루트에 대한 ``path`` 상대 경로. 전혀 다른 데 있으면 절대 경로.

    올리지 않고 물러난다: 패키지가 다른 데 설치된 checkout도 쓸 만한 기록을 얻고, 그 경우에는 절대
    경로가 솔직한 답이다.
    """
    root = Path(__file__).resolve().parents[1]
    try:
        return path.resolve().relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def train_env() -> dict[str, str]:
    """자식 프로세스의 환경: 루프 자신의 환경에 스레드 풀 전부를 못박은 것(:data:`OMP_THREADS`)."""
    env = dict(utf8_env())
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = OMP_THREADS
    return env


def train_config(dataset: Dataset, seed: int, card: dict[str, Any], plan: dict[str, Any]) -> dict:
    """뽑기 하나에 대한 실행기 config. ``nodes/training.py``가 조립하는 방식 그대로.

    필드마다 같게 두되 일부러 빠뜨린 것이 둘 있다. ``simulate`` — 이 카드들은 실패 훈련을 선언하지
    않는다. ``paired_baseline`` — 루프는 원장이 행 단위 델타를 보고할 수 있게 시도마다 자기 최고 기록을
    가리키게 하는데, 이 팔에는 가리킬 "지금까지"가 없다. 이 벤치마크가 판정에 쓰는 짝 비교는 *팔* 사이의
    것이고, 학습 안이 아니라 판정기가 test 행에서 계산한다.
    """
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
    # 여기 상수가 아니라 카드에서 읽는다: LLM 팔의 학습은 profiler가 쓴 값에 묶여 있고, 상한이
    # 다르면 다른 실험이 된다.
    limit = (card.get("constraints") or {}).get("memory_limit_mb")
    if limit:
        config["memory_limit_mb"] = limit
    return config


def spawn(command: list[str], log_path: Path) -> tuple[int, bool]:
    """train.py 호출 하나를 돌린다. ``(returncode, timed_out)``을 돌려주고 콘솔 로그를 쓴다."""
    console = ""
    returncode = -1
    timed_out = False
    try:
        completed = subprocess.run(  # noqa: S603 - 고정된 스크립트, 셸 없음
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
    """config를 쓰고 ``train.py``를 돌리고, 그 뽑기가 낸 것을 돌려준다."""
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

    # ``train.py``는 어떤 실패든 결과 파일로 바꾸고 exit 0으로 끝내므로 ``status``가 권위이고,
    # return code는 파일이 아예 없을 때만 의미가 있다.
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
        # 점수가 안 난 뽑기는 0.0이 아니라 ``None``이다: 0도 정당한 balanced_accuracy이고
        # max()에서 "없음"을 이겨 버린다.
        "val_score": metrics.get(dataset.metric),
        "val_metrics": metrics,
        "val_split": result.get("split"),
        "model_file": (directory / MODEL_FILENAME).as_posix()
        if (directory / MODEL_FILENAME).exists()
        else None,
        "elapsed_sec": round(elapsed, 2),
        "dir": directory.as_posix(),
    }


def best_by_val(records: list[dict[str, Any]], k: int) -> dict[str, Any] | None:
    """**앞 k회** 뽑기 중 우승자. validation만 보고 고른다.

    ``k``는 LLM 팔의 실제 학습 횟수다. 뽑기 순서는 어떤 점수보다도 먼저 고정된 시드가 정했으므로
    "앞 k회"는 보고 나서 한 선택이 아니다. 동점은 먼저 나온 뽑기가 가져가고, 그게 ``best``에 대한 루프
    자신의 규칙이다 — 나중 시도는 현재 보유자를 *이겨야* 한다.
    """
    scored = [item for item in records[:k] if item.get("val_score") is not None]
    if not scored:
        return None
    return max(scored, key=lambda item: (float(item["val_score"]), -int(item["draw"])))


def score_test(dataset: Dataset, record: dict[str, Any]) -> dict[str, Any] | None:
    """``nodes/holdout.py``가 하듯, 뽑기 하나의 저장된 모델을 떼어 둔 test 행에서 채점한다.

    같은 호출이다: 그 뽑기 *자신의* config에 ``--score-model``을 더한다. 분할이 데이터 경로·정답 결측
    정책·시드의 함수이므로, 학습이 돌아간 그 파일을 다시 쓰는 것이 떼어 둔 행과 채점되는 행을 같게
    보장하는 방법이다.

    어떤 앞자리 ``k``가 고르는 뽑기만 채점한다 — :func:`test_candidates` 참고.
    """
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
    """test 분할이 볼 수 있는 뽑기 번호: 앞자리 우승자들, 그 밖에는 없다.

    판정 비교는 LLM 팔이 쓰게 되는 ``k``에서의 우승자를 필요로 하고, 그 집합은 많아야 기록 보유자 한
    사슬이다. 그것만 채점하면 test 행을 보는 횟수가 질문에 답할 수 있는 최소 집합으로 유지되고, 그 전부가
    validation만으로 골라졌으므로 어떤 test 점수도 선택에 영향을 줄 수 없다. 덤으로 ``selection_gap``이
    어떤 ``k``가 고를 수 있었던 모델에 대해서만 정확히 보고된다.
    """
    chosen = {(best_by_val(records, k) or {}).get("draw") for k in range(1, draws + 1)}
    return sorted(draw for draw in chosen if isinstance(draw, int))


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
    # (데이터셋, 시드)마다 생성기 하나: 재현되고, 다시 돌려도 "앞 k회" 규칙이 뜻을 지킨다. 시드
    # 하나가 아니라 쌍으로 seed하므로 시드 42의 두 데이터셋이 같은 모델 수열을 뽑지 않는다.
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
        # 저장소 상대 경로. ``TRAIN_SCRIPT``는 절대 경로이고 이 파일은 커밋된다: 절대 형태는 돌린
        # 사람을 공개 기록에 써 넣고("C:/Users/<이름>/...") 다음 기계에서 같은 문자열이 아니라서,
        # 독자가 옮긴 checkout과 다른 실행기를 구분할 수 없다. 이 기록의 목적은 팔들을 학습시킨
        # 스크립트가 *어느 것*인지 대는 것이다.
        "train_script": repo_relative(TRAIN_SCRIPT),
        # 예산 맞춤 규칙을 미리 계산해 둔다: 앞자리 길이마다 val로 고른 우승자. 그래서 판정기는
        # 아무것도 다시 돌리거나 다시 정하지 않고 LLM 팔의 k를 읽어 낸다.
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
