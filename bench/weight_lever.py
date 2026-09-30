"""speeddating의 이김은 재계획이었나, 카드에서 읽은 수 하나였나?

``docs/HARD-BAR.md``는 루프를 실제로 돌게 만들고도 구분되지 않음으로 끝났다 — 판정 분모인 이진
4개 중 이김 하나. 그 하나에 사후 설명을 적어 두었다: 규칙 폴백은 ``class_weight``를 1에서
``WEIGHT_STEP = 1.5``씩, 그것도 Critic이 그 가지를 다시 고를 때만 올리므로, 5.07을 요구하는 카드
앞에서 speeddating은 1.5에 얼어붙었고 LLM 팔은 그 비율을 한 걸음에 읽었다. 그 문서는 스스로 그
설명을 사후·n=4라고 적고 확인은 밖에 남겼다. 이 파일이 그 확인이다.

사전 등록은 ``docs/WEIGHT-LEVER.md``이고, 이 적합들이 존재하기 전에 커밋됐다.

레버 하나. 각 다리는 기록된 ``bar65nollm`` 우승자의 ``train_config.json``에서
``hyperparams.class_weight``만 바꾼 것이다 — 같은 계열, 같은 나머지 하이퍼파라미터, 같은 전처리,
같은 행, 같은 시드, 같은 지표. 떨어뜨린 키는 ``paired_baseline`` 하나이고 이유는
``bench/random_search.py``와 같다: 루프는 매 시도를 자기 최고 기록에 겨누어 행 단위 Δ를 장부에
적지만, 적합 한 번에는 겨눌 "지금까지"가 없다. 그 키가 가리키는 파일은 gitignore라, 두면 이
기계에서만 풀린다.

이 설명은 *차등* 예측을 하고, 그래서 실패할 수 있다: 회수가 speeddating(가중치 1.5 대 비율
5.07)에서 일어나고 spambase(비율 1.54, 줄 것이 없음)에서는 일어나지 않아야 한다. "무언가
올라간다"만 예측하는 실험은 잡음에도 통과한다.

비교 상대인 팔을 다시 돌리지 않는다. ``bar65llm``과 ``bar65nollm``은 ``HARD-BAR.md``의 기록된
산출물이고 이 모듈은 읽기만 한다 — 즉 주 Δ는 독립적으로 다시 잰 것이 아니라 *재사용*이고, test
행도 세 번째 비교를 새로 얹는 같은 행이다. 둘 다 판정 파일에 적는다.

팔들이 돌았던 것처럼 ``OMP_NUM_THREADS=1``로 돌린다. 값은 출력에 기록된다.

사용법:
    python -m bench.weight_lever --fit          # 36회 적합 (API 호출 없음)
    python -m bench.weight_lever                # 판정
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

# 판정 분모인 이진 4개. ``bench/datasets.py`` 순서이고, 다시 나열하지 않고 ``JUDGED``에서 받는
# 이유는 그 모듈이 적어 둔 것과 같다 — 분모를 따로 베껴 적은 기준은 다른 문서들이 세는 분모와
# 어긋날 수 있다.
DATASETS: tuple[str, ...] = tuple(item.name for item in JUDGED)

SEEDS: tuple[int, ...] = (42, 43, 44)

# 이 파일이 읽기만 하고 쓰지 않는 기록된 팔.
RULES = "bar65nollm"
LOOP = "bar65llm"

# 이 실험이 적합하는 다리 셋. 기준이 말하는 것은 ``ratio``이고 나머지 둘은 대조군이다. 각각이
# 무엇을 대조하는지는 ``docs/WEIGHT-LEVER.md``에 있다.
RATIO = "ratio"
BALANCED = "balanced"
REFIT = "refit"
VARIANTS: tuple[str, ...] = (RATIO, BALANCED, REFIT)

# 커밋된 카드가 말하는 값. import 때 읽지 않고 박아 둔다 — 오늘의 카드에서 자기 예측을 다시
# 계산하는 사전 등록은 아무것도 예측하지 않는다. 커밋된 카드가 여전히 이 값을 내는지는
# ``tests/test_bench_weight_lever.py``가 확인한다.
CARD_RATIOS: dict[str, float] = {
    "adult": 3.18,
    "bank-marketing": 7.55,
    "speeddating": 5.07,
    "spambase": 1.54,
}

# 기록된 규칙 팔 우승자가 실제로 들고 있던 양성 클래스 가중치. 세 시드에서 모두 같으므로 사다리가
# 분할마다 다른 단에서 얼어붙은 것은 아니다. ``None``은 spambase — 비율이 1.54라 사다리가 건드린
# 적이 없다. 값들은 ``WEIGHT_STEP = 1.5``의 거듭제곱이다: 1.5, 2.25, 5.0625(설정이 마지막을
# 5.062로 반올림한다).
#
# 설명 전체가 바로 *이* 수에 대한 것이라 박아 둔다. 다시 만든 아카이브가 다른 단에서 나오면 이
# 실험이 재는 대상이 바뀌는데, 모듈이 조용히 다른 것을 재는 대신 테스트가 그것을 잡는다.
RECORDED_WEIGHTS: dict[str, float | None] = {
    "adult": 2.25,
    "bank-marketing": 5.062,
    "speeddating": 1.5,
    "spambase": None,
}

# ``params`` 목록에 ``class_weight``를 내놓는 계열. 이 집합 밖의 규칙 팔 우승자는 추정기가 무시할
# 키를 받는 대신 칸이 거부된다 — ``xgboost``의 레버는 ``scale_pos_weight``이고, xgboost 설정에
# ``class_weight``를 써 넣으면 일어나지 않은 처치를 기록하는 것이 된다. 기록된 우승자 열둘은 전부
# ``hist_gbdt`` 아니면 ``logreg``다.
#
# 위 :data:`CARD_RATIOS`와 달리 나열하지 않고 메뉴에서 유도한다. 이건 예측이 아니라 가드다 — 묻는
# 것이 "이 계열의 메뉴가 그 키를 내놓는가"이고, registry에 대한 사실이라 어느 순간에도 정답이
# 하나다. 손으로 베낀 첫 초안이 그것을 틀렸다 — registry id가 아닌 ``svm``을 적었고, 그래서 id이면서
# ``class_weight``를 실제로 받는 ``svc``를 빠뜨렸다.
WEIGHT_FAMILIES: frozenset[str] = frozenset(
    entry["id"] for entry in registry() if "class_weight" in entry["params"]
)

# 이 기준이 말하는 데이터셋에 대해 ``HARD-BAR.md``가 소수 네 자리로 발표한 주 Δ. 판정은 이 값을
# 쓰지 않는다 — ``bar65llm − bar65nollm``을 스스로 채점한 다리에서 정확히 다시 계산한 뒤 대조한다.
# 여기 두는 이유는 둘이 어긋나는 것이 아무도 확인하지 않은 수가 아니라 실패가 되게 하는 것이다.
PRIMARY_DELTAS: dict[tuple[str, int], float] = {
    ("speeddating", 42): 0.0819,
    ("speeddating", 43): 0.0777,
    ("speeddating", 44): 0.0761,
}

# 다시 계산한 주 Δ가 ``HARD-BAR.md``가 발표한 소수 네 자리에서 얼마나 떨어져도 되는가.
PUBLISHED_TOLERANCE = 5e-5

# 크기 바. 여기서 잰 것이 아니라 수입한 것이다 — ``REPEATS.md``에서 ``llm`` 팔의 같은 설정 반복이
# 낸 반폭의 중앙값이고, 낡은 프롬프트·낡은 바에서 쟀다. 이 실험은 바닥을 다시 재지 않으므로(그건
# 그것 자체로 하나의 실험이다) 판정 파일에 수의 출처를 적는다.
NOISE_FLOOR = 0.0122
NOISE_FLOOR_SOURCE = "docs/REPEATS.md (낡은 프롬프트·낡은 바에서 잰 수입니다)"

# 재적합 바닥. ``FINDINGS-mimic.md``의 A2에서 왔다 — 키 하나까지 같은 xgboost 설정을 다시
# 적합하니 balanced_accuracy가 0.0026 떨어진 곳에 앉았다. 그 흔들림은 *적합*에서, xgboost의
# histogram 경로에서 나왔다. ``hist_gbdt``와 ``logreg``에도 그런 것이 있는지는 여기서 처음 잰다.
# ``refit`` 다리가 이보다 더 움직인 칸은 처치 측정을 담을 수 없다.
REFIT_TOLERANCE = 0.0026
REFIT_TOLERANCE_SOURCE = "docs/FINDINGS-mimic.md A2 (xgboost 재적합, 이 경로에서는 처음 잽니다)"

# 실행 시드가 아니다. 파일 안의 모든 쌍이 한 재추출 수열에서 나오게, 그리고 판정 파일마다
# ``seed`` 하나를 읽는 ``bench/recheck.py``가 전부 다시 계산할 수 있게 박아 둔다.
RESAMPLE_SEED = JUDGED_SEED

WEIGHT_RUNS = RUNS_DIR / "weightlever"

# 적합 하나당 subprocess timeout. ``bench/random_search.py``가 spawn할 때 쓰는 것과 같다.
TIMEOUT_SEC = 3600
OMP_THREADS = "1"


def _key(dataset: str, seed: int) -> str:
    """판정 항목의 이름. 양쪽이 다 움직이므로 둘 다 들어간다."""
    return f"{dataset}-seed{seed}"


def run_dir(dataset: str, seed: int, variant: str) -> Path:
    return WEIGHT_RUNS / f"{dataset}-seed{seed}" / variant


def recorded_winner_dir(dataset: str, seed: int) -> Path:
    """기록된 규칙 팔 우승자의 iteration 디렉터리. 그 ``history.json``에서 읽는다."""
    directory = ARTIFACTS_DIR / f"{RULES}-{dataset}-seed{seed}"
    history = read_json_object(directory / "history.json")
    if history is None:
        raise Refusal(f"{dataset} 시드 {seed}: {RULES}의 history.json이 없습니다")
    iteration = dict(history.get("best") or {}).get("iteration")
    if not isinstance(iteration, int):
        raise Refusal(f"{dataset} 시드 {seed}: best.iteration이 정수가 아닙니다 ({iteration!r})")
    return directory / "train" / f"iter_{iteration:02d}"


def positive_weight(config: dict[str, Any]) -> float | None:
    """설정 안의 양성 클래스 가중치. 키가 없으면 ``None``.

    맵의 키는 **문자열**로 도착한다 — ``json``에는 정수 키가 없으므로 루프의 ``{1: 1.5}``는
    ``train_config.json``에 ``{"1": 1.5}``로 앉는다. int로 되읽으면 조용히 아무것도 못 찾고
    "가중치 없음"이라고 보고하는데, 그건 이 실험이 다른 상태로 취급하는 바로 그것이다.
    """
    weight = dict(config.get("hyperparams") or {}).get("class_weight")
    if isinstance(weight, dict):
        for key in ("1", 1):
            if key in weight:
                return float(weight[key])
        return None
    return None


def treated_config(dataset: str, seed: int, variant: str) -> dict[str, Any]:
    """기록된 규칙 팔 우승자의 설정에서 ``class_weight``만 바꾼 것.

    두 가지에는 맞춰 주지 않고 거부한다. 둘 다 일어나지 않은 처치가 일어났다고 기록에 적게 만들기
    때문이다: 추정기에 ``class_weight``가 없는 계열, 그리고 :data:`RECORDED_WEIGHTS`가 박아 둔
    단이 아닌 기록된 가중치.
    """
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
    # 모듈 docstring 참고: 루프는 매 시도를 최고 기록에 겨누지만 적합 한 번에는 겨눌 것이 없고,
    # 이 키가 가리키는 파일은 gitignore다.
    config.pop("paired_baseline", None)
    hyperparams = dict(config.get("hyperparams") or {})
    if variant == RATIO:
        hyperparams["class_weight"] = {"0": 1.0, "1": CARD_RATIOS[dataset]}
    elif variant == BALANCED:
        hyperparams["class_weight"] = "balanced"
    elif variant == REFIT:
        pass  # 대조군은 기록된 값을 건드리지 않은 것이다
    else:
        raise Refusal(f"알 수 없는 처치: {variant!r}")
    config["hyperparams"] = hyperparams
    return config


# --------------------------------------------------------------------------- #
# 다리 셋 적합하기
# --------------------------------------------------------------------------- #


def train_env() -> dict[str, str]:
    """자식 프로세스의 환경: 루프 자신의 것에 스레드 풀을 전부 못박은 것.

    위생이 아니다. A2는 스레드 수에 따라 balanced_accuracy가 0.0077 벌어지는 것을 쟀고, 그건
    짝지은 판정 하나의 반폭의 0.99배다 — 그리고 이 실험의 ``refit`` 대조군 전체가 재적합이 무엇을
    움직이지 *않는다*는 주장이다.
    """
    env = dict(utf8_env())
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = OMP_THREADS
    return env


def fit_one(dataset: str, seed: int, variant: str) -> dict[str, Any]:
    """설정을 쓰고 ``train.py``를 한 번 돌린다. 적합이 낸 것을 돌려준다."""
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
        completed = subprocess.run(  # noqa: S603 - 고정된 스크립트, 셸 없음
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

    # ``train.py``는 모든 실패를 결과 파일로 바꾸고도 0으로 끝나므로 ``status``가 기준이고,
    # return code는 파일이 아예 없을 때만 뜻이 있다.
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


# --------------------------------------------------------------------------- #
# 판정하기
# --------------------------------------------------------------------------- #


def weight_leg(dataset: str, seed: int, variant: str) -> Winner:
    """적합된 다리 하나. 이 모듈이 test 점수를 본 적 없는 :class:`Winner`로 돌려준다.

    ``recorded_test``가 ``nan``인 것은 일부러이고 지름길이 아니다. holdout은 루프가 끝난 뒤 그
    루프의 우승자에 대해 한 번 돈다 — 어떤 루프의 우승자도 아니었던 다리는 기록된 test 점수가
    어디에도 없고, 이건 ``bench/replan.py``의 ``first_iteration_winner``와 같은 처지다. 그 확인을
    대신하는 것이 지문이다: ``_adjudicate``는 이 다리의 test 행이 기록된 두 팔의 행과 같은 해시를
    내지 않으면 칸을 거부한다.
    """
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
    """이 칸이 받칠 수 있는 비교. ``WEIGHT-LEVER.md``가 나열한 순서다."""
    pairs: list[tuple[str, str, str]] = []
    if LOOP in scored and RULES in scored:
        # 읽는 것이 아니라 다시 계산한다: 기준의 절반 바는 *이* 수의 절반이고,
        # :data:`PRIMARY_DELTAS`는 그것을 HARD-BAR.md가 발표한 값과 대조만 한다.
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

    # 마지막에 둔다. 중간에 거부되면 다리가 남지 않는다.
    if legs is not None:
        legs.update(scored)


def adjudicate(
    dataset: str, seed: int, resamples: int, legs: dict[str, Scored] | None = None
) -> DatasetVerdict:
    """(데이터셋, 시드) 하나. 거부는 ``paired.py``처럼 쌍을 비운다."""
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


# --------------------------------------------------------------------------- #
# 어느 칸이 측정을 담을 수 있는지 정하는 재적합 대조군
# --------------------------------------------------------------------------- #


def refit_check(results: dict[tuple[str, int], DatasetVerdict]) -> list[dict[str, Any]]:
    """기록된 설정을 다시 적합하면 그것이 재현되는가? 예측: 예측이 동일하다.

    이것을 통과하지 못한 칸은 처치 측정을 담을 수 없다 — ``WEIGHT-LEVER.md``는 흔들림을 빼는 대신
    그 칸을 판정 밖에 둔다. 부호를 모르는 흔들림은 보정이 아니다.
    """
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
    """다시 계산한 주 Δ를 ``HARD-BAR.md``가 찍은 소수 네 자리와 대조한다."""
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


# --------------------------------------------------------------------------- #
# 사전 등록된 기준
# --------------------------------------------------------------------------- #


def _cell(result: DatasetVerdict | None, refit_ok: bool) -> dict[str, Any]:
    """(데이터셋, 시드) 하나를 조건 셋이 읽는 것만으로 줄인 것."""
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
        # 조건 1: 0을 벗어나고 *동시에* 설명하려는 격차의 절반 이상.
        "recovery_clears_zero": (not recovery.identical_predictions) and recovery.ci_low > 0,
        "recovery_reaches_half": recovery.delta >= half,
        "remaining": remaining.delta,
        "remaining_ci": [remaining.ci_low, remaining.ci_high],
        # 조건 2: 남은 것이 0과 구분되지 않거나 바닥 아래다.
        "remaining_closed": (
            remaining.identical_predictions
            or (remaining.ci_low <= 0 <= remaining.ci_high)
            or abs(remaining.delta) < NOISE_FLOOR
        ),
        # 조건 3, 차등 데이터셋에서.
        "recovery_under_floor": recovery.identical_predictions or abs(recovery.delta) < NOISE_FLOOR,
    }


def criterion(results: dict[tuple[str, int], DatasetVerdict]) -> dict[str, Any]:
    """``docs/WEIGHT-LEVER.md``에 이 적합들보다 먼저 커밋된 기준을 판정한다.

    > **설명이 산다는 조건** — 셋 **전부** 충족:
    >
    > 1. **회수**: speeddating 시드 42·43·44 **전부**에서 ``ratio − bar65nollm``의 95% CI가 0을
    >    걸치지 않게 양수이고, 그 Δ가 같은 시드의 주 Δ의 **절반 이상**이다.
    > 2. **남은 격차**: speeddating 세 시드 **전부**에서 ``bar65llm − ratio``의 CI가 0을 걸치거나
    >    |Δ| < 0.0122이다.
    > 3. **차등**: spambase 세 시드 **전부**에서 ``ratio − bar65nollm``의 |Δ| < 0.0122이다.
    >
    > **설명이 죽는 조건**: 1이 깨진다. 그 밖은 **부분 지지**.

    과반이 아니라 시드 전원 일치를 요구하는 이유는 ``HARD-BAR.md``와 같다: ``REPEATS.md``의 R2가
    *같은* 설정의 반복에서 0을 벗어난 짝지은 Δ를 12쌍 중 4쌍에서 냈다. 그래서 한 시드의 구간은
    처치의 증거가 아니고, n=3에서 셋 중 둘은 동전과 구분하기 어렵다.

    우선순위는 미리 정해 둔다: 없거나 거부된 칸이 있으면 부분 판정. ``refit`` 대조군이
    :data:`REFIT_TOLERANCE`보다 더 움직인 칸은 판정 밖이고, speeddating 세 칸 중 하나라도 판정
    밖이면 판정 불가다 — 양성 예측이 전적으로 그 세 칸에 걸려 있다.
    """
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


def payload(results: dict[tuple[str, int], DatasetVerdict], resamples: int) -> dict[str, Any]:
    ordered = [
        (dataset, seed) for dataset in DATASETS for seed in SEEDS if (dataset, seed) in results
    ]
    return {
        "generated_by": "bench/weight_lever.py",
        "preregistration": "docs/WEIGHT-LEVER.md",
        "checks": "docs/HARD-BAR.md",
        # 문서에만 적지 않고 파일에도 적는다: 비교 상대인 팔이 기록된 것이라 주 Δ는 재사용이고,
        # 여기 test 행은 세 번째 비교를 새로 얹는 같은 행이다. 독립 측정의 수를 세는 사람에게는
        # 둘 다 필요하다.
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
