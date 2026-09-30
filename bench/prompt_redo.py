"""``docs/RESULTS.md``의 헤드라인 판정이 지금 저장소가 쓰는 프롬프트에서도 살아남는가?

``docs/RESULTS.md``의 ``llm`` 실행 여섯은 전부 커밋 **셋**보다 먼저 만들어졌다. 그중 둘은 계획
프롬프트가 "이 시스템이 할 수 있는 것"이라고 적는 내용을 다시 썼고, 하나는 그 실행들이 쫓던 바를
옮겼다:

* ``d3e7f83``은 ``MODEL_REGISTRY``의 ``params``/``notes``를 실행기의 실제 폭까지 넓혔고
  (``docs/REGISTRY-GAP.md``), 그 목록은 계획·모델 선택 프롬프트에
  "## Models this system can actually run"으로 그대로 렌더링된다;
* ``0ef38f8``은 shape 가드를 넣으면서 같은 ``notes``를 다시 고치고, *실행기*가 하이퍼파라미터 키
  다섯 개를 다루는 방식까지 바꿨다(:data:`CODE_SENSITIVE_KEYS` 참고);
* ``12fa74b``은 ``auto`` 바의 바닥을 순위 천장으로 올렸고, 그게 이 여섯 카드 중 둘을 옮겼다 —
  bank-marketing 0.7465 → 0.8400, speeddating 0.7514 → 0.7712.

세 번째는 종류가 다른 변경이고 ``docs/PROMPT-REDO.md``는 그걸 적지 않았다: 앞의 둘은 계획이 무엇을
말할 수 있는지를 바꾸고, 세 번째는 *루프가 언제 멈춰도 되는지*를 바꾼다. 이 실험의 무료 절반이
찾아냈고, 무료 절반은 그러라고 있다. 더 높은 바를 마주한 실행은 더 많이 학습하고 주 짝은 그 학습 횟수에
맞춰지므로, ``fits``와 ``goal_threshold``를 팔마다 함께 공표하고 :func:`movement`가 둘을 한 표에 넣는다
— 독자가 움직인 학습 횟수에서 움직인 바를 추론하게 두지 않는다.

그래서 **헤드라인 표의 모든 행은 이 저장소가 더 이상 갖고 있지 않은 프롬프트에서 나온 수**이고,
jungle-chess 덧붙임도 그렇다. ``REGISTRY-GAP.md``가 스스로 그렇게 적었다: "이후 실행을 기존 표에
같은 열로 적으면 안 되고, 비교하려면 이 커밋 이후로 여섯 팔을 다시 돌리는 것이 필요합니다."
이 모듈이 그 재실행이고, ``docs/PROMPT-REDO.md``가 그 사전 등록이다.

**기준을 여기 다시 적지 않는다 — 재사용한다.** :func:`bench.paired.criterion`을
``challenger="llm2"``로 부르므로, 바("이진 4개 중 3개 이상에서 CI가 0을 걸치지 않게 이긴다"),
분모(:data:`bench.datasets.JUDGED_NAMES`), "부분 판정"과 "구분되지 않음" 사이의 우선순위가 모두 이미
커밋된 것이고, 두 번 타이핑되면서 어긋날 수 없다. :attr:`DatasetVerdict.dataset`이 접미사 없는 맨
데이터셋 이름을 들고 있는 이유도 그것이다: 기준이 그 이름으로 세므로, 장식된 이름은 분모에서 빠져나가
없는 데이터셋으로 조용히 읽힌다.

넷이 아니라 여섯 데이터셋. 애초에 분모에 없던 둘(회귀, 다중분류)은 ``RESULTS.md``가 보고하는 것과
같은 이유로 다시 돌리고 보고한다 — 따로, 절대 평균에 섞지 않고. 바에 더하면 그들이 존재하기 전에
못박힌 수를 움직이게 된다.

기록된 팔들은 다리로 추가해 다시 채점한다. 비용이 0이고 전이 둘을 산다:

* ``no_llm2 − no_llm``(``code_shift``) — 규칙 팔은 모델을 부르지 않으므로 이 짝은 *코드*가 움직였을
  때만 움직인다. ``docs/REPLAN-SEEDS.md``가 데이터셋 하나의 시드 셋에서 예측 동일로 쟀고, 여기서는
  여섯에 묻는다. 코드가 움직였는지는 말하지만 *어느* 코드인지는 말하지 않는다 — 실행기와 목표 유도가
  둘 다 이 짝의 같은 편에 있고, 그래서 ``spans_goal_change``를 Δ 옆에 공표한다.
* ``llm2 − llm``(``prompt_shift``) — 기준이 아니라 관찰. ``code_shift``가 동일로 나오면 이 행에 남는
  레버는 프롬프트이고, 동일이 아니면 이 행에는 레버가 둘이라 프롬프트의 효과라고 부를 수 없다. 이
  저장소의 원장 규칙을 저장소 자신의 측정에 적용한 것이다.

그 두 짝은 둘 다 코드 버전을 넘고, ``12fa74b``이 옮긴 두 카드에서는 목표 변경까지 넘는다. 두 다리가
서로 다른 바를 마주한 짝은 버리지 않고 표시한다 — 버리면 그 전이에 대한 유일한 측정을 버리는 것이다 —
그리고 표시는 공표되는 Δ에 붙어서, 파일과 그 유보를 기억하는 독자 사이에서 잃어버릴 수 없다.

``random`` 팔을 다시 뽑지 않고 재사용하는 이유. ``bench/random_search.py``는 목표를 전혀 읽지 않고,
판정은 다시 학습하지 않고 저장된 ``model.joblib``을 다시 채점하므로, 다시 뽑으면 질문은 안 바뀌고
뽑기만 바뀐다. 이게 어긋날 수 있는 유일한 경로는 하이퍼파라미터가 ``0ef38f8``이 *실행기의* 처리를
바꾼 키에 걸린 뽑기다 — 그 뽑기는 지금 다르게 학습되고, ``llm2``에 짝지우면 코드 버전을 넘어 비교하는
것이 된다. :func:`random_exposure`가 시드 42의 서른 뽑기 전부를 정확히 그 다섯 키에 대해 확인하고
결과를 판정 파일에 쓰므로, 주장이 아니라 확인할 수 있는 것이 된다.

사용법:
    python -m bench.prompt_redo                  # 범위 안의 여섯 데이터셋
    python -m bench.prompt_redo adult spambase   # 그보다 적으면 부분 판정이고, 그렇게 적는다
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from automl_agent.config import read_json_object
from automl_agent.threads import describe_thread_state, thread_state
from bench.datasets import BY_NAME, JUDGED_NAMES, JUDGED_SEED, Dataset
from bench.paired import (
    ARTIFACTS_DIR,
    OUT_DIR,
    RESAMPLES,
    SCORE_TOLERANCE,
    DatasetVerdict,
    Refusal,
    Scored,
    Winner,
    criterion,
    loop_winner,
    paired_delta,
    random_winner,
    report,
    reproduce,
    split_identity,
)
from bench.predictions import bundle_path, write_bundle
from bench.random_search import run_dir as random_run_dir
from bench.replan import run_shape

# 여섯 전부. 숫자가 들어온 뒤에 범위가 넓어지거나 좁아질 수 없도록 명령줄이 아니라 여기 못박는다.
# 이진 넷이 기준의 분모이고, 나머지 둘은 다시 돌려 그 옆에 보고하되 평균에 섞지 않는다
# (이유는 ``bench/datasets.py``가 길게 적는다).
DATASETS: tuple[str, ...] = (
    "adult",
    "bank-marketing",
    "speeddating",
    "spambase",
    "house_sales",
    "jungle-chess",
)

# 시드 하나이고, 판정된 시드다. ``RESULTS.md``가 기준을 시드 42에 못박은 이유는 세 팔이 공유하는
# 유일한 시드이기 때문이다. 다른 시드로 헤드라인을 다시 돌리면 붙은 판정이 없는 표가 나온다.
SEED = JUDGED_SEED

# 새 팔들. thread-id 접두사에 숫자를 붙인 이유는 ``llm-adult-seed42``와 그 옆 다섯이
# ``RESULTS.md``의 기록된 아카이브이고, 여기서 아무것도 그 안에 써서는 안 되기 때문이다.
ARMS: tuple[tuple[str, str], ...] = (("llm2", "llm2"), ("no_llm2", "nollm2"))

# ``RESULTS.md`` 자신의 팔들. 전이 둘을 위해 다리로 추가해 다시 채점한다. 무료다: 모델을 부르지도
# 다시 학습하지도 않는다.
RECORDED_LOOP: tuple[str, str] = ("llm", "llm")
RECORDED_RULES: tuple[str, str] = ("no_llm", "nollm")

CHALLENGER = ARMS[0][0]
RULES = ARMS[1][0]

# ``0ef38f8``이 실행기의 처리를 바꾼 다섯 키: clamp 셋, 표기 관문 하나, 이진 전용 버림 하나, 그리고
# 값을 ``xgboost``까지 넘겨 ``fit`` 안에서 죽던 ``early_stopping_rounds <= 0`` 분기. 그중 하나라도 든
# 기록된 ``random`` 뽑기는 오늘 코드에서 다르게 학습되고, 그 팔을 재사용하는 것이 짝에 코드 변경을
# 몰래 들여올 수 있는 유일한 경로다. :func:`random_exposure` 참고.
CODE_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "early_stopping",
        "early_stopping_rounds",
        "scale_pos_weight",
        "validation_fraction",
        "n_iter_no_change",
    }
)

# 헤드라인 표가 공표한 값. 독자가 문서 둘을 열지 않도록 판정 파일이 비교 대상을 들고 있게 한다.
# 주 비교 = 예산 맞춘 ``llm − random@k``.
RECORDED_PRIMARY: dict[str, float] = {
    "adult": 0.1599,
    "bank-marketing": 0.0860,
    "speeddating": 0.1267,
    "spambase": -0.0019,
    "house_sales": 0.3088,
    "jungle-chess": 0.2326,
}

# ``RESULTS.md``의 핵심 질문: "``llm`` > ``no_llm``이 아니면 기여하는 것은 LLM이 아니라
# 루프 구조입니다." 이 대비는 예산을 맞출 필요가 없다 — 두 팔이 분할과 상한을 공유한다.
RECORDED_VS_RULES: dict[str, float] = {
    "adult": 0.0046,
    "bank-marketing": 0.0913,
    "speeddating": 0.0652,
    "spambase": -0.0128,
    "house_sales": 0.0040,
    "jungle-chess": -0.0225,
}

# 데이터셋마다 기록된 ``llm`` 팔의 학습 횟수. 시스템이 나빠지지 않았는데도 판정이 약해질 수 있는
# 기제가 이것이다: 주 짝이 *이* 수에 맞춰지고, 더 많이 학습한 실행은 더 강한 ``random@k``를 마주한다.
RECORDED_FITS: dict[str, int] = {
    "adult": 1,
    "bank-marketing": 1,
    "speeddating": 2,
    "spambase": 1,
    "house_sales": 1,
    "jungle-chess": 1,
}

# 어느 이진 데이터셋이 "4개 중 3개"를 떠받쳤는가. 결과 표가 무엇이 움직였는지 말할 수 있게 적는다.
RECORDED_WINS: tuple[str, ...] = ("adult", "bank-marketing", "speeddating")


# --------------------------------------------------------------------------- #
# leg
# --------------------------------------------------------------------------- #


def _arm_legs(dataset: Dataset, out: DatasetVerdict) -> tuple[dict[str, Winner], dict[str, Any]]:
    """새 팔 둘의 우승자, 그리고 각 실행이 자기를 어떻게 썼는지."""
    winners: dict[str, Winner] = {}
    shapes: dict[str, dict[str, Any]] = {}
    for arm, prefix in ARMS:
        directory = ARTIFACTS_DIR / f"{prefix}-{dataset.name}-seed{SEED}"
        winner = loop_winner(arm, directory, dataset.metric)
        if winner is None:
            out.missing_arms.append(arm)
            continue
        winners[arm] = winner
        shapes[arm] = run_shape(directory)
    return winners, shapes


def _recorded_legs(
    dataset: Dataset, out: DatasetVerdict
) -> tuple[dict[str, Winner], dict[str, Any]]:
    """이 데이터셋에서의 ``RESULTS.md`` 팔들. 읽기만 한다. jungle-chess까지 여섯 다 있다."""
    winners: dict[str, Winner] = {}
    shapes: dict[str, dict[str, Any]] = {}
    for arm, prefix in (RECORDED_LOOP, RECORDED_RULES):
        directory = ARTIFACTS_DIR / f"{prefix}-{dataset.name}-seed{SEED}"
        winner = loop_winner(arm, directory, dataset.metric)
        if winner is None:
            out.missing_arms.append(arm)
            continue
        winners[arm] = winner
        shapes[arm] = run_shape(directory)
    return winners, shapes


def random_exposure(dataset: Dataset) -> dict[str, list[str]]:
    """하이퍼파라미터가 ``0ef38f8``이 실행기를 바꾼 키에 닿는 기록된 뽑기들.

    비어 있는 것이 이 팔의 재사용을 허가하는 답이다: 그 키가 하나도 없는 뽑기는 두 코드 버전에서 같게
    학습되므로, ``llm2``에 짝지우면 레버 둘이 아니라 하나를 비교한다. 비어 있지 않아도 판정을 멈추지는
    않는다 — 판정 파일에 쓰고, 사전 등록이 그 데이터셋의 ``random`` 짝은 코드가 섞인 비교로 인용하라고
    적는다.

    **실행기만 확인하고 그 밖에는 아무것도 확인하지 않는다.** 이 팔에 닿을 수 있는 것이 그게 전부다.
    ``12fa74b``이 여섯 바 중 둘을 옮겼지만 그건 ``random`` 짝에 전혀 닿지 못한다:
    ``bench/random_search.py``는 목표를 읽지 않으므로 바가 무엇이든 뽑기와 val로 고르는 규칙이 같다.
    그래서 여기 결과가 비었다는 것은 코드가 하나도 안 바뀌었다는 주장이 아니라, *이 팔이 지나는 경로에서*
    코드가 안 바뀌었다는 더 좁은 주장이다.

    ``summary.json``이 아니라 뽑기마다 커밋된 ``train_config.json``에서 읽는다: config는 실행기가 받은
    것이고 summary는 돌아온 것을 기록한다.
    """
    found: dict[str, list[str]] = {}
    directory = random_run_dir(dataset, SEED)
    for draw in sorted(directory.glob("draw_*")):
        config = read_json_object(draw / "train_config.json")
        if config is None:
            continue
        keys = sorted(
            key for key in dict(config.get("hyperparams") or {}) if key in CODE_SENSITIVE_KEYS
        )
        if keys:
            found[draw.name] = keys
    return found


def _pairs_to_take(scored: dict[str, Scored], has_random: bool) -> list[tuple[str, str, str]]:
    """비교들. 재사용하는 기준이 읽을 수 있는 순서로.

    앞의 셋은 ``bench.paired._adjudicate``를 정확히 그대로 따른다 — 같은 kind, 같은 예산 맞춤 규칙,
    같은 ``best_of_5`` 부 비교. 다르게 만든 짝에서 평가된 기준은 같은 문장을 입은 다른 기준이기
    때문이다. ``criterion(results, "llm2")``는 ``b``가 ``random``으로 시작하는 첫 ``primary`` 짝을
    가져가고, 그래서 예산 맞춘 짝을 무엇보다 먼저 붙이며, ``b``가 그렇지 않은 ``llm2 − no_llm2``는
    같은 kind를 달아도 안전하다.
    """
    pairs: list[tuple[str, str, str]] = []
    for arm in (CHALLENGER, RULES):
        if arm in scored and has_random:
            fits = scored[arm].winner.fits
            pairs.append((arm, f"random@k={fits}", "primary"))
            if fits != 5:
                pairs.append((arm, "random@k=5", "best_of_5"))
    if CHALLENGER in scored and RULES in scored:
        pairs.append((CHALLENGER, RULES, "primary"))
    if RULES in scored and RECORDED_RULES[0] in scored:
        # 규칙 팔은 모델을 부르지 않으므로 코드가 움직였을 때만 움직인다.
        pairs.append((RULES, RECORDED_RULES[0], "code_shift"))
    if CHALLENGER in scored and RECORDED_LOOP[0] in scored:
        # ``code_shift``가 아니라고 말해 주기 전까지는 한 행에 프롬프트와 코드가 같이 있다 — 관찰.
        pairs.append((CHALLENGER, RECORDED_LOOP[0], "prompt_shift"))
    return pairs


def _adjudicate(
    dataset: Dataset, resamples: int, out: DatasetVerdict, legs: dict[str, Scored] | None = None
) -> None:
    metric = dataset.metric
    winners, shapes = _arm_legs(dataset, out)
    if not winners:
        # 데이터에 대한 거부가 아니다: 두 실행이 아직 없다. 그렇게 적는 이유는, 여기서 "구분되지
        # 않음"으로 읽히는 판정은 아무도 하지 않은 질문에 답하는 것이기 때문이다.
        raise Refusal(
            f"{dataset.name}: llm2도 no_llm2도 없습니다 — "
            "bench/scripts/run_prompt_redo.sh를 먼저 돌려야 판정할 수 있습니다"
        )
    recorded, recorded_shapes = _recorded_legs(dataset, out)
    winners.update(recorded)
    shapes.update(recorded_shapes)

    random_dir = random_run_dir(dataset, SEED)
    has_random = (random_dir / "summary.json").exists()
    if not has_random:
        out.missing_arms.append("random")
    else:
        # 새 팔마다 자기 학습 횟수의 앞자리 하나, 그리고 부 비교용으로 5. 두 팔의 k가 우연히 같을
        # 때 두 번 다시 유도하지 않도록 dict를 공유한다.
        needed = {5, *(winners[arm].fits for arm, _ in ARMS if arm in winners)}
        for k in sorted(needed):
            if k < 1:
                raise Refusal(f"{dataset.name}: 새 팔의 학습 횟수가 {k}회로 기록됐습니다")
            leg = random_winner(random_dir, k, metric)
            winners[leg.arm] = leg

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
        raise Refusal(
            f"{dataset.name}: 팔마다 test 지문이 다릅니다 ({listing}) — 판정하지 않습니다"
        )
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
            "selection_gap": s.winner.val_score - s.test_score,
            "dir": s.winner.directory.as_posix(),
            **shapes.get(name, {}),
        }

    for a_name, b_name, kind in _pairs_to_take(scored, has_random):
        out.pairs.append(paired_delta(scored[a_name], scored[b_name], metric, resamples, SEED, kind))

    # 중간에 거부가 나면 다리를 남기지 않도록 맨 마지막에: 짝이 비워진 항목의 다리를 든 번들은
    # 판정이 공표하지 않는 수를 ``recheck``에 내주게 된다.
    if legs is not None:
        legs.update(scored)


def adjudicate(
    dataset: Dataset, resamples: int, legs: dict[str, Scored] | None = None
) -> DatasetVerdict:
    """데이터셋 하나. ``paired.py``처럼 거부는 짝을 비운다."""
    out = DatasetVerdict(dataset=dataset.name, metric=dataset.metric, task=dataset.task)
    try:
        _adjudicate(dataset, resamples, out, legs)
    except Refusal as exc:
        out.verdict = "refused"
        out.refusal = str(exc)
        out.pairs = []
    return out


# --------------------------------------------------------------------------- #
# 기록된 표 옆에서, 무엇이 움직였는가
# --------------------------------------------------------------------------- #


def _pair(result: DatasetVerdict, a: str, b_prefix: str, kind: str) -> Any:
    return next(
        (p for p in result.pairs if p.a == a and p.kind == kind and p.b.startswith(b_prefix)),
        None,
    )


def _bar(arms: dict[str, dict[str, Any]], *names: str) -> float | None:
    """이 팔들이 마주한 ``auto`` 바. ``names`` 중 바를 기록한 첫 번째에서 읽는다.

    바는 팔의 속성이 아니라 카드와 코드의 속성이다 — 한 코드 버전의 모든 팔이 같은 방식으로 유도한다 —
    그래서 그중 아무거나가 그 버전을 대표해 답한다. ``random`` 다리에는 바가 아예 없으므로
    (``bench/random_search.py``는 목표를 읽지 않는다) 여기 넘기지 않고 아래 표시도 달지 않는다.
    """
    for name in names:
        bar = arms.get(name, {}).get("goal_threshold")
        if bar is not None:
            return float(bar)
    return None


def _publish_pairs(result: DatasetVerdict) -> list[dict[str, Any]]:
    """파일에 들어가는 그대로의 Δ들. 목표 변경을 넘는 짝에는 그 표시를 달아서.

    ``code_shift``와 ``prompt_shift``는 코드 버전을 넘고, ``12fa74b``이 바닥을 다시 깐 카드에서는 바까지
    넘는다: 두 다리가 서로 다른 점수에서 멈출 수 있었으니, 그 짝은 레버를 하나 넘게 들고 있고 어느 하나의
    것이라고 돌릴 수 없다. 버리지 않고 표시해서 남긴다 — 버리면 그 전이의 유일한 측정을 버리는 것이다 —
    그리고 산문이 아니라 *Δ에* 표시한다. 독자가 기억해야 하는 유보는 결국 잊히는 유보이기 때문이다.

    ``recheck``은 이 dict를 키로 읽으므로(``a``, ``b``, ``kind``, 그리고 Δ 필드들) 덧붙은 키는 그것이
    검산하는 것을 바꾸지 않고 함께 실려 간다.
    """
    rows: list[dict[str, Any]] = []
    for delta in result.pairs:
        row = dict(vars(delta))
        a_bar = result.arms.get(delta.a, {}).get("goal_threshold")
        b_bar = result.arms.get(delta.b, {}).get("goal_threshold")
        if a_bar is not None and b_bar is not None and a_bar != b_bar:
            row["spans_goal_change"] = True
            row["goal_threshold_a"] = a_bar
            row["goal_threshold_b"] = b_bar
        rows.append(row)
    return rows


def movement(results: list[DatasetVerdict]) -> list[dict[str, Any]]:
    """데이터셋마다 헤드라인 Δ 둘, 그때와 지금, 그리고 옆에 학습 횟수.

    ``fits``가 각주가 아니라 이 표에 있는 이유는, 시스템이 안 움직였는데 주 Δ가 움직이는 경로가 그것이기
    때문이다: 짝이 루프 팔 자신의 학습 횟수에 맞춰지고, ``RESULTS.md``가 그 부작용을 이미 공표했다 —
    "k가 작으면 ``random@k``가 약해집니다". 더 많이 다시 계획한 실행은 더 강한 상대를 마주하고 그만큼
    작은 이김을 보고한다.

    바가 같은 행에 있는 이유도 같고, 한 걸음 더 뒤에 있다. ``12fa74b``이 이 여섯 중 둘을 올렸고, 올라간
    바는 계획자가 하나도 안 움직여도 학습 횟수가 움직이는 가장 값싼 경로다: bank-marketing의 규칙 팔은
    두 코드 버전에서 반복 1·2를 비트 단위로 같게 내고도 계속 갔다. 0.7833이 0.7465는 넘었고 0.8400은
    넘지 못하기 때문이다. 옆에 바 없이 ``fits``만 읽으면 그게 더 많이 다시 계획한 계획자로 읽힌다.
    """
    rows: list[dict[str, Any]] = []
    for result in results:
        primary = _pair(result, CHALLENGER, "random", "primary")
        vs_rules = _pair(result, CHALLENGER, RULES, "primary")
        # 한 버전의 어느 팔이든 그 바를 대표해 답한다. 규칙 팔만 돌았을 때도(무료 절반의 상태)
        # 표가 루프 팔의 행으로 읽히도록 루프 팔을 먼저 본다.
        recorded_bar = _bar(result.arms, RECORDED_LOOP[0], RECORDED_RULES[0])
        now_bar = _bar(result.arms, CHALLENGER, RULES)
        rows.append(
            {
                "dataset": result.dataset,
                "judged": result.dataset in JUDGED_NAMES,
                "fits_recorded": RECORDED_FITS.get(result.dataset),
                "fits_now": result.arms.get(CHALLENGER, {}).get("fits"),
                "threshold_recorded": recorded_bar,
                "threshold_now": now_bar,
                "bar_moved": (
                    None
                    if recorded_bar is None or now_bar is None
                    else recorded_bar != now_bar
                ),
                "critic_runs_now": result.arms.get(CHALLENGER, {}).get("critic_runs"),
                "stop_reason_now": result.arms.get(CHALLENGER, {}).get("stop_reason"),
                "primary_recorded": RECORDED_PRIMARY.get(result.dataset),
                "primary_now": None if primary is None else primary.delta,
                "primary_ci": (
                    None if primary is None else [primary.ci_low, primary.ci_high]
                ),
                "vs_rules_recorded": RECORDED_VS_RULES.get(result.dataset),
                "vs_rules_now": None if vs_rules is None else vs_rules.delta,
                "vs_rules_ci": (
                    None if vs_rules is None else [vs_rules.ci_low, vs_rules.ci_high]
                ),
                "was_a_win": result.dataset in RECORDED_WINS,
            }
        )
    return rows


def core_question(results: list[DatasetVerdict]) -> dict[str, Any]:
    """같은 넷에서 같은 방식으로 센 ``llm2 − no_llm2`` — ``RESULTS.md`` 자신의 부 판정.

    그 문서에서 기준 문장이 아니라 산문으로 사전 등록됐다: "``no_llm`` 대비도 같은
    방식으로 함께 냅니다. ``llm`` > ``no_llm``이 아니면 기여하는 것은 LLM이 아니라 루프
    구조입니다." 그 문서는 거기에 4개 중 3개 모양을 적용하고 미달을 기록했다(이김 2 / 걸침 1 / 짐 1).
    이 읽기가 나오는 값에 따라 골라지거나 버려질 수 없도록, 표를 눈으로 보지 않고 여기서 계산한다.

    이건 기준이 **아니다**. 기준은 :func:`bench.paired.criterion`이고, 그건 ``random`` 대비다.
    """
    wins, losses, ties, absent = [], [], [], []
    for result in results:
        if result.dataset not in JUDGED_NAMES:
            continue
        pair = _pair(result, CHALLENGER, RULES, "primary")
        if result.verdict != "ok" or pair is None:
            absent.append(result.dataset)
        elif pair.identical_predictions or (pair.ci_low <= 0 <= pair.ci_high):
            ties.append(result.dataset)
        elif pair.ci_low > 0:
            wins.append(result.dataset)
        else:
            losses.append(result.dataset)
    missing = sorted(JUDGED_NAMES - {r.dataset for r in results}) + sorted(absent)
    tally = f"이김 {len(wins)} / 걸침 {len(ties)} / 짐 {len(losses)}"
    if missing:
        status = f"부분 — 이진 {len(JUDGED_NAMES)}개 중 일부가 없습니다 ({', '.join(missing)}). {tally}"
    elif len(wins) >= 3:
        status = f"루프 구조 위에 LLM이 더 얹습니다 — 이진 4개 중 {len(wins)}개 ({tally})"
    else:
        status = f"미달 — 3개 이상이 아닙니다 ({tally}). 기록된 표와 같은 방향입니다"
    return {
        "contrast": f"{CHALLENGER} − {RULES}",
        "note": "docs/RESULTS.md의 부 판정입니다. 사전 등록된 기준은 random 대비입니다",
        "wins": wins,
        "losses": losses,
        "indistinguishable": ties,
        "not_evaluated": missing,
        "recorded": RECORDED_VS_RULES,
        "status": status,
    }


def payload(results: list[DatasetVerdict], resamples: int) -> dict[str, Any]:
    return {
        "generated_by": "bench/prompt_redo.py",
        "preregistration": "docs/PROMPT-REDO.md",
        "remeasures": "docs/RESULTS.md",
        "seed": SEED,
        "resamples": resamples,
        "score_tolerance": SCORE_TOLERANCE,
        "threads": thread_state(),
        # 데이터셋마다 비어 있다는 사실이 코드 변경을 넘어 기록된 ``random`` 팔을 재사용하는 근거다.
        # 주장하지 않고 공표한다. :func:`random_exposure` 참고.
        "random_code_exposure": {
            r.dataset: random_exposure(BY_NAME[r.dataset]) for r in results
        },
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
                "pairs": _publish_pairs(r),
            }
            for r in results
        ],
        "movement": movement(results),
        # 다시 적지 않고 재사용한다: ``RESULTS.md``와 같은 바, 같은 분모, 같은 우선순위.
        "criterion": criterion(results, CHALLENGER, SEED),
        "criterion_no_llm": criterion(results, RULES, SEED),
        "core_question": core_question(results),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "헤드라인 표를 지금 프롬프트로 다시 잰다 (docs/PROMPT-REDO.md의 사전 등록, "
            "기준은 docs/RESULTS.md의 것을 그대로 재사용)"
        )
    )
    parser.add_argument(
        "datasets",
        nargs="*",
        default=[],
        help=f"이름 (기본: {' '.join(DATASETS)} — 그보다 적으면 부분 판정입니다)",
    )
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument(
        "--out", type=Path, default=None, help=f"기본: bench/runs/paired/prompt-redo-seed{SEED}.json"
    )
    return parser.parse_args(argv)


def _movement_text(recorded: float | None, now: float | None, ci: list[float] | None) -> str:
    if recorded is None:
        return "—"
    if now is None:
        return f"{recorded:+.4f} → (없음)"
    span = "" if ci is None else f" [{ci[0]:+.4f}, {ci[1]:+.4f}]"
    return f"{recorded:+.4f} → {now:+.4f}{span}"


def _bar_text(row: dict[str, Any]) -> str:
    """바. 움직였을 때만 크게 적는다 — "그대로"가 줄줄이 있는 열은 눈이 그냥 지나가게 만든다."""
    recorded, now = row["threshold_recorded"], row["threshold_now"]
    if recorded is None or now is None:
        return "—"
    if not row["bar_moved"]:
        return f"{now:.4f}"
    return f"{recorded:.4f}→{now:.4f} !"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    names = args.datasets or list(DATASETS)
    unknown = [n for n in names if n not in DATASETS]
    if unknown:
        print(
            f"사전 등록의 범위 밖 데이터셋: {', '.join(unknown)} (범위: {', '.join(DATASETS)})",
            file=sys.stderr,
        )
        return 2

    print(
        f"프롬프트 재측정 · 시드 {SEED} · 재추출 {args.resamples}회 · "
        f"{describe_thread_state(thread_state())}"
    )

    legs: dict[str, dict[str, Scored]] = {}
    results: list[DatasetVerdict] = []
    for name in DATASETS:
        if name not in names:
            continue
        collected: dict[str, Scored] = {}
        results.append(adjudicate(BY_NAME[name], args.resamples, collected))
        if collected:
            legs[name] = collected
    report(results)

    out_path = args.out or OUT_DIR / f"prompt-redo-seed{SEED}.json"
    if not legs:
        # 다른 판정기들과 같은 규칙: Δ를 하나도 공표하지 않는 판정 파일은 인수 없는
        # ``python -m bench.recheck``이 읽는 디렉터리에 놓여서 그걸 실패시키고, 그건 아직 재지 않은
        # 측정이 아니라 고장난 벤치마크로 읽힌다.
        print(
            f"판정할 다리가 없습니다 — {out_path.as_posix()}를 쓰지 않았습니다. "
            "bench/scripts/run_prompt_redo.sh를 먼저 돌리세요",
            file=sys.stderr,
        )
        return 1
    # 한 번만 만든다: 노출 검사와 기준을 돌리는 일이고, 판정 사본이 둘이면 찍힌 수와 쓰인 수가
    # 어긋나게 된다.
    body = payload(results, args.resamples)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    target = bundle_path(out_path)
    write_bundle(target, legs, {r.dataset: r.metric for r in results})
    print(f"예측 번들 → {target.as_posix()} ({target.stat().st_size / 1024:.0f} KB)")

    print()
    print("기록된 표와 나란히 (기준은 이진 4개만, 나머지 둘은 따로 관찰):")
    print(
        f"   {'데이터셋':<16}{'학습':>9}{'바':>18} {'주 비교 (기록 → 지금)':<30} "
        f"{'llm − no_llm (기록 → 지금)':<28}"
    )
    for row in body["movement"]:
        fits = f"{row['fits_recorded']}→{row['fits_now']}"
        mark = "" if row["judged"] else " (따로)"
        primary = _movement_text(row["primary_recorded"], row["primary_now"], row["primary_ci"])
        rules = _movement_text(row["vs_rules_recorded"], row["vs_rules_now"], row["vs_rules_ci"])
        print(
            f"   {row['dataset'] + mark:<16}{fits:>9}{_bar_text(row):>18} "
            f"{primary:<30} {rules:<28}"
        )

    # 바가 학습 횟수의 위쪽에 있으므로, 움직인 바는 그것이 낸 숫자를 읽기 전에 말해야 하고 그
    # 아래 각주에 적을 것이 아니다.
    moved = [row["dataset"] for row in body["movement"] if row["bar_moved"]]
    if moved:
        print()
        print(
            f"주의 — {', '.join(moved)}에서 auto 바가 12fa74b 이후로 올라갔습니다. "
            "이 카드의 code_shift·prompt_shift는 코드만이 아니라 목표까지 넘는 쌍이고, "
            "판정 파일이 그 쌍에 spans_goal_change를 답니다:"
        )
        for entry in body["datasets"]:
            for pair in entry["pairs"]:
                if pair.get("spans_goal_change"):
                    print(
                        f"   {entry['dataset']:<16}{pair['kind']:<13}"
                        f"{pair['a']} {pair['goal_threshold_a']:.4f} 대 "
                        f"{pair['b']} {pair['goal_threshold_b']:.4f}"
                    )

    exposure = {name: found for name, found in body["random_code_exposure"].items() if found}
    print()
    if exposure:
        print(
            "주의 — 아래 데이터셋의 random 뽑기가 0ef38f8이 손댄 키를 들고 있습니다. "
            "그 쌍은 코드가 섞인 비교로 인용해야 합니다:"
        )
        for name, found in exposure.items():
            print(f"   {name}: {found}")
    else:
        print(
            "random 뽑기 전부가 0ef38f8이 실행기에서 손댄 다섯 키를 하나도 들고 있지 않습니다 — "
            "그 팔을 다시 뽑지 않고 재사용하는 근거입니다. 12fa74b의 바 변경은 이 팔에 닿지 "
            "않습니다 (random_search.py는 목표를 읽지 않습니다)"
        )

    print()
    print(f"기준 (llm2): {body['criterion']['status']}")
    print(f"기준 (no_llm2): {body['criterion_no_llm']['status']}")
    print(f"핵심 질문 (llm2 − no_llm2): {body['core_question']['status']}")
    print(f"판정 → {out_path.as_posix()}")
    refused = [r.dataset for r in results if r.verdict != "ok"]
    if refused:
        print(f"거부된 데이터셋: {', '.join(refused)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
