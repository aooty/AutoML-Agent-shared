"""세 팔을 비교하는 여섯 데이터셋.

공개되어 있고 못박혀 있다: 모든 행이 ``fetch_openml(data_id=...)``에서 오므로 누구든 이 벤치마크가
읽는 CSV를 다시 만들어 ``RESULTS.md``의 숫자를 검산할 수 있다. 이 폴더의 존재 이유가 그것이다 —
``FINDINGS-mimic.md``의 측정은 사용 협약이 걸린 파일에서 나왔고, 이 기계 밖의 누구도 그중 하나도
검증할 수 없다.

못박는 것은 ``data_id`` 하나다. OpenML의 ``version`` 인자는 그 옆에서 군더더기다: id는 한 데이터셋의
업로드된 한 버전을 지목하고, 버전까지 넘기면 첫 번째와 어긋날 수 있는 두 번째 것이 생긴다.

이진 4개 + 다중분류 1개 + 회귀 1개이고, **판정받는 것은 이진 4개뿐이다.** 나머지 둘은 **집계에 섞지
않고 따로 관찰한다**: ``balanced_accuracy_cut_headroom``, ``class_weight``, 랭킹 상한 진단,
``roc_auc``/``pr_auc``가 모두 이진 전용 코드 경로라서, 회귀나 다중분류 실행은 Critic의 다른 절반을
밟고 그 결과는 나머지 넷과 같은 자에 올릴 수 없다.

**그래서 :data:`JUDGED`는 뺄셈이 아니라 목록이다.** ``RESULTS.md``, ``SPG.md``, ``REPEATS.md``,
``ROWBUDGET.md``의 사전 등록된 기준은 전부 "4개 중 n개"이고, 판정기들은 예전에 회귀를 제외해서 그 넷을
찾았다 — task 타입이 정확히 둘일 때만 옳다. 다중분류를 더하는 순간 커밋된 기준이 모두 조용히 "5개 중
n개"가 되어 문서 네 개에서 바가 사후에 움직였을 것이다. 그러니 판정 집합은 여기서 한 번, 긍정형으로
적는다: 새 task 타입은 :data:`ALL`에 들어가 보고되고, 자기가 생기기 전에 고정된 분모에는 못 들어간다.

**이진 4개 모두에서 양성 클래스가 소수 클래스다.** 맞춰 놓은 것이 아니라
:func:`automl_agent.dataset.targets.encode_target`이 정렬 순서로 라벨을 코딩하고,
``<=50K`` < ``>50K``, ``1`` < ``2``, ``0`` < ``1`` 모두 드문 라벨을 코드 1에 놓기 때문이다. 그래서
``recall``, ``balanced_accuracy_cut_headroom``, ``class_weight`` 축이 ``RESULTS.md``의 모든 행에서 같은
것을 뜻한다 — 다수 클래스가 뒤로 정렬되는 데이터셋에서는 "recall이 떨어졌다"가 다른 클래스에 대한
문장이 된다.

**``note``의 수는 OpenML frame이 아니라 CSV에서 잰 것이다.** CSV를 왕복하면 dtype을 다시 추론하고,
세 팔이 읽는 것은 CSV다. ``python -m bench.fetch``가 그 수를 출력한다. note가 출력과 어긋나면
틀린 쪽은 note다.

여섯 개가 어떻게 골라졌고 무엇이 왜 떨어졌는지는 ``bench/README.md``의 "데이터셋 선정과 기각"에 있다
— 기각 이유마다 행 수·결측률·클래스 비율이 붙어 있어서 측정 문서 쪽이 제자리다.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# resolve하지 않은 상대 경로이고, 그게 요점이다: 이 경로들은 데이터셋 카드(비공개 ``data.path``
# 블록)에 써 넣어지고 카드는 커밋된다. 거기 절대 경로가 들어가면 한 기계의 홈 디렉터리를 저장소에
# 공개하는 것이고 아무도 돌릴 수 없는 카드가 된다 — ``automl_agent.scripts.profile``은 받은
# 문자열을 resolve하지 않고 그대로 기록한다.
#
# 무엇에 대한 상대냐면 저장소 루트다. 어차피 이 모듈들이 거기서 돌아야 한다(``__init__`` 참고).
# 다른 곳에서 돌리면 첫 읽기에서 요란하게 어긋난다.
BENCH_DIR = Path("bench")
# ``*.csv``는 저장소 전체에서 무시되므로(.gitignore 참고) 받아 온 행은 로컬에 남고 스크립트와
# 카드와 결과만 커밋된다.
DATA_DIR = BENCH_DIR / "data"
# 카드는 커밋한다. MIMIC 작업의 규칙과 정반대다 — 거기서는 사용 협약이 걸린 행으로 카드를
# 만들었으니 ``local/``이 삼켰다. 여기서는 행이 공개이고 카드는 모든 팔의 prompt가 렌더된 집계다.
# 커밋해 두는 것이 세 팔 모두 데이터에 대해 같은 설명을 받았음을 독자가 확인하는 방법이다.
CARD_DIR = BENCH_DIR / "cards"

# 저렴한 팔을 어느 시드에서 반복하는가, 그리고 판정이 그 시드를 쓰지 않는 이유.
#
# ``JUDGED_SEED``는 세 팔이 공유하는 유일한 시드이므로 짝 비교가 존재하는 유일한 시드다 — LLM
# 팔은 한 번만 돈다. 데이터셋 다섯에 실행당 약 $5가 예산이기 때문이다. 나머지 두 시드는
# ``--no-llm``과 random search만 돌고, 목적은 주장되는 차이 옆에서 **시드 레버가 얼마나 큰지**를
# 말하는 것이다. A2가 스레드 수에 쓴 것과 같은 계기다: 환경 자체의 폭보다 작은 효과는 점추정이
# 무슨 말을 하든 발견이 아니다.
#
# 이렇게 못박아 두면 갈림길의 정원도 막힌다. 성공 기준은 ``JUDGED_SEED`` 하나에서만 평가되므로,
# 나중에 시드를 더해도 잡음을 설명할 수는 있지만 답이 예쁘게 나오는 시드를 캐낼 수는 없다.
#
# 다섯이 아니라 셋인 이유: 데이터셋 5 × 저렴한 팔 2 × 시드 3 × fit 5면 이미 150 fit이고, 거기서
# 읽어 내는 것은 표준편차가 아니라 세 점의 폭이다.
JUDGED_SEED = 42
NOISE_SEEDS: tuple[int, ...] = (43, 44)
CHEAP_SEEDS: tuple[int, ...] = (JUDGED_SEED, *NOISE_SEEDS)


@dataclass(frozen=True)
class Dataset:
    """벤치마크의 한 행. ``name``은 CSV 파일명의 줄기이자 thread-id의 접두사다."""

    name: str
    data_id: int
    target: str
    task: str
    metric: str
    note: str

    @property
    def csv_path(self) -> Path:
        return DATA_DIR / f"{self.name}.csv"

    @property
    def csv_arg(self) -> str:
        """같은 경로를 ``--data`` 인자 모양으로, 슬래시로.

        Windows에서 ``str(Path(...))``는 ``bench\\data\\adult.csv``를 내고, 그 문자열이 커밋되는
        카드의 ``data.path``에 그대로 들어간다. Linux의 독자는 멀쩡해 보이는 카드에서
        file-not-found를 받는다. 슬래시는 어느 플랫폼에서든 ``pathlib``이 제대로 읽으므로 이쪽이
        옮겨 다닐 수 있는 모양이다.
        """
        return self.csv_path.as_posix()

    def card_path(self, seed: int) -> Path:
        """*이 시드에서의* 이 데이터셋 카드. 시드가 이름에 들어가는 것은 사소한 일이 아니다.

        카드는 자기 기준선이 측정된 split 프로토콜을 기록하고,
        :func:`automl_agent.nodes.profiling.assert_card_matches_protocol`은 ``--seed``가 어긋나는
        실행을 거부한다 — 옳은 거부다. 그 기준선에서 유도된 목표 기준값이 시도들이 채점되는 행과
        다른 행에서 측정된 것이 되기 때문이다. 그래서 시드마다 카드 하나, 그리고 다른 시드의 것과
        혼동할 수 없는 파일명.
        """
        return CARD_DIR / f"{self.name}-seed{seed}.json"


# ``metric``은 각 팔이 판정받는 목표 지표이고, 플래그에 맡기지 않고 여기서 못박는다: 세 팔은
# 같은 수로 채점되어야 하고, ``balanced_accuracy``는 이 저장소의 이진 진단이 그것을 중심으로
# 세워진 지표다(``balanced_accuracy_cut_headroom``, 랭킹 상한, ``class_weight``). 회귀가 ``r2``를
# 받는 것도 같은 이유다 — 카드의 기준선이 보고하는 지표라서 ``--goal-mode auto``가 바를 유도할
# 거리가 있다.
BINARY: tuple[Dataset, ...] = (
    Dataset(
        name="adult",
        data_id=1590,
        target="class",
        task="binary",
        metric="balanced_accuracy",
        note="48,842행 · 수치 6 + 범주형 8 · 결측 0.95%(3열이 1% 초과) · 소수 클래스 23.9%",
    ),
    Dataset(
        name="bank-marketing",
        data_id=1461,
        target="Class",
        task="binary",
        metric="balanced_accuracy",
        note="45,211행 · 수치 7 + 범주형 9 · 결측 없음 · 소수 클래스 11.7%(MIMIC과 같은 불균형) "
        "· 열 이름이 V1..V16으로 익명화됨",
    ),
    Dataset(
        name="speeddating",
        data_id=40536,
        target="match",
        task="binary",
        metric="balanced_accuracy",
        note="8,378행 · 수치 61 + 범주형 59 · 결측 1.83%(35열이 1% 초과) · 소수 클래스 16.5% "
        "· field(259개)는 카디널리티 상한으로 버려짐",
    ),
    Dataset(
        name="spambase",
        data_id=44,
        target="class",
        task="binary",
        metric="balanced_accuracy",
        note="4,601행 · 수치 57 · 결측 없음 · 소수 클래스 39.4% · 가장 좁은 슬라이스",
    ),
)

MULTICLASS: tuple[Dataset, ...] = (
    Dataset(
        name="jungle-chess",
        data_id=41027,
        target="class",
        task="multiclass",
        metric="balanced_accuracy",
        note="44,819행 · 수치 6(범주형 0) · 결측 없음 · 3개 클래스 51.5/38.9/9.7% · "
        "집계에 섞지 않고 따로 관찰. roc_auc·pr_auc는 이 task에서 계산되지 않고"
        "(binary_only), f1·precision·recall은 macro 평균으로 나옵니다 — "
        "balanced_accuracy는 이진에서와 같은 정의(클래스별 recall의 평균)라서 "
        "이진 4개와 같은 열에 적을 수 있는 유일한 수입니다",
    ),
)

REGRESSION: tuple[Dataset, ...] = (
    Dataset(
        name="house_sales",
        data_id=42731,
        target="price",
        task="regression",
        metric="r2",
        note="21,613행 · 수치 21(범주형 0) · 결측 없음 · 회귀 · 집계에 섞지 않고 따로 관찰. "
        "id는 없고 날짜는 date_year/month/day로 이미 분해돼 있으며, zipcode(70개)는 CSV에서 "
        "정수로 읽혀 수치로 들어감 — 세 팔 모두에 똑같이 적용되는 조건",
    ),
)

ALL: tuple[Dataset, ...] = BINARY + MULTICLASS + REGRESSION

BY_NAME: dict[str, Dataset] = {item.name: item for item in ALL}

# 사전 등록된 기준이 세는 대상 — "이진 4개 중 3개" 같은 것들. 긍정형으로 적고,
# :data:`BINARY`에서 유도하지 않고 값이 같게 둔다. 두 사실이 같은 사실이 아니기 때문이다:
# ``BINARY``는 "이 중 무엇이 두 클래스 정답인가"이고 ``JUDGED``는 "실행 전에 커밋된 기준이 이 중
# 무엇에 대해 쓰였는가"다. 나중에 새 질문을 위해 이진 데이터셋을 더할 때, 자기 없이 고정된 바에
# 끼지 않을 수 있고, 그 결정은 tuple이 자라서 분모가 움직이는 대신 *여기서* diff에 보이는 편집이
# 된다.
#
# 판정기들은 전에 이걸 ``task != "regression"``으로 적었다. task 타입이 둘일 때는 옳고 셋이 된
# 순간 틀렸다 — 모듈 docstring 참고.
JUDGED: tuple[Dataset, ...] = BINARY
JUDGED_NAMES: frozenset[str] = frozenset(item.name for item in JUDGED)
