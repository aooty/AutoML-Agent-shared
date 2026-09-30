# `bench/` 안내 — 무엇이 어디 있고, 어느 판정이 아직 유효한가

이 폴더는 **이 저장소의 주장을 측정한 코드와 실행 기록**입니다. 파이썬 모듈 19개(공용 7 + 측정별
판정 11 + `__init__.py`), 실행 스크립트 10개(`scripts/`), 실행 아카이브 151개(`runs/`)가 들어
있습니다.

**측정 문서 12개는 이 폴더가 아니라 [`docs/`](../docs/)에 있습니다.** 설계 문서와 같은 자리에
모아 두었습니다 — 이 폴더에는 돌아가는 것만 남깁니다.

**이 파일은 지도일 뿐이고 판정을 내지 않습니다.** 아래 표의 숫자는 전부 각 문서에서 옮긴
것이고, 어긋나면 문서가 옳습니다. 문서들의 본문은 사전 등록된 것이라 고치지 않으므로, 이 지도가
따라가야 하는 쪽입니다.

## 처음 읽는다면

| 순서 | 읽을 것 | 왜 |
| --- | --- | --- |
| 1 | [RESULTS.md](../docs/RESULTS.md) | 이 저장소의 핵심 주장을 처음 측정한 문서입니다. 세 팔(`llm` / `--no-llm` / random search)을 같은 예산·같은 분할로 비교합니다 |
| 2 | [REPEATS.md](../docs/REPEATS.md) | **자를 재는 문서입니다.** 같은 설정을 다시 돌리면 점수가 얼마나 흔들리는지. 이 수를 모르면 위 문서의 Δ를 읽을 수 없습니다 |
| 3 | [HARD-BAR.md](../docs/HARD-BAR.md) → [BAR-NOISE.md](../docs/BAR-NOISE.md) | 가장 어려운 조건에서 다시 물은 것과, 그 판정이 쓴 자를 그 조건에서 직접 잰 것. **두 번째가 첫 번째의 결론 하나를 뒤집습니다** |

나머지는 특정 질문에 대한 답이라 필요할 때 찾아 읽으면 됩니다.

## 이 폴더의 관례

측정마다 **문서 + 판정 모듈 + 실행 스크립트 + 테스트를 실행 전에 정확히 한 커밋으로** 올립니다.
결과는 문서 본문을 고치지 않고 **날짜 붙은 `# 결과` 절**로 보탭니다. git 히스토리가 순서를
증명하므로, 결과를 보고 기준을 정한 것이 아님을 제3자가 확인할 수 있습니다.

대조군은 **CLI 플래그가 아니라 git worktree**로 만듭니다 ([ROWBUDGET.md:50](../docs/ROWBUDGET.md)이 세운
관례). 실험용 플래그가 제품에 남지 않고, 남는 것은 스크립트와 판정뿐입니다.

판정 상태는 두 가지입니다 — `ok`와 `refused`. 지문이 어긋나거나 팔이 빠지면 추측하지 않고
거부합니다.

## 측정 12개

| 문서 | 물은 것 | 판정 | 판정 파일 |
| --- | --- | --- | --- |
| [RESULTS.md](../docs/RESULTS.md) | `llm` 팔이 random search를 이기는가 | **기준 충족** — 시드42 이진 4개 중 3개에서 이김 (spambase 구분 안 됨) | `seed4{2,3,4}.json` · `jungle-chess-seed42.json` |
| [REPEATS.md](../docs/REPEATS.md) | 같은 설정 반복의 산포가 CI 반폭의 절반을 넘는가 | **R1 미충족** — span 중앙값 0.0051, 반폭 0.0122, 비 **0.42**. 그러나 **R2 확인** — 같은 설정 반복끼리의 짝지은 Δ가 **12쌍 중 4쌍에서 0을 벗어났습니다** | `repeats-seed42.json` |
| [REPLAN.md](../docs/REPLAN.md) | 진단·재계획이 예산을 점수로 바꾸는가 | **구분되지 않음** — 이김 1 / 걸침 2 / 짐 0 (과반 2 필요) | `replan-seed42.json` |
| [ROWBUDGET.md](../docs/ROWBUDGET.md) | row budget 절이 계획을 바꾸는가 | **P1 충족** — 이진 4개 중 3개에서 계획이 달라짐. P2 예고=실측 일치 | `rowbudget-seed42.json` |
| [SPG.md](../docs/SPG.md) | `--search-past-goal`이 점수를 올리는가 | **구분되지 않음** — 이김 0 / 걸침 3 / 짐 1. 결론: 점수가 아니라 **정보를 사는 거래** | `spg-seed42.json` |
| [PROMPT-REDO.md](../docs/PROMPT-REDO.md) | 헤드라인 표가 지금 프롬프트에서도 나오는가 | **기준 충족** — 이긴 데이터셋까지 기록과 같음 | `prompt-redo-seed42.json` |
| [REPLAN-SEEDS.md](../docs/REPLAN-SEEDS.md) | 재계획 이득이 시드와 새 프롬프트를 견디는가 | **기준 충족** — 세 시드 다, 최소 Δ +0.0507 | `replan-seeds-speeddating.json` |
| [REGISTRY-GAP.md](../docs/REGISTRY-GAP.md) | jungle-chess에서 `llm` 팔이 왜 졌는가 | **수리** — 고장난 것은 계획 능력이 아니라 **계획이 읽은 목록**이었습니다 | (문서만) |
| [WEIGHT-LEVER.md](../docs/WEIGHT-LEVER.md) | speeddating의 이김이 카드에서 읽은 한 수였는가 | **부분 지지** — 회수 충족, 남은 격차 깨짐, 차등 충족. API 0원 | `weight-lever.json` |
| [WEIGHT-START.md](../docs/WEIGHT-START.md) | 사다리 첫 단을 카드 비율로 옮긴 코드가 루프에서도 값을 내는가 | **판정 A: 처방이 산다** (게이트 셋 전부 통과). **판정 B: 구분되지 않음 — 이김 1개가 0개가 됩니다.** API 0원 | `weight-start.json` |
| [HARD-BAR.md](../docs/HARD-BAR.md) | 루프가 실제로 돌 때 LLM이 규칙 폴백을 이기는가 | **구분되지 않음** — 이김 1 / 그 밖 1 / 짐 2 (3 필요) | `hard-bar.json` |
| [BAR-NOISE.md](../docs/BAR-NOISE.md) | 위 판정이 쓴 자를 그 조건에서 직접 재면 | **N1 충족** (0.59배) · **N2 갈림** · N3 확인 · N4 관찰 | `bar-noise.json` |

문서 없이 모듈만 있는 것이 하나 있습니다 — `class_weight_key.py`. 아래 정정 사슬의 세 번째
줄이 그것입니다.

**표의 Δ를 인용할 때 함께 읽어야 하는 문장이 `REPEATS.md`에 있습니다.** R2가 확인됐으므로,
짝지은 Δ 하나가 0을 벗어나는 것은 **처치의 증거로는 부족합니다** — 같은 설정 반복이 그것을 낼 수
있다는 것이 실제로 관측됐습니다(`REPEATS.md:239-241`). 특히 루프가 두 번 이상 돈 실행에서
그렇습니다.

## 어느 판정이 나중에 수정됐는가

**이 절이 이 지도의 핵심입니다.** 문서 본문은 고치지 않는 관례이므로, 뒤집힌 결론은 덧붙임 절
안에 있고 밖에서는 보이지 않습니다.

| 처음 적은 것 | 나중에 | 어떻게 됐나 |
| --- | --- | --- |
| `REPEATS.md`의 잡음 바닥 **0.0122** | `HARD-BAR.md`가 다른 조건에 **수입** → `BAR-NOISE.md`가 그 조건에서 직접 잼 | 실제로는 **0.0062**. 큰 쪽으로 틀렸지만, 각 셀의 반폭 대비 비가 **0.59**여서 판정이 느슨해진 것은 아닙니다 |
| `HARD-BAR.md`의 **짐 두 칸** (adult 시드43, spambase 시드44) | `BAR-NOISE.md`가 각각 3회 반복 | **adult는 재현되지 않았습니다** (기록된 실행이 세 반복 범위 밖 아래). **spambase는 단단해졌습니다** — 이 저장소에 불리한 쪽 |
| `REPLAN-SEEDS.md`의 "+0.0480은 `class_weight` **한 키**의 효과" | `class_weight_key.py`가 네 다리로 분해 | **두 레버였습니다** (`class_weight` + `preprocessing.missing_count`). 한 줄에 두 레버는 이 저장소가 해석을 거부하는 모양이라, 덧붙임이 그렇게 다시 적었습니다 |
| `RESULTS.md`의 jungle-chess **짐** (−0.0225) | `REGISTRY-GAP.md` | 계획이 "`class_weight`를 쓸 수 없다"고 적었는데 **사실이 아니었습니다.** 실행기의 한계가 아니라 능력 목록의 누락 |
| `HARD-BAR.md`의 **이김 한 칸** (speeddating) | `WEIGHT-START.md`가 그 처방을 코드에 넣고 다시 돌림 (`d37217e`) | **이김 1개가 0개가 됐습니다.** 규칙 팔(무료)이 좋아져서 유료 팔이 이길 카드가 하나 줄었습니다 — 이 저장소의 주장에 **가장 불리한 결과**이고, 저장소가 자기 코드를 고쳐서 스스로 만든 것입니다 |
| `WEIGHT-LEVER.md`의 남은 격차 | 같은 코드 수정 | 결과가 바뀌었으므로 그 뒤 실험은 **옛 코드를 worktree로 되살려** 돌립니다. `BAR-NOISE.md`의 14실행이 그렇게 돌았습니다 |
| `SPG.md`의 표 | 덧붙임 (2026-08-31) | 제3자가 다시 계산할 수 있도록 예측 번들을 함께 커밋 |

읽는 순서를 하나로 줄이면 이렇습니다: **어떤 문서의 Δ를 인용하기 전에 그 문서 맨 끝의 덧붙임
절을 먼저 보십시오.**

아직 아무도 재지 않은 것도 여기 적습니다. `BAR-NOISE.md`가 네 셀 전부에서 반복끼리 학습 횟수가
갈리는 것을 발견했습니다(3/5/3 · 4/5/5 · 3/3/5 · 4/5/4). **"같은 명령이 왜 다른 횟수로 도는가"는
측정되지 않았습니다.**

## 데이터셋 선정과 기각

여섯 개가 어떻게 골라졌고 무엇이 왜 떨어졌는지. 수는 전부 CSV에서 잰 것이고
`python -m bench.fetch`가 출력하는 값과 같아야 합니다 — 어긋나면 아래 표가 틀린 것입니다.
선정 자체는 `datasets.py`가 코드로 못박아 두고, 이유는 여기 있습니다.

### 판정 분모에 든 이진 4개

| 데이터셋 | 무엇을 채우려고 골랐나 |
| --- | --- |
| `adult` | 가장 큰 혼합형 표이고, 결측이 실재하지만 아주 작은(0.95%) 유일한 것. 기준 사례입니다 — 진단할 것이 가장 적은 곳에서 팔들이 구분되지 않으면 어디서도 구분되지 않습니다 |
| `bank-marketing` | 소수 클래스 11.7%, 공개 행으로 재현한 MIMIC 코호트의 불균형(11.0%). `class_weight` / `balanced_accuracy_cut_headroom` 축이 가장 넓고, 이 저장소가 진단 경로에 대해 증거를 가장 많이 가진 데이터셋입니다. 열 이름이 `V1`..`V16`으로 익명화돼 있고 라벨이 `1`/`2`인 것은 우연이지만 남겨 둘 만합니다 — LLM 팔이 열 *이름*에서 추론할 거리를 못 받는 유일한 곳이라, 여기서 이기면 숫자만으로 이긴 것입니다 |
| `SpeedDating` | 결측 사례. 120열 중 35열이 1% 넘게 비어 있고(한 열은 78.5%), 그래서 `impute` / `missing_indicator` / `missing_count`가 실제로 동원됩니다. 검토한 다른 이진 후보는 결측이 사실상 없어서, 이 저장소가 가장 많이 측정한 레버 셋이 세 팔 모두에서 안 밟힌 채 남았을 것입니다. one-hot 카디널리티 상한을 넘는 열(`field`, 259개)도 하나 있어서 버려지는 일이 가정이 아니라 카드에 보입니다 |
| `spambase` | 수치 57열, 범주형 없음, 결측 없음, 행이 가장 적음(4,601 → test 약 920행). 좁은 슬라이스는 의도입니다 — 짝 구간이 가장 넓은 곳이라 행이 차이를 못 가릴 때 판정이 "구분되지 않음"을 인정하는지 시험합니다. 45,000행에서만 도는 벤치마크는 그걸 알 수 없습니다 |

### 이진 4개 판정이 이미 나온 뒤에 더한 다중분류 1개

`jungle_chess_2pcs_raw_endgame_complete` — 44,819행, **세** 클래스 51.5 / 38.9 / 9.7%. 스물여섯이
아니라 셋인 것은 의도입니다: 이진 4개에 가장 가까운 다중분류 모양이라, 바뀌는 것이 코드 경로뿐이고
기하까지 같이 바뀌지 않습니다. 소수 클래스 9.7%는 `bank-marketing`의 11.7%와 같은 범위라
`balanced_accuracy`가 균형 잡힌 정답에서처럼 `accuracy`로 주저앉지 않고 뜻을 가집니다. 행 수는
`adult` 범위라 짝 구간이 충분히 좁고, "구분되지 않음"이 test 1,200행의 부작용이 아니라 발견이 됩니다.

동원하지 **못하는** 것도 그대로 적습니다: 정수 특성 6개, 결측 없음, 범주형 없음, 그리고 계획자가
추론할 거리가 아니라 체스 엔드게임을 설명하는 열 이름. task 타입 탐침이고 두 번째 `SpeedDating`이
아닙니다.

### 기각된 다중분류 후보 — 같은 방식으로 재고 떨어뜨렸습니다

| 후보 | 기각 이유 |
| --- | --- |
| `letter` (id 6) | 20,000행, 26클래스, 각 3.9~4.1%. *너무* 균형 잡혀서 기각: 거기서는 `balanced_accuracy`와 `accuracy`가 거의 겹치므로 다른 다섯이 판정받는 지표가 더 어려운 수이기를 그만두고 비교의 뜻이 조용히 바뀝니다 |
| `kr-vs-k` (id 1481) | 28,056행, 18클래스, 가장 작은 클래스가 27행. `KDDCup09_appetency`와 같은 이유 — 전체 27행인 클래스에 대한 macro 평균은 그 한 클래스의 구간이 지배합니다 |
| `connect-4` (id 40668) | 67,557행, 범주형 42열, 3클래스 65.8 / 24.6 / 9.5%. 전부 범주형인 유일한 후보라 원했지만 행에서 기각: 67k행에 데이터셋당 15 fit은 다른 다섯이 돌아간 예산에 안 맞습니다 |
| `har` (id 1478) | 10,299행, 수치 561열, 균형 잡힌 6클래스. 모양에서 기각 — 10k행에 561특성은 여기 아무것도 차지하지 않은 p/n 구석이고, 그 변화를 task 타입 변화에 섞으면 둘 다 측정되지 않습니다 |

### 기각된 이진 후보

| 후보 | 기각 이유 |
| --- | --- |
| `electricity` (id 151) | 45,312행, 거의 균형 — "행 많고 열 적은" 구석을 채웠을 것입니다. 시계열이라 이 저장소의 무작위 분할에서는 인접한 30분이 양쪽에 걸칩니다. 세 팔을 똑같이 부풀리므로 비교는 여전히 공정하지만, `RESULTS.md`의 모든 행에 같은 유보를 달아야 하고 절대값은 아무 뜻도 없게 됩니다 |
| `KDDCup09_appetency` (id 1111) | 결측 69.8%로 구할 수 있는 가장 강한 결측 사례. 정답에서 기각 — 50,000행에 소수 1.78%면 양성이 890개이고, 그만큼으로 낸 `balanced_accuracy`는 구간이 지배합니다. 230열 중 205열이 1% 넘게 비었고 몇 열은 *전부* 비었습니다 |
| `Diabetes130US` (id 43874) | 101,766행. 데이터셋당 15 fit에는 너무 크고, Fairlearn 변종은 결측이 아예 없어서 이걸 원한 이유가 사라집니다 |
| `APSFailure` (id 41138) | 원했습니다(76k행, 결측 8%, 소수 1.8%). 이 네트워크에서 다운로드가 `IncompleteRead`로 실패합니다 — 두 번, 서로 다른 바이트 수에서 |

## 부품

### 공용 — 여러 측정이 함께 씁니다

| 모듈 | 하는 일 |
| --- | --- |
| `datasets.py` | 여섯 데이터셋과 각각이 왜 여기 있는지. **판정 분모는 그중 이진 4개**(`adult`·`bank-marketing`·`spambase`·`speeddating` = `JUDGED_NAMES`)이고, `jungle-chess`(다중분류)와 `house_sales`(회귀)는 진단 경로가 달라서 **집계에 섞지 않고 따로 관찰**합니다 |
| `fetch.py` | OpenML에서 CSV를 다시 만듭니다. **다른 무엇보다 먼저 한 번** 돌립니다 |
| `cards.py` | (데이터셋, 시드)마다 카드를 만들고, 팔들이 필요한 것이 적혀 있는지 확인합니다 |
| `paired.py` | **짝지은 부트스트랩** — 같은 test 행에 대고 두 팔의 차이와 95% CI를 냅니다. 모든 판정의 바탕 |
| `predictions.py` | 판정을 계산한 배열을 저장해서, 판정을 나중에 다시 계산할 수 있게 합니다 |
| `recheck.py` | 커밋된 판정을 예측 번들에서 재계산하고, **어긋나면 크게 알립니다** |
| `random_search.py` | 세 번째 팔. LLM에게 보여 주는 것과 **같은 메뉴**에서 뽑습니다 |

### 측정별 판정 모듈

`bar_noise.py` · `hard_bar.py` · `prompt_redo.py` · `repeats.py` · `replan.py` ·
`replan_seeds.py` · `rowbudget.py` · `spg.py` · `weight_lever.py` · `weight_start.py` ·
`class_weight_key.py`

이름이 문서와 짝을 이룹니다 (`bar_noise.py` ↔ `BAR-NOISE.md`). 각 모듈은
`python -m bench.<이름>`으로 판정을 다시 낼 수 있고, 재료가 없으면 추측하지 않고 거부합니다.

**문턱 하나는 여기서 검산할 수 없습니다.** `weight_lever.py`의 `REFIT_TOLERANCE = 0.0026`은
[docs/FINDINGS-mimic.md](../docs/FINDINGS-mimic.md)의 A2에서 왔고, 그 문서의 수는 재배포를
허용하지 않는 임상 데이터에서 나왔습니다. 이 폴더의 나머지 전부는 `fetch_openml`로 다시 만들 수
있지만 그 하나는 아닙니다 — `weight-lever.json`이 `refit_tolerance_source`에 출처를 적어 두는
이유입니다.

### 실행 스크립트 — `bench/scripts/`

[run_bar_noise.sh](scripts/run_bar_noise.sh) · [run_hard_bar.sh](scripts/run_hard_bar.sh) ·
[run_prompt_redo.sh](scripts/run_prompt_redo.sh) · [run_repeats.sh](scripts/run_repeats.sh) ·
[run_replan.sh](scripts/run_replan.sh) · [run_replan_seeds.sh](scripts/run_replan_seeds.sh) ·
[run_rowbudget.sh](scripts/run_rowbudget.sh) · [run_spg.sh](scripts/run_spg.sh) ·
[run_weight_lever.sh](scripts/run_weight_lever.sh) ·
[run_weight_start.sh](scripts/run_weight_start.sh)

저장소 루트에서 실행합니다 — 각 스크립트가 `cd "$(dirname "$0")/../.."`로 스스로 올라갑니다.

**이 열 개는 2026-09-12에 `bench/`에서 `bench/scripts/`로 옮겼습니다.** 그 전에 커밋된 문서는
`bench/run_<이름>.sh`로 적고 있고, 본문 무수정 관례상 고치지 않았습니다 — 아래 여덟 곳입니다.

| 어디 | 무엇 |
| --- | --- |
| [HARD-BAR.md:26](../docs/HARD-BAR.md#L26) · [:114](../docs/HARD-BAR.md#L114) | `run_prompt_redo.sh`를 줄 번호까지 걸어 둔 링크 둘. **링크로는 열리지 않습니다** |
| [BAR-NOISE.md:116](../docs/BAR-NOISE.md#L116) · [PROMPT-REDO.md:260](../docs/PROMPT-REDO.md#L260) · [REPEATS.md:144](../docs/REPEATS.md#L144) · [REPLAN-SEEDS.md:200](../docs/REPLAN-SEEDS.md#L200) · [ROWBUDGET.md:128](../docs/ROWBUDGET.md#L128) · [SPG.md:149](../docs/SPG.md#L149) | 산문·명령줄로 적은 `bench/run_*.sh` |

**문서 12개도 같은 날 `bench/`에서 `docs/`로 옮겼습니다.** 문서를 가리키는 경로가 54개 파일에
흩어져 있었고, 그중 판정 파일 11개의 필드 27줄(`"preregistration": "docs/HARD-BAR.md"` 같은
것)까지 함께 고쳤습니다. `git diff`가 그 27줄이 전부 경로 문자열이고 **숫자·기준·분모는 하나도
바뀌지 않았음**을 보입니다 — 사전 등록이 지키는 것은 그 셋이지 파일 위치가 아닙니다.

고치지 못한 것도 적습니다. 동결 문서 본문이 형제 파일을 상대 경로로 걸어 둔 링크 **6개가
죽었습니다** — `RESULTS.md`·`WEIGHT-LEVER.md`가 가리키는 `random_search.py`, `WEIGHT-LEVER.md`의
`replan.py`, `WEIGHT-START.md`의 `runs/paired/weight-start.json`(각각 `bench/` 아래에 그 이름으로
있습니다), 그리고 루트에 있던 연구 기록 둘이 `docs/`로 들어가면서 `REPLAN-SEEDS.md`의
`../FINDINGS-mimic.md`와 `RESULTS.md`의 `../DESIGN-loop-stagnation-v2.md`(둘 다 이제 같은
`docs/` 안에 있으니 `../`만 떼면 열립니다). 그리고 산문으로 `bench/<이름>.md`라 적은 곳이 여섯 남아 있습니다
([HARD-BAR.md:8](../docs/HARD-BAR.md#L8)에 둘 · [:484](../docs/HARD-BAR.md#L484) ·
[RESULTS.md:54](../docs/RESULTS.md#L54)·[:56](../docs/RESULTS.md#L56) ·
[WEIGHT-START.md:38](../docs/WEIGHT-START.md#L38)).

**`.py`는 옮기지 않았습니다.** 동결 문서 일곱이 `python -m bench.recheck` 같은 실행 명령을
18곳에서 쓰고, 그건 제3자가 검산할 때 실제로 타이핑하는 명령입니다. 링크는 깨져도 검색 한 번이면
되지만 명령은 그렇지 않습니다.

## `runs/` 안

| 경로 | 무엇 | 크기 |
| --- | --- | --- |
| `runs/artifacts/` | 실행 아카이브 — 계획·프롬프트·설정·점수 | **151개 실행** |
| `runs/paired/` | 판정 파일과 예측 번들 | JSON 15 + `.npz` 15 |
| `runs/random/` | random 팔의 학습 기록 | 18개 |
| `runs/weightlever/` | `WEIGHT-LEVER.md`의 재학습 | 12개 |

**`runs/artifacts/`에는 쓰지 않습니다.** 기록된 아카이브이고, 각 실행이 자기 `run_config.json`에
경로를 적어 두었습니다.

## 여기 없는 것

- **데이터 CSV** (`bench/data/`) — gitignore입니다. `python -m bench.fetch`로 OpenML에서
  다시 만듭니다 (`datasets.py`의 `data_id`가 고정되어 있습니다). **체크섬은 일부러 쓰지
  않습니다** — CSV 바이트는 그것을 쓴 pandas 버전에 따라 달라지므로, 해시는 행이 같은 사람에게도
  실패합니다. 대신 행·열·결측 비율·클래스 균형을 찍고, 그게 어긋나는 것이 진짜 실패입니다
  (`fetch.py:11-15`)
- **학습된 모델**(`model.joblib`), 예측 배열(`val_predictions.npz`), 로그(`*.log`) — gitignore.
  아카이브에는 JSON 기록만 커밋됩니다
- **판정을 다시 쓰는 도구** — 사전 등록된 기준과 분모는 사후에 바꾸지 않습니다

## 새 측정을 추가한다면

1. 문서를 먼저 쓰고 **판정 기준과 문턱을 숫자로** 박습니다
2. 판정 모듈, 실행 스크립트, 테스트를 함께 씁니다
3. **네 파일을 한 커밋으로, 실행 전에** 올립니다
4. 무료로 확인할 수 있는 절반(`--no-llm`, 재학습)을 먼저 돌립니다 — 파이프라인이 깨져 있으면
   토큰이 아니라 CPU로 알게 됩니다
5. 결과는 본문을 고치지 않고 날짜 붙은 절로 보탭니다
6. 다른 문서의 결론을 수정하게 됐으면 **그 문서에도 덧붙임을 달고, 이 파일의 정정 사슬에
   한 줄 적습니다**

4번이 그냥 조언이 아닙니다. `BAR-NOISE.md`의 무료 절반이 판정 모듈의 결함을 찾아냈고 — 반복이
없으면 규칙 다리를 모으기 전에 거부해서 사전 등록한 확인에 도달할 수 없었습니다 — 유료 12실행
전에 고쳤습니다. `ROWBUDGET.md`는 유료 대조군을 먼저 돌렸다가 **계획 호출 5건을 태웠습니다** —
worktree에 `bench/data/`가 없어서 다섯 실행이 전부 `data_issue`로 죽었습니다(`ROWBUDGET.md:145`).
무료로 한 번 돌렸으면 CPU만 썼을 자리입니다. 그 preflight는 지금 `run_rowbudget.sh`에 있습니다.
