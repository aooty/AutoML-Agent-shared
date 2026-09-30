# AutoML 실행 최종 보고서 — `house_sales` (target: `price`, metric: `r2`)

## 요약

목표는 달성되었습니다. 임계값 r2 = 0.7718(ridge baseline 0.6957 대비 chance 방향으로 25% margin 적용)을 첫 번째 시도에서 넘었고, 최고 검증 점수는 `hist_gbdt`의 r2 = 0.8905 (95% CI 0.8724~0.9062)입니다. 총 1회 시도(허용 1회) 만에 종료되어, stop reason은 `goal_reached`입니다. 한 번도 학습·선택에 쓰이지 않은 최종 테스트 20%에서의 점수는 r2 = 0.8831 (95% CI 0.8510~0.9090)로, 이것이 이 실행이 실제로 입증한 수치입니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `learning_rate=0.05`, `max_iter=800`, `max_leaf_nodes=64`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=30`, `random_state=42` | `status=ok` — r2 = 0.8905 (CI 0.8724~0.9062), mae = 63821.40, rmse = 115644.37, `train_r2` = 0.9608, `train_val_gap` = 0.0703, 학습 3.363초, early stopping이 `stopped_at_iter=240` (max 800)에서 정지 | 없음 (`critic: null`) — 첫 시도가 목표를 통과해 진단이 실행되지 않음 |

## 최고 성능 구성

iteration 1, model `hist_gbdt`. 아래는 히스토리에 기록된 **적용된** 값입니다.

- `hyperparams`:
  - `learning_rate`: 0.05
  - `max_iter`: 800
  - `max_leaf_nodes`: 64
  - `min_samples_leaf`: 20
  - `l2_regularization`: 1.0
  - `early_stopping`: true
  - `validation_fraction`: 0.1
  - `n_iter_no_change`: 30
  - `random_state`: 42
  - `dropped_hyperparams`: 없음 (executor가 거부한 키 없음)
- `preprocessing` (적용값): `impute: median`, `scale: false`, `missing_indicator: false`, `missing_count: false`
- 입력: 카드의 numeric 21개 열 전체, 파생·인코딩·삭제 없음
- 내부 early stopping: `fit_rows=11670`, `held_out_rows=1297`, `stopped_at_iter=240`
- 프로토콜: seed 42, 비층화 3분할 (train 60% / validation 20% / test 20%) — baseline ridge와 동일한 분할

| 지표 | validation (선택에 사용) | 최종 테스트 (한 번만 채점) |
|---|---|---|
| r2 | **0.8905** (CI 0.8724~0.9062) | **0.8831** (CI 0.8510~0.9090) |
| mae | 63821.40 | — |
| rmse | 115644.37 | — |
| train_r2 | 0.9608 | — |

검증 0.8905와 테스트 0.8831의 차이 +0.0074는 **선택 편향의 크기**입니다 — 이 실행은 검증 숫자를 보고 최고 시도를 골랐기 때문입니다. 다만 검증 점수가 테스트 CI(0.8510~0.9090) 안에 들어 있어, 이 행들만으로는 그 차이를 0과 구분할 수 없습니다. 어느 쪽으로 읽어도 테스트 점수와 그 CI 하단(0.8510)이 모두 임계값 0.7718을 상회하며, ridge baseline CI 상단(0.7130)과도 겹치지 않습니다.

## 원인 분석

이 실행에서 critic은 **한 번도 실행되지 않았습니다** (시도 1회, 진단 0회). 따라서 여러 verdict를 가로지르는 패턴은 존재하지 않으며, 여기서 보고할 "진단"도 없습니다. 0.8905라는 점수는 **첫 계획 하나가 낸 결과**이고, 진단·재계획 경로는 이 결과에 전혀 기여하지 않았습니다. 즉 이 실행은 diagnose-and-replan 루프가 도움이 된다는 증거도, 해가 된다는 증거도 제공하지 않습니다.

비교 가능한 차이로 말할 수 있는 것은 baseline과의 대비 하나뿐입니다. ridge baseline은 r2 = 0.6957 (CI 0.6735~0.7130), 이번 `hist_gbdt`는 검증 r2 = 0.8905 (CI 0.8724~0.9062)로 두 구간이 겹치지 않습니다 — 이 데이터가 분리해 주는 차이입니다. 카드상 `lat`(target_corr strong), `grade`, `sqft_living` 등 강한 비선형·지리 구조가 있는 반면 `long`·`zipcode`의 선형 상관은 약하거나 없다는 점과, 선형 모델 대비 트리 앙상블의 이 격차는 정합적입니다. 다만 이것은 baseline 카드와 이번 시도 사이의 대비이지, critic이 내린 진단이 아닙니다.

시도 자체의 `train_val_gap` = 0.0703 (train_r2 0.9608 vs validation 0.8905)은 기록된 사실이지만, 시도가 하나뿐이므로 이를 "성능을 제한한 원인"으로 단정할 근거는 이 실행에 없습니다. `unsupported_claims`는 비어 있고, `dropped_hyperparams`도 비어 있어 계획된 설정은 전부 그대로 적용되었습니다.

## 다음 단계 제안

1. **정규화 강도만 바꾼 소수의 재시도로 `train_val_gap` = 0.0703의 실체를 확인** — `min_samples_leaf`를 20 → 40/80, `max_leaf_nodes`를 64 → 31, `l2_regularization`을 1.0 → 5.0 정도로 각각 하나씩 움직여 봅니다. 단, 검증 CI 폭이 약 ±0.017(테스트는 ±0.03)이므로 **0.01 미만의 변화는 개선으로 보고하지 말아야** 합니다. early stopping이 800 중 240에서 멈춘 만큼 `max_iter`를 더 늘리는 것은 효과가 없을 가능성이 높습니다.
2. **결측 관련 레버(`missing_indicator`, `missing_count`)에는 iteration을 쓰지 말 것** — 카드의 `missing.overall_rate`는 0.0이고 결측 열은 0개입니다. 추가되는 열은 상수가 되어 아무 정보도 싣지 않습니다. 같은 이유로 `impute` 전략(현재 `median`) 변경도 이 데이터에서는 무의미합니다.
3. **`zipcode`의 표현을 executor 바깥, 즉 데이터 카드 단계에서 바꿔 볼 것** — 현재 `zipcode`는 thousands+ 크기의 정수 numeric 열(target_corr weak)로 들어가고 있습니다. executor는 재인코딩·파생·삭제를 하지 않으므로, 범주형 one-hot(카드의 `max_cardinality=50` 제한 내에서 병합 필요) 또는 우편번호 단위 지역 지표를 **카드/CSV 생성 시점에** 만들어 넣는 실험이 필요합니다. 이는 트리가 lat/long 분할로 근사하고 있는 지역 효과를 더 직접적으로 주는 유일한 경로입니다.
4. **비교 대상 모델 계열을 1~2개 더 확보** — `random_forest`나 `xgboost`(후자는 `early_stopping_rounds`만 지정, `eval_set`/`callbacks`는 전달 불가)를 같은 분할·같은 seed에서 한 번씩 돌려, 0.88 수준이 `hist_gbdt` 고유의 성과인지 트리 계열 공통의 상한인지 확인합니다. 앙상블 결합(stacking/blending/seed averaging)은 executor가 지원하지 않으므로 계획에 넣지 말고, 필요하면 별도 파이프라인에서 다뤄야 합니다.

> 참고: `Data caveats`에 기록된 주의사항은 없으므로 위 제안 중 특정 열·값·분할의 신뢰성 때문에 배제되는 항목은 없습니다. 다만 3번은 executor 능력 밖(파생·인코딩 불가)이라는 점에서 "카드를 다시 만드는 작업"이며, 계획서에 적어도 실행되지 않는다는 점을 분명히 해 둡니다.