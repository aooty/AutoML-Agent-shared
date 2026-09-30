# spambase · balanced_accuracy 목표 달성 보고서

## 요약

목표는 달성되었습니다. 목표 임계값은 `balanced_accuracy` 0.9294였고, 첫 번째 시도의 `hist_gbdt`가 검증 슬라이스에서 0.9338848294025625 (95% CI 0.9161~0.9490)를 기록해 임계값을 넘었습니다. 루프는 5회 예산 중 1회만 사용하고 `goal_reached`로 종료되었으며, 따라서 비교 대상이 되는 다른 시도는 존재하지 않습니다. 한 번도 사용되지 않은 최종 테스트 20%에서 같은 모델은 `balanced_accuracy` = 0.9481 (95% CI 0.9322~0.9635)로, 이 실행이 실제로 입증한 수치는 이 값입니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.08`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `early_stopping=False` | `status=ok` · `balanced_accuracy` = 0.9339 (CI 0.9161~0.9490), `roc_auc` = 0.9792, `train_time_sec` = 3.765 → 임계값 0.9294 초과, 목표 달성 | 없음 (`critic: null` — 목표 달성으로 진단 단계에 진입하지 않음) |

## 최고 성능 구성

- **iteration**: 1
- **model**: `hist_gbdt`
- **hyperparams** (실제 적용값, `dropped_hyperparams` 없음):
  - `max_iter`: 400
  - `learning_rate`: 0.08
  - `max_leaf_nodes`: 31
  - `l2_regularization`: 1.0
  - `early_stopping`: false
- **preprocessing** (해당 시도가 기록한 적용값):
  - `impute`: `median`
  - `scale`: false
  - `missing_indicator`: false
  - `missing_count`: false
- **프로토콜**: stratified 3-way split, train 60% / val 20% / test 20%, seed 44 (카드의 baseline과 동일한 분할)

검증 슬라이스 지표:

| 지표 | 값 |
|---|---|
| `balanced_accuracy` | 0.9339 (95% CI 0.9161~0.9490) |
| `accuracy` | 0.9380 |
| `f1` | 0.9207 |
| `precision` / `recall` | 0.9272 / 0.9144 |
| `specificity` | 0.9534 |
| `roc_auc` | 0.9792 |
| `pr_auc` / `average_precision` | 0.9591 |
| `balanced_accuracy_at_best_cut` | 0.9426 |
| `cut_headroom` | 0.008715 |
| `brier` | 0.051284 |
| `calibration_error` | 0.047245 |
| `train_balanced_accuracy` | 0.9992 |
| `train_val_gap` | 0.065357 |

**보류된 테스트 측정값**: 같은 모델을 첫 fit 이전에 떼어두고 어떤 결정에도 쓰이지 않은 20% 행에서 한 번 채점한 결과 `balanced_accuracy` = 0.9481 (95% CI 0.9322~0.9635)입니다. 검증값 0.9339과의 차이 0.0142가 선택 편향의 크기이며, 루프는 검증 숫자를 보고 선택했습니다. 다만 검증값이 테스트 CI(0.9322~0.9635) 안에 들어오므로, 이 행 수로는 그 차이를 0과 구분할 수 없습니다. 참고로 검증 CI의 하한(0.9161)은 임계값 0.9294보다 낮아 검증 기준의 통과 여유(약 0.0045)는 구간 폭보다 훨씬 작고, 임계값 초과를 더 강하게 뒷받침하는 것은 하한이 0.9322인 테스트 측정값입니다.

## 원인 분석

`재계획` 기록에 따르면 critic은 **한 번도 실행되지 않았습니다** (시도 1회, 진단 0회). 따라서 여러 verdict에 걸친 패턴이라는 것이 존재하지 않으며, 이 점수는 **첫 계획 하나가 낸 결과**입니다. 진단·재계획 경로가 도움이 되었다는 증거도, 해가 되었다는 증거도 이 실행에는 없습니다 — 이 실행은 그 루프에 대해 어느 쪽으로도 말해주지 않습니다.

시도가 하나뿐이므로 시도 간 이동으로 이야기를 만들 여지도 없습니다. 비교 가능한 기준선은 카드의 baseline인 `logreg` (median impute + standard scale, 동일 분할·동일 seed)뿐이고, 그 검증 `balanced_accuracy`는 0.9058 (CI 0.8823~0.9239), `roc_auc` 0.9672입니다. 이번 시도의 검증 CI(0.9161~0.9490)와 baseline CI(0.8823~0.9239)는 일부 겹치므로, 검증 슬라이스 하나만으로는 두 값의 차이를 결정적이라고 말할 수 없습니다. 반면 `roc_auc`는 0.9672 → 0.9792로, `balanced_accuracy_at_best_cut`은 baseline의 0.9233 → 0.9426으로 움직였습니다.

그 밖의 metric들은 진단이 아니라 **관측 사실**로만 적어 둡니다(어떤 진단도 내려지지 않았으므로 원인으로 제시하지 않습니다): `cut_headroom`은 0.008715로 작고, `train_val_gap`은 0.065357이며 `train_balanced_accuracy`는 0.9992입니다. `calibration_error`는 0.047245로, 확률값을 확률로 읽을 경우 평균 약 4.7 퍼센트포인트 어긋납니다. 계획의 `unsupported_claims`는 비어 있고 `dropped_hyperparams`도 비어 있어, 계획한 구성이 그대로 실행되었습니다.

## 다음 단계 제안

데이터 카드에 별도로 기록된 caveat는 없으므로, 아래 항목은 모두 이 실행의 실측치에만 근거합니다.

1. **결정 임계값/클래스 가중치 쪽에 예산을 쓰지 말 것.** `cut_headroom`이 0.008715로, 이 모델의 랭킹을 어떻게 잘라도 얻을 수 있는 최대치(`balanced_accuracy_at_best_cut` 0.9426)까지의 거리가 검증 CI 폭보다 작습니다. 남은 개선은 운영점이 아니라 랭킹 축(모델 family 또는 특징)에 있습니다. 참고로 이 실행 환경에서는 임계값 탐색 자체가 불가하며, 유일한 레버는 `class_weight` / `scale_pos_weight`입니다.
2. **랭킹 축에서 family를 한 번 더 갈아보기 (`xgboost`), 단 기대치를 구간과 함께 정할 것.** 지금의 `roc_auc`는 0.9792입니다. 실행 환경 노트에 기록된 family 교체의 실측 폭은 `roc_auc` 0.0022~0.0077이고 쌍대 해상도는 0.003~0.006 수준이므로, 개선이 나와도 이 슬라이스에서는 재표본 잡음과 구분되지 않을 수 있습니다. 즉 "임계값 통과"가 아니라 "랭킹이 실제로 더 좋아지는지"를 묻는 확인 실험으로만 쓰고, `roc_auc`·`pr_auc`·`balanced_accuracy_at_best_cut`을 같이 읽어야 합니다.
3. **과적합 폭을 줄이는 정규화 스윕을 `hist_gbdt` 내부에서 1~2회.** `train_balanced_accuracy` 0.9992 / `train_val_gap` 0.065357은 훈련 데이터를 사실상 암기한 상태입니다. `max_leaf_nodes`를 31에서 낮추거나 `l2_regularization`을 1.0보다 올리는 방향, 혹은 `early_stopping=True`가 후보입니다. 다만 같은 family 내 하이퍼파라미터 재튜닝의 실측 폭은 `roc_auc` 0.0032 수준으로 해상도 경계에 있으므로, 개선이 아니라 "일반화 폭 축소가 랭킹을 해치지 않는지" 확인 목적으로 돌리는 것이 맞습니다.
4. **확률값을 산출물로 쓸 계획이면 보정을 파이프라인 밖에서 준비할 것.** `brier` 0.051284, `calibration_error` 0.047245이며 이 실행기에는 재보정 레버가 없습니다(`CalibratedClassifierCV`·임계값 이동 모두 불가). 스팸 판정에 확률 컷을 붙여 운용할 것이라면, 보정 단계를 실행기 외부의 기능으로 추가하는 일이 선행 과제이고, 그것이 해금되면 지금 랭킹(`roc_auc` 0.9792)이 갖고 있는 여유를 운영점에서 실제로 회수할 수 있습니다.