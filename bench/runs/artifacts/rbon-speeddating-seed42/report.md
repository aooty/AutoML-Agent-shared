# AutoML 최종 리포트 — `speeddating` / `match` (binary_classification)

## 요약

**목표는 달성하지 못했습니다.** 목표 임계값은 `balanced_accuracy` ≥ 0.7514였고, 루프가 선택한 최고 구성(iteration 1, `hist_gbdt`)의 검증 점수는 **0.6853** (95% CI 0.6553~0.7129), 한 번도 사용되지 않은 최종 테스트 20%에서의 점수는 **0.7281** (95% CI 0.7010~0.7604)로 두 측정 모두 임계값에 미달했습니다. 반복 예산은 1회였고 그 1회를 모두 사용했으며(`stop_reason: max_iterations`), 시도는 총 1건입니다. 다만 이 시도의 `balanced_accuracy_at_best_cut`이 **0.8026**이라는 점은, 순위(ranking) 자체는 목표선 위에 있고 부족분이 주로 결정 지점(operating point)에 있음을 시사합니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=false`, `class_weight={"0":1.0,"1":4.0}` / preprocessing: `impute=none`, `scale=false`, `missing_indicator=false`, `missing_count=true` | status `ok` — 검증 `balanced_accuracy` **0.6853** (CI 0.6553~0.7129), `roc_auc` 0.8691, `recall` 0.4420 / `specificity` 0.9286, `cut_headroom` 0.1173, `train_val_gap` 0.3146, 학습 10.205초 | 없음 (critic 미실행) |

`dropped_hyperparams`는 비어 있고(모든 키가 estimator에 그대로 적용됨), `unsupported_claims`도 비어 있습니다.

## 최고 성능 구성

아래 값은 계획서가 아니라 **실제로 실행된 attempt(iteration 1)의 기록**에서 그대로 인용한 것입니다.

- **model**: `hist_gbdt`
- **hyperparams** (적용값):
  - `max_iter: 400`
  - `learning_rate: 0.06`
  - `max_leaf_nodes: 31`
  - `min_samples_leaf: 20`
  - `l2_regularization: 1.0`
  - `early_stopping: false`
  - `class_weight: {"0": 1.0, "1": 4.0}`
- **preprocessing** (적용값): `impute: none` (imputer 없음 — NaN을 트리가 직접 분기), `scale: false`, `missing_indicator: false`, `missing_count: true`
- **프로토콜**: stratified 3-way split, seed 42, train 60% / validation 20% / test 20% (카드 baseline과 동일 분할)

검증 슬라이스 지표:

| metric | value |
|---|---|
| `balanced_accuracy` | **0.6853** (95% CI 0.6553~0.7129) |
| `roc_auc` | 0.8691 |
| `pr_auc` / `average_precision` | 0.5858 |
| `f1` | 0.4900 |
| `accuracy` | 0.8484 |
| `precision` | 0.5495 |
| `recall` | 0.4420 |
| `specificity` | 0.9286 |
| `balanced_accuracy_at_best_cut` | 0.8026 |
| `cut_headroom` | 0.1173 |
| `brier` | 0.10591 |
| `calibration_error` | 0.061968 |
| `train_balanced_accuracy` | 0.9999 |
| `train_val_gap` | 0.3146 |
| `train_time_sec` | 10.205 |

**최종 held-back 측정**: 동일 모델을 루프 중 한 번도 읽지 않은 테스트 20%에서 1회 채점한 결과 `balanced_accuracy` = **0.7281** (95% CI 0.7010~0.7604). 검증 0.6853과의 차이는 0.0428이며, 이 차이가 이 실행의 **선택 편향(selection effect)의 크기**입니다 — 루프의 모든 선택은 검증 숫자만 보고 이루어졌고, 모델에 대해 이 실행이 실제로 입증한 값은 테스트 숫자입니다. 다만 두 구간(0.6553~0.7129, 0.7010~0.7604)은 겹치므로, 이 차이를 "테스트에서 실제로 더 좋아졌다"는 결론으로 읽어서는 안 됩니다. 어느 쪽 숫자로 보든 임계값 0.7514는 넘지 못했습니다(테스트 CI 상단만 0.7604로 임계값을 걸칩니다).

## 원인 분석

**이 실행에서 critic은 한 번도 실행되지 않았습니다(시도 1회, 진단 0회).** 따라서 "verdict들 사이의 패턴"은 존재하지 않고, 여기서 보고할 점수는 첫 계획 하나가 낸 결과입니다. 진단·재계획 경로가 도움이 되었다는 증거도, 해가 되었다는 증거도 이 실행에는 없습니다 — 이 실행은 그 루프에 대해 아무것도 말해주지 않습니다.

시도가 1건뿐이므로 시도 간 이동을 해석할 수도 없습니다. 참고로 비교 가능한 기준선은 카드의 `logreg` baseline(`balanced_accuracy` 0.6685, CI 0.6425~0.6950)인데, iteration 1의 검증 구간(0.6553~0.7129)과 크게 겹칩니다. 즉 **이 데이터의 검증 20%로는 iteration 1과 baseline을 구분할 수 없습니다**; 0.6853 대 0.6685의 0.0168 차이는 구간 폭보다 작으므로 개선으로 보고하지 않습니다.

진단이 없으므로 아래는 원인 규명이 아니라, 다음 계획을 세울 때 읽어야 할 **관측 사실**로만 적습니다(어느 것도 critic이 내린 판단이 아닙니다).

- 두 축을 분리해서 읽으면: 순위 축의 지표 `balanced_accuracy_at_best_cut`은 0.8026이고, 실제 달성 점수와의 거리인 `cut_headroom`은 0.1173입니다. 즉 이 모델의 순위를 어떤 지점에서 자르느냐만으로도 임계값 0.7514를 상당히 넘길 여지가 이 검증 슬라이스에는 있었습니다.
- 결정 지점은 여전히 다수 클래스 쪽에 크게 치우쳐 있습니다: `recall` 0.4420 대 `specificity` 0.9286. `class_weight={"0":1,"1":4}`는 클래스 빈도비 5.07보다 낮은 값이었습니다.
- 적합 쪽은 `train_balanced_accuracy` 0.9999, `train_val_gap` 0.3146으로 훈련 데이터를 사실상 암기한 상태입니다(`early_stopping=false`, `max_iter=400`).
- 확률값은 완벽히 읽을 만한 수준은 아닙니다: `brier` 0.10591, `calibration_error` 0.061968(평균 약 6%p 오차). 이 실행 환경에는 재보정 레버가 없으므로 이는 해석용 정보입니다.

## 다음 단계 제안

기록된 `Data caveats`는 없으므로 특정 열·값·분할을 배제해야 하는 제약은 없습니다. 아래는 이 실행에서 실제로 관측된 숫자에만 근거합니다.

1. **가장 먼저 결정 지점(operating point)을 옮긴다 — `class_weight`를 4.0보다 크게.** 근거는 `cut_headroom` 0.1173과 `recall` 0.4420 / `specificity` 0.9286의 비대칭입니다. `cut_headroom`이 이렇게 클 때 남은 격차는 순위가 아니라 컷의 위치에 있고, 이 실행에서 컷을 움직이는 유일한 레버가 `class_weight`입니다. 같은 `hist_gbdt` 구성에서 `class_weight='balanced'`(빈도비 5.07 지점)와 명시값 `{"0":1,"1":8}`을 각각 1회씩 시도해 `recall`/`specificity`가 어느 지점에서 교차하는지, 그리고 `cut_headroom`이 실제로 줄어드는지를 확인하십시오. 단, `balanced_accuracy_at_best_cut` 0.8026은 이 검증 슬라이스의 최선 컷 값이지 보장이 아니고, 검증 구간 폭이 ±0.03 수준이므로 그보다 작은 변화는 개선으로 세지 말아야 합니다.
2. **과적합을 줄여 순위 축을 조금 더 확보한다.** `train_val_gap` 0.3146, `train_balanced_accuracy` 0.9999는 400 부스팅 라운드를 조기 종료 없이 돌린 결과입니다. `early_stopping=true` + `validation_fraction`(train의 일부를 내부 검증으로 뗀다는 비용을 감수), 또는 `max_leaf_nodes`를 낮추고 `min_samples_leaf`·`l2_regularization`을 올리는 방향을 1회 시도할 가치가 있습니다. 단, 실행 환경 문서상 같은 계열 내 하이퍼파라미터 재조정이 순위 축(`roc_auc`)에서 움직인 폭은 작았습니다 — 이 항목은 1번보다 기대값이 낮은 후속 순번으로 두십시오.
3. **계열 교체는 `xgboost` + `scale_pos_weight` 한 번으로.** `impute: none`을 유지할 수 있는 다른 계열이 `xgboost`이고, 여기서는 `scale_pos_weight`로 1번과 동일한 컷 이동을 시험할 수 있습니다. 계열 교체가 순위 축에서 주는 폭은 이 환경에서 0.0022~0.0077 사이로 관측되어 있어 단독으로는 임계값 격차를 메우기 어렵습니다 — 반드시 1번의 컷 이동과 **함께** 걸고, 손실이 생겼을 때 원인이 두 개가 되지 않도록 컷 설정을 1번의 최선값으로 고정해서 돌리십시오.
4. **반복 예산을 1회 이상으로 늘려 critic이 최소 한 번은 돌게 한다.** 이번 실행은 진단 0회로 끝났고, 그 결과 재계획 경로에 대해 아무 증거도 남기지 못했습니다. 또한 선택 편향(검증 대비 0.0428)을 판단할 근거도 시도 1건뿐이라 매우 약합니다. 최소 3~4회 예산이면 1~3번을 순서대로 검증하면서 각 시도의 CI 겹침 여부를 근거로 실제 개선과 리샘플링 잡음을 구분할 수 있습니다.

참고로 **`missing_indicator`에는 반복을 쓰지 마십시오**: 최고 구성이 `impute: none`을 쓰고 있고, NaN을 직접 분기하는 계열에서 이 옵션은 문서상 비트 단위로 중복(redundant)으로 확인되어 있습니다. 현재 사용 중인 `missing_count`는 유지해도 무해하지만, 그 자체로 이득이 관측된 레버는 아닙니다.