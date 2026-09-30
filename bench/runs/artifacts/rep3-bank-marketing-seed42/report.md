# 최종 리포트 — bank-marketing (binary_classification)

## 요약

목표는 달성되었습니다. 목표 지표 `balanced_accuracy`의 임계값은 0.7465였고, 1회 시도(예산 5회 중 1회)에서 `hist_gbdt`가 검증 `balanced_accuracy=0.8763` (95% CI 0.8647~0.8866)을 기록해 임계값과 카드 기준선(logreg, 0.662, CI 0.6496~0.6765)을 모두 큰 폭으로 넘었습니다. 한 번도 사용되지 않은 최종 테스트 20%에서의 점수는 `balanced_accuracy=0.8530` (95% CI 0.8415~0.8645)으로, 이것이 이 실행이 실제로 입증한 수치입니다. 첫 계획이 곧바로 기준을 통과했기 때문에 critic(진단·재계획) 경로는 한 번도 실행되지 않았습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=40`, `l2_regularization=1.0`, `early_stopping=false`, `class_weight={"0":1.0,"1":6.0}` | status `ok` — 검증 `balanced_accuracy=0.8763` (CI 0.8647~0.8866), `roc_auc=0.9415`, `recall=0.8639`, `specificity=0.8888`, 학습 7.521초 → 목표 달성 | 없음 (목표 달성으로 루프 종료, critic 미실행) |

## 최고 성능 구성

- **model**: `hist_gbdt`
- **hyperparams** (해당 시도에 적용된 값, `dropped_hyperparams`는 없음):
  ```json
  {
    "max_iter": 400,
    "learning_rate": 0.06,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "early_stopping": false,
    "class_weight": {"0": 1.0, "1": 6.0}
  }
  ```
- **preprocessing** (해당 시도의 기록된 값):
  ```json
  {"impute": "median", "scale": false, "missing_indicator": false, "missing_count": false}
  ```
  (카드의 `preprocessing.scale=true`는 요청값이며, 트리 계열에서는 스케일러가 적용되지 않았습니다.)
- **프로토콜**: stratified 3-way split, seed 42, train 60% / validation 20% / test 20%. 검증 점수는 카드 기준선과 동일한 분할에서 산출되어 직접 비교 가능합니다.

검증 슬라이스 지표:

| 지표 | 값 |
|---|---|
| `balanced_accuracy` | **0.8763** (95% CI 0.8647~0.8866) |
| `roc_auc` | 0.9415 |
| `pr_auc` / `average_precision` | 0.6397 |
| `f1` | 0.6392 |
| `accuracy` | 0.8859 |
| `precision` | 0.5072 |
| `recall` | 0.8639 |
| `specificity` | 0.8888 |
| `balanced_accuracy_at_best_cut` | 0.8859 |
| `cut_headroom` | 0.009564 |
| `brier` | 0.081202 |
| `calibration_error` | 0.091778 |
| `train_balanced_accuracy` | 0.9394 |
| `train_val_gap` | 0.063018 |

**최종 held-back 테스트(20%, 루프 중 한 번도 사용되지 않은 행)**: `balanced_accuracy=0.8530` (95% CI 0.8415~0.8645).

검증 0.8763 대비 테스트 0.8530으로 **+0.0234 차이**가 있으며, 이 차이가 선택 편향의 크기입니다 — 루프는 검증 숫자만 보고 최종 구성을 선택했습니다. 보고해야 할 성능은 테스트 0.8530이며, 이 값도 임계값 0.7465를 명확히 상회합니다.

## 원인 분석

- **critic 진단은 0건입니다.** 시도 1회, 진단 0회로, 첫 계획이 임계값을 통과하면서 루프가 즉시 종료되었습니다. 따라서 "여러 verdict에 걸친 패턴"은 존재하지 않으며, 이 점수는 **첫 계획 하나가 낸 결과**입니다. 이 실행은 진단·재계획 경로가 도움이 된다는 증거도, 해가 된다는 증거도 제공하지 않습니다.
- 시도가 1건뿐이므로 **시도 간 비교로 설명할 움직임 자체가 없습니다.** 아래 서술은 진단이 아니라, 이 한 번의 측정이 무엇을 보여주는지에 대한 관찰입니다.
- 기준선(logreg 0.662, CI 0.6496~0.6765)과 최고 시도(0.8763, CI 0.8647~0.8866)의 구간은 전혀 겹치지 않으므로, 이 개선은 리샘플링 잡음으로 설명되지 않는 크기입니다. 다만 이 실행에서는 모델 계열 교체(`logreg`→`hist_gbdt`)와 운영점 이동(`class_weight={"0":1,"1":6}`)이 **같은 시도에서 함께** 바뀌었기 때문에, 개선분이 랭킹 축(`roc_auc` 0.9101→0.9415)과 운영점 축 중 어디에서 얼마씩 왔는지는 이 데이터로 분리되지 않습니다.
- 현재 구성에서 운영점 축은 거의 소진되어 있습니다: `cut_headroom=0.009564`, `balanced_accuracy_at_best_cut=0.8859`. 즉 남은 여지는 대부분 랭킹(모델 계열·피처) 쪽이며, `class_weight`를 더 조정해 얻을 수 있는 양은 검증 CI 폭(약 ±0.011)보다도 작습니다.
- `plan.unsupported_claims`에 `feature_engineering`이 기록되어 있으나, 해당 계획 본문은 "one-hot 인코딩된 동일 행렬을 그대로 사용", "missingness 열은 추가하지 않음(카드상 실제 NaN이 0이고 'unknown'은 이미 별도 one-hot 열이므로 indicator는 상수가 됨)"이라고 **제약을 명시적으로 인지한 서술**입니다. 계획이 파생 피처에 의존하지 않았으므로, 이 항목 때문에 시도가 무언가를 잃었다고 볼 근거는 없습니다.
- 확률값 자체는 신뢰도가 낮습니다: `calibration_error=0.091778`(평균 약 9%p 오차), `brier=0.081202`. `balanced_accuracy`·`roc_auc`는 이에 영향받지 않지만, 확률을 그대로 의사결정 수치로 쓰려면 별도 처리가 필요합니다(이 executor에는 recalibration 레버가 없습니다).
- `train_val_gap=0.063018`(train 0.9394 vs val 0.8763)은 과적합이 파국적이지 않은 수준이며, 테스트 점수(0.8530)도 검증에서 크게 벗어나지 않았습니다.

## 다음 단계 제안

1. **보고 기준을 테스트 점수로 고정하고, 추가 튜닝은 랭킹 축에서만 계획한다.** `cut_headroom=0.009564`이므로 `class_weight`/`scale_pos_weight` 재조정으로 얻을 수 있는 양은 검증 CI 폭보다 작아 측정으로 구분되지 않습니다. 남은 예산을 쓸 경우 `hist_gbdt` 내부 재튜닝(`learning_rate`/`max_leaf_nodes`/`l2_regularization`) 또는 `xgboost`로의 계열 교체처럼 `roc_auc`를 움직이는 변경만 시도하되, 이 환경에서 쌍대 비교의 해상도는 대략 `roc_auc` 0.003~0.006 수준이라는 점을 감안해 그 이하의 차이는 개선으로 보고하지 않아야 합니다.
2. **`unknown` sentinel 문제는 카드 상류에서 해결한다 (V16 81.8%, V9 28.8%, V4 4.1%, V2 0.6%).** 이 값들은 결측 표기로 흔히 쓰이지만 변환되지 않은 상태로 실측치처럼 집계·학습에 들어갔습니다. executor는 열을 드롭·재인코딩할 수 없고(열은 계획이 아니라 카드에서 빠져야 합니다), 실제 NaN이 0%이므로 `missing_indicator`/`missing_count`도 상수·무의미합니다 — 즉 **루프 안에서는 손댈 수 없는 항목**입니다. 데이터 준비 단계에서 각 열의 `unknown`을 진짜 결측으로 변환할지, 별도 범주로 유지할지 결정한 뒤 카드를 재생성해야 하며, 그래야 `impute: none`(hist_gbdt/xgboost의 NaN 분기)이나 V16 제외 같은 선택지가 비로소 평가 가능해집니다.
3. **강한 타깃 상관 열(V12, `target_corr: strong`)이 예측 시점에 실제로 사용 가능한 정보인지 운영 측에 확인한다.** 이 열이 사후적으로만 알 수 있는 값이라면 현재의 0.8530도 실사용 성능을 과대평가한 것이 되며, 확인 결과에 따라 해당 열을 제외한 카드로 재실행해 성능 하한을 측정해야 합니다. 이는 수치가 아니라 검증해야 할 전제입니다.
4. **확률값을 사용해야 하는 용도라면 루프 밖에서 보정한다.** `calibration_error=0.091778`은 예측 확률이 평균 약 9%p 어긋난다는 뜻이고, 이 executor는 recalibration·임계값 탐색을 제공하지 않습니다. 순위(랭킹)만 필요한 용도라면 현재 모델을 그대로 쓰고, 확률 기반 의사결정이 필요하면 별도 파이프라인에서 보정을 붙이는 것이 맞습니다.