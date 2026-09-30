# AutoML 최종 리포트 — `adult` (binary_classification, balanced_accuracy)

## 요약

목표를 달성했습니다. 첫 번째 시도에서 `hist_gbdt`가 검증 balanced_accuracy **0.8460** (95% CI 0.8372~0.8546)을 기록해 목표 임계값 0.8263과 baseline(logreg) 0.7684를 모두 상회했고, 루프는 1회 시도(예산 5회 중)로 종료되었습니다. 최종 홀드백 테스트(한 번도 사용되지 않은 20% 행)에서는 **balanced_accuracy = 0.8352** (95% CI 0.8267~0.8430)로, 임계값 0.8263을 여전히 넘습니다. 즉 이 실행이 실제로 입증한 값은 검증 점수가 아니라 0.8352입니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=false`, `class_weight={"0":1.0,"1":3.0}` | status `ok` — balanced_accuracy **0.8460** (CI 0.8372~0.8546), roc_auc 0.9297, pr_auc 0.8308, recall 0.8519 / specificity 0.8402, `cut_headroom` 0.000651, `train_val_gap` 0.045585, 11.946s | 없음 (목표 달성으로 루프 종료, critic 미실행) |

## 최고 성능 구성

- **iteration**: 1
- **model**: `hist_gbdt`
- **hyperparams** (해당 시도에 실제 적용된 값):
  ```json
  {
    "max_iter": 400,
    "learning_rate": 0.06,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": false,
    "class_weight": {"0": 1.0, "1": 3.0}
  }
  ```
- **preprocessing** (해당 시도에 실제 적용된 값): `impute: median`, `scale: false`, `missing_indicator: false`, `missing_count: false`
- **dropped_hyperparams**: 없음 (실행기가 거부한 키 없음)
- **프로토콜**: stratified 3-way split, seed 44 — train 60% / val 20% / test 20%, `predict()` 기본 결정 규칙

| 지표 | 검증(val 20%) | 홀드백 테스트(test 20%) |
|---|---|---|
| balanced_accuracy | **0.8460** (CI 0.8372~0.8546) | **0.8352** (CI 0.8267~0.8430) |
| roc_auc | 0.9297 | — |
| pr_auc / average_precision | 0.8308 | — |
| accuracy | 0.8430 | — |
| f1 | 0.7219 | — |
| precision / recall | 0.6263 / 0.8519 | — |
| specificity | 0.8402 | — |
| balanced_accuracy_at_best_cut | 0.8467 (`cut_headroom` 0.000651) | — |
| brier / calibration_error | 0.1047 / 0.0912 | — |
| train_balanced_accuracy / train_val_gap | 0.8916 / 0.0456 | — |

검증 0.8460과 테스트 0.8352의 차이 **+0.0109**가 선택 편향의 크기입니다 — 루프는 검증 숫자만 보고 최종 구성을 골랐으므로, 모델에 대해 이 실행이 보증하는 값은 테스트 0.8352 쪽입니다. 두 구간(0.8372~0.8546 / 0.8267~0.8430)은 겹치지 않으므로, 이 차이는 리샘플 잡음만으로 설명되지 않습니다.

## 원인 분석

이 실행에서는 critic이 **한 번도 실행되지 않았습니다**(시도 1회, 진단 0회). 첫 계획이 곧바로 임계값을 넘겨 루프가 종료되었기 때문에, 검증 verdict가 존재하지 않고 따라서 "verdict들 사이의 패턴"이라고 서술할 대상이 없습니다. 이 점수는 전적으로 첫 번째 계획 하나가 낸 결과이며, 진단·재계획 경로가 도움이 되었다는 증거도 해가 되었다는 증거도 이 실행에는 없습니다. 시도가 1회뿐이므로 시도 간 차이(구간 비교)로 성능 제약 요인을 규명하는 작업도 불가능합니다 — 아래는 진단이 아니라, 다음 실행에서 확인해야 할 지점을 지목하는 관측 사실입니다.

- 계획서에는 `feature_engineering`이 `unsupported_claims`로 표시되어 있으나, 계획 본문을 읽으면 파생 피처를 만들겠다는 것이 아니라 "one-hot 지표 열과 6개 수치열은 트리에 스케일이 불필요하므로 `scale=false`", "결측 열을 추가하지 않는다"는 **제약 준수 서술**입니다. 즉 이 시도가 미지원 기능에 의존해 무언가를 잃은 것은 아니고, 0.8460은 이 실행기가 실제로 낸 값입니다. 피처 조합·재인코딩은 실패한 항목이 아니라 아직 존재하지 않는 기능이며, 그래서 아래 `다음 단계 제안`에 둡니다.
- `dropped_hyperparams`는 비어 있어, 계획된 설정이 그대로 반영되었습니다.
- 참고 관측: `cut_headroom = 0.000651`은 기본 `predict()` 컷이 이 랭킹이 허용하는 최적점(0.8467)에 사실상 붙어 있음을 의미하고, recall 0.8519 / specificity 0.8402는 균형이 잡힌 지점입니다. 남은 여지는 operating point가 아니라 랭킹 축(roc_auc 0.9297, pr_auc 0.8308) 쪽에 있다는 표시입니다. 다만 이는 단일 시도의 지표 읽기이며, 확인된 진단이 아닙니다.

## 다음 단계 제안

이 데이터에는 별도로 기록된 caveat이 없으므로(`Data caveats`: 없음), 특정 열·값·split을 배제해야 할 제약은 없습니다. 그럼에도 결측 관련 열은 아래처럼 조심해서 다룹니다.

1. **랭킹 축을 겨냥한 소폭 탐색 1~2회.** `cut_headroom`이 0.000651로 사실상 0이므로 `class_weight` 재조정으로 얻을 것은 거의 없습니다. 남은 예산은 `hist_gbdt` 용량 쪽(`max_leaf_nodes` 63, `learning_rate` 0.04 + `max_iter` 600, `l2_regularization` 조정)이나 `xgboost` 계열 교체에 쓰는 편이 낫습니다. 다만 실행기 문서가 밝힌 대로 동일 계열 내 재튜닝은 roc_auc로 0.003 수준, 계열 교체도 0.002~0.008 범위였고 쌍대 해상도가 0.003~0.006이므로, **개선을 주장하려면 구간이 겹치지 않는지 반드시 확인**해야 합니다.
2. **`impute: none`으로 트리가 NaN을 직접 분기하게 해보기.** 결측은 `workclass`(0.0573), `occupation`(0.0575), `native-country`(0.0175) 세 열, 전체율 0.0095에 그칩니다. `hist_gbdt`는 `impute: none`을 받을 수 있으므로 median 대체 대신 NaN 분기를 한 번 시험할 가치는 있습니다. 반면 `missing_indicator`는 `impute: none`과 함께 쓰면 문서상 예측이 비트 단위로 동일해지는 **중복** 레버이므로 조합하지 말고, `missing_count` 역시 단독 효과가 0.0003 수준으로 보고된 만큼 우선순위에서 뺍니다.
3. **검증-테스트 격차(+0.0109)를 전제로 목표를 설정.** 테스트 0.8352는 임계값 0.8263 위지만 여유는 약 0.009로 CI 폭보다 크지 않습니다. 추가 예산을 쓴다면 검증 점수가 아니라 "검증에서 최소 0.01 이상 벌어야 테스트에서 유지된다"는 기준으로 후보를 걸러, 선택 편향에 먹히지 않을 마진을 확보하는 것이 실질적입니다.
4. **기능으로 요청할 항목(현재 실행기 밖).** `capital-gain`/`capital-loss`의 로그 변환, `education`×`occupation` 상호작용, `fnlwgt`(target_corr: none) 제거 같은 피처 작업은 실행기가 지원하지 않아 이번 실행에서 **시험되지 않았습니다**. 이는 계획의 실패가 아니라 미구현 역량이며, 파생·드롭 기능이 추가되면 랭킹 축(roc_auc 0.9297)에서 남은 여지를 처음으로 검증할 수 있게 됩니다. 확률값을 그대로 쓸 계획이라면 `calibration_error` 0.0912(평균 약 9%p 오차)도 함께 다뤄야 하지만, 현재 실행기에 재캘리브레이션 레버가 없으므로 이것도 기능 요청 항목입니다.