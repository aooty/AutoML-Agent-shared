# AutoML 최종 리포트 — `spambase` (binary_classification, 목표: `balanced_accuracy` 최대화)

## 요약

루프의 목표 판정 기준(validation 20% 슬라이스)에서는 목표를 달성했습니다: 1회차 `hist_gbdt`가 `balanced_accuracy` **0.9493** (95% CI 0.9319~0.9640)로 임계값 0.9421과 baseline `logreg` 0.9228을 모두 넘었고, 5회 예산 중 **1회**만 사용하고 `goal_reached`로 종료했습니다. 다만 단 한 번만 채점되는 held-back test 20%에서의 같은 모델 점수는 **0.9394** (95% CI 0.9216~0.9547)로, 임계값 0.9421을 약간 밑돕니다 — validation과의 차이 +0.0099가 선택 편향의 크기이며, 이 행들만으로는 그 차이를 0과 구분할 수 없습니다. 즉 "이 데이터에서 hist_gbdt가 baseline보다 확실히 낫다"는 결론은 지지되지만, "0.9421을 안정적으로 넘는다"는 결론은 held-back 측정 하나로는 확정되지 않습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=300`, `learning_rate=0.08`, `max_depth=7`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.1` / preprocessing: `impute=median`, `scale=false` | `status=ok` — validation `balanced_accuracy` **0.9493** (CI 0.9319~0.9640), `roc_auc` 0.9908, `recall` 0.9309 / `specificity` 0.967742, `train_val_gap` 0.030549, 학습 2.3초 → 목표(0.9421) 초과로 루프 종료 | 없음 (`critic: null`) — 1회차에서 목표가 달성되어 진단 단계가 실행되지 않았습니다. `dropped_hyperparams` 없음, `unsupported_claims` 없음 |

## 최고 성능 구성

**iteration 1 / `hist_gbdt`** (히스토리에 기록된 실제 적용값)

- `hyperparams`: `max_iter=300`, `learning_rate=0.08`, `max_depth=7`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `early_stopping=true`, `validation_fraction=0.1`
- `preprocessing` (해당 attempt에 기록된 적용값): `impute=median`, `scale=false`, `missing_indicator=false`, `missing_count=false`
  - 데이터에 결측이 전혀 없으므로(`overall_rate=0.0`) imputer는 실질적으로 무동작이며, 트리 계열이라 스케일링은 생략되었습니다.
- 프로토콜: stratified 3-way split, seed 42, train 60% / validation 20% / test 20% (카드 baseline과 동일 split)
- 학습 시간: 2.3초

| 지표 | validation (선택에 사용됨) | held-back test (한 번만 채점) |
|---|---|---|
| `balanced_accuracy` | **0.9493** (95% CI 0.9319~0.9640) | **0.9394** (95% CI 0.9216~0.9547) |

두 값의 차이 **+0.0099**(validation이 높음)가 바로 선택 편향의 크기입니다. 이 실행은 validation 숫자만 보고 모델을 골랐으므로, 모델의 성능을 대표하는 값은 held-back test의 0.9394입니다. validation 점수 0.9493은 test CI(0.9216~0.9547) 안에 들어가므로, 이 행 수로는 두 측정의 차이를 0과 구분할 수 없습니다.

validation 슬라이스의 나머지 지표: `accuracy` 0.9533, `f1` 0.9400, `precision` 0.9493, `recall` 0.9309, `specificity` 0.967742, `roc_auc` 0.9908, `pr_auc` / `average_precision` 0.9869, `brier` 0.035822, `calibration_error` 0.019038, `balanced_accuracy_at_best_cut` 0.957, `cut_headroom` 0.007659, `train_balanced_accuracy` 0.9799, `train_val_gap` 0.030549.

## 원인 분석

- **비교 가능한 attempt가 하나뿐**입니다. 시도 간 이동으로 이야기를 만들 근거가 없으므로, 유일하게 해석 가능한 비교는 baseline 대비입니다: `logreg` 0.9228 (CI 0.903~0.941) → `hist_gbdt` 0.9493 (CI 0.9319~0.9640). 두 구간은 일부 겹치므로 차이 +0.0265를 "이 슬라이스에서 확실히 분리된 개선"으로 못박기는 어렵지만, 개선의 방향과 근거는 ranking 축에서 훨씬 선명합니다: `roc_auc`가 0.9772 → 0.9908, `balanced_accuracy_at_best_cut`이 0.9415 → 0.957로 올라갔습니다. 목표에 남아 있던 거리는 결정 규칙이 아니라 순위 자체에 있었고, 계열 교체(linear → boosted trees)가 그 축을 실제로 움직였습니다.
- **operating point 축에는 남은 여지가 거의 없습니다.** `cut_headroom`이 0.007659에 불과하고 `recall` 0.9309 / `specificity` 0.967742로 두 방향이 이미 균형에 가깝습니다. 즉 `class_weight='balanced'`나 `scale_pos_weight`로 얻을 수 있는 최대치는 0.008 미만이고, 이는 이 슬라이스의 CI 폭(약 ±0.016)보다 작아 측정으로 확인조차 되지 않습니다. 계획이 1회차에서 결정 규칙을 건드리지 않은 판단은 결과적으로 맞았습니다.
- **일반화 여유가 얇은 것이 실질적 제약**입니다. `train_balanced_accuracy` 0.9799 대 validation 0.9493으로 `train_val_gap`이 0.030549, 그리고 validation → test에서 다시 0.0099가 빠져 test 0.9394가 임계값 0.9421 바로 아래에 놓였습니다. 임계값과의 거리가 CI 폭보다 훨씬 작으므로, 지금 상태는 "임계값 근처에서 측정 잡음과 같은 크기의 마진"이라고 읽는 것이 정확합니다.
- **확률 품질은 문제가 아닙니다.** `brier` 0.035822, `calibration_error` 0.019038 — 예측 확률은 평균적으로 약 2%p 오차 수준이며, 여기에 recalibration 레버는 존재하지 않으므로 조치 대상도 아닙니다.
- `dropped_hyperparams`와 `unsupported_claims` 모두 비어 있습니다. 즉 이 attempt는 계획한 설정을 그대로 실행했고, executor의 제약 때문에 검증되지 못한 항목은 없습니다. `critic`이 `null`인 것은 진단이 실패했기 때문이 아니라 1회차에서 목표 판정이 통과되어 루프가 즉시 종료된 결과입니다.

## 다음 단계 제안

1. **다른 seed로 같은 구성을 재측정해 임계값 마진을 확정하기 (최우선).** 현재 결론의 유일한 약점은 test 0.9394가 임계값 0.9421 아래이고, 그 차이가 CI 폭보다 작다는 점입니다. seed/split은 실행 설정으로 루프 안에서 바꿀 수 없으므로 운영자 측에서 seed만 바꾼 동일 구성 실행을 2~3회 추가해야 합니다. 여러 seed의 held-back 점수가 대체로 0.9421 위에 모이면 지금 구성으로 목표를 선언할 수 있고, 그렇지 않으면 아래 2·3번이 필요하다는 것이 확정됩니다. (교차검증·OOF는 이 executor에서 불가하므로 seed 반복이 유일한 대체 수단입니다.)
2. **일반화 여유를 넓히는 방향으로만 hist_gbdt를 재조정.** `train_val_gap` 0.030549와 test에서의 0.0099 하락은 모두 과적합 쪽을 가리킵니다. `l2_regularization`을 1.0 → 3~10, `max_depth` 7 → 4~5, `max_leaf_nodes` 31 → 15로 낮추고 `learning_rate` 0.08 / `max_iter` 300과 `early_stopping=True`는 유지하는 조합을 1~2회 시도할 만합니다. 단, 동일 계열 내 재조정은 다른 샘플에서 `roc_auc` 0.0032 정도의 폭으로 관측된 레버이며, 이 슬라이스의 CI 폭(±0.016)보다 작습니다 — 점수 상승을 기대하는 근거가 아니라 train/val 간격 축소로 판단해야 합니다.
3. **ranking 축 확인용으로 `xgboost` 또는 `random_forest`를 한 번 교체 시도.** 남은 거리가 operating point가 아니라 순위에 있다는 것은 `cut_headroom` 0.007659가 이미 말해 줍니다. 계열 교체는 다른 샘플에서 `roc_auc` 0.0022~0.0077의 폭으로 관측되었고 그 하한은 재조정과 구분되지 않으므로, "더 큰 레버"로 기대하지 말고 현재 `roc_auc` 0.9908을 넘는 순위가 존재하는지 확인하는 진단 목적의 1회 시도로 예산을 잡는 것이 맞습니다.
4. **`class_weight` / 결측 관련 레버에는 예산을 쓰지 않기.** `cut_headroom`이 0.008 미만이라 결정 규칙 이동의 상한이 측정 해상도 이하이고, 이 데이터는 결측률이 0.0이므로 `missing_indicator`·`missing_count`는 상수 열만 추가합니다. 추가 성능이 필요하다면 남은 경로는 (a) 위의 계열/정규화 탐색, 또는 (b) executor 바깥에서의 피처 가공(비율·상호작용·로그 변환 등 — 이 실행에서는 파생·삭제가 모두 불가하므로 데이터 카드 자체를 바꿔 다시 넣어야 함)입니다. 카드에 별도 기록된 데이터 주의사항은 없으므로, 위 제안들과 상충하는 열/split 신뢰성 문제는 확인되지 않았습니다.