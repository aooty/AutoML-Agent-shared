# AutoML 실행 최종 보고서 — `spambase` / `balanced_accuracy`

## 요약

목표는 달성되었습니다. 1회 시도(예산 1회 중 1회)에서 `hist_gbdt`가 검증 `balanced_accuracy` **0.9539** (95% CI 0.9398~0.9673)를 기록해 목표 임계값 0.9421을 넘었고, 루프는 첫 시도에서 종료되었습니다. 반복 중 한 번도 사용되지 않은 최종 테스트 20% 행에서의 점수는 **0.9454** (95% CI 0.9288~0.9598)로, 이 값이 이 실행이 실제로 입증한 수치입니다. 카드의 `logreg` 베이스라인(0.9228, CI 0.903~0.941)보다 높지만, 두 CI가 겹치는 범위가 있으므로 개선폭 자체는 이 슬라이스에서 정밀하게 분리되지 않습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.08`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `early_stopping=False` | status `ok` — 검증 `balanced_accuracy`=0.9539 (CI 0.9398~0.9673), `roc_auc`=0.9909, `train_val_gap`=0.0461, 학습 3.999초 | 없음 (`critic: null` — 목표 달성으로 루프 종료) |

`dropped_hyperparams`는 비어 있어, 제안된 하이퍼파라미터는 모두 estimator에 그대로 전달되었습니다. `unsupported_claims`도 비어 있습니다.

## 최고 성능 구성

- **model**: `hist_gbdt`
- **hyperparams** (해당 시도에 적용된 값): `max_iter=400`, `learning_rate=0.08`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `early_stopping=false`
- **preprocessing** (해당 시도의 적용값): `impute="median"`, `scale=false`, `missing_indicator=false`, `missing_count=false`
- **프로토콜**: stratified 3-way split, seed 42, train 60% / validation 20% / test 20% (카드 베이스라인과 동일 분할)

| 지표 | 검증(20%) | 최종 테스트(20%, 반복 중 미사용) |
|---|---|---|
| `balanced_accuracy` | **0.9539** (CI 0.9398~0.9673) | **0.9454** (CI 0.9288~0.9598) |

두 값의 차이 **+0.0085**(검증이 더 높음)는 선택 편향의 크기입니다 — 루프는 검증 점수를 보고 최종 모델을 골랐기 때문입니다. 다만 검증 점수가 테스트 CI 안에 들어 있어, 이 행 수로는 그 차이가 0과 구분되지 않습니다. 테스트 점수 0.9454는 임계값 0.9421을 넘지만 여유는 0.0033뿐이고, 테스트 CI 하단(0.9288)은 임계값 아래입니다.

검증 슬라이스의 나머지 지표: `accuracy`=0.9576, `f1`=0.9456, `precision`=0.9549, `recall`=0.9365, `specificity`=0.9713, `roc_auc`=0.9909, `pr_auc`/`average_precision`=0.9881, `brier`=0.0342, `calibration_error`=0.0251, `balanced_accuracy_at_best_cut`=0.9568, `cut_headroom`=0.0029, `train_balanced_accuracy`=1.0, `train_val_gap`=0.0461.

## 원인 분석

**진단은 존재하지 않습니다.** `Outcome`의 재계획 항목에 기록된 대로 critic은 한 번도 실행되지 않았습니다(시도 1회, 진단 0회). 첫 계획이 곧바로 임계값을 넘겨 루프가 종료되었기 때문에, 이 점수는 전적으로 **첫 계획 하나**가 낸 결과입니다. 따라서 이 실행은 진단·재계획 경로가 도움이 된다는 증거도, 해가 된다는 증거도 제공하지 않습니다.

시도가 하나뿐이므로 시도 간 움직임에서 원인을 구성할 수도 없습니다. 비교 가능한 참조점은 카드의 `logreg` 베이스라인 하나이며, 베이스라인 CI(0.903~0.941)와 이번 시도의 검증 CI(0.9398~0.9673)는 상단·하단이 근접하게 겹치므로, "선형 → 부스팅 트리 전환의 이득 크기"를 이 데이터로 수치로 확정할 수는 없습니다. 확정적으로 말할 수 있는 것은 두 가지뿐입니다.

- 목표 임계값 0.9421은 베이스라인의 `ranking_ceiling` 0.9415보다 위였습니다(`exceeds_ranking_ceiling: true`). 즉 베이스라인의 순위(ranking) 자체로는 어떤 컷을 골라도 도달할 수 없는 목표였고, 모델 계열 교체가 필요한 종류의 목표였다는 점은 목표 정의에서 이미 주어진 사실입니다.
- 달성된 구성에서 `cut_headroom`=0.0029, `balanced_accuracy_at_best_cut`=0.9568입니다. 즉 기본 결정 규칙이 이미 이 모델 순위의 거의 최적점 근처에서 자르고 있고, 남은 여지는 operating point 축이 아니라 ranking 축(모델 계열 또는 피처)에 있습니다. 이는 관측된 지표일 뿐이며 진단이 아닙니다.

`train_balanced_accuracy`=1.0, `train_val_gap`=0.0461은 학습 데이터를 완전히 적합했음을 보여주는 관측치이지만, 비교 대상 시도가 없으므로 이것이 성능을 제약했는지 여부는 이 실행으로 판단할 수 없습니다.

## 다음 단계 제안

데이터 카드에 별도로 기록된 주의사항은 없으므로(`Data caveats`: 없음), 특정 열·값·분할을 배제해야 할 근거는 없습니다. 아래 제안은 이번 실행이 남긴 실제 여백만을 근거로 합니다.

1. **테스트 여유폭을 굳히는 것을 최우선으로.** 테스트 0.9454는 임계값 0.9421을 0.0033만 넘고, 테스트 CI 하단은 임계값 아래입니다(0.9288). 분할·시드는 실행 설정이라 루프 안에서 바꿀 수 없으므로, 운영자 레벨에서 **다른 seed의 동일 프로토콜 실행을 2~3회 추가**해 같은 구성이 임계값을 반복해서 넘는지 확인해야 합니다. 이것이 없으면 현재 통과는 한 번의 분할 결과입니다.
2. **추가 예산은 ranking 축에만 쓰기.** `cut_headroom`=0.0029이므로 `class_weight`/`scale_pos_weight`로 얻을 수 있는 것은 거의 없습니다. 쓸 수 있는 레버는 계열 교체(`xgboost`, `random_forest`)와 `hist_gbdt` 내부 재튜닝(`learning_rate`, `max_leaf_nodes`, `l2_regularization`, `max_iter`)입니다. 단, 사전 관측된 크기는 계열 교체 0.0022~0.0077 `roc_auc`, 동일 계열 재튜닝 0.0032 `roc_auc` 수준이고 이번 검증 CI 폭은 약 0.028입니다 — **단일 시도 간 차이는 이 슬라이스에서 구분되지 않을 것**이므로, 반복 시드 평균으로 비교하도록 실행 설계를 바꾸지 않는 한 "개선"으로 보고할 수 없습니다.
3. **정규화 방향의 재튜닝을 한 슬롯 배정.** `train_balanced_accuracy`=1.0이므로 `l2_regularization` 상향 또는 `max_leaf_nodes` 하향, `early_stopping=true`(+`validation_fraction`) 조합을 한 번 시도해 볼 값은 있습니다. 다만 이는 가설이며(이번 실행에 진단이 없었으므로 진단으로 뒷받침되지 않음), 2번의 해상도 한계가 그대로 적용됩니다.
4. **확률값을 다운스트림에서 쓸 계획이면 캘리브레이션은 루프 밖에서 처리.** `brier`=0.0342, `calibration_error`=0.0251은 확률이 평균 약 2.5%p 어긋나 있음을 뜻하고, 이 executor에는 재캘리브레이션 레버가 없습니다(`CalibratedClassifierCV`나 컷 이동은 제안 대상이 아님). 순위 지표(`roc_auc`=0.9909)는 캘리브레이션과 무관하므로, 필요한 경우 별도 파이프라인 단계로 **구축**해야 하며 AutoML 시도 예산으로 해결할 항목이 아닙니다.