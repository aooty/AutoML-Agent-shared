## 요약

목표는 달성되었습니다. 임계값 `balanced_accuracy` ≥ 0.8248에 대해 iteration 1의 `hist_gbdt`가 검증 슬라이스에서 0.8484 (95% CI 0.8400~0.8556)를 기록해, 첫 시도에서 바로 통과했습니다(5회 예산 중 1회 사용, stop reason `goal_reached`). 한 번도 학습·선택에 쓰이지 않은 최종 테스트 20%에서의 점수는 0.8435 (95% CI 0.8352~0.8518)로, 이 값이 이번 실행이 실제로 입증한 성능입니다. 베이스라인 `logreg`의 0.7664 (CI 0.7575~0.7762)와 비교하면 검증·테스트 어느 쪽으로 보아도 구간이 겹치지 않는 명확한 개선입니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `min_samples_leaf=20`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=25`, `class_weight='balanced'` / preprocessing: `impute='none'`, `scale=False`, `missing_indicator=False`, `missing_count=False` | status `ok`, `balanced_accuracy=0.8484` (CI 0.8400~0.8556) → 임계값 0.8248 통과. `roc_auc=0.9295`, `balanced_accuracy_at_best_cut=0.8497`, `cut_headroom=0.001257`, `train_val_gap=0.02043`, 학습 8.132초 | 없음 (`critic: null` — 목표 달성으로 루프가 종료되어 진단이 생성되지 않음) |

`dropped_hyperparams`는 비어 있어, `class_weight='balanced'`를 포함한 모든 키가 실제 estimator에 적용되었습니다.

## 최고 성능 구성

- **model**: `hist_gbdt`
- **hyperparams** (해당 attempt에 적용된 값):
  - `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `min_samples_leaf=20`
  - `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=25`
  - `class_weight='balanced'`
- **preprocessing** (해당 attempt에 적용된 값): `impute='none'`, `scale=False`, `missing_indicator=False`, `missing_count=False`
  - 즉 결측치는 대치하지 않고 트리가 NaN 분기를 직접 학습했습니다(카드의 `impute: median`, `scale: true`는 요청값일 뿐이며 이 실행에는 적용되지 않았습니다).
- **프로토콜**: stratified 3-way split, seed 42, train 60% / val 20% / test 20% (베이스라인과 동일 분할)
- **학습 시간**: 8.132초 (제한 600초)

| 지표 | 검증(선택에 사용) | 최종 테스트(한 번도 쓰이지 않은 행) |
|---|---|---|
| `balanced_accuracy` | **0.8484** (CI 0.8400~0.8556) | **0.8435** (CI 0.8352~0.8518) |

검증과 테스트의 차이는 +0.0050이며, 이것이 이번 실행의 **선택 편향(selection effect)** 크기입니다 — 루프의 모든 선택은 검증 숫자로 이루어졌기 때문입니다. 다만 검증 점수 0.8484가 테스트 CI(0.8352~0.8518) 안에 들어 있어, 이 행들만으로는 그 차이를 0과 구분할 수 없습니다. 테스트 CI 하단(0.8352)도 임계값 0.8248을 넘습니다.

검증 슬라이스의 나머지 지표: `accuracy=0.8397`, `f1=0.7209`, `precision=0.6178`, `recall=0.8652`, `specificity=0.831674`, `roc_auc=0.9295`, `pr_auc / average_precision=0.8294`, `balanced_accuracy_at_best_cut=0.8497`, `cut_headroom=0.001257`, `brier=0.107308`, `calibration_error=0.100639`, `train_balanced_accuracy=0.8689`, `train_val_gap=0.02043`.

## 원인 분석

시도가 1회뿐이므로 attempt 간 이동으로 이야기를 만들 수 없습니다. 대신 두 축(랭킹 / 작동점)을 분리해 읽으면 이번 결과가 왜 첫 시도에 통과했는지가 그대로 보입니다.

- **임계값이 베이스라인의 랭킹 천장 위에 있었다는 점이 결정적이었습니다.** 목표 0.8248 > 베이스라인 `balanced_accuracy_at_best_cut` 0.8238이므로, `logreg` 랭킹 위에서 작동점만 옮기는 어떤 방법으로도 통과가 불가능했습니다(goal 블록의 `exceeds_ranking_ceiling: true`, `ks_shortfall: 0.002`가 같은 사실을 가리킵니다). 따라서 랭킹 생성기 자체를 바꾸는 것이 필수였고, `hist_gbdt`는 `roc_auc`를 0.9075 → 0.9295로, 랭킹 천장을 0.8238 → 0.8497로 올려 임계값을 천장 아래로 끌어내렸습니다.
- **작동점은 이미 최적에 붙어 있습니다.** `cut_headroom=0.001257`로, `predict()`의 기본 컷이 이 랭킹이 허용하는 최댓값(0.8497)에서 0.0013밖에 떨어져 있지 않습니다. `recall=0.8652` vs `specificity=0.831674`로 recall 쪽으로 아주 약간 기울어 있을 뿐이므로, `class_weight`를 더 만지는 데 예산을 쓰면 얻을 수 있는 최대치가 0.0013입니다 — 측정 잡음(검증 CI 폭 약 0.0156)보다 작습니다. 남은 개선 여지는 전부 **랭킹 축**, 즉 모델 패밀리와 피처 쪽에 있습니다.
- **과적합은 제약 요인이 아니었습니다.** `train_val_gap=0.02043`, `train_balanced_accuracy=0.8689` vs 검증 0.8484로 격차가 작고, `early_stopping=True`가 8.1초 안에 반복 수를 스스로 제한했습니다. 정규화를 더 조이는 방향의 진단 근거는 없습니다.
- **확률값은 확률로 읽을 수 없습니다.** `calibration_error=0.100639` (평균 약 10퍼센트포인트 편차), `brier=0.107308`. 이는 `class_weight='balanced'`로 양성 쪽을 가중해 컷을 옮긴 구성에서 구조적으로 예상되는 결과이고, 목표 지표(`balanced_accuracy`)와 랭킹 지표(`roc_auc`, `pr_auc`)에는 영향이 없습니다. 다만 하위 시스템이 확률을 그대로 쓴다면 이 숫자가 실제 제약입니다.
- **`unsupported_claims: ["feature_engineering"]`은 이번 attempt의 손실이 아닙니다.** iteration 1의 plan 텍스트를 읽으면 해당 문구는 one-hot 인코딩된 범주형 컬럼과 `missing_indicator`/`missing_count`를 **쓰지 않겠다는** 근거를 설명하는 대목에서 나온 것으로, 파생 피처에 의존한 계획이 아니었습니다(substring 검사가 걸린 사례). 따라서 0.8484는 이 executor가 실제로 낸 숫자이며, 피처 엔지니어링 부재는 "실패한 것"이 아니라 "아직 만들지 않은 능력"으로 다음 단계에 속합니다.

기록된 데이터 주의사항은 없으며(`(없음)`), 결측(전체 0.95%, 최악 컬럼 `occupation` 5.75%)은 `impute='none'`으로 트리가 직접 분기해 처리했습니다.

## 다음 단계 제안

1. **작동점(`class_weight`) 튜닝에는 더 이상 예산을 쓰지 마십시오.** `cut_headroom=0.001257`이 그 레버로 얻을 수 있는 전부를 이미 값으로 알려 주고 있고, 이는 검증 CI 폭(약 0.0156)보다 훨씬 작아 개선으로 보고할 수 없는 크기입니다. 남은 4회는 랭킹 축에 배정하는 것이 맞습니다.
2. **랭킹 축에서 패밀리/하이퍼파라미터 탐색을 1~2회만, 기대치를 미리 낮춘 상태로.** 구체적으로 `xgboost`(`impute='none'` 유지, `scale_pos_weight`로 컷 보정)와 `hist_gbdt`의 용량 확대(`max_leaf_nodes` 63, `learning_rate` 0.03 + `max_iter` 상향)가 후보입니다. 단, 가이드에 기록된 측정 크기는 패밀리 교체 0.0022~0.0077 `roc_auc`, 동일 패밀리 내 재튜닝 0.0032이고 `roc_auc` 차이의 식별 해상도가 0.003~0.006이므로, 나오는 차이의 상당 부분은 이 데이터로 구분되지 않을 가능성이 높습니다. 판단은 `balanced_accuracy`가 아니라 `balanced_accuracy_at_best_cut`과 `roc_auc`로, 그리고 CI 겹침 여부로 내려야 합니다.
3. **확률 품질이 필요하면 그것은 이 루프 밖에서 해결해야 합니다.** executor에는 recalibration 레버가 없으므로(`CalibratedClassifierCV`·컷 이동 모두 불가), `calibration_error=0.100639`를 줄이려면 파이프라인 외부의 후처리 단계로 구축해야 합니다. 참고 측정이 필요하면 `class_weight`를 빼고 동일 구성으로 1회 적합해 `brier`/`calibration_error`를 비교할 수 있지만, 그 경우 컷이 소수 클래스 쪽에서 멀어져 `balanced_accuracy`는 떨어질 것을 전제하고 진단용으로만 읽어야 합니다.
4. **파생 피처 능력을 executor에 추가하는 것을 별도 작업으로 제안합니다.** 현재 executor는 컬럼 결합·상호작용·재인코딩을 전혀 하지 못하며(`missing_indicator`/`missing_count`만 예외), 이 데이터에서 `capital-gain`·`capital-loss`(둘 다 high skew), `education-num`×`hours-per-week` 같은 조합은 표현조차 되지 않았습니다. 랭킹 천장을 0.8497에서 더 올리는 가장 가능성 있는 경로는 하이퍼파라미터가 아니라 이쪽이며, 이는 계획으로 얻을 수 없고 능력으로 만들어야 합니다. (`missing_indicator`는 `impute='none'`과 함께 쓰면 비트 단위로 중복임이 확인되어 있으므로 제외.)