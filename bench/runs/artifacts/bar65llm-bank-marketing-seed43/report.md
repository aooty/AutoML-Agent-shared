# AutoML 최종 리포트 — bank-marketing (`balanced_accuracy`)

## 요약

**목표는 달성하지 못했습니다.** 목표선은 `balanced_accuracy` 0.8816이었고, 5회 시도 중 최고 검증 점수는 iteration 4의 `xgboost` 0.8750 (95% CI 0.8640~0.8851)으로 0.0066 부족했습니다. 루프 종료 후 한 번도 사용되지 않은 held-back 테스트 20%에서 같은 모델은 `balanced_accuracy` 0.8655 (95% CI 0.8545~0.8754)를 기록했으며, 이 값이 이 런이 실제로 입증한 성능입니다. 5회 시도를 모두 소진했고(critic 진단 4회), 베이스라인 `logreg` 0.6616 대비로는 크게 개선되었지만 목표선에는 도달하지 못했습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `learning_rate=0.06`, `max_iter=500`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `class_weight='balanced'`; `impute:none` | `balanced_accuracy` **0.8560** (CI 0.8447~0.8686), `roc_auc` 0.9293, `balanced_accuracy_at_best_cut` 0.8729, `cut_headroom` 0.0169, `train_val_gap` 0.0878 | `wrong_model_family` — 최적 cut으로도 0.8729로 목표 미달, 남은 거리는 ranking 축에 있다고 판단 → `xgboost`로 교체 지시 |
| 2 | `xgboost` | `n_estimators=1200`, `learning_rate=0.04`, `max_depth=7`, `min_child_weight=3`, `subsample=0.8`, `colsample_bytree=0.7`, `reg_lambda=2.0`, `scale_pos_weight=7.5`, `early_stopping_rounds=60`; `impute:none` | `balanced_accuracy` **0.8363** (CI 0.8235~0.8486), `roc_auc` 0.9301, best_cut 0.8763, `cut_headroom` 0.0400, `recall` 0.7580 / `specificity` 0.9146, `train_val_gap` 0.1293 | `hyperparam` — family 교체는 paired Δ −0.0197로 오히려 손실, 손실 대부분은 cut 위치 → `scale_pos_weight` 상향 + 규제 강화 지시 |
| 3 | `xgboost` | `n_estimators=1500`, `learning_rate=0.03`, `max_depth=5`, `min_child_weight=8`, `subsample=0.8`, `colsample_bytree=0.7`, `reg_lambda=5.0`, `reg_alpha=0.5`, `scale_pos_weight=14.0`, `early_stopping_rounds=80`; `impute:none` | `balanced_accuracy` **0.8728** (CI 0.8613~0.8843), `roc_auc` 0.9318, best_cut 0.8754, `cut_headroom` 0.0026, `recall` 0.8875 / `specificity` 0.8581, `train_val_gap` 0.0543 | `hyperparam` — operating point 레버 소진(headroom 0.0026), 미사용 레버인 preprocessing 블록 probe 지시(`impute:median` + `missing_count`, `scale_pos_weight` 12.0) |
| 4 | `xgboost` | iteration 3과 동일 + `scale_pos_weight=12.0`; `impute:median`, `missing_count=true` | `balanced_accuracy` **0.8750** (CI 0.8640~0.8851) ← 최고, `roc_auc` 0.9321, best_cut 0.8775, `cut_headroom` 0.0025, `train_val_gap` 0.0523 | `wrong_model_family` — iteration 3 대비 paired Δ +0.0022 (CI −0.0014~+0.0059, P=0.863)로 구분 불가. 남은 1회를 ranking 축 최대 이동에 쓰라며 `hist_gbdt` 대용량 설정 지시 |
| 5 | `hist_gbdt` | `learning_rate=0.05`, `max_iter=800`, `max_leaf_nodes=63`, `min_samples_leaf=10`, `l2_regularization=0.5`, `early_stopping=false`, `class_weight={0:1.0, 1:12.0}`; `impute:median`, `missing_count=true` | `balanced_accuracy` **0.8367** (CI 0.8224~0.8497), `roc_auc` 0.9264, best_cut **0.8679**(전 시도 중 최저), `cut_headroom` 0.0312, `train_val_gap` 0.1484 | 없음 (평가 직후 루프 종료) |

## 최고 성능 구성

**iteration 4 / `xgboost`** — 아래 값은 executor가 실제로 만든 estimator에서 읽은 적용값입니다 (`dropped_hyperparams`는 비어 있음).

```
model: xgboost
hyperparams:
  n_estimators: 1500
  learning_rate: 0.03
  max_depth: 5
  min_child_weight: 8
  subsample: 0.8
  colsample_bytree: 0.7
  reg_lambda: 5.0
  reg_alpha: 0.5
  scale_pos_weight: 12.0
  early_stopping_rounds: 80
preprocessing (applied):
  impute: median
  scale: false
  missing_indicator: false
  missing_count: true
internal_validation: fit_rows 24413 / held_out_rows 2713 (validation_fraction 0.1),
                     stopped_at_iter 1487 of max_iter 1500
train_time_sec: 12.238
```

| 측정 슬라이스 | `balanced_accuracy` | 95% CI |
|---|---|---|
| 검증 20% (루프가 선택에 사용한 값) | 0.8750 | 0.8640~0.8851 |
| **held-back 테스트 20% (선택에 한 번도 쓰이지 않음)** | **0.8655** | 0.8545~0.8754 |

두 값의 차이 **+0.0096(검증이 높음)** 은 선택 편향의 크기입니다 — 이 런은 검증 숫자를 보고 최고 시도를 골랐기 때문입니다. 다만 검증 점수 0.8750은 테스트 CI(0.8545~0.8754)의 상단 근처에 걸쳐 있어, 이 행들만으로는 그 차이를 0과 구분할 수 없습니다. 어느 쪽이든 목표선 0.8816은 테스트 CI 밖에 있습니다.

검증 슬라이스의 그 외 지표: `roc_auc` 0.9321, `pr_auc` / `average_precision` 0.6194, `f1` 0.6079, `accuracy` 0.8662, `precision` 0.4625, `recall` 0.8866, `specificity` 0.8635, `balanced_accuracy_at_best_cut` 0.8775, `cut_headroom` 0.0025, `brier` 0.0989, `calibration_error` 0.1218, `train_balanced_accuracy` 0.9273, `train_val_gap` 0.0523.

## 원인 분석

**1) 구분 가능한 차이와 구분 불가능한 차이를 먼저 갈라야 합니다.** iteration 3(0.8728, CI 0.8613~0.8843)과 iteration 4(0.8750, CI 0.8640~0.8851)는 구간이 거의 완전히 겹치고 paired Δ도 +0.0022 (CI −0.0014~+0.0059, P=0.863)이므로 **이 데이터가 분리하지 못하는 두 시도**입니다. 즉 "median imputation + `missing_count` 추가가 점수를 올렸다"는 서술은 근거가 없습니다. 반대로 크기가 확실한 이동은 두 개뿐입니다: iteration 1→2의 −0.0197(CI −0.0296~−0.0097)과 iteration 2→3의 +0.0168(CI +0.0095~+0.0237). 두 이동 모두 ranking이 아니라 **operating point**의 이동이었습니다 — `recall`이 0.8289 → 0.7580 → 0.8875로 움직이는 동안 `roc_auc`는 0.9293 → 0.9301 → 0.9318, 즉 CI 폭보다 작은 범위에서만 변했습니다.

**2) 진짜 병목은 ranking 축이고, 그 축은 5회 내내 거의 움직이지 않았습니다.** 다섯 시도의 `balanced_accuracy_at_best_cut`은 0.8729 / 0.8763 / 0.8754 / 0.8775 / 0.8679 — 폭 0.0096이고 **전부 목표선 0.8816 아래**입니다. 즉 어떤 시도에서도 완벽한 임계값을 골랐다 해도 목표에 닿지 못했습니다. 최고 시도의 `cut_headroom`은 0.0025로, 클래스 가중치나 임계값으로 얻을 수 있는 최대치가 남은 거리 0.0066의 38%에 불과합니다. KS는 0.7550으로 목표가 요구하는 0.7632에 미달합니다.

**3) critic의 4개 진단은 같은 벽에 두 번씩 부딪힌 패턴입니다.** iteration 1과 4에서 `wrong_model_family`, 2와 3에서 `hyperparam`. `hyperparam` 두 번은 성공적으로 작동했습니다 — `scale_pos_weight` 7.5와 14.0이 recall/specificity 교차점을 bracket하면서 `cut_headroom`을 0.0400 → 0.0026으로 소진시켰고, 그것이 이 런의 실질적 이득 전부입니다. `wrong_model_family` 두 번은 모두 대가만 치렀습니다: iteration 2는 −0.0197, iteration 5는 0.8367로 best_cut 기준 전 시도 중 최저(0.8679)이자 `train_val_gap` 0.1484로 최대 과적합. 두 tree family 사이 ranking 폭(best_cut 0.8679~0.8775)이 남은 거리와 비슷한 크기라는 것이 두 번째 family 교체의 근거였지만, 실제로 얻은 것은 음의 이동이었습니다. **executor가 제공하는 레버(family 교체 · family 내 하이퍼파라미터 · 클래스 가중치 · imputation/missingness 컬럼) 중 ranking을 목표선까지 밀어올린 것은 하나도 없었습니다.**

**4) preprocessing probe는 구조적으로 빈 probe였습니다.** 카드의 `missing.overall_rate`는 0.0이고 `columns_with_missing`은 0입니다. 'unknown'은 NaN이 아니라 하나의 범주 수준으로 one-hot 인코딩되어 들어갑니다(V16 81.8%, V9 28.8%, V4 4.1%, V2 0.6%). 따라서 iteration 4의 `impute:median`은 채울 값이 없었고 `missing_count`는 전 행 동일한 상수 컬럼이었을 가능성이 높습니다 — critic이 "카드의 unknown 비율이 곧 결측 정보"라고 읽은 것은 카드의 결측 집계와 어긋납니다. iteration 3→4가 구분 불가능했던 것도 이 해석과 일관됩니다.

**5) `unsupported_claims`에 대하여.** iteration 1~3의 계획에 `feature_engineering`이 플래그되었으나, 이 기록에는 해당 iteration의 계획 원문이 없어 계획이 그 기능에 **의존**했는지 단순히 **언급**했는지 확인할 수 없습니다. 따라서 이 플래그를 성능 손실의 원인으로 돌리지 않습니다. 확실한 사실은 executor가 파생·상호작용·재인코딩·컬럼 삭제를 수행하지 않는다는 것뿐이며, 이는 아래 제안에서 "만들어야 할 것"으로 다룹니다.

**6) 목표선 자체의 근거.** 목표 0.8816은 derived이며 참조 정보에 `exceeds_ranking_ceiling: true`, `ranking_ceiling: 0.8415`, `ks_shortfall: 0.0802`가 기록되어 있습니다. 즉 이 목표선은 베이스라인의 ranking이 지지하던 범위를 처음부터 넘어서 설정되었고, 이번 런은 `passable_margin: 0.531`에 해당하는 완화선 기준의 통과 가능성 쪽에 더 가까운 결과를 냈습니다.

## 다음 단계 제안

1. **먼저 caveat를 해소하십시오: 'unknown' 마커를 파일 단계에서 NaN으로 변환한 뒤 재프로파일.** V16(81.8%), V9(28.8%), V4(4.1%), V2(0.6%)의 'unknown'은 현재 실측값으로 집계·학습되고 있어 이 열들의 `target_corr`·`outlier_rate`와 베이스라인 점수 모두 그 가정 위에 있습니다. executor는 열을 변형할 수 없으므로 이 변환은 카드 밖(원본 CSV)에서 이루어져야 합니다. 변환이 끝나면 `impute:none`(NaN 분기), `missing_indicator`, `missing_count`가 처음으로 **실제 내용이 있는** 레버가 되고, iteration 4가 무엇도 테스트하지 못했던 자리를 메울 수 있습니다. 단, executor 문서가 명시하듯 이 레버의 크기는 확립되어 있지 않으므로(무작위 분할 0.0097 vs 연속 분할 +0.0011, 구간이 0을 포함) 예상 이득을 숫자로 약속하지 말고 probe로 다루십시오.
2. **ranking 축을 여는 유일한 남은 수단은 executor 밖에서 만든 파생 컬럼입니다.** 5회 모두에서 `balanced_accuracy_at_best_cut`이 0.8679~0.8775에 갇혀 목표선 아래였고, `cut_headroom`은 최고 시도에서 0.0025입니다. 즉 임계값·클래스 가중치·family 교체·같은 family 내 재튜닝으로는 구조적으로 닿을 수 없습니다(family 교체는 이 런에서 두 번 시도해 −0.0197과 최저 best_cut 0.8679를 기록). 파생 특징(예: 연속형 V6/V12/V13/V14/V15의 로그·구간화, 범주×연속 상호작용)은 **CSV에 미리 계산해 넣어 카드의 열로 들어가게** 해야 합니다 — 계획서에 적어도 executor는 만들지 않습니다.
3. **V12의 사용 가능성 감사.** 카드에서 `target_corr: strong`은 V12 하나뿐이고 이 모델의 ranking은 사실상 그 열에 걸려 있습니다. 이 열이 예측 시점에 실제로 관측 가능한 값인지(사후에만 알 수 있는 값이 아닌지)를 원본 소유자와 확인하십시오. 확인 결과가 부정적이면 현재 0.8655라는 테스트 점수 자체가 운용 가능한 수치가 아니게 되며, 확인이 긍정적이면 이 열 중심의 파생(제안 2)이 가장 수익률 높은 방향이 됩니다.
4. **목표선 0.8816의 재검토, 그리고 다음 런의 예산 배분.** 참조 정보가 `exceeds_ranking_ceiling: true`, `required_ks: 0.7632` (달성 0.7550), `ks_shortfall: 0.0802`를 기록하고 있으므로, 특징을 바꾸지 않은 채 반복 횟수만 늘리는 것은 같은 0.86~0.88 구간을 재표집하는 데 그칠 것입니다. 다음 런에서는 (a) 제안 1·2로 입력 행렬을 먼저 바꾸고, (b) `scale_pos_weight`는 이미 확정된 12~14 구간에 고정한 채 iteration당 레버를 하나만 움직여 귀속 가능성을 지키고, (c) family 교체에는 더 이상 iteration을 쓰지 마십시오 — 이 파일에서 두 번 측정해 두 번 손실이었습니다. 아울러 최고 시도의 `calibration_error` 0.1218 / `brier` 0.0989는 확률값이 평균 12%p 어긋나 있음을 뜻하므로, 산출 확률을 그대로 의사결정에 쓰려는 계획은 이 런의 결과로는 지지되지 않습니다(executor에는 재보정 레버가 없습니다).