# AutoML 최종 보고서 — bank-marketing (`Class`, binary_classification)

## 요약

목표에 도달하지 못했습니다. 목표는 `balanced_accuracy` ≥ 0.8811(maximize)였고, 4회 시도(예산 5회) 중 최고 검증 점수는 iteration 2의 `hist_gbdt`가 기록한 **0.8624 (95% CI 0.8510~0.8737)** 로, 임계값보다 0.0187 낮으며 신뢰구간 전체가 임계값 아래에 있습니다. 같은 모델을 한 번도 사용되지 않은 held-back 테스트 20%에서 채점한 결과는 **0.8628 (95% CI 0.8511~0.8736)** 로, 이것이 이 실행이 실제로 입증한 수치입니다. 루프는 연속 미개선(정체)으로 5회를 다 쓰지 않고 4회에서 조기 종료되었고, critic 진단은 3회 생성되었습니다(logreg 베이스라인 0.6603 대비로는 크게 개선되었으나 목표선에는 미달).

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `learning_rate` 0.06, `max_iter` 500, `max_leaf_nodes` 63, `l2_regularization` 1.0, `early_stopping` false, `class_weight` {0:1, 1:6} | `balanced_accuracy` 0.8266 (CI 0.8135~0.8391), `roc_auc` 0.9308, `recall` 0.7382 / `specificity` 0.9150, `train_val_gap` 0.1509, `cut_headroom` 0.04463 | `overfitting` — train 0.9775 vs val 0.8266. 용량 축소·정규화 강화(31 leaves, l2 5.0, `min_samples_leaf` 40, 낮은 lr·긴 부스팅) 처방 |
| 2 | `hist_gbdt` | `learning_rate` 0.03, `max_iter` 900, `max_leaf_nodes` 31, `max_depth` 6, `min_samples_leaf` 40, `l2_regularization` 10.0, `early_stopping` false, `class_weight` {0:1, 1:7} | **최고** `balanced_accuracy` 0.8624 (CI 0.8510~0.8737), `roc_auc` 0.9351, `recall` 0.8488 / `specificity` 0.8760, `train_val_gap` 0.0584, `cut_headroom` 0.0123, `balanced_accuracy_at_best_cut` 0.8747 | `wrong_model_family` — 과적합은 해소(paired Δ +0.0358, CI +0.0266~+0.0449)되었고 잔여분은 ranking 축. `xgboost`(row/column subsampling, `scale_pos_weight` 7) 로의 계열 교체 처방 |
| 3 | `xgboost` | `n_estimators` 1500, `learning_rate` 0.03, `max_depth` 6, `min_child_weight` 5, `subsample` 0.8, `colsample_bytree` 0.6, `reg_lambda` 5.0, `scale_pos_weight` 7.0, `early_stopping_rounds` 60 | `balanced_accuracy` 0.8423 (CI 0.8287~0.8534), `roc_auc` 0.9316, `balanced_accuracy_at_best_cut` 0.8687, `train_val_gap` 0.1062, `internal_validation.stopped_at_iter` 1500 = `max_iter` 1500 | `wrong_model_family` (자기 처방 반박) — paired Δ −0.0201 (CI −0.0288~−0.0117)로 계열 교체는 기각. early stopping이 발동하지 않아 과적합 재발. iteration 2 구성 복귀 + `missing_count` 1개 열 추가 처방 |
| 4 | `hist_gbdt` | `learning_rate` 0.015, `max_iter` 2500, `max_leaf_nodes` 31, `max_depth` 6, `min_samples_leaf` 40, `l2_regularization` 10.0, `early_stopping` false, `class_weight` {0:1, 1:6.5} | `balanced_accuracy` 0.8565 (CI 0.8445~0.8675), `roc_auc` 0.9338, `balanced_accuracy_at_best_cut` 0.8742, `cut_headroom` 0.0177, `train_val_gap` 0.0716 | 없음 — 평가 직후 루프 종료(정체) |

## 최고 성능 구성

iteration 2, 모델 `hist_gbdt`. 아래는 히스토리에 기록된 **적용된** 값입니다.

**hyperparams**
```
learning_rate       = 0.03
max_iter            = 900
max_leaf_nodes      = 31
max_depth           = 6
min_samples_leaf    = 40
l2_regularization   = 10.0
early_stopping      = false
class_weight        = {"0": 1.0, "1": 7.0}
```
`dropped_hyperparams`: 없음(빈 목록) — 계획한 모든 키가 실제로 적용되었습니다.

**preprocessing (적용값)**
```
impute            = none     # hist_gbdt가 NaN을 직접 분기. 단, 이 데이터에는 NaN이 0건
scale             = false
missing_indicator = false
missing_count     = false
```

**프로토콜**: stratified 3-way split, seed 44, train 60% / validation 20% / test 20%. `train_time_sec` 14.864 (예산 600초).

**점수**

| | validation (선택에 사용) | held-back test (1회 채점) |
|---|---|---|
| `balanced_accuracy` | 0.8624 (CI 0.8510~0.8737) | **0.8628 (CI 0.8511~0.8736)** |

두 값의 차이는 −0.0004이며, 이것이 이 실행의 선택 편향(selection effect)의 크기입니다 — 루프의 모든 비교는 validation 숫자로만 이루어졌습니다. 검증 점수가 테스트 CI 안에 들어 있으므로 이 행들만으로는 그 차이를 0과 구분할 수 없습니다. 어느 쪽으로 읽어도 임계값 0.8811에는 미달합니다.

기타 validation 지표(iteration 2): `f1` 0.6096, `accuracy` 0.8728, `precision` 0.4756, `recall` 0.8488, `specificity` 0.8760, `roc_auc` 0.9351, `pr_auc` / `average_precision` 0.6284, `balanced_accuracy_at_best_cut` 0.8747, `cut_headroom` 0.0123, `brier` 0.0889, `calibration_error` 0.1057, `train_balanced_accuracy` 0.9208, `train_val_gap` 0.0584.

참고: 카드 베이스라인 `logreg`(median impute + standard scale)은 `balanced_accuracy` 0.6603 (CI 0.6465~0.6746)이므로, 이 구성은 베이스라인 대비로는 명확히(구간 비중첩) 개선된 결과입니다.

## 원인 분석

critic 진단은 3회 존재하며, 그 흐름은 일관됩니다: **과적합 → 해소 → 남은 잔여분은 operating point가 아니라 ranking 축**.

- **구간으로 분리되는 차이만 읽으면**: iteration 1 → 2 의 +0.0358 (paired CI +0.0266~+0.0449)와 iteration 2 → 3 의 −0.0201 (paired CI −0.0288~−0.0117)만이 해상도를 넘는 실제 이동입니다. 반면 iteration 2 (0.8624, CI 0.8510~0.8737)와 iteration 4 (0.8565, CI 0.8445~0.8675)는 구간이 크게 겹치므로 **이 데이터로는 구분되지 않습니다** — `learning_rate` 0.03→0.015, `max_iter` 900→2500, `class_weight` 1:7→1:6.5 의 재조정이 무엇을 바꿨다고 서술할 근거가 없습니다. 즉 이 계열 내부 튜닝은 iteration 2에서 이미 정체했습니다.
- **1차 제약은 과적합이었고, 그것은 해결되었습니다.** iteration 1의 `train_val_gap` 0.1509 / `train_balanced_accuracy` 0.9775는 명백한 암기였고, 정규화 처방으로 0.0584 / 0.9208까지 내려갔습니다. 이 축에서는 더 얻을 것이 남아 있지 않습니다.
- **최종 제약은 ranking입니다.** 최고 구성의 `cut_headroom`은 0.0123에 불과하고 `balanced_accuracy_at_best_cut` = 0.8747입니다. 즉 **어떤 결정 규칙(임계값)으로도** 이 순위표에서는 0.8811에 0.0064 모자랍니다. `class_weight` 레버도 이미 교차점 근처입니다: 1:6에서 `recall` 0.7382 < `specificity` 0.9150, 1:7에서 `recall` 0.8488 > `specificity` 0.8760 으로 최적점을 양쪽에서 감싸고 있어 여기서 나올 수 있는 양은 최대 0.012 수준이고 0.0187을 덮지 못합니다.
- **계열 교체 축은 실측으로 기각되었습니다.** `xgboost`는 동일 파이프라인에서 `balanced_accuracy_at_best_cut` 0.8687로 오히려 낮았고(`hist_gbdt` 0.8747), 두 계열이 이 지표에서 만든 폭은 0.0060 — 남은 부족분과 같은 크기입니다. 게다가 iteration 3은 `internal_validation.stopped_at_iter` 1500 = `max_iter` 1500 으로 **early stopping이 전혀 발동하지 않았고** `train_val_gap`이 0.1062로 되돌아갔으므로, 이 한 번의 실행은 `xgboost` 계열 자체를 공정하게 시험한 것이 아니라 설정 문제를 관측한 것입니다.
- **처방과 실행이 어긋난 지점이 하나 있습니다.** iteration 3의 critic은 "iteration 2와 동일 구성 + `missing_count` = true"를 처방했지만, 실제로 실행된 iteration 4는 `missing_count` = false 이고 `learning_rate`/`max_iter`/`class_weight`가 바뀐 재튜닝이었습니다. 따라서 처방된 feature-side 실험은 **시행되지 않았습니다**. 다만 데이터 카드상 `missing.overall_rate` = 0.0 이므로 그 처방 자체가 애초에 성립하지 않았습니다 — `missing_count`는 상수 0 열이 됩니다(이 점은 iteration 2의 계획 텍스트가 이미 정확히 지적하고 있습니다).
- **`unsupported_claims: ["feature_engineering"]`** 는 네 시도 모두에 표시되었으나 이는 계획 산문에 대한 부분 문자열 검사입니다. 확인 가능한 iteration 2의 계획 텍스트는 "caveats의 'unknown'은 여기서는 일반 one-hot level이며 NaN이 아니므로 missingness 열은 상수 0이 되어 아무것도 사지 못한다"고 **제약을 명시적으로 언급했을 뿐**, 그 기능에 의존하지 않았습니다. 따라서 이 플래그 때문에 무엇을 잃었다고 서술할 근거는 없습니다. 파생/재인코딩/열 삭제가 executor의 능력 밖이라는 사실은 "실패한 것"이 아니라 "아직 만들지 않은 것"이며, 아래 제안으로 넘깁니다.
- **목표선 자체의 위치도 기록해 둘 필요가 있습니다.** goal 메타데이터는 `ranking_ceiling` 0.8382, `exceeds_ranking_ceiling` true, `required_ks` 0.7622(베이스라인 `ks` 0.6765, `ks_shortfall` 0.0857)로, 0.8811은 베이스라인 순위 품질로부터 유도된 상한을 이미 넘어서는 목표였습니다. 최고 모델은 `balanced_accuracy_at_best_cut` 0.8747(critic 계산 기준 KS 0.7494)까지 올라와 필요 KS에 0.013 정도 남긴 상태에서 멈췄습니다.
- 부수 관측: 양성 가중치를 키우면서 `calibration_error`가 0.0487(it1) → 0.1057(it2), `brier` 0.0743 → 0.0889로 악화되었습니다. 이 실행에는 재보정 레버가 없어 목표 지표로 다룰 수 없지만, 확률값을 확률로 읽어야 하는 용도라면 그대로 쓰기 어렵다는 사실은 남습니다.

## 다음 단계 제안

1. **'unknown' 표기를 데이터 원본에서 해결하고 카드를 다시 생성**하십시오 (V16 81.8%, V9 28.8%, V4 4.1%, V2 0.6%). 현재 이 값들은 실제 측정치로 집계·학습되어 있고 `missing.overall_rate` = 0.0 이므로, `impute: none`의 NaN 분기와 `missing_count` 열은 **지금은 아무 정보도 담지 못합니다**(상수 0). 이는 계획으로 바꿀 수 없고 카드/원본 쪽 작업입니다. 해결되면 (a) NaN 분기, (b) `missing_count`, (c) `impute` 전략 비교라는 세 레버가 처음으로 시험 가능해집니다. 단, 이 변경이 얼마를 벌어줄지는 실행 지침에서 명시적으로 "확립되지 않았다"고 못박은 부분이므로 기대값을 미리 걸지 마십시오. 특히 V16은 81.8%가 'unknown'이어서, 이 열의 결측이 피험자 상태가 아니라 기록 시점/경로를 가리킬 가능성을 함께 점검해야 합니다 — 그렇다면 무작위 split은 그것을 이득으로 오독합니다.
2. **`xgboost`를 한 번 더, 이번에는 실제로 멈추게 해서 재시험**하십시오. iteration 3은 `stopped_at_iter` 1500 = `max_iter` 1500 으로 조기 종료가 발동하지 않았고 `train_val_gap`이 0.1062로 되돌아갔습니다. `n_estimators`를 400~600으로 낮추고 `learning_rate` 0.03 유지, `reg_lambda`를 5.0 이상으로, `min_child_weight`를 올려 iteration 2 수준의 gap(≈0.06)에서 `balanced_accuracy_at_best_cut`을 다시 읽어야 계열 교체 축이 정말 소진되었는지 판정됩니다. 현재 판정은 "설정 실패가 섞인 1회 관측"입니다.
3. **아직 시도되지 않은 계열(예: `random_forest`)을 동일 파이프라인·동일 불균형 설정(`class_weight` 1:7 근방)에서 1회 측정**해 ranking 축의 계열 간 폭을 넓혀 보십시오. 다만 지금까지 관측된 계열 간 `balanced_accuracy_at_best_cut` 폭은 0.0060(0.8687~0.8747)이고 남은 부족분이 0.0064~0.0187이므로, 이 레버 하나로 목표선을 넘길 확률은 낮게 잡아야 합니다. 예산은 전혀 병목이 아닙니다(최고 구성 14.9초 / 600초).
4. **executor에 없는 능력을 만들거나, 목표선을 재유도하십시오.** `balanced_accuracy` 0.8811은 `ranking_ceiling` 0.8382를 초과하도록 유도된 값이고(`required_ks` 0.7622 vs 베이스라인 0.6765), 결정 규칙으로 도달 가능한 최대치(`balanced_accuracy_at_best_cut` 0.8747)조차 아직 그 아래입니다. 남은 0.0064는 원리적으로 **순위 품질 = 특징 정보량** 문제이므로, (a) 열 파생·재인코딩·열 삭제(현재 CANNOT 목록), (b) cross-validation / out-of-fold 기반의 더 안정적인 모델 선택, (c) 임계값 탐색·확률 재보정 중 최소한 (a)를 파이프라인에 추가하지 않는 한 이 루프 안에서의 추가 반복은 iteration 2/4처럼 구간이 겹치는 재튜닝으로 수렴할 가능성이 큽니다. 대안으로, 실제 운영 요구가 무엇인지 다시 확인해 held-back 0.8628 (CI 0.8511~0.8736)을 수용 가능한 결과로 재평가하는 것도 정당한 선택입니다.