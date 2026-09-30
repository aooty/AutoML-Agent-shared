# AutoML 최종 리포트 — `adult` (binary_classification, balanced_accuracy)

## 요약

**목표는 달성되지 못했습니다.** 목표 임계값은 balanced_accuracy 0.9194였고, 5회 시도 중 가장 좋은 검증 점수는 iteration 3의 `hist_gbdt`가 기록한 **0.8470 (95% CI 0.8389~0.8544)** 으로 약 0.072 부족했습니다. 동일 모델을 루프 내내 한 번도 사용하지 않은 held-back 테스트 20%에서 재측정한 값은 **0.8484 (95% CI 0.8410~0.8561)** 입니다. 5회 시도를 모두 소진(`max_iterations`)하고 종료했으며, baseline(logreg, 0.7698)은 크게 상회했지만 목표선은 넘지 못했습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter` 400, `learning_rate` 0.06, `max_leaf_nodes` 31, `min_samples_leaf` 20, `l2_regularization` 1.0, `class_weight` 'balanced', `early_stopping` false | balanced_accuracy **0.8442** (CI 0.8362~0.8517), roc_auc 0.9272, best_cut 0.8451, `cut_headroom` 0.000936, `train_val_gap` 0.0444 | `wrong_model_family` — 컷은 이미 소진, 남은 격차는 ranking 축에 있다고 판단해 xgboost 전환 지시 |
| 2 | `xgboost` | `n_estimators` 900, `learning_rate` 0.05, `max_depth` 8, `min_child_weight` 5, `subsample` 0.8, `colsample_bytree` 0.8, `reg_lambda` 1.0, `scale_pos_weight` 3.18 | balanced_accuracy **0.8354** (CI 0.8268~0.8434), roc_auc 0.9215, `train_val_gap` 0.0999 — iteration 1 대비 Δ -0.0088 (CI -0.0147~-0.0033) 유의한 퇴행 | `hyperparam` — family swap이 -0.0088을 냈으므로 family가 아니라 설정 문제로 재진단, 용량을 낮추고 정규화를 올릴 것 |
| 3 | `hist_gbdt` | `max_iter` 800, `learning_rate` 0.04, `max_leaf_nodes` 15, `max_depth` 6, `min_samples_leaf` 40, `l2_regularization` 3.0, `class_weight` 'balanced', `early_stopping` false | balanced_accuracy **0.8470** (CI 0.8389~0.8544) — 런 최고, roc_auc 0.9291, best_cut 0.8479, `cut_headroom` 0.000921, `train_val_gap` 0.0180. iteration 1 대비 Δ +0.0028 (CI -0.0011~+0.0067) — 구분되지 않음 | `underfitting` — gap 0.0180, train 0.8650로 train/val이 함께 낮으므로 용량을 넓히라고 지시 |
| 4 | `hist_gbdt` | `max_iter` 1000, `learning_rate` 0.04, `max_leaf_nodes` 63, `min_samples_leaf` 10, `l2_regularization` 0.1, `class_weight` 'balanced' | balanced_accuracy **0.8280** (CI 0.8197~0.8362), roc_auc 0.9206, `train_val_gap` 0.1253 — iteration 3 대비 Δ -0.0189 (CI -0.0257~-0.0118) 유의한 퇴행 | `overfitting` — 직전 underfitting 처방이 반증됨, iteration 3 설정으로 되돌려 최고 ranking을 확보할 것 |
| 5 | `hist_gbdt` | `max_iter` 1500, `learning_rate` 0.025, `max_leaf_nodes` 15, `max_depth` 6, `min_samples_leaf` 40, `l2_regularization` 1.5, `max_features` 0.7, `max_bins` 255, `class_weight` {"0":1.0,"1":2.6} | balanced_accuracy **0.8448** (CI 0.8370~0.8531), roc_auc 0.9287, best_cut 0.8475, `cut_headroom` 0.002678, `train_val_gap` 0.0223, calibration_error 0.0880 | (없음 — 마지막 시도, 평가 직후 루프 종료) |

모든 시도에서 `dropped_hyperparams`는 비어 있었으므로, 표에 적힌 설정은 전부 실제로 적용되었습니다.

## 최고 성능 구성

**iteration 3 / `hist_gbdt`** (아래 값은 해당 attempt의 기록에서 읽은 실제 적용값입니다)

```
model: hist_gbdt
hyperparams:
  max_iter: 800
  learning_rate: 0.04
  max_leaf_nodes: 15
  max_depth: 6
  min_samples_leaf: 40
  l2_regularization: 3.0
  class_weight: "balanced"
  early_stopping: false
preprocessing (applied):
  impute: none          # 트리가 NaN 자체를 분기 — 카드의 median/scale 요청은 적용되지 않음
  scale: false
  missing_indicator: false
  missing_count: false
train_time_sec: 18.941
```

| 지표 | 검증(20%) | held-back 테스트(20%) |
|---|---|---|
| balanced_accuracy | **0.8470** (CI 0.8389~0.8544) | **0.8484** (CI 0.8410~0.8561) |
| 목표 임계값 | 0.9194 | 0.9194 |

기타 검증 지표: accuracy 0.8335, f1 0.7149, precision 0.6053, recall 0.8729, specificity 0.8210, roc_auc 0.9291, pr_auc / average_precision 0.8321, `balanced_accuracy_at_best_cut` 0.8479, `cut_headroom` 0.000921, brier 0.1103, calibration_error 0.1096, `train_balanced_accuracy` 0.8650, `train_val_gap` 0.0180.

검증과 테스트의 차이는 0.0014이며, 이 크기가 이 런의 **선택 편향(selection effect)** 규모입니다 — 최종 모델은 검증 점수만 보고 골라졌기 때문입니다. 다만 검증 점수 0.8470은 테스트 CI(0.8410~0.8561) 안에 들어 있어, 이 행들로는 그 차이를 0과 구분할 수 없습니다. 어느 쪽 숫자를 쓰더라도 목표 0.9194에는 약 0.071 부족합니다.

## 원인 분석

critic 진단은 4회 존재하며, 그 흐름은 "family → hyperparam → underfitting → overfitting"으로 한 바퀴 돌았습니다. 여기서 실제로 데이터가 분간해 주는 사실만 정리하면 다음과 같습니다.

**1) 성능은 ~0.845에서 평평합니다.** iteration 1(0.8442, CI 0.8362~0.8517), iteration 3(0.8470, CI 0.8389~0.8544), iteration 5(0.8448, CI 0.8370~0.8531)의 구간은 서로 크게 겹칩니다 — 이 데이터로는 **구분되지 않는 세 시도**이며, iteration 3이 "최고"인 것은 순위상 그렇다는 뜻일 뿐 iteration 1/5보다 좋다고 보고할 수 있는 차이가 아닙니다. critic이 iteration 3에서 스스로 밝힌 대로 Δ +0.0028 (CI -0.0011~+0.0067)은 측정이 해상하지 못하는 움직임입니다.

**2) 해상되는 차이는 두 번의 퇴행뿐이고, 둘 다 "용량 증가"였습니다.** iteration 2(xgboost `max_depth` 8, Δ -0.0088, gap 0.0999)와 iteration 4(`max_leaf_nodes` 63 / `min_samples_leaf` 10 / `l2` 0.1, Δ -0.0189, gap 0.1253)는 구간이 겹치지 않는 실질 퇴행이며, 두 경우 모두 train balanced_accuracy는 각각 0.9353·0.9534로 올라가고 val은 내려갔습니다. 즉 이 14개 열에서 per-tree 용량을 키우면 ordering이 아니라 variance만 삽니다. iteration 3의 `underfitting` 진단은 gap 0.0180만 보고 내려졌지만 iteration 4가 그것을 명확히 반증했고, critic 스스로 다음 회차에 이를 인정했습니다.

**3) 남은 격차는 operating point가 아니라 ranking 축에 있습니다.** 최고 구성의 `cut_headroom`은 0.000921 — 이 모델의 ranking을 어떤 컷으로 자르더라도 얻을 수 있는 최대치는 `balanced_accuracy_at_best_cut` 0.8479이고, 목표까지 남은 0.0724 중 약 1%만 컷으로 커버됩니다. recall 0.8729 vs specificity 0.8210(iteration 3), 0.8442 vs 0.8454(iteration 5)로 두 방향이 이미 균형점 근처에 있어, `class_weight`를 더 손대는 것으로는 측정 가능한 이득이 나오지 않습니다. 목표 0.9194는 KS 0.8388을 요구하는데 이 런에서 달성된 KS는 최대 0.6958이고, goal 정의 자체가 `exceeds_ranking_ceiling: true`(baseline ranking 상한 0.8266 초과)로 기록되어 있습니다.

**4) 실행기 능력의 경계가 곧 이 런의 상한입니다.** 사용 가능한 레버는 모델 family·하이퍼파라미터·불균형 가중치·결측 처리뿐이며, feature 파생/조합, 앙상블 결합, threshold 탐색, calibration, cross-validation은 모두 불가능합니다. 이 런에서 family swap이 움직인 폭은 `balanced_accuracy_at_best_cut` 기준 0.8347~0.8479(폭 0.0132)에 불과해, 필요한 0.07 규모와 자릿수가 다릅니다. iteration 1의 plan에는 `unsupported_claims: ["feature_engineering"]`이 기록되어 있으나 해당 plan 본문이 기록에 남아 있지 않아 **그 계획이 feature 파생에 의존했는지, 단지 제약을 언급했는지는 판정할 수 없습니다** — 따라서 이를 실패 원인으로 귀속하지 않고, 아래 "다음 단계"에서 새로 만들어야 할 능력으로 다룹니다.

요약하면, 한계는 "잘못된 family"나 "잘못된 정규화"가 아니라 **주어진 14개 열과 실행기 레버 안에서 이 문제의 ranking 품질이 roc_auc ≈ 0.929 / KS ≈ 0.696에서 포화한다는 점**이며, 목표선은 그 포화점보다 위에 설정되어 있었습니다.

## 다음 단계 제안

`Data caveats`에 기록된 주의사항은 없으므로 아래 제안을 무효화하는 열/분할 이슈는 없습니다. 다만 실행기의 "will not do" 목록이 그대로 제약입니다.

1. **목표 임계값의 타당성을 먼저 재검토하십시오 (최우선).** 0.9194는 baseline 0.7698에 margin 0.65를 적용해 유도된 값이고, goal 레코드 자체가 `ranking_ceiling` 0.8266을 초과했다고 표시하며 `required_ks` 0.8388 대 `ks` 0.6531을 기록합니다. 이 런의 최고 모델도 KS 0.6958, 최적 컷 상한 0.8479입니다. 즉 **현재 feature 집합에서는 어떤 모델·어떤 컷으로도 0.9194에 도달할 수 없습니다.** 목표를 (예: 최적 컷 상한 근처인) 달성 가능한 수준으로 재유도하거나, 아래 2번처럼 입력 정보를 늘린 뒤 다시 목표를 세워야 합니다.

2. **유일하게 자릿수가 맞는 레버는 feature 파생이며, 이는 루프 밖에서(카드/CSV 단계에서) 만들어야 합니다.** 실행기는 열 조합·비율·상호작용·재인코딩을 일절 하지 못하므로, `capital-gain`/`capital-loss`(둘 다 skew high)의 0 여부 플래그와 구간화, `education-num` × `marital-status`/`occupation`류의 상호작용, `hours-per-week`(outlier_rate 0.2763) 구간화 같은 파생 열을 **데이터셋 자체에 추가한 뒤 새 카드로 루프를 다시 돌리는 것**이 필요합니다. 근거: family swap(0.0022~0.0098)과 family 내 튜닝(이 런에서는 대부분 하방)이 모두 0.01 미만에 머물렀고, 남은 격차는 0.07이기 때문입니다.

3. **hist_gbdt 용량/정규화 재튜닝에는 더 이상 예산을 쓰지 마십시오.** iteration 1/3/5는 이 검증 슬라이스로 구분되지 않고, 용량을 키운 iteration 2/4는 유의하게 나빠졌습니다. 또한 단일 20% 검증 슬라이스의 폭이 ±0.008 수준이고 roc_auc의 paired 해상도가 0.003~0.006이므로, 이보다 작은 개선은 원리적으로 측정할 수 없습니다. 튜닝을 계속하려면 먼저 **평가 프로토콜을 키우는 것(다른 seed로 동일 구성 재측정 등, 이는 plan이 아니라 런 설정 변경 사항)** 이 선행되어야 하며, 그것이 확보되지 않으면 미세 튜닝 결과는 리샘플 노이즈와 구별되지 않습니다.

4. **확률값을 실제 의사결정에 쓸 계획이라면 calibration을 루프 밖에서 처리하고, 구성 선택도 그 기준으로 다시 보십시오.** 최고 구성(iteration 3)은 calibration_error 0.1096 / brier 0.1103으로 평균 약 11%p 어긋나 있고, 통계적으로 구분되지 않는 iteration 5는 같은 balanced_accuracy 수준(0.8448)에서 calibration_error 0.0880 / brier 0.1028로 더 낫습니다. 실행기는 recalibration을 제공하지 않으므로, 확률 보정은 별도 후처리 단계로 만들어야 합니다(balanced_accuracy 자체는 이로 인해 개선되지 않는다는 점을 명시합니다).

> 참고: `missing_indicator`는 `impute: none`(이 런의 실제 전처리)과 함께 쓰면 예측이 비트 단위로 동일해지는 것이 이미 문서화되어 있고, `missing_count`도 참조 측정에서 0.0000~0.0003에 머물렀습니다. 전체 결측률 0.0095(최악 열 0.0575)인 이 데이터에서 결측 관련 열에 시도를 쓰는 것은 권하지 않습니다.