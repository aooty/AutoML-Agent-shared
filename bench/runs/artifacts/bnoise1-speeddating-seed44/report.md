# AutoML 실행 보고서 — `speeddating` (target: `match`)

## 요약

목표는 달성하지 못했습니다. 목표 지표 `balanced_accuracy`의 기준선(threshold)은 0.8882였으나, 5회 예산 중 4회를 사용한 뒤 정체(stalled)로 조기 종료되었고 최고 검증 점수는 iteration 2의 `hist_gbdt`가 기록한 0.7778 (95% CI 0.7510~0.8021)로 기준에 0.1104 미달했습니다. 한 번도 사용되지 않은 최종 테스트 20%에서 같은 모델은 `balanced_accuracy` = 0.7937 (95% CI 0.7663~0.8218)로, 이 역시 기준선에 크게 미달합니다. 다만 baseline `logreg`의 0.6807 (CI 0.654~0.7099)은 명확히 상회했습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `min_samples_leaf=20`, `class_weight='balanced'`, `early_stopping=False`; preprocessing `impute='none'`, `scale=False` | `balanced_accuracy`=0.7248 (CI 0.6969~0.7527), `roc_auc`=0.8598, `pr_auc`=0.6049, `balanced_accuracy_at_best_cut`=0.7922, `cut_headroom`=0.0674, `train_val_gap`=0.2752 (train BA=1.0) | `overfitting` — 400 iter·31 leaves·min_samples_leaf 20이 5,026 train 행을 전부 암기. 용량 축소·정규화 강화 지시 |
| 2 | `hist_gbdt` | `max_iter=600`, `learning_rate=0.03`, `max_leaf_nodes=8`, `max_depth=4`, `min_samples_leaf=60`, `l2_regularization=20.0`, `max_features=0.6`, `class_weight='balanced'`; preprocessing `impute='none'`, `scale=False` | **최고** `balanced_accuracy`=0.7778 (CI 0.7510~0.8021), `roc_auc`=0.8604, `pr_auc`=0.5925, `balanced_accuracy_at_best_cut`=0.7891, `cut_headroom`=0.0113, `train_val_gap`=0.1036 | `wrong_model_family` — 정규화는 +0.0529를 벌었으나 전량이 operating point 이동(recall 0.5254→0.7355, specificity 0.9243→0.8200)에서 나왔고 ranking은 정체. `xgboost`로 family 교체 지시 |
| 3 | `extra_trees` | `n_estimators=800`, `min_samples_leaf=5`, `class_weight='balanced'`; preprocessing `impute='median'`, `scale=False`, `missing_count=True` | `balanced_accuracy`=0.7409 (CI 0.7155~0.7729), `roc_auc`=0.8474, `pr_auc`=0.5481, `balanced_accuracy_at_best_cut`=0.7715, `cut_headroom`=0.0306, `train_val_gap`=0.2434 | `hyperparam` — 지시(`xgboost`)와 다른 family가 실행되었고 preprocessing도 동시에 변경되어 Δ의 귀속 불가. iteration 2 구성으로 복귀해 `class_weight`만 `{"0":1,"1":6}`로 조정 지시 |
| 4 | `hist_gbdt` | iteration 2와 동일, `class_weight={"0":1.0,"1":6.5}` | `balanced_accuracy`=0.7754 (CI 0.7508~0.8014), `roc_auc`=0.8589, `pr_auc`=0.5948, `balanced_accuracy_at_best_cut`=0.7867, `cut_headroom`=0.0113 | (루프 종료 직후로 진단 없음) |

## 최고 성능 구성

iteration 2, `hist_gbdt`. 아래 값은 히스토리에 기록된 **적용된** `hyperparams` / `preprocessing`입니다.

```
model: hist_gbdt
hyperparams:
  max_iter: 600
  learning_rate: 0.03
  max_leaf_nodes: 8
  max_depth: 4
  min_samples_leaf: 60
  l2_regularization: 20.0
  max_features: 0.6
  class_weight: "balanced"
  early_stopping: false
preprocessing:
  impute: "none"        # NaN을 트리가 직접 분기
  scale: false
  missing_indicator: false
  missing_count: false
dropped_hyperparams: []   # 거부된 키 없음
train_time_sec: 8.35
```

검증/테스트 점수:

| 구분 | balanced_accuracy | 95% CI |
|---|---|---|
| validation 20% (선택에 사용) | 0.7778 | 0.7510~0.8021 |
| **held-back test 20% (한 번만 채점)** | **0.7937** | 0.7663~0.8218 |

두 값의 차이는 **-0.0159**(검증 대비 테스트가 더 높음)이며, 이 차이가 이 실행의 선택 편향(selection effect) 크기입니다 — 루프의 모든 선택은 검증 숫자로 이루어졌고, 이 모델에 대해 실제로 입증된 값은 테스트 쪽입니다. 검증 점수가 테스트 CI 안에 들어 있으므로 이 행들로는 그 차이가 0과 구분되지 않습니다. 어느 쪽 숫자로 보아도 목표 0.8882에는 0.09~0.11 미달입니다.

validation 부가 지표 (iteration 2): `roc_auc`=0.8604, `pr_auc`/`average_precision`=0.5925, `f1`=0.5554, `accuracy`=0.8061, `precision`=0.4462, `recall`=0.7355, `specificity`=0.8200, `balanced_accuracy_at_best_cut`=0.7891, `cut_headroom`=0.0113, `brier`=0.1315, `calibration_error`=0.1478, `train_balanced_accuracy`=0.8813, `train_val_gap`=0.1036.

프로토콜: stratified 3-way split, seed 44, train 0.6 / val 0.2 / test 0.2 — 카드의 baseline과 동일한 split.

## 원인 분석

critic 진단은 3건(iteration 1·2·3)이 존재하며, 그 패턴은 일관되게 **ranking 축의 한계**를 가리킵니다.

- **iteration 1 → 2 는 실제 개선이지만, 개선의 성격이 문제였습니다.** paired Δ = +0.0529 (CI +0.0290~+0.0771)로 구간이 0을 넘지 않는 해소된 이득입니다. 그러나 그 이득은 전부 operating point에서 왔습니다: `roc_auc` 0.8598→0.8604, `pr_auc` 0.6049→0.5925(하락), `balanced_accuracy_at_best_cut` 0.7922→0.7891(하락). 즉 정규화는 암기(`train_val_gap` 0.2752→0.1036, train BA 1.0→0.8813)를 없앴을 뿐 순위 품질을 올리지 못했고, 올라간 점수는 `class_weight='balanced'`가 컷을 균형점 쪽으로 옮긴 결과입니다.
- **operating point 레버는 이미 소진되었습니다.** iteration 2의 `cut_headroom`은 0.0113으로, 남은 거리 0.1104의 약 10%에 불과합니다. recall 0.7355 vs specificity 0.8200으로 거의 교차 상태입니다. iteration 4가 이를 직접 확인해 줍니다: `class_weight`를 `{"0":1,"1":6.5}`로 더 밀어 recall 0.7609 / specificity 0.7900이 되었으나 `balanced_accuracy`는 0.7754로, iteration 2(0.7778)와 **CI가 거의 완전히 겹칩니다**(0.7508~0.8014 vs 0.7510~0.8021). 이 데이터로는 두 시도를 구분할 수 없으므로 -0.0024를 "악화"로 서술할 근거는 없습니다 — 읽을 수 있는 결론은 "이 레버로는 더 살 것이 없다"는 것뿐입니다.
- **family 교체는 검증되지 않았습니다.** iteration 2의 critic은 `xgboost`(`impute='none'`을 받는 유일한 다른 family, 따라서 preprocessing 고정 상태에서 family 단독 Δ 측정 가능)를 지시했으나, 실제로 실행된 것은 `extra_trees` + `impute='median'` + `missing_count=True`였습니다. 결과 Δ = -0.0369 (CI -0.0583~-0.0118)은 해소된 하락이지만 **두 개의 레버가 한 행에서 동시에 움직였으므로 어느 쪽 탓인지 귀속할 수 없습니다.** 결과적으로 이 실행은 "gbdt 아닌 family가 더 나은 ranking을 주는가"를 한 번도 깨끗하게 시험하지 못했습니다.
- **남은 거리의 대부분은 ranking에 있습니다.** 시도된 두 family의 `balanced_accuracy_at_best_cut`은 0.7715~0.7922, 폭 0.0207에 불과합니다. 반면 최선의 컷(0.7891)에서 목표(0.8882)까지는 0.0991입니다. 목표가 요구하는 KS는 0.7764인데 baseline KS는 0.5297입니다. 즉 어떤 컷 조정으로도 닿을 수 없는 거리이며, 카드 자체도 `exceeds_ranking_ceiling: true`, `ks_shortfall: 0.2467`로 이 목표가 baseline의 ranking 천장을 넘어선 값임을 기록하고 있습니다.
- **자원·구성 제약은 원인이 아닙니다.** 모든 시도에서 `dropped_hyperparams`는 비어 있고, 최대 학습 시간은 11.9초(예산 600초)였으며, 실패 status도 없습니다.
- `unsupported_claims: ["feature_engineering"]`이 iteration 1~3에 기록되어 있습니다. 이는 계획 산문에 대한 부분 문자열 검사이며, 실행된 구성 자체는 executor가 지원하는 범위 안에 있었습니다(파생 피처 없이 카드의 컬럼만 사용). 따라서 이를 "손실"로 귀속하지 않고, 아래 제안에서 "만들어야 할 것"으로 다룹니다.

요약하면, 이 실행이 보여준 것은 **컷 위치는 이미 최적에 가깝고, 남은 0.09~0.11은 순위 품질 자체에 있으며, executor가 컬럼을 조합할 수 없는 상태에서 hyperparameter·class_weight만으로는 메울 수 없다**는 것입니다. 목표값 0.8882 자체가 baseline 대비 파생된(derived) 공격적 수치라는 점도 함께 읽어야 합니다.

## 다음 단계 제안

Data caveats 섹션에 기록된 주의사항은 없으므로 컬럼·split을 신뢰할 수 없다는 제약은 없습니다. 아래 제안은 모두 executor의 CAN 목록 안에서 실행 가능하거나, 불가능한 경우 그 사실을 명시했습니다.

1. **iteration 2의 critic이 지시했으나 실행되지 않은 `xgboost` 시험을 먼저 소진할 것.** `hist_gbdt`와 동일한 preprocessing(`impute='none'`, `scale=False`, missingness 컬럼 없음)으로 `n_estimators=900`, `learning_rate=0.03`, `max_depth=5`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.6`, `reg_lambda=5.0`, `reg_alpha=0.5`, `scale_pos_weight=5.0`을 그대로 돌리고, **`balanced_accuracy`가 아니라 `roc_auc`/`pr_auc`/`balanced_accuracy_at_best_cut`을 iteration 2의 0.8604 / 0.5925 / 0.7891과 비교**하십시오. 이것이 이 실행에서 유일하게 남은 "레버 하나만 움직인" family 비교이며, 한 행에서 preprocessing을 동시에 바꾸지 않는 것이 핵심입니다. 다만 시도된 두 family의 best-cut 폭이 0.0207뿐이었으므로, 이것이 0.099를 메울 것으로 기대해서는 안 됩니다 — 기대치는 "family 축이 얼마나 남았는지 확정" 수준입니다.
2. **목표값 0.8882의 타당성을 재검토할 것.** 이 값은 `source: derived`이며 baseline 0.6807에 margin 0.65를 적용해 산출된 것으로, 카드 스스로 `exceeds_ranking_ceiling: true`, `required_ks: 0.7764` vs 실제 `ks: 0.5297`, `ks_shortfall: 0.2467`을 기록하고 있습니다. `passable_margin: 0.263`이 제시하는 수준(대략 0.76~0.77 대)은 이미 iteration 2가 검증 0.7778·테스트 0.7937로 넘어섰습니다. 현재 컬럼 집합에서 어느 지점이 현실적 상한인지 합의하지 않으면, 남은 예산은 도달 불가능한 목표를 향해 소모됩니다.
3. **ranking을 올릴 유일한 실질 경로는 executor 밖의 feature engineering이므로, 이를 카드 단계에서 만들어 넣을 것.** executor는 컬럼 조합·비율·상호작용·재인코딩·컬럼 제거를 하지 않습니다(`missing_indicator`/`missing_count`만 예외). 반면 카드는 `like`(target_corr **strong**), `guess_prob_liked`, `attractive_o`/`attractive_partner`, `shared_interests_o`/`shared_interests_partner` 등 **양방향 짝 컬럼**이 존재함을 보여줍니다. 상호 평가의 대칭/차이(예: 양쪽 `like`의 최소값·차이, 선호 가중치와 상대 평점의 내적)는 트리가 축 정렬 분할로는 잘 표현하지 못하는 구조이며, 이런 파생 컬럼을 **입력 CSV에 추가해 새 카드로 다시 실행**하는 것이 ranking 축을 실제로 움직일 가장 유력한 수단입니다. 또한 고유 카디널리티로 드롭된 `field`(180 one-hot 레벨 한도, `max_cardinality=50`)를 상위 그룹으로 사전 축약해 넣는 것도 같은 경로입니다.
4. **`class_weight`/`scale_pos_weight` 미세 조정에는 더 이상 iteration을 쓰지 말 것.** `cut_headroom`이 0.0113이고, iteration 2와 4의 CI가 겹쳐 이 축에서의 두 시도가 통계적으로 구분되지 않습니다. 참고로 `calibration_error`=0.1478, `brier`=0.1315로 확률값은 상당히 과신 상태이지만, 이 executor에는 재캘리브레이션 레버가 없습니다(threshold 탐색·`CalibratedClassifierCV` 모두 불가). 확률값 자체가 필요한 용도라면 그것은 별도 후처리 단계로 분리해야 하며, 현재 루프 안에서 해결할 항목이 아닙니다.