# AutoML 실행 최종 보고서 — `speeddating` / `balanced_accuracy`

## 요약

**목표는 달성되지 못했습니다.** 목표선은 `balanced_accuracy` 0.8882였고, 5회 예산 중 4회를 사용해 도달한 최고 검증 점수는 iteration 2의 `hist_gbdt`가 기록한 **0.7766 (95% CI 0.7493~0.8026)** 으로, 목표까지 0.1116이 남았습니다. 단 한 번도 학습·선택에 쓰이지 않은 최종 테스트 20%에서 같은 모델은 **0.7961 (95% CI 0.7718~0.8198)** 을 기록했고, 이는 baseline logreg의 0.6807 (CI 0.654~0.7099)보다는 분명히 높습니다. 루프는 연속 미개선(정체)으로 iteration 4 이후 조기 종료되었으며, critic은 3회 진단을 내렸습니다(`overfitting` → `wrong_model_family` → `data_issue`).

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `learning_rate` 0.06, `max_iter` 400, `max_leaf_nodes` 31, `max_depth` 6, `l2_regularization` 1.0, `class_weight` balanced, `early_stopping` false / `impute` none, `scale` false | `balanced_accuracy` 0.7256 (CI 0.6971~0.7535), `roc_auc` 0.8516, `balanced_accuracy_at_best_cut` 0.7810, `cut_headroom` 0.0554, `train_val_gap` 0.2721 | `overfitting` — train 0.9977 vs val 0.7256. 용량을 줄이고 `l2`/`min_samples_leaf`를 올려 ranking을 회복하라 |
| 2 | `hist_gbdt` | `learning_rate` 0.03, `max_iter` 300, `max_leaf_nodes` 8, `max_depth` 3, `l2_regularization` 10.0, `min_samples_leaf` 60, `max_features` 0.5, `class_weight` balanced, `early_stopping` false / `impute` none, `scale` false | **최고** `balanced_accuracy` 0.7766 (CI 0.7493~0.8026), `roc_auc` 0.8583, best_cut 0.7849, `cut_headroom` 0.0083, `train_val_gap` 0.0608 | `wrong_model_family` — 용량 처방은 +0.0509(CI +0.0239~+0.0770)로 효과 확인되었으나 ranking은 거의 불변(best_cut +0.0039). 다른 family(`xgboost`)로 이동하라 |
| 3 | `extra_trees` | `n_estimators` 600, `max_depth` 20, `min_samples_leaf` 5, `class_weight` balanced / `impute` median, `scale` false, `missing_count` true | `balanced_accuracy` 0.7409 (CI 0.7156~0.7716), `roc_auc` 0.8468, best_cut 0.7697, `cut_headroom` 0.0288, `train_val_gap` 0.2437 | `data_issue` — iteration 2 대비 -0.0356(CI -0.0608~-0.0111)의 해소된 회귀. 두 개의 구조적으로 다른 ranking이 best_cut 0.7697~0.7849에 갇혔으므로 남은 격차는 사용 가능한 119개 열의 성질이다 |
| 4 | `hist_gbdt` | `learning_rate` 0.04, `max_iter` 400, `max_leaf_nodes` 16, `max_depth` 4, `l2_regularization` 5.0, `min_samples_leaf` 40, `max_features` 0.7, `class_weight` {0:1.0, 1:6.0}, `early_stopping` false / `impute` none, `scale` false | `balanced_accuracy` 0.7708 (CI 0.7414~0.7961), `roc_auc` 0.8583, best_cut 0.7765, `cut_headroom` 0.0057 | (없음 — 평가 직후 루프 종료) |

## 최고 성능 구성

**iteration 2 / `hist_gbdt`** — 아래 값은 executor가 실제로 만든 추정기에서 읽은 적용값입니다(`dropped_hyperparams` 없음).

```
model: hist_gbdt
hyperparams:
  learning_rate: 0.03
  max_iter: 300
  max_leaf_nodes: 8
  max_depth: 3
  l2_regularization: 10.0
  min_samples_leaf: 60
  max_features: 0.5
  class_weight: "balanced"
  early_stopping: false
preprocessing (applied):
  impute: none          # hist_gbdt가 NaN을 직접 분기
  scale: false
  missing_indicator: false
  missing_count: false
protocol: stratified 60/20/20, seed 44
train_time_sec: 5.876
```

| 지표 | 검증(20%) | 최종 테스트(20%, 미사용 행) |
|---|---|---|
| `balanced_accuracy` | **0.7766** (CI 0.7493~0.8026) | **0.7961** (CI 0.7718~0.8198) |
| `roc_auc` | 0.8583 | — |
| `pr_auc` / `average_precision` | 0.6023 | — |
| `recall` / `specificity` | 0.7681 / 0.7850 | — |
| `precision` / `f1` / `accuracy` | 0.4133 / 0.5374 / 0.7822 | — |
| `balanced_accuracy_at_best_cut` / `cut_headroom` | 0.7849 / 0.0083 | — |
| `brier` / `calibration_error` | 0.1421 / 0.1763 | — |
| `train_val_gap` | 0.0608 | — |

검증 점수와 테스트 점수의 차이는 0.0195입니다(기록 표기 -0.0195). 이 차이가 **선택 편향의 크기**입니다 — 루프는 검증 숫자만 보고 최고 시도를 골랐고, 테스트 점수는 그 선택 이후 한 번만 측정된 값입니다. 다만 검증 0.7766은 테스트 CI(0.7718~0.8198) 안에 들어가므로, 이 행 수로는 그 차이를 0과 구분할 수 없습니다. 어느 쪽으로 읽어도 목표선 0.8882에는 두 숫자 모두 크게 미달합니다.

## 원인 분석

**해소된(interval보다 큰) 차이만 놓고 보면 이야기는 두 줄입니다.**

1. **iteration 1의 실패 원인은 용량 과다였고, 그 처방은 실제로 통했습니다.** `train_balanced_accuracy` 0.9977 vs val 0.7256(`train_val_gap` 0.2721)에서 gap 0.0608로 떨어지며 검증 점수가 +0.0509(paired CI +0.0239~+0.0770, P(better) 1.000) 개선되었습니다. 이는 CI 폭보다 큰, 보고할 만한 개선입니다.
2. **그러나 그 개선은 ranking이 아니라 cut 위치 이동이 대부분이었습니다.** `roc_auc`는 0.8516 → 0.8583(+0.0067, 이 측정에서의 paired 분해능 0.003~0.006 대역 근처), `balanced_accuracy_at_best_cut`은 0.7810 → 0.7849(+0.0039)에 머물렀고, 대신 `recall` 0.5362 → 0.7681 / `specificity` 0.9150 → 0.7850으로 운영점이 중앙으로 이동하면서 `cut_headroom`이 0.0554 → 0.0083으로 소진되었습니다. 즉 imbalance 레버(`class_weight`)는 사실상 다 쓰였습니다.

**나머지는 ranking 축의 벽입니다.** `balanced_accuracy_at_best_cut`은 (1+KS)/2이므로, 목표 0.8882는 KS 0.7764를 요구하는데 기록상 최고 ranking의 KS는 0.5698(best_cut 0.7849)입니다. baseline logreg의 KS는 0.5297, `balanced_accuracy_at_best_cut`은 0.7648이었습니다. 즉 **목표선 자체가 이 데이터의 ranking 상한(reference `ranking_ceiling` 0.7649, `exceeds_ranking_ceiling` true, `ks_shortfall` 0.2467) 위에 설정**되어 있었고, 어떤 임계값·가중치 조합으로도 도달할 수 없는 값이었습니다. iteration 2의 남은 격차 0.1116 중 운영점이 공급할 수 있는 몫은 `cut_headroom` 0.0083(7%)뿐이었습니다.

**family 교체는 이 벽을 넘지 못했고, 그 시도는 해석이 반쯤 막혀 있습니다.** iteration 3은 iteration 2 대비 -0.0356(CI -0.0608~-0.0111)의 해소된 회귀였지만, family(`hist_gbdt`→`extra_trees`)와 파이프라인(`impute` none→median, `missing_count` false→true)이 **동시에** 바뀌어 손실의 주인이 둘입니다. 또한 critic 2가 처방한 `xgboost`는 끝까지 실행되지 않았으므로(planner가 `extra_trees`로 대체) **이 실행은 xgboost에 대해 아무 증거도 남기지 않았습니다.** 다만 구조가 크게 다른 두 유도 방식의 best_cut이 0.7697과 0.7849로 0.0152 폭 안에 모였다는 사실은, 남은 0.1033(최적 cut 기준)이 family 선택 문제가 아니라는 정황으로 읽힙니다.

**iteration 4는 iteration 2와 이 데이터로 구분되지 않습니다.** 0.7708 (CI 0.7414~0.7961) vs 0.7766 (CI 0.7493~0.8026) — 차이 0.0058은 구간 폭보다 훨씬 작고 `roc_auc`도 0.8583로 동일 수준입니다. 이 두 시도 사이의 움직임으로 어떤 서사도 만들지 않았습니다. 루프의 정체 종료는 이 무차별성의 결과입니다.

**부수 관찰(주장으로 세우지 않음).** 최고 시도의 `calibration_error`는 0.1763, `brier`는 0.1421로 iteration 1(0.0592 / 0.1060)보다 크게 나빠졌습니다. 이는 개선분이 더 나은 확률이 아니라 운영점 이동이었다는 위 판독과 일치합니다. 그리고 iteration 1·3의 계획에는 `feature_engineering` 관련 `unsupported_claims` 플래그가 붙었으나 이는 계획 산문에 대한 문자열 검사이고, 해당 계획 본문이 기록에 남아 있지 않아 계획이 그 기능에 **의존**했는지는 확인할 수 없습니다. executor는 애초에 파생 피처를 만들 수 없으므로, 이 플래그를 "무언가를 잃었다"로 읽지 않고 아래 제안에 "만들어야 할 것"으로 옮깁니다.

## 다음 단계 제안

기록된 `Data caveats`는 없습니다(별도 주의사항 없음). 따라서 아래 제안을 무효화하는 caveat도 없지만, 그만큼 **왜 값이 결측인지에 대한 확인된 근거도 없다**는 점을 전제로 씁니다.

1. **목표선을 다시 유도하십시오 (최우선).** 0.8882는 `derived` 목표이며 reference 자체가 `exceeds_ranking_ceiling: true`, `required_ks` 0.7764 대 baseline KS 0.5297(`ks_shortfall` 0.2467)을 기록하고 있습니다. 이 실행이 확인한 것은 두 개의 서로 다른 family 모두 `balanced_accuracy_at_best_cut` 0.77~0.785에서 평탄해진다는 사실이며, 이는 목표선이 현재 열 집합으로 도달 불가능한 지점에 있음을 강하게 시사합니다. `passable_margin` 0.263 기준으로 목표를 재설정하고, 현재 결과(테스트 0.7961)를 그 기준으로 재평가하는 것이 남은 예산을 태우기 전에 해야 할 일입니다.

2. **ranking을 올릴 수 있는 유일한 축 — 피처 — 을 executor 밖에서 확보하십시오.** executor는 파생·재인코딩·열 삭제를 하지 않으므로(`missing_indicator`/`missing_count`만 예외), 카드 단계에서 열을 추가해야 합니다. 근거가 되는 두 지점: (a) `field`가 high-cardinality로 드롭되어 있어(`dropped_high_cardinality: ["field"]`) 정보가 통째로 빠졌습니다 — 상류에서 frequency/그룹 인코딩 후 카드에 넣으면 executor가 그대로 소비합니다. (b) 카드가 `like`를 유일한 `target_corr: strong`, `guess_prob_liked`·`attractive_o`·`attractive_partner`·`shared_interests_partner` 등을 `moderate`로 표시합니다 — 양방향 평가의 차/합 같은 대칭 결합은 트리가 열 단위 분기로 표현하기 어려운 형태이므로, 이를 상류에서 만들어 넣는 것이 남은 0.10 격차에 대해 유일하게 "축이 맞는" 레버입니다. 크기는 이 실행에 증거가 없으므로 약속하지 않습니다.

3. **`xgboost`를 단일 레버로 한 번 실행해 미결 질문을 닫으십시오.** critic 2의 처방(`impute: none`, `max_depth` 5, `min_child_weight` 5, `subsample`/`colsample_bytree` 0.8, `reg_lambda` 5.0, `scale_pos_weight` = n_neg/n_pos, `early_stopping_rounds` 40)은 실행되지 않았고, iteration 3은 family와 파이프라인을 동시에 바꿔 귀속이 불가능했습니다. 파이프라인을 iteration 2와 **동일하게** 두고 family만 바꿔 `roc_auc`(기준 0.8583)와 `balanced_accuracy_at_best_cut`(기준 0.7849)만 읽으십시오. 기대치는 낮습니다 — 기록상 family 교체의 ranking 폭은 0.0022~0.0077 `roc_auc`로 격차보다 한 자릿수 작습니다 — 그러나 "family가 아니라 피처가 한계"라는 결론을 확정하는 값싼 확인(약 6~10초)입니다.

4. **결측 열 레버에는 예산을 쓰지 마십시오.** 최고 구성은 `impute: none`이며, 이 조건에서 `missing_indicator`는 문서상 *중복*(예측이 비트 단위로 동일)으로 확정되어 있습니다. `missing_count`는 iteration 3에 이미 포함되었지만 family·impute와 함께 바뀌어 단독 효과를 알 수 없고, 다른 표본에서 0.0003/0.0000으로 측정된 이력이 있습니다. 여기에 이터레이션을 쓰기 전에, `expected_num_interested_in_me`의 78.5% 결측이 **응답자 상태 때문인지 wave/기록 방식 때문인지**를 원본 파일에서 확인하는 것이 먼저입니다(카드에 `wave` 열이 있고 caveat는 비어 있습니다). 후자라면 그 열들은 기록 체제를 학습하게 만들어 랜덤 분할에서만 이득으로 보일 것이고, 이 확인이 그 판단을 열어 줍니다.

5. **확률값이 필요한 용도라면 최고 구성을 그대로 쓰지 마십시오.** iteration 2는 `calibration_error` 0.1763 / `brier` 0.1421로, 순위는 같은 수준이면서 확률은 iteration 1(0.0592 / 0.1060)보다 크게 나쁩니다. executor에는 recalibration 레버가 없으므로(`CalibratedClassifierCV`·임계값 이동 모두 불가), 보정은 이 루프 밖에서 별도로 붙여야 하며 그 결정은 `balanced_accuracy` 최적화와 분리해 내려야 합니다.