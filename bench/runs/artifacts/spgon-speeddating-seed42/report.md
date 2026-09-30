# AutoML 실행 리포트 — speeddating (`match`, binary_classification)

## 요약

목표는 달성되었습니다. 목표 지표 `balanced_accuracy`의 기준선(threshold 0.7514, logreg baseline 0.6685에서 유도)을 iteration 2의 `hist_gbdt`가 검증 점수 0.7784 (95% CI 0.7492~0.8040)로 넘었고, 한 번도 사용되지 않은 최종 테스트 20%에서 0.7865 (95% CI 0.7611~0.8118)를 기록했습니다. 총 5회 예산 중 4회를 사용했으며, 이후 두 번의 시도(xgboost)가 개선을 만들지 못해 정체(stalled)로 조기 종료되었습니다. 다만 검증 CI의 하단 0.7492가 threshold 0.7514 아래로 내려가므로, 검증 슬라이스 단독으로는 통과가 여유 있게 확인되지는 않았고 — 그 통과를 실제로 뒷받침하는 숫자는 held-back 테스트 점수 0.7865입니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `learning_rate` 0.06, `max_iter` 500, `max_leaf_nodes` 31, `min_samples_leaf` 20, `l2_regularization` 1.0, `early_stopping` true, `class_weight` {0:1, 1:4}; `impute: none` | `balanced_accuracy` **0.7308** (CI 0.7022~0.7601), recall 0.5580 / specificity 0.9036, `roc_auc` 0.8666, `balanced_accuracy_at_best_cut` 0.7993, `cut_headroom` 0.0685 | `data_issue` — 부족분 0.0206은 이 슬라이스가 분해하지 못하는 크기. 랭킹은 이미 기준을 허용하므로 격차는 operating point에 있음 → class weight를 8:1로 |
| 2 | `hist_gbdt` | iteration 1과 동일, `class_weight` {0:1, 1:8}만 변경; `impute: none` | `balanced_accuracy` **0.7784** (CI 0.7492~0.8040) — 최고. recall 0.7319 / specificity 0.8250, `cut_headroom` 0.0104, `roc_auc` 0.8639 | `wrong_model_family` — 페어 Δ +0.0477 (CI +0.0229~+0.0706)로 operating point 레버는 소진(`cut_headroom` 0.0685→0.0104). 랭킹 천장 0.7888까지 남은 여유 0.0104뿐 → 다른 family로 랭킹을 올려보라 |
| 3 | `xgboost` | `n_estimators` 600, `learning_rate` 0.05, `max_depth` 6, `min_child_weight` 5, `subsample` 0.85, `colsample_bytree` 0.8, `reg_lambda` 2.0, `reg_alpha` 0.5, `scale_pos_weight` 8.0; `impute: none` | `balanced_accuracy` **0.7198** (CI 0.6877~0.7489), 페어 Δ vs it2 −0.0586 (CI −0.0849~−0.0307). 랭킹은 최고치(`roc_auc` 0.8739, `pr_auc` 0.6051, best_cut 0.8070)지만 `cut_headroom` 0.0872로 폭발, recall 0.5254 / specificity 0.9143, `train_val_gap` 0.2796 | `data_issue` — family swap은 랭킹을 올리고 컷을 망침. 남은 격차는 다시 operating point 축 → `scale_pos_weight` 20으로 |
| 4 | `xgboost` | iteration 3과 동일, `scale_pos_weight` 20.0; 적용된 `preprocessing`은 `impute: median` (요청과 달리 median으로 기록됨), `dropped_hyperparams: ["class_weight"]` | `balanced_accuracy` **0.7431** (CI 0.7102~0.7750), recall 0.5870 / specificity 0.8993, `roc_auc` 0.8747, best_cut 0.8024, `cut_headroom` 0.0593, `train_val_gap` 0.2550 | (없음 — 루프가 여기서 정체로 종료) |

## 최고 성능 구성

**iteration 2 · `hist_gbdt`** — 아래 값은 히스토리에 기록된 *적용된* 하이퍼파라미터와 전처리입니다.

```
model: hist_gbdt
hyperparams:
  learning_rate: 0.06
  max_iter: 500
  max_leaf_nodes: 31
  min_samples_leaf: 20
  l2_regularization: 1.0
  early_stopping: true
  validation_fraction: 0.15
  n_iter_no_change: 30
  class_weight: {"0": 1.0, "1": 8.0}
preprocessing:
  impute: none          # NaN을 모델이 직접 분기 (hist_gbdt 네이티브)
  scale: false
  missing_indicator: false
  missing_count: false
dropped_hyperparams: []   # 요청한 설정이 모두 적용됨
train_time_sec: 4.623
```

검증 슬라이스(20%) 지표:

| 지표 | 값 |
|---|---|
| `balanced_accuracy` | **0.7784** (95% CI 0.7492~0.8040) |
| `recall` / `specificity` | 0.7319 / 0.8250 |
| `precision` | 0.4519 |
| `f1` | 0.5588 |
| `accuracy` | 0.8097 |
| `roc_auc` | 0.8639 |
| `pr_auc` / `average_precision` | 0.5818 |
| `balanced_accuracy_at_best_cut` | 0.7888 (`cut_headroom` 0.0104) |
| `brier` / `calibration_error` | 0.1306 / 0.1430 |
| `train_balanced_accuracy` / `train_val_gap` | 0.9161 / 0.1377 |

**최종 held-back 테스트(20%, 루프 중 한 번도 읽지 않은 행): `balanced_accuracy` = 0.7865 (95% CI 0.7611~0.8118).** 검증 0.7784 대비 +0.0080 차이이며, 이 차이의 크기가 곧 선택 편향(selection effect)의 크기입니다 — 루프는 검증 숫자만 보고 최선을 골랐습니다. 다만 검증 점수가 테스트 CI 안에 들어 있어 이 행들로는 두 숫자가 0과 구분되지 않습니다. 이 실행이 실제로 입증한 성능은 검증 0.7784가 아니라 테스트 0.7865이며, 이는 threshold 0.7514와 baseline 0.6685를 모두 상회합니다.

## 원인 분석

**1) 성능을 실제로 움직인 것은 operating point 축이었고, 그것은 iteration 2에서 이미 소진되었습니다.** iteration 1→2는 단일 레버(class weight 4:1→8:1) 변경으로 페어 Δ +0.0477 (CI +0.0229~+0.0706)을 얻었습니다 — 구간이 0을 포함하지 않는, 이 실행에서 유일하게 명확한 개선입니다. 그 대가로 `cut_headroom`이 0.0685 → 0.0104로 붕괴했고 recall(0.7319)이 specificity(0.8250)를 향해 올라와 crossover 근처에 자리를 잡았습니다. 즉 이후에 class weight를 더 만져 얻을 수 있는 최대치는 `balanced_accuracy_at_best_cut` 0.7888 − 0.7784 = 0.0104, 한 자리 백분율의 절반 수준입니다.

**2) 랭킹 축은 xgboost에서 실제로 올라갔지만, 목표 지표로는 회수되지 않았습니다.** iteration 3/4는 이 실행에서 가장 좋은 랭킹을 만들었습니다: `roc_auc` 0.8666(it1) → 0.8739(it3) → 0.8747(it4), `pr_auc` 0.5818 → 0.6051 → 0.6080, `balanced_accuracy_at_best_cut` 0.7888 → 0.8070 → 0.8024. 그런데 기본 `predict()` 컷이 crossover에서 멀어져(`cut_headroom` 0.0872, 0.0593; recall 0.5254/0.5870 vs specificity 0.9143/0.8993) `balanced_accuracy`는 각각 0.7198, 0.7431에 머물렀습니다. `scale_pos_weight` 8.0이 hist_gbdt의 `class_weight {0:1,1:8}`와 같은 컷을 재현하지 못한 것이 핵심이며, 600 라운드에 조기 종료가 없어 `train_balanced_accuracy` 0.9994 / `train_val_gap` 0.2796까지 확률이 양극단으로 밀린 것이 그 배경입니다(단, executor는 xgboost에 `eval_set`/`early_stopping_rounds`를 넘길 수 없습니다 — 이는 실패가 아니라 도구의 경계입니다).

**3) 구간이 겹치는 비교로 이야기를 만들지 않았습니다.** iteration 1(0.7308, CI 0.7022~0.7601), iteration 3(0.7198, CI 0.6877~0.7489), iteration 4(0.7431, CI 0.7102~0.7750)는 서로 구간이 넓게 겹칩니다 — 이 데이터로는 이 세 시도를 서열화할 수 없습니다. 특히 iteration 3→4의 +0.0233은 각 구간 폭(약 0.06)보다 작으므로 개선으로 보고할 수 없습니다. 반대로 iteration 2와 iteration 3의 차이(−0.0586, 페어 CI −0.0849~−0.0307)는 구간이 0을 넘지 않으므로 손실로 확정됩니다. 최고 점수 0.7784의 CI 하단 0.7492가 threshold 0.7514를 살짝 밑도는 것도 같은 이유로 정직하게 남겨둡니다 — 통과 판정을 지탱하는 것은 held-back 0.7865입니다.

**4) 부수적으로 남은 두 가지 기록.** iteration 4는 `dropped_hyperparams: ["class_weight"]`로 기록되어 있어(xgboost는 `scale_pos_weight`를 씀) 그 키가 도움이 되었을지는 이 실행이 말해주지 못합니다. 또 iteration 4의 적용된 `preprocessing`은 `impute: median`으로 기록되어, 계획된 NaN-native 파이프라인과 달라졌습니다 — 결측률이 0.7852에 이르는 `expected_num_interested_in_me` 같은 열이 있는 데이터에서 이 차이는 무해하지 않을 수 있으나, 동시에 `scale_pos_weight`도 바뀐 시도이므로 두 요인이 한 숫자를 공유하고 있어 분리 불가입니다. iteration 1의 `unsupported_claims`에는 `feature_engineering`이 잡혀 있으나, executor는 파생 피처를 만들 수 없다는 것이 애초의 경계이며, 그 결과 이 실행은 파생 피처(예: 선호도-평가 간 차이/비율)의 가치를 **한 번도 테스트하지 않았습니다** — 실패가 아니라 미측정 영역입니다.

## 다음 단계 제안

1. **xgboost의 좋은 랭킹을 컷 위치와 함께 회수한다 — `scale_pos_weight`를 12~16 범위에서 한 점씩, 다른 레버는 고정.** iteration 4 시점의 랭킹 천장은 `balanced_accuracy_at_best_cut` 0.8024, `cut_headroom` 0.0593으로 여전히 크고, recall 0.5870이 specificity 0.8993보다 한참 낮습니다. 8.0(headroom 0.0872) → 20.0(0.0593)의 이동 방향은 옳았으나 아직 crossover에 못 미쳤습니다. 단, iteration 4는 `impute: median`으로 실행되었으므로 **반드시 `impute: none`을 명시해 iteration 3과 동일한 NaN-native 파이프라인으로 되돌린 뒤** 가중치만 움직여야 델타가 귀속됩니다. 성공하면 최고 검증 0.7784를 랭킹 천장 0.80대에 근접시킬 수 있습니다.
2. **xgboost의 과적합을 랭킹 손실 없이 줄여 컷을 안정화한다.** `train_val_gap` 0.2550~0.2796, `train_balanced_accuracy` 0.998~0.999는 확률이 양극단으로 밀렸다는 뜻이고, 이것이 기본 0.5 컷을 crossover에서 멀리 떨어뜨린 직접 원인입니다. executor는 xgboost에 `eval_set`/`early_stopping_rounds`를 넘길 수 없으므로, 대신 `n_estimators`를 200~300으로 줄이거나 `learning_rate`를 낮추고 `min_child_weight`/`reg_lambda`를 올리는 단일 변경으로 접근하십시오. `roc_auc`가 0.874 수준을 유지하면서 `cut_headroom`이 줄어드는지가 판정 기준입니다.
3. **hist_gbdt 쪽에서는 컷이 아니라 랭킹을 건드려야 한다 — 용량/정규화 재조정 1회.** iteration 2의 `cut_headroom`은 0.0104로, class weight를 더 움직여 얻을 것이 사실상 없습니다. 반면 `train_val_gap` 0.1377은 여유 용량을 뜻하므로, `max_leaf_nodes`/`min_samples_leaf`/`l2_regularization` 중 **하나만** 바꿔 `roc_auc`(현재 0.8639, xgboost의 0.8747보다 낮음)와 `balanced_accuracy_at_best_cut`(0.7888 vs 0.8070)을 끌어올리는 것이 남은 상방입니다. 다만 이 실행에서 관측된 family 내 재튜닝 폭은 작으므로, 기대치는 백분율 한 자리 수준으로 잡으십시오.
4. **현재 executor 밖의 두 레버는 다음 세대 도구로 미뤄 명시적으로 요청한다.** (a) 임계값 스윕/재캘리브레이션 — iteration 2의 `calibration_error` 0.1430, `brier` 0.1306은 확률을 확률로 읽으면 평균 14%p 어긋난다는 뜻이고, `class_weight` 8:1로 컷을 끌어온 부작용입니다. 확률 보정이나 명시적 컷 선택이 가능해지면 랭킹을 건드리지 않고 `_at_best_cut`을 직접 회수할 수 있습니다. (b) 파생 피처 — 이 데이터에는 `pref_o_*`(상대의 선호 가중치)와 `attractive_o`/`funny_o` 등 평가 점수, `like`(target_corr strong), `guess_prob_liked`가 함께 있어 가중합·차이 형태의 조합이 자연스러운데, 현 executor는 컬럼 결합을 전혀 하지 못해 이 축은 **미측정**입니다. 랭킹 천장(약 0.80~0.81)을 넘길 가능성이 남아 있는 곳은 현실적으로 여기입니다. 한편 `missing_indicator`/`missing_count`는 `impute: none`과 함께 쓰면 예측이 비트 단위로 동일한 것으로 이미 확인된 레버이므로, NaN-native 파이프라인을 유지하는 한 반복 예산을 쓰지 마십시오.

> 참고: 이 데이터셋에는 별도로 기록된 data caveat이 없습니다(“없음”). 따라서 위 제안 중 특정 열·값·분할의 신뢰성을 전제로 한 것은 없으며, 대신 이 실행 자체가 남긴 두 개의 기록상 불확실성(iteration 4의 `impute: median` 적용, iteration 4의 `dropped_hyperparams: ["class_weight"]`)을 제안 1에서 명시적으로 해소하도록 설계했습니다.