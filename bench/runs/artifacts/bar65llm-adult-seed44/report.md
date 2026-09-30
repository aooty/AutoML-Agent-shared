# AutoML 실행 리포트 — `adult` / balanced_accuracy

## 요약

**목표는 달성되지 못했습니다.** 목표 기준선은 `balanced_accuracy` 0.9189였고, 5회 시도 중 가장 좋았던 구성(iteration 3, `hist_gbdt`)의 검증 점수는 0.8481 (95% CI 0.8397~0.8557)로 기준에 0.0708 미달했습니다. 한 번도 사용되지 않은 최종 테스트 20%에서 같은 모델은 `balanced_accuracy` 0.8402 (95% CI 0.8317~0.8486)를 기록했습니다. 5회 반복을 모두 소진했으며(중단 사유 `max_iterations`), 카드의 baseline(logreg, 0.7684, CI 0.7575~0.7784)은 명확히 상회했지만 목표 임계값과의 거리는 실행 가능한 레버로 좁혀지지 않았습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `learning_rate` 0.06, `max_iter` 600, `max_leaf_nodes` 31, `min_samples_leaf` 20, `l2_regularization` 1.0, `early_stopping` true, `class_weight` 'balanced' / `impute` 'none' | `balanced_accuracy` 0.8446 (CI 0.8361~0.8521), `roc_auc` 0.9293, `cut_headroom` 0.0011, `train_val_gap` 0.0383 | `wrong_model_family` — 격차의 98.5%가 ranking 축, 미시도 강력 ranker(xgboost)로 전환 권고 |
| 2 | `xgboost` | `learning_rate` 0.04, `n_estimators` 2000, `max_depth` 8, `min_child_weight` 2, `subsample` 0.8, `colsample_bytree` 0.8, `reg_lambda` 2.0, `early_stopping_rounds` 50, `scale_pos_weight` 3.18 / `impute` 'none' | `balanced_accuracy` 0.8395 (CI 0.8311~0.8482), `roc_auc` 0.9261, `train_val_gap` 0.0701 | `hyperparam` — family 전환은 paired Δ -0.0051로 손실, 깊이 증가 방향이 오류. 최고 ranking family로 복귀해 용량 축소 권고 |
| 3 | `hist_gbdt` | `learning_rate` 0.03, `max_iter` 1500, `max_leaf_nodes` 15, `min_samples_leaf` 40, `l2_regularization` 5.0, `max_bins` 255, `early_stopping` true, `class_weight` 'balanced' / `impute` 'none' | **`balanced_accuracy` 0.8481 (CI 0.8397~0.8557)**, `roc_auc` 0.9299, `pr_auc` 0.8302, `cut_headroom` 0.000343, `train_val_gap` 0.0177 | `wrong_model_family` — 100%가 ranking 축(implied KS 0.6968 vs 요구 0.8378). 마지막 미시도 레버로 `missing_count` 단일 변경 권고 |
| 4 | `random_forest` | `n_estimators` 600, `max_depth` 22, `min_samples_leaf` 3, `class_weight` 'balanced' / `impute` 'median', `missing_indicator` true | `balanced_accuracy` 0.8299 (CI 0.8218~0.8382), `roc_auc` 0.9138, `pr_auc` 0.7821, `train_val_gap` 0.0212 | `underfitting` — 4개 시도 중 최저 ranking, train 0.8512 자체가 기준 미달. `missing_count` 단일 변경 재권고 |
| 5 | `hist_gbdt` | iteration 3과 동일 하이퍼파라미터 / `impute` 'none', `missing_count` **true** | `balanced_accuracy` 0.8481, `roc_auc` 0.9299 — iteration 3과 **모든 지표가 소수점까지 동일** | (마지막 시도, critic 미실행) |

## 최고 성능 구성

iteration 3에서 실제로 빌드된 구성입니다(아래 값은 히스토리의 적용값 그대로이며, `dropped_hyperparams`는 비어 있었습니다).

- **model**: `hist_gbdt`
- **hyperparams**: `learning_rate` 0.03, `max_iter` 1500, `max_leaf_nodes` 15, `min_samples_leaf` 40, `l2_regularization` 5.0, `max_bins` 255, `early_stopping` true, `validation_fraction` 0.1, `n_iter_no_change` 40, `class_weight` 'balanced'
- **preprocessing (적용값)**: `impute` 'none', `scale` false, `missing_indicator` false, `missing_count` false
- **internal_validation**: `fit_rows` 26373, `held_out_rows` 2931, `stopped_at_iter` 997 / `max_iter` 1500 (early stopping이 스스로 발동)
- **train_time_sec**: 21.293

| 지표 | 검증(20%) | 최종 테스트(20%, 미사용 행) |
|---|---|---|
| `balanced_accuracy` | 0.8481 (CI 0.8397~0.8557) | **0.8402 (CI 0.8317~0.8486)** |
| `roc_auc` | 0.9299 | — |
| `pr_auc` / `average_precision` | 0.8302 | — |
| `recall` / `specificity` | 0.8682 / 0.8279 | — |
| `precision` / `f1` / `accuracy` | 0.6134 / 0.7189 / 0.8375 | — |
| `balanced_accuracy_at_best_cut` / `cut_headroom` | 0.8484 / 0.000343 | — |
| `brier` / `calibration_error` | 0.1086 / 0.1054 | — |
| `train_balanced_accuracy` / `train_val_gap` | 0.8658 / 0.0177 | — |

검증 0.8481과 테스트 0.8402의 차이 **+0.0078**은 선택 편향의 크기입니다 — 이 실행의 모든 선택은 검증 점수만 보고 이루어졌습니다. 다만 검증 점수가 테스트 CI(0.8317~0.8486) 안에 들어오므로, 이 행들만으로는 그 차이를 0과 구분할 수 없습니다. 어느 쪽 숫자로 보아도 목표 0.9189에는 미달합니다.

## 원인 분석

critic은 4회 실행되었고, 4개의 판정 전체가 같은 결론으로 수렴했습니다: **남은 격차는 operating point가 아니라 ranking 축에 있으며, 이 executor가 가진 레버는 그 크기를 낼 수 없다.**

- **operating point는 이미 소진되었습니다.** 최고 구성의 `cut_headroom`은 0.000343으로, 남은 0.0708의 0.5%에 불과합니다. `recall` 0.8682와 `specificity` 0.8279는 0.040 이내로 균형이 잡혀 있어, `class_weight` / `scale_pos_weight`를 어떻게 움직여도 얻을 수 있는 최대치가 0.0003 수준입니다. 4번의 시도 모두 `cut_headroom`이 0.0004~0.0020 범위였습니다.
- **용량(capacity) 축도 닫혔습니다.** `train_val_gap`은 0.0701(iteration 2, `max_depth` 8) → 0.0383(iteration 1, 31 leaves) → 0.0177(iteration 3, 15 leaves)로 정규화가 실제로 작동했고, 그 시점의 `train_balanced_accuracy` 0.8658 자체가 목표보다 0.053 낮습니다. train과 val이 함께 낮으므로 더 큰 모델·더 많은 iteration이 남은 답이 아닙니다. early stopping은 세 번 모두(323 / 628 / 997) 스스로 멈췄습니다.
- **ranking 축의 측정된 폭이 필요한 거리보다 한 자릿수 작습니다.** `balanced_accuracy_at_best_cut`은 3개 family에 걸쳐 0.8307(`random_forest`) ~ 0.8484(`hist_gbdt`), 폭 0.0177인데, 최적 컷에서도 기준까지 0.0705가 남습니다. KS로 보면 기준 0.9189는 0.8378을 요구하고 측정된 최고 ranking은 0.6968입니다. 목표 정의 자체가 이 점을 이미 기록하고 있습니다: `exceeds_ranking_ceiling` true, `ranking_ceiling` 0.8261, `ks_shortfall` 0.1855.
- **구간이 겹치는 움직임은 이야기로 만들지 않습니다.** iteration 1(0.8446, CI 0.8361~0.8521)과 iteration 3(0.8481, CI 0.8397~0.8557)은 이 데이터가 분리하지 못하는 두 시도입니다(paired Δ +0.0035, CI -0.0007~+0.0077). 즉 family 내 재튜닝은 "아무것도 하지 않은 것"과 구분되지 않았습니다. iteration 2(0.8395, CI 0.8311~0.8482)도 marginal 구간이 iteration 1과 겹치며, paired Δ -0.0051(CI -0.0101~-0.0005)는 부호가 잡히긴 하지만 크기가 0.005 수준으로 남은 0.079와는 무관한 규모입니다. 구간 폭보다 확실히 큰 유일한 차이는 iteration 4의 하락(Δ vs iteration 3 = -0.0181, CI -0.0238~-0.0126)입니다 — 다만 이 행은 family와 전처리(`impute` none→median, `missing_indicator` false→true)를 **동시에** 바꿨으므로 손실의 주인이 둘이고, 어느 레버에도 귀속시킬 수 없습니다. critic이 지적한 대로 이는 계획된 단일 변경 행이 아니었습니다.
- **missingness 열 레버는 이번에 명확히 0으로 측정되었습니다.** iteration 5는 iteration 3에 `missing_count` true만 추가한 단일 변경 행이었고, 모든 지표가 소수점까지 동일했습니다 — `impute` 'none'으로 NaN을 직접 분기하는 트리에서 이 열은 예측을 전혀 바꾸지 못했습니다. 이 방향에는 남은 것이 없습니다.
- **재계획 패턴에 대한 솔직한 평가**: critic은 `wrong_model_family`를 두 번 발부했고 각각 -0.0051, -0.0181을 지불했습니다(후자는 Planner가 처방을 다시 쓴 행). 세 번째 판정에서는 스스로 "family 전환이 기준에 도달하는 경로가 아니다"라고 명시하며 진단을 ceiling 확인용으로 재규정했습니다. 즉 루프는 잘못된 축을 오래 파지는 않았지만, 남은 예산을 기준 달성이 아니라 "왜 달성 불가한지"의 증거 수집에 썼습니다.
- iteration 1의 `unsupported_claims`에 `feature_engineering`이 기록되어 있습니다. 해당 계획 산문 전체가 이 기록에 남아 있지 않으므로 무엇을 의도했는지 단정할 수 없고, 이 플래그는 문자열 검사이며 critic도 이를 "feature 구성은 사용 불가"라는 **제약 언급**으로 읽었습니다. 따라서 이 시도가 그 때문에 무엇을 잃었다고 쓰지 않습니다 — 0.8446은 이 executor가 그 설정에서 내는 값입니다. 다만 feature 구성 능력의 부재는 아래 제안에 남깁니다.

요약하면, 한계는 모델 튜닝 실패가 아니라 **주어진 열·주어진 family 집합으로 만들 수 있는 랭킹의 상한**이었습니다. 목표 0.9189는 baseline의 최적 컷 상한(0.8261)조차 넘어서는 파생 기준이며, 이 실행은 그 상한을 0.8484까지 밀어올렸을 뿐입니다.

## 다음 단계 제안

(기록된 `Data caveats`는 없으므로, 특정 열·값·split을 신뢰할 수 없다고 배제해야 할 항목은 없습니다. 아래 제안은 대신 이 executor의 CAN/CANNOT 경계와 측정된 축 크기에 근거합니다.)

1. **목표 임계값을 먼저 재검토하십시오 (최우선).** 0.9189는 KS 0.8378을 요구하지만 이 파일에서 측정된 최고 ranking은 KS 0.6968이고, 3개 family의 `balanced_accuracy_at_best_cut` 폭은 0.0177입니다. 남은 거리 0.0705는 그 폭의 4배입니다. 예산을 더 쓰기 전에, 이 기준이 이 열 집합으로 달성 가능한 값인지(목표 정의의 `exceeds_ranking_ceiling` true, `ranking_ceiling` 0.8261이 이미 경고하고 있음) 확인하고, 달성 가능한 기준(예: baseline 대비 상대 개선, 또는 0.85 전후의 ranking 기반 기준)으로 재설정하는 것이 가장 큰 효과가 있습니다.
2. **feature 구성 능력을 executor에 추가하십시오.** 현재 executor는 파생·상호작용·재인코딩·열 제거를 일절 하지 않고, 유일한 열 추가 레버(`missing_indicator`, `missing_count`)는 이번 실행에서 `impute` 'none' 하에 예측 불변(iteration 5가 iteration 3과 비트 단위로 동일)으로 확인되었습니다. ranking을 0.14 KS만큼 올릴 수 있는 유일한 축이 feature이므로, `adult`에서 일반적으로 신호가 있는 변환(`capital-gain`/`capital-loss`의 로그·구간화, `education-num`×`hours-per-week` 상호작용, `marital-status`/`relationship`/`occupation` 조합, `native-country`·`education` 같은 중·고카디널리티 열의 target/ordinal 인코딩)을 파이프라인 능력으로 구현한 뒤 같은 `hist_gbdt` 설정에서 단일 변경으로 측정하는 것이 다음 예산의 정당한 사용처입니다. 카드가 `fnlwgt`의 `target_corr`를 'none'으로 보고하므로, 열 제거 능력이 생기면 이 열의 제외도 함께 단일 변경으로 검증할 가치가 있습니다.
3. **측정 인프라를 보강하십시오: 반복 seed 또는 CV.** 이번 실행에서 결정적인 차이(iteration 3 vs 4의 -0.0181)를 제외한 모든 움직임(+0.0035, -0.0051)은 검증 슬라이스의 CI 폭 0.016 안에 들어가, 단일 20% split으로는 family 내 튜닝을 서로 구분할 수 없었습니다. 현재 executor는 CV·out-of-fold를 지원하지 않으므로, 이는 코드 변경 항목입니다. 이것이 해결되면 위 2번의 feature 실험들이 0.003~0.006 규모에서도 판정 가능해집니다.
4. **operating point와 missingness 열에는 더 이상 예산을 쓰지 마십시오.** `cut_headroom` 0.000343, `recall`/`specificity` 0.8682/0.8279는 `class_weight` 조정 여지가 0.0003임을 뜻하고, `missing_count`/`missing_indicator`는 `impute` 'none'과 함께 쓰일 때 이 파일에서 정확히 0을 기록했습니다. 다만 `calibration_error` 0.1054, `brier` 0.1086은 예측 확률을 확률로 해석할 계획이 있다면 별도 문제입니다 — 현재 executor에 recalibration 레버가 없으므로, 확률값이 실제로 필요하다면 이 역시 기능 추가 항목으로 올려야 합니다(`balanced_accuracy`에는 영향 없음).