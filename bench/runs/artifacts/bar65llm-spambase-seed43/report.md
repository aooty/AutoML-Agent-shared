# AutoML 실행 최종 리포트 — `spambase` (binary_classification, 목표: `balanced_accuracy` 최대화)

## 요약

목표는 **달성하지 못했습니다**. 목표 기준선은 `balanced_accuracy` 0.9711이었고, 3회 시도 중 최고 검증 점수는 iteration 1의 `hist_gbdt`가 기록한 **0.9538 (95% CI 0.9392~0.9676)** 로, 기준선이 이 구간 위쪽에 위치하므로 0.0173의 미달은 실재하는 차이입니다. 같은 모델을 루프 내내 한 번도 사용되지 않은 홀드백 테스트 20%에서 한 번 채점한 결과는 **0.9490 (95% CI 0.9343~0.9614)** 이며, 이것이 이번 실행이 실제로 입증한 수치입니다(검증 대비 0.0048 차이). 5회 예산 중 3회만 사용하고 연속 미개선(정체)으로 조기 종료되었으며, baseline `logreg` 0.9173은 세 시도 모두 상회했습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=500`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `early_stopping=False` | `balanced_accuracy`=**0.9538** (CI 0.9392~0.9676), `roc_auc`=0.9840, `balanced_accuracy_at_best_cut`=0.9552, `cut_headroom`=0.0014, `train_balanced_accuracy`=1.0, `train_val_gap`=0.0462, 4.6s | `overfitting` — 500 iter/무 early stopping으로 train을 완전 암기, 남은 격차는 operating point가 아니라 ranking 축(`cut_headroom`이 필요분의 8%)이라 판단, 동일 family 내 capacity 축소 + 정규화 강화 지시 |
| 2 | `hist_gbdt` | `max_iter=800`, `learning_rate=0.04`, `max_leaf_nodes=15`, `min_samples_leaf=40`, `l2_regularization=10.0`, `max_features=0.6`, `early_stopping=False` | `balanced_accuracy`=0.9474 (CI 0.9294~0.9610), `roc_auc`=0.9851, `balanced_accuracy_at_best_cut`=0.9530, `cut_headroom`=0.0056, `train_balanced_accuracy`=0.9914, `train_val_gap`=0.0440, 5.1s | `wrong_model_family` — 정규화는 의도대로 train 포화를 낮췄으나 ranking 상한이 오르지 않음(0.9552→0.9530). paired Δ = -0.0064 (CI -0.0170~+0.0027, P(better)=0.072)로 0과 구분 불가. 서로 다른 capacity 두 점에서 상한이 0.953~0.955로 평탄 → family 교체(`xgboost` + `scale_pos_weight=1.3`) 지시 |
| 3 | `xgboost` | `n_estimators=900`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=1`, `reg_lambda=1.0`, `scale_pos_weight=1.3` | `balanced_accuracy`=0.9479 (CI 0.9308~0.9620), `roc_auc`=0.9850, `balanced_accuracy_at_best_cut`=0.9596, `cut_headroom`=0.0117, `train_balanced_accuracy`=0.9995, `train_val_gap`=0.0517, 4.3s | (없음 — 평가 직후 루프가 정체로 종료되어 진단 대상이 아님) |

세 시도 모두 `dropped_hyperparams`는 비어 있고 `unsupported_claims`도 없습니다. 즉 계획된 설정이 실제로 그대로 적합되었습니다.

## 최고 성능 구성

- **모델**: `hist_gbdt` (iteration 1)
- **적용된 `hyperparams`**: `max_iter=500`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `early_stopping=False`
- **적용된 `preprocessing`** (해당 시도의 기록값): `impute="median"`, `scale=false`, `missing_indicator=false`, `missing_count=false`
- **프로토콜**: stratified 3-way split, seed 43, train 0.6 / val 0.2 / test 0.2 (카드 baseline과 동일 split), 학습 시간 4.617초

| 구분 | `balanced_accuracy` | 95% CI |
|---|---|---|
| 검증(20%) — 루프가 선택에 사용한 수치 | 0.9538 | 0.9392~0.9676 |
| **홀드백 테스트(20%) — 루프 종료 후 1회 채점** | **0.9490** | 0.9343~0.9614 |
| 목표 기준선 | 0.9711 | — |
| baseline `logreg` (검증) | 0.9173 | 0.8995~0.9334 |

검증과 테스트의 차이 0.0048은 **선택 편향의 크기**입니다 — 루프는 검증 점수만 보고 최고 시도를 골랐기 때문입니다. 다만 검증 점수 0.9538은 테스트 CI(0.9343~0.9614) 안에 들어오므로, 이 행들만으로는 그 차이를 0과 구분할 수 없습니다. 어느 쪽 수치로 보아도 목표 0.9711에는 미달입니다.

검증 슬라이스의 부가 지표(iteration 1): `f1`=0.9466, `accuracy`=0.9587, `precision`=0.9629, `recall`=0.9309, `specificity`=0.9767, `roc_auc`=0.9840, `pr_auc`/`average_precision`=0.9751, `brier`=0.0344, `calibration_error`=0.0252, `balanced_accuracy_at_best_cut`=0.9552, `cut_headroom`=0.0014.

## 원인 분석

**세 시도는 이 데이터로는 서로 구분되지 않습니다.** 검증 점수 0.9538 / 0.9474 / 0.9479의 CI(각각 ±0.014 내외)가 모두 크게 겹치며, iteration 1→2의 paired Δ는 -0.0064 (CI -0.0170~+0.0027, P(better)=0.072)입니다. 따라서 "정규화가 성능을 깎았다" 또는 "xgboost가 hist_gbdt보다 낫다/못하다"는 서사는 이 슬라이스가 지지하지 않습니다 — 그 크기의 움직임은 리샘플 잡음과 분리되지 않습니다.

구간보다 큰, 실제로 읽을 수 있는 신호는 두 가지입니다.

1. **격차는 operating point가 아니라 ranking 축에 있습니다.** 세 시도의 `cut_headroom`은 0.0014 / 0.0056 / 0.0117로, 남은 필요분(0.0173~0.0237)의 8%·24%·68%에 불과합니다. 세 시도 모두 `recall` < `specificity`(예: 0.9309 vs 0.9767)이므로 양성 클래스에 가중을 더 주는 방향은 맞지만, 그 방향으로 얻을 수 있는 최대치가 `cut_headroom`이 말하는 그만큼입니다. `class_weight`/`scale_pos_weight`만으로는 목표에 도달할 수 없습니다.

2. **관측된 ranking 상한 자체가 목표선 아래에 있습니다.** `balanced_accuracy_at_best_cut`은 0.9552(hist_gbdt, 고용량) / 0.9530(hist_gbdt, 저용량+강정규화) / 0.9596(xgboost)입니다. 즉 *어떤 임계값을 골라도* 이 세 랭킹으로는 0.9711을 만들 수 없습니다. 목표 정의에도 이 사실이 이미 적혀 있습니다: `exceeds_ranking_ceiling=true`, `ranking_ceiling=0.9304`, `required_ks=0.9422` 대비 baseline `ks=0.8609`(`ks_shortfall=0.0813`). 목표선은 baseline에 margin 0.65를 더해 유도된 값이고, 그 값은 baseline의 랭킹 상한을 넘어서도록 설정되어 있었습니다.

Critic 판정 2건의 패턴은 일관됩니다. 1회차는 `overfitting`(train `balanced_accuracy`=1.0, `train_val_gap`=0.0462)을 지적하고 capacity 축소를 처방했고, 처방은 의도한 대로 작동했습니다 — train 포화가 1.0→0.9914로 내려갔고 `train_val_gap`도 0.0462→0.0440으로 줄었습니다. 그런데 검증 랭킹은 오르지 않았습니다(`roc_auc` 0.9840→0.9851은 이 슬라이스의 해상도 이하, `balanced_accuracy_at_best_cut`은 오히려 0.9552→0.9530). 2회차 판정은 이를 근거로 `wrong_model_family`로 옮겨 family 교체를 처방했고, 3회차 `xgboost`는 상한을 0.9596까지 올렸지만 여전히 0.9711에는 0.0115 부족했고 실제 `balanced_accuracy`는 iteration 1과 구분되지 않았습니다. 다시 말해 **진단-재계획 루프는 두 축(과적합/ranking)을 정확히 짚었지만, 이 executor에게 남아 있던 레버(같은 family 내 capacity, 다른 tree family, 클래스 가중)로는 목표선이 요구하는 크기의 ranking 개선을 만들 수 없었습니다.** 실행 환경의 참고 수치(family 교체의 `roc_auc` 기여 폭 0.0022~0.0077, 동일 family 재튜닝 0.0032, 해상도 0.003~0.006)에 비추어도, 필요한 개선폭은 이 레버들의 규모를 넘습니다.

부수적으로: 비용은 제약이 아니었습니다(최대 5.1초 / 600초). 데이터에 결측이 전혀 없어(`overall_rate`=0.0) `missing_indicator`·`missing_count`·`impute: none` 계열 레버는 애초에 얻을 것이 없는 영역이었고, 실제로 세 시도 모두 사용하지 않았습니다. 확률 품질도 병목이 아닙니다(`brier` 0.0339~0.0346, `calibration_error` 0.0141~0.0252) — 잘 정렬된 랭킹에 임계값이 거의 최적으로 놓인 상태이며, 남은 것은 정렬 자체의 품질입니다. 또한 예산 2회가 미사용으로 남았는데, 이는 정체 조기 종료 규칙 때문이며 남은 시도가 위 상한 문제를 해결했을 것이라는 근거도 이 실행에는 없습니다.

## 다음 단계 제안

우선순위 순으로, 이번 실행의 수치가 지지하는 범위에서만 제안합니다. (기록된 데이터 주의사항은 없으므로 특정 열·값·split을 신뢰할 수 없다는 제약은 이번엔 적용되지 않습니다.)

1. **목표선 0.9711의 타당성을 먼저 재검토하십시오 (가장 중요).** 이 기준은 baseline 0.9173 + margin 0.65로 유도되었고, 목표 정의 자체가 `exceeds_ranking_ceiling=true`, `ks_shortfall=0.0813`을 기록하고 있습니다. 실제 관측된 최고 랭킹 상한은 `xgboost`의 0.9596이며, 이는 임계값을 자유롭게 고를 수 있어도 0.9711에 닿지 못한다는 뜻입니다. 모델 탐색을 더 돌리기 전에 (a) margin을 낮춰 실현 가능한 목표로 재유도하거나, (b) 0.9711을 유지하려면 아래 2번처럼 **feature 쪽**을 바꾸는 결정을 내려야 합니다. 그렇지 않으면 남은 예산은 해상도 이하의 차이를 쫓게 됩니다.

2. **feature 표현을 업스트림에서 바꿔 ranking 상한을 올리십시오 — 이것은 executor 밖의 작업입니다.** 남은 격차의 76~92%가 ranking 축에 있고(1번 근거), 동일 family 재튜닝과 tree family 교체는 이미 각각 한/두 점씩 시도되어 상한이 0.953~0.960에 머물렀습니다. 반면 `spambase`의 57개 열은 거의 전부 `skew: high`인 빈도 변수이고(`word_freq_*`, `char_freq_*`, `capital_run_length_*`), executor는 로그 변환·비율·상호작용 생성이 원천적으로 불가합니다("Derive, encode or drop features"는 금지 목록). 따라서 `log1p` 변환열, `capital_run_length_total / capital_run_length_average` 같은 비율, 상위 상관 열(`word_freq_remove`, `word_freq_your`, `word_freq_000`, `char_freq_%24`)의 파생 조합을 **새 CSV/새 dataset card로 만들어** 다음 실행에 투입하는 것이 유일하게 남은 규모 있는 레버입니다. 단, 그 기대 효과는 이번 실행에서 측정된 바가 없으므로 크기를 약속할 수 없고, 검증해야 할 가설로 다루십시오.

3. **남은 예산을 쓸 경우, 아직 시도되지 않은 tree family 1점 + 양성 가중 1점으로만 좁혀 쓰십시오.** `random_forest`/`extra_trees` 같은 미시도 family를 기본 전처리(`impute="median"`, `scale=false`) 그대로 1회 적합해 `balanced_accuracy_at_best_cut`을 읽으면, "이 executor의 랭킹 상한이 0.96 근처에서 평탄하다"는 가설을 세 번째 독립 관측으로 확정할 수 있습니다. 이어서 최고 랭킹을 낸 구성(현재 `xgboost`, `cut_headroom`=0.0117)에 `scale_pos_weight`를 1.3에서 1.6~1.8로 올려 `recall`(0.9227) / `specificity`(0.9731) 간 불균형만 정리하면, 상한 대비 최대 0.0117을 회수할 수 있습니다 — 목표선까지는 부족하지만 최종 산출물의 실측 점수는 개선됩니다.

4. **평가 해상도를 의사결정 기준에 명시하십시오.** 검증 슬라이스 920행에서 `balanced_accuracy`의 95% CI 폭은 약 ±0.014이고, 이번 실행의 시도 간 차이(0.0064, 0.0005)는 모두 그 아래였습니다. cross-validation이나 seed 평균은 이 executor에서 불가하므로, 향후 실행에서는 (a) 채택/기각 판단을 `roc_auc`·`balanced_accuracy_at_best_cut` 등 ranking 지표와 paired Δ의 CI로 내리고, (b) 0.006 미만의 `balanced_accuracy` 차이를 개선으로 기록하지 않는 규칙을 두는 것을 권합니다. 이번처럼 정체 판정이 사실상 "잡음 안에서의 진동"일 때 예산 2회가 남는 상황을 줄일 수 있습니다.