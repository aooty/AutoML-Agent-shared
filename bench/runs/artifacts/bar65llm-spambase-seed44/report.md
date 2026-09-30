# AutoML 실행 최종 보고서 — `spambase` (binary_classification, 목표 지표 `balanced_accuracy`)

## 요약

**목표는 달성되지 못했습니다.** 목표선은 `balanced_accuracy` ≥ 0.967이었고, 5회 시도 중 최고 성능은 iteration 3의 `extra_trees`로 검증 점수 **0.9384 (95% CI 0.9210~0.9547)**, 한 번도 사용되지 않은 홀드백 테스트 20%에서 **0.9461 (95% CI 0.9290~0.9621)** 이었습니다 — 테스트 구간의 상단(0.9621)조차 목표선에 닿지 못합니다. 반복은 5회 모두 소진되었고(`max_iterations`), critic 진단은 4회 생성되었습니다. baseline(`logreg`, 0.9058)은 분명히 넘었지만(테스트 기준 +0.04 수준), 목표선까지 남은 거리는 랭킹 축에 있어 5회 예산 안에서 좁혀지지 않았습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 (`balanced_accuracy`, 검증) | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=500`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=10`, `l2_regularization=1.0`, `early_stopping=false` | 0.9302 (CI 0.9118~0.9468), `roc_auc` 0.9788, `balanced_accuracy_at_best_cut` 0.9369, `cut_headroom` 0.0067, `train_val_gap` 0.0689 | `overfitting` — train 0.9991 / gap 0.0689, 남은 격차의 82%가 랭킹 축 |
| 2 | `xgboost` | `n_estimators=700`, `learning_rate=0.035`, `max_depth=4`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `reg_lambda=5.0` | 0.9238 (CI 0.9046~0.9411), `roc_auc` 0.9756, best-cut 0.9344, `cut_headroom` 0.0106, `train_val_gap` 0.0539 | `wrong_model_family` — 용량을 줄여 train 0.9991→0.9777이 됐으나 검증은 오르지 않음 → `overfitting` 가설 기각 |
| 3 | `extra_trees` | `n_estimators=800`, `min_samples_leaf=1`, `class_weight=null` | **0.9384 (CI 0.9210~0.9547)**, `roc_auc` 0.9816, `pr_auc` 0.9736, best-cut 0.9461, `cut_headroom` 0.0077 | `wrong_model_family` — 3개 계열 best-cut 폭 0.0117, 최적 컷에서도 목표선까지 0.0209 부족 |
| 4 | `random_forest` | `n_estimators=1000`, `min_samples_leaf=2`, `class_weight=null` (`unsupported_claims: ["resampling"]`) | 0.9219 (CI 0.9009~0.9391), `roc_auc` 0.9751, best-cut 0.9355, `cut_headroom` 0.0136 | `wrong_model_family` — iteration 3 대비 페어 Δ −0.0166 (CI −0.0280~−0.0055, P(개선) 0.003), 이 실행에서 유일하게 해상된 차이 |
| 5 | `extra_trees` | `n_estimators=1500`, `min_samples_leaf=1`, `max_features=0.4`, `bootstrap=false`, `class_weight=null`, `random_state=44` | 0.9353 (CI 0.9190~0.9521), `roc_auc` 0.9798, best-cut 0.9384, `cut_headroom` 0.0031 | (마지막 시도 — 평가 직후 루프 종료, 진단 없음) |

전처리는 5회 모두 동일하게 적용되었습니다: `impute: median`, `scale: false`, `missing_indicator: false`, `missing_count: false`. `dropped_hyperparams`는 전 시도에서 비어 있고 학습 시간은 3.6~6.0초로, 자원 제약에 걸린 시도는 없습니다.

## 최고 성능 구성

iteration 3에서 실제로 실행된 구성입니다(값은 executor가 만든 파이프라인에서 읽힌 적용값).

- **model**: `extra_trees`
- **hyperparams**: `n_estimators=800`, `min_samples_leaf=1`, `class_weight=null`
- **preprocessing** (해당 시도의 `preprocessing` 블록): `impute: median`, `scale: false`, `missing_indicator: false`, `missing_count: false`
- **프로토콜**: stratified 3-way split, seed 44, train 60% / val 20% / test 20%
- **train_time_sec**: 4.464

| 지표 | 검증(20%) | 홀드백 테스트(20%) |
|---|---|---|
| `balanced_accuracy` | **0.9384** (CI 0.9210~0.9547) | **0.9461** (CI 0.9290~0.9621) |
| `roc_auc` | 0.9816 | — |
| `pr_auc` / `average_precision` | 0.9736 | — |
| `f1` | 0.9263 | — |
| `accuracy` | 0.9424 | — |
| `precision` / `recall` | 0.9328 / 0.9199 | — |
| `specificity` | 0.9570 | — |
| `balanced_accuracy_at_best_cut` / `cut_headroom` | 0.9461 / 0.0077 | — |
| `brier` / `calibration_error` | 0.0477 / 0.0435 | — |
| `train_balanced_accuracy` / `train_val_gap` | 0.9991 / 0.0606 | — |

검증 0.9384와 테스트 0.9461의 차이 **−0.0077**(검증이 더 낮음)이 이 실행의 선택 편향 크기입니다 — 루프는 오직 검증 숫자만 보고 최고 시도를 골랐고, 테스트 20%는 첫 학습 전에 떼어놓아 어떤 계획·진단에도 쓰이지 않았습니다. 다만 검증 점수가 테스트 CI(0.9290~0.9621) 안에 들어 있어, 이 행 수로는 그 차이가 0과 구분되지 않습니다. 어느 쪽 숫자로 읽어도 0.967에는 미달입니다.

## 원인 분석

**1) 목표선은 처음부터 이 열들의 도달 가능한 랭킹 위에 있었습니다.** 목표 정의 자체가 `exceeds_ranking_ceiling: true`, `required_ks: 0.934`, `ks_shortfall: 0.0874`를 기록하고 있습니다(baseline KS 0.8466, baseline `ranking_ceiling` 0.9233). 실행이 끝난 뒤에도 같은 그림입니다: 4개 계열이 만든 랭킹의 `balanced_accuracy_at_best_cut`은 0.9344~0.9461 범위이고, 그중 최고(iteration 3, KS 0.8922)조차 **최적 컷에서 목표선까지 0.0209 부족**합니다. 이는 계열 간 best-cut 전체 폭(0.0117)의 약 1.8배입니다.

**2) 운영점(operating point) 레버는 사실상 비어 있었습니다.** `cut_headroom`은 0.0031~0.0136으로 남은 격차의 18~30%에 불과했고, 최고 시도에서 `recall` 0.9199 vs `specificity` 0.9570은 0.037 차이입니다. 즉 `class_weight`/`scale_pos_weight`로 컷을 옮겨도 남은 거리의 70% 이상은 그대로 남습니다. 4회 진단 모두 이 축 분해를 근거로 임계값 레버를 배제했고, 실제로 어떤 시도도 `class_weight`를 건드리지 않았습니다 — 이 판단은 숫자와 일치합니다.

**3) 시도 간 차이는 대부분 이 데이터가 구분하지 못하는 크기입니다.** 5개 시도의 검증 CI(폭 약 ±0.017~0.019)는 서로 광범위하게 겹칩니다. iteration 3 vs iteration 1의 페어 Δ는 +0.0082 (CI −0.0013~+0.0179, P(개선) 0.940)로 **해상되지 않은 움직임**이며, 따라서 "boosting → bagging 전환이 랭킹을 올렸다"고 보고할 근거는 없습니다. iteration 5(0.9353)와 iteration 3(0.9384)의 −0.0031도 같은 이유로 차이가 아닙니다. 이 실행에서 구간보다 큰, 유일하게 해상된 차이는 **iteration 4 `random_forest` vs iteration 3 `extra_trees`의 −0.0166 (CI −0.0280~−0.0055, P(개선) 0.003)** 하나뿐입니다 — bootstrap + best-of-`sqrt` 분할 탐색이 이 행들을 완전 무작위 분할의 fully-grown extra-trees보다 **더 나쁘게** 정렬했다는 것이고, 네 번째로 구조가 다른 랭킹 생성기를 투입해도 랭킹 천장이 올라가지 않았다는 뜻입니다.

**4) `overfitting` 진단은 실험으로 기각되었습니다.** iteration 1의 진단(train 0.9991, `train_val_gap` 0.0689)에 따라 iteration 2는 용량을 줄였고, train은 0.9991→0.9777로 실제로 내려갔지만 검증은 오르지 않았습니다(0.9302→0.9238, 역시 CI 겹침). 이후 iteration 3~5에서 train이 다시 0.9991로 포화됐음에도 최고 점수가 나왔습니다. 즉 이 파일에서 train 포화는 성능을 제약한 활성 결함이 아니며, critic이 iteration 2 이후 `overfitting`을 반복하지 않고 랭킹 축으로 진단을 옮긴 것은 근거에 부합합니다.

**5) 특성 축 레버는 이 카드에서 물리적으로 존재하지 않았습니다.** `missing.overall_rate` 0.0, `columns_with_missing` 0이므로 `missing_indicator`/`missing_count`는 상수 열만 추가하고, `impute: none`은 bagging 계열에서 사용할 수 없습니다. iteration 2의 critic이 처방했던 "`impute: none` + `missing_count`" 방향은 iteration 3의 planner가 근거를 들어 기각했고, 그 판단이 옳았습니다 — 그 처방을 그대로 실행했다면 한 회차를 아무 정보 없이 태웠을 것입니다. executor는 파생/조합/삭제 특성을 만들 수 없으므로, 4개 계열이 소진된 뒤 랭킹 축에 남은 레버가 없었습니다.

**참고 — iteration 4의 `unsupported_claims: ["resampling"]`**: 이 플래그는 계획 산문에 대한 부분 문자열 검사이고, 이 기록에는 해당 iteration의 계획 본문이 남아 있지 않아 계획이 리샘플링에 *의존*했는지 단지 *언급*했는지 확인할 수 없습니다. 실제 실행 파라미터(`n_estimators=1000`, `min_samples_leaf=2`, `class_weight=null`)에는 리샘플링 요소가 없고 `dropped_hyperparams`도 비어 있으므로, iteration 4의 점수를 이 플래그 탓으로 돌리지 않습니다. 다만 리샘플링은 executor가 제공하지 않는 기능이므로, 그것이 필요하다는 가설은 아래 "다음 단계"의 하네스 항목으로 넘깁니다.

**6) 부수 관찰(진단 아님)**: 최고 모델의 `calibration_error` 0.0435 / `brier` 0.0477 — 확률이 평균 4%p 정도 어긋나 있습니다. 이 실행에는 재보정 레버가 없고 목표 지표도 확률을 직접 쓰지 않으므로 점수 부족의 원인으로 볼 수는 없지만, 확률값을 그대로 의사결정에 쓰려면 별도 처리가 필요하다는 뜻입니다.

## 다음 단계 제안

(이 데이터셋에는 별도로 기록된 caveat이 없으므로, 아래 제안은 카드의 집계값과 위 측정치만을 근거로 합니다.)

1. **목표선 0.967의 타당성을 먼저 재협상할 것.** 목표 정의 자체가 `exceeds_ranking_ceiling: true`, `required_ks: 0.934`를 기록하고 있고, 4개 계열·5회 시도 후 관측된 최고 랭킹은 KS 0.8922 / best-cut 0.9461입니다. 즉 이 57개 열과 현재 executor 능력 범위 안에서 0.967은 "튜닝으로 좁힐 격차"가 아니라 **정보 상한 위쪽**입니다. 홀드백 테스트에서 실제로 확인된 값 0.9461 (CI 0.9290~0.9621)을 근거로, 목표선을 0.95 부근으로 재설정하거나(그 경우 현재 구성이 이미 근접) 목표를 유지하려면 아래 2·3의 데이터/하네스 확장을 예산에 포함해야 합니다.
2. **랭킹 축에 정보를 더하는 유일한 경로는 열 자체를 바꾸는 것 — 카드 상류에서 파생 특성을 만들어 넣을 것.** executor는 비율·상호작용·로그 변환을 만들 수 없고(`missing_*` 두 열이 유일한 예외), 미결측 파일이라 그 두 열도 무효입니다. 카드의 57개 열은 거의 전부 `magnitude: sub_unit`, `skew: high`이며 `capital_run_length_average/longest/total`만 `unit`/`tens`입니다. 따라서 상류 파이프라인에서 (a) 빈도 열의 `log1p` 변환, (b) `capital_run_length_*` 3개의 비율·정규화 조합, (c) 강한 상관 열(`word_freq_remove`, `word_freq_your`, `word_freq_000`, `char_freq_%24`)의 상호작용 항을 **새 열로 추가한 새 카드**를 만들고, 동일 seed 44 프로토콜에서 `extra_trees`(iteration 3 설정)로 재측정하는 것이 다음 회차의 가장 큰 미탐색 레버입니다. 근거: 계열 교체는 이미 4회 소진되어 best-cut 폭 0.0117에 그쳤고, 유일하게 해상된 계열 차이는 오히려 하락(−0.0166)이었습니다.
3. **트리 계열과 구조가 다른, `scale: true`를 전제로 하는 랭킹 생성기를 1회 투입할 것.** 실행된 4개 계열은 모두 축 평행 분할 트리(boosting 2종, bagging 2종)로, 이 열들의 고도로 치우친 분포에서 서로 0.0117 폭 안에 갇혔습니다. 모델 목록에 남아 있는 스케일 민감 계열(선형/커널/신경망 계열 중 목록에 실제로 존재하는 것)을 `scale: true`로 한 번 시도해 `roc_auc`와 `balanced_accuracy_at_best_cut`을 iteration 3의 0.9816 / 0.9461과 비교하면, "0.0209 부족이 데이터 상한인가, 트리 계열 편향인가"가 구분됩니다. 단, baseline `logreg`가 이미 `roc_auc` 0.9672 / best-cut 0.9233이라는 점을 감안하면 기대 이득은 크지 않으므로 1회로 제한하고, 2번 항목 뒤에 배치할 것.
4. **해상력 문제를 하네스 쪽에서 고칠 것 — 검증 슬라이스 1개(20%)로는 0.008급 차이를 판정할 수 없습니다.** 이번 실행에서 5개 시도 중 4개가 서로 구분되지 않았고(CI 폭 약 ±0.018), 그 결과 iteration 4가 되어서야 처음으로 해상된 차이가 나왔습니다. 반복 분할 또는 교차검증은 현재 executor가 제공하지 않으므로 **런 설정/하네스 변경 과제**로 올려야 하며, 이것이 없으면 다음 예산도 구분 불가능한 점수들을 비교하며 소진될 위험이 큽니다. 같은 맥락에서, 현재 불가능한 레버(임계값 탐색, 확률 재보정, 리샘플링, 모델 결합)를 계획에 넣기 전에 **먼저 executor 기능으로 만들 것인지**를 결정해야 합니다 — 다만 `cut_headroom` 0.0031~0.0136이 말하듯 임계값·가중치 계열 레버는 만들어도 이 격차의 30% 이상은 메우지 못합니다.