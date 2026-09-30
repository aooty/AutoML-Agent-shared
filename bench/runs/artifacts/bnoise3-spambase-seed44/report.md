# AutoML 실행 최종 보고서 — `spambase` / balanced_accuracy

## 요약

**목표는 달성되지 못했습니다.** 목표 기준은 `balanced_accuracy` ≥ 0.967(maximize)이었으나, 5회 반복 전체에서 최고 검증 점수는 iteration 3의 `extra_trees`가 기록한 **0.9380 (95% CI 0.9206~0.9542)**로 기준선에서 0.0290 부족했습니다. 한 번도 사용되지 않은 held-back 테스트 20%에서 같은 모델은 **0.9434 (95% CI 0.9263~0.9593)**를 기록했지만 이 역시 0.967에 미치지 못합니다. 5회 시도 모두 카드의 baseline(`logreg`, balanced_accuracy 0.9058)을 상회했으나, 4회의 critic 진단이 일관되게 지적한 대로 부족분의 대부분은 threshold나 class_weight가 아니라 **랭킹(ordering) 성능의 한계**에 있었습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `early_stopping=false` | balanced_accuracy **0.9348** (CI 0.9165~0.9512), roc_auc 0.9793, pr_auc 0.9601, `balanced_accuracy_at_best_cut` 0.9408, `cut_headroom` 0.006019, `train_val_gap` 0.064461 | `overfitting` — train 0.9992 vs val 0.9348, cut은 이미 거의 최적이라 남은 0.0322의 81%가 랭킹 축에 있음 |
| 2 | `xgboost` | `n_estimators=1200`(early stop @682), `learning_rate=0.025`, `max_depth=4`, `subsample=0.8`, `reg_lambda=5.0`, `early_stopping_rounds=60` | balanced_accuracy **0.9293** (CI 0.9112~0.9452), roc_auc 0.9784, pr_auc 0.9641, best_cut 0.9341, `cut_headroom` 0.004844, `train_val_gap` 0.047876 | `wrong_model_family` — paired Δ vs it.1 = -0.0055 (CI -0.0138~+0.0034, P(better) 0.083)로 판별 불가. gap은 닫혔으나(0.0645→0.0479) 검증 점수는 따라오지 않음 |
| 3 | `extra_trees` | `n_estimators=1000`, `min_samples_leaf=1`, `class_weight=null` | **최고** balanced_accuracy **0.9380** (CI 0.9206~0.9542), roc_auc 0.9816, pr_auc 0.9739, best_cut 0.9434, `cut_headroom` 0.005446, `train_val_gap` 0.061127 | `wrong_model_family` — paired Δ vs it.1 = +0.0032 (CI -0.0070~+0.0135, P(better) 0.720)로 판별 불가. 3개 family의 best_cut 폭이 0.9341~0.9434(0.0093)에 불과 |
| 4 | `svc` | `kernel='rbf'`, `C=10.0`, `class_weight=null`, `scale=true` | balanced_accuracy **0.9188** (CI 0.8984~0.9371), recall 0.8895 / specificity 0.948029, `train_val_gap` 0.047862 (roc_auc·pr_auc·cut 지표 미보고) | `wrong_model_family` — paired Δ vs it.3 = **-0.0192 (CI -0.0355~-0.0031, P(better) 0.003)**, 이 실행에서 유일하게 해석 가능한 유의한 하락. 단, family와 `scale` 두 레버를 동시에 움직여 원인 귀속 불가 |
| 5 | `extra_trees` | `n_estimators=2000`, `min_samples_leaf=1`, `max_features=0.35`, `class_weight=null`, `bootstrap=false`, `n_jobs=-1`, `random_state=44` | balanced_accuracy **0.9371** (CI 0.9199~0.9536), roc_auc 0.9803, pr_auc 0.9729, best_cut 0.9397, `cut_headroom` 0.002642, `train_val_gap` 0.062023 | (없음 — 평가 직후 루프 종료) |

모든 시도에서 `dropped_hyperparams`는 비어 있었고 `unsupported_claims`도 발생하지 않았습니다. 학습 시간은 3.4~7.5초로 600초 예산 대비 제약이 전혀 아니었습니다.

## 최고 성능 구성

- **model**: `extra_trees` (iteration 3)
- **hyperparams** (실제 적용값): `n_estimators=1000`, `min_samples_leaf=1`, `class_weight=null`
- **preprocessing** (실제 적용값): `impute='median'`, `scale=false`, `missing_indicator=false`, `missing_count=false`
- **학습 시간**: 5.267초
- **프로토콜**: stratified 60/20/20 split, seed 44 (카드의 baseline과 동일 split)

| 지표 | 값 |
|---|---|
| balanced_accuracy (검증) | **0.9380** (95% CI 0.9206~0.9542) |
| balanced_accuracy (held-back 테스트 20%) | **0.9434** (95% CI 0.9263~0.9593) |
| accuracy / f1 | 0.9424 / 0.9261 |
| precision / recall / specificity | 0.9352 / 0.9171 / 0.958781 |
| roc_auc / pr_auc | 0.9816 / 0.9739 |
| balanced_accuracy_at_best_cut / cut_headroom | 0.9434 / 0.005446 |
| brier / calibration_error | 0.047744 / 0.044216 |
| train_balanced_accuracy / train_val_gap | 0.9991 / 0.061127 |

테스트 점수는 검증 점수보다 0.0054 **높습니다**(검증 대비 -0.0054로 기록됨). 이 차이가 선택 편향의 크기이며, 루프의 모든 선택은 검증 숫자만 보고 이루어졌습니다. 다만 검증 점수 0.9380은 테스트 CI(0.9263~0.9593) 안에 들어 있으므로, 이 행 수로는 그 차이가 0과 구분되지 않습니다. **어느 쪽 숫자로도 0.967 기준을 넘지 못합니다.**

## 원인 분석

critic은 4회 실행되었고, 진단은 `overfitting` 1회 + `wrong_model_family` 3회였습니다. 그 판정들이 만든 패턴은 다음과 같습니다.

1. **부족분은 운영점(operating point)이 아니라 랭킹 축에 있었습니다.** 최고 시도의 `cut_headroom`은 0.005446으로, 남은 0.0290 중 약 19%만이 cut 이동으로 도달 가능한 거리였습니다. recall 0.9171 vs specificity 0.958781의 불균형도 그 0.0054 안에 전부 들어갑니다. 즉 `class_weight`나 `scale_pos_weight` 이동으로 살 수 있는 폭은 남은 거리의 1/5 이하이며, 이는 iteration 1·2·3의 critic이 각각 19%·13%·19%로 반복 확인했습니다. iteration 5에서는 `cut_headroom`이 0.002642까지 더 줄었습니다.

2. **랭킹 성능은 family를 바꿔도 거의 움직이지 않았습니다.** 3개 트리 계열(`hist_gbdt`, `xgboost`, `extra_trees`)의 `balanced_accuracy_at_best_cut`은 0.9341~0.9434, 폭 0.0093에 불과했고, 그 최고값에서 목표까지의 거리 0.0236은 그 폭의 약 3배입니다. KS 기준으로 보면 목표가 요구하는 0.934 대비 달성치는 약 0.8868이며, 목표 정의 자체가 `exceeds_ranking_ceiling: true`, `ks_shortfall: 0.0874`로 baseline 랭킹으로는 통과 불가함을 미리 표시하고 있었습니다. roc_auc 0.9816 / pr_auc 0.9739는 이미 baseline(0.9672 / 0.9394)을 크게 넘어선 값입니다.

3. **iteration 1·2·3·5는 이 데이터로는 서로 구별되지 않습니다.** 검증 CI 폭이 약 0.033~0.039인데 네 시도의 점수는 0.9293~0.9380 범위에 모여 있고, paired Δ의 CI가 모두 0을 포함합니다(it.2 vs it.1: -0.0055, CI -0.0138~+0.0034 / it.3 vs it.1: +0.0032, CI -0.0070~+0.0135). 따라서 "extra_trees가 hist_gbdt보다 좋았다"거나 "iteration 5의 튜닝이 손해였다"는 서술은 이 슬라이스가 지지하지 않습니다. 해석 가능한 유일한 차이는 iteration 4의 `svc` 하락(-0.0192, CI -0.0355~-0.0031)이며, 이마저도 family 교체와 `scale=true`를 동시에 바꿨기 때문에 어느 레버 탓인지 귀속할 수 없습니다.

4. **`overfitting` 진단은 시험되었고, 값을 주지 않았습니다.** iteration 2에서 정규화를 강화해 `train_val_gap`을 0.0645→0.0479, train balanced_accuracy를 0.9992→0.9771로 실제로 낮췄으나 검증 점수는 오르지 않았습니다(오히려 판별 불가 범위 내 소폭 하락). 최고 시도인 iteration 3은 다시 `train_val_gap` 0.0611의 "암기형" 모델이었는데도 최고 점수를 냈습니다. 즉 이 데이터에서 capacity 통제는 목표 지표에 값을 주지 않는 레버로 측정되었습니다.

5. **재계획의 실행 충실도 문제**: critic 1은 "`hist_gbdt`를 정규화해 재적합(단일 축 변경)"을 지시했으나 iteration 2는 `xgboost`로 family까지 바꿨고, critic 2는 `random_forest`를 지시했으나 iteration 3은 `extra_trees`를, critic 4는 "iteration 3을 그대로 재적합해 확인"을 지시했으나 iteration 5는 `n_estimators=2000`, `max_features=0.35`, `bootstrap=false`로 여러 값을 함께 바꿨습니다. 그 결과 (a) 정규화 단독 효과와 (b) iteration 3 점수의 재현성 확인, 두 가지 정보를 이 실행은 확보하지 못했습니다. 이는 성능 부족의 주된 원인은 아니지만, 5회 중 2회의 진단 가치가 희석된 원인입니다.

또한 실행 환경상 파생 피처 생성이 불가능하므로(`missing_indicator`/`missing_count` 외 어떤 조합·변환·삭제도 불가), 57개 열이 지지하는 랭킹 상한 안에서만 탐색이 이루어졌다는 구조적 제약이 있습니다. 캘리브레이션은 `calibration_error` 0.0442 수준으로 완벽하지 않지만, 목표 지표가 랭킹+cut으로 결정되고 recalibration 레버도 없으므로 이번 부족분의 설명 요인은 아닙니다.

## 다음 단계 제안

(데이터 카드에 별도 caveat이 기록되지 않았으므로, 아래 제안을 무효화하는 열·값·split 관련 경고는 없습니다.)

1. **목표 기준 0.967 자체를 재검토하십시오 — 우선순위 1.** 이 기준은 derived이며 메타데이터가 이미 `exceeds_ranking_ceiling: true`, `required_ks: 0.934` vs baseline `ks: 0.8466`(`ks_shortfall: 0.0874`)를 명시하고 있습니다. 실측으로도 3개 family의 최적 cut 상한이 0.9341~0.9434(폭 0.0093)에 머물렀고, held-back 테스트에서도 0.9434(CI 0.9263~0.9593)였습니다. 즉 이 57개 열·이 실행기의 레버 집합에서 0.967은 도달 대상이 아니라 재산정 대상입니다. 현실적 목표를 0.94~0.95대로 재설정하면, 이번 최고 구성이 이미 그 수준의 증거를 제공합니다.

2. **평가 프로토콜의 해상도를 먼저 키우십시오.** 검증 CI 폭 약 0.033에서 5회 중 4회가 서로 구별되지 않았습니다. 현재 실행기는 단일 20% 검증 split만 쓰므로(CV·OOF 불가), 반복 seed 또는 k-fold를 지원하도록 실행기/프로토콜을 확장해야 0.005~0.01 수준의 개선을 판별할 수 있습니다. 이것이 없으면 추가 반복은 리샘플 노이즈를 해석하는 데 소모됩니다.

3. **파생 피처 생성 능력을 실행기 밖에서 확보하십시오.** 남은 부족분의 ~81%가 랭킹 축이고, 랭킹은 family 교체로 0.0093밖에 움직이지 않았습니다. 반면 카드상 57개 열은 거의 전부 `skew: high`의 sub_unit 빈도 열이며 `capital_run_length_*`만 tens 규모입니다. 로그/제곱근 변환, 단어빈도 그룹 합계, `char_freq_%24`·`word_freq_remove`·`word_freq_000`·`word_freq_your`(target_corr strong) 조합항 같은 변환은 현재 실행기가 원칙적으로 만들 수 없습니다(파생·인터랙션·삭제 금지). 이런 열을 데이터 카드 단계에서 미리 만들어 넣는 것이 랭킹 상한을 올릴 수 있는 유일하게 남은 방향입니다.

4. **운영점·캘리브레이션 레버에는 더 이상 반복을 쓰지 마십시오.** 최고 시도의 `cut_headroom`은 0.005446, iteration 5는 0.002642이며 recall/specificity 불균형(0.9171 vs 0.9588)의 전 보정폭이 그 안에 들어갑니다. `class_weight='balanced'`나 임계값 탐색은 최대 0.005 수준의 이득만 가능하므로 0.0290 부족분을 메울 수 없습니다. 만약 4번째 family를 굳이 시도한다면 `random_forest`(critic 2가 지시했으나 끝내 실행되지 않은 유일한 구성)를 **단일 축만 바꿔** 적합해 3-family 상한 0.0093 폭이 실제 상한인지 확인하는 용도로 한정하고, iteration 4처럼 family와 `scale`을 동시에 움직이지 마십시오.