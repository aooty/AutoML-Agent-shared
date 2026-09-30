# AutoML 실행 최종 보고서 — `speeddating` / `balanced_accuracy`

## 요약

**목표는 달성하지 못했습니다.** 목표선은 `balanced_accuracy` 0.8882였고, 5회 반복(허용 5회 전부 소진) 중 가장 좋은 검증 점수는 iteration 4의 `xgboost`가 기록한 **0.7599 (95% CI 0.7317~0.7859)** 로 목표선에 0.1283 부족했습니다. 루프 종료 후 한 번도 사용되지 않은 test 20%에서 같은 모델을 측정한 값은 **0.7860 (95% CI 0.7561~0.8168)** 로, 이 역시 목표선에서 0.1022 미달입니다. 다만 baseline(`logreg`, balanced_accuracy 0.6807, CI 0.654~0.7099)은 검증과 테스트 양쪽에서 분명히 상회했습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `class_weight='balanced'`, `early_stopping=False` / `impute=none`, `scale=false` | balanced_accuracy **0.7248** (CI 0.6969~0.7527), roc_auc 0.8598, best_cut 0.7922, cut_headroom 0.0674, train_val_gap **0.2752** (train BA 1.0) | `overfitting` — 훈련 데이터를 완전 암기. 용량을 강하게 줄여 ranking(roc_auc)을 올릴 것 |
| 2 | `xgboost` | `n_estimators=800`, `learning_rate=0.04`, `max_depth=4`, `min_child_weight=10`, `subsample=0.8`, `colsample_bytree=0.6`, `reg_lambda=10.0`, `reg_alpha=1.0`, `scale_pos_weight=5.07`, `early_stopping_rounds=50` / `impute=none`, `scale=false` | balanced_accuracy **0.7585** (CI 0.7301~0.7841), roc_auc 0.8572, pr_auc 0.5787, best_cut 0.7851, cut_headroom **0.0266**, train_val_gap 0.1776 | `wrong_model_family` — iter1 대비 paired Δ +0.0337 (CI +0.0140~+0.0549)는 실재하지만 ranking은 개선 없음(roc_auc 0.8598→0.8572). 남은 거리의 80%가 ranking 축. 정보(열)를 더할 것 |
| 3 | `mlp` | `hidden_layer_sizes=[128,64]`, `alpha=0.0003`, `learning_rate_init=0.0003`, `batch_size=128`, `max_iter=400`, `early_stopping=True` / `impute=median`, `scale=true`, `missing_count=true` | balanced_accuracy **0.5704** (CI 0.5506~0.5959), roc_auc 0.8063, best_cut 0.7350, cut_headroom 0.1646, 내부 조기중단 iter 13/400 | `wrong_model_family` — paired Δ -0.1881 (CI -0.2178~-0.1602)의 확정된 후퇴. 다만 family와 preprocessing을 동시에 바꿔 원인 귀속 불가. 최고 ranking으로 복귀해 미측정 레버 1개만 분리할 것 |
| 4 | `xgboost` | iter2와 동일 + `missing_count=true` (`impute=none`, `scale=false`, `missing_indicator=false`) | balanced_accuracy **0.7599** (CI 0.7317~0.7859), roc_auc 0.8575, pr_auc 0.5830, best_cut 0.7901, cut_headroom 0.0302, train_val_gap 0.1833 | `wrong_model_family` — iter2 대비 paired Δ +0.0014 (CI -0.0093~+0.0143, P(better) 0.615): 이 데이터로 구분되지 않음. best_cut이 예고된 0.7851~0.7922 밴드 안에 머물러 ranking 상한 확인. 남은 1회는 operating point에만 쓸 것 |
| 5 | `xgboost` | `n_estimators=1200`, `learning_rate=0.03`, `max_depth=5`, `min_child_weight=8`, `subsample=0.85`, `colsample_bytree=0.8`, `reg_lambda=8.0`, `reg_alpha=0.5`, `scale_pos_weight=7.5`, `early_stopping_rounds=60` / `impute=none`, `missing_count=true` | balanced_accuracy **0.7465** (CI 0.7165~0.7751), roc_auc 0.8570, best_cut 0.7903, cut_headroom 0.0438, train_val_gap 0.2102 | (없음 — 평가 직후 루프 종료) |

## 최고 성능 구성

iteration 4에서 **실제로 실행된** 설정입니다(값은 history의 `hyperparams` / `preprocessing`, 즉 executor가 만든 파이프라인에서 읽은 적용값).

- **model**: `xgboost`
- **hyperparams**: `n_estimators=800`, `learning_rate=0.04`, `max_depth=4`, `min_child_weight=10`, `subsample=0.8`, `colsample_bytree=0.6`, `reg_lambda=10.0`, `reg_alpha=1.0`, `scale_pos_weight=5.07`, `early_stopping_rounds=50` (`dropped_hyperparams` 없음)
- **preprocessing (적용값)**: `impute="none"`, `scale=false`, `missing_indicator=false`, `missing_count=true`
- **내부 조기중단**: `fit_rows=4523`, `held_out_rows=503`, `validation_fraction=0.1`, `stopped_at_iter=750/800`
- **train_time_sec**: 7.286 (제한 600초)
- **프로토콜**: stratified 3-way split, seed 44, train 60% / validation 20% / test 20%

검증 슬라이스 지표: `balanced_accuracy` **0.7599 (CI 0.7317~0.7859)**, `roc_auc` 0.8575, `pr_auc` / `average_precision` 0.5830, `f1` 0.5705, `accuracy` 0.8419, `precision` 0.5161, `recall` 0.6377, `specificity` 0.8821, `brier` 0.1122, `calibration_error` 0.0783, `balanced_accuracy_at_best_cut` 0.7901, `cut_headroom` 0.0302, `train_val_gap` 0.1833.

**최종 유보 측정(test 20%, 첫 fit 이전에 분리되어 어떤 결정에도 쓰이지 않은 행): `balanced_accuracy` = 0.7860 (95% CI 0.7561~0.8168).** 검증값 0.7599와의 차이 -0.0261이 선택 편향의 크기이며(루프는 검증 숫자만 보고 최종 구성을 골랐습니다), 검증값이 위 test CI 안에 들어오므로 이 행 수로는 그 차이를 0과 구분할 수 없습니다. 이 실행이 실제로 입증한 성능은 검증 0.7599가 아니라 **test 0.7860**이며, 그 값도 목표선 0.8882에는 0.1022 미달입니다.

## 원인 분석

critic 판정은 4회 존재하고, 그 4회가 한 방향을 가리킵니다: **부족분의 대부분이 operating point가 아니라 ranking 축에 있고, 시도된 어떤 ranking도 목표선을 담을 수 없었다**는 것입니다.

- **목표선 자체가 측정된 ranking 상한 위에 있습니다.** goal의 `reference`가 이미 `ranking_ceiling=0.7649`, `exceeds_ranking_ceiling=true`, `required_ks=0.7764`(baseline `ks=0.5297`, `ks_shortfall=0.2467`)를 기록합니다. 실행 중 측정된 값도 같은 방향입니다: 3개 family에 걸친 `balanced_accuracy_at_best_cut`은 0.7350~0.7922(폭 0.0572)에 머물렀고, 그 최고 cut조차 목표선보다 0.0960 아래입니다. 즉 어떤 threshold를 골랐다 해도 0.8882에는 닿지 않습니다.
- **operating point 여유는 이미 거의 소진됐습니다.** iteration 4에서 `cut_headroom` 0.0302은 남은 0.1283의 24%에 불과하고, iteration 2에서는 0.0266까지 줄어 있었습니다. iteration 1(recall 0.5254 / specificity 0.9243)에서 iteration 2(0.6449 / 0.8721)로의 +0.0337은 확정된 개선이지만, 같은 구간에서 roc_auc는 0.8598→0.8572, pr_auc 0.6049→0.5787로 **ranking은 전혀 개선되지 않았습니다** — 컷이 이동했을 뿐입니다.
- **구분되지 않는 차이로 이야기를 만들지 않기 위해**: iteration 2(0.7585, CI 0.7301~0.7841), iteration 4(0.7599, CI 0.7317~0.7859), iteration 5(0.7465, CI 0.7165~0.7751)는 구간이 서로 크게 겹치며, 특히 iter2→iter4의 `missing_count=true`는 paired Δ +0.0014 (CI -0.0093~+0.0143, P(better) 0.615)로 **이 데이터로는 0과 구분되지 않습니다.** iter4→iter5의 `scale_pos_weight` 5.07→7.5 역시 구간 내 변동이며 개선으로 보고할 수 없습니다. 해상도를 넘는 차이는 두 개뿐입니다: iter1→iter2의 +0.0337(정규화로 overfitting 해소, train_val_gap 0.2752→0.1776, train BA 1.0→0.9361)과 iter3의 -0.1881(mlp 후퇴).
- **iteration 3은 진단 정보를 스스로 깎았습니다.** family(`mlp`)와 preprocessing(`impute` none→median, `scale` false→true, `missing_count` false→true)을 한 번에 바꿨기 때문에 -0.1881이 어느 레버 탓인지 귀속할 수 없습니다. 확실한 것은 ranking 축이 함께 내려갔다는 점(roc_auc 0.8572→0.8063, best_cut 0.7851→0.7350)과, 내부 조기중단이 400회 중 13회에서 발동해 사실상 학습이 진행되지 않았다는 점(train BA 0.6039, train_val_gap 0.0334)입니다. 또한 `mlp`에는 imbalance 레버가 없어 기본 0.5 컷이 recall 0.1594 / specificity 0.9814에 놓였고, 그 결과 `cut_headroom` 0.1646이 순수 낭비로 남았습니다.
- **제약은 예산도, 용량도 아니었습니다.** 최고 구성의 학습 시간은 7.3초(제한 600초)이고 `dropped_hyperparams`는 모든 시도에서 비어 있습니다. iteration 5에서 용량을 다시 키우자 train_val_gap이 0.1833→0.2102로 늘고 검증 점수는 오르지 않았으므로, gbdt 하이퍼파라미터 공간 안에서는 더 살 것이 남아 있지 않다는 iteration 2·4의 판정과 일치합니다.
- **`unsupported_claims`에 대하여**: iteration 2에 `feature_engineering`, `per_column_preprocessing`, iteration 3에 `feature_engineering`이 표시되어 있으나, 이 필드는 계획 산문에 대한 부분 문자열 검사이고 해당 iteration의 계획 본문은 이 보고서에 남아 있지 않습니다. 따라서 그 시도가 무엇을 잃었다고 귀속하지 않습니다. 사실로 말할 수 있는 것은 executor가 파생 변수·열별 전처리를 수행하지 않는다는 점이고(적용된 preprocessing 블록이 그것을 확인합니다), 따라서 **이 실행은 파생 피처가 무엇을 살 수 있었는지 한 번도 시험하지 않았습니다.** 이는 실패 원인이 아니라 다음 단계에서 만들어야 할 능력입니다.

## 다음 단계 제안

1. **목표선을 먼저 재정의하십시오(최우선).** 0.8882는 `derived`이고 그 근거 블록이 스스로 `exceeds_ranking_ceiling=true`, `required_ks=0.7764` vs `ks=0.5297`를 기록합니다. 이 실행은 3개 family·5회 시도로 best_cut 0.7350~0.7922라는 상한을 실측했고, 유보 test에서 0.7860 (CI 0.7561~0.8168)을 입증했습니다. 현재 열 구성으로는 0.8882가 도달 대상이 아니므로, 목표를 (a) baseline 대비 개선(0.6807 → 0.7860, baseline CI 상단 0.7099를 상회) 또는 (b) `roc_auc` 같은 ranking 지표 기준으로 재설정하는 것이 정직한 다음 단계입니다.
2. **ranking을 올리려면 열을 늘려야 합니다 — executor 밖에서.** 남은 부족분의 76%가 ranking 축에 있고, 이 실행이 쓸 수 있었던 정보 레버는 `missing_indicator`(`impute=none`에서는 비트 단위로 중복)와 `missing_count`(측정값 ~0, paired Δ +0.0014, CI가 0을 포함)뿐이었습니다. 실질적으로 미시험인 것은 파생 피처입니다: 카드에서 `like`는 target_corr `strong`, `attractive_partner`/`funny_partner`/`shared_interests_partner`/`guess_prob_liked` 등이 `moderate`이므로, 쌍(pair) 수준 차이·비대칭(예: 자기평가와 상대평가의 차, 상호 `like`·`guess_prob_liked` 조합)과 고카디널리티로 제외된 `field`의 축약 인코딩을 **데이터 카드 단계에서 열로 추가**한 뒤 iteration 4 구성을 그대로 재적합해 paired Δ를 읽으십시오. executor는 열을 파생하지 못하므로 이는 계획이 아니라 업스트림 작업입니다.
3. **threshold 선택 능력을 파이프라인에 추가하십시오(단, 기대치는 0.03 수준).** 최고 구성의 `cut_headroom`은 0.0302이고 3개 family의 best_cut은 ≈0.79에 고정돼 있습니다. 현재 executor는 `predict()` 기본 컷만 쓰고 threshold 탐색을 하지 않으므로, 별도 검증 분할에서 컷을 고르는 절차를 만들면 이 0.03가량을 회수할 수 있습니다 — 목표선을 넘기지는 못하지만 보고 가능한 최고치를 0.79 부근으로 올립니다. `scale_pos_weight`를 5.07→7.5로 올린 iteration 5는 이 여유를 사지 못했으므로(0.7465, 구간 내), 가중치 레버를 더 미는 것은 권하지 않습니다.
4. **missingness가 "언제 기록됐는지"를 학습하는지 별도 분할로 확인하십시오.** `expected_num_interested_in_me`의 결측률은 0.7852, 결측 열은 60개이고 카드에는 `wave`(0.0 결측, tens 규모)가 있습니다 — 결측 패턴이 응답자 상태가 아니라 wave/기록 체제를 가리킬 가능성이 있으며, executor 문서가 경고하듯 stratified random split은 그런 신호를 이득으로 채점합니다. 이번 실행에서 `missing_count`의 실측 효과가 사실상 0이었으므로 현재 최고 구성이 그 함정에 빠졌다는 증거는 없지만, 앞으로 결측 기반 열을 추가하려면 `wave` 기준 그룹 분할(현 프로토콜은 `grouped_by: null`, 변경은 런 설정 레벨의 작업)에서 같은 Δ가 재현되는지 확인하는 것이 전제입니다. 이것이 확인되면 2번의 파생 피처 실험 결과도 신뢰 구간과 함께 해석할 수 있게 됩니다.
5. **더 쓰지 말 것**: gbdt 내부 용량 재튜닝(iter1→2에서 이미 회수, iter5에서 되돌아감), `impute=none`과 함께 쓰는 `missing_indicator`(비트 단위 중복으로 문서에 확정), 그리고 median 대치 + 스케일링 기반 `mlp` 재시도(iter3에서 -0.1881, roc_auc 0.8063). 예산은 제약이 아니었으므로(7.3초 / 600초) 남는 반복은 2~4번의 준비가 끝난 뒤에 쓰는 것이 효율적입니다.