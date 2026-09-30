# AutoML 최종 리포트 — `speeddating` / `balanced_accuracy`

## 요약

목표는 달성하지 못했습니다. 목표선은 `balanced_accuracy` 0.8882(baseline logreg 0.6807 + margin 0.65에서 파생)였고, 5회 반복 전부를 소진한 뒤 최고 검증 점수는 iteration 4의 `hist_gbdt` 0.7791 (95% CI 0.7518~0.8052)로 목표선까지 0.1091 부족했습니다. 이 모델을 한 번도 사용되지 않은 test 20%에서 한 번 채점한 결과는 `balanced_accuracy` 0.7936 (95% CI 0.7694~0.8204)로, baseline(0.6807)은 확실히 넘었지만 목표선에는 여전히 미달입니다. 중단 사유는 `max_iterations`이며 critic 진단은 4회 생성되었습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `learning_rate` 0.06, `max_iter` 400, `max_leaf_nodes` 31, `max_depth` 6, `l2_regularization` 1.0, `class_weight` 'balanced', `early_stopping` false / `impute` none | `balanced_accuracy` 0.7256 (CI 0.6971~0.7535), `roc_auc` 0.8516, `balanced_accuracy_at_best_cut` 0.7810, `cut_headroom` 0.0554, `train_val_gap` 0.2721 | `unknown` (근거 서술은 명백한 과적합: train BA 0.9977 vs val 0.7256). 처방: 용량 대폭 축소 + 내부 early stopping |
| 2 | `hist_gbdt` | `learning_rate` 0.04, `max_iter` 800(early stop @253), `max_leaf_nodes` 8, `max_depth` 3, `l2_regularization` 20.0, `min_samples_leaf` 60, `class_weight` 'balanced' / `impute` none | `balanced_accuracy` 0.7751 (CI 0.7483~0.8012), `roc_auc` 0.8592, `pr_auc` 0.5988, best_cut 0.7814, `cut_headroom` 0.0063, `train_val_gap` 0.0633 | `wrong_model_family` (gap은 0.2721→0.0633으로 해소되었으나 best_cut은 0.7810→0.7814로 사실상 불변) |
| 3 | `extra_trees` | `n_estimators` 800, `min_samples_leaf` 2, `class_weight` 'balanced' / `impute` median | `balanced_accuracy` 0.6459 (CI 0.6194~0.6720), `roc_auc` 0.8473, best_cut 0.7711, `cut_headroom` 0.1252, `train_val_gap` 0.3540 | `overfitting` (iter 2 대비 paired Δ −0.1292, CI −0.1594~−0.0991 — 해상도를 넘는 명확한 퇴행) |
| 4 | `hist_gbdt` | iteration 2와 동일 지점 + `missing_count` true (early stop @211) | **`balanced_accuracy` 0.7791 (CI 0.7518~0.8052)**, `roc_auc` 0.8590, `pr_auc` 0.5981, best_cut 0.7808, `cut_headroom` 0.0017, `train_val_gap` 0.0525 | `unknown` (iter 2 대비 Δ +0.0040, CI −0.0040~+0.0121 — 이 데이터로 구분 불가. 사전 등록된 중단 조건 발동) |
| 5 | `xgboost` | `n_estimators` 1500(early stop @536), `learning_rate` 0.03, `max_depth` 5, `min_child_weight` 5, `subsample` 0.8, `colsample_bytree` 0.7, `reg_lambda` 5.0, `reg_alpha` 0.5, `scale_pos_weight` 5.07 / `impute` none, `missing_count` true | `balanced_accuracy` 0.7551 (CI 0.7261~0.7810), `roc_auc` 0.8586, best_cut 0.7856(전 구간 최고), `cut_headroom` 0.0305, `train_val_gap` 0.1928 | — (평가 직후 루프 종료, 진단 없음) |

## 최고 성능 구성

iteration 4에서 실제로 빌드된 구성(아래 값은 히스토리의 적용값 그대로):

- **model**: `hist_gbdt`
- **hyperparams**: `learning_rate` 0.04, `max_iter` 800, `max_leaf_nodes` 8, `max_depth` 3, `l2_regularization` 20.0, `min_samples_leaf` 60, `early_stopping` true, `validation_fraction` 0.1, `n_iter_no_change` 30, `class_weight` 'balanced'
- **preprocessing (적용값)**: `impute` none(모델이 NaN을 직접 분기), `scale` false, `missing_indicator` false, `missing_count` true
- **internal_validation**: `fit_rows` 4523, `held_out_rows` 503, `stopped_at_iter` 211 / 800
- **train_time_sec**: 4.801
- 프로토콜: stratified 60/20/20, seed 44 (카드 baseline과 동일 분할)

성능:

| 지표 | 값 |
|---|---|
| `balanced_accuracy` (validation) | **0.7791** (95% CI 0.7518~0.8052) |
| `balanced_accuracy` (held-back test 20%, 1회 채점) | **0.7936** (95% CI 0.7694~0.8204) |
| `roc_auc` / `pr_auc` | 0.8590 / 0.5981 |
| `f1` / `accuracy` | 0.5406 / 0.7840 |
| `recall` / `specificity` | 0.7717 / 0.7864 |
| `balanced_accuracy_at_best_cut` / `cut_headroom` | 0.7808 / 0.0017 |
| `brier` / `calibration_error` | 0.1428 / 0.1807 |
| `train_balanced_accuracy` / `train_val_gap` | 0.8316 / 0.0525 |

test 점수는 validation보다 0.0145 높습니다(리포트 상 검증 대비 −0.0145). 이 차이가 선택 편향의 크기이며, 루프는 오직 validation 숫자만 보고 최종 구성을 골랐습니다. 다만 validation 0.7791은 test CI(0.7694~0.8204) 안에 들어가므로, 이 행 수로는 두 값의 차이를 0과 구분할 수 없습니다. 어느 쪽 숫자를 쓰더라도 목표선 0.8882에는 도달하지 못합니다.

> 참고: iteration 4의 `dropped_hyperparams`에는 `missing_count`, `missing_indicator`, `scale`이 기록되어 있습니다. 이들은 estimator 파라미터가 아닌 파이프라인 플래그라 `applied_preprocessing`을 통해 반영된 것으로 critic이 읽었으며, 위 `preprocessing` 블록이 적용값입니다. 이 모호성은 `## 원인 분석`에서 다룹니다.

## 원인 분석

**1. 부족분의 거의 전부가 랭킹 축에 있고, 랭킹은 어떤 레버로도 움직이지 않았습니다.**
5회 시도의 `balanced_accuracy_at_best_cut`은 0.7711~0.7856 사이에 갇혀 있습니다(폭 0.0145). 목표선 0.8882는 KS 0.7764를 요구하는데 이 실행에서 관측된 최고 랭킹은 `roc_auc` 0.8592 / best_cut 0.7856 수준으로, KS 기준 약 0.21의 격차가 남습니다. `cut_headroom`은 iteration 2·4에서 각각 0.0063, 0.0017까지 줄어 있고 `recall` 0.7717 ≈ `specificity` 0.7864이므로, `class_weight` / `scale_pos_weight`(이 실행기에서 유일한 불균형 레버)로 얻을 수 있는 여지는 이미 소진되었습니다. 즉 남은 0.1091은 결정 규칙 문제가 아니라 **모델이 행을 정렬하는 능력, 곧 컬럼의 문제**입니다.

**2. critic 4건의 패턴: 진단은 맞았고 처방은 의도한 축을 정확히 움직였지만, 그 축이 목표선까지 이어지지 않았습니다.**
- iteration 1→2: 과적합 진단이 정확히 적중했습니다(`train_val_gap` 0.2721→0.0633, paired Δ +0.0495, CI +0.0244~+0.0763 — 해상도를 넘는 실제 개선). 그러나 개선분의 대부분은 operating point였고 best_cut은 0.7810→0.7814로 사실상 제자리였습니다.
- iteration 2→3: `wrong_model_family` 처방으로 `extra_trees`를 시도했으나 −0.1292(CI −0.1594~−0.0991)의 명확한 퇴행이었고, `train_val_gap` 0.3540으로 이 실행 최악의 과적합이었습니다. 이 행은 family와 `impute`(none→median)를 동시에 바꿨기 때문에 손실의 귀속도 불가능합니다 — 한 번에 한 축만 바꾸라는 규칙을 어긴 유일한 시도입니다.
- iteration 3→4: 유일하게 남은 랭킹 레버였던 `missing_count`는 Δ +0.0040 (CI −0.0040~+0.0121)로 **이 데이터가 구분하지 못하는 움직임**이었습니다. 실행기 문서가 예고한 대로(`missing_count`는 다른 표본에서 0.0003, 0.0000) 컬럼 한 개로는 아무 것도 사지 못했습니다.
- iteration 4→5: `xgboost` family swap도 `roc_auc` 0.8586으로 iteration 2·4(0.8592, 0.8590)와 사실상 동일했습니다.

**3. 구간이 겹치는 시도들 사이에서 서사를 만들지 않겠습니다.**
iteration 2 (0.7751, CI 0.7483~0.8012), iteration 4 (0.7791, CI 0.7518~0.8052), iteration 5 (0.7551, CI 0.7261~0.7810)의 세 구간은 서로 크게 겹칩니다 — 이 20% 슬라이스로는 **구분되지 않는 세 시도**이며, "iteration 4가 최고"라는 선택은 폭 0.05짜리 구간 안의 0.004 차이에 기반한 것입니다. 해상도를 넘는 차이는 두 개뿐입니다: iteration 1→2의 정규화 효과(+0.0495)와 iteration 3의 퇴행(−0.1292). 결론적으로 이 실행이 확실히 보여준 것은 "용량을 줄이면 좋아진다", "`extra_trees`+median impute는 나쁘다" 두 가지이고, 나머지 family·컬럼 실험은 모두 미해결입니다.

**4. 실행기 제약이 랭킹 상한을 사실상 고정했습니다.**
이 실행기는 파생 피처를 만들지 않습니다(비율·차이·상호작용·재인코딩·컬럼 삭제 불가, 추가 가능한 것은 `missing_indicator`/`missing_count` 두 개뿐). 카드에서 `field`는 high-cardinality로 이미 드롭되었고, 수치 컬럼 61개 대부분은 `target_corr`이 `none`이며 `strong`은 `like` 하나입니다. 즉 5회 반복 전부가 **동일한 고정 피처 행렬 위의 가설 공간**을 탐색했고, 그 공간의 랭킹 상한이 best_cut ~0.78~0.79로 관측된 것입니다. 목표선 0.8882는 baseline의 ranking_ceiling(0.7649)을 이미 넘는 값으로 파생되었고(`exceeds_ranking_ceiling: true`, `ks_shortfall` 0.2467), 이 실행기의 레버 목록 안에서 관측된 어떤 효과 크기(family swap 0.0022~0.0077 roc_auc, 튜닝 0.0032 roc_auc)로도 메울 수 없는 규모였습니다.

**5. 부수적 관찰**: iteration 4의 예측 확률은 랭킹은 좋으나 심하게 과신되어 있습니다(`brier` 0.1428, `calibration_error` 0.1807 — iteration 1의 0.0592, iteration 5의 0.0693과 대비). 이는 `class_weight='balanced'`가 확률을 밀어낸 결과로, 목표 지표에는 영향이 없지만 이 모델의 확률값을 그대로 의사결정에 쓰기는 어렵다는 뜻입니다. 이 실행기에는 재캘리브레이션 레버가 없습니다.

## 다음 단계 제안

(카드의 `Data caveats`는 "없음"이므로 특정 컬럼·분할을 배제하는 제약은 없습니다. 대신 실행기의 기록된 한계 — 단일 20% 검증, 파생 피처 불가, `missing_count`의 미해결 효과 — 를 준수해 제안합니다.)

1. **목표선 0.8882의 타당성을 먼저 재검토할 것.** 이 목표는 baseline + margin 0.65로 파생되었고, 자체 메타데이터가 `exceeds_ranking_ceiling: true` / `required_ks` 0.7764 / `ks_shortfall` 0.2467을 기록하고 있습니다. 실측 상한은 5개 시도 전부에서 best_cut 0.7711~0.7856입니다. 현재 피처 집합과 실행기 제약이 유지된다면 이 목표선은 튜닝으로 도달 가능한 값이 아니므로, (a) 목표를 관측된 랭킹 상한 근처(예: best_cut 0.79 부근) 기준으로 재설정하거나 (b) 아래 2번처럼 피처 파이프라인 자체를 확장해야 합니다. 이 판단이 없으면 다음 예산도 같은 0.78 벽에 소모됩니다.

2. **랭킹을 올릴 유일한 실질 경로는 실행기 외부의 피처 확장 — 카드 단계에서 수행할 것.** 실행기는 컬럼을 만들지 않으므로, 파생 피처는 데이터 카드에 들어가야 합니다. 근거가 있는 후보: (i) 드롭된 `field`를 빈도/타깃 인코딩한 저차원 컬럼으로 다시 넣기(현재 `max_cardinality` 50 규칙에 걸려 정보 자체가 소실), (ii) 상호 평가 쌍 컬럼의 차이·합(`attractive_o` vs `attractive_partner`, `like` vs `guess_prob_liked`, `pref_o_*` vs `*_important`) — speeddating의 매칭은 본질적으로 양방향 일치 문제이고 현재 모델은 두 방향을 각각의 컬럼으로만 볼 수 있으며 실행기는 그 조합을 만들 수 없습니다. (iii) `d_*` 이산화 중복 컬럼(58개 one-hot → 180 레벨)이 원본 수치와 같은 정보를 반복해 트리 분할을 희석하는지 점검 — 컬럼 삭제도 카드 단계에서만 가능합니다. 기대 효과의 크기는 이 실행에서 측정된 바 없으므로 수치로 약속하지 않습니다.

3. **검증 해상도를 먼저 키울 것.** 이 실행의 결정적 약점은 폭 0.05 (iteration 4: 0.7518~0.8052)짜리 단일 20% 슬라이스 위에서 0.004 차이로 최종 구성을 골랐다는 점입니다. iteration 2·4·5는 이 데이터로 구분 불가합니다. 반복 분할/CV 또는 더 넓은 검증 비율은 이 실행기가 지원하지 않는 런 설정이므로, 다음 예산의 첫 항목은 "cross-validation 또는 반복 시드 평가를 실행기에 추가"여야 합니다. 이것이 없으면 family·컬럼 실험은 계속 미해결로 끝납니다.

4. **같은 실행기 안에서 남은 예산을 쓴다면, 유일하게 논리적으로 남은 지점은 iteration 5의 `xgboost`를 정규화하는 것.** 근거: iteration 5는 best_cut 0.7856(전 구간 최고)을 가지면서 `train_val_gap` 0.1928(잔존 과적합)과 `cut_headroom` 0.0305(미소진 operating point)를 동시에 갖는 유일한 행입니다. 즉 `max_depth`를 3~4로 낮추고 `min_child_weight`·`reg_lambda`를 올려 gap을 iteration 2 수준(0.06)까지 줄이면서 `scale_pos_weight`를 5.07 주변에서 미세 조정하면, 실측 `balanced_accuracy`를 best_cut(0.7856) 근처까지 끌어올릴 가능성이 있습니다. 다만 이는 최대 약 +0.03 규모이며 목표선 0.8882와는 무관합니다 — 목표 달성이 아니라 "이 피처 집합의 상한을 확정"하는 용도로만 제안합니다. 이때도 `missing_indicator`는 `impute: none`과 함께라면 증명된 중복이므로 켜지 말 것이고, `missing_count`는 iteration 4에서 이미 미해결(+0.0040, CI가 0을 포함)로 측정되었으므로 단독 검증에 반복을 더 쓰지 말 것을 권합니다.