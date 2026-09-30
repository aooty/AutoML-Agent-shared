## 요약

**목표는 달성되지 않았습니다.** 목표 기준선은 `balanced_accuracy` 0.8403(maximize)이었으나, 5회 예산 중 4회를 사용한 시점에서 최고 검증 점수는 iteration 2의 `xgboost`가 기록한 **0.7752**(95% CI 0.7507~0.8020)로, 기준선에 0.0650 부족한 상태에서 연속 미개선(정체)으로 조기 종료되었습니다. 한 번도 사용되지 않은 최종 테스트 20%에서 같은 모델은 **0.8013**(95% CI 0.7762~0.8288)을 기록했고, 이 역시 기준선 0.8403에는 도달하지 못했습니다(CI 상한 0.8288). Critic 진단은 3회 나왔으며, 남은 격차의 대부분이 threshold(운영점)가 아니라 **ranking 축**에 있다는 점이 반복적으로 확인되었습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `class_weight='balanced'`, `early_stopping=False` / `impute='none'` | `balanced_accuracy` 0.7248 (CI 0.6969~0.7527), `roc_auc` 0.8598, `pr_auc` 0.6049, `balanced_accuracy_at_best_cut` 0.7922, `cut_headroom` 0.0674, `train_val_gap` 0.2752 | `overfitting` — train 1.0 대비 val 0.7248, gap 0.2752. 용량 축소를 지시(`max_leaf_nodes=8`, `min_samples_leaf=40`, `l2_regularization=10.0`, `max_features=0.6`, family 유지) |
| 2 | `xgboost` | `n_estimators=600`, `learning_rate=0.03`, `max_depth=3`, `reg_lambda=20.0`, `subsample=0.8`, `colsample_bytree=0.6`, `min_child_weight=20`, `scale_pos_weight=5.07`, `early_stopping_rounds=40` / `impute='none'` | **최고** `balanced_accuracy` 0.7753 (CI 0.7507~0.8020), `roc_auc` 0.8573, `pr_auc` 0.5844, `balanced_accuracy_at_best_cut` 0.7864, `cut_headroom` 0.0111, `train_val_gap` 0.0890 | `wrong_model_family` — overfitting은 해소(gap 0.2752→0.0890)됐고 paired Δ +0.0504(CI +0.0257~+0.0749)는 실재하지만 전부 운영점 이득이며 ranking은 하락. 다음엔 `xgboost` 고정 + `impute='median'` + `missing_count=True`만 바꾸라고 지시 |
| 3 | `mlp` | `hidden_layer_sizes=[128,64]`, `alpha=0.0003`, `learning_rate_init=0.0008`, `batch_size=128`, `max_iter=300`, `early_stopping=True` / `impute='median'`, `scale=True`, `missing_count=True` | `balanced_accuracy` 0.6121 (CI 0.5882~0.6355), `roc_auc` 0.8133, `pr_auc` 0.5007, `balanced_accuracy_at_best_cut` 0.7400, `train_val_gap` 0.0277, `stopped_at_iter=12/300` | `wrong_model_family` — paired Δ −0.1631(CI −0.1924~−0.1339)로 해소된 회귀. family와 pipeline을 동시에 바꿔 원인이 둘. 다시 iteration 2 구성 + `impute='median'` + `missing_count=True` 단일 변경을 지시 |
| 4 | `random_forest` | `n_estimators=600`, `max_depth=14`, `min_samples_leaf=5`, `class_weight='balanced_subsample'` / `impute='median'`, `missing_count=True` | `balanced_accuracy` 0.7201 (CI 0.6923~0.7477), `roc_auc` 0.8544, `pr_auc` 0.5731, `balanced_accuracy_at_best_cut` 0.7818, `cut_headroom` 0.0617, `train_val_gap` 0.2502 | (없음 — 평가 직후 루프 종료) |

## 최고 성능 구성

**iteration 2 / `xgboost`** — 아래 `hyperparams`/`preprocessing`은 해당 attempt에 기록된 **실제 적용값**입니다.

```
model: xgboost
hyperparams:
  n_estimators: 600
  learning_rate: 0.03
  max_depth: 3
  reg_lambda: 20.0
  subsample: 0.8
  colsample_bytree: 0.6
  min_child_weight: 20
  scale_pos_weight: 5.07
  early_stopping_rounds: 40
preprocessing:
  impute: none          # NaN 분기를 모델이 직접 처리
  scale: false
  missing_indicator: false
  missing_count: false
internal_validation: fit_rows 4523 / held_out_rows 503 (validation_fraction 0.1),
                     stopped_at_iter 599 / max_iter 600
dropped_hyperparams: [] (거부된 키 없음)
train_time_sec: 5.794
```

검증 지표(validation 20%): `balanced_accuracy` **0.7753** (95% CI 0.7507~0.8020), `recall` 0.7355, `specificity` 0.8150, `roc_auc` 0.8573, `pr_auc`/`average_precision` 0.5844, `f1` 0.5501, `precision` 0.4394, `accuracy` 0.8019, `brier` 0.1319, `calibration_error` 0.1439, `balanced_accuracy_at_best_cut` 0.7864, `cut_headroom` 0.0111, `train_val_gap` 0.0890.

최종 held-back 측정(첫 fit 이전에 분리되어 어떤 결정에도 쓰이지 않은 test 20%): `balanced_accuracy` **0.8013** (95% CI 0.7762~0.8288). 검증 0.7753과의 차이 **0.0261**이 이 실행의 선택 편향(selection effect) 크기입니다 — 루프는 검증 숫자만 보고 최고 attempt를 골랐고, 테스트 숫자는 그 선택에 관여하지 않았습니다. 이번 실행에서는 그 차이가 검증에 불리한 방향(테스트가 더 높음)으로 나왔으므로 검증 점수가 낙관적으로 부풀려졌다는 증거는 없습니다. 다만 **테스트 점수와 그 CI 상한(0.8288) 모두 목표 0.8403 아래**이므로, 이 실행이 실제로 입증한 것은 "약 0.80 수준의 `balanced_accuracy`"이며 목표 달성이 아닙니다.

## 원인 분석

**1. 남은 격차는 threshold가 아니라 ranking 축에 있습니다.** Critic 3회 진단이 모두 같은 지점을 가리켰습니다. 어떤 결정 컷을 골라도 얻을 수 있는 최댓값인 `balanced_accuracy_at_best_cut`은 4개 family 전체에서 0.7400(mlp) ~ 0.7922(iteration 1 hist_gbdt) 범위였고, 최고치 0.7922조차 기준선 0.8403에 **0.0481 부족**합니다. 즉 이 실행에서 만들어진 어떤 ranking도, 최적 컷을 준다 해도 목표에 닿지 않습니다. 최고 attempt(iteration 2)의 `cut_headroom`은 0.0111로 운영점 레버는 이미 거의 소진되었습니다(`recall` 0.7355 vs `specificity` 0.8150).

**2. iteration 1→2의 +0.0504는 실재하지만 축을 잘못 산 이득입니다.** paired Δ +0.0504(CI +0.0257~+0.0749)로 해소된 차이인데, 같은 구간에서 `roc_auc` 0.8598→0.8573, `pr_auc` 0.6049→0.5844, `balanced_accuracy_at_best_cut` 0.7922→0.7864로 ranking은 오히려 내려갔고 `cut_headroom`이 0.0674→0.0111로 붕괴했습니다. 목표 지표는 올랐지만 오른 축은 운영점이었습니다.

**3. 구간이 겹치는 움직임은 차이로 읽지 않습니다.** iteration 1(0.7248, CI 0.6969~0.7527)과 iteration 4(0.7201, CI 0.6923~0.7477)는 이 데이터로 구분되지 않습니다 — `random_forest`가 `hist_gbdt`보다 나쁘다/좋다고 말할 근거가 없습니다. 반대로 iteration 3의 0.6121(CI 0.5882~0.6355)은 다른 모든 attempt와 겹치지 않으며, `stopped_at_iter=12/300`과 `train_val_gap` 0.0277이 함께 있는 것으로 보아 mlp family 내부의 underfit입니다. 단, iteration 3은 family와 pipeline(`impute` none→median, `scale` false→true, `missing_count` false→true)을 동시에 바꿨으므로 −0.1631의 책임 주체가 둘이고, family 단독 효과로 귀속시킬 수 없습니다.

**4. Critic이 두 번 지정한 단일 레버 실험이 한 번도 실행되지 않았습니다.** iteration 1의 진단은 "family를 유지한 저용량 `hist_gbdt`"를 지시했으나 iteration 2는 `xgboost`로 family를 바꿨고, iteration 2·3의 진단은 모두 "iteration 2 구성 고정 + `impute='median'` + `missing_count=True`만 변경"을 지시했으나 iteration 3은 `mlp`, iteration 4는 `random_forest`로 갔습니다. 결과적으로 예산 4회 중 3회가 family 교체에 쓰였고, family 교체는 ranking 축에서 `balanced_accuracy_at_best_cut` 폭 0.0522를 만든 반면 필요한 양은 0.0481입니다 — 즉 "다음 family가 역대 최고여야만" 겨우 닿는 크기의 레버에 예산이 집중되었습니다. 정체로 조기 종료된 직접적 이유가 이것입니다.

**5. 목표 기준선 자체가 이 파이프라인의 도달 범위 밖에 설정되었을 가능성이 큽니다.** 기준선은 derived로, baseline `logreg` 0.6807(CI 0.6540~0.7099)에 margin 0.5를 적용해 0.8403이 되었고, 메타데이터는 `exceeds_ranking_ceiling: true`, `ranking_ceiling: 0.7649`, `required_ks: 0.6806` vs baseline `ks: 0.5297`(`ks_shortfall: 0.1509`)를 명시합니다. 즉 목표는 baseline ranking을 컷으로 최적화해도 넘을 수 없는 값이고, KS를 0.15 끌어올릴 만한 새로운 정보/표현을 요구합니다. 그런데 executor는 열 파생·상호작용·재인코딩·앙상블·threshold 탐색·calibration을 모두 할 수 없으며, 추가 가능한 열은 `missing_indicator`/`missing_count` 둘뿐입니다.

**6. `unsupported_claims`에 대한 주의.** iteration 1과 3의 plan에 `feature_engineering` 플래그가 붙었습니다. 이는 plan 산문에 대한 substring 검사이고, 이 기록에는 두 iteration의 plan 원문이 남아 있지 않아 실제로 그 기능에 **의존**했는지 단순 언급했는지 확인할 수 없습니다. 따라서 두 attempt의 점수 손실을 이 플래그에 귀속시키지 않습니다 — 다만 executor가 파생 피처를 만들지 못한다는 사실 자체는 아래 제안에 반영합니다.

**7. 부수적으로 확인된 것:** iteration 4는 `max_depth=14`, `min_samples_leaf=5` 조합으로 `train_val_gap` 0.2502를 다시 만들어 iteration 1의 overfitting 실패 모드를 재현했고, iteration 2의 `calibration_error` 0.1439 / `brier` 0.1319는 이 확률값을 확률로 읽기 어렵다는 뜻입니다(`scale_pos_weight=5.07`이 확률을 위로 밀어냄). 이 실행 환경에는 recalibration 레버가 없으므로 이는 진단 정보이며 수정 대상이 아닙니다.

## 다음 단계 제안

`Data caveats`는 "없음"이므로 어떤 열·값·split을 신뢰할 수 없다고 명시된 제약은 없습니다. 아래 제안은 executor의 CAN/CANNOT 목록과 위 수치에만 근거합니다.

1. **한 번도 실행되지 않은 두 개의 단일 레버 실험을 먼저 소진할 것 (예산 2회).**
   - (a) Critic이 iteration 1에서 지시한 저용량 `hist_gbdt`: `max_iter=250`, `learning_rate=0.04`, `max_leaf_nodes=8`, `min_samples_leaf=40`, `l2_regularization=10.0`, `max_features=0.6`, `class_weight='balanced'`, `impute='none'`. 근거: **ranking 축 최고치는 여전히 iteration 1의 `hist_gbdt`**(`roc_auc` 0.8598, `pr_auc` 0.6049, `best_cut` 0.7922)이고, 그 구성의 유일한 확정된 결함은 `train_val_gap` 0.2752입니다. 최고 ranking family를 유지한 채 분산만 줄이는 조합은 아직 측정되지 않았습니다. 읽을 값은 `balanced_accuracy` 대신 `roc_auc`/`pr_auc`/`balanced_accuracy_at_best_cut`(vs 0.8598 / 0.6049 / 0.7922)과 `train_val_gap`입니다.
   - (b) iteration 2 구성을 하이퍼파라미터까지 그대로 두고 `impute='median'`, `missing_count=True`만 켠 행. 다만 기대값은 낮게 잡아야 합니다 — executor 문서상 `missing_count`는 다른 표본에서 단독 0.0003, indicator 위 0.0000이었고, 이 데이터는 `wave`(recording regime)와 `expected_num_interested_in_me`(missing_rate 0.7852)를 함께 갖고 있어 stratified random split에서는 결측 열이 "언제 기록되었는지"를 학습하고 그것이 이득으로 채점될 위험이 있습니다. 이 행은 "가능성 있는 개선"이 아니라 **Critic이 두 번 지정한 미해결 귀속 실험을 닫기 위한** 것으로 취급하십시오. (`impute='none'`과 함께 쓰는 `missing_indicator`는 문서상 비트 단위로 동일한 예측을 낸다고 확정되어 있으므로 예산을 쓰지 마십시오.)
2. **`mlp`와 `random_forest` 라인은 종료할 것.** `mlp`는 4개 family 중 ranking 최하위(`roc_auc` 0.8133, `best_cut` 0.7400)이고 12 iteration에서 멈췄으며, `random_forest`는 `balanced_accuracy` 0.7201로 iteration 1과 CI가 겹쳐 구분되지 않으면서 `train_val_gap` 0.2502를 재도입했습니다. 남은 예산을 family 순회에 쓰는 것은 필요량 0.0481 대비 family 폭 0.0522라는 산술상 승산이 낮습니다. 또한 iteration 3처럼 family와 pipeline을 동시에 바꾸는 계획은 금지하고, **attempt당 레버 1개** 규칙을 강제하십시오(그렇지 않으면 −0.1631처럼 원인 귀속이 불가능해집니다).
3. **executor 역량을 늘리거나 목표를 재도출할 것 — 이것이 사실상 유일하게 0.0481을 메울 수 있는 경로입니다.** 필요량은 `required_ks` 0.6806 vs 현재 최고 KS 근처 0.5844로, "정보량"이 부족한 상황인데 현재 executor는 열 파생을 전혀 하지 못합니다. 이 데이터는 자기평가/상대평가 쌍(`attractive`↔`attractive_partner`↔`attractive_o`, `like`, `guess_prob_liked`, `pref_o_*`↔`*_important`)이 구조적으로 존재하므로 차이·비율·상호작용 열이 ranking을 올릴 후보이지만, 그것은 **plan이 아니라 데이터 카드/전처리 단계에서 만들어져 들어와야** 합니다. 병행해서, 목표 0.8403은 derived이며 스스로 `exceeds_ranking_ceiling: true`, `ks_shortfall: 0.1509`를 기록하고 있으므로, 관측된 ranking 상한(모든 attempt의 `best_cut` 최댓값 0.7922, held-back 실측 0.8013)을 근거로 기준선을 재협상하는 것도 정당한 조치입니다.
4. **split 유효성을 한 번 점검할 것(모델 개선이 아니라 숫자의 의미 확인).** 프로토콜은 `stratified: true`, `grouped_by: null`이고 각 행은 speed-dating 쌍이며 `wave` 열이 존재합니다. 동일 참가자/동일 wave의 행이 train과 validation/test에 함께 들어갈 수 있는 구조인지 확인하고, 그렇다면 wave 기준 group-aware split에서 같은 iteration 2 구성을 재측정하십시오. 이는 점수를 올리는 작업이 아니라, 0.8013이라는 held-back 수치를 배포 상황의 추정치로 인용해도 되는지를 결정하는 작업이며, 결과에 따라 위 1(b)의 `missing_count` 해석(진짜 정보인가, recording regime인가)도 함께 정리됩니다. 현재 executor는 split·seed를 바꿀 수 없으므로 이 점검은 실행 설정 단계에서 수행되어야 합니다.