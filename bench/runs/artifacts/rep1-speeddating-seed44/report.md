# AutoML 실행 최종 리포트 — `speeddating` / `balanced_accuracy`

## 요약

목표는 달성하지 못했습니다. 목표 기준선은 `balanced_accuracy` 0.7605였고, 루프가 선택한 최고 검증 점수는 iteration 2의 `hist_gbdt`가 기록한 **0.7528 (95% CI 0.7268~0.7792)** 로 기준에 0.0077 미달입니다. 총 5회 예산 중 4회를 사용하고 연속 미개선(정체)으로 조기 종료되었으며, critic 진단은 3회 수행되었습니다. 참고로 한 번도 사용되지 않은 최종 테스트 20%에서 같은 모델은 0.7649 (95% CI 0.7358~0.7929)를 기록했지만, 루프의 판정 기준은 검증 점수였고 테스트 CI가 기준선을 포함하므로 "기준 통과"라고 주장할 수는 없습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=false`, `class_weight={"0":1,"1":5}`, `impute: none` | `balanced_accuracy` 0.7310 (CI 0.7026~0.7617), `recall` 0.5399 / `specificity` 0.9221, `roc_auc` 0.8563, `cut_headroom` 0.0492 | `data_issue` — 격차는 랭킹이 아니라 operating point 축에 있음 → 양성 가중치를 5→12로 올릴 것 |
| 2 | `hist_gbdt` | 위와 동일, `class_weight={"0":1,"1":12}` | **`balanced_accuracy` 0.7528 (CI 0.7268~0.7792)**, `recall` 0.5978 / `specificity` 0.9079, `roc_auc` 0.8585, `cut_headroom` 0.0297 | `data_issue` — 여전히 specificity 쪽에 치우침, `cut_headroom`이 남은 거리의 3.9배 → 가중치를 12→22로 |
| 3 | `hist_gbdt` | 위와 동일, `class_weight={"0":1,"1":22}` | `balanced_accuracy` 0.7461 (CI 0.7179~0.7729), `recall` 0.6051 / `specificity` 0.8871, `roc_auc` 0.8574, `cut_headroom` 0.0351 | `wrong_model_family` — 가중치 3개 설정(5/12/22)이 0.7461~0.7528에서 정체, 랭킹은 평탄 → 동일 파이프라인에서 `xgboost`(`scale_pos_weight=12`)로 계열 교체 |
| 4 | `extra_trees` | `n_estimators=800`, `min_samples_leaf=2`, `bootstrap=false`, `class_weight={"0":1,"1":6}`, `impute: median` | `balanced_accuracy` 0.6553 (CI 0.6289~0.6825), `recall` 0.3478 / `specificity` 0.9629, `roc_auc` 0.8483, `cut_headroom` 0.1175 | (없음 — 마지막 시도, 평가 직후 루프 종료) |

> iteration 3의 critic이 지시한 계열 교체는 `xgboost` + `impute: none`이었지만, 실제로 실행된 iteration 4는 `extra_trees` + `impute: median`이었습니다. 즉 **처방된 `xgboost` 전환은 이번 실행에서 한 번도 검증되지 않았습니다.**

## 최고 성능 구성

iteration 2에서 실제로 빌드된 구성(히스토리의 적용값 그대로):

- **model**: `hist_gbdt`
- **hyperparams**: `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=false`, `class_weight={"0": 1.0, "1": 12.0}`
- **preprocessing (applied)**: `impute: none` (모델이 NaN을 직접 분기), `scale: false`, `missing_indicator: false`, `missing_count: false`
- **dropped_hyperparams**: 없음 / **train_time_sec**: 10.735
- 프로토콜: stratified 60/20/20, seed 44 (카드의 baseline과 동일 분할)

검증 지표: `balanced_accuracy` **0.7528 (CI 0.7268~0.7792)**, `recall` 0.5978, `specificity` 0.907857, `precision` 0.5612, `f1` 0.5789, `accuracy` 0.8568, `roc_auc` 0.8585, `pr_auc` / `average_precision` 0.6070, `balanced_accuracy_at_best_cut` 0.7825, `cut_headroom` 0.029658, `brier` 0.106086, `calibration_error` 0.073463, `train_balanced_accuracy` 0.9990, `train_val_gap` 0.246206.

**보류 테스트(최종 1회 측정)**: `balanced_accuracy` **0.7649 (95% CI 0.7358~0.7929)**. 검증 0.7528 대비 -0.0120이며, 이 차이가 선택 편향의 크기입니다 — 루프의 모든 선택은 검증 숫자만 보고 이루어졌습니다. 다만 검증 점수가 테스트 CI 안에 들어가므로 이 행들만으로는 그 차이를 0과 구분할 수 없습니다. 이 모델에 대해 실제로 입증된 값은 검증 0.7528이 아니라 테스트 0.7649이지만, 테스트 CI가 기준선 0.7605를 포함하므로 기준 통과로 보고할 수는 없습니다.

## 원인 분석

**1) 세 번의 `hist_gbdt` 시도는 이 데이터에서 서로 구분되지 않습니다.** iteration 1/2/3의 CI는 0.7026~0.7617, 0.7268~0.7792, 0.7179~0.7729로 폭이 약 0.055이고 전부 겹칩니다. 즉 가중치 5 → 12 → 22가 만든 0.7310 / 0.7528 / 0.7461의 순서는 이 검증 슬라이스(20%)의 해상도 아래에 있는 움직임이며, 남은 거리 0.0077은 CI 폭의 1/7에 불과합니다. critic 자신도 iteration 3 진단에서 "세 값과 기준선 모두 iteration 3의 CI 안에 있다"고 기록했습니다. 따라서 "가중치 12가 최적이었다"는 식의 서술은 리샘플 노이즈를 설명하는 것에 가깝고, 실제로 이번 실행이 확실히 보여준 것은 (a) `hist_gbdt` 계열이 baseline `logreg` 0.6807 (CI 0.654~0.7099)보다 확실히 낫다는 점, (b) iteration 4의 `extra_trees` 0.6553 (CI 0.6289~0.6825)이 iteration 2와 CI가 겹치지 않을 만큼 확실히 나쁘다는 점, 두 가지입니다.

**2) critic 3회 진단의 패턴: 같은 축을 세 번 밀었고, 그 축은 이미 거의 소진되어 있었습니다.** 진단 유형은 `data_issue` → `data_issue` → `wrong_model_family`였고, 앞의 두 번은 모두 operating point(양성 가중치) 단일 레버를 처방했습니다. 근거는 일관되게 `recall` ≪ `specificity`와 큰 `cut_headroom`이었는데, 실제 반응은 급격히 둔화되었습니다: 가중치를 12→22로 거의 두 배 올려도 `recall`은 0.5978→0.6051, `specificity`는 0.9079→0.8871로만 움직였고 `balanced_accuracy`는 오히려 0.7528→0.7461로 내려갔습니다(단, 위 1)에서 말한 대로 이 하락 자체도 유의하지 않습니다). 즉 세 번의 시도 예산 중 세 번이 해상도 이하의 한 축에 쓰였습니다.

**3) 랭킹 축은 이번 실행에서 사실상 전혀 움직이지 않았습니다.** `roc_auc`는 0.8563 / 0.8585 / 0.8574 (핵심 문서가 명시한 페어드 해상도 0.003~0.006 이내), `balanced_accuracy_at_best_cut`은 0.7802 / 0.7825 / 0.7812로 평탄합니다. 그리고 이것이 이번 실행의 실질적 상한을 규정합니다 — 어떤 컷을 골라도 이 랭킹으로 얻을 수 있는 최대치가 0.78 근처이므로, 기준선 0.7605까지의 여유는 원래부터 매우 얇았습니다. 랭킹을 바꿀 수 있는 두 가지 시도(계열 교체, 용량/정규화 변경)는 각각 이렇게 소비되었습니다: 계열 교체는 처방된 `xgboost` 대신 `extra_trees` + `impute: median`으로 실행되어 **모델 계열과 전처리 두 개가 동시에 변경**되었고(0.0975의 하락에 소유자가 둘), 처방된 `xgboost`는 검증되지 않았습니다. 용량 쪽은 세 번의 `hist_gbdt` 시도가 `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`을 한 번도 바꾸지 않았습니다 — `train_balanced_accuracy` 0.9990, `train_val_gap` 0.2462(학습 데이터를 사실상 암기)라는 신호가 세 번 반복해서 기록되었는데도 그 레버는 시도되지 않았습니다.

**4) 부가 관찰.** iteration 1의 plan에는 `unsupported_claims: ["feature_engineering"]`이 기록되어 있습니다. 이는 계획 산문에 대한 문자열 검사이고, 해당 iteration의 plan 원문은 여기에 남아 있지 않아 그 계획이 파생 피처에 *의존*했는지 단순히 한계를 언급했는지 판별할 수 없습니다. 따라서 이 항목을 성능 손실의 원인으로 귀속하지 않습니다 — 실제로 실행된 것은 카드의 원본 수치 열 그대로이며, 피처 생성 부재는 아래 "다음 단계"에서 만들어야 할 역량으로 다룹니다. 또한 최고 구성의 `calibration_error` 0.0735 / `brier` 0.1061은 확률이 평균 7%p 어긋난다는 뜻이지만, 이 환경에는 재보정 레버가 없고 `balanced_accuracy`는 순위와 컷의 함수이므로 이번 미달의 직접 원인은 아닙니다.

## 다음 단계 제안

1. **처방되었지만 실행되지 않은 단일 레버 교체를 먼저 채운다: `xgboost` + `impute: none`.** iteration 3 critic의 구체 지시(`scale_pos_weight=12`, `n_estimators=600`, `learning_rate=0.06`, `max_depth=6`, `min_child_weight=5`, `subsample=0.9`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `early_stopping_rounds=null`, 전처리는 `impute: none`/`scale: false`/미싱 열 없음)를 그대로 실행하면, iteration 4처럼 계열과 전처리를 동시에 바꾸지 않고 랭킹 축의 이동분만 읽을 수 있습니다. 근거: `roc_auc`가 세 시도에서 평탄했고 `balanced_accuracy_at_best_cut`이 0.78 근처에 갇혀 있어, 남은 격차는 컷이 아니라 랭킹에 있습니다(문서상 계열 교체의 랭킹 이동 폭은 0.0022~0.0077 범위로 관측된 바 있음 — 크기를 보장하는 값이 아니라 기대 범위로만 사용).
2. **`hist_gbdt`의 용량/정규화를 처음으로 움직인다.** `train_balanced_accuracy` 0.9990, `train_val_gap` 0.2462가 세 번 반복 기록됐는데 `max_leaf_nodes`/`min_samples_leaf`/`l2_regularization`/`learning_rate`·`max_iter`는 한 번도 변경되지 않았습니다. 예: `max_leaf_nodes=15`, `min_samples_leaf=100`, `l2_regularization=10`, `max_iter=400`, `learning_rate=0.06`을 유지하고 `class_weight={"0":1,"1":12}` 고정(= 검증된 operating point는 고정, 용량만 단일 변경). 읽을 지표는 `balanced_accuracy_at_best_cut`과 `roc_auc`이며, 이 둘이 오르지 않으면 이 파이프라인의 랭킹 상한이 확인된 것으로 보고 다음 축으로 넘어갑니다.
3. **양성 가중치 미세조정에는 더 이상 예산을 쓰지 않는다.** 검증 슬라이스의 CI 폭이 약 0.055이고 남은 거리는 0.0077이므로, 5~22 사이의 어떤 값도 이 데이터로는 서로 구분되지 않습니다(iteration 1~3 CI 전부 중첩). 이 환경은 CV·OOF·분할 변경을 제공하지 않으므로, 해상도를 높이려면 루프 밖에서 더 큰 평가 데이터나 반복 분할을 확보하는 것이 선행 조건입니다. 그 전까지는 CI 폭보다 작은 개선을 목표로 하는 계획을 세우지 않는 것이 옳습니다.
4. **미싱 정보 열과 피처 생성은 "왜 결측인지"를 먼저 문서화한 뒤에만 손댄다.** 이 데이터에는 별도 caveat이 기록되어 있지 않고, 카드에는 `expected_num_interested_in_me`가 78.52% 결측, 결측 열이 60개, 그리고 이미 `has_null`과 `wave` 열이 원본 피처로 들어 있습니다. 최고 구성은 `impute: none`이므로 `missing_indicator`는 정의상 중복(동일 예측)이고, `missing_count`는 결측이 피험자 상태가 아니라 수집 회차(wave)를 지시할 경우 "기록 체제"를 학습하는 열이 됩니다 — 어느 쪽인지 판별할 근거가 이번 실행에 없으므로 지금 추가를 권하지 않습니다. 대신 결측 발생 원인(회차별 설문 문항 변경 여부 등)을 데이터 제공자 측에서 확정해 caveat으로 기록하는 것을 선행 작업으로 권합니다. 같은 이유로, 카드에서 제외된 `field`(high cardinality)나 파생 피처(예: 상호 평가 차이·비율)는 executor가 만들 수 없으므로, 필요하다면 루프 밖 전처리에서 열로 만들어 카드에 포함시켜야 하며 이는 랭킹 상한 0.78을 실제로 끌어올릴 수 있는 유일한 미검증 경로입니다.