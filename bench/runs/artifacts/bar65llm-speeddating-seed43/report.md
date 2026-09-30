# AutoML 실행 최종 보고서 — `speeddating` / `balanced_accuracy`

## 요약

**목표는 달성하지 못했습니다.** 목표 기준선은 `balanced_accuracy` 0.8879였으나, 5회 예산 중 4회를 사용해 얻은 최고 검증 점수는 iteration 2의 `hist_gbdt` 0.7843 (95% CI 0.7584~0.8131)로 **0.1036 부족**했습니다. 같은 모델을 한 번도 사용되지 않은 테스트 20%에서 채점한 결과는 0.7940 (95% CI 0.7658~0.8205)입니다. 연속 미개선(정체)으로 iteration 4 이후 조기 종료되었고, critic은 3회 진단·재계획했습니다. 다만 baseline logreg 0.6798 (CI 0.6551~0.7086)은 명확히 넘어섰습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `class_weight='balanced'`, `early_stopping=False`; `impute:none`, `missing_count=True` | `balanced_accuracy` 0.7473 (CI 0.7196~0.7803), `roc_auc` 0.8601, `balanced_accuracy_at_best_cut` 0.7849, `cut_headroom` 0.0376, `train_val_gap` 0.2524 | `overfitting` — train 0.9998 vs val 0.7473. 용량 규제 처방 |
| 2 | `hist_gbdt` | `max_iter=200`, `learning_rate=0.05`, `max_depth=4`, `max_leaf_nodes=8`, `min_samples_leaf=40`, `l2_regularization=10.0`, `max_features=0.6`, `class_weight={0:1,1:4}`; `impute:none`, `missing_count=True` | **최고 검증** 0.7843 (CI 0.7584~0.8131), `roc_auc` 0.8552, `best_cut` 0.7847, `cut_headroom` 0.000388, `train_val_gap` 0.0706 | `wrong_model_family` — gap은 닫혔으나 ranking(roc_auc, best_cut)은 정체. bagging 계열 전환 처방 |
| 3 | `extra_trees` | `n_estimators=600`, `max_features=0.4`, `min_samples_leaf=3`, `class_weight='balanced'`, `bootstrap=False`; `impute:median`(강제), `missing_count=True` | 0.6765 (CI 0.6502~0.7062), `roc_auc` 0.8355, `best_cut` 0.7649, `cut_headroom` 0.0884, `train_val_gap` 0.3232 — 실행 중 최저 | `hyperparam` — 무제한 깊이로 재차 암기. 계열 재전환 대신 iteration 2 구성 복귀 처방 |
| 4 | `hist_gbdt` | `max_iter=500`, `learning_rate=0.03`, `max_depth=6`, `max_leaf_nodes=16`, `min_samples_leaf=30`, `l2_regularization=5.0`, `max_features=0.6`, `class_weight={0:1,1:4}`; `impute:none`, `missing_count=True` | 0.7713 (CI 0.7432~0.8013), `roc_auc` 0.8585, `best_cut` 0.7834, `cut_headroom` 0.0121, `train_val_gap` 0.1871 | (없음 — 평가 직후 루프 종료) |

`dropped_hyperparams`는 4회 모두 비어 있고 오류도 없습니다. 학습 시간은 최대 10.4초로 예산은 제약이 아니었습니다.

## 최고 성능 구성

iteration 2에서 **실제로 빌드된** 구성입니다 (아래 값은 히스토리의 `hyperparams` / `preprocessing`, 즉 적용된 값입니다).

- model: `hist_gbdt`
- hyperparams:
  ```json
  {
    "max_iter": 200,
    "learning_rate": 0.05,
    "max_depth": 4,
    "max_leaf_nodes": 8,
    "min_samples_leaf": 40,
    "l2_regularization": 10.0,
    "max_features": 0.6,
    "class_weight": {"0": 1.0, "1": 4.0},
    "early_stopping": false
  }
  ```
- preprocessing: `{"impute": "none", "scale": false, "missing_indicator": false, "missing_count": true}`
  (즉 결측은 대치하지 않고 트리가 NaN을 직접 분기, 행별 결측 개수 컬럼 1개만 추가)
- 프로토콜: stratified 60/20/20, seed 43 (카드의 baseline과 동일 분할)

검증 지표: `balanced_accuracy` **0.7843** (95% CI 0.7584~0.8131), `recall` 0.7536, `specificity` 0.8150, `roc_auc` 0.8552, `pr_auc` / `average_precision` 0.5895, `f1` 0.5599, `precision` 0.4454, `accuracy` 0.8049, `balanced_accuracy_at_best_cut` 0.7847, `cut_headroom` 0.000388, `brier` 0.1324, `calibration_error` 0.1477, `train_balanced_accuracy` 0.8549, `train_val_gap` 0.0706, 학습 4.929초.

**보류된 테스트 20%에서의 측정: `balanced_accuracy` 0.7940 (95% CI 0.7658~0.8205).** 검증 점수 0.7843과의 차이 약 0.0097이 이 실행의 선택 편향(selection effect) 크기입니다 — 루프는 검증 숫자만 보고 최고 시도를 골랐기 때문에, 모델에 대해 이 실행이 실제로 입증한 값은 테스트 점수 쪽입니다. 다만 검증 점수가 테스트 CI 안에 들어 있으므로, 이 행 수로는 두 값의 차이를 0과 구분할 수 없습니다. 어느 숫자를 쓰더라도 목표 0.8879에는 미달입니다.

## 원인 분석

critic 진단은 3건 존재하며, 세 진단을 관통하는 사실은 하나입니다: **부족분이 operating point가 아니라 ranking(순위 매기기)에 있고, 이 실행이 쓸 수 있는 레버로는 ranking이 거의 움직이지 않았습니다.**

- **operating point 축은 iteration 2에서 완전히 소진되었습니다.** iteration 1의 `cut_headroom`은 0.0376이었고(recall 0.5761 vs specificity 0.9186), `class_weight`를 `{0:1,1:4}`로 옮긴 iteration 2에서 recall 0.7536 / specificity 0.8150으로 균형이 잡히며 `cut_headroom`이 0.000388까지 떨어졌습니다. 남은 거리 0.1036 대비 임계값 축이 살 수 있는 몫은 사실상 0입니다.
- **ranking 축은 4회 동안 거의 고정되었습니다.** `balanced_accuracy_at_best_cut`는 0.7849 → 0.7847 → 0.7649 → 0.7834로, 두 개의 모델 계열과 약 4배에 이르는 용량 범위를 지났음에도 전체 폭이 0.0200에 불과합니다. `roc_auc`도 0.8355~0.8601 범위입니다. 목표가 요구하는 KS는 0.7758인데 달성 KS는 약 0.57 수준이며, 목표 정의 자체가 `exceeds_ranking_ceiling: true`, `ks_shortfall: 0.2642`를 기록하고 있습니다. **즉 0.8879는 이 실행에서 관측된 어떤 ranking을 어떤 임계값으로 잘라도 도달할 수 없는 지점입니다** (최고 `best_cut` 0.7849).
- **구간이 겹치는 움직임은 이야기로 만들지 않습니다.** iteration 2(0.7584~0.8131)와 iteration 4(0.7432~0.8013)의 CI는 크게 겹치며, 두 시도는 이 데이터로 구분되지 않습니다 — 규제된 `hist_gbdt` 안에서의 추가 용량 조정은 성과가 있었다고 말할 수 없습니다. 반면 동일 행 기준 paired 비교가 남아 있는 두 건은 해상도를 넘습니다: iteration 1→2는 Δ +0.0370 (CI +0.0120~+0.0645, P(better) 0.998), iteration 2→3은 Δ −0.1078 (CI −0.1367~−0.0797)로 명확한 개선과 명확한 퇴행입니다. 그리고 +0.0370은 `roc_auc`가 0.8601→0.8552로 오히려 소폭 하락한 가운데 발생했으므로, 그 이득의 사실상 전부는 ranking이 아니라 cut 위치 이동이었습니다.
- **iteration 3은 두 레버가 동시에 움직인 시도입니다.** `extra_trees`는 NaN을 직접 받을 수 없어 `impute`가 `median`으로 강제 downgrade되었으므로, −0.1078의 소유자가 계열 전환과 대치 방식 두 개입니다. 그 안에서 귀속 가능한 결함은 `min_samples_leaf=3` + 무제한 깊이로 인한 `train_val_gap` 0.3232이며, 즉 **bagging 계열은 정상 용량에서 시험된 적이 없습니다.** 이는 이 실행의 미해결 항목으로 남았습니다.
- 부수적으로, 최고 구성의 확률값은 그대로 쓰기 어렵습니다: `brier` 0.1324, `calibration_error` 0.1477 (iteration 1은 0.1026 / 0.0603). 양의 클래스에 4배 가중을 준 결과이며, 이 실행에는 재보정 레버가 없습니다.
- `unsupported_claims`에 iteration 1과 3에서 `feature_engineering`이 기록되어 있습니다. 이는 계획 산문에 대한 부분 문자열 검사로 약한 증거이고, 두 계획의 전체 문면이 여기에 없어 계획이 실제로 파생 피처에 **의존**했는지는 확인할 수 없습니다. 따라서 그 항목 때문에 점수를 잃었다고 말하지 않습니다 — 확실한 것은 실행된 파이프라인에 파생 컬럼이 하나도 없었다는 사실이며, 이는 아래 제안으로 넘깁니다.

요약하면, 성능을 제한한 것은 과적합(iteration 1에서 진단·해소됨)도 임계값(iteration 2에서 소진됨)도 아니고, **주어진 컬럼을 그대로 넣었을 때 이 모델들이 만들어낼 수 있는 순위 품질의 상한**이었습니다.

## 다음 단계 제안

(이 데이터셋에는 별도로 기록된 caveat이 없으므로, 특정 컬럼·값·분할을 배제하는 제약은 적용되지 않습니다.)

1. **목표 수치를 먼저 재검토할 것.** 0.8879는 baseline 0.6798에 margin 0.65를 적용한 파생 임계값이고, 목표 블록 스스로 `exceeds_ranking_ceiling: true` / `required_ks: 0.7758` / `passable_margin: 0.237`을 기록합니다. 관측된 `balanced_accuracy_at_best_cut` 최고치가 0.7849인 상황에서, 남은 1회 예산이든 추가 예산이든 하이퍼파라미터로 이 차이를 메울 수 있다는 근거는 없습니다. 현실적 목표(예: `passable_margin` 기준선)로 재설정하거나, 아래 2번처럼 입력 신호를 늘리는 것이 전제 조건입니다.
2. **ranking을 올릴 유일한 유망 경로는 피처이며, 이는 executor 밖에서 해야 합니다.** 이 executor는 비율·차이·상호작용 등 어떤 파생 컬럼도 만들 수 없고(`missing_indicator`/`missing_count`만 예외), 실제로 4회 모두 원본 컬럼 그대로 학습되었습니다. `target_corr`가 strong/moderate로 표시된 `like`, `guess_prob_liked`, `attractive_partner`/`funny_partner`/`shared_interests_partner`와, 상대의 선호(`pref_o_*`)와 본인 평가 사이의 대응 구조는 쌍(pair) 데이터 특유의 상호작용 신호를 담고 있을 가능성이 높은데, 현재 모델은 이를 분기 조합으로만 근사합니다. **파생 컬럼을 CSV/카드 단계에서 추가해 새 카드로 다시 루프를 돌리는 것**이 `best_cut` 0.785 천장을 실제로 시험하는 유일한 방법입니다. 또한 고카디널리티로 드롭된 `field`(그룹화·빈도 인코딩 후 재투입)도 같은 경로에서 되살릴 수 있습니다.
3. **executor 내부에 남은 미시험 레버는 두 개뿐이고, 기대 효과는 작다는 전제로만 쓸 것.** (a) `xgboost` + `impute:none`은 NaN을 직접 분기하는 아직 시도되지 않은 계열이며, iteration 2 수준의 규제(얕은 깊이, 강한 `lambda`, `max_features`/`colsample` 축소)와 `class_weight`/`scale_pos_weight`를 iteration 2와 동일하게 두어 계열만 단독으로 움직이면 귀속 가능한 1행이 됩니다. (b) `extra_trees`/`random_forest`를 **정상 용량**(`min_samples_leaf` 20~40, `max_depth` 제한)으로 재시험 — iteration 3은 무제한 깊이 때문에 계열이 아니라 용량을 측정한 셈이므로 미해결 항목입니다. 다만 참고 자료상 계열 전환의 `roc_auc` 이동폭은 0.0022~0.0077이었고 부족분은 0.1036이므로, **이 두 행이 목표를 달성할 것으로 기대해서는 안 됩니다.** `missing_indicator`는 `impute:none`과 함께 쓰면 예측이 비트 단위로 동일하다는 것이 확인된 사항이므로 예산을 쓰지 마십시오.
4. **분할 구조를 별도로 검증할 것(미측정 가설).** 이 데이터는 speed dating 이벤트 행이고 `wave` 컬럼이 존재하는 반면 프로토콜의 `grouped_by`는 `null`입니다. 동일 참가자의 여러 행이 train과 validation/test에 나뉘어 들어갈 수 있다면 0.784/0.794는 낙관적으로 읽힐 수 있습니다. 이 executor는 분할·seed를 바꿀 수 없으므로, 원본 파일에서 참가자/wave 단위의 그룹 분할로 동일 iteration 2 구성을 한 번 재적합해 보는 외부 확인이 필요합니다. 이는 **측정되지 않은 가설**이며, 확인되면 위 1~3의 기준선 자체를 다시 잡아야 하고, 기각되면 현재 숫자를 그대로 신뢰할 수 있게 됩니다.

추가로, 확률값을 의사결정에 쓸 계획이라면 최고 구성의 `calibration_error` 0.1477을 기억하십시오. 이 실행에는 재보정 레버가 없으므로, 확률 품질이 필요하면 `class_weight` 가중을 낮춘 구성(iteration 1: `brier` 0.1026, `calibration_error` 0.0603)과 `balanced_accuracy` 사이의 트레이드오프를 명시적으로 선택해야 합니다.