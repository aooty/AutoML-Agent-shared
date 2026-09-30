# AutoML 최종 리포트 — bank-marketing (balanced_accuracy)

## 요약

목표는 달성하지 못했습니다. 목표 임계값은 `balanced_accuracy` 0.8811이었고, 5회 시도 중 최고 검증 점수는 iteration 4의 `hist_gbdt`가 기록한 **0.8652 (95% CI 0.8526~0.8758)** 로 0.0159 부족했습니다. 한 번도 사용되지 않은 홀드백 테스트 20%에서는 같은 모델이 **0.8646 (95% CI 0.8530~0.8743)** 을 기록해, 검증 점수와 실질적으로 동일했습니다(즉 선택 편향은 이 데이터에서 0과 구분되지 않습니다). 베이스라인 `logreg` 0.6603 대비로는 +0.205의 큰 개선이지만, 이 파일·이 feature set에서 4번의 critic 진단이 일관되게 지적한 대로 남은 격차는 operating point가 아니라 **ranking(순위 매기기) 능력**에 있었고 5회 예산 안에서는 그 축을 움직일 레버가 없었습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `learning_rate` 0.06, `max_iter` 500, `max_leaf_nodes` 63, `l2_regularization` 1.0, `class_weight` {0:1, 1:6}, `early_stopping` false | balanced_accuracy **0.8266** (CI 0.8135~0.8391), recall 0.7382 / specificity 0.9150, `cut_headroom` 0.0446, `train_val_gap` 0.1509 | `hyperparam` — 컷이 crossing에서 벗어나 있고 용량이 과다. 양성 가중치 상향 + 용량 축소 처방 |
| 2 | `hist_gbdt` | `learning_rate` 0.06, `max_iter` 300, `max_leaf_nodes` 31, `l2_regularization` 1.0, `class_weight` {0:1, 1:9} | balanced_accuracy **0.8652** (CI 0.8532~0.8756), recall 0.8648 / specificity 0.8655, `cut_headroom` 0.0063, `balanced_accuracy_at_best_cut` 0.8715 | `wrong_model_family` — 처방대로 +0.0386(paired CI +0.0281~+0.0481) 얻었고 operating point는 소진. 남은 격차는 ranking에 있다며 family swap 처방 |
| 3 | `xgboost` | `n_estimators` 700, `learning_rate` 0.05, `max_depth` 6, `subsample` 0.8, `reg_lambda` 2.0, `scale_pos_weight` 9.0 | balanced_accuracy **0.8491** (CI 0.8358~0.8600), `balanced_accuracy_at_best_cut` 0.8701, `train_val_gap` 0.1132 | `wrong_model_family` — swap이 ranking을 전혀 못 올림(best_cut 0.8715→0.8701). hist_gbdt 복귀 + `missing_indicator`/`missing_count`를 단일 레버로 처방 |
| 4 | `hist_gbdt` | `learning_rate` 0.05, `max_iter` 400, `max_leaf_nodes` 31, `l2_regularization` 0.5, `min_samples_leaf` 40, `class_weight` {0:1, 1:8.5}, `missing_count` true | balanced_accuracy **0.8652** (CI 0.8526~0.8758) — 최고, `cut_headroom` 0.0045, `balanced_accuracy_at_best_cut` 0.8697 | `wrong_model_family` — iteration 2와 paired Δ +0.00005 (CI -0.0059~+0.0069, 구분 불가). ranking 축 4회 평탄. `missing_indicator`를 유일한 레버로 재처방 |
| 5 | `random_forest` | `n_estimators` 600, `min_samples_leaf` 3, `class_weight` {0:1, 1:9} | balanced_accuracy **0.8498** (CI 0.8374~0.8611), `balanced_accuracy_at_best_cut` 0.8655 | (루프 종료 — 진단 없음) |

## 최고 성능 구성

iteration 4에서 실제로 빌드된 구성입니다(아래 값은 모두 history의 적용값이며, `dropped_hyperparams`는 비어 있었습니다).

- **model**: `hist_gbdt`
- **hyperparams**: `learning_rate` 0.05, `max_iter` 400, `max_leaf_nodes` 31, `l2_regularization` 0.5, `min_samples_leaf` 40, `early_stopping` false, `class_weight` {"0": 1.0, "1": 8.5}
- **preprocessing (적용값)**: `impute` median, `scale` false, `missing_indicator` false, `missing_count` true
- **프로토콜**: stratified 60/20/20, seed 44 (카드 baseline과 동일 split)
- **학습 시간**: 8.811s (예산 600s)

| 지표 | 검증(선택에 사용됨) | 홀드백 테스트(1회 채점) |
|---|---|---|
| `balanced_accuracy` | **0.8652** (CI 0.8526~0.8758) | **0.8646** (CI 0.8530~0.8743) |

검증과 테스트의 차이는 +0.0006이며, 이것이 이 실행의 **선택 편향 크기**입니다 — 루프는 검증 숫자를 보고 최고 시도를 골랐고, 테스트 숫자는 그 선택 이후 단 한 번 측정된 값입니다. 검증 점수가 테스트 CI 안에 들어오므로 이 행들로는 편향이 0과 구분되지 않습니다. 어느 쪽으로 읽어도 목표 0.8811에는 각각 0.0159 / 0.0165 미달입니다.

참고 지표(검증): `roc_auc` 0.9331, `pr_auc` 0.6259, recall 0.8611 / specificity 0.8694, `f1` 0.6049, `precision` 0.4662, `accuracy` 0.8684, `balanced_accuracy_at_best_cut` 0.8697, `cut_headroom` 0.0045, `brier` 0.0922, `calibration_error` 0.1064, `train_val_gap` 0.0649.

## 원인 분석

critic은 4회 실행되었고, 2~4회차 진단이 모두 같은 결론(`wrong_model_family`, 실질적으로는 "ranking 축의 천장")으로 수렴했습니다.

**1) operating point 축은 iteration 2에서 이미 소진되었습니다.** iteration 1→2의 개선 +0.0386(paired CI +0.0281~+0.0481)은 CI 폭보다 크므로 실재하는 개선이고, 그 내용은 전적으로 컷 이동입니다: recall/specificity 격차가 0.1768(0.7382 vs 0.9150) → 0.0007(0.8648 vs 0.8655)로 붕괴하고 `cut_headroom`이 0.0446 → 0.0063으로 줄었습니다. 최고 구성(iteration 4)의 `cut_headroom`은 0.0045로, 남은 0.0159의 28%에 불과합니다. 즉 `class_weight`/`scale_pos_weight`를 더 밀어도 산술적으로 목표에 닿지 않습니다(가중치 8.5와 9.0이 crossing을 양쪽에서 감싸고 있습니다).

**2) ranking 축은 4회에 걸쳐 평탄했습니다.** `balanced_accuracy_at_best_cut`은 0.8712 → 0.8715 → 0.8701 → 0.8697 → 0.8655로, 두 개의 gbdt 계열과 `random_forest`를 합쳐도 폭 0.006 이내이고 최고값조차 목표보다 0.0114 낮습니다. 목표 0.8811은 파생 규칙상 KS 0.7622를 요구하는데(베이스라인 KS 0.6765, 요구 증분 0.0857), 이 실행이 달성한 KS는 약 0.7430 수준으로 요구 증분의 약 3/4에서 멈췄습니다. 목표 정의 자체가 `ranking_ceiling` 0.8382를 넘어서도록(`exceeds_ranking_ceiling: true`) 설정되었다는 점도 함께 읽어야 합니다.

**3) CI가 겹치는 움직임은 설명하지 않습니다.** iteration 2와 iteration 4는 paired Δ +0.00005 (CI -0.0059~+0.0069)로 **이 데이터가 구분하지 못하는 두 시도**입니다. 게다가 이 전환에서는 `learning_rate`·`max_iter`·`l2_regularization`·`min_samples_leaf`·`class_weight`·`missing_count`가 동시에 움직였으므로, 설령 차이가 있었더라도 어느 레버에도 귀속할 수 없습니다. 반대로 CI/paired 구간보다 큰 차이는 두 개뿐입니다: iteration 1→2(+0.0386, 컷 이동)와 iteration 2→3(-0.0160, CI -0.0239~-0.0069, `xgboost` swap의 명확한 후퇴). iteration 3과 5는 서로, 그리고 최고 구성과도 CI가 부분적으로 겹칩니다.

**4) critic이 두 번 처방한 단일 레버 검증은 실제로 실행되지 않았습니다.** iteration 3과 4의 진단은 모두 "`missing_indicator`를 유일하게 움직이는 레버로 하여 iteration 2 구성을 재적합"을 처방했지만, iteration 4는 `missing_count`만 켜고 하이퍼파라미터도 함께 바꿨고, iteration 5는 `random_forest`로 갔습니다. 따라서 "결측 열이 ranking을 올리는가"라는 질문은 이 실행에서 **한 번도 깨끗하게 측정되지 않았습니다**. 다만 데이터 카드가 `missing.overall_rate` 0.0 / `columns_with_missing` 0을 보고한다는 점을 보면, 현재 파일 상태에서 `missing_count`는 사실상 상수 열이고 `missing_indicator`도 상수 열이 되므로, 실행되었더라도 얻을 것이 없었을 가능성이 큽니다. V16의 'unknown' 81.8%, V9 28.8%는 카드 caveat대로 **결측이 아니라 하나의 범주 수준으로 인코딩되어 있기 때문**입니다.

**5) unsupported_claims에 대하여.** 모든 시도에 `feature_engineering` 플래그가 붙었습니다. 기록에 전문이 남아 있는 iteration 4의 `plan_strategy`를 읽으면, 이 계획은 파생 feature를 전제하지 않고 오히려 지원되는 두 결측 열(`missing_count`/`missing_indicator`)의 선택 근거를 논한 것으로, 문자열 검사가 걸린 사례로 보입니다. 따라서 이 시도가 그 플래그 때문에 무엇을 잃었다고 쓸 근거는 없습니다. 다른 시도들의 계획 전문은 기록에 없으므로 귀속하지 않습니다. 대신 사실로 남는 것은 executor 규약 자체입니다: 열 파생·조합·재인코딩·삭제가 불가능하므로, ranking 축을 움직이려면 파일 쪽에서 열이 바뀌어야 합니다.

**6) 과적합·비용·보정은 원인이 아닙니다.** 최고 구성의 `train_val_gap`은 0.0649(train 0.9301 vs val 0.8652)로 iteration 1의 0.1509에서 이미 정리되었고, 학습 8.8s / 예산 600s로 비용 제약도 없었습니다. `calibration_error` 0.1064·`brier` 0.0922는 확률값이 체계적으로 어긋나 있음을 보여주지만, `balanced_accuracy`는 확률을 읽지 않고 이 실행에는 재보정 레버도 없으므로 격차의 설명이 될 수 없습니다.

## 다음 단계 제안

1. **데이터 카드 caveat를 먼저 해소한다 — 'unknown'을 실제 결측으로 변환한 파일을 다시 프로파일링한다.** V16 81.8%, V9 28.8%, V4 4.1%, V2 0.6%의 'unknown'이 지금은 정상 범주값으로 계산되어 있고(`missing.overall_rate` 0.0), 그 결과 `missing_count`는 상수 열, `missing_indicator`는 전부 상수 열, `impute: none`(NaN 분기)도 무의미해집니다. 이 변환 없이는 critic이 두 번 처방한 결측 레버를 아무리 실행해도 측정할 것이 없습니다. 변환 후에야 `hist_gbdt` + `impute: none`(NaN 자체 분기) 대 `median` + `missing_count`의 비교가 의미를 가집니다. 단, executor 문서가 명시한 대로 결측 레버의 크기는 이 프로젝트에서 **확립되지 않았고**(랜덤 split 0.0097 roc_auc, 연속 split +0.0011로 구간이 0을 포함), 랜덤 split은 "기록 체제(recording regime)"를 성능으로 오인할 수 있습니다. 따라서 이 레버는 "예상 이득"이 아니라 "미측정 가설"로 취급하고, 다른 레버와 섞지 말고 단일 전환으로 한 번만 측정하십시오.
2. **남은 격차를 operating point에서 찾지 마십시오. 예산은 ranking 축에만 쓰십시오.** `cut_headroom` 0.0045는 남은 0.0159의 28%이고 recall 0.8611 / specificity 0.8694는 이미 crossing 위입니다. `class_weight`를 8.5·9.0 근방에서 또 흔드는 시도는 측정 가능한 이득이 없습니다. 대신 executor 밖에서 **열을 만들어 카드에 넣는 것**이 유일하게 남은 ranking 레버입니다: `target_corr`가 strong인 V12와 skew high인 V6/V13/V14/V15에 대한 파생 열(로그/구간화/비율 등)을 원본 csv에 추가한 뒤 동일 프로토콜로 재실행하십시오. executor는 파생·조합·재인코딩을 하지 않으므로 이 작업은 반드시 파일 쪽에서 이루어져야 합니다.
3. **목표값의 타당성을 재검토하십시오.** 임계값 0.8811은 baseline KS 0.6765에서 파생되었고 `required_ks` 0.7622, `ks_shortfall` 0.0857, `exceeds_ranking_ceiling: true`로 명시되어 있습니다. 이 실행은 KS를 약 0.7430까지 끌어올려 요구 증분의 약 3/4을 확보했으며, 그 이상은 4번의 시도에서 폭 0.006 안에 갇혔습니다. 새 열이나 새 데이터 없이 동일 feature set으로 재시도하는 것은 근거가 약합니다 — 열을 늘리거나(2번), `passable_margin` 0.523 기준의 완화된 목표(≈0.8382의 `ranking_ceiling` 부근)를 채택하는 쪽 중 하나를 선택해 명시하십시오.
4. **이미 실행된 family swap을 반복하지 마십시오.** `xgboost`(iteration 3)는 -0.0160(paired CI -0.0239~-0.0069)로 해소된 후퇴였고 `random_forest`(iteration 5)의 `balanced_accuracy_at_best_cut` 0.8655는 세 계열 중 최저였습니다. gbdt 구현 간 교체는 이 파일에서 `best_cut` 0.8655~0.8715 폭만을 커버하는 반면 필요한 것은 0.0114 이상입니다. 굳이 모델 쪽을 더 쓴다면 `hist_gbdt` 한 계열 안에서 용량/정규화만 좁게 재탐색하고(현재 `train_val_gap` 0.0649이므로 여지는 크지 않음), 그 결과를 `balanced_accuracy`가 아니라 `balanced_accuracy_at_best_cut`과 `roc_auc`로 판정하십시오 — 이 두 축을 분리해 읽지 않으면 컷 이동을 개선으로 오독하게 됩니다.