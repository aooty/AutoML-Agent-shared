# AutoML 실행 보고서 — `adult` / balanced_accuracy

## 요약

목표는 **달성하지 못했습니다**. 목표 기준선은 `balanced_accuracy` 0.9194였고, 5회 예산 중 3회를 사용한 뒤 연속 미개선(정체)으로 조기 종료되었으며, 최고 성능은 iteration 1의 `hist_gbdt`가 기록한 검증 `balanced_accuracy` **0.8442 (95% CI 0.8362~0.8517)** 로 기준선까지 0.0752가 부족합니다. 이 최고 구성을 한 번도 사용되지 않은 테스트 20%에서 채점한 결과는 **0.8432 (95% CI 0.8354~0.8515)** 입니다. 세 번의 시도 중 서로 구별 가능한 차이는 없었고(모든 CI가 중첩), 부족분은 전부 랭킹(ranking) 축에 남아 있습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=false`, `class_weight='balanced'` / `impute='none'`, `scale=false` | `balanced_accuracy` **0.8442** (CI 0.8362~0.8517), `roc_auc` 0.9272, `pr_auc` 0.8288, `balanced_accuracy_at_best_cut` 0.8451, `cut_headroom` 0.000936, `train_val_gap` 0.0444 — **최고** | `wrong_model_family`: 기준선까지 0.0752 남았고 CI 폭(0.0155)의 약 5배. 운영점은 소진(`cut_headroom`이 남은 격차의 1.2%), KS 0.6531→~0.6902 vs 필요값 0.8388. 다음은 `xgboost` 계열 교체 제안 |
| 2 | `xgboost` | `n_estimators=900`, `learning_rate=0.04`, `max_depth=7`, `subsample=0.9`, `reg_lambda=2.0`, `scale_pos_weight=3.18` / `impute='none'`, `scale=false` | `balanced_accuracy` 0.8418 (CI 0.8342~0.8501), `roc_auc` 0.9255, `pr_auc` 0.8265, `balanced_accuracy_at_best_cut` 0.8436, `cut_headroom` 0.001841, `train_val_gap` 0.0783 — iteration 1과 **구별 불가**(paired Δ −0.0024, CI −0.0068~+0.0020, P(better) 0.145) | `wrong_model_family`: 앞선 계열 교체 처방이 아무것도 사지 못했음을 인정. 두 계열의 `balanced_accuracy_at_best_cut` 차이는 0.0015인데 남은 거리는 0.0743(약 50배) → 세 번째 계열 교체는 산술적으로 경로가 아님. 유일한 미시도 레버인 `missing_count`를 iteration 1 설정 고정 하에 1회만 측정하라고 처방 |
| 3 | `hist_gbdt` | iteration 1과 동일한 하이퍼파라미터 / 적용된 전처리는 `impute='median'`, `scale=false`, `missing_indicator=true`, `missing_count=true` | `balanced_accuracy` 0.8442 (CI 0.8362~0.8517) — iteration 1과 **모든 지표가 소수점까지 동일**, `train_time_sec` 23.864 | critic 미실행(평가 직후 루프 종료) |

`dropped_hyperparams`는 세 시도 모두 비어 있으므로, 계획한 설정이 조용히 무시된 경우는 없습니다.

## 최고 성능 구성

iteration 1, `hist_gbdt` — 아래는 히스토리에 기록된 **적용된** 값입니다.

```
model: hist_gbdt
hyperparams:
  max_iter: 400
  learning_rate: 0.06
  max_leaf_nodes: 31
  min_samples_leaf: 20
  l2_regularization: 1.0
  early_stopping: false
  class_weight: "balanced"
preprocessing:
  impute: none            # 모델이 NaN을 직접 분기
  scale: false
  missing_indicator: false
  missing_count: false
protocol: stratified 60/20/20, seed 43 (카드의 baseline과 동일 분할)
train_time_sec: 13.664
```

검증 지표: `balanced_accuracy` 0.8442 (95% CI 0.8362~0.8517), `roc_auc` 0.9272, `pr_auc` / `average_precision` 0.8288, `f1` 0.7163, `accuracy` 0.8376, `precision` 0.6154, `recall` 0.8567, `specificity` 0.8317, `brier` 0.1081, `calibration_error` 0.0986, `balanced_accuracy_at_best_cut` 0.8451, `cut_headroom` 0.000936, `train_balanced_accuracy` 0.8886, `train_val_gap` 0.0444.

**보류된 테스트 20% 채점: `balanced_accuracy` = 0.8432 (95% CI 0.8354~0.8515).** 검증 0.8442 대비 −0.0010이며, 이 차이가 선택 편향의 크기입니다 — 루프는 검증 숫자만 보고 최고 시도를 골랐습니다. 다만 검증 점수가 테스트 CI 안에 들어 있어, 이 행 수로는 그 편향을 0과 구분할 수 없습니다. 어느 쪽이든 목표 0.9194에는 크게 미치지 못합니다.

## 원인 분석

**부족분은 전부 랭킹 축에 있습니다.** 두 critic 판정(`재계획` 2회, 둘 다 `wrong_model_family`)은 같은 근거를 반복해서 제시했습니다: iteration 1에서 `cut_headroom`은 0.000936으로 남은 격차 0.0752의 1.2%, iteration 2에서도 0.001841로 남은 격차 0.0776의 약 2%였고, `recall`(0.8567 / 0.8383)과 `specificity`(0.8317 / 0.8453)가 거의 같은 지점에 있었습니다. 즉 `class_weight='balanced'`나 `scale_pos_weight=3.18`로 운영점을 더 움직여 얻을 수 있는 양은 0.002 미만이며, 실행기에서 임계값 탐색은 애초에 불가능한 레버입니다.

**계열 교체는 측정 가능한 이득을 주지 못했습니다.** iteration 1 → 2의 `balanced_accuracy` 차이는 −0.0024이고 paired CI가 −0.0068~+0.0020으로 0을 포함하며, 두 시도의 개별 CI(0.8362~0.8517 / 0.8342~0.8501)도 크게 겹칩니다. `roc_auc` 0.9272 vs 0.9255, `pr_auc` 0.8288 vs 0.8265, `balanced_accuracy_at_best_cut` 0.8451 vs 0.8436 — 모두 이 데이터가 분해하지 못하는 크기입니다. 따라서 `hist_gbdt`가 `xgboost`보다 낫다는 이야기는 이 실행에서 할 수 없습니다. 두 번째 critic이 스스로 적었듯, 자신이 처방한 계열 교체는 아무것도 사지 못했고 관측된 랭킹 상한 폭(0.8436~0.8451, 0.0015)은 남은 거리(0.0743)의 50분의 1에 불과합니다.

**반면 기준선 대비 계열 교체 자체는 실재하는 이득이었습니다** — 이것은 CI 폭보다 훨씬 큰 차이입니다. baseline `logreg`의 0.7698 (CI 0.7586~0.7798) → 0.8442, `roc_auc` 0.908 → 0.9272, `balanced_accuracy_at_best_cut` 0.8266 → 0.8451, KS 0.6531 → ~0.6902. 그러나 목표가 요구하는 KS는 0.8388이고, 목표 정의 자체가 `exceeds_ranking_ceiling: true`, `ks_shortfall: 0.1857`로 기록되어 있습니다 — **0.9194는 baseline의 랭킹으로는 어떤 임계값을 골라도 도달 불가능한 값으로 파생된 기준선**이며, 실제로 이번에 측정된 두 강력한 트리 계열의 랭킹 상한도 0.845 부근에서 멈췄습니다.

**용량(capacity) 문제도 아닙니다.** iteration 1은 `train_balanced_accuracy` 0.8886으로 **훈련 데이터에서조차 기준선 0.9194를 넘지 못했고**, `train_val_gap`은 0.0444에 불과합니다. iteration 2에서 `max_depth=7`, 900라운드로 용량을 늘렸을 때 훈련 점수는 0.9200까지 올랐지만 `train_val_gap`이 0.0783으로 커지고 검증 점수는 (구별 불가한 범위에서) 오히려 내려갔습니다 — 추가 용량이 랭킹이 아니라 암기로 흡수되고 있다는 신호입니다.

**iteration 3은 의도된 실험을 깨끗하게 수행하지 못했습니다.** critic은 `impute='none'` + `missing_count=true`만 바꾸어 전처리 단독 기여를 측정하라고 처방했지만, 기록된 적용 전처리는 `impute='median'` + `missing_indicator=true` + `missing_count=true`입니다(카드 정책상 `missing_indicator`는 `impute='none'`과 함께면 비트 단위로 중복이라는 사실이 이미 확정되어 있습니다). 게다가 결과 지표가 iteration 1과 소수점까지 완전히 동일합니다. 따라서 이 시도는 "결측 정보 열이 아무 값도 없었다"는 증거로 읽을 수도, "변경이 실제로 파이프라인에 반영되었는지"를 확인해야 하는 미해결 항목으로 읽을 수도 있습니다 — 어느 쪽이든 유일하게 남아 있던 정보 축 레버는 **아직 유효하게 측정되지 않았습니다**.

한 가지 정리해 둘 점: iteration 1의 `unsupported_claims`에 `feature_engineering`이 올라와 있지만, 계획 본문은 실행기에 파생 열을 요구한 것이 아니라 GBDT가 내부적으로 `education-num × hours-per-week × marital-status` 같은 비선형 구조를 스스로 표현할 수 있다는 점을 설명한 것입니다. 즉 이 시도는 사용할 수 없는 기능에 의존해 무언가를 잃은 것이 아니며, 0.8442는 이 실행기가 실제로 낸 숫자입니다. 다만 "열을 새로 만드는 능력"이 없다는 사실은 아래 제안의 근거가 됩니다.

## 다음 단계 제안

기록된 `Data caveats`는 없으므로, 아래 제안은 특정 열·값·분할의 신뢰성 문제로 배제되는 것이 없습니다.

1. **목표 기준선 0.9194를 재검토하십시오 (가장 우선).** 이 값은 baseline에서 `margin=0.65`로 파생된 값이고, 정의 자체에 `exceeds_ranking_ceiling: true`, `required_ks: 0.8388`, `ks_shortfall: 0.1857`이 기록되어 있습니다. 실제로 측정된 것은 두 개의 강한 트리 계열이 `balanced_accuracy_at_best_cut` 0.8436~0.8451, `roc_auc` 0.9255~0.9272에서 멈춘다는 사실이고, 훈련 점수(0.8886)조차 기준선 아래입니다. 이 14개 열 + 실행기가 허용하는 레버 조합으로는 0.9194가 도달 대상이 아닐 가능성이 큽니다. 관측된 랭킹 상한(0.8451)을 기준으로 목표를 다시 파생하면, 이후 예산이 "이미 소진된 운영점"과 "구별 불가능한 계열 교체"에 낭비되지 않습니다.

2. **iteration 3을 올바른 형태로 1회 재실행해 결측 정보 레버를 닫으십시오.** iteration 1 설정을 그대로 두고 `impute='none'`, `missing_indicator=false`, `missing_count=true`만 적용하고, `applied_preprocessing`이 실제로 그렇게 기록되는지 확인합니다. 지금은 적용 전처리가 처방과 다르고 지표가 iteration 1과 완전히 동일해서 "효과 없음"과 "변경 미반영"을 구분할 수 없습니다. 비용은 24초 수준이며, 결과가 CI 폭 0.0155 밖으로 움직이지 않으면 이 축은 공식적으로 종료됩니다. (참고로 카드는 `missing_count`가 무작위 분할에서 0.0003, 지시자 위에서 0.0000이었다고 적고 있으므로, 기대값은 크지 않습니다.)

3. **정보 축은 실행기 밖에서 열어야 합니다 — 파생 열을 CSV/카드 단계에서 만드십시오.** 실행기는 비율·차이·상호작용·재인코딩·열 삭제를 하지 못하고(`missing_indicator`/`missing_count`가 유일한 예외), 운영점은 이미 `cut_headroom` 0.000936으로 소진되었으므로 남은 유일한 축은 입력 열 자체입니다. 카드가 지목하는 후보는 명확합니다: `capital-gain`(skew high, `magnitude: sub_unit`)의 0 스파이크 플래그와 로그 변환, `capital-loss`도 동일, `hours-per-week`(outlier_rate 0.2763)의 구간화, `education-num`(target_corr strong) × `marital-status`/`hours-per-week` 상호작용, 그리고 `fnlwgt`(target_corr none — 표본 가중치 성격)의 제거. 트리가 일부를 스스로 표현하므로 기대치는 크지 않게 잡되, 이것이 이 실행에서 유일하게 시도되지 않은 축입니다.

4. **같은 계열 안에서 하이퍼파라미터를 더 돌리는 데 남은 예산을 쓰지 마십시오.** iteration 1의 `train_balanced_accuracy` 0.8886이 이미 기준선 아래이고, iteration 2가 용량을 키웠을 때 `train_val_gap`만 0.0444 → 0.0783으로 벌어졌습니다. 세 번째 계열 교체도 같은 이유로 배제됩니다(계열 간 상한 폭 0.0015 vs 남은 거리 0.0743). 대신 확률값을 실제로 사용할 계획이 있다면 `calibration_error` 0.0986 / `brier` 0.1081을 별도 항목으로 다루십시오 — 이 실행기에는 재교정 레버가 없으므로(`CalibratedClassifierCV`, 임계값 이동 모두 불가) 이는 파이프라인 밖에서 구축해야 하는 작업이며, `balanced_accuracy`에는 영향이 없지만 0.10 수준의 과신은 그대로 남아 있습니다.