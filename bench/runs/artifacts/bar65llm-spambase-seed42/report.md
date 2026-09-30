# AutoML 실행 최종 보고서 — `spambase` (binary_classification, 목표: `balanced_accuracy` 최대화, 기준선 통과 임계값 0.973)

## 요약

**목표는 달성하지 못했습니다.** 4회 시도(예산 5회 중) 동안 가장 좋았던 검증 점수는 iteration 2의 `xgboost`로 `balanced_accuracy` = **0.9493 (95% CI 0.9327~0.9636)**이며, 목표 임계값 0.973에는 0.0237 부족했고 임계값은 이 측정의 CI 상단(0.9636) 밖에 있습니다. 동일 모델을 루프 내에서 한 번도 사용되지 않은 held-back 테스트 20%에서 한 번 채점한 결과는 **0.9417 (95% CI 0.9250~0.9568)** 로, 이것이 이 실행이 실제로 입증한 수치입니다. 연속 미개선(정체)으로 5회 중 4회에서 조기 종료되었고, critic 진단은 3회 생성되었습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=500`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=10`, `l2_regularization=1.0`, `early_stopping=False` | `balanced_accuracy`=0.9488 (CI 0.9327~0.9640), `roc_auc`=0.9910, `balanced_accuracy_at_best_cut`=0.9601, `cut_headroom`=0.0113, `train_val_gap`=0.0512 | `overfitting` — train 1.0 / gap 0.0512로 500 iteration 예산이 2,760행을 암기. 남은 격차 0.0242 중 operating point로 회수 가능한 것은 0.0113뿐 → ranking 문제로 보고 `hist_gbdt` 재정규화 처방 |
| 2 | `xgboost` | `n_estimators=900`, `learning_rate=0.035`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.7`, `min_child_weight=2`, `reg_lambda=3.0`, `reg_alpha=0.5`, `gamma=1e-06` | `balanced_accuracy`=**0.9493** (CI 0.9330~0.9636), `roc_auc`=0.9910, `balanced_accuracy_at_best_cut`=0.9602, `cut_headroom`=0.0109, `train_val_gap`=0.0435 | `wrong_model_family` — 정규화는 capacity 축에서만 작동(train 1.0→0.9928), ranking은 불변(`roc_auc` 0.9910→0.9910, best_cut 0.9601→0.9602). 두 gbdt의 best_cut 차이 0.0001 vs 최적 컷 기준 잔여 0.0128 → bagging 계열(`random_forest`) 처방 |
| 3 | `random_forest` | `n_estimators=1200`, `min_samples_leaf=1`, `min_samples_split=2`, `class_weight=None`, `n_jobs=-1`, `random_state=42` | `balanced_accuracy`=0.9484 (CI 0.9337~0.9632), `roc_auc`=0.9882, `balanced_accuracy_at_best_cut`=0.9542, `cut_headroom`=0.0058, `train_val_gap`=0.0516 | `wrong_model_family` (2회 연속) — paired Δ vs iter2 = −0.0009 (CI −0.0116~+0.0095). 트리 3계열의 best_cut 폭이 0.0060에 불과, 임계값은 KS 0.9460을 요구하나 현재 최고 0.9204 → 축이 다른 유일한 생성기 `svc`(RBF, 스케일링 동반) 처방 |
| 4 | `svc` | `kernel='rbf'`, `C=10.0`, `probability=True`, `class_weight=None`, `cache_size=500`, `random_state=42` (전처리 `scale=True`) | `balanced_accuracy`=0.9355 (CI 0.9185~0.9523), `roc_auc`=0.9806, `balanced_accuracy_at_best_cut`=0.9408, `cut_headroom`=0.0053, `train_val_gap`=0.0313 | 없음 (평가 직후 루프 종료 — 진단 대상 아님) |

## 최고 성능 구성

iteration 2, `xgboost`. 아래 값은 히스토리에 기록된 **적용된** `hyperparams` / `preprocessing`입니다(`dropped_hyperparams`는 비어 있음).

```json
{
  "model": "xgboost",
  "hyperparams": {
    "n_estimators": 900,
    "learning_rate": 0.035,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "min_child_weight": 2,
    "reg_lambda": 3.0,
    "reg_alpha": 0.5,
    "gamma": 1e-06
  },
  "preprocessing": {
    "impute": "median",
    "scale": false,
    "missing_indicator": false,
    "missing_count": false
  }
}
```

프로토콜: stratified 3분할(train 60% / validation 20% / test 20%), seed 42. 학습 시간 4.334초.

| 지표 | 검증(validation 20%) | held-back 테스트(20%) |
|---|---|---|
| `balanced_accuracy` | **0.9493** (95% CI 0.9330~0.9636) | **0.9417** (95% CI 0.9250~0.9568) |
| `f1` | 0.9400 | — |
| `accuracy` | 0.9533 | — |
| `precision` / `recall` | 0.9493 / 0.9309 | — |
| `specificity` | 0.9677 | — |
| `roc_auc` / `pr_auc` | 0.9910 / 0.9879 | — |
| `balanced_accuracy_at_best_cut` / `cut_headroom` | 0.9602 / 0.0109 | — |
| `brier` / `calibration_error` | 0.0333 / 0.0171 | — |
| `train_balanced_accuracy` / `train_val_gap` | 0.9928 / 0.0435 | — |

검증 0.9493 → 테스트 0.9417의 차이 **+0.0076**은 선택 편향의 크기입니다. 루프는 검증 숫자만 보고 4개 시도 중 최고를 골랐으므로 그만큼 낙관적으로 기울 수밖에 없습니다. 다만 검증 점수가 테스트 CI(0.9250~0.9568) 안에 들어오므로, 이 테스트 행 수로는 그 차이를 0과 구분할 수 없습니다. 어느 쪽 숫자로도 0.973 임계값에는 도달하지 못했습니다(테스트 기준 −0.0313).

## 원인 분석

**1) 상위 3개 시도는 이 데이터로 구분되지 않습니다.** iteration 1(0.9488, CI 0.9327~0.9640), 2(0.9493, CI 0.9330~0.9636), 3(0.9484, CI 0.9337~0.9632)의 CI는 서로 거의 완전히 겹치고, critic이 계산한 paired Δ도 iter2 vs iter1 = +0.0006 (CI −0.0078~+0.0083), iter3 vs iter2 = −0.0009 (CI −0.0116~+0.0095)입니다. 즉 "xgboost가 hist_gbdt보다 좋았다"는 식의 서술은 리샘플 노이즈를 설명하는 것에 불과합니다. 검증 슬라이스 하나의 CI 폭이 약 0.030인데, 루프가 쫓던 차이는 그보다 한 자릿수 작았습니다.

**2) 부족분은 operating point가 아니라 ranking 축에 있습니다.** 세 차례 진단이 모두 같은 숫자를 지목했습니다. 최고 시도에서 `cut_headroom`은 0.0109로, 남은 격차 0.0237의 46%밖에 되지 않습니다. 즉 `class_weight` / `scale_pos_weight`로 컷을 옮겨 이론적 최적점까지 가더라도 `balanced_accuracy_at_best_cut` = 0.9602에서 멈추며, 이는 임계값보다 여전히 0.0128 낮습니다. `recall` 0.9309 < `specificity` 0.9677이라 컷이 약간 보수적이라는 신호는 있지만, 그 가치의 상한이 바로 0.0109입니다.

**3) 그리고 ranking은 트리 계열 안에서 움직이지 않았습니다.** iteration 1→2에서 `roc_auc`는 0.9910→0.9910, best_cut은 0.9601→0.9602 — 정규화(train 1.0→0.9928, gap 0.0512→0.0435)는 capacity 축만 정리하고 순위에는 아무 것도 주지 않았습니다. 세 트리 계열의 best_cut 전체 폭은 0.9542~0.9602(0.0060)로, 최적 컷 기준 잔여 0.0128의 절반 수준입니다. 유일하게 CI 폭보다 큰, 따라서 읽을 만한 변화는 iteration 4의 `svc`입니다: `roc_auc` 0.9910→0.9806(−0.0104), best_cut 0.9602→0.9408(−0.0194)로 축이 다른 가설 공간(RBF 커널)은 이 데이터에서 트리 순위보다 **나빴습니다**. `balanced_accuracy` 자체(0.9355 vs 0.9493)는 CI가 겹쳐 단독으로는 결론이 되지 않지만, ranking 지표의 하락 폭은 이 실행에서 관측된 해상도(약 0.003~0.006)를 넘습니다.

**4) 임계값 자체가 이 실행이 접근할 수 있는 범위 밖에 있었습니다.** 목표 정의는 `exceeds_ranking_ceiling: true`, `required_ks: 0.946`, `ks_shortfall: 0.063`을 명시합니다. 기준선 `logreg`의 KS는 0.883, 이번 최고 모델의 KS는 0.9204(critic 계산)로 0.946에 미달합니다. 실행이 기준선(0.9228)과 `ranking_ceiling`(0.9415)을 모두 넘어섰음에도 목표에 닿지 못한 것은, 남은 0.024가 하이퍼파라미터·컷·모델 계열 교체로 채워지는 종류가 아니라 **입력 표현(feature) 쪽에 있다는 뜻**입니다. 그리고 이 executor는 파생 피처·열별 전처리·컬럼 변환을 할 수 없으므로(추가 가능한 열은 `missing_indicator`/`missing_count`뿐이고, 결측률 0.0인 이 데이터에서는 상수열), 루프에는 그 축을 시험할 수단이 아예 없었습니다. `unsupported_claims`는 모든 시도에서 비어 있으므로, 계획이 사용 불가 기능에 의존해 손실을 본 시도는 없습니다 — 위 숫자는 이 executor가 이 표현으로 낼 수 있는 값입니다.

## 다음 단계 제안

(`Data caveats`에 기록된 항목이 없어, 특정 열·값·분할을 배제하는 제약은 없습니다. 아래 제안은 executor의 CAN/CANNOT 경계를 기준으로 구분했습니다.)

1. **측정 해상도를 먼저 올릴 것 — 이것 없이는 다음 튜닝도 리샘플을 쫓게 됩니다.** 검증 슬라이스 CI 폭 ≈0.030에 비해 iteration 1~3의 차이는 0.001 수준이었습니다. 현재 executor는 cross-validation과 out-of-fold 예측을 지원하지 않으므로(고정 20% 단일 분할), **k-fold 또는 반복 분할 채점을 실행 프로토콜에 추가**하는 것이 우선 과제입니다. 이것이 열리면 "gbdt 재튜닝이 실제로 뭔가를 주는지"를 처음으로 판정할 수 있습니다.
2. **ranking 축을 실제로 움직일 수 있는 유일한 남은 레버는 feature 표현이며, 지금은 executor 기능 확장이 필요합니다.** 카드상 57개 열 중 거의 전부가 `magnitude: sub_unit`, `skew: high`이고 `capital_run_length_*`만 `tens` 스케일에 outlier_rate 0.08~0.12입니다. 이런 희소·왜곡 카운트에 대한 `log1p`류 단조 변환은 트리 순위를 바꾸지 않지만 `svc`/`logreg` 계열의 순위에는 직접 영향을 주며, 열별 전처리(트리엔 원값, 커널엔 변환값)는 현재 금지되어 있습니다. **열 단위 변환 및 열별 전처리 파이프라인을 executor에 추가**한 뒤 `svc`를 재측정할 것을 권합니다 — iteration 4가 실패한 것은 커널 자체보다 표준화만 얹은 원 스케일 왜곡 분포일 가능성이 남아 있고, 이 실행은 그것을 분리하지 못했습니다.
3. **`class_weight` / `scale_pos_weight`에는 남은 예산을 쓰지 말 것.** 최고 시도의 `cut_headroom` 0.0109(그리고 iteration 3·4는 0.0058/0.0053)가 그 레버의 상한이며, 필요한 것은 0.0237입니다. 완벽한 컷을 잡아도 0.9602에서 멈춥니다. 이 축은 이미 값이 매겨져 소진되었습니다.
4. **목표 임계값 0.973을 재검토할 것.** 목표 정의 자체가 `exceeds_ranking_ceiling: true`, `ks_shortfall: 0.063`, `passable_margin: 0.242`를 기록하고 있고, 실측 결과는 기준선 0.9228과 `ranking_ceiling` 0.9415를 모두 넘긴 상태에서 held-back 0.9417 (CI 0.9250~0.9568)입니다. 현 표현·현 executor 능력으로 0.973은 도달 가능한 지점이 아니라는 것이 4회 시도의 결론이므로, (a) 임계값을 held-back 기준 달성 가능 구간으로 재협상하거나 (b) 1·2번의 기능 확장을 선행 조건으로 명시한 뒤 재실행하는 편이, 다섯 번째 트리 하이퍼파라미터 점을 찍는 것보다 정보량이 큽니다.