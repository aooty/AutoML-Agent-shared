# AutoML 실행 최종 보고서 — `spambase` / `balanced_accuracy`

## 요약

목표를 달성하지 못했습니다. 목표는 validation `balanced_accuracy` ≥ 0.9614였고, 5회 시도 중 최고는 iteration 5의 `hist_gbdt`로 validation 0.9612 (95% CI 0.9456~0.9733) — 임계값에 0.0002 부족했으며, 루프는 최대 반복 횟수(5회) 소진으로 종료되었습니다. 같은 모델을 첫 학습 이전에 분리해 둔 test 20%에서 한 번만 측정한 결과는 `balanced_accuracy` 0.9427 (95% CI 0.9279~0.9575)로, 임계값에서 더 크게 벗어났습니다. 참고로 이 임계값 자체가 baseline(logreg, 0.9228)의 ranking ceiling 0.9415를 넘어서도록 설정된(`exceeds_ranking_ceiling: true`) 공격적인 목표였습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter` 400, `learning_rate` 0.06, `max_leaf_nodes` 31, `min_samples_leaf` 20, `l2_regularization` 1.0, `early_stopping` false | balanced_accuracy **0.9534** (CI 0.9384~0.9671), roc_auc 0.9912, best_cut 0.9585, `cut_headroom` 0.0051, `train_val_gap` 0.0461 | `overfitting` — train_balanced_accuracy 0.9995 vs val 0.9534. 동일 family를 저분산 설정으로 재적합 지시 |
| 2 | `xgboost` | `n_estimators` 900, `learning_rate` 0.03, `max_depth` 4, `subsample` 0.8, `reg_lambda` 3.0 | balanced_accuracy **0.9517** (CI 0.9358~0.9660), roc_auc 0.9914, best_cut 0.9589, `cut_headroom` 0.0072, `train_val_gap` 0.0363 | `wrong_model_family` — 지시된 `hist_gbdt` 재적합이 아니라 family가 교체됨. gap은 해소되었으나 ranking은 0.0004 이동. bagged family 시도 지시 |
| 3 | `random_forest` | `n_estimators` 1200, `min_samples_leaf` 1, `class_weight` {0:1.0, 1:1.35} | balanced_accuracy **0.9545** (CI 0.9400~0.9681), roc_auc 0.9892, best_cut 0.9546, `cut_headroom` 0.000145, `calibration_error` 0.0524 | `wrong_model_family` — 운용점(operating point)이 소진됨(recall 0.9448 ≈ specificity 0.9642). ranking은 boosting보다 나쁨. `hist_gbdt` + `impute: none` + `missing_count` 지시 |
| 4 | `svc` | `C` 10.0, `kernel` rbf, `class_weight` {0:1.0, 1:1.15}, `scale` true | balanced_accuracy **0.9337** (CI 0.9179~0.9500) — 유일하게 임계값을 CI로 배제 | `wrong_model_family` — iteration 3 대비 paired Δ -0.0207 (CI -0.0352~-0.0067, P(better) 0.003)로 해소된 회귀. 지시한 preprocessing 변경은 반영되지 않아 미검증. 마지막 시도는 `hist_gbdt` 저분산 설정으로 복귀 지시 |
| 5 | `hist_gbdt` | `learning_rate` 0.03, `max_iter` 900, `max_leaf_nodes` 15, `max_depth` 6, `min_samples_leaf` 10, `l2_regularization` 0.5, `early_stopping` false, `class_weight` {0:1.0, 1:1.2} | balanced_accuracy **0.9612** (CI 0.9456~0.9733), roc_auc 0.9922, best_cut 0.9645, `cut_headroom` 0.0033 — 임계값 0.0002 미달 | (마지막 시도, critic 미실행) |

## 최고 성능 구성

iteration 5, `hist_gbdt`. 아래 `hyperparams`/`preprocessing`은 해당 시도 이력에 기록된 **실제 적용값**입니다.

```json
{
  "model": "hist_gbdt",
  "hyperparams": {
    "learning_rate": 0.03,
    "max_iter": 900,
    "max_leaf_nodes": 15,
    "max_depth": 6,
    "min_samples_leaf": 10,
    "l2_regularization": 0.5,
    "early_stopping": false,
    "class_weight": {"0": 1.0, "1": 1.2},
    "random_state": 42
  },
  "preprocessing": {
    "impute": "median",
    "scale": false,
    "missing_indicator": false,
    "missing_count": false
  }
}
```

- `dropped_hyperparams`: 없음. `status`: ok. `train_time_sec`: 5.912 (예산 600초).
- 분할 프로토콜: stratified 60/20/20, seed 42 (카드 baseline과 동일 분할).

| 측정 | balanced_accuracy | 95% CI |
|---|---|---|
| validation 20% (선택에 사용됨) | **0.9612** | 0.9456~0.9733 |
| test 20% (첫 적합 전에 분리, 단 한 번 채점) | **0.9427** | 0.9279~0.9575 |

두 값의 차이 **+0.0185**(validation이 높음)가 이 실행의 **선택 편향(selection effect)** 크기입니다. 루프는 validation 숫자만 보고 5개 시도 중 최고를 골랐으므로, 이 모델에 대해 실제로 입증된 것은 test의 0.9427이며 0.9612가 아닙니다. test CI(0.9279~0.9575) 상한도 임계값 0.9614에 미치지 못합니다.

validation 기준 그 밖의 지표(iteration 5): `f1` 0.9541, `accuracy` 0.9641, `precision` 0.9608, `recall` 0.9475, `specificity` 0.97491, `roc_auc` 0.9922, `pr_auc` 0.9889, `balanced_accuracy_at_best_cut` 0.9645, `cut_headroom` 0.003288, `brier` 0.031229, `calibration_error` 0.018214, `train_balanced_accuracy` 0.9979, `train_val_gap` 0.036651.

## 원인 분석

critic 진단은 4회 존재하며, 그 패턴은 비교적 일관됩니다.

**1) 측정 해상도가 대부분의 시도 간 차이를 삼켰습니다.** validation CI 폭은 대략 ±0.014~0.015입니다. iteration 1(0.9534, CI 0.9384~0.9671), 2(0.9517, 0.9358~0.9660), 3(0.9545, 0.9400~0.9681), 5(0.9612, 0.9456~0.9733)의 구간은 서로 크게 겹치므로, 이 데이터로는 **네 시도를 구분할 수 없습니다**. 특히 iteration 1 → 5의 +0.0078은 구간 폭보다 작아 "저분산 설정이 ranking을 개선했다"고 단정할 수 없습니다(다만 `roc_auc` 0.9912→0.9922, `balanced_accuracy_at_best_cut` 0.9585→0.9645는 같은 방향을 가리키는 보조 신호입니다). 유일하게 해소된 차이는 iteration 4의 `svc`로, iteration 3 대비 paired Δ -0.0207 (CI -0.0352~-0.0067)의 명확한 회귀였습니다 — 단 이 행은 family와 `scale` 두 레버가 동시에 움직여 손실의 귀속이 불가능합니다.

**2) 남은 격차는 운용점이 아니라 ranking에 있었고, ranking 여유가 실제로 거의 없었습니다.** critic이 반복해 지적한 대로 `cut_headroom`은 iteration 3에서 0.000145까지 소진되었고(recall 0.9448 ≈ specificity 0.9642), 최고 시도인 iteration 5에서도 0.0033에 불과합니다. 즉 `class_weight`나 임계값 이동으로 살 수 있는 몫은 극히 작았습니다. 동시에 완벽한 컷을 가정한 `balanced_accuracy_at_best_cut`조차 iteration 1~3에서 0.9546~0.9589(폭 0.0043)에 머물러, 임계값 0.9614는 세 트리 family의 **최대치보다도 위**에 있었습니다. iteration 5에서 이 값이 0.9645로 올라 처음으로 임계값을 넘었지만, 그 여유는 0.0031이며 실제 컷은 0.9612에서 멈췄습니다.

**3) family 교체는 4회 시도에도 ranking을 거의 옮기지 못했습니다.** `hist_gbdt`/`xgboost`/`random_forest`는 서로 구분되지 않고, `svc`는 확실히 나빴습니다. 성능을 실제로 제한한 것은 "잘못된 family"가 아니라 **이 executor에서 남은 레버가 사실상 하나(같은 boosting family 내부의 용량/정규화 설정)뿐이었다는 점**입니다. iteration 1의 `overfitting` 진단(train 0.9995, gap 0.0461)은 방향으로는 옳았고 iteration 5에서 gap 0.0367까지 내려갔지만, train_balanced_accuracy는 여전히 0.9979로 학습 분할을 거의 암기한 상태입니다.

**4) critic 지시가 두 번 그대로 실행되지 않았습니다.** iteration 2는 지시된 `hist_gbdt` 재적합 대신 `xgboost`가, iteration 4는 지시된 `hist_gbdt` + `impute: none` + `missing_count` 대신 `svc`(+`scale` 활성)가 실행되었습니다. 결과적으로 5회 중 실질적으로 "가장 유망한 family를 저분산 지점에서 검증"한 시도는 iteration 5 단 한 번이며, 예산의 관점에서는 낭비가 있었습니다(모든 학습이 3.2~5.9초, 600초 예산의 1% 미만).

**5) 다만 critic이 iteration 3에서 제시한 "특징 행렬(missingness) 레버"는 이 데이터에서 애초에 아무 효과가 없었을 것입니다.** 데이터 카드는 57개 열 전부 `missing_rate` 0.0, `columns_with_missing` 0, `overall_rate` 0.0입니다. 즉 `impute: none`은 대치할 값이 없고 `missing_count`는 상수 0 열이 되므로, 이 방향은 미검증인 채로 남았지만 검증 가치도 없습니다. 실행되지 않은 것이 손실은 아니었습니다.

정리하면: 이 실행의 병목은 (a) 임계값 0.9614가 측정된 모든 family의 best-cut 상한 근처에 놓여 있었다는 점, (b) 운용점 여유가 0.003 수준으로 소진되어 남은 몫이 전부 ranking 축에 있었다는 점, (c) executor가 특징 파생을 지원하지 않아 ranking을 올릴 수 있는 유일한 레버가 boosting 용량 설정이었다는 점, 그리고 (d) validation 단일 20% 분할의 해상도(±0.015)가 임계값까지의 거리(0.007~0.010)보다 커서 어떤 시도도 결정적으로 판정될 수 없었다는 점입니다. test 0.9427이 validation 0.9612보다 0.0185 낮은 것은 (d)의 직접적 귀결입니다.

## 다음 단계 제안

1. **`hist_gbdt`의 저분산 방향을 한 단계 더 밀어붙일 것 (최우선, 비용 거의 없음).** iteration 1→5에서 `learning_rate` 0.06→0.03, `max_leaf_nodes` 31→15로 옮겼을 때 `roc_auc` 0.9912→0.9922, `balanced_accuracy_at_best_cut` 0.9585→0.9645로 두 ranking 지표가 함께 올라간 유일한 이동이 있었습니다. 그런데도 `train_balanced_accuracy` 0.9979 / `train_val_gap` 0.0367로 여전히 암기 여지가 남아 있습니다. `learning_rate` 0.015~0.02, `max_iter` 2000~3000, `max_leaf_nodes` 7~8, `min_samples_leaf` 20~30, `l2_regularization` 2.0~5.0, `early_stopping` false, `class_weight` {0:1.0, 1:1.15~1.25}를 2~3점 정도 훑을 것을 권합니다. 학습 시간이 5.9초/600초였으므로 예산상 전혀 부담이 없습니다.
2. **성공 판정 기준을 CI에 맞게 재설정하거나 여러 seed로 재실행할 것.** 임계값까지의 거리(0.0002~0.0100)가 validation CI 폭(±0.015)보다 작은 상태에서는 "달성/미달성" 판정 자체가 리샘플 노이즈에 좌우됩니다. executor 안에서는 분할·seed·CV를 바꿀 수 없으므로, 루프 **밖에서** 동일 구성(iteration 5)을 여러 seed로 재실행해 test 점수 분포를 확보하고, 목표를 baseline의 ranking ceiling(0.9415) 대비로 다시 유도하는 것이 필요합니다. 이것이 확보되면 1번의 미세 튜닝 결과를 신뢰할 수 있게 판정할 수 있습니다.
3. **ranking을 더 올리려면 특징 쪽을 데이터 파일 단계에서 손볼 것.** 카드상 57개 열 대부분이 `skew: high`의 `sub_unit` 빈도이고 `capital_run_length_*` 3개는 `tens` 규모의 고왜도·고이상치 열입니다. `cut_headroom` 0.0033은 운용점에 남은 게 없다는 뜻이므로 남은 축은 ranking뿐인데, executor는 파생·변환·삭제를 하지 않습니다. 따라서 log1p 변환이나 capital_run 비율 같은 열은 **입력 CSV/데이터 카드에 미리 만들어 넣는 방식**으로만 시험할 수 있습니다. 이것이 트리 family 내부 튜닝으로 닫히지 않은 유일한 실질적 여지입니다.
4. **다음 시도에서 하지 말아야 할 것을 명시적으로 배제할 것.** (a) missingness 레버(`impute: none`, `missing_indicator`, `missing_count`) — 57개 열 전부 `missing_rate` 0.0이므로 상수 열 추가 외에 아무 일도 하지 않습니다. iteration 3의 critic 지시는 이 데이터에서는 무효였습니다. (b) 새로운 family 탐색 — 3개 트리 family가 CI로 구분되지 않고 `svc`는 해소된 회귀(-0.0207)였으므로, 남은 예산을 5번째 family에 쓰는 것보다 1번의 용량 탐색에 쓰는 편이 근거가 강합니다. (c) 임계값 스윕/확률 보정 — executor가 지원하지 않으며, `cut_headroom` 0.0033 / `calibration_error` 0.0182가 이미 그 여지가 작음을 보여줍니다.