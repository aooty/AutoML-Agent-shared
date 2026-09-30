## 요약

목표는 달성되었습니다. 단일 시도에서 `hist_gbdt`가 validation `balanced_accuracy` **0.8401** (95% CI 0.8310~0.8480)을 기록해 임계값 0.8248과 baseline(logreg, 0.7664)을 모두 넘었습니다. 반복 예산 1회 중 1회를 사용해 첫 계획이 바로 기준을 통과했고, 그 시점에서 루프가 종료되었습니다. 한 번도 사용되지 않은 test 20%에서 같은 모델을 다시 측정한 값은 **0.8358** (95% CI 0.8270~0.8443)이며, 이 수치가 이번 실행이 실제로 입증한 성능입니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `learning_rate=0.06`, `max_iter=500` (early_stopping, iter 158에서 정지), `max_leaf_nodes=63`, `min_samples_leaf=30`, `l2_regularization=1.0`, `class_weight={"0":1.0,"1":2.4}` / `impute: none`, `scale: false` | `status: ok` — validation `balanced_accuracy` 0.8401 (CI 0.8310~0.8480), `roc_auc` 0.9286, `train_val_gap` 0.0376 → 목표 달성 | 없음 (목표 달성으로 critic 미실행) |

## 최고 성능 구성

**iteration 1 / `hist_gbdt`** — 아래 값은 해당 시도의 히스토리에 기록된 *적용된* 하이퍼파라미터와 전처리입니다.

```
model: hist_gbdt
hyperparams:
  learning_rate: 0.06
  max_iter: 500
  max_leaf_nodes: 63
  min_samples_leaf: 30
  l2_regularization: 1.0
  early_stopping: true
  validation_fraction: 0.1
  n_iter_no_change: 30
  class_weight: {"0": 1.0, "1": 2.4}
preprocessing:
  impute: none          # NaN을 모델이 직접 분기
  scale: false
  missing_indicator: false
  missing_count: false
dropped_hyperparams: []
internal_validation: fit_rows 26373 / held_out_rows 2931, stopped_at_iter 158 (max_iter 500)
protocol: stratified 60/20/20, seed 42
train_time_sec: 8.139
```

| 지표 | validation | 비고 |
|---|---|---|
| `balanced_accuracy` | **0.8401** (95% CI 0.8310~0.8480) | 선택 기준 지표 |
| `balanced_accuracy` (held-back test) | **0.8358** (95% CI 0.8270~0.8443) | 루프 종료 후 1회만 측정 |
| `roc_auc` | 0.9286 | |
| `pr_auc` / `average_precision` | 0.8281 | |
| `f1` | 0.7247 | |
| `accuracy` | 0.8512 | |
| `precision` / `recall` | 0.6499 / 0.8190 | |
| `specificity` | 0.8613 | |
| `balanced_accuracy_at_best_cut` | 0.8500 | `cut_headroom` 0.0099 |
| `brier` / `calibration_error` | 0.0985 / 0.0700 | |
| `train_balanced_accuracy` / `train_val_gap` | 0.8778 / 0.0376 | |

validation 0.8401과 test 0.8358의 차이 **+0.0044**가 선택 편향의 크기입니다 — 루프는 validation 숫자만 보고 결정을 내렸으므로, 이 차이는 그 선택이 validation 쪽으로 얼마나 유리하게 읽혔는지를 나타냅니다. 단, validation 값이 test의 95% CI(0.8270~0.8443) 안에 들어 있어 이 행들만으로는 그 차이를 0과 구분할 수 없습니다. 두 값 모두 임계값 0.8248 위에 있습니다.

## 원인 분석

이번 실행에서는 critic이 한 번도 실행되지 않았습니다(시도 1회, 진단 0회). 첫 계획이 바로 임계값을 넘어 루프가 종료되었기 때문에, **설명할 진단 패턴이 존재하지 않습니다.** 0.8401은 오직 첫 계획 하나가 낸 점수이며, 이 실행은 진단·재계획 경로가 도움이 된다는 증거도, 해가 된다는 증거도 제공하지 않습니다.

또한 시도가 하나뿐이므로 시도 간 비교로 어떤 레버가 얼마를 벌었는지 분해할 수 없습니다. 비교 가능한 유일한 기준선은 카드의 baseline(logreg + median impute + standard scale, `balanced_accuracy` 0.7664, CI 0.7575~0.7762, `roc_auc` 0.9075, `balanced_accuracy_at_best_cut` 0.8238)이며, 이 대비 검증 점수는 +0.0737로 두 CI가 겹치지 않는 크기의 차이입니다. 다만 이 차이 안에서 모델 계열 교체(순위 축)와 `class_weight`에 의한 절단점 이동(운영점 축)이 각각 얼마를 담당했는지는 이번 실행의 측정만으로는 분리되지 않습니다. baseline의 순위 상한이 0.8238인데 현재 모델의 `balanced_accuracy_at_best_cut`이 0.8500이라는 사실은 순위 축이 함께 올라갔음을 시사하지만, 이는 두 지점 비교이지 진단이 아닙니다.

`dropped_hyperparams`는 비어 있어 제안된 설정이 그대로 적용되었습니다. `unsupported_claims`에 `feature_engineering`이 기록되어 있으나, iteration 1의 계획 본문은 "No scaling (tree family), no added missingness columns"라고 명시하며 파생 피처에 의존하지 않았습니다 — 이는 문자열 검사가 계획의 *언급*에 반응한 것이므로, 이 시도가 무언가를 잃었다고 볼 근거는 없습니다. 피처 생성 기능 부재는 아래 제안 항목의 "만들어야 할 것"에 해당합니다.

## 다음 단계 제안

1. **운영점 축(`class_weight`)은 더 손대지 말 것.** `cut_headroom`이 0.0099에 불과하고 `recall` 0.8190 / `specificity` 0.8613로 이미 균형점 근처입니다. 즉 `class_weight={"0":1.0,"1":2.4}`는 기본 `predict()` 절단이 `balanced_accuracy` 최적점 바로 옆에 놓이게 했고, 남은 여지는 순위(ranking) 쪽입니다. 이 레버를 재탐색하는 데 반복을 쓰면 리샘플 노이즈 폭(CI 폭 약 ±0.008) 이하의 움직임만 얻을 가능성이 큽니다.
2. **순위 축을 노리는 계열 교체 1회: `xgboost` + `impute: none`, `scale_pos_weight`로 동일한 절단 위치 재현.** 실행기 문서상 계열 교체가 `roc_auc`에서 움직인 폭은 0.0022~0.0077이고 페어드 해상도는 0.003~0.006이므로, 이득이 나오더라도 작을 수 있다는 점을 예산에 반영해야 합니다. 현재 `roc_auc` 0.9286에서 유의하게 구분되는 개선만 채택하고, CI가 겹치는 결과는 개선으로 보고하지 마십시오.
3. **`missing_indicator` / `missing_count`에는 반복을 쓰지 말 것.** 이 데이터는 결측률이 전체 0.0095, 최악 열 0.0575(`occupation`, `workclass`, `native-country`)로 낮고, 무엇보다 `impute: none`을 쓰는 계열에서 `missing_indicator`는 중복(예측이 비트 단위로 동일)임이 실행기 문서에서 확인되어 있습니다. `missing_count`도 별도 이득이 보고되지 않았습니다.
4. **확률값을 그대로 쓸 계획이라면 캘리브레이션을 별도로 처리할 준비.** `calibration_error` 0.0700, `brier` 0.0985 — 순위는 좋지만 확률은 평균 약 7%p 어긋나 있습니다. 이 실행기에는 재캘리브레이션 레버가 없으므로(`CalibratedClassifierCV`·절단점 이동 모두 불가), 임계값 기반 의사결정이나 확률 리포팅이 필요하면 루프 밖의 후처리 단계로 설계해야 합니다. 같은 맥락에서, 파생 피처(비율·상호작용·재인코딩)는 현재 실행기가 수행하지 못하므로 순위 축을 더 밀어올리려면 카드 단계에서 피처를 추가하는 파이프라인을 먼저 만들어야 합니다 — 그것이 열리면 `balanced_accuracy_at_best_cut` 0.8500이라는 현재의 순위 상한 자체를 올릴 수 있습니다.

> 별도로 기록된 `Data caveats`는 없으므로, 위 제안은 특정 열·값·분할의 신뢰성 문제로 인해 배제되는 항목이 없습니다.