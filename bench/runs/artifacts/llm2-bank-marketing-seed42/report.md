# AutoML 실행 보고서 — bank-marketing (binary_classification)

## 요약

목표는 달성되었습니다. 목표 지표 `balanced_accuracy` 임계값 0.84에 대해, 첫 번째 시도의 `hist_gbdt`가 검증 슬라이스에서 **0.8705** (95% CI 0.8588~0.8808)를 기록했고, 최종적으로 한 번도 사용되지 않은 홀드백 테스트 20%에서 **0.8554** (95% CI 0.8434~0.8667)로 임계값을 넘겼습니다. 소요 시도는 5회 예산 중 **1회**이며, 학습 시간은 7.874초였습니다. 카드의 baseline(`logreg`, balanced_accuracy 0.662)에 비해 큰 폭의 개선이지만, 개선의 축이 무엇인지는 아래 `## 원인 분석`에서 구분해 읽어야 합니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.08`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `class_weight='balanced'`, `early_stopping=False` | status `ok` — `balanced_accuracy`=0.8705 (CI 0.8588~0.8808), `roc_auc`=0.9387, `train_val_gap`=0.0775 | 없음 (목표 달성으로 loop 종료, critic 미실행) |

## 최고 성능 구성

iteration 1, `hist_gbdt`. 아래 `hyperparams`와 `preprocessing`은 history에 기록된 **실제 적용값**입니다 (`dropped_hyperparams`는 비어 있음 — 제안된 키가 모두 수용되었습니다).

```json
{
  "model": "hist_gbdt",
  "hyperparams": {
    "max_iter": 400,
    "learning_rate": 0.08,
    "max_leaf_nodes": 31,
    "l2_regularization": 1.0,
    "class_weight": "balanced",
    "early_stopping": false
  },
  "preprocessing": {
    "impute": "median",
    "scale": false,
    "missing_indicator": false,
    "missing_count": false
  }
}
```

프로토콜: stratified 3-way split, seed 42, train 60% / validation 20% / test 20% (카드 baseline과 동일한 split).

| 지표 | 검증(validation 20%) |
|---|---|
| `balanced_accuracy` | **0.8705** (95% CI 0.8588~0.8808) |
| `roc_auc` | 0.9387 |
| `pr_auc` / `average_precision` | 0.6324 |
| `f1` | 0.6270 |
| `accuracy` | 0.8807 |
| `precision` | 0.4943 |
| `recall` | 0.8573 |
| `specificity` | 0.8838 |
| `balanced_accuracy_at_best_cut` | 0.8812 |
| `cut_headroom` | 0.0107 |
| `brier` | 0.0845 |
| `calibration_error` | 0.0935 |
| `train_balanced_accuracy` | 0.9480 |
| `train_val_gap` | 0.0775 |
| `train_time_sec` | 7.874 |

**홀드백 테스트 측정 (1회, 루프 종료 후):** `balanced_accuracy` = **0.8554** (95% CI 0.8434~0.8667).

검증 0.8705와 테스트 0.8554의 차이 **+0.0151**(검증이 더 높음)은 선택 편향의 크기입니다. 루프는 검증 숫자만 보고 최선을 골랐으므로, 이 모델에 대해 이 실행이 실제로 입증한 값은 테스트의 0.8554입니다. 두 값 모두 임계값 0.84 위에 있고, 테스트 CI의 하한(0.8434)도 0.84를 넘습니다.

## 원인 분석

`재계획` 라인이 명시하듯 **critic 검증(verdict)은 0건**입니다. 첫 계획이 임계값을 넘겨 루프가 즉시 종료되었으므로, 시도 간 패턴이라 부를 것이 존재하지 않습니다. 이 점수는 **첫 계획 하나가 낸 결과**이며, 이 실행은 진단·재계획 경로가 도움이 된다는 증거도, 해가 된다는 증거도 제공하지 않습니다. 아래는 진단이 아니라, 단일 시도에서 이미 측정되어 기록된 값들의 판독입니다.

- **비교 가능한 차이는 baseline과의 차이 하나뿐입니다.** 시도가 1건이므로 시도 간 비교는 없습니다. 카드 baseline(`logreg`)의 `balanced_accuracy` 0.662 (CI 0.6496~0.6765)와 iteration 1의 0.8705 (CI 0.8588~0.8808)는 구간이 겹치지 않으므로, 이 데이터가 분리해 주는 실질적 차이입니다.
- **이 차이는 두 축에 나뉘어 있습니다.** 랭킹 축에서 `roc_auc`는 0.9101 → 0.9387로 올랐고, 운영점 축에서는 baseline의 `balanced_accuracy_at_best_cut`이 이미 0.84였다는 점이 결정적입니다. 즉 baseline은 랭킹으로는 0.84에 도달할 수 있었지만 기본 `predict()` 컷이 recall 0.3478에 머물러 0.662밖에 내지 못했습니다. iteration 1은 `class_weight='balanced'`로 컷을 옮겨 recall 0.8573 / specificity 0.8838의 균형을 만들었고, 동시에 랭킹 상한 자체를 `balanced_accuracy_at_best_cut` 0.8812로 끌어올렸습니다. 두 축 모두 움직였다는 것까지가 이 데이터가 말해 주는 범위이고, 각 축의 기여를 정량 배분할 추가 측정은 이 실행에 없습니다.
- **남은 여지는 랭킹 쪽에 있습니다.** `cut_headroom`=0.0107은 매우 작습니다. 기본 컷은 이미 이 랭킹의 최적점(0.8812) 근처에 있으므로, 여기서 `class_weight`를 'balanced'보다 무겁게 밀어 얻을 수 있는 최대치는 0.011 남짓이며, 그 이상은 모델 family나 특징(feature) 쪽 문제입니다.
- **확률값은 그대로 읽기 어렵습니다.** `brier`=0.0845, `calibration_error`=0.0935 — 예측 확률이 평균 약 9%p 어긋나 있습니다. 랭킹 지표(`roc_auc`, `pr_auc`)와 목표 지표는 이 영향을 받지 않지만, 이 모델의 확률을 위험 점수로 그대로 쓰려는 용도라면 별개의 문제입니다. 이 실행 환경에는 재보정 수단이 없습니다.
- **과적합은 심하지 않습니다.** `train_val_gap`=0.0775 (train 0.9480 vs val 0.8705)로, `max_iter=400` + `early_stopping=False` 조합에서 예상 범위이며 검증 CI 폭(약 0.022)에 비해 크긴 하지만 붕괴 수준은 아닙니다.
- **`unsupported_claims: ["feature_engineering"]`에 대하여:** iteration 1의 계획 문장은 "the same 16-column encoded matrix", "No imputation needed (no true NaNs) and no scaling for a tree family"라고 적고 있을 뿐, 파생 특징을 만들어 성능을 얻겠다고 의존하지 않았습니다. 이 플래그는 계획 산문에 대한 substring 검사가 걸린 것으로 보이며, **이 시도가 무엇을 잃었다고 볼 근거는 없습니다**. 특징 조합·재인코딩은 executor가 할 수 없는 일이므로, 실패 원인이 아니라 다음 단계에서 파이프라인 바깥에 만들어야 할 항목입니다.

## 다음 단계 제안

1. **테스트 점수를 기준선으로 고정하고, 개선은 랭킹 축에서만 노려라.** 이 실행이 입증한 값은 검증 0.8705가 아니라 테스트 0.8554 (CI 0.8434~0.8667)입니다. `cut_headroom`=0.0107이므로 `class_weight`를 'balanced'보다 무겁게 하는 명시적 weight map(예: `{"0": 1, "1": 12}`)에 남은 여지는 0.011 미만이고, 그 크기는 검증 CI 폭(약 0.022)보다 작아 이 split에서는 개선으로 보고할 수 없습니다. 예산을 쓸 곳은 랭킹 축, 즉 `roc_auc` / `balanced_accuracy_at_best_cut`을 올리는 방향입니다. 다만 문서에 기록된 측정에서 family 교체는 0.0022~0.0077, 동일 family 내 하이퍼파라미터 재튜닝은 0.0032 폭이며 roc_auc 차이의 분해 해상도는 0.003~0.006이므로, 한 번의 시도로 분리 가능한 차이를 얻기는 어렵다는 것을 전제로 계획하십시오.
2. **`V16`·`V9`의 'unknown'을 원본 파일 수준에서 정리한 뒤 재측정하라 — 이것이 caveat의 자체 해결책이다.** `V16`은 81.8%, `V9`는 28.8%가 'unknown'이며, 카드는 이 값들이 변환되지 않은 채 실측치로 집계·채점되었다고 명시합니다. 현재 실행에서는 이 열들이 그대로 one-hot으로 들어갔고, executor는 열을 드롭하거나 재인코딩할 수 없습니다(`missing_indicator`는 전체 열 일괄 적용이며, 'unknown'은 NaN이 아니어서 애초에 걸리지도 않습니다). 따라서 "V16을 결측 처리해 보자"는 플랜은 이 executor에서 실행 불가입니다. 해야 할 일은 데이터 준비 단계에서 'unknown'을 실제 NaN으로 변환해 카드를 다시 생성하는 것이고, 그것이 열리면 `impute: none`(hist_gbdt가 NaN을 직접 분기)과 `missing_count` 같은 결측 레버를 비로소 의미 있게 평가할 수 있습니다. 지금 상태에서 결측 레버를 제안하는 것은 근거가 없습니다.
3. **결측 레버의 크기를 이 데이터셋의 숫자로 인용하지 말고, 검사부터 하라.** 실행 문서의 결측 관련 수치(0.0097 / 0.0065 / 0.0151 등)는 다른 임상 샘플에서 나온 값이고, contiguous split에서 전부 0 근처로 붕괴했습니다. 이 bank-marketing 파일에는 진짜 NaN이 0%이므로 그 숫자들은 예산 근거가 되지 않습니다. 2번에서 'unknown'을 NaN으로 바꾼 뒤에는, 그 결측이 응답자의 상태를 나타내는지 아니면 수집 시점·경로(recording regime)를 나타내는지를 먼저 확인하십시오. 후자라면 stratified random split은 그것을 이득으로 잘못 채점합니다.
4. **확률 보정이 필요한 용도라면 이 루프 밖에서 처리하라.** `calibration_error`=0.0935, `brier`=0.0845 — 랭킹은 좋지만 확률은 평균 약 9%p 어긋납니다. 이 실행 환경에는 재보정 수단도 임계값 탐색 수단도 없으므로, 확률값을 그대로 쓰는 다운스트림(예: 기대수익 기준 컷)이 있다면 별도 파이프라인에서 보정하고, 그 이후 컷을 최적화하면 `balanced_accuracy_at_best_cut`=0.8812까지의 0.011은 회수 가능합니다. 목표 지표만 본다면 우선순위는 낮습니다.