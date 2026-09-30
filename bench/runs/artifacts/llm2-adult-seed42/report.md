## 요약

목표는 달성되었습니다. `balanced_accuracy` 임계값 0.8248에 대해, 첫 번째 시도인 `hist_gbdt`가 검증 슬라이스에서 0.8460 (95% CI 0.8377~0.8530)을 기록했고, 한 번도 사용되지 않은 홀드백 테스트 20%에서 0.8433 (95% CI 0.8347~0.8516)으로 재확인되었습니다. 사용된 반복은 5회 중 1회이며, 목표 달성으로 루프가 즉시 종료되었습니다. 카드의 baseline(`logreg`, 0.7664, CI 0.7575~0.7762)과 비교하면 두 구간이 겹치지 않으므로 이 개선은 이 데이터가 실제로 구분해 주는 크기입니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `class_weight='balanced'`, `early_stopping=true`, `validation_fraction=0.1` / `impute: none`, `scale: false` | `status: ok` — `balanced_accuracy=0.8460` (CI 0.8377~0.8530), 임계값 0.8248 초과. 내부 조기 종료는 131 iter에서 멈춤 (fit_rows 26373, held_out_rows 2931). 학습 시간 6.973초 | 없음 (`critic: null` — 목표 달성으로 진단이 실행되지 않음) |

## 최고 성능 구성

iteration 1, `hist_gbdt`. 아래 값은 히스토리에 기록된 **적용된(applied)** 하이퍼파라미터와 전처리이며, `dropped_hyperparams`는 비어 있습니다.

- `hyperparams`: `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `class_weight='balanced'`, `early_stopping=true`, `validation_fraction=0.1`
- `preprocessing`: `impute: none`, `scale: false`, `missing_indicator: false`, `missing_count: false` — 즉 결측치는 대치되지 않고 모델이 NaN 분기를 직접 학습했습니다
- `internal_validation`: `fit_rows=26373`, `held_out_rows=2931`, `stopped_at_iter=131` (400 상한에 도달하지 않음)
- 분할/시드: 고정 프로토콜 stratified 60/20/20, seed 42 (카드 baseline과 동일 분할)

검증 슬라이스 지표:

| 지표 | 값 |
|---|---|
| `balanced_accuracy` | **0.8460** (95% CI 0.8377~0.8530) |
| `roc_auc` | 0.9296 |
| `pr_auc` / `average_precision` | 0.8291 |
| `accuracy` | 0.8371 |
| `f1` | 0.7172 |
| `precision` / `recall` | 0.6134 / 0.8631 |
| `specificity` | 0.8290 |
| `balanced_accuracy_at_best_cut` / `cut_headroom` | 0.8499 / 0.003872 |
| `brier` / `calibration_error` | 0.108302 / 0.105264 |
| `train_balanced_accuracy` / `train_val_gap` | 0.8606 / 0.014579 |

**홀드백 최종 측정**: 동일 모델을 첫 fit 이전에 떼어내어 어떤 결정에도 쓰이지 않은 테스트 20% 행에서 한 번 채점 → `balanced_accuracy=0.8433` (95% CI 0.8347~0.8516). 검증 0.8460 대비 −0.0027이고, 이 차이가 선택 편향의 크기입니다 — 루프의 선택은 모두 검증 숫자로 이루어졌습니다. 단 검증 점수가 테스트 CI 안에 들어 있어, 이 행 수로는 그 차이를 0과 구분할 수 없습니다. 이 실행이 실제로 입증한 값은 테스트 쪽 0.8433입니다.

## 원인 분석

이 실행에서 critic은 한 번도 실행되지 않았습니다(시도 1회, 진단 0회). 따라서 **설명할 진단 패턴이 존재하지 않습니다**. 0.8460이라는 점수는 첫 계획 하나가 낸 결과이고, 진단·재계획 경로는 이 결과에 전혀 기여하지 않았습니다 — 그 경로가 도움이 된다는 증거도, 해가 된다는 증거도 이 실행에는 없습니다. 시도가 하나뿐이므로 시도 간 이동(레버 비교)에서 원인을 추론할 근거도 없습니다.

계획 텍스트에 대한 `unsupported_claims: ["feature_engineering"]`는 산문에 대한 부분 문자열 검사입니다. iteration 1의 `plan_strategy`를 읽으면 실제로는 "one-hot encoded adult matrix" 위에 그대로 fit한다고만 서술하고 있고, 새로운 파생 변수나 상호작용에 의존하는 부분이 없습니다. 즉 이 시도가 feature engineering 때문에 무언가를 잃은 것이 아니며, 0.8460은 이 executor의 능력 범위에서 나온 값 그대로입니다. 파생 변수는 `## 다음 단계 제안`에서 "만들어야 할 것"으로 다룹니다.

진단 대신 남는 것은 이 한 시도가 스스로 보고한 계측값이며, 이는 원인 규명이 아니라 다음 반복의 방향 표지로만 읽어야 합니다. `cut_headroom=0.003872`는 기본 0.5 컷이 이미 이 랭킹에서 얻을 수 있는 최적점(`balanced_accuracy_at_best_cut=0.8499`)에 거의 붙어 있다는 뜻이므로, 남은 여지는 operating point 축이 아니라 랭킹 축(모델 계열·특징)에 있습니다. `train_val_gap=0.014579`이고 조기 종료가 131 iter에서 걸렸다는 점은 용량 과다로 인한 과적합이 병목이 아니라는 신호입니다. 반면 `calibration_error=0.105264`, `brier=0.108302`은 확률값이 평균 약 10.5%p 어긋나 있음을 보여주는데, 이는 `class_weight='balanced'`로 작동점을 옮긴 모델에서 예상되는 부작용이며 `balanced_accuracy` 자체에는 벌점이 없습니다.

## 다음 단계 제안

(이 데이터셋에는 별도로 기록된 caveat이 없으므로, 아래 제안을 무효화하는 열·값·분할 관련 경고는 없습니다.)

1. **비교 대상 없이 종료된 상태를 메우기 — 남은 반복으로 랭킹 축만 흔들어 볼 것.** 목표는 이미 넘겼지만 후보는 하나뿐이어서, 이 구성이 이 executor의 상한인지 알 수 없습니다. `xgboost`를 동일한 `impute: none` 파이프라인으로 한 번, `hist_gbdt`의 용량만 바꾼 구성(예: `learning_rate` 하향 + `max_leaf_nodes=63`)을 한 번 돌려 `roc_auc`(현재 0.9296)와 `balanced_accuracy_at_best_cut`(현재 0.8499)을 비교하십시오. 단 검증 CI 폭이 약 ±0.008이고 계열 교체·재튜닝의 알려진 크기는 roc_auc 기준 0.0022~0.0077 수준이므로, 구간이 겹치는 차이는 개선으로 보고하지 말아야 합니다.
2. **작동점(`class_weight`, 컷) 튜닝에는 반복을 쓰지 말 것.** `cut_headroom=0.003872`는 컷 조정으로 얻을 수 있는 최대치가 0.004 미만이라는 계측값이고, 이는 검증 CI 폭보다 작습니다. 같은 이유로 `missing_indicator`/`missing_count`도 배제합니다 — `impute: none`으로 NaN 분기를 이미 쓰는 계열에서 `missing_indicator`는 중복(예측이 비트 단위로 동일)이라는 것이 정리되어 있고, `missing_count`도 이 조건에서 기대할 근거가 없습니다.
3. **특징 쪽에서 더 얻으려면 executor 밖에서 데이터 카드를 바꿔야 함.** 파생·상호작용·열 삭제는 이 executor가 하지 않는 일이므로 계획으로는 해결되지 않습니다. 구체적으로 `fnlwgt`는 카드에서 `target_corr: none`이자 `distinct: high`인 표본 가중치성 열이고, `education`과 `education-num`은 사실상 같은 정보를 두 번 넣고 있습니다. 이런 열의 제거나 재인코딩은 카드/원본 파일 단계에서 처리한 뒤 다시 fit해야 하며, 그때 비로소 "특징이 병목인가"를 랭킹 축 지표로 검증할 수 있습니다.
4. **확률값을 그대로 쓸 계획이라면 별도 처리 경로가 필요함.** `calibration_error=0.105264`, `brier=0.108302`로 확률은 약 10%p 어긋나 있고, 이 루프에는 recalibration 레버가 없습니다(`CalibratedClassifierCV`나 컷 이동은 실행되지 않음). 다운스트림이 확률을 요구한다면 보정은 루프 외부 단계로 설계하고, 이 루프의 산출물은 순위·결정 라벨 용도로만 사용하십시오. 참고로 `roc_auc`/`pr_auc`는 보정과 무관하므로 랭킹 비교(제안 1)는 이 문제와 별개로 유효합니다.