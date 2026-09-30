# AutoML 실행 최종 보고서 — `adult` (binary_classification, balanced_accuracy)

## 요약

**목표는 달성하지 못했습니다.** 목표 임계값은 `balanced_accuracy` 0.8248이었으나, 단 1회의 시도에서 얻은 최고 검증 점수는 `hist_gbdt`의 0.8030964412527815 (95% CI 0.79534~0.812614)로, CI 상단조차 임계값에 도달하지 못했습니다. 반복 예산이 1회로 제한되어 있어 루프는 `max_iterations`로 종료되었고, 두 번째 시도나 critic 진단은 한 번도 실행되지 않았습니다. 다만 baseline(`logreg`, 0.7664, CI 0.7575~0.7762)보다는 명확히 개선되었으며, 한 번도 사용되지 않은 테스트 20%에서도 0.8008 (95% CI 0.7916~0.8105)을 기록했습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `learning_rate=0.06`, `max_iter=500`, `max_leaf_nodes=63`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=30` | `status=ok` / `balanced_accuracy=0.8031` (CI 0.79534~0.812614), `roc_auc=0.9298`, `train_val_gap=0.0265`, `stopped_at_iter=132` — 임계값 0.8248 미달 | 없음 (critic 미실행) |

## 최고 성능 구성

- **iteration**: 1
- **model**: `hist_gbdt`
- **hyperparams** (해당 시도에 적용된 값, `dropped_hyperparams`는 빈 목록 — 모든 키가 그대로 반영됨):
  - `learning_rate: 0.06`
  - `max_iter: 500`
  - `max_leaf_nodes: 63`
  - `min_samples_leaf: 20`
  - `l2_regularization: 1.0`
  - `early_stopping: true`
  - `validation_fraction: 0.1`
  - `n_iter_no_change: 30`
- **preprocessing** (해당 시도에 적용된 값): `impute: none`, `scale: false`, `missing_indicator: false`, `missing_count: false`
- **internal_validation**: `fit_rows=26373`, `held_out_rows=2931`, `stopped_at_iter=132` (`max_iter=500` 미소진)
- **train_time_sec**: 7.263

검증 지표 (validation 20%):

| metric | 값 |
|---|---|
| `balanced_accuracy` | **0.8030964412527815** (95% CI 0.79534~0.812614) |
| `accuracy` | 0.8760364418057119 |
| `f1` | 0.7190906982138715 |
| `precision` / `recall` | 0.7852077001013171 / 0.6632434745400085 |
| `specificity` | 0.942949 |
| `roc_auc` | 0.9297715058406604 |
| `pr_auc` / `average_precision` | 0.8313697643233551 |
| `balanced_accuracy_at_best_cut` | 0.8502 |
| `cut_headroom` | 0.047104 |
| `brier` / `calibration_error` | 0.087195 / 0.008111 |
| `train_balanced_accuracy` / `train_val_gap` | 0.8295947302765381 / 0.026498 |

**최종 held-back 측정**: 동일 모델을 첫 학습 이전에 분리해 두고 어떤 결정에도 쓰이지 않은 테스트 20% 행에서 한 번 채점한 결과 `balanced_accuracy=0.8008` (95% CI 0.7916~0.8105)입니다. 검증 점수 0.8031보다 0.0023 낮으며, **이 차이가 선택 편향의 크기**입니다 — 이 실행의 모든 선택은 검증 점수를 보고 이루어졌습니다. 다만 검증 점수가 테스트 CI(0.7916~0.8105) 안에 들어 있어, 이 행 수로는 그 차이를 0과 구분할 수 없습니다. 어느 쪽 숫자로 보아도 임계값 0.8248에는 미달입니다.

## 원인 분석

- **critic 진단은 0건입니다.** 시도 1회, 진단 0회이므로 verdict 사이의 패턴이라 할 것이 존재하지 않습니다. 위 점수는 **첫 계획 하나가 낸 결과**이며, 이 실행은 진단·재계획 경로가 도움이 된다는 증거도 해가 된다는 증거도 제공하지 않습니다. 종료 사유도 "목표 달성"이 아니라 `max_iterations`(반복 예산 1회 소진)였습니다.
- **시도가 1건이므로 시도 간 비교 자체가 없습니다.** baseline과의 비교만 가능하며, 그 차이는 구간보다 큽니다: `balanced_accuracy` 0.8031 (CI 0.79534~0.812614) 대 baseline 0.7664 (CI 0.7575~0.7762)로 구간이 겹치지 않고, `roc_auc`도 0.9298 대 0.9075입니다. 즉 선형 baseline에서 gradient-boosted tree로의 교체는 이 데이터에서 실제 개선으로 읽을 수 있습니다.
- 목표까지의 남은 거리는 검증 기준 0.8248 − 0.8031 = 0.0217입니다. 이 시도의 진단 지표는 `cut_headroom=0.047104`, `balanced_accuracy_at_best_cut=0.8502`, `recall=0.6632` 대 `specificity=0.942949`를 보고하고 있습니다. **이는 진단이 아니라 이 시도가 보고한 관측값입니다** — 어떤 critic도 이를 해석한 바 없으므로, 아래 `다음 단계 제안`에서 검증할 가설로만 취급합니다.
- `plan.unsupported_claims`에 `feature_engineering`이 기록되어 있으나, 해당 iteration의 plan 원문을 읽으면 계획은 파생 피처를 **의존하지 않았습니다**. plan은 오히려 "no indicator columns are added, since they are provably redundant under impute:none", "class_weight is not a hist_gbdt parameter" 처럼 실행기의 제약을 명시적으로 인지하며 서술했고, 실제로 `dropped_hyperparams`는 비어 있습니다. 따라서 이 플래그로 인해 시도가 무엇을 잃었다고 볼 근거는 없습니다 — 0.8031은 이 실행기·이 피처 집합에서 그 구성이 낸 그대로의 숫자입니다.
- 참고로 **class_weight / scale_pos_weight 계열의 운영점 레버는 이 실행에서 한 번도 사용되지 않았습니다**(계획이 hist_gbdt에서 지원되지 않는다고 판단해 설정하지 않음). 즉 두 축(랭킹 축과 운영점 축) 중 운영점 축은 미측정 상태로 남아 있습니다.

## 다음 단계 제안

기록된 `Data caveats`가 없으므로 특정 열·값·분할을 배제하는 제약은 없습니다. 우선순위 순으로:

1. **운영점 축을 한 번 실제로 움직여 보기 (최우선).** `cut_headroom=0.047104`는 남은 격차 0.0217보다 크고, `recall=0.6632` 대 `specificity=0.942949`의 비대칭은 기본 0.5 컷이 balanced_accuracy 최적점(`balanced_accuracy_at_best_cut=0.8502`)에서 떨어져 있다는 뜻입니다. 실행기는 추정기 자체의 `class_weight`를 통한 재조정을 지원하므로, 동일한 `hist_gbdt` 구성에 `class_weight={"0": 1, "1": 2}`, 이어서 `{"0": 1, "1": 3}` 정도의 **완만한** 명시적 가중치를 시도하십시오(`'balanced'`는 비율을 클래스 빈도 3.18에 고정하는 한 점일 뿐이며 최적점이라는 보장이 없습니다). 시도 후 `dropped_hyperparams`를 반드시 확인하십시오 — 이 sklearn 버전의 `HistGradientBoostingClassifier`가 `class_weight`를 받지 않으면 키가 드롭되어 기록되고, 그 경우는 아래 2번으로 넘어갑니다.
2. **`class_weight`가 드롭될 경우 `xgboost` + `scale_pos_weight`로 동일한 축을 공격.** `impute: none`(NaN 분기 유지), `scale: false`, `early_stopping_rounds`만 지정(실행기가 eval set을 스스로 분리해 `internal_validation`에 보고)하고 `scale_pos_weight`를 2~3 범위에서 조정하십시오. 이는 운영점 축을 확실히 움직일 수 있는 유일한 경로이며, 부수적으로 tree family 교체가 랭킹 축에서 얼마를 주는지도 함께 읽힙니다 — 단 과거 측정에서 family 교체의 `roc_auc` 이동폭은 0.0022~0.0077로 페어드 해상도(0.003~0.006)와 겹치므로, 랭킹 개선을 기대 근거로 쓰지는 마십시오.
3. **랭킹 축의 재튜닝에는 예산을 크게 배분하지 말 것.** `train_val_gap=0.026498`은 과적합이 심하지 않음을 시사하고, early stopping이 `stopped_at_iter=132`에서 멈춰 `max_iter=500`은 제약이 아니었으며, 같은 family 내부 하이퍼파라미터 재튜닝은 과거 측정에서 `roc_auc` 0.0032 폭에 그쳤습니다. 여유가 남으면 `learning_rate` 0.03 + `max_leaf_nodes` 31 / `min_samples_leaf` 50처럼 용량을 낮춘 1회 정도로 제한하십시오.
4. **`missing_indicator` / `missing_count`에는 반복을 쓰지 말 것.** `impute: none`으로 NaN을 직접 분기하는 family에서 `missing_indicator`는 예측값이 비트 단위로 동일하다는 점이 이미 확인되어 있어 명백히 중복입니다. `missing_count`도 별도 이득이 보고되지 않았습니다. 이 데이터의 결측은 `workclass`(0.0573), `occupation`(0.0575), `native-country`(0.0175) 세 열, 전체 비율 0.0095로 규모 자체가 작습니다.
5. **반복 예산을 1회 이상으로 올려 재실행.** 이번 실행은 critic이 한 번도 돌지 않았기 때문에 진단·재계획 경로에 대해 아무것도 말해주지 않습니다. 1번·2번 시도를 담을 수 있는 최소 3~4회 예산으로 재실행하면, 남은 격차가 운영점 축(현재 `cut_headroom` 0.047)에 있는지 랭킹 축에 있는지를 처음으로 데이터로 판별할 수 있습니다. 참고로 baseline의 랭킹 상한(`balanced_accuracy_at_best_cut=0.8238`)은 임계값 0.8248보다 낮았고, 이번 모델은 0.8502로 그 상한을 이미 넘겼습니다 — 즉 랭킹은 원리상 임계값을 지지할 수 있는 수준이며, 남은 문제는 기본 컷을 그 지점으로 옮길 수 있는지입니다.