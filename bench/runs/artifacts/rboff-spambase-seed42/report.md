# AutoML 실행 최종 보고서 — `spambase` (binary_classification, 목표 지표 `balanced_accuracy`)

## 요약

목표는 달성되었습니다: 첫 시도에서 `hist_gbdt`가 검증 `balanced_accuracy` **0.9493** (95% CI 0.9331~0.9647)을 기록해 임계값 0.9421과 baseline `logreg` 0.9228을 모두 넘었고, 루프는 1회 시도(할당 1회) 만에 종료되었습니다. 다만 한 번도 사용되지 않은 최종 테스트 20%에서의 점수는 **0.9385** (95% CI 0.9206~0.9534)로 검증 점수보다 0.0108 낮고, 임계값 0.9421 자체는 이 held-back 행에서는 넘지 못했습니다. 즉 "목표 달성"은 루프가 판단 근거로 삼은 검증 슬라이스의 결과이며, 이 실행이 실제로 입증한 값은 테스트 점수 0.9385입니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=500`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=15`, `l2_regularization=1.0`, `early_stopping=true`, `validation_fraction=0.1`, `n_iter_no_change=30` (내부 조기 종료: 123 iter에서 정지, fit 2484행 / held-out 276행) | status `ok`, 검증 `balanced_accuracy=0.9493` (CI 0.9331~0.9647), `roc_auc=0.9890`, `pr_auc=0.9846`, `train_val_gap=0.038018`, 학습 2.432초 → 목표 달성으로 루프 종료 | 없음 (`critic: null`) |

`dropped_hyperparams`는 비어 있어, 계획된 하이퍼파라미터가 모두 그대로 estimator에 적용되었습니다. `unsupported_claims`도 비어 있습니다.

## 최고 성능 구성

- **model**: `hist_gbdt` (iteration 1)
- **hyperparams** (해당 시도 히스토리의 적용값):
  - `max_iter=500`
  - `learning_rate=0.06`
  - `max_leaf_nodes=31`
  - `min_samples_leaf=15`
  - `l2_regularization=1.0`
  - `early_stopping=true`, `validation_fraction=0.1`, `n_iter_no_change=30`
  - `class_weight` 미지정 (기본 0.5 컷)
- **preprocessing** (해당 시도의 적용값): `impute="median"`, `scale=false`, `missing_indicator=false`, `missing_count=false`
- **프로토콜**: stratified 3-way split, seed 42, train 60% / validation 20% / test 20% (card의 baseline과 동일 split)

| 지표 | 검증(validation 20%) | 최종 테스트(held-back 20%) |
|---|---|---|
| `balanced_accuracy` | **0.9493** (95% CI 0.9331~0.9647) | **0.9385** (95% CI 0.9206~0.9534) |
| `roc_auc` | 0.9890 | — |
| `pr_auc` / `average_precision` | 0.9846 | — |
| `f1` | 0.9400 | — |
| `accuracy` | 0.9533 | — |
| `precision` / `recall` | 0.9493 / 0.9309 | — |
| `specificity` | 0.967742 | — |
| `balanced_accuracy_at_best_cut` / `cut_headroom` | 0.9524 / 0.003059 | — |
| `brier` / `calibration_error` | 0.03786 / 0.015495 | — |
| `train_balanced_accuracy` / `train_val_gap` | 0.9874 / 0.038018 | — |

검증 0.9493과 테스트 0.9385의 차이 **+0.0108**은 선택 편향의 크기입니다 — 루프는 검증 숫자만 보고 이 구성을 골랐습니다. 다만 검증 점수 0.9493은 테스트 CI(0.9206~0.9534) 안에 들어오므로, 이 행 수로는 그 차이를 0과 구분할 수 없습니다. 배포 판단은 검증 0.9493이 아니라 테스트 0.9385(그리고 그 CI가 임계값 0.9421을 걸치고 있다는 사실)를 근거로 해야 합니다.

## 원인 분석

`재계획` 기록에 따르면 critic 진단은 **0건**입니다. 첫 계획이 임계값을 바로 넘겨 루프가 종료되었기 때문에 critic은 한 번도 실행되지 않았고, 따라서 **설명할 수 있는 verdict 패턴이 존재하지 않습니다**. 이 점수는 전적으로 첫 계획 하나가 낸 결과이며, 이 실행은 진단·재계획 경로가 도움이 된다는 증거도, 해가 된다는 증거도 제공하지 않습니다.

시도가 1건뿐이므로 시도 간 비교로 성능 한계를 규명할 근거도 없습니다. 어떤 레버가 얼마를 벌었는지에 대한 진단은 이 실행에서 이루어지지 않았고, 아래는 진단이 아니라 단일 시도가 측정해 남긴 사실의 확인입니다.

- baseline `logreg`의 `balanced_accuracy` CI는 0.903~0.941, `hist_gbdt`의 검증 CI는 0.9331~0.9647로 **겹칩니다**. 즉 이 검증 슬라이스(920행 규모)만으로는 두 모델의 목표 지표 차이(0.9228 → 0.9493)를 "차이"로 단정할 수 없습니다. 다만 랭킹 축에서는 `roc_auc` 0.9772(CI 0.9675~0.9844) → 0.9890, `pr_auc` 0.9647(CI 0.9519~0.9762) → 0.9846으로 baseline CI 상단 밖으로 움직였고, `balanced_accuracy_at_best_cut`도 0.9415 → 0.9524로 올라, 트리 앙상블이 baseline의 랭킹 상한(0.9415, 임계값 아래) 문제를 실제로 해소했다는 점은 이 데이터에서 확인됩니다.
- 운영점(operating point) 쪽 여지는 거의 없습니다: `cut_headroom=0.003059`로 기본 0.5 컷이 이미 최적 컷에 근접해 있고 `recall=0.9309` / `specificity=0.967742`가 대칭에 가깝습니다. 남은 격차는 컷이 아니라 랭킹 쪽에 있습니다.
- 과적합 징후는 크지 않습니다: `train_balanced_accuracy=0.9874` 대비 검증 0.9493, `train_val_gap=0.038018`이며 내부 조기 종료가 500 중 123 iter에서 멈췄습니다.
- 확률값 자체도 그대로 읽을 만합니다: `brier=0.03786`, `calibration_error=0.015495`.

## 다음 단계 제안

Data caveats 섹션에 기록된 주의사항은 없으므로(결측률 0.0, 비수치 열 0개, 대상 결측 0행), 아래 제안은 데이터 신뢰성 문제가 아니라 측정 해상도와 남은 축(랭킹)에 초점을 둡니다.

1. **판정 근거를 검증에서 held-back으로 옮겨 재확인하라.** 테스트 `balanced_accuracy=0.9385`의 CI(0.9206~0.9534)가 임계값 0.9421을 걸치고 있어, "임계값 초과"는 아직 held-back 행에서 확정되지 않았습니다. 더 많은 행이나 다른 seed의 실행(현 실행에서는 split·seed가 설정으로 고정되어 있으므로 별도 실행이 필요)으로 같은 구성을 다시 채점해, 0.9421 초과가 재현되는지를 먼저 확인하는 것이 가장 값싼 다음 행동입니다.
2. **추가 예산은 랭킹 축(모델 패밀리·하이퍼파라미터)에만 쓰고, 사전에 해상도를 정해두라.** `cut_headroom=0.003059`이므로 `class_weight`/`scale_pos_weight` 같은 운영점 레버에 iteration을 쓰는 것은 회수 가능액이 0.003 수준입니다. 대신 동일 파이프라인에서 `xgboost`(`early_stopping_rounds` 사용 가능) 또는 `random_forest`로의 패밀리 교체와 `hist_gbdt` 내부 재튜닝(`learning_rate`, `max_leaf_nodes`, `min_samples_leaf`, `l2_regularization`)을 비교하십시오. 단, 검증 CI 폭이 약 ±0.016(0.9331~0.9647)이므로 그보다 작은 `balanced_accuracy` 변화는 개선으로 보고하지 말고, 판단은 `roc_auc`/`pr_auc`/`balanced_accuracy_at_best_cut`으로 하되 이조차 페어 해상도가 대략 0.003~0.006이라는 점을 전제로 두어야 합니다.
3. **결측 관련 레버는 건너뛰라.** `missing.overall_rate=0.0`, `columns_with_missing=0`이므로 `missing_indicator`는 상수 열, `missing_count`는 항등 0이 되어 아무 정보도 추가하지 않습니다. 같은 이유로 `impute`의 선택(현재 `median`)은 사실상 no-op이며, `impute: none`로 바꿔도 얻을 것이 없습니다. 여기에 iteration을 쓰지 마십시오.
4. **executor 밖에서 만들어야 하는 것: 임계값 선택과 다중 seed 집계.** 이 루프는 임계값 탐색, cross-validation/out-of-fold, 모델 결합, 파생 피처 생성을 하지 않습니다. 배포 시 `recall`/`specificity` 비대칭이 요구된다면 `balanced_accuracy_at_best_cut=0.9524`가 그 상한을 이미 알려주므로, 컷 선택은 루프 밖의 별도 절차로 구현해야 합니다. 또한 단일 20% 검증 슬라이스라는 구조가 이번 ±0.016 해상도의 원인이므로, 반복 split 기반 평가를 루프 외부에 갖추면 위 2번의 비교가 비로소 의미 있는 크기 판정이 됩니다.