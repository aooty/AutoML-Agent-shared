# AutoML 최종 리포트 — bank-marketing (`Class`, binary_classification)

## 요약

목표는 달성되었습니다. 단 1회의 시도로 `hist_gbdt`가 validation `balanced_accuracy` **0.8772** (95% CI 0.8664~0.8865)를 기록해 목표 임계값 0.7465와 baseline(logreg, 0.662, CI 0.6496~0.6765)을 모두 크게 상회했습니다. 루프 종료 후 한 번도 사용되지 않은 held-back test 20%에서의 점수는 **0.8606** (95% CI 0.8490~0.8712)로, 이것이 이 실행이 실제로 입증한 수치입니다. 첫 계획이 곧바로 임계값을 넘겼기 때문에 critic(진단·재계획)은 한 번도 실행되지 않았습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `min_samples_leaf=40`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=30`, `class_weight={"0":1.0,"1":6.0}`, `random_state=42` | status `ok` — validation `balanced_accuracy`=0.8772 (CI 0.8664~0.8865), `roc_auc`=0.9415, `pr_auc`=0.6378, `recall`=0.8837, `specificity`=0.870741, `train_val_gap`=0.029344, 학습 4.929초, 내부 early stopping은 163/400에서 정지 | 없음 (목표 달성으로 루프 종료, critic 미실행) |

## 최고 성능 구성

`iteration 1`, model `hist_gbdt`. 아래는 히스토리에 기록된 **적용된** 값입니다.

- `hyperparams`:
  - `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `min_samples_leaf=40`
  - `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=30`
  - `class_weight={"0": 1.0, "1": 6.0}`, `random_state=42`
  - `dropped_hyperparams`: 없음 (모든 키가 estimator에 그대로 전달됨)
- `preprocessing` (적용값): `impute=median`, `scale=false`, `missing_indicator=false`, `missing_count=false`
- `internal_validation`: `fit_rows=24413`, `held_out_rows=2713`, `validation_fraction=0.1`, `stopped_at_iter=163` (`max_iter=400`)
- 프로토콜: stratified 3-way split, seed 42, train 60% / validation 20% / test 20% (카드 baseline과 동일 split)

측정값:

| metric | validation | held-back test |
|---|---|---|
| `balanced_accuracy` | **0.8772** (CI 0.8664~0.8865) | **0.8606** (CI 0.8490~0.8712) |
| `roc_auc` | 0.9415 | — |
| `pr_auc` / `average_precision` | 0.6378 | — |
| `recall` | 0.8837 | — |
| `specificity` | 0.870741 | — |
| `precision` | 0.4753 | — |
| `f1` | 0.6182 | — |
| `accuracy` | 0.8723 | — |
| `balanced_accuracy_at_best_cut` | 0.8835 (`cut_headroom`=0.006258) | — |
| `brier` / `calibration_error` | 0.088987 / 0.113344 | — |
| `train_balanced_accuracy` / `train_val_gap` | 0.9066 / 0.029344 | — |

validation 0.8772와 test 0.8606의 차이 **+0.0167**은 선택 편향(selection effect)의 크기입니다. 루프는 validation 숫자를 보고 선택을 했으므로, 이 모델에 대해 보고할 수 있는 값은 test 0.8606입니다. 두 구간(validation CI 0.8664~0.8865, test CI 0.8490~0.8712)은 겹치지 않으며, 두 수치 모두 임계값 0.7465를 여유 있게 넘습니다.

## 원인 분석

- **critic 진단은 0건입니다.** 첫 시도가 임계값을 넘겨 루프가 즉시 종료되었기 때문에, 이 점수는 오직 첫 계획 하나가 낸 결과입니다. 따라서 진단·재계획 경로에 대해 이 실행은 **아무런 증거도 제공하지 않습니다** — 도움이 되었다는 증거도, 해가 되었다는 증거도 없습니다. 시도가 1회뿐이므로 "시도 간 변화"라는 패턴 자체가 존재하지 않고, 여기서 성능 제약 요인을 진단으로 제시할 근거도 없습니다.
- 확인 가능한 것은 baseline과의 비교뿐입니다. logreg baseline은 `balanced_accuracy` 0.662 (CI 0.6496~0.6765), `roc_auc` 0.9101, `balanced_accuracy_at_best_cut` 0.84였고, 이번 `hist_gbdt`는 `roc_auc` 0.9415, `balanced_accuracy_at_best_cut` 0.8835입니다. 두 CI가 겹치지 않으므로 이 차이는 리샘플 노이즈로 설명되지 않습니다. 즉 개선은 **랭킹 축**(모델 패밀리 교체)과 **작동점 축**(`class_weight={"0":1,"1":6}`으로 recall 0.8837 ↔ specificity 0.8707을 맞춘 것) 양쪽에서 함께 왔습니다 — 다만 시도가 1회이므로 두 축의 기여를 이 실행 안에서 분해할 수는 없습니다.
- `plan.unsupported_claims`에 `feature_engineering`이 기록되어 있으나, 해당 계획의 본문은 "fit … on the full one-hot + numeric matrix"로 주어진 행렬을 그대로 쓴다고 명시하고 있습니다. 계획이 파생 피처를 전제로 하지 않았으므로 이 시도가 그것 때문에 무언가를 잃었다고 볼 근거는 없습니다. 이는 계획 산문에 대한 substring 체크의 오탐으로 읽는 것이 타당하며, 피처 생성은 "실패한 것"이 아니라 "이 executor에 없는 기능"으로 다음 단계에 속합니다.

## 다음 단계 제안

1. **작동점 축에 예산을 더 쓰지 마십시오.** `cut_headroom`=0.006258은 기본 결정 규칙이 이미 이 랭킹에서 거의 최적의 위치에 있음을 뜻합니다(`balanced_accuracy` 0.8772 vs `balanced_accuracy_at_best_cut` 0.8835). `class_weight` 재조정으로 얻을 수 있는 최대치가 validation CI 폭(약 ±0.010)보다 작으므로, 개선을 시도한다면 **랭킹 축**(모델 패밀리·피처)만이 남은 여지입니다. 단, 문서화된 랭킹 축 이동 크기는 roc_auc 기준 0.0022~0.0077, 페어드 해상도는 0.003~0.006 수준이므로 `xgboost` 교체나 `hist_gbdt` 재튜닝이 유의한 차이로 확인될 가능성은 높지 않다는 점을 예산 산정에 반영해야 합니다.
2. **`unknown` 센티널을 데이터 소스에서 정리하십시오 (카드 caveat의 자체 해결책).** V16은 81.8%, V9는 28.8%, V4는 4.1%, V2는 0.6%가 `unknown`이며, 변환되지 않은 상태로 실제 측정값처럼 집계·학습되었습니다. 따라서 이 열들의 `target_corr`·중요도에 기대는 어떤 분석도 지금은 신뢰할 수 없고, executor는 열을 재인코딩·삭제할 수 없으므로 이 작업은 **원본 파일/카드 단계**에서 `unknown` → 결측으로 바꿔야 합니다. 그렇게 하면 `hist_gbdt`에서 `impute: none`(NaN 분기)을 쓰는 경로가 열립니다 — 이때 `missing_indicator`는 NaN 분기와 중복(비트 단위 동일 예측)이므로 함께 쓰지 마십시오.
3. **위 정리를 하기 전에, 결측이 "대상의 상태"인지 "기록 방식/시점"인지 먼저 확인하십시오.** V16처럼 81.8%가 `unknown`인 열은 결측 자체가 기록 레짐을 식별할 수 있고, stratified random split은 그것을 이득으로 계상해 버립니다. `unknown`의 발생 이유가 문서로 확인되지 않는 한 결측 관련 컬럼(`missing_indicator`, `missing_count`)의 이득을 성능 개선으로 보고하지 말아야 합니다. 참고로 문서화된 실험에서 `missing_count`는 random split에서 0.0003, indicator 위에서 0.0000이었습니다.
4. **확률값을 쓰려면 별도 캘리브레이션 단계가 필요합니다.** `roc_auc` 0.9415로 순위는 좋지만 `calibration_error`=0.113344, `brier`=0.088987이며, 이 executor에는 재캘리브레이션 레버가 없습니다(`CalibratedClassifierCV`·임계값 이동 모두 불가). 예측 확률을 의사결정 수치로 소비할 계획이라면 루프 밖 파이프라인에서 캘리브레이션을 붙이고 held-back test 위에서 다시 측정해야 합니다. 분류 라벨만 쓰는 용도라면 현 구성(test `balanced_accuracy` 0.8606)으로 그대로 진행 가능합니다.