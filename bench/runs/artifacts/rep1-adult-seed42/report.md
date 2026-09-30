## 요약

목표는 달성되었습니다. 1회 시도만으로 `hist_gbdt`가 검증 `balanced_accuracy` **0.8427** (95% CI 0.8338~0.8508)을 기록해 임계값 0.8248을 넘었고, 로지스틱 회귀 baseline 0.7664(CI 0.7575~0.7762)를 크게 상회했습니다. 한 번도 학습·선택에 쓰이지 않은 최종 테스트 20%에서도 **0.8397** (95% CI 0.8298~0.8478)로, 이 실행이 실제로 입증한 성능은 이 숫자입니다. 예산 5회 중 1회만 사용하고 `goal_reached`로 종료되었으므로 critic 진단은 한 번도 실행되지 않았습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `learning_rate=0.06`, `max_iter=600`, `max_leaf_nodes=63`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=true`, `validation_fraction=0.1`, `n_iter_no_change=30`, `class_weight={"0":1.0,"1":2.6}` / `impute: none`, `scale: false` | `status: ok` — `balanced_accuracy` 0.8427 (CI 0.8338~0.8508), `roc_auc` 0.9293, `pr_auc` 0.8301, `recall` 0.8310 / `specificity` 0.8544, `train_val_gap` 0.0383, 학습 8.738초, `early_stopping`이 160/600 iteration에서 정지 | 없음 (목표 달성으로 루프 종료 — critic 미실행) |

## 최고 성능 구성

- **model**: `hist_gbdt`
- **hyperparams** (해당 attempt에 적용된 값, `dropped_hyperparams`는 비어 있음):
  - `learning_rate: 0.06`
  - `max_iter: 600`
  - `max_leaf_nodes: 63`
  - `min_samples_leaf: 20`
  - `l2_regularization: 1.0`
  - `early_stopping: true`, `validation_fraction: 0.1`, `n_iter_no_change: 30`
  - `class_weight: {"0": 1.0, "1": 2.6}`
- **preprocessing** (해당 attempt에 적용된 값): `impute: none`, `scale: false`, `missing_indicator: false`, `missing_count: false` — 즉 결측 대치를 하지 않고 트리가 NaN을 직접 분기했으며, 표준화도 하지 않았습니다(카드의 `impute: median` / `scale: true`는 요청값일 뿐 이 실행에 적용되지 않았습니다).
- **internal_validation**: `fit_rows: 26373`, `held_out_rows: 2931`, `stopped_at_iter: 160`
- **프로토콜**: stratified 3분할, seed 42, train 60% / validation 20% / test 20%

| 지표 | validation | 최종 held-back test |
|---|---|---|
| `balanced_accuracy` | **0.8427** (CI 0.8338~0.8508) | **0.8397** (CI 0.8298~0.8478) |
| `roc_auc` | 0.9293 | — |
| `pr_auc` / `average_precision` | 0.8301 | — |
| `f1` | 0.7245 | — |
| `accuracy` | 0.8488 | — |
| `precision` / `recall` | 0.6422 / 0.8310 | — |
| `specificity` | 0.8544 | — |
| `balanced_accuracy_at_best_cut` / `cut_headroom` | 0.8502 / 0.0075 | — |
| `brier` / `calibration_error` | 0.0999 / 0.0761 | — |
| `train_balanced_accuracy` / `train_val_gap` | 0.8810 / 0.0383 | — |

검증 0.8427과 테스트 0.8397의 차이 **+0.0030**이 선택 편향의 크기입니다 — 루프는 검증 숫자만 보고 최고 attempt를 골랐기 때문입니다. 다만 검증 점수가 테스트 CI(0.8298~0.8478) 안에 들어 있어, 이 행들만으로는 그 차이를 0과 구분할 수 없습니다. 두 값 모두 임계값 0.8248을 상회합니다.

## 원인 분석

이 실행에서는 **critic이 한 번도 실행되지 않았습니다** (시도 1회, 진단 0회). 첫 계획이 곧바로 임계값을 넘어 루프가 종료되었으므로, 여러 attempt 사이의 패턴이나 진단 근거로 설명할 것이 존재하지 않습니다. 위 점수는 전적으로 **첫 계획 하나가 낸 결과**이며, 이 실행은 진단·재계획 경로가 도움이 되는지 해가 되는지에 대해 어느 방향으로도 증거를 제공하지 않습니다. 어떤 지표가 성능을 "제한했다"는 서술은 진단이 아니라 사후 추측이 되므로, 여기서는 하지 않습니다.

관측된 사실만 기록하면 다음과 같습니다.

- 성능 향상은 baseline 대비 **랭킹 축과 운영점 축 양쪽에서** 관찰됩니다: `roc_auc`가 0.9075 → 0.9293, `balanced_accuracy_at_best_cut`이 0.8238 → 0.8502로 올랐고(랭킹), 동시에 `class_weight={"0":1.0,"1":2.6}`로 기본 `predict()` 컷이 그 최적점 근처(`cut_headroom` 0.0075)에 놓였습니다(운영점). baseline은 `recall` 0.6008에서 컷이 크게 어긋나 있었습니다.
- `recall` 0.8310과 `specificity` 0.8544가 거의 균형을 이루고 있어, `balanced_accuracy` 관점에서 운영점에 남은 여지는 0.0075뿐입니다.
- attempt가 1건이므로 CI가 겹치는 attempt 간 비교로 서술할 움직임 자체가 없습니다. baseline과의 차이(0.7664 CI 0.7575~0.7762 대 0.8427 CI 0.8338~0.8508)는 구간이 겹치지 않아 이 데이터로 구분되는 차이입니다.
- `plan.unsupported_claims`에 `feature_engineering`이 기록되어 있으나, iteration 1의 계획 본문은 모델 교체(`hist_gbdt`), `impute: none`, `class_weight` 조정만 서술하고 파생 피처를 전제하지 않습니다. 즉 substring 검사에 의한 오탐으로 보이며, 이 attempt가 그로 인해 잃은 것은 없습니다. 피처 생성은 "실패한 것"이 아니라 "아직 만들지 않은 도구"로서 다음 단계에 속합니다.

## 다음 단계 제안

1. **남은 예산으로 랭킹 축을 더 검증한다.** `cut_headroom`이 0.0075에 불과하므로 `class_weight` 재조정으로 얻을 수 있는 여지는 거의 없고, 남은 개선은 `roc_auc`/`balanced_accuracy_at_best_cut`(현재 0.9293 / 0.8502) 쪽에 있습니다. 같은 파이프라인(`impute: none`, `scale: false`)을 유지한 채 `xgboost`로 계열을 바꿔 1회, 그리고 `hist_gbdt` 내부에서 `learning_rate`/`max_leaf_nodes`/`min_samples_leaf`를 조정해 1회 시도하는 것이 자연스럽습니다. 다만 계열 교체가 주는 크기는 관측 사례에서 0.0022~0.0077 범위로 폭이 넓고, 동일 계열 재튜닝은 0.0032 정도였으며 이 차이들의 해상도는 0.003~0.006 수준이므로, 개선이 CI 폭보다 작으면 개선으로 보고하지 말아야 합니다.
2. **`missing_indicator`/`missing_count`는 지금 구성에 추가하지 않는다.** `impute: none`으로 트리가 NaN을 직접 분기하는 조건에서 `missing_indicator`는 중복이며(비트 단위로 동일한 예측이 재현됨), `missing_count`도 단독 이득이 관측되지 않았습니다. 결측은 `workclass` 0.0573, `occupation` 0.0575, `native-country` 0.0175로 전체 0.0095 수준이고, 이 데이터셋에는 결측 발생 원인에 관한 caveat이 기록되어 있지 않으므로(caveat 없음) 이 레버에 iteration을 쓸 근거가 없습니다.
3. **확률값을 쓸 계획이라면 캘리브레이션을 별도로 다룬다.** `brier` 0.0999, `calibration_error` 0.0761 — 예측 확률이 평균 약 7.6%p 어긋나 있습니다. 이 실행 환경에는 재캘리브레이션 레버가 없으므로(`CalibratedClassifierCV`나 컷 이동 불가), 순위(`roc_auc` 0.9293)만 필요한 용도라면 문제가 없지만 확률 자체를 의사결정에 쓸 경우 파이프라인 밖에서 후처리를 설계해야 합니다.
4. **피처 파생이 필요하다면 executor 밖에서 준비한다.** 현재 executor는 컬럼 조합·재인코딩·삭제를 하지 못하므로(`education`과 `education-num`의 중복, `capital-gain`/`capital-loss`의 높은 skew 같은 것을 다룰 방법이 없음), 파생 피처를 시험하려면 데이터 카드 단계에서 컬럼을 만들어 넣어야 합니다. 이는 위 1번의 랭킹 축 개선 여지가 소진된 뒤에 착수할 항목입니다.

또한 이번 실행은 첫 시도에서 종료되어 진단·재계획 경로가 검증되지 않았습니다. 그 경로의 유효성을 알고 싶다면, 임계값을 더 높게 잡거나 최소 시도 횟수를 지정해 critic이 실제로 작동하는 실행을 별도로 한 번 돌리는 것이 필요합니다.