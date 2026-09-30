# AutoML 실행 리포트 — `adult` (binary_classification, balanced_accuracy)

## 요약

목표를 달성했습니다. 목표 임계값은 `balanced_accuracy` 0.8274였고, 첫 번째 시도의 `hist_gbdt`가 검증 슬라이스에서 0.8426 (95% CI 0.8341~0.8505)을 기록해 임계값을 CI 하단까지 포함해 넘겼습니다. 반복은 5회 예산 중 1회만 사용했고, 루프는 첫 시도에서 목표 달성으로 종료되었습니다. 한 번도 사용되지 않은 최종 테스트 20%에서의 점수는 0.8435 (95% CI 0.8362~0.8526)로, 이것이 이 실행이 실제로 입증한 수치입니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `early_stopping=false`, `class_weight={"0":1.0,"1":3.0}` / preprocessing: `impute=none`, `scale=false`, `missing_indicator=false`, `missing_count=false` | status `ok` — `balanced_accuracy` **0.8426** (CI 0.8341~0.8505), `roc_auc` 0.9278, `recall` 0.8502 / `specificity` 0.8350, `train_time_sec` 11.716 → 임계값 0.8274 달성, 루프 종료 | 없음 (`critic: null`, 목표 달성으로 진단 미실행) |

## 최고 성능 구성

- **model**: `hist_gbdt`
- **hyperparams** (해당 시도의 적용값): `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `early_stopping=false`, `class_weight={"0": 1.0, "1": 3.0}`
- **preprocessing** (해당 시도의 적용값): `impute: none`, `scale: false`, `missing_indicator: false`, `missing_count: false` — 즉 결측은 대치하지 않고 트리가 NaN을 직접 분기했습니다 (데이터 카드의 `impute: median` / `scale: true`는 요청값일 뿐, 이 실행에서는 적용되지 않았습니다).
- **dropped_hyperparams**: 없음 (executor가 거부한 키 없음)
- **프로토콜**: stratified 3-way split, seed 43, train 60% / validation 20% / test 20%

| 지표 | 검증(20%) | 최종 테스트(20%, 미사용 행) |
|---|---|---|
| `balanced_accuracy` | **0.8426** (CI 0.8341~0.8505) | **0.8435** (CI 0.8362~0.8526) |

두 수치의 차이(요약된 값으로 검증 대비 −0.0008)가 이 실행의 **선택 편향 크기**입니다. 루프는 검증 숫자만 보고 선택했기 때문에, 검증 점수는 선택에 쓰인 값이고 테스트 점수는 선택에 전혀 관여하지 않은 값입니다. 다만 검증 점수가 테스트 CI 안에 들어 있으므로, 이 행들만으로는 그 차이를 0과 구분할 수 없습니다.

검증 슬라이스의 나머지 지표:

| 지표 | 값 |
|---|---|
| `accuracy` | 0.8386733544886887 |
| `f1` | 0.716036036036036 |
| `precision` / `recall` | 0.6184251478369125 / 0.8502353444587077 |
| `specificity` | 0.835038 |
| `roc_auc` | 0.9277500687448095 |
| `pr_auc` / `average_precision` | 0.8296808360008627 |
| `balanced_accuracy_at_best_cut` | 0.8441 |
| `cut_headroom` | 0.001463 |
| `brier` | 0.106031 |
| `calibration_error` | 0.093468 |
| `train_balanced_accuracy` / `train_val_gap` | 0.8884940164096943 / 0.045858 |
| `train_time_sec` | 11.716 |

## 원인 분석

**critic은 한 번도 실행되지 않았습니다 (시도 1회, 진단 0회).** 따라서 여러 verdict를 가로지르는 패턴은 존재하지 않으며, 이 점수는 전적으로 첫 계획 하나가 낸 결과입니다. 진단·재계획 경로가 도움이 되었다는 증거도, 해가 되었다는 증거도 이 실행에는 없습니다. 아래는 진단이 아니라, 목표 설정 문서(`reference`)와 baseline 카드가 이미 기록해 둔 수치와 이번 시도의 측정값을 나란히 놓은 관찰입니다.

- 목표 설정 단계에서 이미 기록된 사실: 임계값 0.8274는 baseline logreg의 **ranking ceiling 0.8266을 초과**(`exceeds_ranking_ceiling: true`)했습니다. 즉 baseline의 순위 자체로는 어떤 컷을 골라도 임계값에 도달할 수 없었고, 운영점(operating point)만 움직이는 레버로는 통과가 불가능한 목표였습니다.
- 이번 시도는 그 순위 축을 실제로 움직였습니다: `roc_auc` 0.908 → 0.9278, `pr_auc` 0.7723 → 0.8297, `balanced_accuracy_at_best_cut` 0.8266 → 0.8441. `balanced_accuracy`는 0.7698 → 0.8426으로, baseline CI(0.7586~0.7798)와 이번 시도 CI(0.8341~0.8505)는 겹치지 않습니다.
- 운영점 축은 이미 거의 소진되었습니다. `cut_headroom` 0.001463 — 즉 `class_weight={"0":1,"1":3}` 아래의 기본 `predict()` 컷이 이 순위가 허용하는 최적 컷과 0.0015 차이입니다. `recall` 0.8502 대 `specificity` 0.8350의 잔여 비대칭도 이 크기 안에 들어갑니다. 남은 개선 여지는 운영점이 아니라 순위(모델 패밀리·특징)에 있습니다.
- 시도가 1건뿐이므로 CI가 겹치는 시도 간 이동을 해석할 대상 자체가 없습니다. `train_val_gap` 0.045858은 과적합이 폭주한 수준은 아니지만, 이 한 점만으로 규제 방향을 진단할 근거도 없습니다.
- `plan.unsupported_claims`에 `feature_engineering`이 올라와 있으나, 해당 iteration의 plan 텍스트를 읽으면 `missing_indicator`/`missing_count`를 **쓰지 않는 이유를 명시적으로 적어 둔 것**이고(“no missing_indicator (proven bit-for-bit redundant under impute:none) and no missing_count”), 실행 불가능한 기능에 의존한 부분은 없습니다. 문자열 검사가 언급만으로 발화한 경우로, 이 시도가 그 때문에 무언가를 잃은 것은 아닙니다. 즉 0.8426/0.8435는 파생 특징 없이 이 executor가 낸 값 그대로입니다.
- 남은 불만족 지점은 확률 품질입니다: `calibration_error` 0.093468, `brier` 0.106031. 순위는 좋지만 확률은 평균 9.3%p 어긋나 있습니다(양성 가중치 3.0을 준 결과와 일관됩니다). 이는 goal metric에는 영향이 없지만, 확률값을 그대로 소비하는 용도에는 부적합합니다.

## 다음 단계 제안

데이터 카드에 별도로 기록된 caveat은 없으므로(“없음”), 아래 제안은 특정 컬럼·값·split의 신뢰성 문제에 걸리지 않습니다. 대신 executor의 실제 능력 경계와 이번 측정의 해상도를 기준으로 씁니다.

1. **운영점 레버(`class_weight`) 추가 튜닝은 하지 마십시오.** `cut_headroom`이 0.001463이므로, 가중치를 3.0에서 2.5나 3.5로 옮겨 얻을 수 있는 최대치는 0.0015 수준입니다. 이는 이번 검증 CI 폭(0.8341~0.8505, 약 0.016)의 1/10이어서, 개선이 있어도 이 데이터로는 측정되지 않습니다. 같은 이유로 `missing_indicator`도 `impute: none`과 함께 쓰면 예측이 비트 단위로 동일하다고 이미 확립되어 있으므로 반복을 쓸 가치가 없습니다.
2. **남은 예산은 순위 축에만 쓰고, 결과를 “개선”이라 부르기 전에 CI와 비교하십시오.** 후보는 (a) `xgboost`로 패밀리 교체(`impute: none` 유지, `scale_pos_weight`로 동일한 운영점 이동), (b) `hist_gbdt` 내부 재튜닝(`max_leaf_nodes`, `l2_regularization`, `learning_rate`×`max_iter` 트레이드오프)입니다. 다만 기록된 크기는 패밀리 교체 0.0022~0.0077 `roc_auc`, 동일 패밀리 내 재튜닝 0.0032 `roc_auc`이고 짝지은 해상도는 0.003~0.006이므로, 대부분의 결과는 이번 시도와 구분되지 않을 가능성이 큽니다. 목표는 이미 달성되었으니 이 작업의 목적은 “더 높은 숫자”가 아니라 “현재 구성이 국소 최적인지 확인”으로 두는 것이 정직합니다.
3. **`feature_engineering` 능력을 executor 쪽에 만드는 것을 별도 작업으로 올리십시오.** 이 카드에는 `capital-gain`/`capital-loss`가 skew high, `education`과 `education-num`이 사실상 같은 정보를 두 인코딩으로 담고 있으며, `hours-per-week`의 outlier_rate가 0.2763입니다. 파생·재인코딩·컬럼 제거는 현재 executor가 전면 금지하는 항목이므로 계획으로는 시도할 수 없습니다. 이는 “이번에 실패한 것”이 아니라 “아직 없는 레버”이며, 순위 축의 상한(현재 `balanced_accuracy_at_best_cut` 0.8441)을 올릴 수 있는 유일한 미탐색 경로입니다.
4. **확률값을 쓰는 소비처가 있다면 재보정 단계를 파이프라인 밖에 두십시오.** `calibration_error` 0.093468, `brier` 0.106031이며 executor에는 재보정 레버가 없습니다(`CalibratedClassifierCV`나 컷 이동은 계획으로 전달되지 않습니다). 순위 지표(`roc_auc` 0.9278)와 `balanced_accuracy`는 보정과 무관하므로 이번 결론은 유지되지만, 확률을 임계값 정책이나 기대값 계산에 쓰려면 별도 보정 없이 그대로 쓰면 안 됩니다.