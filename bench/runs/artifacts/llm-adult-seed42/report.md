# AutoML 실행 최종 보고서 — `adult` / `balanced_accuracy`

## 요약

목표는 달성되었습니다. 1회차 시도에서 `hist_gbdt`가 검증 `balanced_accuracy` **0.8481** (95% CI 0.8392~0.8550)을 기록해 목표 임계값 0.8248을 넘었고, 루프는 5회 예산 중 **1회**만 사용하고 `goal_reached`로 종료되었습니다. 한 번도 사용되지 않은 최종 테스트 20%에서의 점수는 **0.8445** (95% CI 0.8360~0.8531)로, 검증 점수보다 0.0036 낮지만 이 역시 임계값을 상회합니다. 이번 목표는 baseline `logreg`의 랭킹 상한(`balanced_accuracy_at_best_cut` = 0.8238)을 이미 넘어서는 수준이어서 결정 임계값 조정만으로는 도달할 수 없었고, 실제로 랭킹 축(roc_auc 0.9075 → 0.9299)이 함께 움직인 것이 달성의 핵심입니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=300`, `learning_rate=0.08`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=25`, `class_weight='balanced'` / preprocessing `impute='none'`, `scale=False` | `status=ok` — `balanced_accuracy=0.8481` (CI 0.8392~0.8550), `roc_auc=0.9299`, `recall=0.8644`, `specificity=0.831943`, `cut_headroom=0.002651`, `train_val_gap=0.021667`, 학습 8.416초 | 없음 (`critic=null`) — 목표 달성으로 루프가 즉시 종료되어 진단이 생성되지 않음 |

## 최고 성능 구성

아래 값은 실행기가 실제로 만든 파이프라인에서 읽은 **적용값**(history의 `hyperparams` / `preprocessing`)입니다. `dropped_hyperparams`는 비어 있어, 제안된 설정이 모두 그대로 반영되었습니다.

- **model**: `hist_gbdt`
- **hyperparams**:
  - `max_iter=300`
  - `learning_rate=0.08`
  - `max_leaf_nodes=31`
  - `min_samples_leaf=20`
  - `l2_regularization=1.0`
  - `early_stopping=true`
  - `validation_fraction=0.1`
  - `n_iter_no_change=25`
  - `class_weight='balanced'`
- **preprocessing**: `impute='none'` (트리가 NaN을 직접 분기), `scale=false`, `missing_indicator=false`, `missing_count=false`
- **프로토콜**: stratified 3-way split, seed 42, train 60% / validation 20% / test 20% (카드 baseline과 동일 분할)
- **학습 시간**: 8.416초 (제한 600초)

| 지표 | 검증(validation, 20%) | 최종 테스트(held-back, 20%) |
|---|---|---|
| `balanced_accuracy` | **0.8481** (CI 0.8392~0.8550) | **0.8445** (CI 0.8360~0.8531) |
| `roc_auc` | 0.9299 | — |
| `pr_auc` / `average_precision` | 0.8306 | — |
| `f1` | 0.7207 | — |
| `accuracy` | 0.8397 | — |
| `precision` / `recall` | 0.6179 / 0.8644 | — |
| `specificity` | 0.831943 | — |
| `balanced_accuracy_at_best_cut` / `cut_headroom` | 0.8508 / 0.002651 | — |
| `brier` / `calibration_error` | 0.107117 / 0.100564 | — |
| `train_balanced_accuracy` / `train_val_gap` | 0.8698 / 0.021667 | — |

검증 0.8481과 테스트 0.8445의 차이 **0.0036**이 이번 실행의 선택 편향 크기입니다. 루프는 전적으로 검증 숫자만 보고 선택했으므로, 이 모델에 대해 실제로 입증된 값은 테스트 쪽 0.8445입니다. 다만 검증 점수는 테스트 CI(0.8360~0.8531) 안에 들어 있어, 이 행들만으로는 그 차이를 0과 구분할 수 없습니다. 어느 쪽으로 읽어도 임계값 0.8248은 테스트 CI 하한(0.8360)보다 낮아, 달성 판정은 재표본 노이즈에 뒤집히지 않습니다.

## 원인 분석

- **시도가 1회뿐이므로 "시도 간 변화"에 대한 서사는 없습니다.** 비교 대상은 카드의 baseline `logreg` 하나이며, 두 CI(baseline 0.7575~0.7762 / 최고 0.8392~0.8550)는 전혀 겹치지 않으므로 +0.0817의 `balanced_accuracy` 개선은 이 데이터에서 분명하게 구분되는 차이입니다.
- **이번 목표는 애초에 운영점(operating point) 조정만으로는 불가능한 목표였습니다.** goal 정보가 명시한 대로 임계값 0.8248은 baseline의 랭킹 상한 `balanced_accuracy_at_best_cut=0.8238`을 넘고(`exceeds_ranking_ceiling: true`), 필요 `ks`는 0.6496인데 baseline은 0.6476으로 0.002 부족했습니다. 즉 `logreg`의 확률 순위를 어디서 잘라도 목표에 닿지 못하는 상태였습니다.
- **실제로 움직인 것은 랭킹 축과 운영점 축 둘 다이며, 목표를 넘긴 결정적 요인은 랭킹 축입니다.** roc_auc 0.9075 → 0.9299, pr_auc 0.7706 → 0.8306, 그리고 랭킹 상한 자체가 0.8238 → 0.8508로 올라갔습니다. 상한이 임계값 위로 올라간 뒤에야 운영점이 의미를 갖는데, `class_weight='balanced'`가 baseline의 recall 0.6008을 0.8644로 끌어올려(specificity 0.831943) 기본 `predict()` 컷을 거의 최적점에 놓았습니다. 그 증거가 `cut_headroom=0.002651`입니다 — 임계값 탐색이 가능했더라도 추가로 얻을 수 있던 폭이 0.0027에 불과합니다.
- **과적합은 성능 제약 요인이 아니었습니다.** `train_val_gap=0.021667`, `train_balanced_accuracy=0.8698` 대 검증 0.8481로 격차가 작고, `early_stopping=True`가 `max_iter=300` 안에서 반복 수를 스스로 제한해 8.4초에 끝났습니다. 즉 용량(capacity)과 시간 예산 모두 여유가 남은 상태에서 목표에 도달했습니다.
- **남아 있는 유일한 명확한 약점은 확률 품질입니다.** `calibration_error=0.100564`, `brier=0.107117` — 예측 확률이 평균 약 10%p 어긋나 있습니다. 이는 `class_weight='balanced'`로 양성 쪽을 밀어 올린 결과와 일관되며, 순위 기반 지표(`roc_auc`, `pr_auc`)와 `balanced_accuracy`에는 영향을 주지 않지만 확률값을 그대로 쓰는 용도에는 부적합합니다. 이 실행기에는 재보정 레버가 없어 측정만 되고 교정되지는 않았습니다.
- **`unsupported_claims: ["feature_engineering"]`에 성능 손실을 귀속시키지 않습니다.** 해당 플래그는 계획 산문에 대한 부분 문자열 검사에서 걸린 것이고, 1회차 계획 본문을 읽으면 "education-num x marital-status x capital-gain" 은 *트리가 내부적으로 잡아내는 비선형 구조*를 설명한 대목이지, 파생 컬럼을 만들어 쓰겠다는 전제가 아닙니다. 실제 설정에도 파생 피처는 없고, 실행기가 만든 파이프라인은 계획대로였습니다. 따라서 이 실행은 "명시적 상호작용 피처가 도움이 되는가"를 **시험하지 않았을 뿐**이며, 그것은 아래 제안의 항목입니다.

## 다음 단계 제안

이 데이터셋에는 별도로 기록된 caveat이 없어(`(없음)`), 어떤 컬럼·값·분할도 신뢰 불가로 지목되지 않았습니다. 아래 제안은 모두 실행기가 실제로 지원하는 레버 안에 있습니다.

1. **추가 예산을 성능 향상에 쓰기 전에, 이미 달성된 상태임을 인정하고 목적을 바꿀 것.** 테스트 0.8445, CI 하한 0.8360 > 임계값 0.8248이므로 목표 관점에서 더 짜낼 이유는 약합니다. `cut_headroom=0.002651`은 운영점 축이 사실상 소진되었다는 뜻이므로, 명시적 `class_weight` 가중치 맵 탐색(예: `{"0":1,"1":3}` 계열)에 iteration을 쓰는 것은 최대 0.0027짜리 여지를 CI 폭 0.016 안에서 쫓는 일입니다 — 권하지 않습니다.
2. **그래도 점수를 더 올리려면 랭킹 축만 건드릴 것: `xgboost`로의 family 교체, 또는 동일 family에서 용량 확대(`max_leaf_nodes=63`, `learning_rate=0.05` + `max_iter` 상향).** `train_val_gap=0.021667`과 8.4초/600초 사용률이 용량·시간 여유를 보여 주므로 안전한 방향입니다. 단 기대치는 낮게 잡아야 합니다: 참고 측정에서 family 교체는 roc_auc 0.0022~0.0077, 동일 family 재튜닝은 0.0032 폭이었고 짝지은 roc_auc 차이의 분해능은 0.003~0.006이므로, 결과가 현재 CI(±0.008)와 겹치면 "개선"으로 보고해서는 안 됩니다. 판정 기준은 `balanced_accuracy`가 아니라 `roc_auc`와 `balanced_accuracy_at_best_cut`의 이동으로 두십시오.
3. **결측 관련 레버에는 iteration을 쓰지 말 것.** 현재 구성은 `impute='none'`이고, 이 조건에서 `missing_indicator`는 예측이 비트 단위로 동일한 중복 레버로 확인되어 있습니다. `missing_count`도 참고 측정에서 단독 0.0003 / 지시자 위에서 0.0000이었습니다. 결측률 자체도 낮습니다(전체 0.0095, 최악 컬럼 0.0575: `occupation` 0.0575, `workclass` 0.0573, `native-country` 0.0175).
4. **확률값을 쓰는 소비처가 있다면, 그것은 튜닝이 아니라 도구 확충 과제로 분리할 것.** `calibration_error=0.100564`는 현 실행기에서 측정만 가능하고 교정 레버가 없습니다. 재보정(`CalibratedClassifierCV` 등)과 임계값 탐색, 그리고 (2번에서도 언급한) 명시적 상호작용/파생 피처 생성은 모두 실행기 CANNOT 목록에 속하므로, 다음 라운드에 "해봤다"고 보고될 수 없는 항목입니다. 이 세 가지가 필요하다면 파이프라인 기능으로 먼저 구현해야 하며, 구현되면 (a) 확률 소비 용도의 신뢰성, (b) `cut_headroom`이 남는 구성에서의 운영점 최적화, (c) 명시적 상호작용이 `hist_gbdt`의 내부 분기 대비 추가 가치를 갖는지를 각각 검증할 수 있게 됩니다.