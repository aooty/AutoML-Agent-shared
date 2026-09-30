# AutoML 실행 최종 보고서 — `adult` (binary_classification, balanced_accuracy)

## 요약

목표는 달성되었습니다. 목표선은 `balanced_accuracy` 0.8248이었고, 최고 구성(iteration 2, `hist_gbdt`)의 검증 점수는 **0.8456 (95% CI 0.8369~0.8531)** 으로 하한값조차 목표선을 0.0121 상회했습니다. 5회 예산 중 4회를 사용하고 연속 미개선(정체)으로 조기 종료되었으며, 실질적인 성능 도약은 첫 시도에서 baseline `logreg` (0.7664, CI 0.7575~0.7762)를 `hist_gbdt`로 교체한 순간에 이미 끝났습니다. 한 번도 사용되지 않은 홀드백 테스트 20%에서의 점수는 **0.8386 (95% CI 0.8294~0.8471)** 으로, 이 값이 이번 실행이 실제로 입증한 수치입니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=True`, `n_iter_no_change=25`, `class_weight={"0":1.0,"1":2.6}`, `impute='none'` | balanced_accuracy **0.8421** (CI 0.8338~0.8502), roc_auc 0.9295, at_best_cut 0.8521, cut_headroom 0.00995, train_val_gap 0.0326 | `hyperparam` — 목표선 통과, 과적합 아님. 남은 여지는 운영점 ≤0.010, 나머지는 랭킹 축. 용량을 키워 랭킹 상한을 밀어볼 것 |
| 2 | `hist_gbdt` | `max_iter=800`, `learning_rate=0.03`, `max_leaf_nodes=63`, `min_samples_leaf=40`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=40`, `class_weight={"0":1.0,"1":2.6}`, `impute='none'` | balanced_accuracy **0.8456** (CI 0.8369~0.8531) — 실행 최고, roc_auc 0.9289, at_best_cut 0.8500, cut_headroom 0.00438, train_val_gap 0.0293 | `hyperparam` — iteration 1 대비 paired Δ +0.0035 (CI −0.0007~+0.0073), 0과 구분 불가. roc_auc·pr_auc·at_best_cut은 오히려 소폭 하락 → 용량 증설로 랭킹 상한은 오르지 않았고, 기본 컷이 최적에 가까워진 효과. 다른 계열을 시도할 것 |
| 3 | `xgboost` | `n_estimators=600`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `min_child_weight=1`, `scale_pos_weight=2.75`, `impute='none'` | balanced_accuracy **0.8452** (CI 0.8373~0.8535), roc_auc 0.9304, pr_auc 0.8334, at_best_cut 0.8513, cut_headroom 0.00615, train_val_gap **0.0509** | `wrong_model_family` — iteration 2 대비 paired Δ −0.0005 (CI −0.0047~+0.0042). 세 시도 모두 서로 구분 불가. gbdt 계열 3개 점의 at_best_cut 폭 0.0021, roc_auc 폭 0.0015로 paired 해상도(0.003~0.006) 미만 → 구조가 다른 학습기를 시도할 것 |
| 4 | `random_forest` | `n_estimators=800`, `min_samples_leaf=2`, `class_weight={"0":1.0,"1":3.0}`, `n_jobs=-1`, `random_state=42`, `impute='median'` | balanced_accuracy **0.8350** (CI 0.8271~0.8433), roc_auc 0.9187, pr_auc 0.7967, at_best_cut 0.8375, cut_headroom 0.00253, train_val_gap **0.0644** | (critic 없음 — 정체로 루프 종료) |

## 최고 성능 구성

**iteration 2 / `hist_gbdt`** — 아래 값은 히스토리에 기록된 *적용된* `hyperparams`와 `preprocessing`(실제로 만들어진 파이프라인)입니다. `dropped_hyperparams`는 비어 있어, 제안한 설정이 모두 그대로 적용되었습니다.

```
model: hist_gbdt
hyperparams:
  max_iter: 800
  learning_rate: 0.03
  max_leaf_nodes: 63
  min_samples_leaf: 40
  l2_regularization: 1.0
  early_stopping: true
  validation_fraction: 0.1
  n_iter_no_change: 40
  class_weight: {"0": 1.0, "1": 2.6}
preprocessing:
  impute: none          # hist_gbdt가 NaN을 직접 분기
  scale: false
  missing_indicator: false
  missing_count: false
protocol: stratified 60/20/20, seed 42
train_time_sec: 12.848 (예산 600s)
```

| 지표 | 검증 (20%) | 홀드백 테스트 (20%) |
|---|---|---|
| **balanced_accuracy** | **0.8456** (95% CI 0.8369~0.8531) | **0.8386** (95% CI 0.8294~0.8471) |
| f1 | 0.7256 | — |
| accuracy | 0.8477 | — |
| precision / recall | 0.6376 / 0.8417 | — |
| specificity | 0.8496 | — |
| roc_auc / pr_auc | 0.9289 / 0.8292 | — |
| balanced_accuracy_at_best_cut / cut_headroom | 0.8500 / 0.00438 | — |
| brier / calibration_error | 0.1008 / 0.0794 | — |
| train_balanced_accuracy / train_val_gap | 0.8750 / 0.0293 | — |

검증 0.8456 대비 테스트 0.8386으로 **−0.0071** 차이가 있고, 이 차이가 선택 편향(4개 시도 중 검증 점수 최대값을 고른 효과)의 크기입니다. 루프의 모든 결정은 검증 숫자로 이뤄졌으므로, 모델의 성능으로 인용해야 하는 값은 테스트 0.8386입니다. 다만 검증 점수가 테스트 CI(0.8294~0.8471) 안에 들어오므로, 이 행들만으로는 그 차이를 0과 구분할 수 없습니다. 테스트 CI 하한 0.8294도 목표선 0.8248을 상회합니다.

## 원인 분석

**실제로 크기가 확인된 유일한 개선은 모델 계열 교체 하나입니다.** baseline `logreg` 0.7664 (CI 0.7575~0.7762) → iteration 1 `hist_gbdt` 0.8421 (CI 0.8338~0.8502)로, 두 구간이 전혀 겹치지 않고 폭보다 훨씬 큰 +0.0757입니다. 동시에 랭킹 축이 함께 올라갔습니다: roc_auc 0.9075 → 0.9295, `balanced_accuracy_at_best_cut` 0.8238 → 0.8521. 목표선 0.8248은 baseline의 랭킹 상한(0.8238)을 넘도록 설정된 값이었으므로, 이 계열 교체 없이는 어떤 컷 조정으로도 통과가 불가능했습니다.

**iteration 1·2·3은 이 데이터로 서로 구분되지 않습니다.** 세 CI(0.8338~0.8502, 0.8369~0.8531, 0.8373~0.8535)가 광범위하게 겹치고, critic이 계산한 paired Δ도 iteration 2 vs 1이 +0.0035 (CI −0.0007~+0.0073), iteration 3 vs 2가 −0.0005 (CI −0.0047~+0.0042)로 모두 0을 포함합니다. 따라서 "learning_rate를 낮추고 leaf를 늘려 0.0035를 얻었다"는 이야기는 성립하지 않으며, 그 변화의 실체는 랭킹 상한이 아니라 기본 컷의 위치였습니다(`at_best_cut` 0.8521 → 0.8500으로 오히려 하락, `cut_headroom` 0.00995 → 0.00438로 소진). gbdt 계열 3개 점을 다 합쳐 `roc_auc` 폭 0.0015, `at_best_cut` 폭 0.0021 — 둘 다 paired 해상도 0.003~0.006 미만이므로, **동일 계열 내부의 하이퍼파라미터 튜닝은 이 문제에서 측정 가능한 값을 만들지 못했습니다.**

**구분 가능한 유일한 후속 신호는 iteration 4의 랭킹 축 후퇴입니다.** `random_forest`의 balanced_accuracy 0.8350 자체는 CI(0.8271~0.8433)가 앞선 시도들과 겹쳐 목표 지표상으로는 단정할 수 없지만, 랭킹 지표의 낙폭은 해상도를 넘습니다: roc_auc 0.9187 (gbdt 0.9289~0.9304보다 0.010~0.012 낮음), pr_auc 0.7967 (0.8292~0.8334보다 0.033~0.037 낮음), `at_best_cut` 0.8375 (0.8500~0.8521보다 0.0125~0.0146 낮음). 즉 이 열들에서 gbdt의 랭킹 상한 ≈0.85는 우연이 아니고, 부스팅이 아닌 배깅 트리는 그보다 낮은 상한을 가집니다. 여기에는 `impute='median'`으로의 강제 변경이 함께 들어 있어(random_forest는 NaN 분기 불가) 두 개의 원인이 한 행에 섞여 있다는 점도 감안해야 합니다.

**성능을 제한한 것은 운영점이 아니라 랭킹 축입니다.** 최고 구성의 `cut_headroom`은 0.00438에 불과하고(iteration 4는 0.00253), recall 0.8417 / specificity 0.8496으로 두 값이 거의 교차 지점에 있습니다. 즉 `class_weight`를 어떻게 움직여도 남은 이득의 상한은 약 0.004이고, 그 이상은 전부 랭킹 — 즉 모델 계열 또는 피처 — 에서 와야 합니다. 그리고 랭킹 축에서는 gbdt 두 계열이 이미 0.9289~0.9304의 좁은 대역에 수렴했습니다. 이것이 루프가 "정체"로 조기 종료된 실질적 이유입니다.

**부수적으로 확인된 두 가지.** (1) iteration 3·4의 `train_val_gap`이 0.0509·0.0644로 커졌지만 검증 점수는 오르지 않았습니다 — 추가 용량이 학습 행에만 쓰였다는 뜻이고, 최고 구성(0.0293)이 가장 건전합니다. (2) 최고 구성의 `calibration_error` 0.0794 / `brier` 0.1008은 확률값이 평균 약 8%p 어긋난다는 의미입니다. 이는 `balanced_accuracy`에 영향을 주지 않았지만(랭킹 지표는 보정 무관), 확률을 그대로 쓰는 하위 활용에는 문제가 됩니다. 이 실행기에는 재보정 레버가 없어 측정만 되고 교정되지 않았습니다.

`unsupported_claims`에 대해: iteration 1과 4의 계획이 `feature_engineering`으로 표시되었습니다. 이 표시는 계획 산문에 대한 부분문자열 검사이고, 두 iteration의 계획 본문은 이 기록에 남아 있지 않아 계획이 실제로 그 기능에 *의존*했는지 확인할 수 없습니다. 따라서 두 시도가 그 때문에 무엇을 잃었다고 쓰지 않습니다 — 기록된 숫자는 이 실행기가 파생 피처 없이 달성한 값 그대로이며, 피처 생성은 아래 "다음 단계"에서 **만들어야 할 것**으로 다룹니다.

## 다음 단계 제안

목표는 이미 달성되었으므로, 아래는 "더 짜낼 곳"이 아니라 **다음 이득이 실제로 있는 축**에 대한 제안입니다. 이 데이터셋에는 별도로 기록된 caveat이 없어(`Data caveats`: 없음) 특정 열·값·분할을 배제하는 제약은 없습니다.

1. **동일 계열 내 추가 하이퍼파라미터 탐색은 중단할 것.** 근거가 분명합니다: `hist_gbdt`·`xgboost` 3개 점이 roc_auc 폭 0.0015, `at_best_cut` 폭 0.0021로 paired 해상도(0.003~0.006) 아래에 갇혔고, `cut_headroom` 0.00438은 `class_weight`/`scale_pos_weight` 조정의 상한을 ~0.004로 못 박습니다. 남은 예산을 여기에 쓰면 리샘플 노이즈를 사는 것입니다.

2. **다음 이득은 실행기 밖의 피처 축에서 사야 합니다 — 카드 자체를 바꿀 것.** 실행기는 파생/삭제/재인코딩을 하지 않으므로(`missing_indicator`/`missing_count`만 예외), 상호작용이나 비율 열은 **입력 CSV/카드 단계에서** 만들어 넣어야 합니다. 카드가 가리키는 후보는 구체적입니다: `capital-gain`/`capital-loss`(둘 다 skew high, target_corr moderate/weak)의 순액·이진화, `education-num`(target_corr strong) × `hours-per-week`, `age` 구간화, 그리고 `fnlwgt`(target_corr **none**, distinct high, outlier_rate 0.0297 — 원래 표본 가중치 열) 제거. 이 중 `fnlwgt` 제거는 열을 카드에서 빼는 것으로만 가능하고, 랭킹 상한 ≈0.85가 피처 한계인지 계열 한계인지를 구분해 줍니다.

3. **결측 레버에는 iteration을 쓰지 말 것.** `impute='none'`을 쓰는 계열에 `missing_indicator`를 붙이면 예측이 비트 단위로 동일하다는 것이 문서화되어 있고, `missing_count`도 별도 이득이 확인되지 않았습니다. 게다가 이 파일의 전체 결측률은 0.0095, 최악 열도 0.0575(`occupation`)에 불과합니다. 다만 `workclass`/`occupation`/`native-country`의 결측이 **왜** 발생했는지는 카드가 말해주지 않으므로, 그 사유(기록 체계/조사 시점 등)를 확인해 caveat으로 남기는 작업이 선행 과제입니다 — 그것이 정리되기 전까지 결측 파생 열을 이득으로 계산하지 마십시오.

4. **확률을 쓰려면 보정을, 결론을 굳히려면 시드 반복을 별도 단계로 둘 것.** `calibration_error` 0.0794는 이 루프에서 교정 불가능하므로(재보정 레버 없음), 확정된 최고 구성을 루프 밖에서 재보정(예: 별도 홀드백으로 등온/시그모이드 보정)하는 단계를 파이프라인에 추가해야 합니다. 또한 모든 결정이 단일 20% 검증 분할(seed 42)에서 이뤄졌고 검증-테스트 차이 −0.0071은 이 행 수로는 0과 구분되지 않으므로, 결론의 안정성이 필요하다면 분할/시드는 루프 내부에서 바꿀 수 없으니 **런 설정 수준에서 서로 다른 seed로 전체 실행을 2~3회 반복**해 최고 구성의 점수 분포를 확인하는 것이 가장 값싼 검증입니다.