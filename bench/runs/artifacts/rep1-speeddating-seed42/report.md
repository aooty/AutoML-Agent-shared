# AutoML 최종 리포트 — `speeddating` / `balanced_accuracy`

## 요약

목표를 달성했습니다. 목표 기준선 `balanced_accuracy` 0.7514에 대해, 2회차 시도의 `hist_gbdt`가 검증 0.7793 (95% CI 0.7462~0.8066), 그리고 한 번도 사용되지 않은 최종 테스트 20%에서 0.7902 (95% CI 0.7617~0.8173)를 기록했습니다. 총 5회 예산 중 2회만 사용했고, critic 진단은 1회 발생했습니다. 카드의 baseline(`logreg`, 0.6685)과 비교하면 검증 기준 +0.1108이며, 개선의 대부분은 랭킹 품질이 아니라 결정 경계(operating point)가 이동한 결과입니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=false`, `class_weight={"0":1.0,"1":3.5}`, `preprocessing: impute='none', scale=false` | `balanced_accuracy` 0.6903 (CI 0.6603~0.7207) — 기준 미달(-0.0611). `recall` 0.4457 / `specificity` 0.9350, `roc_auc` 0.8698, `balanced_accuracy_at_best_cut` 0.7987, `cut_headroom` 0.1084, `train_val_gap` 0.3096 | `failure_type: hyperparam`. 랭킹은 이미 충분(`cut_headroom` 0.1084가 남은 격차 0.0611의 177%)하고 부족분 전체가 operating point 축에 있다고 판단. 양성 클래스 가중치를 `'balanced'`(≈5.07) 수준까지 올리고 나머지는 고정하라고 지시 |
| 2 | `hist_gbdt` | `max_iter=250`, `learning_rate=0.06`, `max_leaf_nodes=15`, `min_samples_leaf=40`, `l2_regularization=5.0`, `early_stopping=false`, `class_weight={"0":1.0,"1":6.0}`, `preprocessing: impute='none', scale=false` | **`balanced_accuracy` 0.7793 (CI 0.7462~0.8066)** — 목표 달성. `recall` 0.7029 / `specificity` 0.8557, `roc_auc` 0.8739, `pr_auc` 0.6055, `cut_headroom` 0.0247, `train_val_gap` 0.1716 | — (목표 달성으로 루프 종료, 진단 없음) |

두 시도의 `dropped_hyperparams`는 모두 비어 있고 `unsupported_claims`도 비어 있습니다. 실패·OOM·타임아웃은 없었습니다(`train_time_sec` 10.756 / 6.148, 제한 600초).

## 최고 성능 구성

iteration 2, 모델 `hist_gbdt`. 아래 값은 히스토리의 적용된 `hyperparams` / `preprocessing`을 그대로 인용한 것입니다.

```
model: hist_gbdt
hyperparams:
  max_iter: 250
  learning_rate: 0.06
  max_leaf_nodes: 15
  min_samples_leaf: 40
  l2_regularization: 5.0
  early_stopping: false
  class_weight: {"0": 1.0, "1": 6.0}
  random_state: 42
preprocessing:
  impute: none          # hist_gbdt가 NaN을 직접 분기 — 대치 없음
  scale: false
  missing_indicator: false
  missing_count: false
split: stratified 60/20/20, seed 42
```

| 지표 | 검증(20%) | 최종 테스트(20%, 미사용 행) |
|---|---|---|
| `balanced_accuracy` | **0.7793** (CI 0.7462~0.8066) | **0.7902** (CI 0.7617~0.8173) |
| `recall` | 0.7029 | — |
| `specificity` | 0.855714 | — |
| `precision` | 0.4899 | — |
| `f1` | 0.5774 | — |
| `accuracy` | 0.8305 | — |
| `roc_auc` | 0.8739 | — |
| `pr_auc` / `average_precision` | 0.6055 | — |
| `balanced_accuracy_at_best_cut` | 0.804 | — |
| `cut_headroom` | 0.0247 | — |
| `brier` / `calibration_error` | 0.1180 / 0.1115 | — |
| `train_val_gap` | 0.1716 | — |

검증 0.7793 대 테스트 0.7902의 차이(-0.0108, 테스트가 오히려 높음)가 이 실행의 선택 편향 크기입니다. 루프는 검증 숫자만 보고 최고 시도를 골랐으므로, 모델에 대해 이 실행이 실제로 입증하는 값은 테스트 0.7902입니다. 다만 검증 점수가 테스트 CI(0.7617~0.8173) 안에 들어 있어, 이 행들만으로는 두 값의 차이를 0과 구분할 수 없습니다. 어느 쪽으로 읽어도 목표 0.7514는 두 측정 모두에서 초과했습니다.

## 원인 분석

critic 진단은 1회 존재하며(iteration 1), 그 진단대로 한 번 재계획한 뒤 목표를 통과했습니다.

- **성능을 제약한 것은 랭킹이 아니라 결정 경계였습니다.** iteration 1은 이미 `roc_auc` 0.8698, `balanced_accuracy_at_best_cut` 0.7987 — 즉 컷만 제대로 놓이면 목표를 0.047 초과할 수 있는 랭킹을 갖고 있었는데, 기본 `predict()` 컷이 다수 클래스 쪽에 크게 치우쳐 `recall` 0.4457 / `specificity` 0.9350으로 갈렸습니다. `cut_headroom` 0.1084가 남은 격차 0.0611의 177%라는 critic의 계산이 정확히 이 상황을 지목했습니다.
- **두 시도의 차이는 리샘플 잡음이 아닙니다.** iteration 1의 CI 상한 0.7207과 iteration 2의 CI 하한 0.7462가 겹치지 않으므로, 0.6903 → 0.7793의 이동(+0.0890)은 이 데이터가 실제로 분리해 주는 차이입니다.
- **그 이동의 내용도 operating point 축입니다.** `balanced_accuracy_at_best_cut`은 0.7987 → 0.804로 0.0053만 움직였고, `roc_auc`는 0.8698 → 0.8739로 0.0041 움직였습니다 — 둘 다 이 환경에서 알려진 짝지은 해상도(약 0.003~0.006) 수준이라 랭킹이 개선되었다고 주장할 수 없습니다. 반면 `cut_headroom`은 0.1084 → 0.0247로 줄었습니다. 즉 이번 실행에서 목표를 넘긴 것은 `class_weight`를 3.5 → 6.0으로 올려 컷을 recall/specificity 교차점 근처로 옮긴 결과이며, `balanced_accuracy` = (recall + specificity)/2 관점에서 recall 0.4457→0.7029, specificity 0.9350→0.8557의 교환이 전부입니다.
- **다만 iteration 2는 한 축에 두 레버를 함께 움직였습니다.** critic은 가중치만 `'balanced'`(≈5.07)로 올리라고 지시했으나, 실제 계획은 가중치를 6.0으로 잡으면서 동시에 용량을 줄였습니다(`max_leaf_nodes` 31→15, `min_samples_leaf` 20→40, `l2_regularization` 1.0→5.0, `max_iter` 400→250). 결과적으로 `train_val_gap`이 0.3096→0.1716, `train_balanced_accuracy`가 0.9999→0.9509로 내려간 것은 확인되지만, +0.0890 중 가중치 몫과 정규화 몫은 이 두 행으로 분리할 수 없습니다. 목표를 넘겼기 때문에 비용은 없었지만, 다음 미세조정 때 기준점으로 삼기에는 귀속이 깨끗하지 않은 관측입니다.
- **부작용 하나는 기록해 둘 만합니다.** `calibration_error`가 0.0608 → 0.1115, `brier`가 0.1052 → 0.1180으로 악화됐습니다. 클래스 가중치를 올린 대가로 예측 확률은 확률로 읽기 어려워졌습니다 — `balanced_accuracy`만 목표라면 무해하지만, 이 모델의 확률값을 그대로 쓰려면 평균 약 11%p의 편차를 감안해야 합니다. 이 실행 환경에는 재보정 레버가 없어 측정만 되고 교정되지 않았습니다.

## 다음 단계 제안

1. **`class_weight`를 5.0 / 7.0으로 한 번씩 고정 비교하되, 기대치를 미리 낮춰 잡을 것.** 남은 `cut_headroom`은 0.0247이므로 이 축에서 얻을 수 있는 최대치는 검증 0.804 근처입니다. 그런데 검증 CI 폭이 약 ±0.03이므로 0.0247짜리 개선은 이 슬라이스에서 잡음과 구분되지 않습니다. 즉 "가중치 미세조정으로 더 짜낸다"는 방향은 실행할 수는 있으나, 결과를 개선으로 보고해서는 안 되는 구간입니다. 오히려 iteration 2의 두 레버 혼합을 풀어(가중치 6.0 + iteration 1의 용량 설정) 어느 쪽이 무엇을 벌었는지 귀속시키는 데 1회를 쓰는 편이 재현성 측면에서 남는 것이 많습니다.
2. **더 얻을 것이 있다면 랭킹 축이며, 같은 파이프라인(`impute='none'`)으로 `xgboost`를 한 번 시도할 것.** `balanced_accuracy_at_best_cut`이 0.804에서 멈춰 있다는 것은 컷을 완벽히 놓아도 0.804가 상한이라는 뜻이고, 그 위로 가려면 `roc_auc`/`pr_auc`가 올라가야 합니다. 이 환경에서 계열 교체는 `roc_auc` 0.0022~0.0077 범위로 관측된 적이 있어 크기를 미리 약속할 수는 없지만, operating point 축이 사실상 소진된 지금 남은 유일한 축입니다.
3. **`missing_indicator` / `missing_count`에는 예산을 쓰지 말 것.** 현재 최고 구성은 `impute='none'`으로 `hist_gbdt`가 NaN을 직접 분기하고 있고, 이 조합에서 `missing_indicator`는 비트 단위로 동일한 예측을 낸다는 것이 이미 확인된 사실입니다. `missing_count`도 랜덤 분할에서 0.0003 이하로 측정됐습니다. 특히 `expected_num_interested_in_me`(결측률 0.7852)처럼 결측이 `wave`(설문 회차)에 연동될 가능성이 있는 열에서는, 층화 랜덤 분할이 "기록 체제 식별"을 이득으로 잘못 계상할 위험이 있습니다 — 얻는 것은 없고 잘못 읽을 위험만 있는 레버입니다.
4. **이 실행 환경 밖에서 두 가지 기능을 확보할 것: 임계값 선택과 확률 재보정, 그리고 `field` 열의 재인코딩.** 현재 executor는 항상 `predict()`의 기본 컷을 쓰므로 `class_weight`로 컷을 간접 조작하는 우회로만 있고, 그 대가가 `calibration_error` 0.1115입니다. 검증셋에서 컷을 직접 고르는 절차를 갖추면 `balanced_accuracy_at_best_cut` 0.804를 원칙적으로 회수할 수 있고, 재보정을 붙이면 확률값 자체를 하위 의사결정에 쓸 수 있게 됩니다. 또 카드에서 `field`(고카디널리티, 1개 열)가 드롭되었는데, executor는 파생·재인코딩을 하지 않으므로 이 신호를 쓰려면 데이터 카드 단계에서 축약된 형태로 넣어야 합니다. 셋 다 "이번에 실패한 것"이 아니라 "아직 만들지 않은 것"입니다.

*참고: 이 데이터셋에는 별도로 기록된 data caveat이 없어, 위 권고를 무효화하는 열/값/분할 관련 경고는 없습니다. 3번에서 언급한 결측-회차 연동 우려는 카드의 집계(결측률 0.7852, `wave` 열 존재)와 executor 문서의 일반 지침에서 도출한 주의사항이며, 확인된 caveat이 아닙니다.*