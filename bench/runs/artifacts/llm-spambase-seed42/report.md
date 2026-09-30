# AutoML 최종 리포트 — spambase (balanced_accuracy)

## 요약

목표 지표(`balanced_accuracy` ≥ 0.9421)는 **검증 기준으로 1회 시도에서 달성**되었습니다. 최고 구성은 iteration 1의 `hist_gbdt`로 검증 `balanced_accuracy` = **0.9484** (95% CI 0.9322~0.9623)이며, 카드의 `logreg` 베이스라인 0.9228과 베이스라인 랭킹 상한(`balanced_accuracy_at_best_cut` 0.9415)을 모두 넘었습니다. 다만 한 번도 사용되지 않은 테스트 20%에서의 점수는 **0.9362** (95% CI 0.9167~0.9534)로 목표선 0.9421보다 낮았으므로, "목표 달성"은 선택이 이루어진 검증 슬라이스에서의 사실이며 미사용 행에서 재확인된 것은 아닙니다. 5회 예산 중 1회만 사용하고 `goal_reached`로 조기 종료했습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=500`, `learning_rate=0.08`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.15`, `n_iter_no_change=25`, `random_state=42` | `status=ok` · `balanced_accuracy`=0.9484 (CI 0.9322~0.9623) · `roc_auc`=0.9874 · `train_val_gap`=0.0350 · 학습 2.193초 · 목표선 0.9421 초과 → 루프 종료 | critic 기록 없음 (`critic: null` — 목표 달성으로 진단 단계가 실행되지 않음) |

`dropped_hyperparams`는 비어 있고 `unsupported_claims`도 비어 있으므로, 계획이 요청한 설정은 전부 그대로 적용되었습니다.

## 최고 성능 구성

**iteration 1 · `hist_gbdt`** (실제로 빌드된 값 기준)

```
model: hist_gbdt
hyperparams:
  max_iter: 500
  learning_rate: 0.08
  max_leaf_nodes: 31
  l2_regularization: 1.0
  early_stopping: true
  validation_fraction: 0.15
  n_iter_no_change: 25
  random_state: 42
preprocessing:
  impute: median          # 이 파일은 결측 0% 이므로 실질적 no-op
  scale: false
  missing_indicator: false
  missing_count: false
protocol: stratified 60/20/20, seed 42
```

| 지표 | 검증(20%) | 최종 테스트(20%, 미사용 행) |
|---|---|---|
| `balanced_accuracy` | **0.9484** (CI 0.9322~0.9623) | **0.9362** (CI 0.9167~0.9534) |
| `accuracy` | 0.9522 | — |
| `f1` | 0.9387 | — |
| `precision` / `recall` | 0.9466 / 0.9309 | — |
| `specificity` | 0.9660 | — |
| `roc_auc` / `pr_auc` | 0.9874 / 0.9842 | — |
| `balanced_accuracy_at_best_cut` / `cut_headroom` | 0.9540 / 0.0056 | — |
| `brier` / `calibration_error` | 0.0375 / 0.0159 | — |
| `train_balanced_accuracy` / `train_val_gap` | 0.9834 / 0.0350 | — |

검증 0.9484 대 테스트 0.9362의 차이 **+0.0122**가 이 실행의 선택 편향 크기입니다. 루프는 검증 숫자만 보고 최고 시도를 골랐으므로, 모델에 대해 실제로 입증된 값은 테스트 쪽 0.9362이며 이 값은 목표선 0.9421 아래입니다. 다만 검증 점수는 테스트 CI(0.9167~0.9534) 안에 들어오므로, 이 행 수(약 920행)로는 두 값의 차이가 0과 구분되지 않습니다 — 즉 "실제로 더 낮다"고 단정할 근거도 없습니다.

## 원인 분석

- **베이스라인 대비 개선의 성격은 랭킹 개선입니다.** 베이스라인 `logreg`는 `ks`=0.883, `balanced_accuracy_at_best_cut`=0.9415로 어떤 컷을 골라도 목표선 0.9421에 닿을 수 없는 구조였습니다(`exceeds_ranking_ceiling: true`, `ks_shortfall` 0.0012). 계획대로 선형 계열을 트리 부스팅으로 교체한 결과 `roc_auc`가 0.9772 → 0.9874, `pr_auc`가 0.9647 → 0.9842로 올라가 랭킹 자체가 개선되었고, 이것이 목표 달성의 실질적 원인입니다.
- **운영점(operating point)은 이미 거의 최적입니다.** `cut_headroom`=0.0056, 즉 이 모델의 랭킹을 어떤 임계값으로 잘라도 `balanced_accuracy`는 최대 0.9540까지만 갑니다. `recall` 0.9309 대 `specificity` 0.9660으로 두 축이 크게 비대칭도 아닙니다(불균형은 1.54:1). 따라서 `class_weight`류 레버로 짜낼 여지는 0.006 미만이며, 남은 개선분은 모델 계열/특징 쪽에 있습니다.
- **비교할 다른 시도가 없습니다.** 시도 이력이 1건이고 `critic`은 `null`이므로, "시도 간 패턴"으로 설명할 수 있는 것이 없습니다. 유일한 시도의 검증 CI 폭은 약 ±0.015(0.9322~0.9623)이며, 이 폭은 목표선까지의 여유(0.9484 − 0.9421 = 0.0063)와 검증–테스트 격차(0.0122)를 모두 삼킬 만큼 큽니다. **이 실행의 실질적 한계는 모델 용량이나 전처리가 아니라 측정 해상도입니다**: 20% 단일 검증 슬라이스 하나로 0.006 수준의 마진을 판정했고, 예산 5회 중 1회에서 멈췄기 때문에 그 마진이 재현되는지는 확인되지 않았습니다.
- **과적합은 문제로 보이지 않습니다.** `train_balanced_accuracy` 0.9834 대 검증 0.9484, `train_val_gap` 0.0350 — 내부 `early_stopping`이 반복 수를 스스로 잘랐고, 이 정도 격차는 부스팅 트리에서 규제를 급히 올려야 할 신호는 아닙니다.
- **확률값은 읽을 만합니다.** `brier` 0.0375, `calibration_error` 0.0159 — 예측 확률이 평균 약 1.6%p 오차이므로 심한 과확신은 없습니다. 이 실행 환경에는 재캘리브레이션 레버가 없으므로 이는 진단으로만 씁니다.
- `unsupported_claims`는 비어 있고 `dropped_hyperparams`도 비어 있어, 실행기의 제약 때문에 이 시도가 잃은 것은 없습니다. 즉 0.9484는 이 실행기가 실제로 낸 숫자 그대로입니다.

## 다음 단계 제안

데이터 카드에 기록된 주의사항은 없고(결측 0%, 비수치형 0열, 그룹 누수 없음) 전 열이 수치형이므로, 아래 제안은 어떤 caveat와도 충돌하지 않습니다.

1. **최우선: 마진의 재현성을 다른 seed로 확인.** 이번 실행에서 목표선 초과 폭(0.0063)이 검증 CI 폭(±0.015)과 검증–테스트 격차(0.0122)보다 작습니다. 루프 안에서는 split·seed가 고정 설정이라 바꿀 수 없으므로, **동일한 `hist_gbdt` 구성을 seed만 달리한 여러 실행으로 재측정**해 `balanced_accuracy`의 분포를 보십시오. 0.9421을 안정적으로 넘는지가 확인되지 않으면 이 구성을 "0.948 모델"로 배포 문서에 쓰면 안 됩니다. 이것이 나머지 제안 전부의 판정 기준을 만들어 줍니다.
2. **남은 여지는 랭킹 축에만 있으므로 용량/계열을 건드리되, 기대치를 작게 잡으십시오.** `cut_headroom`이 0.0056이므로 `class_weight='balanced'`나 `scale_pos_weight`에는 예산을 쓰지 마십시오(최대 이득이 이미 0.006 미만으로 상한이 잡혀 있습니다). 대신 남은 4회 예산으로 (a) `max_leaf_nodes=63`, `learning_rate=0.05`, `max_iter` 유지 + `early_stopping` 유지, (b) `xgboost` 계열 스왑을 각각 1회 시도하는 편이 낫습니다. 다만 참고 측정에서 동일 계열 재튜닝은 `roc_auc` 0.0032, 계열 스왑은 0.0022~0.0077 범위였고 짝지은 해상도가 0.003~0.006이므로, **여기서 나오는 개선분은 이 슬라이스에서 잡음과 구분되지 않을 가능성이 높습니다.** 개선을 주장하려면 1번의 다중 seed 측정과 함께 읽어야 합니다.
3. **결측 관련 레버(`missing_indicator`, `missing_count`, `impute: none`)는 시도하지 마십시오.** 이 파일의 `overall_rate`는 0.0이고 결측 열은 0개이므로 세 레버 모두 정의상 no-op이며, 이번 시도의 `impute: median`도 실질적으로 아무 일도 하지 않았습니다. 예산 1회를 확실히 낭비하는 항목입니다.
4. **더 큰 도약을 원한다면 실행기 밖에서 특징 변환을 준비하십시오.** 57열 중 거의 전부가 `skew: high`인 카운트/빈도 열이고 `capital_run_length_*`는 `outlier_rate` 0.08~0.12입니다. 현재 실행기는 파생·변환·상호작용·열 삭제를 하지 않으므로(로그/비율 변환은 계획에 써도 실행되지 않음), 로그 변환이나 `capital_run_length` 비율 같은 열은 **데이터 카드 단계에서 추가되어야** 합니다. 트리 계열은 단조 변환에 불변이라 `hist_gbdt` 자체 이득은 크지 않을 것이므로, 이 작업의 가치는 주로 선형·거리 기반 계열을 동일 프로토콜에서 재비교할 수 있게 만드는 데 있습니다.