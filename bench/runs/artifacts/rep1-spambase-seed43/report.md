# AutoML 실행 보고서 — spambase (balanced_accuracy)

## 요약

목표는 달성되었습니다. 1회 시도, 즉 첫 계획 하나로 `hist_gbdt`가 검증 `balanced_accuracy=0.9533` (95% CI 0.9387~0.9670)을 기록해 목표 기준선 0.938과 baseline 0.9173을 모두 넘었습니다. 루프에서 한 번도 사용되지 않은 최종 테스트 20%에서도 `balanced_accuracy=0.9481` (95% CI 0.9335~0.9606)로 기준선 0.938을 상회했습니다. 총 5회 예산 중 1회만 사용하고 `goal_reached`로 종료되었습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `early_stopping=false` | `status=ok`, 검증 `balanced_accuracy=0.9533` (CI 0.9387~0.9670), `roc_auc=0.9842`, `pr_auc=0.9745` → 목표 달성 | 없음 (목표 달성으로 critic 미실행) |

## 최고 성능 구성

- **model**: `hist_gbdt`
- **hyperparams** (해당 시도의 적용값): `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `early_stopping=false`
- **preprocessing** (해당 시도의 적용값): `impute="median"`, `scale=false`, `missing_indicator=false`, `missing_count=false`
  - 카드의 `missing.overall_rate=0.0`이므로 median imputation은 실질적으로 무연산이며, 트리 계열이라 스케일링은 적용되지 않았습니다.
- **dropped_hyperparams**: 없음 (`[]`) — 제안한 키가 모두 그대로 적용되었습니다.
- **학습 시간**: 3.845초
- **분할 프로토콜**: stratified 60/20/20, seed 43 (카드 baseline과 동일 분할)

검증 지표 (validation 20%):

| metric | 값 |
|---|---|
| balanced_accuracy | **0.9533** (CI 0.9387~0.9670) |
| accuracy | 0.9587 |
| f1 | 0.9465 |
| precision | 0.9655 |
| recall | 0.9282 |
| specificity | 0.978495 |
| roc_auc | 0.9842 |
| pr_auc / average_precision | 0.9745 |
| balanced_accuracy_at_best_cut | 0.9563 |
| cut_headroom | 0.002964 |
| brier | 0.033926 |
| calibration_error | 0.020998 |
| train_balanced_accuracy | 1.0 |
| train_val_gap | 0.046664 |

**최종 보류 테스트 (20%, 루프 중 한 번도 사용되지 않은 행)**: `balanced_accuracy=0.9481` (95% CI 0.9335~0.9606). 검증 0.9533 대비 **+0.0052**이며, 이 차이가 선택 편향(validation 숫자로 선택을 내린 대가)의 크기입니다. 다만 검증 점수가 테스트 CI 안에 들어 있고, 애초에 후보가 1개뿐이어서 선택 압력이 거의 없었기 때문에 이 행들로는 그 차이를 0과 구분할 수 없습니다. 이 실행이 실제로 입증한 숫자는 검증값이 아니라 테스트 0.9481입니다.

## 원인 분석

이 실행에서 critic은 **한 번도 실행되지 않았습니다** (시도 1회, 진단 0회). 첫 계획이 곧바로 기준선을 넘어 루프가 종료되었기 때문에, 여기서 보고할 verdict 패턴은 존재하지 않습니다. 따라서 이 점수는 진단·재계획 경로의 산물이 아니라 **첫 계획 하나가 낸 결과**이며, 이 실행은 diagnose-and-replan 루프가 도움이 되는지 해가 되는지에 대해 어느 쪽으로도 아무것도 말해주지 않습니다.

시도가 1건뿐이므로 시도 간 움직임에서 만들 수 있는 이야기도 없습니다. 계획서 자체는 목표 기준선 0.938이 baseline logreg 랭킹의 상한(`balanced_accuracy_at_best_cut=0.9304`, `ks=0.8609`)보다 위에 있어 어떤 operating point 조정으로도 도달 불가라는 점을 근거로 모델 계열 교체(선형 → 부스팅 트리)를 택했고, 실제로 랭킹 축 지표가 함께 올라갔습니다(`roc_auc` 0.974 → 0.9842, `pr_auc` 0.9548 → 0.9745, `balanced_accuracy_at_best_cut` 0.9304 → 0.9563). 이 시도의 `unsupported_claims`는 비어 있고 `dropped_hyperparams`도 없으므로, 계획이 실행기가 못 하는 기능에 의존해 잃은 것은 없습니다.

성능을 제한하는 요인에 대해 확정적으로 말할 수 있는 것은 없습니다 — 진단이 없었고, 남은 격차를 측정할 두 번째 시도도 없습니다. 관찰 사실만 기록하면: `cut_headroom=0.002964`은 기본 결정 규칙이 이미 이 모델 랭킹에서 거의 최적 지점에 있다는 뜻이고(즉 `class_weight`/`scale_pos_weight`로 얻을 여지가 매우 작음), `train_balanced_accuracy=1.0`과 `train_val_gap=0.046664`은 학습 데이터를 완전히 맞춘 상태를 보여줍니다. 이 두 값은 다음 시도의 방향을 가리키는 단서일 뿐 진단이 아니며, 이 실행에서 검증된 바 없습니다.

## 다음 단계 제안

데이터 카드에는 별도로 기록된 caveat이 없습니다(`(없음)`), 따라서 특정 열·값·분할을 배제해야 하는 제약은 없습니다. 아래 제안은 모두 실행기가 실제로 지원하는 레버만 사용합니다.

1. **결론을 더 굳히려면 남은 예산을 성능 추가가 아니라 재현성 확인에 쓸 것.** 검증 CI 폭은 약 ±0.014, 테스트 CI 폭은 약 ±0.014로, 이 데이터 크기에서 0.003~0.006 미만의 차이는 구분되지 않습니다. seed·분할은 루프 내부에서 바꿀 수 없으므로, 확인이 필요하면 **루프 밖에서 다른 seed로 동일 구성(`hist_gbdt`, 위 하이퍼파라미터)을 재실행**하는 방식이어야 합니다. 이는 0.9481이 분할 운이 아니라는 점을 확인해 줍니다.

2. **더 높은 점수를 노린다면 랭킹 축만 건드릴 것.** `cut_headroom=0.002964`이므로 operating point 레버(`class_weight='balanced'`, `scale_pos_weight`)에 시도를 쓰는 것은 사실상 낭비입니다. 남은 격차는 랭킹, 즉 모델 계열/용량 쪽에 있습니다. 다만 참고 측정에서 계열 교체는 `roc_auc` 0.0022~0.0077, 같은 계열 내 재튜닝은 0.0032 범위였고 이 데이터의 구분 해상도가 0.003~0.006이므로, `xgboost`나 `random_forest`로의 1회 교체가 개선으로 읽힐 것이라 기대해서는 안 됩니다 — "개선"으로 보고할 수 있는 것은 CI 폭보다 큰 차이뿐입니다.

3. **용량/정규화 프로브를 1회만 던져볼 것.** `train_balanced_accuracy=1.0`, `train_val_gap=0.046664`이므로 `l2_regularization`을 올리거나 `max_leaf_nodes`를 줄인 구성(예: `l2_regularization=5.0`, `max_leaf_nodes=15`, 나머지 동일)을 한 번 재보는 것이 가장 값싼 확인입니다. 단, 결과 차이가 검증 CI 폭(약 0.028) 안에 들어오면 "동일하다"로 기록하고 채택 근거로 쓰지 말아야 합니다. `early_stopping=True`는 학습 행을 내주므로(2,760행에서 10% 손실) 우선순위가 낮습니다.

4. **확률값을 다운스트림에서 쓸 계획이면 보정은 이 루프 밖에서 처리할 것.** `brier=0.033926`, `calibration_error=0.020998`로 확률 오차는 평균 약 2%p 수준이며 나쁘지 않지만, 실행기에는 recalibration 레버가 없습니다(`CalibratedClassifierCV`나 임계값 이동은 제안 대상이 아님). 확률 기반 의사결정이 필요하다면 별도 파이프라인 단계로 구축해야 하며, 이는 분류 지표(현재 목표)에는 영향을 주지 않습니다.