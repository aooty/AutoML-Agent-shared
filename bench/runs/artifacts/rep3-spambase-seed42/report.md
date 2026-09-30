# AutoML 최종 리포트 — `spambase` (balanced_accuracy)

## 요약

목표는 달성되었습니다. 1회 시도로 `hist_gbdt`가 검증 `balanced_accuracy` **0.9534** (95% CI 0.9384~0.9671)를 기록해 목표선 **0.9421**과 baseline `logreg` **0.9228**을 모두 넘었고, 한 번도 사용되지 않은 최종 테스트 20%에서도 **0.9441** (95% CI 0.9273~0.9586)로 목표선을 넘었습니다. 루프는 첫 시도에서 목표를 충족해 종료되었으므로(사용 1/5회), critic 진단·재계획은 한 번도 실행되지 않았습니다. 다만 테스트 점수의 목표선 초과 폭(+0.0020)은 해당 CI 폭에 비해 매우 작으므로, "확실히 넘었다"가 아니라 "이 행들에서는 넘었고 여유는 좁다"로 읽어야 합니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `min_samples_leaf=20`, `early_stopping=False` | status `ok` — 검증 `balanced_accuracy` 0.9534 (CI 0.9384~0.9671), `roc_auc` 0.9912, `train_val_gap` 0.04613, `train_time_sec` 3.826 → 목표 달성으로 루프 종료 | 없음 (목표 달성으로 critic 미실행) |

`dropped_hyperparams`는 비어 있고 `unsupported_claims`도 비어 있습니다 — 계획한 설정이 그대로 적용되었습니다.

## 최고 성능 구성

- **iteration**: 1
- **model**: `hist_gbdt` (HistGradientBoostingClassifier)
- **hyperparams** (해당 attempt에 적용된 값): `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `min_samples_leaf=20`, `early_stopping=False`
- **preprocessing** (해당 attempt에 적용된 값): `impute="median"`, `scale=false`, `missing_indicator=false`, `missing_count=false`
- **프로토콜**: stratified 3-way split, seed 42, train 60% / validation 20% / test 20% (baseline과 동일 분할)
- **학습 시간**: 3.826초

| 구분 | balanced_accuracy |
|---|---|
| 검증 (선택에 사용됨) | **0.9534** (95% CI 0.9384~0.9671) |
| 최종 held-back 테스트 (1회만 채점, 선택에 미사용) | **0.9441** (95% CI 0.9273~0.9586) |
| 목표선 / baseline | 0.9421 / 0.9228 |

검증과 테스트의 차이 **+0.0094**는 선택 편향의 크기입니다 — 이 실행의 모든 선택은 검증 숫자로 이루어졌으므로, 모델에 대해 실제로 입증된 값은 테스트 쪽 0.9441입니다. 다만 검증 점수가 테스트 CI(0.9273~0.9586) 안에 들어 있으므로, 이 행 수로는 그 차이를 0과 구분할 수 없습니다.

검증 슬라이스의 부가 지표(모두 검증 기준): `accuracy` 0.9576, `f1` 0.9455, `precision` 0.9575, `recall` 0.9337, `specificity` 0.973118, `roc_auc` 0.9912, `pr_auc` / `average_precision` 0.9883, `brier` 0.034933, `calibration_error` 0.024347, `balanced_accuracy_at_best_cut` 0.9585, `cut_headroom` 0.00509, `train_balanced_accuracy` 0.9995, `train_val_gap` 0.04613.

## 원인 분석

이 실행에는 critic 진단이 **0건**입니다(시도 1회, 진단 0회). 첫 계획이 곧바로 목표선을 넘겨 루프가 끝났기 때문에, 위 점수는 전적으로 **첫 계획 하나가 낸 결과**입니다. 따라서 진단·재계획 경로에 대해서는 이 실행이 유리한 증거도 불리한 증거도 제공하지 않습니다 — 그 경로가 도움이 되는지 여부는 여기서 측정되지 않았습니다.

시도가 하나뿐이므로 시도 간 이동(어떤 레버가 얼마를 벌었는지)에 대한 비교 근거도 존재하지 않습니다. 확인 가능한 사실은 다음 세 가지뿐이며, 이는 진단이 아니라 단일 측정치의 서술입니다.

- 목표선 0.9421은 baseline `logreg` 랭킹의 상한(`balanced_accuracy_at_best_cut` = 0.9415)보다 높게 설정되어 있었습니다. 즉 baseline의 결정 임계값을 어떻게 움직여도 도달할 수 없는 지점이었고, 이번 시도는 결정 규칙이 아니라 모델 계열 자체를 바꿔(선형 → 부스팅 트리) 랭킹 축을 올린 구성입니다. 검증 `roc_auc` 0.9912 대 baseline 0.9772가 그 축의 이동을 보여줍니다.
- 검증 `cut_headroom`은 0.00509로 작습니다. 즉 기본 임계값이 이 랭킹에서 거의 최적에 가깝고, 남은 여지는 임계값/`class_weight`가 아니라 랭킹(모델 계열·특징) 쪽에 있습니다. 계획이 `class_weight`를 쓰지 않은 판단은 이 숫자와 일관됩니다.
- `train_balanced_accuracy` 0.9995 대 검증 0.9534(`train_val_gap` 0.04613)는 학습 데이터를 거의 완전히 맞추는 용량 설정임을 뜻합니다. 이것이 성능을 제한했는지 여부는 비교 시도가 없어 이 실행으로는 판정할 수 없습니다.

## 다음 단계 제안

1. **목표선 여유가 좁다는 점을 먼저 해소한다.** 테스트 0.9441은 목표선 0.9421을 +0.0020 넘겼을 뿐이고 CI는 0.9273~0.9586입니다. 같은 데이터·같은 프로토콜에서 seed를 바꾼 별도 실행(분할·seed는 루프 안에서 바꿀 수 없는 구성값이므로 새 실행으로)을 몇 회 돌려 목표 달성이 분할 우연이 아닌지 확인하는 것이 가장 저렴하고 가치 있는 다음 작업입니다. 이는 "달성"이라는 결론 자체의 신뢰도를 결정합니다.
2. **남은 예산 4회는 랭킹 축에만 쓴다.** `cut_headroom` 0.00509이므로 `class_weight='balanced'`나 `scale_pos_weight` 같은 operating-point 레버는 기대값이 거의 없습니다. 대신 (a) `hist_gbdt` 용량 재조정(`learning_rate`↓ + `max_iter`↑, 또는 `min_samples_leaf`↑ / `l2_regularization`↑로 `train_val_gap` 0.04613 축소 시도), (b) 파이프라인을 그대로 두고 `xgboost`로 계열 교체 — 두 가지가 남은 축의 정당한 후보입니다. 단, 실행 환경에서 관측된 계열 교체 폭은 `roc_auc` 0.0022~0.0077, 동일 계열 내 재튜닝은 0.0032 수준이고 이 단일 20% 검증 슬라이스의 `balanced_accuracy` CI 폭은 약 0.029이므로, **이 크기의 개선은 이 분할 하나로는 구분되지 않습니다.** 개선 여부 판정은 1번의 다중 seed 재측정과 묶어서 해야 합니다.
3. **결측 관련 레버는 이 데이터에서 제안하지 않는다.** 카드상 `missing.overall_rate = 0.0`, 결측 열 0개이므로 `missing_indicator` / `missing_count` / `impute: none`은 추가할 정보가 없습니다. 적용된 `impute="median"`도 실질적으로 무작동이며, 여기에 iteration을 쓰는 것은 낭비입니다.
4. **진단·재계획 경로는 별도로 검증이 필요하다.** 이번 실행은 critic을 한 번도 실행하지 않았으므로 그 경로의 유용성에 대한 근거가 없습니다. 더 어려운 목표선(예: 테스트 기준 0.95 이상)이나 다른 데이터셋에서 최소 2~3회 진단이 발생하는 실행을 한 번 확보해야, 이 루프에서 재계획이 실제로 점수를 올리는지 말할 수 있습니다.
5. **확률값을 그대로 쓰려면 별도 작업이 필요하다.** 검증 `calibration_error` 0.024347, `brier` 0.034933 — 평균 약 2.4%p 어긋납니다. 현재 executor에는 재보정 레버가 없으므로(`CalibratedClassifierCV`·임계값 이동 불가) 이는 iteration으로 해결할 항목이 아니고, 확률 자체가 필요한 용도라면 파이프라인 밖에서 보정 단계를 **구축**해야 하는 항목입니다.