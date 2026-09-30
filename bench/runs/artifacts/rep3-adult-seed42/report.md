## 요약

목표는 달성되었습니다. 목표 지표 `balanced_accuracy`의 임계값 0.8248을 첫 번째 시도에서 넘겼고, 검증 점수는 **0.8451 (95% CI 0.8368~0.8534)** 입니다. 사용한 반복은 5회 중 1회이며, 첫 계획이 곧바로 기준을 통과해 루프가 종료되었습니다. 한 번도 사용되지 않은 홀드백 테스트 20%에서의 점수는 **0.8405 (95% CI 0.8320~0.8492)** 로, 이 값이 이번 실행이 실제로 입증한 성능입니다(카드의 logreg 베이스라인 검증 점수 0.7664, CI 0.7575~0.7762 대비 상회).

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `learning_rate=0.06`, `max_iter=500`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=False`, `class_weight={"0":1.0,"1":3.0}` | status `ok` — `balanced_accuracy=0.8451` (CI 0.8368~0.8534), 임계값 0.8248 통과 → 루프 종료 | 없음 (목표 달성으로 critic 미실행) |

## 최고 성능 구성

iteration 1, `hist_gbdt`. 아래 `hyperparams` / `preprocessing`은 해당 시도 히스토리에 기록된 **실제 적용값**입니다. `dropped_hyperparams`는 비어 있어, 제안한 설정이 그대로 반영되었습니다.

- **model**: `hist_gbdt`
- **hyperparams**: `learning_rate=0.06`, `max_iter=500`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=false`, `class_weight={"0": 1.0, "1": 3.0}`
- **preprocessing**: `impute="none"`(트리가 NaN을 직접 분기), `scale=false`, `missing_indicator=false`, `missing_count=false`
- **프로토콜**: stratified 3-way split, seed 42, train 60% / validation 20% / test 20% (카드 베이스라인과 동일)
- **train_time_sec**: 13.737

검증 슬라이스 지표:

| metric | 값 |
|---|---|
| `balanced_accuracy` | 0.8451 (CI 0.8368~0.8534) |
| `balanced_accuracy_at_best_cut` | 0.8470 |
| `cut_headroom` | 0.001908 |
| `recall` | 0.8395 |
| `specificity` | 0.850646 |
| `accuracy` | 0.8480 |
| `f1` | 0.7255 |
| `precision` | 0.6387 |
| `roc_auc` | 0.9287 |
| `pr_auc` / `average_precision` | 0.8309 |
| `brier` | 0.102842 |
| `calibration_error` | 0.082122 |
| `train_balanced_accuracy` | 0.8990 |
| `train_val_gap` | 0.053939 |

**홀드백 테스트 점수**: `balanced_accuracy=0.8405` (95% CI 0.8320~0.8492). 검증 0.8451과의 차이 +0.0046은 선택 편향의 크기입니다 — 루프는 검증 숫자만 보고 결정했고, 테스트 슬라이스는 첫 fit 이전에 분리되어 어떤 결정에도 쓰이지 않았습니다. 다만 검증 점수가 테스트 CI(0.8320~0.8492) 안에 들어 있어, 이 행들만으로는 그 차이를 0과 구분할 수 없습니다.

## 원인 분석

**critic 진단은 0건입니다.** 첫 시도가 임계값을 넘겨 루프가 즉시 종료되었으므로 critic이 한 번도 실행되지 않았고, 따라서 "여러 판정에 걸친 패턴"이라는 것이 존재하지 않습니다. 이 점수는 전부 첫 계획 하나가 낸 것이며, 이번 실행은 진단·재계획 경로가 도움이 된다는 증거도, 해가 된다는 증거도 제공하지 않습니다.

비교 가능한 시도가 하나뿐이라 시도 간 변화로부터 원인을 구성할 수 없습니다. 아래는 진단이 아니라 단일 측정에서 읽히는 사실입니다.

- 베이스라인 `logreg`의 `balanced_accuracy_at_best_cut`은 0.8238이었고 목표 임계값 0.8248은 그보다 높게 설정되어 있었습니다(`exceeds_ranking_ceiling: true`). 즉 임계값 이동만으로는 도달할 수 없는 목표였고, 실제로 달성된 점수는 `roc_auc` 0.9075 → 0.9287, `pr_auc` 0.7706 → 0.8309처럼 순위 매김(ranking) 축이 함께 움직인 상태에서 나왔습니다.
- 운영점(operating point) 축은 거의 소진되었습니다: `cut_headroom`이 0.001908, `balanced_accuracy_at_best_cut`이 0.8470으로 기본 `predict()` 컷이 최적점에 근접해 있고, `recall` 0.8395와 `specificity` 0.850646이 거의 대칭입니다. 남은 여지는 `class_weight` 조정이 아니라 순위 매김 쪽에 있습니다.
- 확률값 자체는 그대로 읽기 어렵습니다: `calibration_error` 0.082122(평균 약 8%p 오차), `brier` 0.102842. 이 실행 환경에는 재보정 레버가 없어 측정만 되고 교정되지 않았습니다.
- `train_val_gap` 0.053939, `train_balanced_accuracy` 0.8990 — 과적합이 파국적이지는 않지만 학습 쪽이 앞서 있습니다. 이것이 성능을 제한했다고 단정할 근거는 이 한 번의 측정에 없습니다.
- `plan.unsupported_claims`에 `feature_engineering`이 기록되어 있으나, 해당 계획 본문은 파생 피처를 사용하겠다고 한 것이 아니라 전처리 선택(`impute: 'none'`, 스케일링 없음, `missing_indicator` 미사용이 `impute: none`에서 중복이라는 점)만 서술했습니다. 즉 계획이 사용 불가 기능에 **의존하지 않았고**, 이 시도가 그 때문에 무엇을 잃은 것은 아닙니다. 피처 생성 기능은 "실패한 것"이 아니라 "아직 없는 도구"이므로 다음 단계에 둡니다.

이 데이터에 대해 별도 기록된 caveat은 없습니다.

## 다음 단계 제안

1. **남은 예산은 순위 매김(ranking) 축에만 쓸 것.** `cut_headroom`이 0.001908이므로 `class_weight`를 2.5~4 사이로 재조정해도 회수 가능한 양은 CI 폭(±약 0.008)보다 작아 측정으로 구분되지 않습니다. 대신 용량 계열(`max_leaf_nodes` 63, `min_samples_leaf` 10~50, `l2_regularization` 0.1~10, `learning_rate` 0.03 + `max_iter` 상향)이나 `xgboost`로의 패밀리 교체를 시도하고, 판정은 `balanced_accuracy`가 아니라 `roc_auc` / `pr_auc` / `balanced_accuracy_at_best_cut`으로 하십시오. 단, 문서화된 경험치상 한 패밀리 내 재튜닝은 `roc_auc` 0.003 수준, 패밀리 교체는 0.002~0.008 수준이며 이 측정들의 해상도가 0.003~0.006이므로, 개선을 주장하려면 CI 겹침 여부를 먼저 확인해야 합니다.
2. **단일 20% 검증 슬라이스라는 제약을 별도로 해소할 것.** 이 실행에는 교차검증도 out-of-fold 예측도 없고, 최종 테스트 슬라이스는 이미 한 번 소비되었습니다. 이 구성이 실제로 안정적인지는 루프 밖에서 다른 seed의 split 또는 CV로 `hist_gbdt` 동일 설정을 재적합해 확인하는 것이 가장 값싼 검증입니다. 이는 +0.0046의 검증-테스트 차이가 선택 편향인지 단순 리샘플 변동인지 가려줍니다.
3. **확률값을 쓰려면 재보정을 파이프라인 밖에 붙일 것.** `calibration_error` 0.082122는 순위는 좋아도 확률이 체계적으로 어긋나 있음을 뜻하며, 이 실행기에는 교정 레버가 없습니다. 점수 임계값을 운영에 쓰거나 확률을 리스크로 해석해야 한다면, 별도 홀드아웃에서의 사후 보정(및 그 보정 후의 컷 선택)을 실행기 외부 작업으로 계획하십시오. 목표 지표에는 영향이 없습니다.
4. **피처 쪽 실험은 카드/도구 확장 과제로 분리할 것.** 실행기는 파생·상호작용·인코딩 변경·열 삭제를 하지 않으므로, 예컨대 `fnlwgt`(카드상 `target_corr: none`, 표본 가중치 성격)를 빼거나 `capital-gain`/`capital-loss`(`skew: high`)를 변환해 보려면 계획이 아니라 데이터 카드/전처리 단계를 바꿔야 합니다. 어느 쪽도 이번 실행에서 시험되지 않았으므로, 효과를 예상 수치로 약속하지 말고 "미검증 가설"로 남겨두고 도구를 먼저 만드는 편이 낫습니다. 반대로 `missing_indicator`는 `impute: "none"`과 함께 쓰면 비트 단위로 동일한 예측을 낸다고 이미 확인된 항목이므로, 여기에 반복을 쓰지 마십시오.