# 최종 리포트 — bank-marketing (balanced_accuracy)

## 요약

목표는 달성되었습니다. 목표 임계값은 `balanced_accuracy` 0.7465(baseline 0.662 기준 파생)였고, 1회 시도만으로 검증 기준 **0.8736 (95% CI 0.8621~0.8843)** 을 기록했습니다. 한 번도 학습·선택에 쓰이지 않은 최종 테스트 20%에서도 **0.8528 (95% CI 0.8410~0.8645)** 로 임계값을 크게 상회했습니다. 총 5회 예산 중 1회만 사용하고 `goal_reached`로 종료되었으므로, critic 진단·재계획 경로는 이 결과에 전혀 개입하지 않았습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `min_samples_leaf=20`, `early_stopping=false`, `class_weight={"0":1.0,"1":6.0}` / preprocessing: `impute="none"`, `scale=false`, `missing_indicator=false`, `missing_count=false` | status `ok`, `balanced_accuracy=0.8736` (CI 0.8621~0.8843), `roc_auc=0.9395`, `pr_auc=0.6375`, `recall=0.8582`, `specificity=0.889028`, `train_val_gap=0.064663`, 학습 7.49초 | 없음 (목표 달성으로 critic 미실행) |

## 최고 성능 구성

- **model**: `hist_gbdt`
- **hyperparams (실제 적용값)**: `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `min_samples_leaf=20`, `early_stopping=false`, `class_weight={"0": 1.0, "1": 6.0}`
- **preprocessing (실제 적용값, 해당 attempt에서 읽음)**: `impute="none"`, `scale=false`, `missing_indicator=false`, `missing_count=false` — 즉 결측 대치를 하지 않고 트리가 NaN 분기를 직접 학습했으며, 스케일링·결측 파생열은 모두 없습니다.
- **dropped_hyperparams**: 없음 (executor가 거부한 키 없음)
- **프로토콜**: stratified 3-way split, seed 42, train 60% / validation 20% / test 20%

| 지표 | validation | 최종 테스트(held-back) |
|---|---|---|
| `balanced_accuracy` | **0.8736** (CI 0.8621~0.8843) | **0.8528** (CI 0.8410~0.8645) |
| `roc_auc` | 0.9395 | — |
| `pr_auc` / `average_precision` | 0.6375 | — |
| `f1` | 0.6367 | — |
| `accuracy` | 0.8854 | — |
| `precision` / `recall` | 0.5061 / 0.8582 | — |
| `specificity` | 0.889028 | — |
| `balanced_accuracy_at_best_cut` / `cut_headroom` | 0.883 / 0.009374 | — |
| `brier` / `calibration_error` | 0.081695 / 0.091758 | — |
| `train_balanced_accuracy` / `train_val_gap` | 0.9382884875043336 / 0.064663 | — |

검증 0.8736 대비 테스트 0.8528으로 **-0.0208** 차이가 있습니다. 루프는 오직 검증 숫자만 보고 최종 모델을 골랐으므로, 이 차이가 곧 **선택 편향(selection effect)의 크기**입니다. 이 실행이 실제로 입증한 값은 테스트 쪽 0.8528이며, 두 구간(0.8621~0.8843 vs 0.8410~0.8645)이 겹치지 않는다는 점도 검증 점수를 그대로 성능으로 인용하면 안 되는 이유입니다.

## 원인 분석

- **critic 진단은 0건입니다.** 첫 시도가 곧바로 임계값(0.7465)을 넘겨 루프가 종료되었기 때문에, 진단·재계획 경로는 한 번도 실행되지 않았습니다. 따라서 "시도 간 패턴"이라고 부를 것이 존재하지 않으며, 이 점수는 **첫 계획 하나가 낸 결과**입니다. 이 실행은 diagnose-and-replan 루프가 도움이 된다는 증거도, 해가 된다는 증거도 제공하지 않습니다.
- 시도가 1건뿐이므로 **비교 가능한 대안이 없고**, 어떤 레버가 얼마를 기여했는지는 이 데이터로 분리할 수 없습니다. baseline(`logreg`, median impute + standard scale) 대비 `balanced_accuracy` 0.662 → 0.8736, `roc_auc` 0.9101 → 0.9395로 올라갔지만, 그중 랭킹 축(모델 패밀리 교체)과 운영점 축(`class_weight`)의 배분은 측정되지 않았습니다. baseline의 `balanced_accuracy_at_best_cut`이 0.84였던 점을 감안하면 두 축이 함께 움직였다는 것까지가 이 데이터가 말할 수 있는 범위입니다.
- 진단 대신, 남은 여지가 어디에 있는지에 대한 **관측치**만 기록해 둡니다(이는 critic의 판정이 아니라 attempt 자체 지표입니다): `cut_headroom=0.009374`는 매우 작아 기본 `predict()` 컷이 이미 이 랭킹의 최적점 근처(`balanced_accuracy_at_best_cut=0.883`)에 있습니다. 즉 `class_weight` 추가 조정으로 얻을 수 있는 양은 검증 CI 폭(약 0.022)보다도 작습니다. `recall=0.8582` vs `specificity=0.889028`도 이미 균형에 가깝습니다.
- `train_val_gap=0.064663`은 400 iteration / `early_stopping=false` 구성으로서는 과도한 과적합 신호로 보기 어렵습니다. 반면 `calibration_error=0.091758`, `brier=0.081695`는 확률값이 평균 약 9%p 벗어나 있음을 뜻합니다 — 랭킹 지표는 영향받지 않지만, 확률 자체를 쓰는 용도에는 그대로 신뢰할 수 없고 이 executor에는 재보정 레버가 없습니다.
- `plan.unsupported_claims`에 `feature_engineering`이 기록되어 있으나, 해당 계획 본문을 읽으면 one-hot 인코딩과 결측 파생열이 **불필요/불가능하다는 점을 언급**한 것이며 그 기능에 의존하지 않았습니다. 따라서 이 attempt가 그 때문에 잃은 것은 없고, 파생 피처는 "실패한 것"이 아니라 "아직 만들지 않은 것"으로 아래 제안에 둡니다.

## 다음 단계 제안

1. **`class_weight`/운영점 재조정에 예산을 쓰지 마십시오.** `cut_headroom=0.009374`는 검증 CI 폭(0.8621~0.8843, 약 0.022)보다 작아, 조정해도 이 데이터에서는 개선으로 구분되지 않습니다. 남은 여지는 랭킹 축(패밀리·피처)에 있습니다.
2. **랭킹 축 확인은 "동일 파이프라인 + 다른 트리 패밀리" 1회로 제한하십시오.** 예: `impute="none"`, `scale=false`를 유지한 `xgboost`(`scale_pos_weight`로 동일 취지의 가중, `early_stopping_rounds`는 executor가 내부 eval set을 잡아줌). 다만 패밀리 교체의 관측된 크기는 `roc_auc` 0.0022~0.0077 수준이고 `roc_auc` 차이의 짝지은 분해능은 0.003~0.006이므로, **개선이 나와도 구분 불가일 가능성이 높다는 전제로** 실행해야 합니다. 하이퍼파라미터 재튜닝(같은 패밀리 내 0.0032 수준)은 이보다 더 기대치가 낮습니다.
3. **V16 / V9의 `'unknown'` 의미를 데이터 소유자에게 확인해 카드 단계에서 해결하십시오.** V16은 81.8%, V9는 28.8%가 `'unknown'`이며, 카드의 caveat대로 이 값들은 변환되지 않은 상태로 실측치처럼 집계·학습에 들어갔습니다. 이것이 "실제 범주"인지 "기록 누락 표기"인지가 정해지지 않은 한, 이 열들에 근거한 해석·중요도 논의는 신뢰할 수 없습니다. 또한 executor는 열을 추가·삭제·재인코딩할 수 없으므로(결측 파생열 2개만 예외) 해결은 **카드/원본 파일 쪽**에서 이루어져야 합니다. 확인이 끝나면 V16을 카드에서 제외하거나 명시적 결측으로 재표기한 뒤 재측정하는 선택지가 열립니다.
   - 같은 이유로 `missing_indicator` / `missing_count`는 권하지 않습니다: 카드의 실제 결측률은 0%이고, `impute="none"`과 함께 쓰면 `missing_indicator`는 예측이 비트 단위로 동일해지는 중복 레버로 이미 확인되어 있습니다.
4. **V12(`target_corr="strong"`)가 예측 시점에 실제로 사용 가능한 값인지 운영 담당자와 확인하십시오.** 이 열은 수치 피처 중 유일하게 강한 타깃 상관을 보이며 `roc_auc=0.9395`의 상당 부분을 떠받칠 가능성이 있습니다. 만약 결과가 확정된 뒤에만 얻어지는 값이라면 현재 점수는 배포 성능을 과대평가합니다 — 이는 모델 튜닝이 아니라 데이터 정의로만 해결됩니다.
5. **확률값을 그대로 쓸 계획이라면 별도 보정 단계를 파이프라인 밖에 두십시오.** `calibration_error=0.091758`은 약 9%p 오차를 뜻하지만, 이 executor는 재보정을 지원하지 않습니다(`CalibratedClassifierCV`·컷 이동 불가). 순위·임계 판단만 필요하면 현재 구성으로 충분합니다.