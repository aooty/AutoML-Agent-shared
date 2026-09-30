# AutoML 최종 리포트 — `spambase` / balanced_accuracy

## 요약

**목표는 달성하지 못했습니다.** 목표 기준선은 `balanced_accuracy` 0.967이었으나, 3회 시도 중 최고 검증 점수는 iteration 1의 `hist_gbdt`가 기록한 **0.9302 (95% CI 0.9118~0.9468)** 로 기준에 0.0368 부족했습니다. 최종 held-back 테스트(한 번도 사용되지 않은 20% 행)에서는 같은 모델이 **0.9467 (95% CI 0.9313~0.9615)** 를 기록했지만, 이 역시 CI 상한(0.9615)이 0.967에 닿지 않습니다. 루프는 5회 예산 중 3회만 사용하고 연속 미개선(정체)으로 조기 종료되었으며, critic 진단은 2회 모두 동일하게 `wrong_model_family`였습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=10`, `l2_regularization=1.0`, `early_stopping=False` | `balanced_accuracy` **0.9302** (CI 0.9118~0.9468), `roc_auc` 0.9789, `pr_auc` 0.9577, `cut_headroom` 0.0081, `train_val_gap` 0.0689, 4.708s | `wrong_model_family` — 남은 격차의 78%는 랭킹 축(`balanced_accuracy_at_best_cut` 0.9383, 요구 KS 0.934 vs 0.8766). 용량 부족 아님(`train_balanced_accuracy` 0.9991). xgboost로 패밀리 교체 권고 |
| 2 | `xgboost` | `n_estimators=900`, `learning_rate=0.03`, `max_depth=5`, `min_child_weight=3`, `subsample=0.8`, `colsample_bytree=0.7`, `reg_lambda=2.0`, `reg_alpha=0.5` | `balanced_accuracy` **0.9210** (CI 0.9024~0.9388), `roc_auc` 0.9770, `pr_auc` 0.9579, `cut_headroom` 0.0171, `train_val_gap` 0.0677, 4.183s | `wrong_model_family` (반복) — 두 GBDT의 랭킹 상한이 0.9383 vs 0.9381로 0.0002 차이. 구조적으로 다른 랭킹 생성기(RBF `svc`, 표준화) 권고 |
| 3 | `svc` | `kernel='rbf'`, `C=10.0` (전처리 `scale=True`) | `balanced_accuracy` **0.9188** (CI 0.8984~0.9371), `train_balanced_accuracy` 0.9666, `train_val_gap` 0.0479, 3.35s | (없음 — 평가 직후 루프 종료) |

## 최고 성능 구성

- **iteration**: 1
- **model**: `hist_gbdt`
- **hyperparams (실제 적용값)**: `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=10`, `l2_regularization=1.0`, `early_stopping=False` — `dropped_hyperparams`는 비어 있음(거부된 키 없음)
- **preprocessing (실제 적용값)**: `impute='median'`, `scale=False`, `missing_indicator=False`, `missing_count=False` (데이터에 결측이 0.0%이므로 impute는 실질적 no-op)
- **프로토콜**: stratified 3-way split, train 60% / val 20% / test 20%, seed 44 (카드 baseline과 동일 분할), `train_time_sec` 4.708

| 지표 | 검증(20%) | held-back 테스트(20%) |
|---|---|---|
| `balanced_accuracy` | **0.9302** (CI 0.911845~0.946761) | **0.9467** (CI 0.9313~0.9615) |
| `roc_auc` | 0.9789 | — |
| `pr_auc` / `average_precision` | 0.9577 | — |
| `f1` | 0.9164 | — |
| `accuracy` | 0.9348 | — |
| `precision` / `recall` | 0.9242 / 0.9088 | — |
| `specificity` | 0.951613 | — |
| `balanced_accuracy_at_best_cut` / `cut_headroom` | 0.9383 / 0.008074 | — |
| `brier` / `calibration_error` | 0.052167 / 0.047834 | — |
| `train_balanced_accuracy` / `train_val_gap` | 0.9990808 / 0.068855 | — |

검증 0.9302와 테스트 0.9467의 차이 **0.0165** 가 이 실행의 선택 편향(selection effect) 크기입니다 — 루프는 오직 검증 숫자만 보고 최고 시도를 골랐고, 테스트 20%는 첫 fit 이전에 분리되어 어떤 결정에도 쓰이지 않았습니다. 이번 실행에서는 테스트 쪽이 더 높게 나왔으므로 검증 슬라이스가 다소 비관적이었다고 읽어야 하며, 어느 쪽 숫자로도 0.967 기준선은 넘지 못했습니다(테스트 CI 상한 0.9615 < 0.967).

## 원인 분석

critic 진단은 2회 존재하며, **두 번 모두 `wrong_model_family`** 로 동일했습니다. 그 근거와 이후 실측을 함께 읽으면 병목은 명확합니다.

- **세 시도는 이 데이터로 구분되지 않습니다.** 0.9302 (CI 0.9118~0.9468), 0.9210 (CI 0.9024~0.9388), 0.9188 (CI 0.8984~0.9371) — 세 구간이 모두 겹칩니다. 즉 `hist_gbdt` → `xgboost` → `svc` 사이의 -0.0092, -0.0022 수준 변동은 개선/악화로 해석할 수 없는 리샘플 폭 안의 움직임이며, 패밀리 교체가 이 검증 슬라이스에서 아무 것도 해결하지 못했다는 사실 자체가 실질적 정보입니다.
- **부족분은 운영점(cut)이 아니라 랭킹 축에 있습니다.** iteration 1의 `cut_headroom`은 0.008074에 불과하고(`balanced_accuracy_at_best_cut` 0.9383), iteration 2도 0.017057(0.9381)입니다. 즉 임계값을 최적으로 옮겨도 최대 0.938대이며, 목표 0.967까지 남는 약 0.0287은 순전히 "행을 더 잘 순서 매기는" 문제입니다. KS로 보면 목표가 요구하는 0.934에 대해 실측 최고는 0.8766이고, baseline 기준으로도 `ks_shortfall` 0.0874, `exceeds_ranking_ceiling=true`였습니다. 게다가 `recall` 0.9088 vs `specificity` 0.9516처럼 컷은 이미 거의 균형점에 있어 `class_weight` 계열 레버의 여지도 작습니다.
- **용량 부족이 아니라 일반화 한계입니다.** iteration 1은 `train_balanced_accuracy` 0.9991, `train_val_gap` 0.068855로 2,760 학습 행을 사실상 암기했습니다. 더 강한 정규화·더 얕은 트리·서브샘플링을 적용한 iteration 2(`train_val_gap` 0.0677)와 완전히 다른 결정면인 RBF `svc`(`train_val_gap` 0.0479)도 검증 랭킹을 올리지 못했습니다. 두 GBDT의 랭킹 상한이 0.9383 vs 0.9381로 0.0002 차이였다는 critic의 관찰은, "같은 축의 다른 점"을 더 찍는 것이 무의미하다는 뜻으로 정확히 검증되었습니다.
- **예산은 제약이 아니었습니다.** 세 시도의 학습 시간은 각 3.35~4.708초로 600초 예산의 1% 미만이고, 5회 중 2회가 남은 채 정체로 종료되었습니다. 즉 남은 격차는 시간이나 탐색량이 아니라, 이 executor가 쓸 수 있는 레버 집합(피처 파생 불가, 앙상블 결합 불가, 임계값 탐색 불가, CV 불가) 안에서 랭킹을 더 올릴 수단이 소진되었다는 데 있습니다.
- **`unsupported_claims`는 세 시도 모두 비어 있고 `dropped_hyperparams`도 비어 있으므로**, 계획이 없는 기능에 의존하다 실패한 흔적은 없습니다. 숫자는 그대로 이 executor가 이 설정에서 달성한 값입니다.

참고로 목표 자체의 성격도 기록해 둘 필요가 있습니다: 0.967은 baseline(`logreg` 0.9058)의 랭킹 상한 0.9233을 이미 초과하도록 유도된 값(`derived`, `margin` 0.65)이며, `passable_margin` 0.185 기준에서도 상당히 공격적인 목표였습니다.

## 다음 단계 제안

이 데이터셋에는 별도로 기록된 `Data caveats`가 없으므로(결측 0.0%, 비수치 열 0개, 열 신뢰성 경고 없음) 아래 권고를 무효화하는 주의사항은 없습니다. 다만 executor의 CANNOT 목록에 걸리는 항목은 "지금 돌릴 계획"이 아니라 "먼저 만들어야 하는 기능"으로 분리했습니다.

1. **남은 트리 패밀리 1점만 값싸게 확인하되, 기대치를 낮추어 잡을 것 (지금 실행 가능).** `random_forest`를 `impute='median'`, `scale=False`로 한 번 돌려 `roc_auc` / `balanced_accuracy_at_best_cut`을 읽습니다. 근거는 두 GBDT의 랭킹 상한 차이가 0.0002였다는 점이므로 기대 이득은 CI 폭(±0.018 내외)보다 작을 가능성이 높습니다 — 판정은 `balanced_accuracy` 단독이 아니라 `balanced_accuracy_at_best_cut`이 0.9383을 유의하게 넘는지로만 해야 합니다. 이 한 번으로 "랭킹 상한이 패밀리와 무관하게 0.938 부근에 고정"인지가 확정됩니다.
2. **피처 변환 기능을 executor에 추가하는 것이 가장 근거 있는 다음 투자입니다 (기능 구축).** 카드상 57개 열이 거의 전부 `skew='high'`, `magnitude='sub_unit'`이고 `capital_run_length_*`만 `tens` 규모이며 `outlier_rate`가 0.17~0.22에 달하는 열들이 있습니다(`word_freq_re` 0.2176, `word_freq_free` 0.208, `char_freq_%24` 0.1763). 현재 executor는 파생·변환·상호작용·열 제거가 모두 불가하므로 `log1p`류 변환이나 빈도 열 간 비율을 시험한 적이 **없습니다**. 남은 0.0287은 랭킹 축이며 랭킹은 표현(feature)과 패밀리에서만 오므로, 패밀리는 3종을 소진한 지금 남은 것은 표현 쪽입니다.
3. **평가 프로토콜의 해상도를 먼저 올릴 것 (설정 변경).** 검증 CI 폭이 약 0.035(0.9118~0.9468)이고 검증-테스트 차가 0.0165인 상황에서, 단일 20% 슬라이스로는 0.9188~0.9302 범위의 후보를 서열화할 수 없습니다. repeated split 또는 k-fold 기반 선택(현재 executor는 CV 불가)을 도입하면 이후 시도들이 "리샘플을 설명"하는 대신 실제 개선을 측정하게 되며, 학습 시간이 4초대이므로 비용은 무시할 수 있습니다.
4. **목표값 0.967의 타당성을 재협상할 것.** 0.967은 `derived` 기준이며 baseline 랭킹 상한 0.9233을 이미 초과(`exceeds_ranking_ceiling=true`, `required_ks` 0.934 vs baseline KS 0.8466)하도록 설정되었고, 실측 최고 랭킹 상한도 0.9383에 머물렀습니다. 위 1~3을 수행해도 남는 격차가 0.0287 규모라면, 현실적 합격선은 `passable_margin` 0.185에 해당하는 수준으로 재설정하거나, 목표 자체를 랭킹 지표(`roc_auc` 0.9789 / `pr_auc` 0.9577 기준)로 다시 정의하는 편이 정직합니다. 아울러 `calibration_error` 0.0478은 확률값이 평균 4.8%p 어긋나 있음을 보여주므로, 확률을 그대로 쓰는 다운스트림이 있다면 recalibration 기능(현재 불가)이 별도 과제로 남습니다.