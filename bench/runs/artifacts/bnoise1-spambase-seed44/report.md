# AutoML 실행 최종 보고서 — `spambase` / `balanced_accuracy`

## 요약

**목표는 달성되지 못했습니다.** 목표 임계값은 `balanced_accuracy` 0.967이었으나, 5회 예산 중 3회를 사용한 뒤 연속 미개선(정체)으로 조기 종료되었고, 최고 검증 점수는 iteration 1의 `hist_gbdt`가 기록한 0.9302 (95% CI 0.9118~0.9468)로 임계값보다 0.0368 낮았습니다. 한 번도 사용되지 않은 held-back 테스트 20%에서 같은 모델은 0.9459 (95% CI 0.9295~0.9608)를 기록해, 검증 점수보다 오히려 높았지만 여전히 임계값 0.967에는 도달하지 못했습니다(CI 상단 0.9608). 세 번의 시도(`hist_gbdt`, `xgboost`, `svc`)는 서로 신뢰구간이 크게 겹쳐 이 데이터로는 구분되지 않으며, 부족분의 대부분은 결정 임계점이 아니라 랭킹 축에 있습니다(`cut_headroom` 최대 0.0094).

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=500`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=10`, `l2_regularization=1.0`, `early_stopping=False` | `balanced_accuracy` **0.9302** (CI 0.9118~0.9468), `roc_auc` 0.9788, `balanced_accuracy_at_best_cut` 0.9369, `cut_headroom` 0.006674, `train_val_gap` 0.068855 | `overfitting` — `train_balanced_accuracy` 0.9991로 학습 데이터를 암기, 남은 격차 0.0368 중 82%는 랭킹 축. 용량 축소·정규화 강화(`hist_gbdt`, `max_leaf_nodes=15`, `min_samples_leaf=30`, `l2_regularization=10.0`) 처방 |
| 2 | `xgboost` | `n_estimators=600`, `learning_rate=0.03`, `max_depth=4`, `reg_lambda=10.0`, `subsample=0.8`, `colsample_bytree=0.7`, `min_child_weight=3` | `balanced_accuracy` **0.9228** (CI 0.9054~0.9402), `roc_auc` 0.9763, `balanced_accuracy_at_best_cut` 0.9302, `cut_headroom` 0.007365, `train_val_gap` 0.050044 | `wrong_model_family` — iteration 1과의 차이 -0.0074 (CI -0.0150~+0.0011, P(better) 0.045)는 측정으로 구분 불가. 분산은 줄었으나(`train_val_gap` 0.0689→0.0500) 랭킹은 따라오지 않음. 부스팅 밖의 랭킹 생성기(`svc` rbf, 표준화) 처방 |
| 3 | `svc` | `kernel=rbf`, `C=10.0`, `probability=True` | `balanced_accuracy` **0.9188** (CI 0.8984~0.9371), `roc_auc` 0.9677, `balanced_accuracy_at_best_cut` 0.9282, `cut_headroom` 0.009434, `calibration_error` 0.019876 | (critic 미실행 — 평가 직후 루프 종료) |

세 시도 모두 `status: ok`, `dropped_hyperparams` 없음, `unsupported_claims` 없음, 학습 시간 3.9~5.4초로 자원 제약은 전혀 문제가 아니었습니다.

## 최고 성능 구성

- **model**: `hist_gbdt` (iteration 1)
- **hyperparams** (해당 attempt에 적용된 값): `max_iter=500`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=10`, `l2_regularization=1.0`, `early_stopping=False`
- **preprocessing** (해당 attempt에 적용된 값): `impute=median`, `scale=false`, `missing_indicator=false`, `missing_count=false`
- **프로토콜**: stratified 3-way split, train 60% / validation 20% / test 20%, seed 44 (카드의 baseline과 동일 분할)

| 측정 | balanced_accuracy | 비고 |
|---|---|---|
| validation 20% (선택에 사용) | **0.9302** (95% CI 0.911845~0.946761) | 루프가 이 숫자로 최고 시도를 골랐습니다 |
| held-back test 20% (1회 채점) | **0.9459** (95% CI 0.9295~0.9608) | 첫 학습 전에 분리되어 어떤 결정에도 쓰이지 않은 행 |

두 숫자의 차이(0.0156)가 **선택 편향의 크기**입니다. 이 실행이 실제로 입증하는 것은 test 쪽 0.9459이며, 검증 점수 0.9302는 test CI(0.9295~0.9608) 안에 들어가므로 이 행들만으로는 차이가 0과 구분되지 않습니다. 어느 쪽으로 읽어도 목표 0.967에는 미달합니다.

validation에서의 부수 지표: `accuracy` 0.9348, `f1` 0.9164, `precision` 0.9242, `recall` 0.9088, `specificity` 0.951613, `roc_auc` 0.9788, `pr_auc` / `average_precision` 0.9577, `balanced_accuracy_at_best_cut` 0.9369, `cut_headroom` 0.006674, `brier` 0.052615, `calibration_error` 0.048631, `train_balanced_accuracy` 0.9991, `train_val_gap` 0.068855, `train_time_sec` 5.429.

baseline(`logreg` + median impute + standard scale)은 같은 분할에서 `balanced_accuracy` 0.9058 (CI 0.8823~0.9239)이었습니다. 최고 시도의 검증 CI는 baseline CI와 겹치므로, 검증 슬라이스만으로는 "baseline보다 확실히 낫다"고 말할 수 없고, held-back test 0.9459가 baseline 대비 개선을 지지하는 유일한 독립 증거입니다.

## 원인 분석

critic 판정은 2회 존재하며(`overfitting`, `wrong_model_family`), 그 사이에서 읽을 수 있는 패턴은 다음과 같습니다.

1. **부족분은 결정 임계점이 아니라 랭킹 축에 있습니다.** 세 시도의 `cut_headroom`은 0.006674 / 0.007365 / 0.009434로, 남은 격차(각각 0.0368 / 0.0442 / 0.0482)의 17~20%에 불과합니다. 즉 `class_weight`나 `scale_pos_weight` 같은 운영점 레버를 완벽히 조정해도 임계값에 도달할 수 없습니다. `recall`(0.9088)과 `specificity`(0.9516)의 0.043 불균형을 전부 교정해도 상한이 그 0.0067입니다. critic 두 판정 모두 같은 결론에 도달했고, 이 부분은 숫자로 확정적입니다.
2. **목표 임계값 자체가 관측된 랭킹 한계 위에 있습니다.** 목표 정의는 `exceeds_ranking_ceiling: true`, `required_ks: 0.934`, `ks_shortfall: 0.0874`를 명시하고 있습니다. 실제로 세 시도의 `balanced_accuracy_at_best_cut`은 0.9369 / 0.9302 / 0.9282 — **어떤 임계점을 골라도 최고가 0.9369**이므로 0.967까지 0.0301이 남습니다. 이 0.0301은 문서화된 랭킹 축 레버 크기(같은 family 내 재튜닝 `roc_auc` 0.0032, family 교체 0.0022~0.0077)보다 한 자릿수 크며, 이 executor가 쓸 수 있는 레버 하나로 메울 수 있는 폭이 아닙니다.
3. **시도 간 `balanced_accuracy` 움직임은 이 데이터로 구분되지 않습니다.** 0.9302 / 0.9228 / 0.9188의 CI(0.9118~0.9468, 0.9054~0.9402, 0.8984~0.9371)는 전부 서로를 포함합니다. iteration 1→2의 paired Δ는 -0.0074 (CI -0.0150~+0.0011)로 critic 자신도 "측정이 분해하지 못하는 움직임"이라고 기록했습니다. 따라서 "정규화가 성능을 떨어뜨렸다"거나 "svc가 더 나빴다"는 서술은 이 슬라이스에서 리샘플을 설명하는 것에 가깝습니다. 구간 폭보다 큰 차이로 말할 수 있는 것은 **학습-검증 격차의 축소**(`train_balanced_accuracy` 0.9991→0.9729→0.9666, `train_val_gap` 0.0689→0.0500→0.0479)와 **그 축소가 검증 랭킹으로 옮겨지지 않았다는 사실**(`roc_auc` 0.9788→0.9763→0.9677, `balanced_accuracy_at_best_cut` 0.9369→0.9302→0.9282)입니다. 즉 iteration 1의 암기는 실재했지만, 그것이 성능의 병목은 아니었습니다.
4. **처방과 실행이 한 번 어긋나 귀인이 흐려졌습니다.** iteration 1의 critic은 `concrete_changes`에서 `hist_gbdt`를 유지한 채 용량만 줄이라고(같은 family, 전처리·`class_weight` 불변) 지정했는데, 실제 iteration 2는 `xgboost`로 실행되었습니다. 결과적으로 "정규화 강화"와 "family 교체"가 같은 시도에 섞여, 어느 쪽이 -0.0074(구분 불가 범위)를 만들었는지 분리할 수 없습니다. iteration 2의 critic이 제시한 대안 중 하나였던 `random_forest`(완전 성장 트리)는 끝까지 시도되지 않았고, 예산 2회가 미사용으로 남았습니다.
5. **부수적으로**, iteration 1의 확률값은 과신 상태입니다(`brier` 0.052615, `calibration_error` 0.048631). 가장 잘 보정된 시도는 iteration 3(`calibration_error` 0.019876)이었지만 랭킹은 가장 나빴습니다. 이 실행 환경에는 재보정 레버가 없으므로 이는 진단으로만 남습니다.

정리하면, 성능을 제한한 것은 과적합이나 잘못된 family가 아니라 **이 특성 집합(카드 그대로의 57개 수치 컬럼) + 이 executor의 레버 조합이 만들 수 있는 랭킹의 한계**이며, 목표 0.967은 그 한계 위에 설정되어 있었습니다. `Data caveats`에 기록된 주의사항은 없으므로, 데이터 신뢰성 문제로 돌릴 근거도 없습니다.

## 다음 단계 제안

1. **임계값의 타당성을 먼저 재확정할 것.** 목표 메타데이터가 `exceeds_ranking_ceiling: true`, `required_ks: 0.934`(달성 최고 KS 0.8738), `ks_shortfall: 0.0874`를 이미 명시하고, 실측 `balanced_accuracy_at_best_cut` 최고치도 0.9369입니다. 추가 예산을 쓰기 전에 0.967이 이 특성 집합에서 달성 가능한 값인지, 아니면 held-back 0.9459(CI 0.9295~0.9608) 수준으로 목표를 재설정할지를 결정해야 합니다. 이 결정 없이 iteration을 더 쓰면 0.003 수준의 레버로 0.03을 메우는 시도를 반복하게 됩니다.
2. **남은 랭킹 축 후보 중 아직 측정되지 않은 family를 1회로 소진할 것: `random_forest`(깊이 제한 없음, `n_estimators` 충분히 크게, `impute=median`, `scale=false`).** iteration 2의 critic이 대안으로 지목했으나 실행되지 않은 유일한 지점이며, 부스팅 2점 + rbf-SVM 1점으로 덮인 best-cut 폭은 0.9282~0.9369(0.0087)뿐이어서 family 교체의 실제 폭이 여기서는 아직 측정되지 않았습니다. 단, 문서화된 family 교체 효과가 `roc_auc` 0.0022~0.0077임을 감안하면 이것만으로 0.967에 도달할 기대는 두지 말아야 합니다.
3. **특성 쪽에 손을 대려면 executor 밖에서 해야 함을 계획에 반영할 것.** 이 executor는 파생·변환·상호작용·컬럼 삭제를 하지 않으며(`missing_indicator`/`missing_count`가 유일한 예외이고, 이 데이터는 `overall_rate: 0.0`이라 둘 다 무의미), 카드의 57개 컬럼은 거의 전부 `skew: high`의 sub_unit 빈도값입니다. 로그 변환·빈도 비율 등 랭킹을 올릴 가능성이 있는 표현은 **데이터셋 카드에 컬럼으로 추가되어 들어와야** 하며, 이는 플랜이 아니라 상류 데이터 준비 작업입니다. 이것이 열리면 지금 0.0301로 막혀 있는 랭킹 축에 처음으로 새 여지가 생깁니다.
4. **측정 해상도를 먼저 올릴 것.** 검증 20%(약 920행)에서 `balanced_accuracy` CI 폭은 ±0.017~0.019이고, 세 시도가 전부 그 안에서 겹쳤습니다. 현재 프로토콜(단일 20% 검증, CV 불가, seed 고정)에서는 0.003~0.008급 레버를 판정할 수 없으므로, 추가 탐색을 하기로 한다면 반복 분할/여러 seed에 걸친 재채점을 루프 설정 수준에서 도입해야 합니다. 그러지 않으면 다음 5회 예산도 이번처럼 "구분되지 않는 세 점"으로 끝납니다.
5. **운영점 레버(`class_weight='balanced'`, `scale_pos_weight`)와 보정 관련 작업에는 예산을 쓰지 말 것.** `cut_headroom`이 0.0067~0.0094로 이미 최적 컷에 근접해 있고 남은 격차의 20% 미만이며, 이 환경에는 임계값 탐색·확률 재보정 레버가 아예 없습니다(불균형비 1.54도 경미).