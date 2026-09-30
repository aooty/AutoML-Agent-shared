# AutoML 실행 보고서 — bank-marketing (balanced_accuracy)

## 요약

목표는 달성하지 못했습니다. 목표 임계값은 `balanced_accuracy` 0.8811이었고, 5회 시도 중 최고 검증 점수는 iteration 3의 `xgboost` 0.8668 (95% CI 0.8557~0.8771)로, 신뢰구간 전체가 임계값 아래에 있어 미달은 실측입니다. 같은 모델을 루프 내내 한 번도 쓰이지 않은 최종 테스트 20%에서 한 번 채점한 결과는 0.8681 (95% CI 0.8573~0.8786)로, 검증 점수와 사실상 같습니다. 다만 baseline (`logreg`, balanced_accuracy 0.6603)과 baseline의 랭킹 상한 0.8382는 크게 넘어섰습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `class_weight='balanced'`, `early_stopping=False` / `impute=median`, `scale=False` | balanced_accuracy 0.8586 (CI 0.8456~0.8699), roc_auc 0.9328, best_cut 0.8702, cut_headroom 0.0116, train_val_gap 0.0786 | `wrong_model_family` — cut_headroom가 남은 격차의 절반뿐이므로 랭킹 축을 움직여야 한다며 `xgboost` + `impute: none` 처방 |
| 2 | `xgboost` | `n_estimators=700`, `learning_rate=0.04`, `max_depth=6`, `subsample=0.8`, `reg_lambda=3.0`, `scale_pos_weight=8.0` / `impute=none`, `scale=False` | balanced_accuracy 0.8557 (CI 0.8435~0.8679), roc_auc 0.9338, best_cut 0.8721, cut_headroom 0.0164, train_val_gap 0.0959 | `data_issue` — iteration 1 대비 paired Δ −0.0028 (CI −0.0099~+0.0046)로 구분 불가. 남은 랭킹은 열이 담은 정보에서 와야 한다며 `impute: median` + `missing_count` 처방 |
| 3 | `xgboost` | `n_estimators=600`, `learning_rate=0.04`, `max_depth=5`, `subsample=0.8`, `reg_lambda=5.0`, `scale_pos_weight=10.0` / `impute=median`, `scale=False`, `missing_indicator=True`, `missing_count=True` | **balanced_accuracy 0.8668 (CI 0.8557~0.8771)**, roc_auc 0.9353, pr_auc 0.6324, best_cut 0.8741, cut_headroom 0.0073, train_val_gap 0.0509 | `data_issue` — iteration 1 대비 paired Δ +0.0082 (CI +0.0012~+0.0154)이나 세 레버가 동시에 움직여 귀속 불가. `missing_indicator`만 떼어내 단독 가격 측정 처방 |
| 4 | `xgboost` | iteration 3과 동일 / `impute=median`, `scale=False`, `missing_indicator=False`, `missing_count=True` | balanced_accuracy 0.8668 (CI 0.8557~0.8771) — iteration 3과 비트 단위로 동일 (roc_auc 0.9352519, brier 0.098841, best_cut 0.8741) | `wrong_model_family` — paired Δ가 정확히 +0.0000. 결측 관련 레버는 0으로 확정되었고 operating point는 `scale_pos_weight` 8~10 사이로 갇혔으므로, 마지막 반복은 GBDT 밖 계열(`random_forest`)에 쓰라고 처방 |
| 5 | `random_forest` | `n_estimators=800`, `min_samples_leaf=2`, `class_weight='balanced_subsample'` / `impute=median`, `scale=False`, `missing_indicator=True`, `missing_count=True` | balanced_accuracy 0.7864 (CI 0.7700~0.8005), roc_auc 0.9299, best_cut 0.8673, cut_headroom 0.0809, train_val_gap 0.1921 | (루프 종료로 critic 미실행) |

## 최고 성능 구성

iteration 3, `xgboost`.

적용된 하이퍼파라미터 (`hyperparams`, `dropped_hyperparams`는 비어 있음):

```
n_estimators      = 600
learning_rate     = 0.04
max_depth         = 5
subsample         = 0.8
reg_lambda        = 5.0
scale_pos_weight  = 10.0
```

실제로 만들어진 전처리 (`preprocessing`):

```
impute            = median
scale             = false
missing_indicator = true
missing_count     = true
```

프로토콜은 seed 44의 계층화 3분할(train 60% / validation 20% / test 20%)이며, 카드의 baseline과 동일한 분할입니다. 학습 시간 8.25초.

| 지표 | 검증 (20%) | 최종 테스트 (20%, 루프 미사용) |
|---|---|---|
| balanced_accuracy | 0.8668 (95% CI 0.8557~0.8771) | **0.8681 (95% CI 0.8573~0.8786)** |
| recall | 0.8752 | — |
| specificity | 0.8583 | — |
| roc_auc | 0.9353 | — |
| pr_auc / average_precision | 0.6324 | — |
| f1 | 0.5945 | — |
| accuracy | 0.8603 | — |
| precision | 0.4502 | — |
| brier / calibration_error | 0.0988 / 0.1261 | — |
| balanced_accuracy_at_best_cut / cut_headroom | 0.8741 / 0.0073 | — |
| train_balanced_accuracy / train_val_gap | 0.9177 / 0.0509 | — |

테스트와 검증의 차이는 −0.0013(테스트가 더 높음)이며, 이 차이가 선택 편향의 크기입니다 — 루프는 검증 점수만 보고 최고 시도를 골랐습니다. 검증 점수가 테스트 CI 안에 들어오므로 이 행들만으로는 그 편향이 0과 구분되지 않습니다. 어느 쪽 숫자로 보아도 임계값 0.8811에는 미달입니다. iteration 4는 검증에서 iteration 3과 비트 단위로 같은 결과를 냈으므로, `missing_indicator` 없이도 동일한 구성으로 재현할 수 있습니다.

## 원인 분석

critic 진단은 4회 존재하고, 그 패턴은 일관됩니다: **격차는 operating point가 아니라 랭킹 축에 있었고, 이 executor에게 랭킹 축 레버가 남아 있지 않았습니다.**

- **operating point는 일찌감치 소진되었습니다.** iteration 1의 `cut_headroom`은 0.0116, iteration 3/4에서는 0.0073으로 줄었습니다. 즉 완벽한 임계값을 골라도 최고 도달점은 `balanced_accuracy_at_best_cut` 0.8741이며, 임계값 0.8811에 0.0070 부족합니다. 게다가 `scale_pos_weight=8.0`에서는 recall 0.8185 < specificity 0.8929, `=10.0`에서는 recall 0.8752 > specificity 0.8583이므로 최적점이 8~10 사이에 갇혀 있고, 그 안에서 벌 수 있는 양은 0.0073의 일부입니다.
- **모델 계열 교체는 랭킹을 거의 움직이지 못했습니다.** iteration 1 → 2의 paired Δ는 −0.0028 (CI −0.0099~+0.0046)로 이 데이터가 구분하지 못하는 차이입니다(두 시도의 CI 0.8456~0.8699 대 0.8435~0.8679도 크게 겹칩니다). 시도된 세 계열의 `balanced_accuracy_at_best_cut`는 0.8673(`random_forest`) ~ 0.8741(`xgboost`)로 폭 0.0068에 불과했고, 이는 요구되는 KS 0.7622(최고 달성 0.7482)를 채우기에 부족합니다. iteration 5의 `random_forest`는 CI(0.7700~0.8005)가 다른 시도들과 겹치지 않는 유일한 큰 차이인데, 방향이 아래쪽입니다 — train_val_gap 0.1921, cut_headroom 0.0809으로 과적합과 극단적으로 어긋난 작동점이 동시에 나타났습니다.
- **결측 관련 열은 값이 0으로 실측되었습니다.** iteration 3 → 4는 `missing_indicator`만 뗀 유일한 전처리 단독 전환이었고, paired Δ가 정확히 +0.0000, 모든 지표가 비트 단위로 동일했습니다. iteration 1 → 3의 +0.0082 (CI +0.0012~+0.0154)는 해상도를 넘는 차이이지만 `impute`, `missing_count`, `missing_indicator`, 하이퍼파라미터가 함께 움직였으므로 어느 레버에도 귀속되지 않습니다. 카드가 지목한 'unknown' 레짐(V16 81.8%, V9 28.8%, V4 4.1%, V2 0.6%)은 원본 파일에서 결측이 아니라 **문자열 범주값**이므로 `missing_indicator`/`missing_count`가 애초에 볼 수 없는 정보였습니다 — 이것이 결측 레버가 정확히 0이었던 구조적 이유입니다.
- **용량(capacity)은 제약이 아니었습니다.** train_val_gap이 0.0959(iteration 2) → 0.0509(iteration 3)로 줄면서 점수는 올랐으므로, 더 깊은/더 긴 부스팅이 답이 아닙니다.
- **critic 진단의 순환.** `wrong_model_family` → `data_issue` → `data_issue` → `wrong_model_family` 순서로 두 진단을 왕복했는데, 그 사이 각각의 처방이 실제로 산 것은 −0.0028(구분 불가)과 +0.0000(정확히 0)이었습니다. 즉 진단 자체가 틀렸다기보다, executor가 쓸 수 있는 레버 집합(추정기 교체, 하이퍼파라미터, 클래스 가중치, 두 개의 결측 열)이 남은 0.0070의 랭킹 격차를 메우기에 구조적으로 부족했습니다.
- **unsupported_claims.** iteration 1, 2, 5의 계획 텍스트에서 `feature_engineering` 문자열이 감지되었습니다. 이는 부분 문자열 검사이고, 해당 시도들이 실제로 사용한 전처리는 기록된 `preprocessing` 블록 그대로(파생 열 없음)입니다. 따라서 "특징 공학으로 잃은 점수"라고 볼 근거는 없고, 다만 **파생 특징이라는 축은 이 런에서 한 번도 시험되지 않았다**는 사실만 남습니다. iteration 3, 4는 unsupported_claims가 비어 있습니다.
- 참고로 확률 자체는 신뢰하기 어렵습니다: 최고 구성의 `calibration_error`는 0.1261, `brier` 0.0988입니다. 랭킹 지표(roc_auc 0.9353)는 이 영향을 받지 않지만, 확률값을 그대로 업무에 쓰려면 별도 보정이 필요합니다(이 executor에는 보정 레버가 없습니다).

## 다음 단계 제안

1. **'unknown' 범주를 데이터 카드 단계에서 결측으로 변환한 파일을 새로 만들고 다시 돌리기.** 카드의 네 caveat(V2 0.6%, V4 4.1%, V9 28.8%, V16 81.8%)은 이 값들이 변환되지 않았고 따라서 해당 열의 통계와 baseline 점수가 'unknown'을 실측치로 계산한 결과라고 명시합니다. 현재 파일에서는 결측률이 0이므로 `missing_indicator`/`missing_count`가 물리적으로 아무 정보도 담지 못했고(iteration 3→4의 Δ = +0.0000이 정확히 그 증거), 랭킹 축의 유일한 데이터 레버가 봉인된 상태였습니다. 변환은 executor가 할 수 없는 일(재인코딩·열 삭제 불가)이므로 카드/입력 파일 쪽에서 처리해야 하며, 이것이 풀리면 결측 레버를 처음으로 실제로 가격 매길 수 있습니다. 단, caveat가 경고하듯 V16처럼 81.8%가 'unknown'인 열의 지표는 거의 상수에 가까워질 수 있고, 이 열의 결측이 "대상의 상태"가 아니라 "기록 레짐"을 나타낼 위험이 있으므로, 변환 후 얻은 이득은 무작위 분할 점수만으로 확정하지 말고 별도 확인이 필요합니다.
2. **파생 특징 축을 executor 기능으로 열기.** 남은 격차는 최적 임계값을 써도 0.0070인 **랭킹** 격차이고(best_cut 0.8741 vs 임계값 0.8811), 시도된 세 계열의 best_cut 폭은 0.0068뿐이었습니다. 즉 같은 열 집합에서 추정기를 더 바꿔봐야 도달 근거가 없습니다. 반면 파생·상호작용·재인코딩은 이 런에서 한 번도 시험되지 않은 유일한 랭킹 축입니다(계획 텍스트에서 언급만 되고 실행되지 않았음). 특히 카드에서 `target_corr='strong'`인 V12와 skew가 높은 V6/V13/V14/V15의 변환·조합이 첫 후보입니다.
3. **랭킹 축을 건드리지 않는 미세 조정은 마지막에 한 번만.** `scale_pos_weight`를 8.0과 10.0 사이(예: 9.0)로 보간하면 recall 0.8752 / specificity 0.8583의 불균형을 줄일 수 있지만, `cut_headroom`이 0.0073이므로 상한이 명확합니다. 임계값 도달용 수단이 아니라, 랭킹이 개선된 뒤 작동점을 마무리하는 수단으로만 쓰는 것이 맞습니다.
4. **확률을 쓸 계획이면 보정 단계를 파이프라인에 추가.** 최고 구성의 `calibration_error` 0.1261은 예측 확률이 평균 12.6%p 어긋난다는 뜻입니다. 현재 executor에는 보정 레버가 없으므로, 이는 구성 조합으로 해결할 수 없고 기능으로 추가해야 하는 항목입니다. balanced_accuracy 자체는 개선되지 않지만, 이 모델을 의사결정에 쓰는 쪽에서는 필수입니다.