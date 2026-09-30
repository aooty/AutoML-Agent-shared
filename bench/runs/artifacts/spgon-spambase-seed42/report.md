# AutoML 최종 리포트 — `spambase` / balanced_accuracy

## 요약

루프는 검증 기준(threshold 0.9421)을 넘긴 구성을 찾았고(iteration 3, `hist_gbdt`, validation balanced_accuracy **0.9531**, 95% CI 0.9384~0.9676), 5회 반복을 모두 사용한 뒤 `max_iterations`로 종료했습니다. 다만 **한 번도 사용되지 않은 최종 테스트 20%에서의 점수는 0.9363 (95% CI 0.9166~0.9512)으로 기준선 0.9421을 밑돕니다** — 검증 0.9531과의 차이 +0.0168이 곧 선택 편향(selection effect)의 크기이며, 루프의 "달성" 판정은 검증 숫자만 보고 내려진 것입니다. 5회 시도의 검증 점수는 0.9406~0.9531 범위에 모두 몰려 있고 각 시도의 95% CI(폭 ≈0.029~0.033)가 서로 겹치므로, 이 데이터로는 최상위 시도들을 서로 구분할 수 없습니다. baseline(`logreg` 0.9228, CI 0.903~0.941) 대비로는 분명히 개선되었으나, 목표선 자체가 baseline의 ranking ceiling 0.9415보다 높게 설정되어 있었다는 점을 함께 읽어야 합니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=500`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=30` | balanced_accuracy **0.9530** (CI 0.9360~0.9673), roc_auc 0.9893, best_cut 0.9578, `cut_headroom` 0.0048, `train_val_gap` 0.0379 | `hyperparam` — 기준선은 넘겼으나 CI가 기준선 아래까지 닿음. 운영점은 거의 최적, 남은 여지는 ranking 축이라 판단해 강한 정규화 처방 |
| 2 | `hist_gbdt` | `learning_rate=0.03`, `max_iter=1500`, `max_leaf_nodes=15`, `min_samples_leaf=40`, `l2_regularization=5.0`, `max_bins=255`, `early_stopping=True`, `n_iter_no_change=50` | balanced_accuracy **0.9406** (CI 0.9232~0.9557), roc_auc 0.9890, best_cut 0.9556, `cut_headroom` 0.0150, `train_val_gap` 0.0376 | `data_issue` — paired Δ vs it1 = −0.0124 (CI −0.0228~−0.0026, P(improve)=0.007)로 정규화 처방이 반박됨. recall 0.9171 < specificity 0.9642 → 미시도 레버인 `class_weight` 지목 |
| 3 | `hist_gbdt` | iteration 1과 동일 + `class_weight='balanced'` | balanced_accuracy **0.9531** (CI 0.9384~0.9676), roc_auc 0.9885, best_cut 0.9545, `cut_headroom` 0.0014, `train_val_gap` 0.0357 → **최고 검증 점수** | `wrong_model_family` — paired Δ vs it1 = +0.000074 (CI −0.0068~+0.0072, P=0.497), 즉 구분 불가. `cut_headroom`이 0.0014로 소진되어 남은 여지는 ranking 축뿐 → family 교체 지시 |
| 4 | `random_forest` | `n_estimators=1000`, `min_samples_leaf=1`, `min_samples_split=2`, `bootstrap=True`, `class_weight='balanced'`, `n_jobs=-1` | balanced_accuracy **0.9531** (CI 0.9383~0.9673), roc_auc 0.9886, best_cut 0.9554, `cut_headroom` 0.0023, `train_balanced_accuracy=1.0`, `train_val_gap` 0.0469 | `wrong_model_family` — paired Δ vs it3 = +0.0000 (CI −0.0102~+0.0086). family 교체가 ranking을 해상도(0.003~0.006) 이상으로 움직이지 못함. 잎 크기 정규화 + `extra_trees` 지시 |
| 5 | `extra_trees` | `n_estimators=1500`, `min_samples_leaf=4`, `min_samples_split=2`, `bootstrap=False`, `class_weight='balanced'`, `n_jobs=-1`, `random_state=42` | balanced_accuracy **0.9498** (CI 0.9339~0.9636), roc_auc 0.9878, best_cut 0.9532, `train_val_gap` 0.0096, `calibration_error` 0.1319, `brier` 0.0590 | (마지막 반복 — critic 없음) |

모든 시도에서 `dropped_hyperparams`는 비어 있고 `unsupported_claims`도 비어 있습니다. 즉 계획한 설정은 전부 실제로 적용되었고, 사용 불가 기능에 의존해 무효가 된 시도는 없습니다.

## 최고 성능 구성

선택된 것은 **iteration 3**이며, 아래는 실행기가 실제로 만든 estimator에서 읽은 값입니다.

- **model**: `hist_gbdt`
- **hyperparams** (applied): `max_iter=500`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=30`, `class_weight='balanced'`
- **preprocessing** (applied): `impute='median'`, `scale=false`, `missing_indicator=false`, `missing_count=false`
- **프로토콜**: stratified 60/20/20, seed 42 (카드의 baseline과 동일 분할)
- **학습 시간**: 3.554초

| 구분 | balanced_accuracy | 95% CI |
|---|---|---|
| validation (선택에 사용된 슬라이스) | **0.9531** | 0.9384~0.9676 |
| 최종 테스트 20% (첫 fit 이전에 격리, 어떤 결정에도 사용되지 않음) | **0.9363** | 0.9166~0.9512 |

두 숫자의 차이 **+0.0168**은 이 루프가 검증 점수를 보고 5개 중 최고를 고른 데서 발생한 선택 편향의 크기입니다. 이 실행이 실제로 입증한 성능은 **0.9363**이고, 그 CI(0.9166~0.9512)는 목표선 0.9421을 포함하므로 테스트 슬라이스 역시 "통과"와 "기준선 근처"를 분리해 주지 못합니다.

검증 슬라이스의 그 밖의 지표(iteration 3): f1 0.9433, accuracy 0.9554, precision 0.9446, recall 0.9420, specificity 0.9642, roc_auc 0.9885, pr_auc / average_precision 0.9848, `balanced_accuracy_at_best_cut` 0.9545, `cut_headroom` 0.0014, `brier` 0.0367, `calibration_error` 0.0109, `train_balanced_accuracy` 0.9888, `train_val_gap` 0.0357.

## 원인 분석

**1) 시도들 사이의 차이는 대부분 이 데이터가 구분할 수 없는 크기다.** iteration 1(0.9530), 3(0.9531), 4(0.9531), 5(0.9498)의 95% CI는 폭이 약 0.029~0.033이고 서로 완전히 겹칩니다. 따라서 "class_weight로 올렸다", "random_forest가 동급이다", "extra_trees가 조금 낮다"는 식의 서사는 재표본(resample) 잡음을 설명하는 것에 불과합니다. 실제로 critic의 paired 검정도 같은 결론을 냈습니다: it3 vs it1 Δ=+0.000074 (P(improve)=0.497), it4 vs it3 Δ=+0.0000 (P=0.475). **이 실행에서 해상된(resolved) 차이는 단 하나** — iteration 2의 강한 정규화가 iteration 1 대비 Δ=−0.0124 (CI −0.0228~−0.0026)로 명확히 나빠진 것입니다. 즉 "capacity 과다가 문제"라는 iteration 1의 `hyperparam` 진단은 반증되었습니다: `train_balanced_accuracy`는 0.9909→0.9783으로 떨어졌지만 `train_val_gap`은 0.0379→0.0376으로 사실상 그대로였고, 일반화 이득 없이 표현력만 깎였습니다.

**2) 운영점(operating point) 축은 실질적으로 소진됐다.** `cut_headroom`은 0.0048(it1) → 0.0150(it2) → 0.0014(it3) → 0.0023(it4)로 움직였습니다. 최고 구성에서 어떤 임계값을 골라도 얻을 수 있는 최대치는 `balanced_accuracy_at_best_cut` 0.9545, 즉 현재보다 0.0014뿐입니다. recall 0.9420 vs specificity 0.9642는 여전히 recall이 낮지만, 그 격차를 더 좁혀도 되돌아올 양이 0.0014 수준이라는 뜻입니다. `class_weight` 계열 레버를 더 쓰는 것은 남은 예산의 낭비입니다.

**3) 남은 여지는 ranking 축인데, 시도한 세 family가 모두 같은 평지에 있었다.** roc_auc는 0.9878~0.9893, `balanced_accuracy_at_best_cut`는 0.9532~0.9578 범위에 갇혀 있습니다. 이 폭(≈0.0015 / ≈0.0046)은 critic이 인용한 paired 해상도 대역 0.003~0.006 안이거나 그 경계선이므로, boosting(`hist_gbdt`) → bagging(`random_forest`, `extra_trees`) 교체가 ranking을 개선했다고 말할 근거가 없습니다. 두 번 연속 내려진 `wrong_model_family` 진단은 각각 +0.0000을 벌었습니다. 다시 말해 **모델 선택은 이미 수렴했고, 성능을 제한한 것은 family도 하이퍼파라미터도 아니라 이 57개 컬럼 + 2,760개 학습행이 만들어낼 수 있는 순위 자체의 상한**입니다.

**4) 목표선 자체가 측정 해상도보다 좁게 잡혀 있었다.** goal 정의를 보면 threshold 0.9421은 baseline의 `ranking_ceiling` 0.9415를 이미 넘고(`exceeds_ranking_ceiling: true`, `ks_shortfall` 0.0012), baseline CI 상단은 0.941입니다. 반면 검증/테스트 슬라이스(각 920행)의 단일 측정 CI 폭은 ~0.030입니다. 통과선과 측정 오차가 이 비율일 때, 검증에서 0.9531을 얻고 테스트에서 0.9363을 얻는 것은 예상 가능한 결과이며, 두 슬라이스 모두 CI가 기준선을 포함합니다. 이 실행의 결론은 "기준선을 안정적으로 넘었다"가 아니라 **"baseline보다 확실히 좋고, 기준선 0.9421 부근에서 판정 불가"**입니다.

**5) 데이터 쪽에서 쓸 수 있는 레버는 애초에 없었다.** 카드의 `missing.overall_rate`는 0.0, `columns_with_missing`은 0입니다. 따라서 `median` imputation은 사실상 no-op이고, `missing_indicator` / `missing_count`는 상수 컬럼만 추가하므로 어떤 이득도 기대할 수 없습니다. 5회 모두 전처리를 고정한 것은 이 데이터에서 옳은 선택이었습니다.

## 다음 단계 제안

기록된 `Data caveats`는 없으므로 컬럼·값·분할을 무효화하는 제약은 없습니다. 아래 제안은 실행기의 CAN 목록 안에서만 구성했습니다.

1. **최우선: 판정을 반복 측정으로 옮길 것 (다중 seed 재실행).** 이 실행의 모든 판단은 폭 0.03의 단일 슬라이스 위에 있고, 검증→테스트에서 0.0168이 사라졌습니다. 실행기는 seed·분할·CV를 바꿀 수 없으므로, 같은 파이프라인(iteration 3 구성)을 **seed만 바꿔 3~5회 별도 실행**하고 테스트 점수의 분포를 보십시오. 얻는 것: "0.9421을 넘었는가"라는 질문에 단일 측정이 아니라 분포로 답할 수 있고, 지금까지 서로 구분 불가였던 it1/it3/it4의 순위도 비로소 결정 가능해집니다. 이것이 해결되지 않으면 아래 2~4번의 어떤 개선도 0.03 잡음에 묻힙니다.
2. **미시도 family인 `xgboost`로 ranking 축을 한 번 더 찍어볼 것.** 근거: 남은 여지는 ranking 축뿐(`cut_headroom` 0.0014)이고, 실측된 family 교체 폭은 0.0022~0.0077로 "작지만 0은 아닌" 구간입니다. 단 `eval_set` / `early_stopping_rounds` / `callbacks`는 이 실행기에서 차단되므로 고정 `n_estimators`와 `max_depth`·`subsample`·`colsample_bytree`·`reg_lambda`만으로 구성하고, 불균형은 `scale_pos_weight`로 다루십시오. 판정 기준은 balanced_accuracy가 아니라 **roc_auc / `balanced_accuracy_at_best_cut`가 현행 0.9885 / 0.9545를 해상도 대역 0.003~0.006 이상 넘는지**로 두어야 합니다(그렇지 않으면 iteration 4·5의 +0.0000을 반복합니다).
3. **`class_weight`·임계값 계열 시도는 중단할 것.** `cut_headroom` 0.0014가 상한이므로, 어떤 가중치 맵이나 임계값 조정도 최고 구성에서 0.0014 이상을 줄 수 없습니다. 이 축에 예산을 쓰는 대신 2번(ranking) 또는 4번(입력)에 배분하십시오.
4. **더 큰 개선을 원한다면 실행기 밖에서 입력 표현을 바꿔야 한다.** 실행기는 파생·조합·인코딩·컬럼 제거를 하지 않으므로(미결측 데이터라 missingness 컬럼도 무의미), 57개 원컬럼의 순위 상한이 그대로 천장입니다. 로그/랭크 변환된 빈도 컬럼이나 `capital_run_length_*` 파생값이 필요하다면 **카드(데이터셋 자체)에 컬럼을 추가한 뒤 재실행**하는 것이 유일한 경로입니다. 카드의 프로파일에 따르면 거의 모든 word/char 빈도 컬럼이 `skew: high`이고 `char_freq_%21`·`char_freq_%24`·`capital_run_length_*`는 outlier_rate가 6~22%이므로, 이 변환은 트리 계열보다 `logreg` 같은 스케일 민감 family에서 더 큰 차이를 만들 가능성이 있습니다 — 다만 그 크기는 이 실행에서 측정된 바 없으므로 기대치를 숫자로 약속할 수는 없습니다.