# 최종 리포트 — bank-marketing / balanced_accuracy

## 요약

목표는 달성되었습니다. 1회 시도(예산 5회 중 1회)에서 `hist_gbdt`가 검증 `balanced_accuracy` **0.8780** (95% CI 0.8671~0.8881)을 기록해 임계값 0.7465와 baseline `logreg` 0.6620을 모두 큰 폭으로 넘겼습니다. 루프 종료 후 한 번만 채점한 held-back 테스트 20%에서도 **0.8656** (95% CI 0.8547~0.8755)로, 임계값을 여유 있게 상회합니다. 첫 시도에서 목표를 통과했기 때문에 critic 진단이나 후속 반복은 생성되지 않았습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=30`, `class_weight={"0":1.0,"1":6.0}` | status `ok` — `balanced_accuracy=0.8780` (CI 0.8671~0.8881), `roc_auc=0.9417`, `pr_auc=0.6457`, `recall=0.8837` / `specificity=0.8722`, `cut_headroom=0.004906`, `train_val_gap=0.029121`, 학습 7.032초 | 없음 (목표 달성으로 루프 종료, `critic: null`) |

`dropped_hyperparams`는 비어 있어, 계획된 하이퍼파라미터는 모두 그대로 적용되었습니다.

## 최고 성능 구성

- **model**: `hist_gbdt`
- **hyperparams** (해당 attempt의 적용값): `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=30`, `class_weight={"0": 1.0, "1": 6.0}`
- **preprocessing** (해당 attempt의 적용값): `impute=median`, `scale=false`, `missing_indicator=false`, `missing_count=false`
- **프로토콜**: stratified 3-way split, seed 42, train 60% / val 20% / test 20% (카드 baseline과 동일한 split)

| 지표 | 검증 (val 20%) | held-back 테스트 (test 20%) |
|---|---|---|
| `balanced_accuracy` | **0.8780** (95% CI 0.8671~0.8881) | **0.8656** (95% CI 0.8547~0.8755) |

검증과 테스트의 차이 **+0.0124**(검증이 더 높음)가 이 실행의 **선택 편향 크기**입니다. 루프는 검증 숫자만 보고 최종 구성을 골랐으므로, 이 모델에 대해 실제로 입증된 값은 테스트의 0.8656 쪽입니다. 두 구간이 부분적으로 겹치는 만큼 차이 자체가 큰 편은 아니며, 어느 값을 쓰더라도 임계값 0.7465는 넘습니다.

기타 검증 지표(참고): `accuracy=0.8736`, `f1=0.6206`, `precision=0.4783`, `recall=0.8837`, `specificity=0.872244`, `roc_auc=0.9417`, `pr_auc / average_precision=0.6457`, `balanced_accuracy_at_best_cut=0.8829`, `cut_headroom=0.004906`, `brier=0.088171`, `calibration_error=0.112497`, `train_balanced_accuracy=0.9071`, `train_val_gap=0.029121`.

## 원인 분석

- **시도가 1회이므로 시도 간 변화로 서사를 만들 근거가 없습니다.** 비교 가능한 기준선은 카드의 `logreg` 하나이며, 그 차이는 구간 폭보다 훨씬 큽니다: baseline `balanced_accuracy` 0.6620 (CI 0.6496~0.6765) 대 0.8780 (CI 0.8671~0.8881)로 구간이 전혀 겹치지 않습니다.
- **이번 향상은 두 축 모두에서 왔고, 그중 랭킹 축의 기여가 실질적입니다.** 랭킹 축: `roc_auc` 0.9101 → 0.9417, `pr_auc` 0.5557 → 0.6457, `balanced_accuracy_at_best_cut` 0.84 → 0.8829. 즉 선형 모델을 트리 앙상블로 교체한 것이 순서 자체를 개선했습니다. 운영점 축: baseline은 `recall=0.3478`로 소수 클래스를 거의 놓치고 있었고, `class_weight={"0":1,"1":6}`이 컷을 `recall=0.8837` / `specificity=0.8722`의 균형점으로 옮겼습니다.
- **남은 여지는 운영점이 아니라 랭킹에 있습니다.** `cut_headroom=0.004906`은 현재 기본 `predict()` 컷이 이 랭킹에서 얻을 수 있는 최선(`balanced_accuracy_at_best_cut=0.8829`)에 거의 붙어 있다는 뜻입니다. 따라서 `class_weight` 비율을 더 만지는 것은 0.005 이내의 움직임이며, 이는 검증 구간 폭(약 ±0.010)보다 작아 이 데이터에서는 개선으로 보고할 수 없는 크기입니다.
- **과적합은 문제가 아닙니다.** `train_val_gap=0.029121`, `train_balanced_accuracy=0.9071` 대 검증 0.8780으로, `early_stopping=True`(+`n_iter_no_change=30`)와 `l2_regularization=1.0`이 400회 부스팅을 무리 없이 억제했습니다. 학습 시간 7.032초로 600초 예산도 여유롭습니다.
- **확률값은 그대로 쓰면 안 됩니다.** `calibration_error=0.112497`, `brier=0.088171`. `class_weight` 1:6은 컷을 옮기는 대가로 양성 확률을 체계적으로 부풀립니다. 이 실행에는 재보정 레버가 없으므로(측정만 되고 교정되지 않음), 순위 기반 지표는 신뢰해도 "예측 확률 = 가입 확률"로 읽는 소비처는 별도 보정이 필요합니다.
- **`unsupported_claims: ["feature_engineering"]`은 이번 결과의 원인이 아닙니다.** iteration 1의 plan 본문을 읽으면 파생 변수 생성을 실행 계획에 넣은 곳이 없고("median impute, no scaling, no missingness columns"로 명시), "the model family or the features"라는 진단 문구에서 단어가 걸린 substring 오탐입니다. 따라서 이 attempt가 feature engineering 때문에 무언가를 잃은 것은 아니며, 그 능력은 아래 "다음 단계"의 신설 항목으로 다룹니다.
- **한계로 남는 것은 데이터 표현 쪽입니다.** 카드 caveats에 따르면 `V16`의 81.8%, `V9`의 28.8%, `V4`의 4.1%, `V2`의 0.6%가 결측 표기로 흔히 쓰이는 `'unknown'` 문자열이며, 변환 없이 실제 범주값으로 학습되었습니다. 즉 이 실행의 `missing.overall_rate=0.0`은 "결측이 없다"가 아니라 "결측이 범주 레벨로 위장되어 있다"는 뜻이고, `impute`/`missing_indicator` 계열 레버가 애초에 작동할 표면이 없었습니다.

## 다음 단계 제안

목표는 이미 달성됐으므로, 아래는 (a) 신뢰성 확보와 (b) 랭킹 축의 추가 여지 탐색 순서입니다.

1. **`V12`가 예측 시점에 실제로 관측 가능한지 확인하는 것을 최우선으로.** 카드에서 `V12`만 `target_corr: strong`이고, 나머지 수치형은 `none`~`weak`입니다. 이 한 열이 랭킹의 상당 부분을 지고 있을 가능성이 높습니다. 만약 이 값이 결과가 확정된 뒤에야 알 수 있는 종류의 측정이라면 0.8656은 배포 성능이 아니라 누출된 성능이므로, 모델 개선보다 이 확인이 먼저입니다. 확인 결과에 따라 "이 열을 카드에서 제외한 재실행"이 필요할 수 있고(executor는 열을 드롭할 수 없으므로 카드 단계에서 빼야 합니다), 그 재실행이 배포 가능한 상한을 처음으로 알려줍니다.
2. **`'unknown'` 문자열을 상류(카드 생성 단계)에서 진짜 결측으로 변환한 뒤, `hist_gbdt`에 `impute: none`으로 한 번 재측정.** 이것이 caveats 자체의 해소책입니다. 지금은 `V16`(81.8%)·`V9`(28.8%)의 `'unknown'`이 하나의 정상 레벨로 들어가 있어 트리가 "측정되지 않음"과 "특정 값"을 구분할 수 없습니다. 변환 후에는 `impute: none`으로 트리가 NaN 분기를 직접 학습할 수 있고, 이때 `missing_indicator`는 중복이므로 함께 켜지 마십시오(동일 예측이 재현된 사례가 보고되어 있습니다). 다만 주의: `V16`의 `'unknown'`이 "이전 캠페인 접촉 이력 자체가 없음"처럼 **기록 체제**를 나타내는 경우, stratified random split은 그 정보를 이득으로 채점해 줍니다. 그러므로 이 변환은 "얼마를 벌어준다"는 기대치 없이, 어떤 방향으로 얼마나 움직이는지 읽는 목적으로만 한 번 쓰고, 값이 나오면 시간 기반(연속) 분할로도 재확인할 것을 권합니다. 이 caveat가 해소되기 전에는 결측 관련 레버의 크기를 인용하지 마십시오 — 이 파일에는 안정적인 수치가 없습니다.
3. **랭킹 축을 노리는 시도는 최대 1~2회, 기대치를 낮춰 잡고.** 남은 예산으로는 (a) `xgboost` 등 다른 트리 계열로의 교체, (b) `hist_gbdt` 내부 용량 조정(`max_leaf_nodes` 상향 + `learning_rate` 하향, `min_samples_leaf` 조정)이 후보입니다. 다만 계열 교체는 `roc_auc` 0.0022~0.0077, 동일 계열 재튜닝은 0.0032 규모로 관측된 바 있고, 현재 검증 구간 폭(0.8671~0.8881)이 그보다 넓습니다. 즉 개선이 나와도 이 split에서는 구별되지 않을 가능성이 큽니다. 반대로 `class_weight` 재조정은 `cut_headroom=0.004906` 때문에 상한이 0.005 미만이므로 **권하지 않습니다**.
4. **확률을 소비할 계획이라면 루프 밖에서 보정.** `calibration_error=0.112497`은 약 11%포인트 어긋남을 뜻하고, 이 실행에는 재보정 수단이 없습니다. 기대수익 계산이나 콜 리스트 컷오프처럼 확률의 절대값을 쓰는 용도라면, 이 모델 산출물에 대해 별도 보정 단계를 붙이고 그 이후 임계값을 정하십시오. 순위/컷 기반 사용(상위 N% 콜)에는 지금 상태로 문제가 없습니다.