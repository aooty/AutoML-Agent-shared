## 요약

목표는 달성되었습니다. 1차 시도의 `hist_gbdt`가 검증 r2 = **0.8902** (95% CI 0.8725~0.9063)를 기록해 목표 임계값 0.7718을 0.118 상회했고, 구간 전체가 임계값과 ridge baseline CI 상한(0.7130)보다 위에 있어 통과가 경계선에 걸린 결과가 아닙니다. 총 5회 예산 중 3회를 사용했고, 2·3차 시도가 1차와 구분 가능한 개선을 내지 못해 정체(stalled)로 조기 종료되었습니다. 한 번도 사용되지 않은 테스트 20%에서 같은 모델을 재채점한 결과는 r2 = **0.8825** (95% CI 0.8518~0.9070)로, 검증 점수와 같은 수준입니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=600`, `learning_rate=0.06`, `max_leaf_nodes=63`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=30` | r2 **0.8902** (CI 0.8725~0.9063), mae 64,148, rmse 115,774, `train_r2`=0.9589, `train_val_gap`=0.0687, 3.1s | `hyperparam` — 임계값 통과는 견고하나 `train_val_gap`=0.0687(train_mae 43,492 vs val_mae 64,148)이 남아 있어, 용량(capacity) 축을 축소 방향으로 한 번 탐색하라고 지시 |
| 2 | `hist_gbdt` | `max_iter=3000`, `learning_rate=0.03`, `max_leaf_nodes=31`, `max_depth=8`, `min_samples_leaf=40`, `l2_regularization=10.0`, `max_bins=255`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=60` | r2 0.8899 (CI 0.8733~0.9040), mae 64,295, rmse 115,926, `train_r2`=0.9464, `train_val_gap`=0.0565, 6.4s | `hyperparam` — iteration 1 대비 Δ=-0.0003 (CI -0.0068~+0.0068, P(better)=0.46)로 **측정으로 구분 불가**. 모든 규제 노브를 2~10배 움직여도 검증 r2가 움직이지 않음 → 용량 축 탐색 중단, 다른 트리 계열을 시도하라고 지시 |
| 3 | `random_forest` | `n_estimators=600`, `max_features=0.4`, `min_samples_leaf=1`, `min_samples_split=2`, `bootstrap=True`, `n_jobs=-1`, `random_state=42` | r2 0.8767 (CI 0.8638~0.8886), mae 69,251, rmse 122,700, `train_r2`=0.9831, `train_val_gap`=0.1064, 4.4s | (critic 없음 — 루프 종료) |

세 시도 모두 `status=ok`, `dropped_hyperparams` 없음, `unsupported_claims` 없음이며 학습 시간은 600s 예산의 1% 이하였습니다.

## 최고 성능 구성

iteration 1, `hist_gbdt`. 아래 값은 해당 시도의 history에 기록된 **실제 적용된** 값입니다.

- `hyperparams`: `max_iter=600`, `learning_rate=0.06`, `max_leaf_nodes=63`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=30`
- `preprocessing` (applied): `impute="median"`, `scale=false`, `missing_indicator=false`, `missing_count=false`
- 프로토콜: seed 42, 비층화 3분할 (train 60% / validation 20% / test 20%), 데이터셋 카드의 ridge baseline과 동일한 분할

| 지표 | validation (선택에 사용) | held-back test (1회 채점) |
|---|---|---|
| r2 | **0.8902** (95% CI 0.8725~0.9063) | **0.8825** (95% CI 0.8518~0.9070) |
| mae | 64,148.44 | — |
| rmse | 115,774.42 | — |
| train_r2 / train_mae | 0.9589 / 43,491.61 | — |
| train_val_gap (r2) | 0.0687 | — |

검증 0.8902와 테스트 0.8825의 차이 **+0.0077**이 이번 실행의 선택 편향(selection effect) 크기입니다 — 루프는 검증 숫자만 보고 최고 시도를 골랐으므로, 모델에 대해 실제로 입증된 값은 테스트 쪽 0.8825입니다. 다만 검증 점수가 테스트 CI(0.8518~0.9070) 안에 있어, 이 행들만으로는 그 차이를 0과 구분할 수 없습니다. 두 수치 모두 임계값 0.7718과 baseline CI 상한 0.7130보다 충분히 위입니다.

## 원인 분석

- **임계값은 첫 시도에서 이미 해결된 문제였습니다.** ridge baseline(r2 0.6957, CI 0.6735~0.713)에서 `hist_gbdt`로 계열을 바꾼 것만으로 검증 r2가 0.6957 → 0.8902로 올라갔습니다. 이 이동은 검증 슬라이스의 CI 폭(약 0.034)보다 훨씬 크므로 실재하는 차이입니다. 카드가 시사하는 구조(가격이 `lat`/`long`/`zipcode` 같은 지리 코드와 `sqft_living`·`grade`의 비선형 상호작용에 지배됨)와 일치합니다.
- **이후 두 시도는 이 데이터로 1차와 구분되지 않습니다.** iteration 2는 iteration 1과 짝지은 비교에서 Δ=-0.0003 (CI -0.0068~+0.0068), iteration 3(0.8767, CI 0.8638~0.8886)은 iteration 1의 CI(0.8725~0.9063)와 겹칩니다. 즉 "규제를 10배 강화했더니 나빠졌다", "random_forest가 hist_gbdt보다 못하다" 같은 서사는 이 측정으로 지지되지 않습니다. 검증 슬라이스의 CI 폭이 약 0.031~0.034이므로, **r2 0.03 미만의 변화는 이 프로토콜에서 개선으로 읽을 수 없습니다.**
- **해석 가능한 유일한 움직임은 학습/검증 격차 쪽입니다.** iteration 2에서 `max_leaf_nodes` 63→31, `min_samples_leaf` 20→40, `l2_regularization` 1→10, `learning_rate` 0.06→0.03으로 용량을 대폭 줄였을 때 `train_r2`는 0.9589→0.9464, `train_val_gap`은 0.0687→0.0565로 내려갔지만 검증 r2는 움직이지 않았습니다. 반대로 iteration 3의 `random_forest`(`min_samples_leaf=1`, 완전 성장)는 `train_r2`=0.9831, `train_val_gap`=0.1064로 암기량이 가장 컸지만 검증 점수는 역시 같은 대역에 머물렀습니다. 종합하면 **약 0.89는 용량 설정의 문제가 아니라, 이 21개 수치 열을 그대로 넣는 구성 공간의 상한**입니다. Critic이 두 번 연속 `hyperparam`으로 진단하면서도 두 번째에는 "규제 축 탐색을 멈추라"고 방향을 바꾼 것이 이 평탄한 응답면의 신호입니다.
- **테스트하지 못한 레버가 하나 있고, 그것은 무해합니다.** iteration 2의 critic 처방에는 `impute: "none"`이 포함됐지만 실제 파이프라인에는 `impute="median"`이 적용되었습니다(`applied_preprocessing` 기준). 다만 이 카드의 결측률은 `overall_rate=0.0`, `columns_with_missing=0`이므로 imputation 전략·`missing_indicator`·`missing_count`는 어느 쪽이든 동일한 행렬을 만듭니다 — 즉 이 레버는 애초에 얻을 것이 없는 축이며, 이 미실행이 점수를 깎은 원인이 아닙니다.
- **`unsupported_claims`는 세 시도 모두 비어 있고 `dropped_hyperparams`도 없습니다.** 계획한 하이퍼파라미터는 모두 그대로 전달되었으므로, 0.89 대역은 executor 제약이 깎아낸 값이 아니라 이 executor가 이 열 구성에서 실제로 도달하는 값입니다. 성능을 더 올리는 데 필요한 것(파생 특성, 타깃 변환, 교차검증)은 executor의 CANNOT 목록에 있어 이 루프 안에서는 시험 자체가 불가능했습니다.

## 다음 단계 제안

이번 실행은 목표를 이미 통과했으므로, 아래는 통과를 지키기 위한 것이 아니라 **0.89 대역의 천장을 실제로 넘기 위한** 제안입니다. (기록된 `Data caveats`는 없으며, 아래 판단은 카드의 집계 사실 — 결측 0%, 타깃 skew high, 지리 열 존재 — 에만 의존합니다.)

1. **파생 특성을 데이터셋 카드 단계에서 만들어 넣기 (가장 큰 기대값).** executor는 열을 조합·인코딩·삭제하지 못하므로, 열은 계획이 아니라 카드가 바꿔야 합니다. 우선순위는 (a) `zipcode`를 수치 코드가 아닌 범주형으로 one-hot(카드 `max_cardinality=50`, `zipcode`는 distinct medium이므로 인코딩 예산 안), (b) `house_age = date_year - yr_built` 및 `years_since_renovation`, (c) `sqft_living / sqft_lot`, `sqft_living / sqft_living15` 같은 비율. 근거: 규제·계열을 10배씩 흔들어도 검증 r2가 CI 폭 미만으로만 움직였다는 점은 남은 신호가 하이퍼파라미터가 아니라 표현(representation)에 있다는 뜻입니다.
2. **타깃 로그 변환을 파이프라인 기능으로 추가하고 재측정.** 카드는 `price`의 skew를 high, outlier_rate 0.053으로 보고하고, 최고 모델의 rmse(115,774)가 mae(64,148)의 1.8배라 상위 꼬리가 오차를 지배합니다. 현재 executor는 타깃 변환을 하지 못하므로 이는 "시도"가 아니라 **만들어야 하는 기능**입니다(`TransformedTargetRegressor` 수준). 구현되면 같은 `hist_gbdt` 설정으로 곧바로 비교 가능한 한 줄이 생깁니다.
3. **0.03보다 작은 차이를 볼 수 있는 측정으로 바꾸기.** 검증 20% 단일 슬라이스의 CI 폭이 0.031~0.034여서, 향후 하이퍼파라미터 튜닝의 실제 이득 대부분이 잡음과 구분되지 않습니다. 이 루프에서는 분할·시드·프로토콜이 고정 설정이므로, 다음 실행에서는 k-fold 또는 반복 분할을 지원하도록 스코어링을 확장한 뒤에 미세 튜닝에 예산을 쓰는 것이 순서입니다. 그 전에는 `l2_regularization`·`max_leaf_nodes` 류의 재탐색을 권하지 않습니다 — iteration 1↔2가 이미 그 결과를 보여줬습니다.
4. **세 번째 계열로 `xgboost`를 1회만 확인.** `hist_gbdt`와 `random_forest`가 서로 겹치는 구간에 있으므로 "0.89가 이 열 구성의 천장"이라는 해석의 반증 시도로 값이 있습니다. 단, 기대 이득은 CI 폭(≈0.03)을 넘기 어려우니 1·2번보다 우선순위는 낮고, `eval_set`/`early_stopping_rounds`는 executor가 차단하므로 고정 `n_estimators`+낮은 `learning_rate`로 계획해야 합니다.
5. **하지 말 것 (예산 낭비 확정):** `impute`(median/mean/most_frequent/none) 변경, `missing_indicator`, `missing_count`. 카드의 `missing.overall_rate=0.0`, `columns_with_missing=0`이므로 이 세 레버는 동일한 입력 행렬을 만들며, 어떤 차이가 관측되더라도 resample 잡음입니다.