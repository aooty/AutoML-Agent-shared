## 요약

목표는 달성되었습니다. 첫 번째 시도에서 `hist_gbdt`가 검증 `r2` **0.8902** (95% CI 0.8725~0.9063)를 기록해, 목표 임계값 0.7718과 baseline(`ridge`, `r2` 0.6957, CI 0.6735~0.7130)을 모두 뚜렷하게 상회했습니다 — 두 구간은 전혀 겹치지 않으므로 이 개선은 리샘플 노이즈로 설명되지 않습니다. 5회 예산 중 1회만 사용하고 `goal_reached`로 조기 종료했으며, 학습 시간은 3.104초로 제약(600초, 2GB) 대비 여유가 컸습니다. 루프가 한 번도 사용하지 않은 최종 테스트 20%에서의 점수는 `r2` **0.8825** (CI 0.8518~0.9070)로, 이 수치가 이번 실행이 실제로 입증한 성능입니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=600`, `learning_rate=0.06`, `max_leaf_nodes=63`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=30`, `random_state=42` | `status=ok` — 검증 `r2`=0.8902 (CI 0.8725~0.9063), `mae`=64148.44, `rmse`=115774.42, `train_r2`=0.9589, `train_val_gap`=0.0687, 3.104초 | 없음 (목표 달성으로 루프 종료, critic 미실행) |

## 최고 성능 구성

- **iteration**: 1
- **model**: `hist_gbdt`
- **hyperparams** (해당 시도에 적용된 값): `max_iter=600`, `learning_rate=0.06`, `max_leaf_nodes=63`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=30`, `random_state=42`
- **preprocessing** (해당 시도에 적용된 값): `impute=median`, `scale=False`, `missing_indicator=False`, `missing_count=False` — 이 데이터는 결측률이 0.0이므로 imputer는 실질적으로 아무 값도 채우지 않았습니다.
- `dropped_hyperparams`는 비어 있어, 계획한 모든 키가 그대로 estimator에 적용되었습니다.
- **입력**: 카드의 numeric 21개 열 전부, `train_subsample` 없음(전체 학습 행). 분할은 실행 프로토콜 그대로 train 60% / validation 20% / test 20%, 비층화, seed 42.

| 구분 | r2 | mae | rmse |
|---|---|---|---|
| 검증 (선택에 사용) | 0.8902 (CI 0.8725~0.9063) | 64148.44 | 115774.42 |
| 학습 | 0.9589 | 43491.61 | — |
| **최종 held-back 테스트 (1회 채점)** | **0.8825 (CI 0.8518~0.9070)** | — | — |

검증 0.8902와 테스트 0.8825의 차이 **+0.0077**이 이번 실행의 선택 편향 크기입니다. 루프는 검증 숫자만 보고 최적 시도를 골랐으므로, 모델에 대해 주장할 수 있는 값은 테스트 0.8825입니다. 다만 검증 점수가 테스트 CI(0.8518~0.9070) 안에 들어 있으므로, 이 테스트 행 수만으로는 그 차이를 0과 구분할 수 없습니다. 어느 쪽으로 읽어도 목표 임계값 0.7718은 테스트 CI 하한(0.8518)보다도 위입니다.

## 원인 분석

- 시도가 1회뿐이므로 **시도 간 변화에서 이야기를 만들 수 없습니다**. 비교 가능한 대조군은 카드의 baseline 하나입니다: `ridge` 0.6957 (CI 0.6735~0.7130) → `hist_gbdt` 0.8902 (CI 0.8725~0.9063). 두 구간의 간격이 넓어(약 +0.19) 이 차이는 이 데이터로 분명히 구분되는 개선입니다.
- 개선의 성격은 카드의 열 프로파일과 일치합니다. `lat`은 target과 strong 상관, `long`은 none, `zipcode`는 weak로 기록되어 있는데 — 이는 위치 효과가 단일 좌표에 대한 선형 기울기가 아니라 좌표 조합에 대한 비선형 구간 효과라는 신호입니다. `ridge`는 이를 표현할 수 없고, 부스팅 트리는 축 정렬 분할의 누적으로 표현합니다. `sqft_living`/`grade`/`sqft_above`(모두 strong)와의 곱셈적 상호작용도 같은 방향입니다.
- 남은 여유는 **과적합 쪽에 있습니다**: `train_r2`=0.9589 vs 검증 0.8902, `train_val_gap`=0.0687(양수 = 검증이 학습보다 나쁨). `max_leaf_nodes=63`에 `min_samples_leaf=20`, `l2_regularization=1.0`은 13k 학습 행에 대해 다소 관대한 설정이며, `early_stopping=True`가 라운드 수는 스스로 잘랐지만 트리 복잡도 자체를 줄이지는 않습니다. 다만 이 gap이 검증 CI 폭(약 ±0.017)과 비슷한 규모라는 점은 유의해야 하며, 정규화를 조인다고 검증 점수가 오른다는 증거는 이번 실행에 없습니다 — 아직 시험되지 않은 가설입니다.
- `plan.unsupported_claims`는 비어 있고 `dropped_hyperparams`도 비어 있으므로, 계획 중 실행되지 않은 부분 때문에 잃은 성능은 없습니다. 절대 점수는 이 executor가 이 카드로 달성한 값 그대로입니다.
- critic 진단은 `null`입니다(1회차에서 목표 달성 → 진단 단계 미실행). 따라서 "critic 판정의 패턴"으로부터 도출할 결론은 없습니다.

## 다음 단계 제안

이 데이터셋에는 별도로 기록된 caveat이 없으므로, 아래 제안은 실행 환경의 능력 경계만을 제약으로 삼습니다.

1. **테스트 점수 0.8825를 보고 기준으로 고정하고, 정규화 강화를 1회 실험으로 검증.** `train_val_gap`=0.0687이 유일하게 관측된 개선 여지입니다. 같은 `hist_gbdt`에서 `max_leaf_nodes`를 31로, `min_samples_leaf`를 40~50으로, `l2_regularization`을 5~10으로 올리고 `learning_rate=0.04`로 낮춘 구성을 시도하되, **검증 CI 폭(약 0.017)보다 작은 차이는 개선으로 보고하지 말 것**. 이 실험이 답하는 질문은 "점수가 오르나"가 아니라 "현재 구성이 복잡도 때문에 손실 중인가"입니다.
2. **다른 단일 estimator 계열로 견고성 확인** — `xgboost` 또는 `random_forest`를 동일 전처리(`impute=median`, `scale=False`)로 1회. 목적은 최고점 경쟁이 아니라, 0.89 수준이 특정 부스팅 구현의 산물이 아님을 확인하는 것입니다. 단, `xgboost`에는 `eval_set`/`early_stopping_rounds`/`callbacks`를 쓸 수 없으므로 `n_estimators`를 고정값으로 계획해야 합니다.
3. **위치·면적 파생 특징은 executor 밖에서, 즉 데이터 카드 단계에서 준비.** executor는 열을 만들거나 결합하거나 재인코딩하지 않습니다(비율·차이·상호작용·원핫 모두 불가). `zipcode`는 현재 `int64` continuous로 들어가 크기 순서가 무의미한데도 분할 기준으로 쓰이고 있습니다. 카드 생성 시 `zipcode`를 저카디널리티 범주로 원핫 인코딩하거나(카드의 `max_cardinality=50` 정책 확인 필요), `sqft_living` 대비 `sqft_living15` 비율 같은 열을 미리 만들어 넣으면 현재 트리가 축 정렬 분할로 근사하고 있는 구조를 직접 제공할 수 있습니다.
4. **target의 높은 왜도(skew=high, outlier_rate=0.053)에 대한 대응은 로그 변환된 target 열로 카드를 다시 만드는 방식으로만 가능.** executor는 target 변환을 지원하지 않으며, 변환 시 `r2`가 다른 스케일에서 계산되어 이번 baseline·임계값과 직접 비교할 수 없게 된다는 점을 함께 기록해야 합니다. 현재 `mae`=64148 / `rmse`=115774의 큰 격차는 고가 구간의 큰 오차가 지배적임을 시사하므로, 고가 주택 예측 정확도가 중요한 용도라면 이 경로가 우선순위이고, 그렇지 않다면 3번이 먼저입니다.