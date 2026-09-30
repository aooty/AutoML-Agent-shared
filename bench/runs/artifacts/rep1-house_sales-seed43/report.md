# house_sales 회귀 실행 최종 보고서

## 요약

목표는 달성되었습니다. 목표 지표인 `r2` 0.7831(baseline ridge 0.7108 + 마진) 기준에 대해, 첫 시도에서 `hist_gbdt`가 검증 `r2` **0.8699** (95% CI 0.8342~0.8961)를 기록해 임계값을 명확히 넘겼습니다. 5회 예산 중 **1회 시도**만 사용하고 루프가 종료되었습니다. 반복 중 한 번도 사용되지 않은 최종 테스트 20%에서의 점수는 **r2 = 0.8934** (95% CI 0.8738~0.9132)입니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=600`, `learning_rate=0.06`, `max_leaf_nodes=63`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=False` | `status=ok` · 검증 `r2`=0.8699 (CI 0.8342~0.8961), `mae`=67663.61, `rmse`=132451.23, `train_r2`=0.9897, `train_val_gap`=0.1198, 학습 5.505초 | 없음 (critic 미실행 — 첫 시도에서 목표 달성) |

## 최고 성능 구성

- **iteration**: 1
- **model**: `hist_gbdt`
- **hyperparams** (실제 적용값, `dropped_hyperparams` 없음):
  - `max_iter`: 600
  - `learning_rate`: 0.06
  - `max_leaf_nodes`: 63
  - `min_samples_leaf`: 20
  - `l2_regularization`: 1.0
  - `early_stopping`: false
- **preprocessing** (해당 시도가 기록한 적용값):
  - `impute`: `median`
  - `scale`: false
  - `missing_indicator`: false
  - `missing_count`: false
- **프로토콜**: 비층화 3분할(train 60% / validation 20% / test 20%), seed 43 — 카드의 baseline과 동일 분할이므로 직접 비교 가능. 입력은 카드의 numeric 21개 열 전체이며 파생·인코딩·드롭 없음.

| 지표 | 검증(validation 20%) | 최종 테스트(held-back 20%) |
|---|---|---|
| `r2` | 0.8699 (95% CI 0.8342~0.8961) | **0.8934** (95% CI 0.8738~0.9132) |
| `mae` | 67663.61 | — |
| `rmse` | 132451.23 | — |
| `train_r2` | 0.9897 | — |
| `train_val_gap` | 0.1198 | — |

검증과 테스트의 차이는 -0.0235(검증 0.8699 → 테스트 0.8934)입니다. 이 실행의 모든 선택은 검증 점수로 이루어졌으므로, 이 차이가 선택 편향의 크기입니다. 이 실행이 실제로 입증하는 값은 held-back 테스트 점수 **0.8934**이며, 검증 점수는 선택에 사용된 수치입니다. (참고로 두 구간은 서로 겹치므로, 이 데이터가 두 값을 분리해 주지는 않습니다.)

## 원인 분석

- **critic 진단은 0건입니다.** 시도 1회, 진단 0회로, 첫 계획이 곧바로 임계값을 넘겨 루프가 종료되었습니다. 따라서 verdict들 사이의 패턴이라 할 것이 존재하지 않으며, 이 점수는 **첫 계획 하나가 낸 결과**입니다. 이 실행은 진단·재계획 경로가 도움이 되는지 해가 되는지에 대해 아무것도 말해 주지 않습니다.
- 시도가 1건뿐이므로 **시도 간 변화로 성능을 설명할 근거도 없습니다.** 비교 가능한 기준선은 카드의 baseline `ridge` (검증 `r2` 0.7108, CI 0.6849~0.7328) 하나이고, `hist_gbdt`의 검증 CI(0.8342~0.8961)와 baseline CI(0.6849~0.7328)는 겹치지 않으므로 **선형 모델 → gradient-boosted tree 계열 전환은 이 데이터에서 구간 폭보다 큰, 실재하는 개선**입니다. 이는 카드가 보여 주는 구조(`lat`, `sqft_living`, `grade`의 강한 `target_corr`와 target의 높은 skew·outlier_rate 0.053)와도 일관됩니다.
- `train_r2`=0.9897 대비 검증 `r2`=0.8699로 `train_val_gap`=0.1198의 과적합이 관찰됩니다. 다만 이것은 **관측된 수치이지 진단이 아닙니다** — 이 실행에서 정규화/용량 축소를 실제로 비교한 시도는 없으므로, 이 gap이 성능을 제한했다고 단정할 근거는 없습니다.
- `plan.unsupported_claims`는 비어 있고, `dropped_hyperparams`도 비어 있습니다. 즉 계획이 실행기가 못 하는 기능에 의존해 손실을 본 흔적은 없습니다. 결측치가 전혀 없는 데이터(`overall_rate`=0.0)여서 `missing_indicator`/`missing_count` 레버는 애초에 적용 대상이 아니며, `impute: median`도 실질적으로 아무 값도 바꾸지 않았습니다.

## 다음 단계 제안

1. **용량·정규화 스윕으로 `train_val_gap`=0.1198을 실제로 검증한다.** 같은 `hist_gbdt` 계열에서 `max_leaf_nodes`(예: 31/63), `l2_regularization`(예: 1.0/10.0), `min_samples_leaf`(예: 20/50), `learning_rate`(0.03~0.06 + 그에 맞춘 `max_iter`)를 한 축씩 바꿔 2~3회 시도. 단, 검증 CI 폭이 약 ±0.03이므로 **그보다 작은 차이는 개선으로 보고하지 말 것** — 이 슬라이스는 그 크기의 차이를 분리하지 못합니다.
2. **`early_stopping=True` + `validation_fraction`의 손익을 직접 측정한다.** 현재는 `early_stopping=False`로 train 12,967행 전부를 사용해 baseline과 비교 가능성을 확보했습니다. 조기 종료는 train의 10%를 떼어내므로 순손익이 자명하지 않으며, 한 번의 시도로 `internal_validation` 카운트와 함께 확인하는 것이 값싼 검증입니다.
3. **다른 단일 트리 앙상블(`xgboost`, `random_forest`)로 계열 민감도를 확인한다.** 같은 고정 분할·seed에서 한 번씩 적합해, 0.87 수준이 `hist_gbdt` 특유의 결과인지 트리 계열 일반의 결과인지 구분합니다(스태킹·블렌딩은 실행기가 지원하지 않으므로 계획에 넣지 말 것).
4. **파생 피처가 필요하다고 판단되면 계획이 아니라 카드 쪽에서 처리한다.** 실행기는 비율·차분·상호작용·재인코딩·드롭을 하지 않습니다. `zipcode`(현재 정수 numeric, `target_corr`=weak)의 범주형 인코딩이나 `lat`/`long` 기반 위치 피처처럼 가장 유망한 후보는 데이터셋 카드 단계에서 열로 만들어 넣어야 하며, 이것이 열리면 위치 구조를 트리가 축 정렬 분할로 근사하는 현재의 한계를 시험할 수 있습니다.

> `Data caveats`에 별도로 기록된 주의사항은 없습니다. 따라서 위 제안 중 특정 열·값·분할의 신뢰성 때문에 배제되는 항목은 없으나, 4번은 실행기 능력 범위 밖의 작업(카드 재생성)이라는 점을 전제로 합니다.