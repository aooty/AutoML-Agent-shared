# AutoML 실행 최종 보고서 — `jungle-chess`

## 요약

목표는 달성되었습니다. 목표 지표 `balanced_accuracy`의 임계값 0.633에 대해, 첫 번째 시도의 `hist_gbdt`가 검증 기준 **0.8016** (95% CI 0.7918~0.8139)을 기록했고, 한 번도 사용되지 않은 최종 테스트 20%에서 **0.7991** (95% CI 0.7875~0.8096)을 기록했습니다. 사용한 반복은 5회 중 1회이며, 첫 계획이 곧바로 기준선(logreg, `balanced_accuracy` 0.5106, CI 0.5024~0.5177)과 임계값을 모두 크게 넘어 루프가 종료되었습니다. critic 진단은 한 번도 수행되지 않았습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=500`, `learning_rate=0.1`, `max_depth=12`, `max_leaf_nodes=255`, `l2_regularization=1.0`, `class_weight='balanced'`, `early_stopping=False` | status `ok` — 검증 `balanced_accuracy`=0.8016 (CI 0.7918~0.8139), `accuracy`=0.8524, `f1`=0.7955, `train_val_gap`=0.1955, 학습 24.993초 | 없음 (목표 달성으로 루프 종료, critic 미실행) |

## 최고 성능 구성

- **model**: `hist_gbdt`
- **hyperparams** (해당 시도에 실제로 적용된 값, `dropped_hyperparams`는 비어 있음):
  - `max_iter=500`
  - `learning_rate=0.1`
  - `max_depth=12`
  - `max_leaf_nodes=255`
  - `l2_regularization=1.0`
  - `class_weight='balanced'`
  - `early_stopping=False`
- **preprocessing** (해당 시도의 적용값): `impute='median'`, `scale=False`, `missing_indicator=False`, `missing_count=False`
- **프로토콜**: stratified 3분할, seed 42, train 60% / validation 20% / test 20% (카드의 baseline과 동일한 분할)

| 지표 | 검증 (validation 20%) | 최종 테스트 (held-back 20%) |
|---|---|---|
| `balanced_accuracy` | **0.8016** (CI 0.7918~0.8139) | **0.7991** (CI 0.7875~0.8096) |
| `accuracy` | 0.8524 | — |
| `f1` | 0.7955 | — |
| `precision` | 0.7900 | — |
| `recall` | 0.8016 | — |
| `train_balanced_accuracy` | 0.9972 | — |
| `train_val_gap` | 0.1955 | — |

검증 0.8016과 테스트 0.7991의 차이 **+0.0025**가 이 실행의 선택 편향(selection effect) 크기입니다. 루프는 검증 숫자만 보고 최종 구성을 골랐으므로, 모델에 대해 실제로 입증된 값은 테스트 0.7991입니다. 다만 검증 점수가 테스트 CI(0.7875~0.8096) 안에 들어 있어, 이 행들만으로는 그 차이를 0과 구분할 수 없습니다. 두 값 모두 임계값 0.633을 CI 하한에서도 크게 상회합니다.

## 원인 분석

이 실행에서는 **critic이 한 번도 실행되지 않았습니다** (시도 1회, 진단 0회). 따라서 verdict 간의 패턴이라 할 것이 존재하지 않으며, 이 점수는 전적으로 **첫 계획 하나가 낸 결과**입니다. 진단·재계획 경로는 이 결과에 아무 기여도 하지 않았고, 그 경로가 유용하다는 증거도 해롭다는 증거도 이 실행에는 없습니다.

시도가 하나뿐이므로 시도 간 비교(구간 겹침 여부를 따질 대상)도 없습니다. 또한 성능을 제한한 요인에 대한 **진단은 이루어지지 않았습니다** — 아래는 진단이 아니라, 다음 단계를 정할 때 확인이 필요한 관측 사실입니다.

- `train_balanced_accuracy`=0.9972 대 검증 0.8016으로 `train_val_gap`=0.1955입니다. 이는 고용량 설정(`max_leaf_nodes=255`, `max_depth=12`, `early_stopping=False`)에서 예상 범위에 있는 값이지만, 용량을 줄였을 때 일반화가 좋아지는지 나빠지는지는 이 실행에서 측정되지 않았습니다.
- 계획 본문은 결과에서 `balanced_accuracy_at_best_cut`과 `cut_headroom`을 읽어 다음 시도를 정하겠다고 했으나, 이 태스크는 3-class(`n_classes=3`)이고 해당 진단값은 보고된 `metrics`에 존재하지 않습니다. 즉 운영점(operating point) 축과 랭킹 축을 분리해 읽는 근거는 이 실행에 없습니다. `plan.unsupported_claims`는 비어 있고, 계획이 실행기에 없는 기능에 의존한 흔적은 없습니다 — 이 숫자는 이 실행기가 실제로 낸 값입니다.
- 이 실행의 해상도 한계도 기록해 둘 필요가 있습니다: 검증 CI 폭은 약 ±0.011, 테스트 CI 폭은 약 ±0.011입니다. 이보다 작은 변화는 이 데이터로 개선이라 부를 수 없습니다.

## 다음 단계 제안

데이터 카드에 별도로 기록된 caveat은 없으므로, 아래 제안은 모두 사용 가능한 컬럼과 고정된 분할 프로토콜 안에서 성립합니다.

1. **용량 축을 한 번 측정한다** — 남은 예산으로 `hist_gbdt`를 `max_leaf_nodes` 31~63, `max_depth` 6~8, `l2_regularization` 10.0 정도로 낮춘 구성과, 별도로 `early_stopping=True`를 켠 구성을 각각 한 번 돌립니다. 목적은 점수 추격이 아니라 `train_val_gap`=0.1955가 실제 일반화 손실인지 확인하는 것입니다. 판단 기준은 명시적으로 두어야 합니다: 검증 차이가 ±0.011보다 작으면 이 데이터로는 구분되지 않는 두 구성이라고 결론 내리고, 더 단순한(학습이 빠른) 쪽을 채택합니다. `early_stopping=True`는 train의 10%를 내부 검증으로 떼어가므로 그 손실이 함께 계산된다는 점도 감안해야 합니다.
2. **소수 클래스 가중치를 `'balanced'` 한 점에 고정하지 않는다** — 클래스 비율은 0.5146 / 0.3887 / 0.0967(imbalance_ratio 5.32)이고 `'balanced'`는 그 빈도에 비율을 못박은 한 점일 뿐입니다. 클래스 코드 0..2를 모두 덮는 명시적 가중치 맵(예: 소수 클래스만 `'balanced'` 대비 상·하로 흔든 두 지점)을 시도해 `balanced_accuracy`의 클래스별 recall 반응을 봅니다. 단, 3-class이므로 `cut_headroom`으로 운영점 잔여폭을 확인할 수 없다는 제약을 전제해야 하며, 관측된 차이가 CI 폭 미만이면 운영점 튜닝의 여지가 없다고 읽는 편이 안전합니다.
3. **특징 결합은 실행기 밖에서 만들어 데이터 카드로 넣는다** — 이 태스크의 신호는 piece strength와 board position의 상호작용에 있을 가능성이 크지만, 실행기는 비율·차이·상호작용·다항항을 만들지 못합니다(`missing_indicator`/`missing_count`가 유일한 예외이며, 결측률이 0.0이므로 둘 다 무의미합니다). 따라서 strength 차이, file/rank 거리 같은 파생 컬럼은 데이터 준비 단계에서 계산해 카드에 포함시킨 뒤 새 실행을 돌려야 합니다. 이것이 열리면 트리가 분할로 근사해야 했던 관계를 직접 입력으로 줄 수 있습니다.
4. **단일 분할의 해상도를 넘어서려면 프로토콜 밖에서 검증한다** — 분할·seed·테스트 슬라이스는 실행 설정으로 고정되어 있고 cross-validation과 out-of-fold 예측은 실행기가 제공하지 않습니다. 0.7991이라는 값을 운영 의사결정에 쓰려면, 동일 파이프라인을 여러 seed로 반복 분할해 재현하는 별도 스크립트를 만드는 것이 필요합니다. 이는 현재 ±0.011 폭 안에 묻혀 있는 개선들을 앞으로 판정 가능하게 만들어 줍니다.