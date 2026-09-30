# 최종 리포트 — bank-marketing (balanced_accuracy)

## 요약

목표는 **달성**되었습니다. 임계값 `balanced_accuracy` ≥ 0.7452에 대해, 1회차 시도의 `hist_gbdt`가 검증 슬라이스에서 **0.8515 (95% CI 0.8382~0.8614)** 를 기록했고, 한 번도 사용되지 않은 최종 테스트 20%에서 **0.8558 (95% CI 0.8436~0.8682)** 를 냈습니다. 사용된 반복은 5회 중 **1회**이며, 첫 계획이 곧바로 임계값(그리고 baseline 0.6603, ranking_ceiling 0.8382)을 넘어 루프가 종료되었습니다. 따라서 이 실행에서 critic 진단·재계획은 단 한 번도 실행되지 않았습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=False`, `class_weight={"0":1.0,"1":6.0}` | `status=ok` — `balanced_accuracy=0.8515` (CI 0.8382~0.8614), `roc_auc=0.9321`, `recall=0.8119`, `specificity=0.8910`, 학습 7.392초 → 목표(0.7452) 초과 달성 | 없음 (목표 달성으로 루프 종료, critic 미실행) |

## 최고 성능 구성

**iteration 1 / `hist_gbdt`** — 아래 값은 history의 해당 시도에 기록된 **실제 적용값**입니다.

- `hyperparams`:
  - `max_iter: 400`
  - `learning_rate: 0.06`
  - `max_leaf_nodes: 31`
  - `min_samples_leaf: 20`
  - `l2_regularization: 1.0`
  - `early_stopping: false`
  - `class_weight: {"0": 1.0, "1": 6.0}`
  - `dropped_hyperparams`: 없음 (빈 목록)
- `preprocessing` (적용된 값): `impute: median`, `scale: false`, `missing_indicator: false`, `missing_count: false`
- 프로토콜: 카드의 고정 분할 — stratified, train 0.6 / val 0.2 / test 0.2, seed 44
- 학습 시간: 7.392초

성능:

| 지표 | 검증(val) | 비고 |
|---|---|---|
| `balanced_accuracy` | **0.8514706634440905** (95% CI 0.8382~0.8614) | 선택에 사용된 수치 |
| `roc_auc` | 0.9320777661182478 | baseline 0.9087 |
| `pr_auc` / `average_precision` | 0.6214733388483193 | |
| `recall` | 0.8119092627599244 | |
| `specificity` | 0.891032 | |
| `precision` | 0.496818970503181 | |
| `f1` | 0.6164334409759599 | |
| `accuracy` | 0.8817739438177394 | |
| `balanced_accuracy_at_best_cut` | 0.8698 | 진단용 |
| `cut_headroom` | 0.018329 | 진단용 |
| `brier` / `calibration_error` | 0.082023 / 0.0827 | 진단용 |
| `train_balanced_accuracy` / `train_val_gap` | 0.9384460670820203 / 0.086975 | |

**최종 held-back 측정**: 같은 모델을 첫 학습 전에 떼어 두고 어떤 결정에도 쓰지 않은 테스트 20%에서 한 번 채점 → `balanced_accuracy = 0.8558` (95% CI 0.8436~0.8682). 검증값 0.8515 대비 **-0.0043**(테스트가 오히려 높음)이며, 이 차이가 선택 편향의 크기입니다. 다만 검증 점수가 테스트 CI 안에 들어 있어, 이 행들만으로는 그 차이를 0과 구분할 수 없습니다. 이 실행이 실제로 입증한 수치는 검증값이 아니라 테스트값 0.8558입니다.

## 원인 분석

`재계획` 항목이 명시하듯 **critic 진단은 0건**입니다. 첫 시도가 곧바로 임계값을 넘겨 루프가 종료되었으므로, 여러 시도 사이의 패턴이나 진단의 경향 같은 것은 이 실행에 존재하지 않습니다. 즉 **이 점수는 첫 계획 하나가 낸 결과**이며, 진단·재계획 경로가 도움이 되었다는 증거도 해가 되었다는 증거도 이 실행에는 없습니다. 시도가 1건뿐이므로 시도 간 비교(구간 중첩 여부를 따질 대상)도 존재하지 않고, 성능을 제한한 요인에 대한 "진단"은 이 실행에서 내려진 바 없습니다 — 아래는 진단이 아니라 단일 시도의 계측값을 그대로 읽은 것입니다.

- baseline(`logreg`, median impute + standard scale)은 `balanced_accuracy=0.6603` (CI 0.6465~0.6746), `roc_auc=0.9087`, `recall=0.345`였습니다. 최고 구성은 `roc_auc` 0.9321로 랭킹 축이 올라갔고, 동시에 `recall` 0.8119 / `specificity` 0.8910으로 기본 0.5 컷이 균형점 근처에 놓였습니다. 두 축이 모두 움직였으므로 `balanced_accuracy` 차이 하나만으로 어느 쪽이 얼마를 기여했는지 분해할 수는 없습니다.
- `cut_headroom = 0.018329`, `balanced_accuracy_at_best_cut = 0.8698`: 현재 기본 결정 규칙은 이 모델의 랭킹에서 얻을 수 있는 최선에 가깝습니다. 남은 여지의 대부분은 `class_weight` 재조정이 아니라 랭킹 축(모델 계열/용량, 입력 열)에 있습니다.
- `plan.unsupported_claims`에 `feature_engineering`이 표시되어 있으나, 해당 계획 본문을 읽으면 파생 피처를 만들겠다는 의존이 아니라 "결측 표기 열을 추가하지 않는 이유", "남은 격차가 랭킹인지 운영점인지 판단하겠다"는 서술에서 문자열이 걸린 것입니다. 계획은 존재하지 않는 기능에 의존하지 않았으므로, 이 시도가 그 때문에 무엇을 잃었다고 볼 근거는 없습니다. 피처 생성 기능의 부재는 "실패 원인"이 아니라 다음 단계에서 별도로 다룰 항목입니다.

## 다음 단계 제안

1. **랭킹 축을 겨냥한 1~2회 추가 시도** — `cut_headroom`이 0.0183로 작아 운영점 재조정(`class_weight` 미세 튜닝)으로 얻을 수 있는 상한이 작습니다. 대신 같은 파이프라인에서 `xgboost`로 계열을 바꾸거나 `hist_gbdt`의 용량(`max_leaf_nodes` 증가, `learning_rate` 감소 + `max_iter` 증가)을 키워 `roc_auc`와 `balanced_accuracy_at_best_cut`의 변화를 읽으십시오. 단, 계열 교체·재튜닝의 통상적 효과 크기는 소수 셋째 자리 수준이며 이 검증 슬라이스의 CI 폭이 약 ±0.012이므로, **CI 폭보다 작은 변화는 개선으로 보고하지 말 것**을 전제로 진행해야 합니다. 이미 목표는 달성되었으므로 이는 확인·여유 확보용 작업입니다.
2. **`unknown` 표기의 의미를 원본에서 확정하는 것(캐비어트의 자체 해결)** — V16의 `unknown`이 81.8%, V9가 28.8%, V4가 4.1%, V2가 0.6%입니다. 카드는 이들을 실제 측정값으로 집계했으므로, 해당 열의 `target_corr`·`outlier_rate`는 이 값을 포함한 수치입니다. 이 실행의 계획은 V16의 `unknown`을 "직전 캠페인 접촉 없음"이라는 상태로 간주했지만, **그 해석은 이 데이터로 검증되지 않았습니다.** 원본 정의를 확인해 (a) 상태인지 (b) 기록 누락인지를 확정하기 전에는, 이들 열의 재인코딩이나 결측 표기 열 추가를 근거로 성능 개선을 약속하지 말아야 합니다. 확정되면 "V9/V16을 결측으로 재인코딩할지"라는 선택지가 열리고, 그 작업은 executor가 열을 바꿀 수 없으므로 **카드/입력 데이터 단계에서** 처리되어야 합니다.
3. **`missing_indicator` / `missing_count`에 반복을 쓰지 말 것** — 카드의 `missing.overall_rate`가 0.0이고 결측 열이 0개이므로 두 열은 상수가 되어 아무 정보도 추가하지 않습니다. `unknown`은 이미 one-hot 수준으로 들어가 있습니다. 이 판단은 위 2번이 확정되어 실제 NaN이 생길 때에만 재검토 대상입니다.
4. **확률값을 그대로 쓰려면 별도 보정 단계를 외부에 마련할 것** — `calibration_error = 0.0827`, `brier = 0.082023`으로, 랭킹은 좋지만 예측 확률은 평균 약 8퍼센트포인트 어긋나 있습니다. 이 실행 환경에는 보정 레버가 없으므로(`CalibratedClassifierCV`·컷 이동 모두 불가), 확률을 의사결정에 직접 쓰는 용도라면 파이프라인 밖에서 보정을 붙이고 그 효과를 별도로 측정해야 합니다. `balanced_accuracy`만 목표라면 이 항목은 우선순위가 낮습니다.