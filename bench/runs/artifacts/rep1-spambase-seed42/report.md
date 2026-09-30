# AutoML 실행 리포트 — `spambase` (balanced_accuracy)

## 요약

목표는 달성되었습니다. 목표 임계값은 `balanced_accuracy` 0.9421이었고, 1회 시도만으로 `hist_gbdt`가 검증 슬라이스에서 0.9539 (95% CI 0.9398~0.9673)를 기록했습니다. 한 번도 학습·선택에 쓰이지 않은 최종 테스트 20%에서도 0.9454 (95% CI 0.9288~0.9598)로 임계값을 넘었습니다. 총 5회 예산 중 1회만 사용했고, 첫 계획이 곧바로 통과했기 때문에 critic(진단·재계획) 경로는 한 번도 실행되지 않았습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `learning_rate=0.08`, `max_iter=400`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `early_stopping=false` | 성공(`status: ok`) — 검증 `balanced_accuracy` 0.9539 (CI 0.9398~0.9673), `roc_auc` 0.9909, `pr_auc` 0.9881, 학습 3.885초. 목표 0.9421 초과로 루프 종료 | 없음 (목표 달성으로 critic 미실행) |

## 최고 성능 구성

- **iteration**: 1
- **model**: `hist_gbdt`
- **hyperparams** (해당 시도에 실제 적용된 값, `dropped_hyperparams`는 없음):
  - `learning_rate`: 0.08
  - `max_iter`: 400
  - `max_leaf_nodes`: 31
  - `l2_regularization`: 1.0
  - `early_stopping`: false
- **preprocessing** (해당 시도가 실제로 만든 파이프라인):
  - `impute`: `median`
  - `scale`: false
  - `missing_indicator`: false
  - `missing_count`: false
- **프로토콜**: stratified 3-way split, seed 42, train 60% / validation 20% / test 20%

| 지표 | 검증(validation 20%) | 최종 테스트(held-back 20%) |
|---|---|---|
| `balanced_accuracy` | **0.9539** (95% CI 0.9398~0.9673) | **0.9454** (95% CI 0.9288~0.9598) |

기타 검증 지표: `accuracy` 0.9576, `f1` 0.9456, `precision` 0.9549, `recall` 0.9365, `specificity` 0.9713, `roc_auc` 0.9909, `pr_auc` / `average_precision` 0.9881, `balanced_accuracy_at_best_cut` 0.9568, `cut_headroom` 0.002905, `brier` 0.034233, `calibration_error` 0.025134, `train_balanced_accuracy` 1.0, `train_val_gap` 0.046105.

검증 0.9539 대비 테스트 0.9454로 **-0.0085** 차이가 있습니다. 이 차이가 선택 편향(selection effect)의 크기입니다 — 루프는 검증 숫자만 보고 최종 모델을 골랐기 때문입니다. 단, 검증 점수가 테스트 CI(0.9288~0.9598) 안에 들어오므로 이 행 수로는 그 차이를 0과 구분할 수 없습니다. 실제로 이 실행이 입증하는 값은 테스트 점수 0.9454이며, 이 값도 임계값 0.9421을 넘습니다.

## 원인 분석

- **critic 진단은 0건입니다.** 첫 시도가 곧바로 목표를 넘겨 루프가 종료되었으므로 critic이 한 번도 호출되지 않았습니다. 따라서 "시도 간 패턴"이나 "진단이 지적한 병목" 같은 서술은 존재하지 않습니다. 이번 점수는 **첫 계획 하나가 낸 결과**이며, 이 실행은 진단·재계획 루프가 도움이 된다는 증거도, 해가 된다는 증거도 제공하지 않습니다.
- 시도가 1건뿐이라 **시도 간 비교로 설명할 수 있는 움직임도 없습니다.** 유일한 비교 대상은 카드의 baseline(`logreg`, 검증 `balanced_accuracy` 0.9228, CI 0.903~0.941)이며, 첫 시도 0.9539와 baseline 0.9228은 서로의 CI 밖에 있어 이 데이터에서 구분 가능한 차이입니다. baseline의 `balanced_accuracy_at_best_cut`는 0.9415로 임계값 0.9421보다 낮았기 때문에, 계획이 판단한 대로 임계값·operating point 조정만으로는 목표에 도달할 수 없는 구조였고 모델 패밀리 교체가 실제로 랭킹 자체를 올렸습니다(`roc_auc` 0.9772 → 0.9909).
- 그 이상은 진단이 아니라 관측된 숫자에 대한 기술에 그칩니다: `cut_headroom`이 0.002905로 매우 작아 기본 결정 규칙이 이미 이 랭킹의 거의 최적 지점에 있고, `train_balanced_accuracy` 1.0 / `train_val_gap` 0.046105로 학습 데이터를 완전히 맞춘 상태입니다. 이 값들은 다음 단계의 근거로만 사용하고, 무엇이 성능을 "제한했다"는 진단으로 읽지 않았습니다 — 그런 진단은 이번 실행에서 내려지지 않았습니다.

## 다음 단계 제안

1. **선택 편향의 크기를 실제로 재기 위해 시드를 바꿔 같은 구성을 재실행하기.** 현재 검증-테스트 차이 -0.0085는 테스트 CI 폭(약 ±0.015)보다 작아 0과 구분되지 않습니다. 동일한 `hist_gbdt` 구성으로 seed만 달리한 실행을 몇 번 돌려 테스트 점수가 임계값 0.9421 위에 안정적으로 머무는지 확인하는 것이, 지금 남은 4회 예산으로 얻을 수 있는 가장 정보량이 큰 결과입니다.
2. **operating point 레버(`class_weight`, `scale_pos_weight`)에는 예산을 쓰지 말 것.** `cut_headroom`이 0.002905이므로 어떤 cut을 골라도 최대 0.0029밖에 올라가지 않으며, 이는 검증 CI 폭보다 훨씬 작아 측정 불가능한 크기입니다. 남은 여지는 랭킹 축에 있습니다.
3. **랭킹 축을 조금 더 밀어보려면 용량/정규화 쪽을 한 번만 시도.** `train_balanced_accuracy` 1.0, `train_val_gap` 0.046105는 학습 데이터를 완전히 적합한 상태를 보여줍니다. `max_iter`를 낮추거나 `l2_regularization`·`max_leaf_nodes`를 조정하는 재튜닝, 또는 `xgboost`로 패밀리를 바꾸는 시도가 후보입니다. 다만 이 executor의 측정 해상도 기준으로 패밀리 내 재튜닝은 `roc_auc` 기준 0.003 수준, 패밀리 교체는 0.002~0.008 수준의 움직임이므로, 현재 `roc_auc` 0.9909에서 유의한 개선이 나올 가능성은 낮다고 보고 기대치를 낮게 잡아야 합니다.
4. **특징 공학이 필요하다고 판단되면 executor 밖에서 처리해야 함.** 이 executor는 비율·상호작용·컬럼 삭제 등 파생 특징을 만들지 않고, 결측 관련 두 컬럼도 이 데이터에서는 결측률이 0.0%라 상수가 되어 쓸모가 없습니다(`missing_indicator`/`missing_count` 모두 false로 실행됨). `capital_run_length_*`의 로그 변환 같은 아이디어를 검증하려면 데이터 카드 단계에서 컬럼을 바꿔 다시 프로파일링해야 하며, 이는 계획으로는 실행할 수 없습니다.

> 참고: 이 데이터셋에는 별도로 기록된 `Data caveats`가 없으므로, 위 제안 중 특정 컬럼·값·split의 신뢰성 때문에 배제되는 항목은 없습니다.