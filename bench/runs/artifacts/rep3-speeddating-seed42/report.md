## 요약

목표는 달성되었습니다. 목표 지표 `balanced_accuracy`의 기준선은 0.7514였고, iteration 2의 `hist_gbdt`가 검증 슬라이스에서 **0.7864 (95% CI 0.7568~0.8140)** 를 기록해 기준을 넘었습니다. 한 번도 사용되지 않은 최종 테스트 20%에서의 점수는 **0.7901 (95% CI 0.7650~0.8158)** 로, 이 값이 이번 실행이 실제로 입증한 성능입니다. 총 5회 예산 중 2회 시도, critic 진단 1회로 종료되었습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `learning_rate` 0.06, `max_iter` 400, `max_leaf_nodes` 31, `min_samples_leaf` 20, `l2_regularization` 1.0, `early_stopping` false, `class_weight` {0:1.0, 1:3.5} / preprocessing: `impute` none, `scale` false, `missing_indicator` false, `missing_count` true | `balanced_accuracy` 0.6932 (CI 0.6619~0.7211) — 기준 0.7514 미달. `roc_auc` 0.8719, `pr_auc` 0.5918, `balanced_accuracy_at_best_cut` 0.7952, `cut_headroom` 0.1020, `recall` 0.4493 vs `specificity` 0.9371, `train_val_gap` 0.3067 | `failure_type: hyperparam`. 랭킹 축은 이미 baseline 한계(0.7712)를 넘었고(`balanced_accuracy_at_best_cut` 0.7952), 부족분 0.0582보다 `cut_headroom` 0.1020이 더 크므로 문제는 operating point. `class_weight`의 양성 가중치를 3.5 → 약 8로 크게 올리고 나머지는 고정하라 |
| 2 | `hist_gbdt` | `learning_rate` 0.06, `max_iter` 250, `max_leaf_nodes` 15, `min_samples_leaf` 40, `l2_regularization` 3.0, `early_stopping` false, `class_weight` {0:1.0, 1:9.0} / preprocessing: `impute` none, `scale` false, `missing_indicator` false, `missing_count` true | `balanced_accuracy` **0.7864** (CI 0.7568~0.8140) — 기준 달성. `roc_auc` 0.8734, `pr_auc` 0.6121, `recall` 0.7464, `specificity` 0.8264, `cut_headroom` 0.0133, `train_val_gap` 0.1495 | 없음 (목표 달성으로 루프 종료 — 마지막 시도는 진단 대상이 아님) |

## 최고 성능 구성

아래 값은 iteration 2에서 executor가 실제로 만든 파이프라인/추정기에서 읽은 적용값입니다.

- **model**: `hist_gbdt`
- **hyperparams**: `learning_rate` 0.06, `max_iter` 250, `max_leaf_nodes` 15, `min_samples_leaf` 40, `l2_regularization` 3.0, `early_stopping` false, `class_weight` {"0": 1.0, "1": 9.0}
- **preprocessing (applied)**: `impute` none (hist_gbdt가 NaN을 직접 분기), `scale` false, `missing_indicator` false, `missing_count` true
- **프로토콜**: stratified 3-way split, seed 42, train 60% / val 20% / test 20%
- **train_time_sec**: 5.929 (`dropped_hyperparams` 없음)

검증(선택에 사용된) 점수와 held-back 테스트 점수:

| 지표 | 검증 20% | 테스트 20% (1회 측정) |
|---|---|---|
| `balanced_accuracy` | 0.7864 (CI 0.7568~0.8140) | **0.7901 (CI 0.7650~0.8158)** |

두 값의 차이(보고값 -0.0037)가 이번 실행의 선택 편향 크기입니다. 선택은 전적으로 검증 숫자로 이루어졌고, 검증 점수가 테스트 CI 안에 들어오므로 이 행들만으로는 차이를 0과 구분할 수 없습니다.

검증 슬라이스의 나머지 지표: `f1` 0.5683, `accuracy` 0.8132, `precision` 0.4588, `recall` 0.7464, `specificity` 0.8264, `roc_auc` 0.8734, `pr_auc` / `average_precision` 0.6121, `balanced_accuracy_at_best_cut` 0.7997, `cut_headroom` 0.0133, `brier` 0.1283, `calibration_error` 0.1339, `train_balanced_accuracy` 0.9359, `train_val_gap` 0.1495.

## 원인 분석

critic 진단은 1회 존재하며, 그 진단의 축 판정은 데이터로 확인됩니다.

- **iteration 1의 부족분은 랭킹이 아니라 operating point였습니다.** iteration 1에서 이미 `roc_auc` 0.8719(baseline 0.8505), `pr_auc` 0.5918(0.5507), `balanced_accuracy_at_best_cut` 0.7952(baseline의 한계 0.7712)를 기록했습니다. 즉 그 시점의 랭킹을 어디서 자르기만 해도 기준 0.7514는 이미 넘을 수 있었고, 실제 점수 0.6932와의 거리 `cut_headroom` 0.1020은 남은 부족분 0.0582보다 컸습니다. `recall` 0.4493 vs `specificity` 0.9371이라는 비대칭이 그 자체로 컷이 보수적인 쪽에 놓여 있었음을 보여줍니다.
- **두 시도는 구간이 겹치지 않으므로 개선은 실재합니다.** iteration 1 CI 상한 0.7211 < iteration 2 CI 하한 0.7568이므로, 0.6932 → 0.7864의 이동은 리샘플 잡음으로 설명되지 않습니다.
- **그 개선은 랭킹 축이 아니라 컷 이동에서 나왔습니다.** `roc_auc`는 0.8719 → 0.8734로 0.0015만 움직였는데, 이 환경에서 roc_auc 차이의 분해 해상도는 대략 0.003~0.006이므로 랭킹은 사실상 변하지 않았다고 읽어야 합니다. `balanced_accuracy_at_best_cut`도 0.7952 → 0.7997로 거의 그대로입니다. 반면 `cut_headroom`은 0.1020 → 0.0133으로 붕괴하고 `recall`/`specificity`가 0.4493/0.9371 → 0.7464/0.8264로 교차 지점 근처까지 이동했습니다. 즉 `balanced_accuracy` 상승분은 거의 전부 operating point이며, critic의 진단 방향은 옳았습니다.
- **다만 iteration 2는 두 개의 레버를 동시에 움직였으므로 귀속이 섞여 있습니다.** critic이 제시한 `concrete_changes`는 `class_weight` {0:1, 1:8}만 바꾸고 나머지를 그대로 두라는 것이었지만, 실제 실행은 `class_weight` 9.0에 더해 용량 축소(`max_leaf_nodes` 31→15, `max_iter` 400→250, `min_samples_leaf` 20→40, `l2_regularization` 1.0→3.0)를 함께 적용했습니다. `train_val_gap`은 0.3067 → 0.1495로 줄었지만, 얻은 0.0932 중 무엇이 가중치이고 무엇이 용량인지는 이 두 행으로 분리되지 않습니다.
- **확률값의 품질은 나빠졌습니다.** `brier` 0.1037 → 0.1283, `calibration_error` 0.0577 → 0.1339. 무거운 클래스 가중치가 확률을 양성 쪽으로 밀어낸 결과이며, 이 executor에는 재보정 레버가 없습니다. 랭킹 지표(`roc_auc`, `pr_auc`)와 `balanced_accuracy` 자체에는 영향이 없지만, 예측값을 "확률"로 소비할 계획이라면 이 구성 그대로는 부적합합니다.
- **iteration 1 계획에는 `unsupported_claims: ["feature_engineering"]`이 기록되어 있으나, 해당 계획 본문이 이 기록에 남아 있지 않아 계획이 그 기능에 실제로 의존했는지 확인할 수 없습니다.** 이 플래그는 문자열 검사이므로, 이것을 iteration 1의 손실 원인으로 귀속하지 않습니다. executor는 파생/조합 피처를 만들지 않으므로, 피처 엔지니어링은 "실패한 것"이 아니라 "아직 없는 역량"으로 아래 제안에 둡니다.

## 다음 단계 제안

1. **iteration 2의 두 변경을 분리하는 ablation 1회.** iteration 1의 원래 용량(`max_iter` 400, `max_leaf_nodes` 31, `min_samples_leaf` 20, `l2_regularization` 1.0)에 `class_weight` {0:1, 1:9}만 얹어 재실행하십시오. 현재 기록으로는 0.0932의 개선이 가중치 때문인지 용량 축소 때문인지 귀속되지 않으며, 이 한 줄이 그 귀속을 확정합니다. 재현/운영 시 어느 하이퍼파라미터를 지켜야 하는지가 여기서 결정됩니다.
2. **operating point 레버는 사실상 소진되었으므로 더 짜내지 마십시오.** `cut_headroom`이 0.0133까지 내려왔고 `balanced_accuracy_at_best_cut`은 0.7997입니다. 즉 `class_weight`를 더 올려서 얻을 수 있는 최대치는 0.013 남짓이고, 이는 이 검증 슬라이스의 CI 폭(±0.03 수준)보다 작아 개선으로 측정되지도 않습니다. 미세 조정(7~11 사이 탐색)에 예산을 쓰는 것은 리샘플을 설명하는 일에 가깝습니다.
3. **추가 이득이 필요하면 랭킹 축을 건드리십시오 — 같은 파이프라인에서 `xgboost` + `scale_pos_weight`.** `impute` none, `scale` false, `missing_count` true를 그대로 두고 family만 바꾸는 단일 변경으로, 남은 격차가 랭킹에 있는지 확인합니다. 단, 이 환경에서 family 교체가 `roc_auc`에 준 폭은 0.0022~0.0077로 관측되었고 해상도는 0.003~0.006이므로, 작은 쪽 결과가 나오면 "구분되지 않음"으로 보고해야 합니다. 큰 도약을 기대하는 근거는 없습니다.
4. **`missing_indicator`에는 예산을 쓰지 말고, 확률 소비가 필요하면 루프 밖에서 보정하십시오.** `impute: none`과 함께 쓰인 `missing_indicator`는 NaN 분기와 중복이어서 예측이 비트 단위로 동일했다는 점이 이미 정리되어 있으므로 시도 대상이 아닙니다(`missing_count`는 이미 켜져 있습니다). 대신 `calibration_error` 0.1339가 문제가 되는 용도라면, 재보정과 임계값 선택은 이 executor의 역량 밖이므로 루프 외부 파이프라인으로 구축해야 합니다 — 같은 이유로 `field`(high cardinality로 드롭됨) 재인코딩이나 파생 피처도 executor 내부가 아니라 데이터 카드 단계에서 해결해야 하는 항목입니다. 이 데이터에는 별도로 기록된 caveat이 없으므로 위 제안들을 무효화하는 열/분할 신뢰성 문제는 없습니다.