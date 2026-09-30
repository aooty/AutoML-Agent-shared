## 요약

목표는 **달성하지 못했습니다**. 목표 기준선은 `balanced_accuracy` 0.9194였으나, 5회 예산 중 4회를 사용해 얻은 최고 검증 점수는 iteration 2의 `xgboost`가 기록한 **0.8379 (95% CI 0.8289~0.8460)** 로, 기준선까지 약 0.0815가 부족했습니다. 한 번도 사용되지 않은 최종 테스트 20%에서의 점수는 **0.8376 (95% CI 0.8284~0.8470)** 으로, 검증 점수와 사실상 동일했습니다. 루프는 연속 미개선(정체)으로 4회 시도 후 조기 종료되었고, critic 진단은 3회 생성되어 세 번 모두 동일한 `wrong_model_family` 판정을 냈습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `learning_rate=0.06`, `max_iter=600`, `max_leaf_nodes=63`, `min_samples_leaf=20`, `l2_regularization=1.0`, `class_weight='balanced'`, `early_stopping=False` / `impute='none'`, `scale=False` | `balanced_accuracy` 0.8368 (CI 0.8283~0.8446), `roc_auc` 0.9233, `pr_auc` 0.8237, `cut_headroom` 0.002472, `train_val_gap` 0.100165 | `wrong_model_family` — 운용점은 이미 소진(`cut_headroom` 0.0025 / 격차 0.0826), 남은 격차는 랭킹 축. `xgboost`로 정규화 강화 처방 |
| 2 | `xgboost` | `learning_rate=0.05`, `n_estimators=800`, `max_depth=6`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=5.0`, `scale_pos_weight=2.0`, `early_stopping_rounds=50` / `impute='none'`, `scale=False` | **최고** `balanced_accuracy` 0.8379 (CI 0.8289~0.8460), `roc_auc` 0.9272, `pr_auc` 0.8295, `balanced_accuracy_at_best_cut` 0.8467, `cut_headroom` 0.008753, `train_val_gap` 0.033761 (626라운드에서 early stop) | `wrong_model_family` — 과적합은 해소(0.1002→0.0338)됐지만 랭킹 상승은 문서화된 계열 교체 폭(0.0022~0.0077 roc_auc) 안. `random_forest` 처방 |
| 3 | `random_forest` | `n_estimators=600`, `min_samples_leaf=2`, `class_weight='balanced_subsample'`, `n_jobs=-1` / `impute='median'`, `scale=False` | `balanced_accuracy` 0.8290 (CI 0.8200~0.8385), `roc_auc` 0.9173, `pr_auc` 0.7991, `balanced_accuracy_at_best_cut` 0.8340, `cut_headroom` 0.005005 | `wrong_model_family` — 트리 계열 3종의 best-cut 편차가 0.0127에 불과, 구조가 다른 `mlp` 처방 |
| 4 | `mlp` | `hidden_layer_sizes=[256,128]`, `alpha=0.0005`, `learning_rate_init=0.001`, `batch_size=256`, `max_iter=400`, `early_stopping=True` / `impute='median'`, `scale=True` | `balanced_accuracy` 0.7823 (CI 0.7719~0.7927), `roc_auc` 0.9075, `pr_auc` 0.7737, `cut_headroom` 0.040868, `train_val_gap` 0.007825 (15회에서 early stop) | (없음 — 평가 직후 루프 종료) |

## 최고 성능 구성

**iteration 2 — `xgboost`** (아래 값은 executor가 실제로 만든 추정기에서 읽은 적용값입니다. `dropped_hyperparams`는 비어 있습니다.)

```
hyperparams:
  learning_rate: 0.05
  n_estimators: 800          # early_stopping_rounds=50, stopped_at_iter=626
  max_depth: 6
  min_child_weight: 5
  subsample: 0.8
  colsample_bytree: 0.8
  reg_lambda: 5.0
  scale_pos_weight: 2.0
  early_stopping_rounds: 50

preprocessing:
  impute: none               # xgboost가 NaN을 직접 분기
  scale: false
  missing_indicator: false
  missing_count: false

internal_validation: held_out_rows=2931, fit_rows=26373, validation_fraction=0.1
train_time_sec: 8.757
```

검증(20%) 지표: `balanced_accuracy` **0.8379 (95% CI 0.8289~0.8460)**, `recall` 0.7963, `specificity` 0.879575, `precision` 0.6753, `f1` 0.7308, `accuracy` 0.8597, `roc_auc` 0.9272, `pr_auc` / `average_precision` 0.8295, `brier` 0.095965, `calibration_error` 0.060817, `balanced_accuracy_at_best_cut` 0.8467, `cut_headroom` 0.008753, `train_val_gap` 0.033761.

최종 held-back 테스트(20%, 루프 중 한 번도 읽지 않은 행): `balanced_accuracy` **0.8376 (95% CI 0.8284~0.8470)**. 검증 0.8379와의 차이 **+0.0004** 가 이 실행의 선택 편향 크기입니다 — 루프는 검증 숫자만 보고 최고 시도를 골랐기 때문입니다. 다만 검증 점수가 테스트 CI 안에 들어오므로, 이 행 수에서는 그 차이를 0과 구분할 수 없습니다. 어느 쪽 숫자로도 0.9194 기준선에는 도달하지 못합니다.

## 원인 분석

**세 번의 critic 진단은 모두 `wrong_model_family`였고, 세 번 모두 같은 사실을 근거로 삼았습니다: 남은 격차가 운용점(operating point)이 아니라 랭킹 축에 있다는 것.** iteration 1의 `cut_headroom`은 0.002472, iteration 2는 0.008753, iteration 3은 0.005005였습니다. 즉 임계값을 최적으로 옮겨도 벌 수 있는 양은 0.0815 격차의 3~11%에 불과합니다. 최고 시도의 `balanced_accuracy_at_best_cut` 0.8467을 그대로 인정해도 기준선까지 0.0727이 남습니다. `class_weight`/`scale_pos_weight`는 이 실행에서 쓸 수 있는 유일한 불균형 레버였고, iteration 2에서 `recall` 0.7963 대 `specificity` 0.8796이라는 근접한 균형을 만들면서 이미 그 레버의 대부분을 소비했습니다.

**서로 구분되지 않는 시도들:** iteration 1(0.8368, CI 0.8283~0.8446), iteration 2(0.8379, CI 0.8289~0.8460), iteration 3(0.8290, CI 0.8200~0.8385)의 구간은 모두 겹칩니다. critic도 iteration 2의 +0.0011을 "증거가 아니다(짝지은 Δ 95% CI −0.0045~+0.0065, P(better) 0.640)"라고 기록했습니다. 따라서 `hist_gbdt` → `xgboost` → `random_forest` 사이의 `balanced_accuracy` 움직임은 이 데이터로 설명할 수 있는 개선이 아니며, 이야기로 만들 대상이 아닙니다. 구간 폭보다 확실히 큰 차이는 하나뿐입니다: **iteration 4의 `mlp` 0.7823 (CI 0.7719~0.7927)** 은 세 트리 모델 어느 구간과도 겹치지 않고 명확히 낮습니다. 다만 그 시도는 `max_iter=400` 중 `stopped_at_iter=15`에서 early stop되어 학습이 거의 진행되지 않았고(`train_balanced_accuracy` 0.7902, `train_val_gap` 0.007825 — 과적합이 아니라 과소적합), "비트리 결정면도 같은 대역에 떨어지는가"라는 원래의 질문에는 제대로 답하지 못했습니다.

**구조적 한계.** 규정된 목표 자체가 참조값에서 이미 도달 불가 신호를 달고 있습니다: `ranking_ceiling` 0.8266에 대해 `exceeds_ranking_ceiling: true`, `required_ks` 0.8388 대비 baseline `ks` 0.6531, `ks_shortfall` 0.1857. 이번 실행이 도달한 최고 랭킹은 `roc_auc` 0.9272 / 암시 KS 0.6934였습니다. 정규화는 요구받은 일을 정확히 했고(iteration 2에서 `train_val_gap` 0.1002 → 0.0338), 그렇게 확보한 용량이 랭킹으로 전환된 폭은 `roc_auc` 0.9233 → 0.9272, `pr_auc` 0.8237 → 0.8295 — 문서화된 계열 교체 폭(0.0022~0.0077 roc_auc) 안입니다. 즉 **결정적인 제약은 하이퍼파라미터나 임계값이 아니라, executor가 파생 특성을 만들 수 없고 단일 추정기 하나만 적합할 수 있는 환경에서 이 14개 열이 담고 있는 순서 정보의 상한**입니다. 예산은 제약이 아니었습니다(최대 19.6초, 한도 600초).

`unsupported_claims`에 대해: iteration 1과 iteration 4의 plan에 `feature_engineering` 문자열 플래그가 붙었지만, 이 플래그는 계획 산문에 대한 substring 검사이며 해당 두 iteration의 plan 원문은 이 보고서에 제공되지 않았습니다. 따라서 그 계획이 파생 특성에 *의존했는지*는 확인할 수 없고, 두 시도가 무엇을 잃었다고 쓰지 않습니다. 원문이 남아 있는 iteration 2의 plan은 파생 특성 부재를 "envelope의 한계"로 *언급만* 했으며 그것에 기대지 않았습니다 — 특성 파생 능력의 부재는 실패 원인이 아니라, 아래 "다음 단계"에서 **만들어야 할 것**으로 다룹니다.

## 다음 단계 제안

1. **기준선 0.9194의 타당성을 먼저 재검토할 것 (최우선).** 이 값은 `derived`이며 참조 블록 자체가 `exceeds_ranking_ceiling: true`, `ks_shortfall: 0.1857`을 기록하고 있습니다. 구조적으로 다른 3개 계열(`hist_gbdt`, `xgboost`, `random_forest`)의 `balanced_accuracy_at_best_cut`이 0.8340~0.8467 범위에 몰려 있고, 비트리(`mlp`)도 0.8232였습니다. 남은 1회 예산을 이 envelope 안에서 더 쓰는 것은 0.008 수준의 움직임만 만들 뿐이므로, **목표를 이 데이터의 랭킹 상한에 맞게 재설정하거나(예: 0.85 대역), 아래 2번처럼 envelope 자체를 확장한 뒤 재도전**하는 것이 정직한 경로입니다.
2. **랭킹 축을 움직이려면 executor 밖에서 특성 파생 파이프라인을 만들어 카드에 반영할 것.** 현재 executor는 열 조합·비율·상호작용·재인코딩을 일절 하지 않으며(`missing_indicator`/`missing_count` 예외뿐), 남은 격차의 89~97%가 랭킹 축에 있습니다(`cut_headroom` 0.0025~0.0088 대 격차 0.0815). adult 열 구성상 시도할 가치가 있는 것은 `capital-gain`/`capital-loss`(skew high, sub_unit 스케일)의 비선형 변환·binning, `education-num` × `hours-per-week`, `marital-status` × `occupation` 같은 교차 항이며, 이들은 계획에 쓸 수 없고 **입력 카드 단계에서 컬럼으로 만들어야** 합니다. 이것이 랭킹 상한을 올릴 수 있는지 검증하기 전까지는 어떤 계열 교체도 같은 0.83~0.85 대역을 반복할 것으로 봅니다.
3. **`mlp` 결과는 결론으로 쓰지 말고 한 번 제대로 다시 측정할 것.** iteration 4는 `stopped_at_iter=15 / max_iter=400`으로 사실상 학습되지 않은 모델이며(`train_balanced_accuracy` 0.7902), "비트리 결정면도 같은 대역인가"라는 검증 목적을 달성하지 못했습니다. 남은 1회 예산을 쓴다면 `early_stopping=False`(또는 `learning_rate_init`를 낮추고 반복을 충분히 확보) + `impute='median'`, `scale=True`로 동일 계열을 재적합해, 스케일 민감 계열의 랭킹(`roc_auc`)이 0.9272를 넘는지만 확인하는 것이 가장 정보량이 큽니다.
4. **확률값을 하류에서 쓸 계획이라면 보정을 루프 밖에 둘 것.** 최고 모델의 `brier` 0.095965, `calibration_error` 0.060817 — 예측 확률이 평균 약 6%p 어긋나 있고, 이 실행 환경에는 재보정 레버가 없습니다(`CalibratedClassifierCV`·임계값 이동 모두 불가). 랭킹 지표(`roc_auc` 0.9272)는 보정과 무관하므로 모델 선택 판단은 그대로 유효하지만, 운용점을 실제로 옮겨야 한다면 `balanced_accuracy_at_best_cut` 0.8467과 현재 0.8379의 차이(0.008753)까지가 이 랭킹에서 임계값으로 회수 가능한 전부임을 전제로 별도 파이프라인에서 처리해야 합니다.

> 참고: 데이터 카드에 별도로 기록된 caveat은 없으므로(“없음”), 위 제안 중 특정 열이나 분할의 신뢰성에 의존해 무효화되는 항목은 없습니다. 다만 `missing_indicator`는 `impute='none'`과 함께 쓰면 비트 단위로 중복이라는 점이 이미 확정되어 있어(최고 구성이 정확히 그 조합), 그 레버에 예산을 쓰는 것은 권하지 않습니다.