# AutoML 최종 리포트 — bank-marketing (`balanced_accuracy`)

## 요약

목표는 **달성하지 못했습니다**. 목표 기준선은 `balanced_accuracy` 0.8811이었고, 5회 시도 중 최고 성능은 iteration 3의 `hist_gbdt`로 검증 점수 **0.8656 (95% CI 0.8540~0.8752)** — 기준선(logreg 0.6603)보다는 크게 높지만 목표에는 0.0155 부족하며, 목표값이 이 CI 위에 있으므로 그 부족분은 리샘플 노이즈로 설명되지 않습니다. 루프는 최대 반복 횟수(5회)를 모두 소진하고 종료되었으며, critic 진단은 4회 생성되었습니다. 이 모델을 한 번도 사용되지 않은 테스트 20%에서 채점한 결과는 **0.8699 (95% CI 0.8585~0.8797)** 로, 역시 목표 미달입니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `early_stopping=false`, `class_weight='balanced'` | `balanced_accuracy`=0.8586 (CI 0.8456~0.8699), `roc_auc`=0.9328, `cut_headroom`=0.01164, `train_val_gap`=0.0786 | `wrong_model_family` — 컷은 거의 소진(best_cut 0.8702), 남은 거리의 절반 이상이 ranking 축에 있다고 판단 |
| 2 | `xgboost` | `n_estimators=1200`, `learning_rate=0.03`, `max_depth=8`, `min_child_weight=2`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=2.0`, `reg_alpha=0.5`, `scale_pos_weight=7.5` | `balanced_accuracy`=0.8194 (CI 0.8036~0.8319) — iteration 1 대비 paired Δ -0.0392 (CI -0.0506~-0.0296)로 **명확한 후퇴**, `train_val_gap`=0.1689 | `overfitting` — 용량 확대가 gap 0.1689만 사고 `roc_auc`는 0.9319로 정체, recall 0.7164 ≪ specificity 0.9223로 컷이 음성 쪽으로 치우침 |
| 3 | `hist_gbdt` | `max_iter=600`, `learning_rate=0.04`, `max_leaf_nodes=15`, `min_samples_leaf=40`, `l2_regularization=5.0`, `early_stopping=false`, `class_weight={"0":1.0,"1":9.0}` | **최고** `balanced_accuracy`=0.8656 (CI 0.8540~0.8752), `roc_auc`=0.9340, `cut_headroom`=0.00728, `train_val_gap`=0.0402 | `hyperparam` — gap은 절반으로 줄었지만 iteration 1 대비 paired Δ +0.0071 (CI -0.0004~+0.0140)로 **판별 불가**, best_cut 0.8729로 ranking 상한이 굳음 |
| 4 | `hist_gbdt` | `max_iter=900`, `learning_rate=0.025`, `max_leaf_nodes=24`, `min_samples_leaf=25`, `l2_regularization=2.0`, `max_bins=255`, `early_stopping=false`, `class_weight={"0":1.0,"1":8.2}` | `balanced_accuracy`=0.8629 (CI 0.8509~0.8733), `roc_auc`=0.9347, best_cut 0.8698, `cut_headroom`=0.00695 | `wrong_model_family` — iteration 3 대비 paired Δ -0.0028 (CI -0.0084~+0.0028)로 판별 불가, 컷은 사실상 소진, 남은 거리는 ranking에 있다고 재확인 |
| 5 | `random_forest` | `n_estimators=800`, `min_samples_leaf=1`, `min_samples_split=2`, `class_weight='balanced_subsample'`, `bootstrap=true`, `random_state=44`, preprocessing `impute='median'` | `balanced_accuracy`=0.6407 (CI 0.6268~0.6540), recall 0.2996 vs specificity 0.9818, `cut_headroom`=0.2304, best_cut 0.8711, `train_val_gap`=0.3593 | (루프 종료로 진단 없음) |

## 최고 성능 구성

**iteration 3 — `hist_gbdt`** (아래 값은 히스토리에 기록된 *적용된* 하이퍼파라미터·전처리이며, `dropped_hyperparams`는 비어 있습니다.)

```
model: hist_gbdt
hyperparams:
  max_iter: 600
  learning_rate: 0.04
  max_leaf_nodes: 15
  min_samples_leaf: 40
  l2_regularization: 5.0
  early_stopping: false
  class_weight: {"0": 1.0, "1": 9.0}
preprocessing:
  impute: none        # hist_gbdt가 NaN을 직접 분기
  scale: false
  missing_indicator: false
  missing_count: false
train_time_sec: 9.999
```

| 지표 | 검증(20%) | 테스트(한 번도 사용되지 않은 20%) |
|---|---|---|
| `balanced_accuracy` | **0.8656** (CI 0.8540~0.8752) | **0.8699** (CI 0.8585~0.8797) |
| `recall` / `specificity` | 0.8828 / 0.848447 | — |
| `roc_auc` | 0.9340 | — |
| `pr_auc` / `average_precision` | 0.6291 | — |
| `f1` / `precision` | 0.5834 / 0.4356 | — |
| `accuracy` | 0.8525 | — |
| `balanced_accuracy_at_best_cut` / `cut_headroom` | 0.8729 / 0.00728 | — |
| `brier` / `calibration_error` | 0.102816 / 0.133202 | — |
| `train_balanced_accuracy` / `train_val_gap` | 0.9058 / 0.0402 | — |

검증과 테스트의 차이는 0.0043(테스트가 더 높음)입니다. 루프의 모델 선택은 **검증 숫자만** 보고 이루어졌으므로 이 차이가 선택 편향의 크기이며, 검증 점수가 테스트 CI(0.8585~0.8797) 안에 들어가므로 이 행들로는 그 차이를 0과 구분할 수 없습니다. 두 숫자 모두 목표 0.8811 아래이고, 테스트 CI의 상단(0.8797)조차 목표에 닿지 않습니다.

## 원인 분석

**1) 성능을 묶은 축은 operating point가 아니라 ranking입니다.** `balanced_accuracy_at_best_cut`은 어떤 임계값을 골라도 그 모델의 순위가 허용하는 최댓값인데, 이 값이 5회 시도에서 0.8702 / 0.8727 / 0.8729 / 0.8698 / 0.8711로 나왔습니다. 세 개의 서로 다른 family(`hist_gbdt`, `xgboost`, `random_forest`)와 서로 다른 용량 설정에도 상한이 0.87 근방에 고정되어 있고, 목표 0.8811은 그 상한보다 **0.0082 이상 위에** 있습니다. 즉 이 실행에서 임계값을 완벽하게 고를 수 있었다 해도 목표에 도달하지 못했습니다. 이는 goal 스펙 자체와도 일치합니다: `exceeds_ranking_ceiling: true`, `required_ks` 0.7622 대비 baseline `ks` 0.6765이며, 최고 시도의 KS도 critic 계산으로 0.7458 수준에 머물렀습니다.

**2) hist_gbdt 내부 재튜닝은 이 데이터로 구분되지 않습니다.** iteration 1(0.8586, CI 0.8456~0.8699), iteration 3(0.8656, CI 0.8540~0.8752), iteration 4(0.8629, CI 0.8509~0.8733)의 구간은 서로 겹치고, paired Δ도 각각 +0.0071(P 0.973), -0.0028(P 0.193)로 0과 분리되지 않습니다. **따라서 이 세 시도의 순서에서 원인을 읽어서는 안 됩니다** — 잎 개수·학습률·`class_weight` 8.2↔9.0 사이의 움직임은 이 검증 슬라이스가 판별하지 못하는 크기입니다. 실제로 확실히 말할 수 있는 것은 iteration 3에서 `train_val_gap`이 0.0786 → 0.0402로 줄어 과적합 여지가 사라졌다는 사실뿐이고, 그 대가로 점수가 오른 것은 측정되지 않았습니다.

**3) 구간보다 큰 차이는 두 건뿐이고, 둘 다 손실입니다.** iteration 2(`xgboost`, `max_depth=8`×1200라운드)는 `train_val_gap` 0.1689와 함께 -0.0392의 확정된 후퇴였고, `roc_auc`는 0.9319로 iteration 1의 0.9328과 사실상 같았습니다 — 용량은 순위를 사지 못하고 암기만 샀습니다. iteration 5(`random_forest`, 무제한 깊이)는 `train_balanced_accuracy`=1.0, `train_val_gap`=0.3593으로 완전 암기했고, 기본 컷에서 recall 0.2996 / specificity 0.9818로 `balanced_accuracy` 0.6407까지 떨어졌습니다. 다만 그 모델의 best_cut은 0.8711로 다른 family와 같은 띠 안에 있었으므로, 이것은 *순위가 나빴다*는 증거가 아니라 *기본 임계값이 크게 어긋났다*(`cut_headroom` 0.2304)는 증거이며, 동시에 family 교체가 이 특징 집합에서 순위 상한을 올리지 못한다는 세 번째 확인입니다.

**4) critic 진단의 패턴.** 4개의 진단은 `wrong_model_family` → `overfitting` → `hyperparam` → `wrong_model_family` 순서였습니다. 첫 `wrong_model_family` 처방(`xgboost`)은 -0.0392를 지불했고, 마지막 `wrong_model_family` 처방(`random_forest`)은 기본 컷 붕괴로 -0.2249를 지불했습니다. critic 자신이 iteration 3·4의 근거에서 이미 정확히 진단했듯 — "family 교체가 실제로 덮은 best_cut 폭은 0.0002인데 남은 거리는 0.0082, 즉 41배" — family 교체는 이 실행에서 사용 가능한 어떤 크기의 레버도 아니었습니다. 그럼에도 executor가 특징을 파생할 수 없어 ranking 축을 움직일 수단이 남아 있지 않았고, 남은 예산은 판별 불가한 재튜닝과 확정된 손실에 소진되었습니다. 시간은 제약이 아니었습니다(최대 15.1초 / 예산 600초).

**5) `unsupported_claims`에 대하여.** iteration 1~4의 계획에서 `feature_engineering` 문자열이 감지되었습니다. 이는 계획 산문에 대한 부분 문자열 검사이며, 확인 가능한 유일한 계획 텍스트(iteration 3의 `plan_strategy`)는 전처리를 `impute='none'`으로 고정한 채 용량과 가중치만 움직인다고 명시하고 있어 **파생 특징에 의존하지 않았습니다**. 따라서 iteration 3의 점수는 이 executor가 실제로 달성한 값이며, 특징 파생 부재를 "실패 원인"으로 기록하지 않습니다 — 그것은 아래 제안에서 *만들어야 할 능력*으로 다룹니다. 나머지 계획의 원문은 본 리포트에 제공되지 않아, 그 시도들이 무엇을 검증하지 못했는지는 단정할 수 없습니다.

## 다음 단계 제안

1. **`unknown` 센티널을 데이터 카드 단계에서 실제 NaN으로 변환한 뒤 iteration 3 구성을 그대로 재적합하라 (최우선).** 카드 caveat가 명시하듯 V16의 81.8%, V9의 28.8%, V4의 4.1%, V2의 0.6%가 결측 표기 `unknown`인데 변환되지 않아 지금은 one-hot의 일반 수준으로 들어가 있습니다. executor는 재인코딩을 할 수 없으므로 이 수정은 **카드/원본 파일 쪽에서만** 가능하며, 수정되면 `impute='none'`의 `hist_gbdt`가 그 결측 분기를 직접 학습할 수 있게 됩니다. 이것이 이 실행에서 유일하게 손대지 않은 *ranking 축* 레버입니다. 단, 같은 caveat 때문에 현재 V16·V9의 magnitude·`target_corr`·기준선 수치는 `unknown`을 실측치로 계산한 결과이므로, **변환 전에 이 열들의 상관 강도를 근거로 삼는 계획은 세우지 마십시오.**
2. **`missing_indicator`에 반복을 쓰지 마라.** CAN 목록에 확정 사항으로 기재되어 있듯, `impute='none'`으로 NaN을 직접 분기하는 family에 인디케이터를 붙이면 예측이 비트 단위로 동일합니다. 임퓨트를 하는 경우의 값은 확정되지 않았고(랜덤 분할과 연속 분할에서 부호·크기가 무너짐), `missing_count`는 랜덤 분할에서 0.0003 / 0.0000이었습니다. 따라서 제안 1은 "결측 열 추가"가 아니라 **센티널을 진짜 NaN으로 고치는 것**이어야 합니다.
3. **파생 특징을 executor 밖에서 만들어 카드에 넣어라.** best_cut 상한이 세 family에서 0.8698~0.8729로 고정된 것은 "모델을 더 찾아라"가 아니라 "현재 열들이 담은 순위 정보가 소진되었다"는 신호입니다. executor는 비율·상호작용·재인코딩·열 삭제를 하지 않으므로, 후보 파생열(예: `target_corr`가 strong인 V12를 축으로 한 정규화·상호작용 항)은 미리 계산해 카드의 열로 제공해야 하며, 이때도 제안 1을 먼저 끝내야 `unknown` 오염이 파생열로 전파되지 않습니다.
4. **동일 예산으로 hist_gbdt/xgboost의 용량·`class_weight` 재탐색은 반복하지 마라.** iteration 1·3·4의 CI가 모두 겹쳐 이 슬라이스가 구분하지 못하며, `cut_headroom`은 이미 0.00695~0.00728로 남은 0.0155보다 작습니다. 가중치 미세조정으로 살 수 있는 최대치는 목표까지 거리의 절반 미만입니다. 아울러 목표 0.8811 자체가 baseline의 `ranking_ceiling` 0.8382를 넘도록 파생된 값(`exceeds_ranking_ceiling: true`, `ks_shortfall` 0.0857)이므로, **제안 1·3으로 입력 정보를 늘리지 않은 채 이 기준을 유지하는 것은 달성 불가한 목표를 재확인하는 데 예산을 쓰는 일**입니다 — 기준 재협상 또는 정보 추가 중 하나를 먼저 결정하십시오.
5. (부수 사항) 확률값을 그대로 쓸 계획이라면 최고 구성의 `calibration_error`=0.1332, `brier`=0.1028을 감안하십시오. 이 실행에는 재보정 레버가 없으므로, 확률 자체가 필요하면 루프 밖에서 보정 단계를 별도로 구축해야 합니다.