# AutoML 실행 리포트 — bank-marketing (balanced_accuracy)

## 요약

목표 지표(`balanced_accuracy` ≥ 0.8817)는 **검증 슬라이스 기준으로 첫 시도에서 달성**되었습니다. 최고 구성은 iteration 1의 `hist_gbdt`로 검증 `balanced_accuracy` = **0.8826** (95% CI 0.8730~0.8916)이며, 카드의 baseline `logreg` 0.662 대비 큰 폭의 상승입니다. 총 5회 예산 중 **1회**만 사용하고 `goal_reached`로 종료되었습니다. 다만 한 번도 사용되지 않은 최종 테스트 20%에서의 같은 모델 점수는 **0.8635** (95% CI 0.8527~0.8727)로, 검증 점수보다 0.0192 낮고 임계값 0.8817 아래입니다 — 이 실행이 실제로 입증한 성능은 후자입니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=500`, `learning_rate=0.06`, `max_leaf_nodes=63`, `l2_regularization=1.0`, `class_weight='balanced'`, `early_stopping=True`, `validation_fraction=0.1` (실제 89 iter에서 조기 종료) | status `ok` / 검증 `balanced_accuracy`=0.8826 (CI 0.8730~0.8916), `roc_auc`=0.9409, `recall`=0.9045, `specificity`=0.8607, `cut_headroom`=0.000671, `train_val_gap`=0.0317, 학습 4.939초 | 없음 — 첫 시도가 임계값을 넘어 루프가 종료되어 critic이 실행되지 않음 |

## 최고 성능 구성

- **model**: `hist_gbdt`
- **hyperparams** (실제 적용값, `dropped_hyperparams` 없음):
  ```json
  {
    "max_iter": 500,
    "learning_rate": 0.06,
    "max_leaf_nodes": 63,
    "l2_regularization": 1.0,
    "class_weight": "balanced",
    "early_stopping": true,
    "validation_fraction": 0.1
  }
  ```
- **preprocessing** (해당 attempt에 기록된 적용값):
  ```json
  {
    "impute": "median",
    "scale": false,
    "missing_indicator": false,
    "missing_count": false
  }
  ```
- **internal_validation**: `fit_rows`=24413, `held_out_rows`=2713, `stopped_at_iter`=89 / `max_iter`=500
- **프로토콜**: stratified 3-way split, seed 42, train 60% / validation 20% / test 20% (카드 baseline과 동일 분할)

| 지표 | 검증 (20%) | 최종 테스트 (미사용 20%) |
|---|---|---|
| `balanced_accuracy` | **0.8826** (CI 0.8730~0.8916) | **0.8635** (CI 0.8527~0.8727) |
| `roc_auc` | 0.9409 | — |
| `pr_auc` / `average_precision` | 0.6434 | — |
| `f1` | 0.6121 | — |
| `accuracy` | 0.8658 | — |
| `precision` / `recall` | 0.4625 / 0.9045 | — |
| `specificity` | 0.8607 | — |
| `balanced_accuracy_at_best_cut` / `cut_headroom` | 0.8833 / 0.000671 | — |
| `brier` / `calibration_error` | 0.0951 / 0.1250 | — |
| `train_balanced_accuracy` / `train_val_gap` | 0.9143 / 0.0317 | — |

검증 0.8826과 테스트 0.8635의 차이 **0.0192**가 이 실행의 선택 편향 크기입니다. 루프는 검증 점수만 보고 목표 달성을 판정했고, 테스트 점수는 루프 종료 후 단 한 번 계산되었습니다. 테스트 CI 상한(0.8727)이 임계값 0.8817보다 낮으므로, **held-back 행에서 임계값 초과가 입증된 것은 아닙니다.** 보고할 성능 수치로는 0.8635를 쓰는 것이 정확합니다.

## 원인 분석

- **critic 진단은 0건입니다.** 시도 1회, 진단 0회로, 첫 계획이 곧바로 임계값을 넘겨 루프가 종료되었기 때문에 critic이 한 번도 실행되지 않았습니다. 따라서 "verdict 간 패턴"은 존재하지 않으며, 이 점수는 **첫 계획 하나가 낸 결과**입니다. 이 실행은 진단·재계획 경로가 도움이 된다는 증거도, 해가 된다는 증거도 제공하지 않습니다.
- 또한 시도가 1회뿐이므로 **시도 간 비교로 성능 제한 요인을 지목할 근거가 없습니다.** 아래는 진단이 아니라 iteration 1의 계측값을 그대로 읽은 것입니다.
  - `cut_headroom`=0.000671 — `class_weight='balanced'`가 만든 기본 결정 규칙이 이 ranking에서 얻을 수 있는 최적 컷(`balanced_accuracy_at_best_cut`=0.8833)에 사실상 붙어 있습니다. 즉 operating point 축은 이미 소진된 상태이고, 검증 점수를 더 올릴 여지는 ranking 축(모델 family / 용량 / 피처)에 있습니다.
  - `train_val_gap`=0.0317, 조기 종료가 89 iteration에서 걸린 점을 보면 500 iteration의 용량을 다 쓰지 않았습니다. 이것이 과적합 때문인지 아닌지는 한 번의 fit으로는 구분되지 않습니다.
  - `calibration_error`=0.1250, `brier`=0.0951 — 확률값 자체는 평균적으로 12.5%p 어긋나 있습니다. `balanced_accuracy`·`roc_auc`는 이에 영향받지 않으므로 이번 점수의 원인은 아니지만, 확률을 그대로 쓰려는 용도에는 부적합합니다(이 executor에는 recalibration 레버가 없습니다).
- **검증 CI 폭(약 0.019)과 검증→테스트 차이(0.0192)의 관계**: 이 슬라이스의 측정 해상도가 그 정도이므로, 앞으로 0.005 수준의 개선을 보고할 때는 그것이 개선인지 리샘플 노이즈인지 구분되지 않는다는 점을 전제해야 합니다.
- `plan.unsupported_claims`에 `feature_engineering`이 기록되어 있으나, iteration 1의 계획 본문은 스케일링과 missingness 컬럼을 **쓰지 않겠다고 명시**했을 뿐 파생 피처에 의존하지 않았습니다. 이는 계획 산문에 대한 substring 검사가 걸린 것으로 읽는 것이 맞고, 이 attempt가 그 때문에 무엇을 잃었다고 볼 근거는 없습니다.

## 다음 단계 제안

1. **`unknown` 문자열 마커를 데이터 단계에서 정리한 뒤 다시 프로파일링할 것 (최우선).** 카드 caveat대로 V16(81.8%), V9(28.8%), V4(4.1%), V2(0.6%)의 `unknown`은 변환되지 않은 실측값으로 집계·baseline에 들어갔습니다. 즉 이 열들의 `target_corr`를 근거로 한 어떤 피처 판단도 지금은 신뢰할 수 없고, executor는 재인코딩·열 삭제를 할 수 없으므로 **루프 밖에서 CSV/카드를 고쳐야** 합니다. `unknown`을 진짜 결측(NaN)으로 바꾸면 `hist_gbdt`에 `impute: "none"`을 주어 NaN 분기를 직접 학습시키는 경로가 열리고, V16처럼 81.8%가 마커인 열은 카드에서 빼는 판단도 비로소 근거를 갖게 됩니다. 지금 카드의 `missing.overall_rate`=0.0은 이 마커가 결측으로 세어지지 않은 결과이므로, **현 상태에서 `missing_indicator`/`missing_count`를 켜는 것은 상수 열을 추가하는 일이며 iteration을 쓸 가치가 없습니다.**
2. **남은 예산은 operating point가 아니라 ranking 축에만 쓸 것.** `cut_headroom`=0.000671이므로 `class_weight` 비율 탐색(예: 명시적 `{"0":1,"1":k}`)은 회수할 여지가 거의 없습니다. 대신 (a) `xgboost`로의 family 교체, (b) `hist_gbdt` 내에서 `learning_rate`를 낮추고 `max_leaf_nodes`·`l2_regularization`을 조정해 89 iteration에서 멈춘 조기 종료 지점을 뒤로 미는 재튜닝을 시도하는 것이 축에 맞습니다. 다만 참고 계측에서 family 교체는 roc_auc 0.0022~0.0077, family 내 재튜닝은 0.0032 규모였고 이는 이 슬라이스의 CI 폭보다 작으므로, **단일 attempt의 개선이 검증 CI 안에서 구분되지 않을 가능성을 미리 인정하고 판독해야 합니다.**
3. **테스트 점수 기준으로 목표 재확인.** 이번 실행에서 검증 0.8826과 테스트 0.8635의 간극이 0.0192였으므로, 첫 시도 통과로 루프를 끝내면 held-back 성능이 임계값 아래일 수 있습니다. 예산이 남았다면 임계값을 넘은 시점에서 멈추기보다 **검증 마진을 0.02 이상 확보할 때까지 ranking 축 시도를 계속**하는 운영이 이 데이터에서는 안전합니다(분할·seed·goal metric은 실행 구성이라 변경 대상이 아님).
4. **확률값을 쓰려면 recalibration은 루프 밖에서.** `calibration_error`=0.1250은 측정되었을 뿐 교정되지 않으며 executor에 교정 레버가 없습니다. 임계값·확률을 실제 의사결정에 쓸 계획이라면 최종 모델을 별도로 재적합/교정하고, `balanced_accuracy` 목표 자체는 이와 무관하다는 점을 함께 기록해 두십시오.