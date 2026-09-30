## 요약

목표(`balanced_accuracy` ≥ 0.7598, maximize)는 **달성**되었습니다. 2회 시도 중 iteration 2의 `hist_gbdt`가 검증 `balanced_accuracy` **0.7710** (95% CI 0.7471~0.8001)을 기록해 기준선 logreg의 0.6798과 목표선 0.7598을 모두 넘었습니다. 같은 모델을 루프 내내 한 번도 쓰이지 않은 held-back 테스트 20%에서 한 번 채점한 결과는 **0.7969** (95% CI 0.7708~0.8206)로, 이 숫자가 이 실행이 실제로 입증한 성능입니다. 5회 예산 중 2회만 사용했고, critic 진단은 1회 발생했습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_depth=6`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `min_samples_leaf=20`, `early_stopping=false`, `class_weight={0:1,1:4}`; preprocessing `impute: none`, `scale: false` | `balanced_accuracy=0.7415` (CI 0.7114~0.7768) — 목표선 미달(-0.0183). `roc_auc=0.8566`, `balanced_accuracy_at_best_cut=0.7827`, `cut_headroom=0.0412`, `recall=0.5616` vs `specificity=0.9214`, `train_val_gap=0.2579` | `data_issue`: 랭킹 축은 이미 충분(baseline ranking ceiling 0.7558 초과), 남은 것은 operating point. `cut_headroom`이 필요 격차의 225%이고 양성 클래스가 과소 호출됨 → `class_weight`를 4→9로 올리고 동시에 용량 축소(`max_iter=250`, `max_leaf_nodes=15`, `max_depth=4`, `l2_regularization=5.0`, `min_samples_leaf=40`), preprocessing은 불변 |
| 2 | `hist_gbdt` | `max_iter=250`, `learning_rate=0.06`, `max_depth=4`, `max_leaf_nodes=15`, `l2_regularization=5.0`, `min_samples_leaf=40`, `early_stopping=false`, `class_weight={0:1,1:9}`; preprocessing `impute: none`, `scale: false` | **`balanced_accuracy=0.7710`** (CI 0.7471~0.8001) — 목표 달성. `roc_auc=0.8581`, `balanced_accuracy_at_best_cut=0.7868`, `cut_headroom=0.0158`, `recall=0.7862` vs `specificity=0.7557`, `train_val_gap=0.1409`, `calibration_error=0.1783` | (없음 — 평가 직후 목표 달성으로 루프 종료) |

두 시도 모두 `status: ok`, `dropped_hyperparams`는 비어 있어 제안한 하이퍼파라미터가 전부 그대로 적용되었습니다. iteration 1의 plan에는 `unsupported_claims: ["feature_engineering"]` 플래그가 있으나, 이는 plan 산문에 대한 substring 검사이며 이 기록에는 해당 plan 원문이 남아 있지 않아 실제로 파생 피처에 의존했는지는 확인할 수 없습니다. 적용된 preprocessing에는 파생 피처가 없고 executor는 애초에 그 기능을 제공하지 않으므로, 이 항목은 "잃은 것"이 아니라 아래 `다음 단계 제안`의 빌드 항목으로 다룹니다.

## 최고 성능 구성

- **model**: `hist_gbdt` (iteration 2)
- **hyperparams** (적용된 값):
  ```
  max_iter=250
  learning_rate=0.06
  max_depth=4
  max_leaf_nodes=15
  l2_regularization=5.0
  min_samples_leaf=40
  early_stopping=false
  class_weight={"0": 1.0, "1": 9.0}
  ```
- **preprocessing** (적용된 값, 해당 attempt에서 그대로 인용): `impute: none` (트리가 NaN을 직접 분기), `scale: false`, `missing_indicator: false`, `missing_count: false`
- **split/protocol**: stratified 60/20/20, seed 43 (카드의 baseline과 동일한 분할이라 직접 비교 가능)
- **train_time_sec**: 5.076

| 지표 | 검증(20%) | held-back 테스트(20%) |
|---|---|---|
| `balanced_accuracy` | **0.7710** (95% CI 0.7471~0.8001) | **0.7969** (95% CI 0.7708~0.8206) |

기타 검증 지표: `roc_auc=0.8581`, `pr_auc`/`average_precision=0.6042`, `f1=0.5198`, `accuracy=0.7607`, `precision=0.3882`, `recall=0.7862`, `specificity=0.7557`, `balanced_accuracy_at_best_cut=0.7868`, `cut_headroom=0.0158`, `brier=0.1535`, `calibration_error=0.1783`, `train_balanced_accuracy=0.9119`, `train_val_gap=0.1409`.

검증과 테스트의 차이는 0.0259입니다. 이 실행의 모든 선택(모델·하이퍼파라미터·최종 후보)은 **검증 숫자만** 보고 이뤄졌으므로 이 차이가 선택 편향의 크기입니다. 다만 검증 점수 0.7710이 테스트 CI(0.7708~0.8206) 안에 들어가므로, 이 행 수로는 그 편향을 0과 구분할 수 없습니다. 대외적으로 인용할 숫자는 테스트 0.7969입니다.

## 원인 분석

critic 진단은 1회(iteration 1에 대해) 존재하며, 그 진단이 지목한 축이 그대로 결과로 나타났습니다.

- **iteration 1의 미달은 랭킹 부족이 아니라 절단점 문제였습니다.** iteration 1은 `roc_auc=0.8566`, `balanced_accuracy_at_best_cut=0.7827`로 이미 목표 note가 넘어야 한다고 지정한 baseline ranking ceiling 0.7558을 상회했습니다. 그런데 기본 0.5 규칙이 너무 보수적으로 잘려 `recall=0.5616` / `specificity=0.9214`, 남은 `cut_headroom=0.0412`가 목표까지의 격차 0.0183의 두 배 이상이었습니다. 즉 부족분 전체가 operating point 축에 있었고, critic이 이를 정확히 그렇게 읽고 `class_weight`를 4→9로 올린 것이 이번 실행의 유일한 실질적 개입입니다.
- **두 시도의 `balanced_accuracy` 차이(0.0295)는 이 데이터로는 통계적으로 분리되지 않습니다.** iteration 1의 CI(0.7114~0.7768)와 iteration 2의 CI(0.7471~0.8001)가 겹치므로, 점수 차이 자체를 "개선"으로 단정할 근거는 이 검증 슬라이스에 없습니다. 대신 분리 가능한 것은 **분해 결과**입니다: 랭킹 축은 사실상 정지했고(`roc_auc` 0.8566→0.8581, `balanced_accuracy_at_best_cut` 0.7827→0.7868 — 실행 노트가 제시한 roc_auc 분해 해상도 0.003~0.006 수준의 미미한 이동), `cut_headroom`은 0.0412→0.0158로, `recall`/`specificity`는 0.5616/0.9214 → 0.7862/0.7557로 크게 이동했습니다. 즉 얻은 것은 랭킹 개선이 아니라 **이미 존재했던 랭킹을 제대로 잘라 쓴 결과**입니다.
- **용량 축소는 과적합을 절반 이상 줄였습니다.** `train_val_gap` 0.2579→0.1409, `train_balanced_accuracy` 0.9994→0.9119. 5,026행 학습에서 400×31-leaf/depth-6이 사실상 암기하던 상태를 벗어났습니다.
- **남은 두 개의 약점.** (1) `cut_headroom=0.0158`이 아직 남아 있고 이제는 `recall`(0.7862)이 `specificity`(0.7557)보다 커서 가중치가 소폭 **오버슈트**했습니다 — critic이 예고한 "recall이 specificity를 넘으면 6~7로 되돌린다"는 조건이 실제로 성립했습니다. (2) 무거운 클래스 가중치의 대가로 확률 품질이 크게 나빠졌습니다: `calibration_error` 0.0577→0.1783, `brier` 0.1007→0.1535. 이 모델의 출력은 "확률"로 읽어서는 안 되고 결정 규칙으로만 써야 합니다(이 executor에는 재보정 레버가 없습니다).
- **성능의 상한을 실제로 제한한 것**은 결국 랭킹 축입니다. 두 시도 모두 `balanced_accuracy_at_best_cut`이 0.783~0.787에 머물렀고, 이는 하이퍼파라미터·클래스 가중치로 더 올릴 수 있는 대상이 아닙니다(모델 계열 또는 피처의 문제). 다만 이번 목표선은 그 상한 아래에서 충족되었으므로, 이 제한은 "실패 원인"이 아니라 "다음 예산이 향할 곳"입니다.

## 다음 단계 제안

1. **`class_weight`를 6~7로 되돌려 operating point를 마무리한다.** 근거: iteration 2에서 `recall=0.7862 > specificity=0.7557`로 가중치가 살짝 넘어갔고, 남은 `cut_headroom=0.0158`(최적 절단점 기준 0.7868)이 회수 가능한 전부입니다. 나머지 preprocessing/하이퍼파라미터는 iteration 2와 바이트 단위로 동일하게 두어 변화 귀속을 유지하십시오. 단, 이 한 번의 조정으로 기대할 수 있는 이동폭(≤0.016)은 검증 CI 폭(약 0.053)보다 작으므로, 개선으로 보고하지 말고 `recall`/`specificity` 균형과 `cut_headroom`으로만 판정해야 합니다.
2. **랭킹 축을 건드리는 시도로 1~2회를 쓴다: `xgboost`(동일하게 `impute: none`)로의 계열 교체.** 근거: 두 시도의 `roc_auc`가 0.8566/0.8581로 정지했고 `balanced_accuracy_at_best_cut`도 0.783~0.787에서 움직이지 않았습니다. 실행 노트의 측정에서 계열 교체가 roc_auc를 움직인 폭은 0.0022~0.0077로, 계열에 따라 재튜닝과 구분되지 않을 수도 있음을 전제로 하십시오. 참고로 `missing_indicator`는 `impute: none`과 함께 쓰면 예측이 비트 단위로 동일해지는 것이 확인된 항목이므로 이 조합에서는 절대 반복 시도하지 마십시오.
3. **확률이 필요한 사용처가 있다면 보정 레버를 executor 기능으로 추가한다.** 근거: 최고 구성의 `calibration_error=0.1783`, `brier=0.1535`. 현재 파이프라인은 재보정을 지원하지 않으므로, 지금 산출물은 랭킹(`roc_auc=0.8581`) 및 고정 절단점 의사결정에만 유효하다고 문서화하고, 확률 사용이 요건이라면 보정 단계를 구축하는 것이 선행 과제입니다.
4. **분할 구조와 결측 발생 원인을 먼저 확인한 뒤에야 파생 피처 작업을 계획한다.** 이 데이터에 대해 별도로 기록된 caveat은 없지만, 그것은 "문제가 없다"는 뜻이 아니라 "확인되지 않았다"는 뜻입니다. 확인할 두 가지: (a) 프로토콜은 `grouped_by: null`(층화 무작위 분할)인데, 이 파일은 스피드데이팅 만남 단위 행이므로 동일 참가자가 여러 행에 걸쳐 검증/테스트 양쪽에 나타날 수 있는지 — 사실이면 검증·테스트 숫자 모두 낙관 편향되며, 분할은 executor 레버가 아니라 실행 설정(run-level config) 변경 사안입니다. (b) `expected_num_interested_in_me`의 결측률 0.7852(및 `wave` 별 기록 방식 차이)가 응답자 상태가 아니라 **기록 시점/방식**을 가리키는지 — 그렇다면 `missing_count`나 결측 관련 열은 무작위 분할에서 이득처럼 채점될 위험이 있으므로(노트의 측정에서 `missing_count` 자체의 이득은 0.0003/0.0000 수준) 우선순위를 낮추어야 합니다. 이 두 확인이 끝나기 전에는 `feature_engineering`(예: 자기 평가와 상대 평가의 차이·비율 열)을 executor 기능으로 구축하는 작업도 착수 근거가 부족합니다 — 확인이 끝나면 그 기능이 랭킹 축을 실제로 움직이는지 검증 가능해집니다.