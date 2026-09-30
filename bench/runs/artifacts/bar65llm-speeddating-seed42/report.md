# AutoML 실행 보고서 — `speeddating` / `balanced_accuracy`

## 요약

**목표는 달성되지 못했습니다.** 목표 기준선은 `balanced_accuracy` ≥ 0.884였지만, 5회 예산 중 4회를 사용한 뒤 연속 미개선(정체)으로 조기 종료되었고, 최고 검증 점수는 iteration 2의 `hist_gbdt`가 기록한 **0.7865 (95% CI 0.7602~0.8114)** 로 기준까지 0.0975 부족했습니다. 한 번도 사용되지 않은 최종 테스트 20%에서 같은 모델은 **0.7977 (95% CI 0.7700~0.8234)** 를 기록했습니다. 4회 시도 모두 랭킹 축(`roc_auc` 0.8691~0.8741, `balanced_accuracy_at_best_cut` 0.7984~0.8053)이 사실상 움직이지 않았고, 이는 목표치가 이 열 구성으로 도달 가능한 범위 밖에 있었다는 것을 시사합니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `class_weight='balanced'`, `early_stopping=False` / `impute: none`, `scale: false` | `balanced_accuracy` **0.7049** (CI 0.6765~0.7358), `roc_auc` 0.8691, `balanced_accuracy_at_best_cut` 0.7984, `cut_headroom` 0.0935, `train_val_gap` **0.2949** | `overfitting` — train 0.9997 vs val 0.7049. 용량을 강하게 줄여 암기를 막고 best_cut 상승 여부를 확인하라 |
| 2 | `hist_gbdt` | `max_iter=300`, `learning_rate=0.05`, `max_leaf_nodes=8`, `min_samples_leaf=60`, `l2_regularization=10.0`, `max_features=0.6`, `max_depth=4`, `class_weight='balanced'`, `early_stopping=False` / `impute: none`, `scale: false` | `balanced_accuracy` **0.7865** (CI 0.7602~0.8114) ← 최고, `roc_auc` 0.8741, `best_cut` 0.8053, `cut_headroom` 0.0188, `train_val_gap` 0.0820 | `wrong_model_family` — 개선분(paired Δ +0.0816, CI +0.0564~+0.1111)의 대부분은 작동점(cut)이고 랭킹은 +0.0069뿐. 계열을 처음으로 바꿔보라 |
| 3 | `xgboost` | `n_estimators=700`, `learning_rate=0.04`, `max_depth=5`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.7`, `reg_lambda=3.0`, `reg_alpha=0.5`, `scale_pos_weight=5.07`, `early_stopping_rounds=50` / `impute: none` | `balanced_accuracy` **0.7256** (CI 0.6917~0.7564), `roc_auc` 0.8738, `best_cut` 0.7986, `cut_headroom` 0.0730, `train_val_gap` 0.2436 | `wrong_model_family` — 계열 교체가 랭킹으로 사 온 것은 0.0000~0.0067. iteration 2 구성으로 복귀해 `missing_count`만 추가하라 |
| 4 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.05`, `max_leaf_nodes=12`, `min_samples_leaf=40`, `l2_regularization=8.0`, `max_features=0.8`, `max_depth=5`, `class_weight='balanced'`, `early_stopping=False` / `impute: none`, `missing_count: true` | `balanced_accuracy` **0.7616** (CI 0.7324~0.7878), `roc_auc` 0.8741, `best_cut` 0.8037, `cut_headroom` 0.0421, `train_val_gap` 0.1797 | (없음 — 평가 직후 루프 종료) |

## 최고 성능 구성

**iteration 2 / `hist_gbdt`** — 아래 값은 실제로 실행된(집행기가 만든 추정기에서 읽은) `hyperparams`와 `applied preprocessing`입니다. `dropped_hyperparams`는 비어 있어, 계획한 값이 그대로 적용되었습니다.

```
model: hist_gbdt
hyperparams:
  max_iter: 300
  learning_rate: 0.05
  max_leaf_nodes: 8
  min_samples_leaf: 60
  l2_regularization: 10.0
  max_features: 0.6
  max_depth: 4
  class_weight: "balanced"
  early_stopping: false
preprocessing:
  impute: none          # NaN을 트리가 직접 분기
  scale: false
  missing_indicator: false
  missing_count: false
split: stratified 60/20/20, seed 42
train_time_sec: 5.548
```

| 지표 | 값 |
|---|---|
| `balanced_accuracy` (검증) | **0.7865** (95% CI 0.7602~0.8114) |
| `balanced_accuracy` (최종 테스트, 1회만 채점) | **0.7977** (95% CI 0.7700~0.8234) |
| `roc_auc` / `pr_auc` | 0.8741 / 0.6043 |
| `recall` / `specificity` | 0.7536 / 0.8193 |
| `precision` / `f1` / `accuracy` | 0.4512 / 0.5645 / 0.8085 |
| `balanced_accuracy_at_best_cut` / `cut_headroom` | 0.8053 / 0.0188 |
| `brier` / `calibration_error` | 0.1297 / 0.1530 |
| `train_balanced_accuracy` / `train_val_gap` | 0.8684 / 0.0820 |

검증 0.7865와 테스트 0.7977의 차이 0.0113이 **선택 편향의 크기**입니다 — 루프는 검증 숫자만 보고 최고 시도를 골랐기 때문입니다. 다만 검증 점수가 테스트 CI(0.7700~0.8234) 안에 들어 있으므로, 이 테스트 행들만으로는 그 차이를 0과 구분할 수 없습니다. 어느 쪽 숫자를 쓰더라도 기준 0.884에는 미달입니다.

> 참고: iteration 2 계획에는 `unsupported_claims: ["feature_engineering"]` 플래그가 붙었지만, 해당 계획 본문은 전처리를 "impute:none / scale:false / no missingness columns 그대로 유지"한다고 명시했을 뿐 파생 피처에 의존하지 않았습니다. 즉 이 플래그 때문에 시도가 무엇을 잃은 것은 아니며, 피처 생성 부재는 아래 `다음 단계 제안`의 항목입니다.

## 원인 분석

critic 판정은 3회 존재하며(`overfitting` 1회, `wrong_model_family` 2회), 그 궤적 자체가 병목을 가리킵니다.

1. **1회차의 진단은 옳았고, 처방으로 살 수 있는 것은 그때 전부 소진되었습니다.** iteration 1 → 2는 `train_val_gap` 0.2949 → 0.0820, paired Δ +0.0816 (CI +0.0564~+0.1111, P(better)=1.000)으로 구간보다 큰 실제 개선입니다. 그러나 같은 구간에서 `balanced_accuracy_at_best_cut`은 0.7984 → 0.8053(+0.0069), `roc_auc`는 0.8691 → 0.8741(+0.0050)로, 문서가 명시한 paired 해상도(0.003~0.006) 안입니다. 즉 개선분의 실질은 **작동점 이동**(`cut_headroom` 0.0935 → 0.0188, `recall` 0.4855 → 0.7536)이고, **랭킹은 움직이지 않았습니다.**

2. **계열 교체(iteration 3)는 랭킹 축에 아무것도 보태지 못했습니다.** `xgboost`의 `roc_auc` 0.8738과 `best_cut` 0.7986은 iteration 2(0.8741 / 0.8053)와 해상도 안에서 구분되지 않습니다. `balanced_accuracy`가 -0.0608(CI -0.0883~-0.0367) 떨어진 것은 랭킹 퇴보가 아니라 `scale_pos_weight=5.07`이 iteration 2의 균형점을 재현하지 못해 `cut_headroom`이 0.0188 → 0.0730으로 되돌아간 결과입니다(`recall` 0.5399 vs `specificity` 0.9114).

3. **iteration 2와 iteration 4는 이 데이터로 구분되지 않습니다.** CI가 0.7602~0.8114와 0.7324~0.7878로 겹치고, `roc_auc`는 0.8741로 동일합니다. 따라서 `missing_count` 추가가 도움이 되었는지 해가 되었는지는 **측정되지 않았다**고 말하는 것이 정확하며, 두 점 사이의 -0.0249는 설명할 대상이 아니라 슬라이스의 흔들림입니다.

4. **남은 거리는 전부 랭킹 축에 있고, 그 축이 4회 동안 0.0069만 움직였습니다.** 네 시도의 `balanced_accuracy_at_best_cut`은 0.7984 / 0.8053 / 0.7986 / 0.8037 — 두 계열, 네 가지 용량 설정을 걸쳐 폭 0.0069입니다. 최고 랭킹을 **완벽하게 자르더라도** 0.8053이므로 기준까지 0.0787이 남습니다. 목표 정의 자체도 이를 예고합니다: `exceeds_ranking_ceiling: true`, 필요 KS 0.768 대 baseline KS 0.5423, baseline `ranking_ceiling` 0.7712. 최고 시도의 `cut_headroom` 0.0188은 남은 0.0975의 약 19%에 불과하므로, `class_weight`/`scale_pos_weight` 같은 작동점 레버로는 원리적으로 도달할 수 없었습니다.

요약하면 실패 원인은 하이퍼파라미터 탐색 부족이나 불균형 처리 미숙이 아니라, **이 119개 열을 그대로 넣은 상태에서 어떤 추정기도 만들어내지 못한 랭킹 상한**입니다. 부수적으로, 최고 구성은 확률 품질이 나쁩니다(`calibration_error` 0.1530, `brier` 0.1297 — iteration 1의 0.0619 / 0.1063, iteration 3의 0.0493 / 0.1045보다 확연히 나쁨). `class_weight='balanced'`가 작동점을 옮긴 대가이며, 이 집행기에는 재보정 레버가 없습니다.

## 다음 단계 제안

기록된 데이터 주의사항은 없으므로(`없음`), 아래는 열 신뢰성 제약 없이 제안하는 항목입니다. 순서는 증거가 가리키는 크기순입니다.

1. **피처 축(카드 수준)에 예산을 쓸 것 — 이 집행기가 할 수 없는 유일한 큰 레버입니다.** 랭킹 축이 두 계열·네 설정에서 0.0069밖에 벌어지지 않았고 남은 거리는 0.0787이므로, 추정기를 더 바꾸는 계획은 0.884에 도달하는 계획이 아닙니다. 집행기는 파생·상호작용·재인코딩·열 제거를 일절 하지 않으므로, 변경은 **입력 파일/카드**에서 이뤄져야 합니다: (a) 쌍 구조를 명시하는 차이·정합 피처(예: 본인 평가 대 상대 평가의 차, `like`·`guess_prob_liked`와 상대측 평가의 상호작용), (b) 카드가 `dropped_high_cardinality`로 버린 `field`를 목표 인코딩 없이도 쓸 수 있는 소수 그룹으로 축약, (c) 원-핫으로 180 레벨을 차지하는 58개 `d_*` 이산화 열이 대응 수치열과 중복인지 확인 후 정리. 이것이 풀리면 `roc_auc`/`balanced_accuracy_at_best_cut`이 처음으로 움직일 수 있고, 그때서야 작동점 레버가 의미를 갖습니다.

2. **분할 단위를 점검할 것 — 현재 프로토콜은 `grouped_by: null`입니다.** 이 데이터는 `wave` 안에서 같은 참가자가 여러 행으로 반복될 수 있는 구조이므로, 행 단위 층화 분할에서는 동일 인물이 train과 validation/test에 동시에 등장할 수 있습니다. 검증 0.7865와 테스트 0.7977이 서로 일관된 것은 두 슬라이스가 **같은 방식**으로 나뉘었기 때문이며, 이 위험을 배제하지는 못합니다. 참가자 ID 기준 group 분할로 한 번 재측정해, 현재의 0.79대가 얼마나 그룹 누수에 의존하는지 확정하는 것을 권합니다(집행기는 분할을 바꿀 수 없으므로 런 설정 차원의 작업입니다).

3. **목표 임계값 0.884의 타당성을 재검토할 것.** 목표 메타데이터 자체가 `exceeds_ranking_ceiling: true`, `required_ks: 0.768`, `ks_shortfall: 0.2257`, `passable_margin: 0.309`를 기록하고 있고, 실제로 관측된 최고 랭킹은 KS ~0.6106 / best_cut 0.8053입니다. 피처가 늘지 않는 한 이 기준은 도달 불가로 보이므로, 현재 열 구성에 대해서는 baseline 0.6685 대비 개선폭(테스트 0.7977)을 성과 지표로 재정의하거나, 1번 항목의 피처 확장을 전제로 임계값을 유지할지 결정해야 합니다.

4. **작동점 레버는 남은 소액만 회수하는 용도로만 쓸 것.** 최고 구성의 `cut_headroom`은 0.0188(남은 거리의 19%)이므로 추가 시도 1회를 여기 쓴다면, `class_weight='balanced'`(빈도비 5.07에 고정된 한 점)를 명시적 가중치 맵(예: `{"0": 1, "1": 3}`, `{"0": 1, "1": 7}`) 몇 점으로 바꿔 `recall` 0.7536 / `specificity` 0.8193 근방을 미세 조정하는 정도가 상한입니다. 동시에 `calibration_error` 0.1530은 이 집행기에서 교정할 수 없으므로, 확률값을 그대로 소비하는 downstream이 있다면 재보정 단계를 파이프라인 **밖에** 구축해야 한다는 점을 명시해 두십시오.