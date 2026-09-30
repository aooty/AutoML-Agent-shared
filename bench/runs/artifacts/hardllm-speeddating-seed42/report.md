# AutoML 실행 보고서 — `speeddating` (target: `match`)

## 요약

목표를 달성하지 못했습니다. 목표는 `balanced_accuracy` ≥ 0.8342였고, 5회 시도 중 최고 성능은 iteration 4의 `xgboost`로 검증 `balanced_accuracy` = 0.7783 (95% CI 0.7503~0.8055)이었으며, 최종 홀드백 테스트에서는 0.7830 (95% CI 0.7554~0.8111)을 기록해 임계값에서 약 0.05 부족했습니다. 다만 baseline(`logreg`, balanced_accuracy 0.6685, CI 0.6425~0.695)은 명확히 상회했습니다. 5회 반복 예산을 모두 소진했고(`max_iterations`), critic 진단은 4회 이루어졌습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=false`; `impute=none`, `missing_count=true` | balanced_accuracy **0.6597** (CI 0.6294~0.6881), roc_auc 0.8728, recall 0.3623 / specificity 0.9571, cut_headroom 0.1486, train_val_gap 0.3403 | `data_issue` — 랭킹은 이미 강함(best_cut 0.8083), 문제는 결정 규칙. 양성 클래스 가중 부족 → `xgboost` + `scale_pos_weight≈5.1` 권고 |
| 2 | `xgboost` | `n_estimators=350`, `learning_rate=0.06`, `max_depth=5`, `reg_lambda=2.0`, `subsample=0.8`, `scale_pos_weight=5.1`; `impute=none`, `missing_count=true` | balanced_accuracy **0.7332** (CI 0.7009~0.7617), roc_auc 0.8724, recall 0.5543 / specificity 0.9121, cut_headroom 0.0649, train_val_gap 0.2612 | `wrong_model_family` — 가중 레버는 거의 소진(cut_headroom 0.0649는 잔여 0.1010의 64%), 잔여는 랭킹에 있음 → `random_forest` + `class_weight='balanced_subsample'` + `median` impute + `missing_indicator` 권고 |
| 3 | `random_forest` | `n_estimators=800`, `min_samples_leaf=2`, `class_weight='balanced_subsample'`, `n_jobs=-1`, `random_state=42`; `impute=median`, `missing_indicator=true`, `missing_count=true` | balanced_accuracy **0.6002** (CI 0.5792~0.6265), roc_auc 0.8710, recall 0.2210 / specificity 0.9793, cut_headroom 0.2036, train_val_gap 0.3995 | `data_issue` — 자기 처방이 회귀(Δ -0.1331, CI -0.1607~-0.1046)했고 family와 pipeline을 동시에 움직여 원인 귀속 불가. 회수 가능 여백이 다시 operating point로 이동 → iteration 2 구성으로 복귀 + `scale_pos_weight=9.0` 권고 |
| 4 | `xgboost` | `n_estimators=500`, `learning_rate=0.05`, `max_depth=4`, `reg_lambda=4.0`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `scale_pos_weight=9.0`, `random_state=42`; `impute=none`, `missing_count=true` | balanced_accuracy **0.7783** (CI 0.7503~0.8055) ← 최고, roc_auc 0.8760, recall 0.7102 / specificity 0.8464, cut_headroom 0.0252, train_val_gap 0.1729 | `wrong_model_family` — operating point 사실상 소진(cut_headroom 0.0252는 잔여 0.0559의 45%), 남은 55%는 랭킹 → 동일 pipeline에서 `hist_gbdt` 대용량 구성 + `class_weight={"0":1,"1":9}` 권고 |
| 5 | `hist_gbdt` | `max_iter=1200`, `learning_rate=0.02`, `max_leaf_nodes=127`, `min_samples_leaf=5`, `l2_regularization=3.0`, `max_bins=255`, `early_stopping=false`, `class_weight={"0":1.0,"1":9.0}`, `random_state=42`; `impute=none`, `missing_count=true` | balanced_accuracy **0.6946** (CI 0.6632~0.7240), roc_auc 0.8674, recall 0.4493 / specificity 0.9400, cut_headroom 0.1001, train_val_gap 0.3054, train_time 91.07s | (없음 — 평가 직후 루프 종료) |

## 최고 성능 구성

iteration 4, `xgboost`. 아래 값은 히스토리에 기록된 **적용된** 하이퍼파라미터/전처리입니다(계획서가 아니라 executor가 실제로 만든 estimator에서 읽은 값).

```
model: xgboost
hyperparams:
  n_estimators: 500
  learning_rate: 0.05
  max_depth: 4
  reg_lambda: 4.0
  subsample: 0.8
  colsample_bytree: 0.8
  min_child_weight: 5
  scale_pos_weight: 9.0
  random_state: 42
preprocessing (applied):
  impute: none          # NaN을 트리가 직접 분기
  scale: false
  missing_indicator: false
  missing_count: true
split protocol: stratified 60/20/20, seed 42
train_time_sec: 4.96
dropped_hyperparams: [] (없음)
```

검증 및 테스트 점수:

| 구분 | balanced_accuracy | 95% CI |
|---|---|---|
| 검증 20% (선택에 사용됨) | 0.7783 | 0.7503~0.8055 |
| 최종 홀드백 테스트 20% (루프 중 미사용, 1회 채점) | **0.7830** | 0.7554~0.8111 |

두 값의 차이는 -0.0047(테스트가 오히려 소폭 높음)이며, 이것이 선택 편향의 크기입니다. 루프는 검증 점수만 보고 최고 시도를 골랐고, 검증 값이 테스트 CI 안에 들어 있으므로 이 행 수로는 차이를 0과 구분할 수 없습니다. 어느 쪽 값으로 읽어도 목표 0.8342에는 미달입니다.

검증 슬라이스의 기타 지표(iteration 4): `roc_auc` 0.8760, `pr_auc`/`average_precision` 0.6134, `f1` 0.5706, `accuracy` 0.8240, `precision` 0.4769, `recall` 0.7101, `specificity` 0.8464, `brier` 0.1203, `calibration_error` 0.1122, `balanced_accuracy_at_best_cut` 0.8035, `cut_headroom` 0.0252, `train_balanced_accuracy` 0.9512, `train_val_gap` 0.1729.

## 원인 분석

critic 진단은 4회 있었고, 그 4개 verdict가 가리키는 방향은 실행 기록과 일관됩니다.

**1) 실제로 움직인 것은 operating point 축뿐이었습니다.** iteration 1 → 2 (Δ +0.0735, CI +0.0500~+0.0994) 와 iteration 2 → 4 (Δ +0.0450, CI +0.0239~+0.0673) 는 모두 구간이 0을 넘지 않는 해상된 개선이며, 그 내용은 recall 0.3623 → 0.5543 → 0.7101, specificity 0.9571 → 0.9121 → 0.8464 로 임계값이 균형점으로 이동한 것입니다. `cut_headroom`도 0.1486 → 0.0649 → 0.0252로 붕괴했습니다. 즉 `scale_pos_weight`라는 유일한 불균형 레버는 iteration 4에서 사실상 소진되었습니다.

**2) 랭킹 축은 5회 동안 거의 움직이지 않았습니다.** `roc_auc`는 0.8728 / 0.8724 / 0.8710 / 0.8760 / 0.8674, `balanced_accuracy_at_best_cut`는 0.8083 / 0.7981 / 0.8038 / 0.8035 / 0.7947입니다. 이 스프레드(roc_auc 약 0.0086, best_cut 약 0.0136)는 executor 문서가 밝힌 paired 해상도(roc_auc 0.003~0.006) 및 검증 CI 폭(±0.03 내외)에 견줘 작아, 세 family(`hist_gbdt`, `xgboost`, `random_forest`)의 랭킹 품질은 **이 데이터로 서로 구분되지 않습니다**. 결정적인 수치는 이것입니다: 완벽한 임계값을 골라도 도달 가능한 최댓값이 0.8035~0.8083 수준이므로, 목표 0.8342까지는 약 0.026이 남습니다. KS로 환산하면 필요치 0.6684 대 실측 최고 0.6166입니다. 즉 **남은 격차는 임계값이나 클래스 가중으로 살 수 있는 것이 아니라 랭킹(=feature 정보량 / 모델 표현력)에 있습니다.**

**3) family 교체는 여기서 값을 사지 못했습니다.** iteration 3(`random_forest`)은 0.6002로 iteration 2 대비 Δ -0.1331 (CI -0.1607~-0.1046)의 해상된 회귀였지만, 이 시도는 family와 pipeline(`impute` none→median, `missing_indicator` false→true)을 동시에 바꿨기 때문에 손실의 소유자가 둘이고 어느 레버 탓인지 귀속할 수 없습니다. critic 본인도 그 점을 명시했습니다. iteration 5는 pipeline을 고정한 채 family만 바꾼 유일한 깨끗한 비교인데, roc_auc 0.8674 / best_cut 0.7947로 랭킹이 개선되지 않았고 balanced_accuracy는 0.6946(CI 0.6632~0.7240)로 떨어졌습니다 — 동일한 가중치 9.0을 넣었음에도 recall 0.4493 / specificity 0.9400으로 임계값이 다시 보수적으로 착지했기 때문입니다. `wrong_model_family` verdict가 두 번 나왔으나 두 번 모두 랭킹을 올리지 못했다는 것이 이 실행의 가장 견고한 사실입니다.

**4) 과적합은 원인이 아니었습니다.** train_val_gap은 0.3403 → 0.2612 → 0.3995 → 0.1729 → 0.3054로 크게 움직였지만 roc_auc는 그와 무관하게 평평했습니다. iteration 4의 정규화 강화(max_depth 5→4, reg_lambda 2.0→4.0, min_child_weight 5, colsample_bytree 0.8)는 gap을 0.2612 → 0.1729로 줄였으나 랭킹은 0.8724 → 0.8760(해상도 이하)에 머물렀습니다. 용량 축은 이미 충분히 눌렸습니다.

**5) 확률 자체의 품질도 최고 구성에서 악화됐습니다.** iteration 4는 `brier` 0.1203, `calibration_error` 0.1122로, iteration 2(0.1021 / 0.0465)보다 나쁩니다. 이는 강한 `scale_pos_weight`가 확률을 양성 쪽으로 밀어 올린 부작용이며, 이 executor에는 재보정 레버가 없습니다. balanced_accuracy만 보면 보이지 않는 비용입니다.

**참고 — `unsupported_claims`:** iteration 5의 계획에 `feature_engineering`이 표시되었습니다. 다만 해당 iteration의 plan 내용(critic의 direction)은 pipeline을 "bit-for-bit" 유지하고 랭킹 생성기만 교체하는 것이며, feature 파생에 의존한 설계가 아닙니다. 따라서 이 표시 때문에 iteration 5가 무언가를 잃었다고 볼 근거는 없고, 파생 feature 부재는 아래 "다음 단계"에서 **만들어야 할 역량**으로 다룹니다.

## 다음 단계 제안

이 데이터셋에는 별도로 기록된 `Data caveats`가 없으므로(“없음”), 특정 열·값·split을 신뢰 불가로 배제하는 제약은 없습니다. 그래도 아래 제안은 executor의 CAN/CANNOT 경계 안에서만 서술합니다.

1. **랭킹 축을 올릴 수 있는 유일한 실질 수단은 feature 파생이며, 그것은 현재 executor 밖에 있습니다 — 먼저 카드 단계에서 열을 만들어 넣으십시오.** 근거: 완벽한 임계값에서도 best_cut이 0.8035~0.8083에 머물러 목표까지 0.026이 남고, 세 family가 이 상한을 공유했습니다. `speeddating`은 쌍(pair) 데이터로, `attractive_important` × `attractive_partner` 같은 선호-평가 정합성, `like` − `guess_prob_liked` 차이, `pref_o_*`와 `*_o` 평점의 매칭 점수 등 자연스러운 상호작용이 존재하는데(카드에서 `like`만이 유일한 `target_corr: strong`, `attractive_o`/`funny_o`/`shared_interests_partner` 등이 moderate), executor는 "비율·차이·상호작용·재인코딩·드롭"을 일절 하지 않습니다. 이런 파생 열을 **데이터 카드/입력 파일 쪽에서 미리 만들어** 넣는 것이 다음 예산의 1순위입니다.
2. **고차원 카디널리티로 버려진 `field` 열을 살릴 수 있는 형태로 사전 인코딩하십시오.** 카드의 `dropped_high_cardinality: ["field"]`(max_cardinality 50)로 인해 이 열은 현재 모델에 전혀 들어가지 않았고, executor는 재인코딩을 하지 않으므로 계획으로는 되돌릴 수 없습니다. 상위 빈도 그룹으로 축약한 저카디널리티 열을 입력 파일에 추가하면 새로운 정보가 랭킹 축에 들어옵니다.
3. **operating point는 더 밀지 마십시오 — 최적 지점을 이미 지나쳤을 가능성이 있습니다.** iteration 4에서 `cut_headroom`이 0.0252로, 임계값을 완벽히 골라도 남은 0.0559의 45%만 회수됩니다. 동시에 `calibration_error`가 0.0465(spw 5.1) → 0.1122(spw 9.0)로 두 배 이상 나빠졌습니다. 예산이 있다면 `scale_pos_weight`를 9.0 위로 올리는 대신 **5.1~9.0 사이(예: 6.5~7.5)에서 1회만** 확인해 recall/specificity 균형과 brier를 동시에 보고 멈추는 것이 합리적입니다. 이는 목표 달성 수단이 아니라 운영 지점 확정용입니다.
4. **family 교체에 추가 예산을 쓰지 마십시오.** `hist_gbdt`·`xgboost`·`random_forest` 세 family의 roc_auc 스프레드가 0.0086, best_cut 스프레드가 0.0136으로 이 슬라이스의 해상도 안에 있습니다. 네 번째 트리 family가 0.026을 메울 것으로 기대할 근거가 기록에 없습니다. 굳이 모델 축을 쓸 경우에는 iteration 3의 실패를 반복하지 않도록 **pipeline을 고정하고 한 번에 한 레버만** 바꾸어(iteration 5처럼) 귀속 가능성을 유지해야 합니다.
5. **(방법론) 단일 20% 검증 슬라이스의 CI가 ±0.03 수준이라는 점을 감안해, 향후 비교는 이 폭보다 큰 차이만 개선으로 취급하십시오.** 이번 실행에서 iteration 1(0.6597)·3(0.6002)·5(0.6946)은 서로 구간이 부분적으로만 겹치지만, 랭킹 지표들은 사실상 구분 불가였습니다. 교차검증은 이 executor에서 지원되지 않으므로(“no cross-validation”), 더 안정적인 비교가 필요하면 실행 설정 자체(split 프로토콜)를 바꿔야 하며, 이는 루프 밖의 작업입니다.