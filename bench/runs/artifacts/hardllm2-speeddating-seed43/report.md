# AutoML 최종 리포트 — `speeddating` / `match` 이진 분류

## 요약

**목표에 도달하지 못했습니다.** 목표는 `balanced_accuracy` ≥ 0.8399였고, 5회 예산 중 4회를 사용한 뒤 연속 미개선(stalled)으로 조기 종료했습니다. 최고 성능은 iteration 2의 `hist_gbdt`로 검증 `balanced_accuracy` = **0.7725** (95% CI 0.7460~0.7987), 동일 모델을 한 번도 사용되지 않은 최종 테스트 20%에서 채점하면 **0.7944** (95% CI 0.7677~0.8205)로, 목표선까지 여전히 약 0.045 부족합니다. 4회 시도 모두 baseline(logreg, `balanced_accuracy` 0.6798)은 넘겼지만, `balanced_accuracy_at_best_cut`이 0.7770~0.7848 범위에 갇혀 있어 **임계값 자체가 이 데이터·이 실행기에서 도달 가능한 랭킹 상한 위**에 있었습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `class_weight='balanced'`, `early_stopping=False`, `impute='none'` | `balanced_accuracy` 0.7491 (CI 0.7222~0.7785), `roc_auc` 0.8587, `balanced_accuracy_at_best_cut` 0.7836, `cut_headroom` 0.0345, `train_val_gap` 0.2509 (train BA 1.0) | `overfitting` — 전 학습행 암기, 변동성이 랭킹을 제한한다고 판단하여 용량 축소·정규화 강화 지시 |
| 2 | `hist_gbdt` | `max_iter=600`, `learning_rate=0.03`, `max_leaf_nodes=8`, `max_depth=4`, `min_samples_leaf=60`, `l2_regularization=10.0`, `class_weight='balanced'`, `early_stopping=False`, `impute='none'` | **`balanced_accuracy` 0.7725 (CI 0.7460~0.7987)**, `roc_auc` 0.8552, `pr_auc` 0.5952, `balanced_accuracy_at_best_cut` 0.7795, `cut_headroom` 0.0070, `train_val_gap` 0.1190, `calibration_error` 0.1555 | `wrong_model_family` — gap은 0.2509→0.1190으로 실제 감소했으나 랭킹은 움직이지 않음(`roc_auc` 0.8587→0.8552). 다른 랭킹 생성기(`xgboost`)로 전환 지시 |
| 3 | `xgboost` | `n_estimators=800`, `learning_rate=0.05`, `max_depth=6`, `subsample=0.8`, `reg_lambda=2.0`, `scale_pos_weight=5.07`, `impute='none'` | `balanced_accuracy` 0.7226 (CI 0.6954~0.7560), `roc_auc` 0.8605, `balanced_accuracy_at_best_cut` 0.7848, `cut_headroom` 0.0622, `train_val_gap` 0.2774 | `data_issue` — iteration 2 대비 페어드 Δ −0.0499 (CI −0.0763~−0.0215, P(better) 0.000)로 **해소된 회귀**. 두 family의 best-cut 폭이 0.0012에 불과해 family 교체는 소진. 운영점(양성 가중치 상향) 회수 지시 |
| 4 | `hist_gbdt` | iteration 2와 동일 용량 + `class_weight={"0":1.0,"1":7.0}`, `missing_count=True`, `impute='none'` | `balanced_accuracy` 0.7710 (CI 0.7468~0.7994), `roc_auc` 0.8574, `balanced_accuracy_at_best_cut` 0.7770, `cut_headroom` 0.0060, recall 0.7862 / specificity 0.7557, `calibration_error` 0.1843 | (critic 미실행 — 마지막 시도) |

## 최고 성능 구성

iteration 2, **`hist_gbdt`**. 아래는 실행기가 실제로 만든 값(`hyperparams` / `preprocessing`)입니다.

```
model: hist_gbdt
hyperparams:
  max_iter: 600
  learning_rate: 0.03
  max_leaf_nodes: 8
  max_depth: 4
  min_samples_leaf: 60
  l2_regularization: 10.0
  class_weight: "balanced"
  early_stopping: false
preprocessing:
  impute: none          # NaN을 모델이 직접 분기
  scale: false
  missing_indicator: false
  missing_count: false
dropped_hyperparams: []   # 요청한 설정이 모두 적용됨
protocol: stratified 60/20/20, seed 43
train_time_sec: 8.042
```

| 지표 | 검증(20%) | 최종 테스트(20%, 미사용 행) |
|---|---|---|
| `balanced_accuracy` | **0.7725** (95% CI 0.7460~0.7987) | **0.7944** (95% CI 0.7677~0.8205) |
| `roc_auc` | 0.8552 | — |
| `pr_auc` / `average_precision` | 0.5952 | — |
| `recall` / `specificity` | 0.7536 / 0.7914 | — |
| `precision` / `f1` / `accuracy` | 0.4160 / 0.5361 / 0.7852 | — |
| `balanced_accuracy_at_best_cut` / `cut_headroom` | 0.7795 / 0.0070 | — |
| `brier` / `calibration_error` | 0.1391 / 0.1555 | — |
| `train_val_gap` | 0.1190 (train BA 0.8915) | — |

루프의 모든 선택은 검증 숫자(0.7725)로 이루어졌고, 테스트 0.7944는 그 선택 이후 단 한 번 채점된 값입니다. 두 값의 차이 −0.0219가 **선택 편향의 크기**이지만, 검증 점수가 테스트 CI(0.7677~0.8205) 안에 들어오므로 이 행들만으로는 0과 구분되지 않습니다. 어느 쪽을 보더라도 목표 0.8399에는 미달이며, 테스트 CI의 상단(0.8205)조차 목표선 아래입니다.

## 원인 분석

critic 판정은 3회 존재하고, 그 3회가 서로 다른 축을 하나씩 소진시키는 뚜렷한 패턴을 만들었습니다.

- **변동성(용량)은 원인이 아니었다.** iteration 1→2에서 `train_val_gap`이 0.2509 → 0.1190, train BA가 1.0 → 0.8915로 암기가 실제로 제거되었지만, 랭킹은 따라오지 않았습니다(`roc_auc` 0.8587 → 0.8552, `balanced_accuracy_at_best_cut` 0.7836 → 0.7795 — 둘 다 소폭 **하락**). 두 시도의 `balanced_accuracy` CI(0.7222~0.7785 / 0.7460~0.7987)는 크게 겹치고 페어드 Δ도 +0.0234 (CI −0.0020~+0.0503)이므로, **iteration 1과 2는 이 데이터로 구별되지 않습니다.** 즉 "정규화로 0.023 벌었다"는 이야기는 쓸 수 없고, 확실하게 말할 수 있는 것은 "gap은 줄었으나 랭킹은 그대로"라는 사실뿐입니다.
- **family 교체는 해소된 손실을 냈다.** iteration 3의 `xgboost`는 iteration 2 대비 페어드 Δ −0.0499 (CI −0.0763~−0.0215, P(better) 0.000)로, 구간 폭보다 큰 실제 회귀입니다. 그런데 랭킹 축에서는 여기서도 거의 움직임이 없었습니다: 두 family의 `balanced_accuracy_at_best_cut`은 0.7836과 0.7848(폭 0.0012), 4회 전체로도 0.7770~0.7848(폭 0.0078)에 머물렀습니다.
- **운영점은 완전히 소진되었다.** iteration 2에서 `cut_headroom`은 이미 0.0070(잔여 격차 0.0674의 10%)이고 recall 0.7536 ≈ specificity 0.7914였습니다. iteration 4에서 가중치를 `{"0":1,"1":7}`로 더 밀자 recall 0.7862 / specificity 0.7557로 **팔이 반대로 기울기만** 했고 `balanced_accuracy`는 0.7710(CI 0.7468~0.7994)로 iteration 2와 구별되지 않았습니다. 참고로 iteration 4는 `class_weight`와 `missing_count`를 동시에 바꿨으므로, 설령 움직임이 있었더라도 두 레버 중 어느 쪽 몫인지 분리할 수 없는 시도였습니다.

정리하면 **남은 격차는 거의 전부 랭킹 축에 있고, 그 랭킹은 하이퍼파라미터·family·클래스 가중치 어느 것으로도 움직이지 않았습니다.** 이 실행기가 만들 수 있는 최고의 랭킹을 최적 컷으로 잘라도 0.7848이며 목표 0.8399까지 0.055가 남습니다(목표가 요구하는 KS 0.6798에 대해 실제 KS는 0.5696 수준). 이는 goal 정의의 `exceeds_ranking_ceiling: true`, `ks_shortfall: 0.1682`와 일치합니다 — 목표선은 baseline 계열 랭킹의 도달 범위 밖에서 설정되었습니다.

한 가지 주의: iteration 1의 plan에 `unsupported_claims: ["feature_engineering"]`가 기록되어 있지만, 이 기록에는 해당 plan 원문이 남아 있지 않아 그 계획이 실제로 파생 피처에 의존했는지 단순 언급이었는지 확인할 수 없습니다. 따라서 iteration 1의 점수를 "피처 엔지니어링을 못 해서 손실이 났다"고 해석하지 않습니다. 파생 피처 부재는 이 실행기의 고정된 제약이며, 아래 다음 단계에 속하는 항목입니다.

## 다음 단계 제안

1. **파생 피처를 CSV 단계에서 만들어 카드에 태워 넣기 (최우선).** 랭킹 상한이 4회 시도에서 0.7770~0.7848로 사실상 고정된 반면 잔여 격차는 0.055이므로, 남은 진짜 여지는 "정보량"뿐입니다. 실행기는 열 결합·비율·차분·상호작용을 만들 수 없으므로(제약 목록의 명시 항목), 예컨대 `like`(target_corr strong), `guess_prob_liked`, `attractive_partner`/`funny_partner`/`shared_interests_partner`(moderate)와 대응하는 `*_o`·`*_important` 열들 사이의 차분·곱 항을 **원본 파일에서 미리 생성**한 뒤 새 카드로 다시 프로파일해야 합니다. 이것이 랭킹 축을 실제로 움직일 수 있는 유일한 레버입니다.
2. **목표선 0.8399의 타당성 재검토.** 이 값은 baseline 0.6798에 margin 0.5를 적용해 파생된 값이고, goal 블록 자체가 `exceeds_ranking_ceiling: true`, `required_ks: 0.6798` vs 실제 KS ≈ 0.5696을 기록합니다. 현재 피처 집합으로는 최적 컷을 골라도 0.785 부근이 상한이므로, 목표를 "랭킹 지표(`roc_auc` 0.8552 → 개선)" 또는 도달 가능한 `balanced_accuracy` 구간으로 다시 정의하지 않으면 예산을 더 써도 같은 벽에 부딪힙니다.
3. **행 독립성(그룹 구조) 확인을 운영자에게 요청.** 현재 프로토콜은 `grouped_by: null`의 행 단위 층화 분할입니다. speeddating 형식의 파일은 동일 참가자가 여러 행에 반복될 수 있고, 그렇다면 검증·테스트 점수가 낙관적으로 편향됩니다. 데이터 카드에 참가자 식별 열이 없어 여기서는 확인 자체가 불가능하므로(카드에 별도 caveat도 없음), **먼저 원본에 참가자 ID가 있는지와 반복 여부를 확인**하는 것이 다음 예산의 전제 조건입니다. 이것이 해소되면 위 1번의 개선폭을 신뢰할 수 있게 되고, 필요 시 그룹 분할 프로토콜로 갈 근거가 됩니다.
4. **확률값이 필요한 용도라면 iteration 1 구성을 별도로 보관.** 최고 구성(iteration 2)은 강한 축소 때문에 `calibration_error` 0.1555 / `brier` 0.1391로, iteration 1(0.0646 / 0.1027)보다 확률 품질이 크게 나쁩니다. 이 실행기에는 재보정 레버가 없으므로(`CalibratedClassifierCV`·컷 이동 불가), 순위만 쓸 것인지 확률을 쓸 것인지에 따라 채택 구성을 나누는 것이 현실적입니다. 남은 1회 예산을 쓴다면 `random_forest` 등 미시도 family 1회가 유일한 저비용 탐색이지만, family 간 best-cut 폭이 0.0012~0.0078에 불과했다는 증거상 기대값은 낮으므로 1번·3번보다 뒤에 두어야 합니다.