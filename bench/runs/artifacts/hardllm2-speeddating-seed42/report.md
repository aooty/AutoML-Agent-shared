# AutoML 최종 리포트 — `speeddating` / `balanced_accuracy`

## 요약

**목표는 달성하지 못했습니다.** 목표 기준선은 `balanced_accuracy` 0.8342였으나, 5회 예산 중 4회를 사용한 뒤 연속 미개선(정체)으로 조기 종료되었고 최고 검증 점수는 iteration 2의 `hist_gbdt`가 기록한 **0.7990 (95% CI 0.7735~0.8229)** 로 기준선에 0.0352 미달했습니다. 한 번도 사용되지 않은 홀드백 테스트 20%에서 같은 모델은 **0.7883 (95% CI 0.7628~0.8103)** 를 기록했습니다. 4회 시도 모두에서 `balanced_accuracy_at_best_cut`(랭킹 상한)이 0.7984~0.8062 범위에 머물러, 임계값을 어떻게 잡아도 이 컬럼 집합과 두 개 모델 패밀리로는 목표선에 도달할 수 없다는 것이 이 런의 핵심 관측입니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=false`, `class_weight='balanced'` | `balanced_accuracy=0.7049` (CI 0.6765~0.7358), `roc_auc=0.8691`, `best_cut=0.7984`, `cut_headroom=0.0935`, `train_val_gap=0.2949` | `overfitting` — train `balanced_accuracy=0.9998` 대비 검증 0.7049. 용량 축소 + 양성 가중 상향 지시 |
| 2 | `hist_gbdt` | `max_iter=300`, `learning_rate=0.05`, `max_depth=3`, `max_leaf_nodes=8`, `min_samples_leaf=60`, `l2_regularization=10.0`, `early_stopping=false`, `class_weight={0:1.0, 1:9.0}` | **최고** `balanced_accuracy=0.7990` (CI 0.7735~0.8229), `roc_auc=0.8736`, `best_cut=0.8050`, `cut_headroom=0.0060`, `train_val_gap=0.0547` | `wrong_model_family` — 과적합은 해소(gap 0.2949→0.0547)되었으나 상승분의 대부분이 운영점 축에서 나왔고 랭킹 상한은 0.8050. `xgboost`로 패밀리 교체 지시 |
| 3 | `xgboost` | `n_estimators=600`, `learning_rate=0.04`, `max_depth=4`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.6`, `reg_lambda=5.0`, `reg_alpha=0.5`, `scale_pos_weight=7.0` | `balanced_accuracy=0.7718` (CI 0.7423~0.8007), `roc_auc=0.8759`, `best_cut=0.8037`, `cut_headroom=0.0319`, `train_val_gap=0.1810` | `data_issue` — 두 패밀리의 랭킹 상한 차이가 0.0013에 불과(필요 상승폭 0.0292). iteration 2 구성 복귀 + 가중 ~8 재조정 지시 |
| 4 | `hist_gbdt` | `max_iter=1200`, `learning_rate=0.015`, `max_leaf_nodes=16`, `min_samples_leaf=30`, `l2_regularization=3.0`, `max_features=0.4`, `early_stopping=false`, `class_weight={0:1.0, 1:8.0}` | `balanced_accuracy=0.7663` (CI 0.7345~0.7957), `roc_auc=0.8770`, `best_cut=0.8062`, `cut_headroom=0.0399`, `train_val_gap=0.1862` | (없음 — 평가 직후 루프 종료) |

참고: iteration 4에 적용된 하이퍼파라미터는 iteration 3 critic이 처방한 값(`max_iter=400`, `learning_rate=0.04`, `max_depth=4`, `max_leaf_nodes=12`, `min_samples_leaf=40`, `l2_regularization=10.0`)과 다르며, 용량을 되돌리는 대신 오히려 키운 설정(`max_iter=1200`, `l2_regularization=3.0`, `max_features=0.4`)이 실행되었습니다. 즉 "낮은 용량 + 가중 8" 조합은 이 런에서 실제로 검증되지 않았습니다. 네 시도 모두 `dropped_hyperparams=[]` 이므로 executor가 거부한 설정은 없습니다.

## 최고 성능 구성

iteration 2, `hist_gbdt`. 아래 값은 executor가 실제로 구성한 estimator에서 읽은 적용값입니다.

```
model: hist_gbdt
hyperparams:
  max_iter: 300
  learning_rate: 0.05
  max_depth: 3
  max_leaf_nodes: 8
  min_samples_leaf: 60
  l2_regularization: 10.0
  early_stopping: false
  class_weight: {"0": 1.0, "1": 9.0}
preprocessing:
  impute: none          # NaN을 모델이 직접 분기 (imputer 미구성)
  scale: false
  missing_indicator: false
  missing_count: false
split: stratified 60/20/20, seed 42
train_time_sec: 5.403
```

| 구분 | balanced_accuracy | 95% CI |
|---|---|---|
| 검증 20% (선택 근거) | **0.7990** | 0.7735 ~ 0.8229 |
| 홀드백 테스트 20% (1회 측정) | **0.7883** | 0.7628 ~ 0.8103 |

두 값의 차이 +0.0107은 **선택 편향의 크기**입니다 — 루프는 검증 숫자만 보고 최고 시도를 골랐기 때문입니다. 다만 검증 점수 0.7990이 테스트 CI(0.7628~0.8103) 안에 들어오므로, 이 행 수로는 그 차이를 0과 구분할 수 없습니다. 이 런이 실제로 입증한 값은 테스트 점수 쪽입니다.

검증 슬라이스의 나머지 지표: `f1=0.5413`, `accuracy=0.7613`, `precision=0.3960`, `recall=0.8551`, `specificity=0.742857`, `roc_auc=0.8736`, `pr_auc=0.6008`, `balanced_accuracy_at_best_cut=0.8050`, `cut_headroom=0.0060`, `brier=0.1637`, `calibration_error=0.2193`, `train_val_gap=0.0547`.

확률값 품질은 나쁩니다(`calibration_error=0.2193`, iteration 1의 0.0619 대비 악화). 이는 `class_weight` 1:9의 예상된 대가이며, 이 하니스에는 재보정 레버가 없습니다. 목표 지표는 랭킹+운영점이므로 점수에는 직접 반영되지 않지만, 이 모델의 출력 확률을 그대로 확률로 해석해서는 안 됩니다.

## 원인 분석

critic 판정은 3회 존재하며(`overfitting` → `wrong_model_family` → `data_issue`), 그 궤적 자체가 원인을 가리킵니다.

**1) 진짜로 확인된 개선은 한 번뿐입니다 — 과적합 해소.** iteration 1(CI 0.6765~0.7358)과 iteration 2(CI 0.7735~0.8229)는 구간이 겹치지 않고, 페어드 Δ도 +0.0941 (95% CI +0.0628~+0.1264)로 보고되었습니다. `train_val_gap` 0.2949→0.0547이 이 이동의 내용이며, critic의 첫 처방(용량 대폭 축소 + 양성 가중 9)은 지시한 대로 작동했습니다.

**2) 그 개선의 거의 전부가 운영점 축에서 나왔고, 랭킹 축은 사실상 움직이지 않았습니다.** 4회에 걸친 `balanced_accuracy_at_best_cut`은 0.7984 → 0.8050 → 0.8037 → 0.8062, 총 폭 0.0078입니다. `roc_auc`도 0.8691 → 0.8736 → 0.8759 → 0.8770으로, 이 하니스에서 페어드로 구분 가능한 해상도(대략 0.003~0.006) 근처를 넘지 못하는 이동입니다. iteration 2 시점에 `cut_headroom`은 이미 0.0060으로, 남은 격차 0.0352의 17%만 임계값 쪽에서 살 수 있었고 나머지 83%는 랭킹이었습니다. 목표 0.8342는 KS 0.6684를 요구하는데 이 런의 최고 랭킹은 KS 0.6100 수준이며, 목표 자체가 baseline 랭킹 상한 0.7712를 설계상 초과(`exceeds_ranking_ceiling=true`, `ks_shortfall=0.1261)`합니다.

**3) iteration 2 이후의 시도들은 이 데이터로 서로 구분되지 않거나, 구분되더라도 하락입니다.** iteration 3(CI 0.7423~0.8007)과 iteration 4(CI 0.7345~0.7957)는 iteration 2의 CI와 겹치므로 단일 슬라이스 구간만으로는 서열을 매길 수 없습니다. 다만 iteration 3에 대해서는 iteration 2 대비 페어드 Δ −0.0272 (95% CI −0.0494~−0.0046)가 보고되어, `wrong_model_family` 처방이 점수를 **잃었다**는 판정 근거가 있습니다. 그리고 잃은 이유는 랭킹이 아니라 운영점입니다: `scale_pos_weight=7.0`에서 `recall=0.6957 < specificity=0.8479`로 컷이 보수적으로 되돌아갔고(`cut_headroom` 0.0060→0.0319), `train_val_gap`은 0.1810으로 다시 벌어졌습니다. iteration 4는 가중을 8로 두었지만 용량을 함께 키운 탓에 동일한 증상(`recall=0.6848 < specificity=0.8479`, `cut_headroom=0.0399`, `gap=0.1862`)을 반복했습니다. 즉 **컷의 위치는 `class_weight` 단독이 아니라 용량·보정 상태와 함께 결정**되며, 잘 맞았던 조합은 "낮은 용량(depth 3 / leaf 8) + 가중 9"라는 단 한 점뿐이었습니다.

**4) 두 패밀리 교체는 진단을 반박했습니다.** critic이 `wrong_model_family`를 근거로 지목한 `xgboost`는 `roc_auc`를 0.8736→0.8759, `best_cut`을 0.8050→0.8037로 옮겼을 뿐입니다. 세 번째 critic이 이를 `data_issue`로 재판정한 것은 타당합니다 — 두 패밀리의 랭킹 상한 차이가 0.0013인데 필요한 상한 상승폭은 0.0292이므로, 세 번째 패밀리로도 산술적으로 메워지지 않습니다. 남은 병목은 estimator가 아니라 **입력 컬럼과 목표선 설정**입니다: 5,026 학습 행 / 241 인코딩 컬럼, 각 평점이 원값과 `d_*` 구간 원핫으로 중복 표현, `expected_num_interested_in_me` 78.5% 결측, 클래스 비 5.07:1.

**5) `unsupported_claims`에 대하여.** iteration 1·3·4의 플랜에서 `feature_engineering` 문자열이 감지되었습니다. 이 기록에는 해당 플랜 본문이 남아 있지 않아 플랜이 그 기능에 *의존*했는지 단지 한계를 *언급*했는지 판별할 수 없으므로, 이 항목을 성능 손실의 원인으로 귀속하지 않습니다. 확실한 사실만 적으면: executor는 파생·조합·삭제·재인코딩을 하지 않으므로 네 시도 모두 카드에 실린 컬럼을 그대로 사용했고, 따라서 **파생 특성이 랭킹을 얼마나 올릴 수 있는지는 이 런에서 전혀 시험되지 않았습니다.**

기록된 데이터 주의사항은 없으므로(`없음`), 위 결론 중 어느 것도 신뢰할 수 없는 컬럼·분할에 근거하고 있지 않습니다.

## 다음 단계 제안

1. **목표선을 재도출하거나 목표 지표를 랭킹 축으로 옮길 것 (최우선).** 0.8342는 baseline 0.6685 + margin 0.5로 파생된 값이고, 메타데이터가 이미 `exceeds_ranking_ceiling=true`, `required_ks=0.6684`, `ks_shortfall=0.1261`이라고 적고 있습니다. 실측으로도 4회 시도의 랭킹 상한이 0.7984~0.8062에 갇혔습니다. 즉 현재 컬럼 집합에서 이 기준선은 하이퍼파라미터·패밀리 예산으로 도달 가능한 대상이 아닙니다. `passable_margin=0.309`에 해당하는 선(≈0.80대 초반)으로 재설정하거나, 판정 지표를 `roc_auc`/`pr_auc`(현재 최고 0.8770 / 0.6161)로 바꿔 랭킹 개선을 직접 보상하도록 하는 것이 다음 런의 예산을 낭비하지 않는 유일한 방법입니다.

2. **랭킹을 올릴 수 있는 유일한 레버는 특성이므로, 카드 상류에서 컬럼을 만들어 넣을 것.** executor는 파생 특성을 만들 수 없으니(`missing_indicator`/`missing_count` 외 추가 불가), 이 작업은 CSV/카드 단계에서 해야 합니다. 근거가 있는 후보: (a) 자기평가와 상대평가의 쌍(pair) 차이·일치도 — `attractive` vs `attractive_partner`, `shared_interests_important` vs `shared_interests_partner` 등, 카드에서 `target_corr`가 `moderate`~`strong`인 축들(`like`가 유일한 `strong`, `attractive_o`/`funny_o`/`shared_interests_partner` 등이 `moderate`)에 집중; (b) `field`(고카디널리티로 드롭됨)를 소수 그룹으로 축약해 다시 투입; (c) 각 평점의 원값과 `d_*` 구간 원핫 중복(180 원핫 레벨)을 한쪽으로 정리해 5,026 행 대비 241 컬럼의 비율을 낮추기. 이 중 어느 것도 이번 런에서 시험되지 않았으므로, 남은 0.0292의 랭킹 상한이 여기 있는지 없는지는 아직 미지입니다.

3. **현재 하니스에서 예산이 남으면, iteration 2 구성을 고정하고 `class_weight`만 좁게 흔들 것 — 단, 상한 기대치는 +0.006.** iteration 2의 `cut_headroom=0.0060`은 그 fit의 랭킹을 이미 거의 최적 지점에서 자르고 있음을 뜻합니다. 그리고 iteration 3·4는 가중을 8·7로 낮추면서 용량도 함께 바꿨기 때문에 "저용량 + 가중 8"이라는 브래킷 중앙은 실제로 미검증입니다. 따라서 `max_iter=300, learning_rate=0.05, max_depth=3, max_leaf_nodes=8, min_samples_leaf=60, l2_regularization=10.0, early_stopping=false`를 비트 단위로 유지한 채 `class_weight={0:1,1:8}` 및 `{0:1,1:10}` 두 점만 재는 것을 권합니다. 이는 목표 달성 경로가 아니라 최고 구성의 확정(consolidation)이며, 리포트에는 그렇게 적어야 합니다.

4. **분할 구조의 타당성을 운영자와 확인할 것.** 이 데이터는 참가자 쌍(pair) 단위 행이고 `wave` 컬럼이 존재하지만, 채점 프로토콜은 `stratified: true, grouped_by: null`의 무작위 층화 분할입니다. 동일 참가자의 행이 학습/검증/테스트에 동시에 걸릴 수 있으며, 그 경우 검증 0.7990과 테스트 0.7883이 **같은 방향으로** 낙관적일 수 있습니다(현재 두 값 차이는 CI 내부이므로 이 가설을 이 숫자만으로는 확인도 반증도 못 합니다). executor는 분할·시드를 바꿀 수 없으므로 이는 하니스 설정 변경 요청 사항입니다. 참가자/wave 단위 그룹 분할로 한 번 재측정해 두면, 위 1·2번에서 얻는 개선이 실제 일반화인지 참가자 누출인지 판별할 수 있습니다.