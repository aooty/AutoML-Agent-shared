# AutoML 실행 최종 보고서 — `adult` (binary_classification, 목표 지표 `balanced_accuracy`)

## 요약

목표는 **달성하지 못했습니다**. 요구 임계값은 `balanced_accuracy` 0.9182였으나, 4회 시도 중 최고 검증 점수는 iteration 2의 `xgboost`가 기록한 **0.8467 (95% CI 0.8391~0.8550)** 으로 0.0715 부족했습니다. 한 번도 사용되지 않은 최종 테스트 20%에서 같은 모델은 **0.8426 (95% CI 0.8346~0.8508)** 을 기록했습니다. 루프는 5회 예산 중 4회를 쓰고 연속 미개선(정체)으로 조기 종료되었으며, baseline `logreg` 0.7664는 크게 넘었지만 임계값에는 도달하지 못했습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=500`, `learning_rate=0.06`, `max_leaf_nodes=63`, `l2_regularization=1.0`, `min_samples_leaf=40`, `class_weight='balanced'`, `early_stopping=False` / `impute: none`, `scale: false` | `balanced_accuracy` **0.8407** (CI 0.8325~0.8498), `roc_auc` 0.9260, `pr_auc` 0.8259, `balanced_accuracy_at_best_cut` 0.8464, `cut_headroom` 0.005654, `train_val_gap` 0.0862 | `wrong_model_family` — 운영점은 이미 소진(`cut_headroom`이 남은 격차의 7%), 남은 93%는 ranking 축. 다른 정규화 체계의 boosted family(`xgboost`) + `missing_count` 제안 |
| 2 | `xgboost` | `n_estimators=1200`, `learning_rate=0.03`, `max_depth=5`, `subsample=0.8`, `colsample_bytree=0.7`, `min_child_weight=5`, `reg_lambda=3.0`, `scale_pos_weight=3.18`, `early_stopping_rounds=60` / `impute: none`, `scale: false`, `missing_count: true` | `balanced_accuracy` **0.8467** (CI 0.8391~0.8550) — 최고, `roc_auc` 0.9293, `pr_auc` 0.8296, `balanced_accuracy_at_best_cut` 0.8495, `cut_headroom` 0.00278, `train_val_gap` 0.0285 | `wrong_model_family` — paired Δ +0.0060 (CI +0.0010~+0.0115)로 미미하고 family/preprocessing이 동시에 바뀌어 귀속 불가. 구조적으로 다른 ranking 생성기(`extra_trees`) 제안 |
| 3 | `extra_trees` | `n_estimators=800`, `min_samples_leaf=2`, `class_weight='balanced'` / `impute: median`, `scale: false`, `missing_count: true` | `balanced_accuracy` **0.8134** (CI 0.8054~0.8211), `roc_auc` 0.9012, `pr_auc` 0.7411, `balanced_accuracy_at_best_cut` 0.8173, `cut_headroom` 0.0039, `train_val_gap` 0.0603 | `wrong_model_family` — paired Δ −0.0333 (CI −0.0404~−0.0269)로 확정 회귀. tree family 교체 폭(0.0322)이 남은 ranking 격차(0.0687)의 절반뿐. iteration 2 구성을 용량만 키워(`max_depth=10`, `min_child_weight=1`, `n_estimators=2000`) 재적합하고, 그래도 ~0.005 미만이면 0.9182를 도달 불가로 보고하라고 지시 |
| 4 | `mlp` | `hidden_layer_sizes=[256,128,64]`, `alpha=0.0003`, `learning_rate_init=0.0008`, `batch_size=256`, `max_iter=300`, `early_stopping=True` (`stopped_at_iter=20`) / `impute: median`, `scale: true`, `missing_count: true` | `balanced_accuracy` **0.7724** (CI 0.7630~0.7826), `roc_auc` 0.9099, `pr_auc` 0.7730, `balanced_accuracy_at_best_cut` 0.8294, `cut_headroom` 0.056981, `recall` 0.6157 vs `specificity` 0.9291, `calibration_error` 0.0219 | (없음 — 평가 직후 루프 종료) |

## 최고 성능 구성

**iteration 2 / `xgboost`** — 아래 값은 실행기가 실제로 만든 추정기에서 읽은 적용값입니다 (`dropped_hyperparams`는 비어 있음).

```
model: xgboost
hyperparams:
  n_estimators: 1200
  learning_rate: 0.03
  max_depth: 5
  subsample: 0.8
  colsample_bytree: 0.7
  min_child_weight: 5
  reg_lambda: 3.0
  scale_pos_weight: 3.18
  early_stopping_rounds: 60
preprocessing:
  impute: none          # NaN을 모델이 직접 분기
  scale: false
  missing_indicator: false
  missing_count: true
internal_validation: held_out_rows=2931, fit_rows=26373,
                     validation_fraction=0.1, stopped_at_iter=1198 / max_iter=1200
train_time_sec: 12.831 (max_train_time_sec=600)
```

검증 지표: `balanced_accuracy` 0.8467 (CI 0.8391~0.8550), `roc_auc` 0.9293, `pr_auc` / `average_precision` 0.8296, `f1` 0.7212, `accuracy` 0.8415, `precision` 0.6227, `recall` 0.8567, `specificity` 0.8368, `brier` 0.1061, `calibration_error` 0.0955, `train_balanced_accuracy` 0.8752, `train_val_gap` 0.0285. 진단용 값: `balanced_accuracy_at_best_cut` 0.8495, `cut_headroom` 0.00278.

**최종 held-back 측정:** 테스트 20%(루프 중 단 한 번도 사용되지 않은 행)에서 `balanced_accuracy` = **0.8426 (95% CI 0.8346~0.8508)**. 검증 0.8467 대비 **+0.0042**이며, 이 차이가 선택 편향의 크기입니다 — 루프는 검증 숫자만 보고 최고 시도를 골랐습니다. 다만 검증 점수가 테스트 CI 안에 들어오므로, 이 행 수에서는 그 차이를 0과 구분할 수 없습니다. 어느 쪽 숫자로 읽어도 임계값 0.9182에는 미달입니다.

## 원인 분석

critic이 3회 실행되어 3개의 진단이 존재하고, 세 진단 모두 `failure_type: wrong_model_family`로 같은 방향을 가리켰습니다. 핵심은 **격차가 운영점(operating point)이 아니라 ranking 축에 있고, 실행기가 ranking 축에서 쓸 수 있는 레버가 모델 교체밖에 없다**는 점입니다.

- **운영점은 세 tree family 모두에서 이미 소진되었습니다.** `cut_headroom`은 0.005654 → 0.00278 → 0.0039로, 남은 격차(각각 0.0775 / 0.0715 / 0.1048)의 4~7%에 불과합니다. iteration 2에서 `recall` 0.8567 vs `specificity` 0.8368은 이미 거의 균형점이며, `scale_pos_weight=3.18`(카드의 `imbalance_ratio`)이 `balanced_accuracy`가 원하는 교차점입니다. 즉 `class_weight`/`scale_pos_weight`를 더 흔들어 얻을 것은 실질적으로 없습니다.
- **ranking 자체가 임계값 아래에서 포화했습니다.** 임계값 0.9182는 KS 0.8364를 요구하지만, 최고 ranking이 허용하는 최적 컷은 `balanced_accuracy_at_best_cut` 0.8495(implied KS 0.6990)입니다 — 0.1374 부족. 시도된 4개 family의 최적 컷 폭은 0.8173~0.8495(0.0322)로, 최고 컷에서 임계값까지의 거리 0.0687의 절반에 불과합니다. baseline의 `balanced_accuracy_at_best_cut` 0.8238에서 0.8495까지 올린 것이 이번 루프가 ranking 축에서 실제로 벌어들인 전부입니다.
- **iteration 1과 2는 이 데이터로 구분되지 않습니다.** 두 시도의 CI(0.8325~0.8498 / 0.8391~0.8550)는 겹칩니다. critic이 보고한 paired Δ는 +0.0060(CI +0.0010~+0.0115)로 부호는 잡히지만 크기가 미미하고, 그 행은 family와 `missing_count`를 동시에 바꿨으므로 어느 레버의 공로도 아닙니다. 따라서 "hist_gbdt → xgboost가 효과가 있었다"는 이야기를 만들지 않습니다. 반대로 interval보다 훨씬 큰 차이는 두 개뿐입니다: iteration 3의 −0.0333(확정 회귀), iteration 4의 0.7724(최고 대비 −0.0743).
- **용량 부족도, 자원 제약도 아닙니다.** iteration 1의 `train_val_gap` 0.0862는 과적합 쪽이었고, iteration 2에서 0.0285로 줄었는데도 `train_balanced_accuracy`는 0.8752 — 학습을 본 행에서조차 0.9182에 못 미칩니다. 학습 시간은 8.4~18.5초로 예산 600초에 한참 못 미치고, `dropped_hyperparams`는 모든 시도에서 비어 있어 거부된 설정도 없습니다.
- **iteration 4는 critic의 처방을 실행하지 않았습니다.** iteration 3의 critic은 "iteration 2 구성을 용량만 키운 `xgboost`(`max_depth=10`, `min_child_weight=1`, `n_estimators=2000`, `early_stopping_rounds=100`)"를 지시했지만 실제로 돌아간 것은 `mlp`입니다. 따라서 마지막 남은 ranking 가설은 **검증되지 않은 상태**로 루프가 끝났습니다. `mlp`는 구조적으로 다른 ranking 생성기라는 의미는 있었지만 최적 컷 0.8294로 `xgboost`의 0.8495보다 낮았고, `mlp`에는 실행기가 쓸 수 있는 불균형 레버가 없어 기본 컷이 `specificity` 0.9291 / `recall` 0.6157로 크게 치우쳐 `cut_headroom` 0.0570을 남겼습니다. 다만 `calibration_error` 0.0219 / `brier` 0.1006으로 확률 품질만은 tree family(0.0955 / 0.1061)보다 뚜렷하게 좋았습니다.
- **`unsupported_claims`에 대하여:** iteration 1과 4의 계획 산문에서 `feature_engineering` 문자열이 감지되었습니다. 이 보고서에는 두 계획의 전문이 남아 있지 않아, 계획이 해당 기능에 *의존*했는지 단지 제약을 *언급*했는지는 확인할 수 없습니다. 따라서 두 시도가 그 때문에 무엇을 잃었다고 쓰지 않겠습니다 — 기록된 숫자는 이 실행기의 레버로 얻은 값 그대로입니다. 파생 피처 부재는 아래 `다음 단계 제안`에서 "만들어야 할 것"으로 다룹니다.

요약하면, 이번 실행은 임계값 0.9182가 **이 14개 열 + 이 실행기의 레버(모델 교체, 불균형 가중, 두 개의 missingness 열)로는 도달 불가능한 위치**에 있다는 증거를 축적했습니다. 목표 정의 자체도 이를 예고합니다: `ranking_ceiling` 0.8238, `exceeds_ranking_ceiling: true`, `ks_shortfall` 0.1888, `passable_margin` 0.245.

## 다음 단계 제안

1. **실행되지 않은 마지막 처방을 1회로 마무리하라 (비용 최소, 정보 최대).** iteration 3 critic의 지시대로 iteration 2와 동일한 파이프라인(`impute: none`, `scale: false`, `missing_count: true`)에서 `xgboost`를 `max_depth=10`, `min_child_weight=1`, `n_estimators=2000`, `learning_rate=0.03`, `subsample=0.9`, `colsample_bytree=0.9`, `reg_lambda=1.0`, `scale_pos_weight=3.18`, `early_stopping_rounds=100`으로만 바꿔 적합하십시오. iteration 2는 12.8초/600초를 썼으므로 예산 위험이 없고, family를 고정한 채 용량만 움직이므로 결과가 단일 레버에 귀속됩니다. 판정 기준은 `balanced_accuracy_at_best_cut`과 `roc_auc`: 0.8495 / 0.9293에서 ~0.005 미만 움직이면 boosted-tree ranking은 포화로 확정입니다.
2. **임계값을 재설정하거나 목표 지표를 ranking 축으로 옮기십시오.** 요구 KS 0.8364 대비 실측 최고 KS는 0.6990이고, 4개 family의 최적 컷 폭(0.0322)이 남은 격차(0.0687)의 절반뿐입니다. 현재 레버 집합에서 현실적인 상한은 `balanced_accuracy` 0.85 부근(테스트 0.8426, 검증 최적 컷 0.8495)입니다. 임계값 0.9182를 유지하려면 그것은 모델링 문제가 아니라 **입력 열의 문제**임을 목표 문서에 명시하는 것이 옳습니다.
3. **피처 파생 능력을 실행기 밖에서 확보하십시오 — 이것이 유일하게 남은 큰 레버입니다.** 실행기는 `missing_indicator`/`missing_count` 외에는 어떤 열도 만들거나 결합하거나 버릴 수 없습니다. `education-num`(target_corr strong), `age`·`hours-per-week`·`capital-gain`(moderate) 조합, `capital-gain`/`capital-loss`의 고왜도 변환, `occupation × education` 상호작용 같은 것은 **카드 단계에서 열로 추가**되어야 합니다. 동시에 `fnlwgt`는 `target_corr: none`인 표본 가중치성 열이므로 카드에서 제외하는 것을 검토하십시오(플랜으로는 드롭할 수 없습니다). 이것이 풀리면 ranking 상한 자체가 올라가고, 그때 비로소 0.9182가 검증 가능한 목표가 됩니다.
4. **다음 예산에서 하지 말 것을 고정하십시오.** (a) 임계값/`class_weight` 재탐색 — tree family의 `cut_headroom`이 0.0028~0.0057로 남은 격차의 4~7%뿐입니다. (b) 또 다른 tree family 교체 — 교체 폭이 이미 격차의 절반 이하로 측정되었습니다. (c) `impute: none`과 `missing_indicator` 병용 — 문서상 비트 단위로 중복이며, 임퓨트 파이프라인에서의 missingness 레버 효과 크기는 이 프로젝트에서 확정되지 않았으므로 예상 이득으로 계산해서는 안 됩니다. 여유 예산은 대신 **선택 편향 축소**에 쓰는 편이 낫습니다: 검증 20% 단일 슬라이스에서 시도 간 CI가 서로 겹치는 상황(iteration 1 vs 2)이 실제로 발생했고, 이번 검증-테스트 차이 +0.0042도 그 슬라이스 하나에 의존한 결과입니다.

> 참고: `Data caveats`에 기록된 주의사항은 없으므로, 위 제안 중 특정 열·값·분할의 신뢰성 때문에 배제되는 항목은 없습니다. 단 3번의 `fnlwgt` 제외와 파생 열 추가는 카드/프로파일러를 고치는 작업이며, 플랜만으로는 실행기에서 시도할 수 없습니다.