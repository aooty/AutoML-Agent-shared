# AutoML 최종 리포트 — `adult` / balanced_accuracy

## 요약

**목표는 달성되지 못했습니다.** 목표 기준선은 `balanced_accuracy` 0.8832였으나, 5회 예산 중 3회를 사용한 뒤 연속 미개선(정체)으로 조기 종료되었고, 최고 검증 점수는 iteration 1의 `hist_gbdt`가 기록한 0.8452 (95% CI 0.8365–0.8529)입니다. 한 번도 사용되지 않은 최종 테스트 20%에서 같은 모델은 0.8416 (95% CI 0.8335–0.8501)을 기록했습니다. 카드의 baseline(`logreg`, validation 0.7664, CI 0.7575–0.7762)보다는 분명히 높고 참조된 `ranking_ceiling` 0.8238도 넘었지만, 요구 기준까지는 약 0.038이 부족합니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.08`, `max_depth=8`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `min_samples_leaf=20`, `early_stopping=false`, `class_weight='balanced'` / `impute: none`, `scale: false` | `balanced_accuracy` **0.8452** (CI 0.8365–0.8529), `roc_auc` 0.9291, `cut_headroom` 0.001507 — 최고 성능 | `wrong_model_family`: 운용점(operating point)은 이미 소진(recall 0.8502 vs specificity 0.8402, `cut_headroom` 0.0015 = 남은 격차의 4%), 남은 96%는 랭킹 축 → 다른 트리 계열 시도 지시 |
| 2 | `xgboost` | `n_estimators=1200`, `learning_rate=0.03`, `max_depth=8`, `min_child_weight=2`, `subsample=0.9`, `colsample_bytree=0.8`, `reg_lambda=1.0`, `scale_pos_weight=3.18`, `early_stopping_rounds=60` / `impute: none`, `scale: false` | `balanced_accuracy` 0.8423 (CI 0.8348–0.8510), `roc_auc` 0.9288, 내부 조기중단 `stopped_at_iter=759` — iteration 1과 **구분 불가** (paired Δ −0.0029, CI −0.0074~+0.0014) | `data_issue`: 두 계열의 `balanced_accuracy_at_best_cut`이 0.8467 / 0.8475(스팬 0.0008)로 랭킹 상한이 거의 동일 → 계열 교체로 남은 0.0357을 덮을 수 없음, 파이프라인(정보량) 레버를 단독으로 1회 시험하라 |
| 3 | `hist_gbdt` | iteration 1과 동일 하이퍼파라미터 / 적용 전처리 `impute: median`, `missing_count: true`, `scale: false`. `dropped_hyperparams`: `missing_count`, `missing_indicator`, `scale` (hyperparams 키로 전달되어 estimator가 거부) | `balanced_accuracy` 0.8452 — iteration 1과 **모든 지표가 자리수까지 동일** | (없음 — 평가 직후 루프 종료) |

## 최고 성능 구성

iteration 1, `hist_gbdt`. 아래 값은 해당 attempt의 `hyperparams` / `preprocessing`(실제 빌드된 파이프라인)에서 그대로 읽은 것입니다.

- **model**: `hist_gbdt`
- **hyperparams**: `max_iter=400`, `learning_rate=0.08`, `max_depth=8`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `min_samples_leaf=20`, `early_stopping=false`, `class_weight='balanced'` (`dropped_hyperparams`는 비어 있음 — 요청된 모든 키가 적용됨)
- **preprocessing**: `impute: none`(모델이 NaN을 직접 분기), `scale: false`, `missing_indicator: false`, `missing_count: false`
- **프로토콜**: seed 42, stratified 60/20/20 — 검증 20%로 채점·선택, 테스트 20%는 루프 종료 후 1회만 채점
- **train_time_sec**: 12.084

| 구분 | balanced_accuracy |
|---|---|
| validation (선택에 사용됨) | **0.8452** (95% CI 0.8365–0.8529) |
| held-back test (한 번도 사용되지 않은 행) | **0.8416** (95% CI 0.8335–0.8501) |

두 값의 차이 +0.0036이 이 런의 **선택 편향 크기**입니다. 루프는 검증 숫자만 보고 선택했으므로, 이 모델에 대해 실제로 입증된 값은 테스트 0.8416입니다. 다만 검증 점수가 테스트 CI 안에 들어 있어 이 행들만으로는 이 차이를 0과 구분할 수 없습니다. 어느 쪽 숫자로 보아도 기준 0.8832에는 미달입니다.

검증 슬라이스의 기타 지표(iteration 1): `roc_auc` 0.9291, `pr_auc` / `average_precision` 0.8319, `accuracy` 0.8426, `f1` 0.7210, `precision` 0.6258, `recall` 0.8502, `specificity` 0.8402, `balanced_accuracy_at_best_cut` 0.8467, `cut_headroom` 0.001507, `brier` 0.10499, `calibration_error` 0.089875, `train_balanced_accuracy` 0.8906, `train_val_gap` 0.045378.

## 원인 분석

critic 진단은 2회 존재하며(iteration 1, 2), 두 진단이 가리키는 방향은 일관됩니다: **남은 격차는 운용점이 아니라 랭킹 축에 있다.**

- **운용점 축은 이미 소진되었습니다.** iteration 1에서 `class_weight='balanced'`가 recall 0.8502 / specificity 0.8402로 두 항을 거의 교차점에 놓았고 `cut_headroom`은 0.001507 — 즉 이 랭킹에서 가능한 최선의 컷조차 0.8467이며, 남은 0.0380 중 96%는 컷 이동으로 회수할 수 없습니다. iteration 2에서 `scale_pos_weight=3.18`은 반대편(recall 0.8348 < specificity 0.8498)으로 약간 넘어갔을 뿐이고 `cut_headroom` 0.005165도 남은 0.0409의 13%에 불과합니다. 이 실행기에는 임계값 탐색 레버가 없으며, 있었더라도 회수량은 이 크기입니다.
- **모델 계열 교체는 값을 지불하지 않았습니다.** iteration 1(0.8452, CI 0.8365–0.8529)과 iteration 2(0.8423, CI 0.8348–0.8510)의 구간은 크게 겹치고 paired Δ는 −0.0029 (CI −0.0074~+0.0014, P(better) 0.092)입니다. **이 데이터로는 두 시도가 구분되지 않으므로**, 이 차이를 성능 하락으로 서술할 근거는 없습니다. 실질적으로 읽을 수 있는 신호는 두 계열의 랭킹 상한이 0.8467 vs 0.8475, 즉 스팬 0.0008로 사실상 같다는 점이고, 이는 최선 컷에서 기준까지 남은 0.0357의 1/45입니다. KS로 보면 기준은 0.7664를 요구하는데 달성된 최선은 0.6950입니다.
- **과적합/용량이 병목이 아닙니다.** iteration 1의 `train_balanced_accuracy` 0.8906, `train_val_gap` 0.045378 — 훈련 측 성능 자체가 기준 0.8832를 겨우 넘는 수준이므로, 분산을 줄여 회수할 여지가 크지 않습니다. iteration 2도 `train_val_gap` 0.063669에 조기중단이 759회에서 발동했습니다. 실행 시간(10–13초)도 제약이 아니었습니다.
- **iteration 3(결측 정보 레버)은 아무 변화를 만들지 못했습니다.** 적용 전처리는 `impute: median` + `missing_count: true`로 기록되었지만 모든 지표가 iteration 1과 **자리수까지 동일**합니다. 동시에 `dropped_hyperparams`에 `missing_count`, `missing_indicator`, `scale`이 남아 있어(전처리 블록에 속하는 키가 hyperparams로도 전달되어 estimator가 거부) 이 시도는 "레버가 효과 없음"과 "레버가 실제로 적용되지 않았음"을 깔끔히 분리하지 못합니다. 어느 해석이든 이 런은 결측 레버에서 측정 가능한 이득을 얻지 못했으며, 데이터 특성(전체 결측률 0.0095, 최악 열 0.0575)도 큰 이득을 기대하기 어렵게 만듭니다.
- **목표 기준 자체가 도달 불가 영역에 설정되었을 가능성이 큽니다.** 목표는 baseline에서 파생된 값(`source: derived`)이고 참조 블록 스스로 `exceeds_ranking_ceiling: true`, `ks_shortfall: 0.1188`, `passable_margin: 0.245`를 기록합니다. 실제로 달성된 검증 0.8452 / 테스트 0.8416은 `ranking_ceiling` 0.8238을 이미 넘었습니다. 즉 이 런은 "모델이 나빴다"보다 "주어진 열이 담고 있는 정보량으로는 0.8832가 서지 않는다"는 방향의 증거를 모았습니다.
- iteration 1의 `unsupported_claims`에 `feature_engineering`이 기록되어 있으나, 해당 plan 본문은 파생 피처에 의존하지 않고 `impute: none`으로 NaN을 직접 분기시키는 전략과 원핫 인코딩된 열을 언급한 것뿐입니다. 따라서 이 시도가 그로 인해 무엇을 잃었다고 볼 근거는 없으며, 파생 피처는 "실패한 것"이 아니라 "이 실행기에 없어서 만들어야 하는 것"으로 다음 단계에 둡니다.

## 다음 단계 제안

기록된 데이터 caveat이 없으므로 열·값·분할의 신뢰성을 이유로 배제되는 항목은 없습니다. 우선순위 순:

1. **목표 임계값을 재검토하십시오(최우선).** 참조 블록 자체가 `exceeds_ranking_ceiling: true`, `ks_shortfall: 0.1188`을 기록하고, 두 트리 계열의 랭킹 상한이 0.8467 / 0.8475로 0.0008 안에 모였습니다. 남은 0.036–0.038을 계열 교체(측정된 스팬 0.0022–0.0077 roc_auc)나 동일 계열 재튜닝(0.0032 roc_auc)으로 덮을 수 있다는 증거는 이 런에 없습니다. 현재 열 구성에 대한 현실적 상한을 기준으로 목표를 다시 도출하고, 달성치 0.8416(테스트)을 baseline 0.7664 대비 개선으로 평가하는 것이 정직한 결론입니다.
2. **랭킹 축을 올릴 유일한 남은 레버는 열 자체이며, 그것은 plan이 아니라 카드에서 해야 합니다.** 이 실행기는 비율·상호작용·재인코딩·열 삭제를 하지 못하므로(`missing_indicator` / `missing_count`가 유일한 예외), `capital-gain`/`capital-loss`의 결합, `education-num × hours-per-week`, `age` 구간화 같은 파생 열은 **소스 CSV와 데이터 카드에 미리 만들어 넣은 뒤** 다시 돌려야 합니다. 근거: `cut_headroom` 0.0015로 운용점은 소진되었고 격차의 96%가 랭킹 축에 남아 있는데, 이 실행기 안에서 랭킹을 움직일 수 있는 레버는 이미 두 계열로 소진되었습니다. 이것이 풀리면 0.8832 도달 가능성을 처음으로 실제로 검증할 수 있습니다.
3. **iteration 3의 재현 문제를 확정하십시오(저비용).** `impute: median` + `missing_count: true`가 기록되었는데 지표가 iteration 1과 비트 단위로 같고, 동시에 같은 키들이 `dropped_hyperparams`에 있습니다. 전처리 키를 `preprocessing` 블록에만 넣어 1회 재실행하면 "레버 무효"와 "레버 미적용"이 분리됩니다. 다만 기대 이득은 낮게 잡으십시오 — 전체 결측률 0.0095, 최악 열 0.0575이고, `impute: none`을 쓰는 계열에서 `missing_indicator`는 이미 중복으로 정리되어 있습니다.
4. **확률값을 쓰려면 별도 캘리브레이션 단계를 파이프라인 밖에 두십시오.** 최고 구성의 `calibration_error` 0.089875, `brier` 0.10499 — 예측 확률이 평균 약 9%p 어긋나 있습니다. 이 실행기에는 재캘리브레이션 레버가 없고 목표 지표로도 삼을 수 없으므로, 확률을 의사결정에 직접 사용하는 소비자가 있다면 런 외부에서 처리해야 합니다. (`balanced_accuracy`/`roc_auc` 자체는 이 문제와 무관합니다.)