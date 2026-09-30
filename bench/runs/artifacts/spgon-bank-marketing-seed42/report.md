# AutoML 최종 리포트 — `bank-marketing` (목표: `balanced_accuracy` 최대화)

## 요약

목표는 달성했습니다. 통과 기준 `balanced_accuracy` **0.7465** 에 대해, 최고 구성인 iteration 1의 `hist_gbdt`가 검증 슬라이스에서 **0.8787 (95% CI 0.8676~0.8887)** 를 기록했고, 한 번도 사용되지 않은 최종 테스트 20%에서 **0.8582 (95% CI 0.8474~0.8692)** 를 기록해 기준선(`logreg` 0.662)과 통과선을 모두 여유 있게 넘었습니다. 5회 예산 중 3회를 사용했고, iteration 2·3이 iteration 1을 넘지 못해 정체(stalled)로 조기 종료되었습니다. 즉 첫 시도가 곧 최종 승자였고, 이후 두 번의 시도는 "어느 축에 여유가 남아 있는지"를 확인하는 데 쓰였습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `min_samples_leaf=30`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=25`, `class_weight={0:1.0, 1:5.0}`, `random_state=42` | `balanced_accuracy` **0.8787** (CI 0.8676~0.8887), `roc_auc` 0.9423, `pr_auc` 0.6499, `recall` 0.8733 / `specificity` 0.8840, `cut_headroom` 0.0078, `train_val_gap` 0.0293, 5.07초 | `failure_type: hyperparam`. 통과선 대비 CI 전체가 ~0.12 위, 과적합 신호 없음. 운영점(operating point) 여유는 사실상 소진(`cut_headroom` 0.0078 < CI 폭 0.0211)이므로 남은 축은 랭킹뿐. `calibration_error` 0.0987 로 과확신은 남아 있음. → 같은 family에서 capacity 상향 처방 |
| 2 | `xgboost` | `n_estimators=700`, `learning_rate=0.04`, `max_depth=6`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=2.0`, `min_child_weight=5`, `scale_pos_weight=5.0`, `random_state=42`, `n_jobs=4` | `balanced_accuracy` **0.8635** (CI 0.8512~0.8759), `roc_auc` 0.9417, `balanced_accuracy_at_best_cut` 0.8857, `cut_headroom` 0.0222, `recall` 0.8233 / `specificity` 0.9037, `train_val_gap` 0.0777, 6.72초 | `failure_type: hyperparam`. iteration 1 대비 쌍대 Δ = **-0.0152** (CI -0.0235~-0.0067, P(improve)=0.000) → 측정 가능한 퇴행. 단 손실은 랭킹이 아니라 운영점: best-cut 0.8857 vs 0.8865 로 사실상 동일한데 `cut_headroom` 이 0.0078→0.0222 로 커짐. 동일한 1:5 비율이 두 엔진에서 다른 컷을 만든다는 것이 확인됨. → 미실행 상태로 남아 있던 `hist_gbdt` capacity 처방으로 복귀 |
| 3 | `hist_gbdt` | `max_iter=800`, `learning_rate=0.03`, `max_leaf_nodes=63`, `l2_regularization=0.5`, `min_samples_leaf=20`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=40`, `class_weight={0:1.0, 1:5.0}`, `random_state=42` | `balanced_accuracy` **0.8757** (CI 0.8647~0.8857), `roc_auc` 0.9417, `pr_auc` 0.6490, `balanced_accuracy_at_best_cut` 0.8841, `cut_headroom` 0.0084, `train_val_gap` 0.0383, 8.11초 | critic 없음(정체로 루프 종료). iteration 1과 CI가 거의 완전히 겹침 → 이 데이터로는 구분되지 않는 두 시도 |

세 시도 모두 `status: ok`, `dropped_hyperparams` 는 전부 비어 있어 제안한 설정은 모두 실제로 적용되었습니다. 학습 시간은 최대 8.11초로 600초 예산의 2% 미만이었습니다.

## 최고 성능 구성

**iteration 1 · `hist_gbdt`** (아래 값은 히스토리에 기록된 *적용된* 값입니다)

```
model: hist_gbdt
hyperparams:
  max_iter: 400
  learning_rate: 0.06
  max_leaf_nodes: 31
  l2_regularization: 1.0
  min_samples_leaf: 30
  early_stopping: true
  validation_fraction: 0.1
  n_iter_no_change: 25
  class_weight: {"0": 1.0, "1": 5.0}
  random_state: 42
preprocessing (applied):
  impute: median
  scale: false
  missing_indicator: false
  missing_count: false
protocol: stratified 60/20/20, seed 42 (카드 baseline과 동일 분할)
```

| 지표 | 검증 20% | 최종 테스트 20% (선택에 한 번도 쓰이지 않음) |
|---|---|---|
| `balanced_accuracy` | **0.8787** (95% CI 0.8676~0.8887) | **0.8582** (95% CI 0.8474~0.8692) |
| `roc_auc` | 0.9423 | — |
| `pr_auc` / `average_precision` | 0.6499 | — |
| `recall` / `specificity` | 0.8733 / 0.8840 | — |
| `precision` / `f1` / `accuracy` | 0.4995 / 0.6355 / 0.8828 | — |
| `balanced_accuracy_at_best_cut` / `cut_headroom` | 0.8865 / 0.0078 | — |
| `brier` / `calibration_error` | 0.0820 / 0.0987 | — |
| `train_balanced_accuracy` / `train_val_gap` | 0.9079 / 0.0293 | — |

검증 0.8787 과 테스트 0.8582 의 차이 **+0.0204** 는 선택 편향의 크기입니다. 루프는 검증 숫자만 보고 3개 시도 중 최고를 골랐으므로, 이 실행이 실제로 입증한 성능은 **테스트 0.8582 (CI 0.8474~0.8692)** 이며, 이 값 역시 통과선 0.7465와 기준선 0.662를 크게 상회합니다. 검증 점수를 성능 보고값으로 쓰면 약 0.02 만큼 과대평가하게 됩니다.

## 원인 분석

- **통과선은 첫 시도에서 이미 넘었고, 이후의 제약은 "무엇이 부족했나"가 아니라 "남은 여유가 어느 축에 있나"였습니다.** iteration 1 의 CI 하한 0.8676 이 이미 기준 0.7465 보다 0.12 이상 높습니다. 기준선 `logreg` 0.662 (CI 0.6496~0.6765) 대비 개선의 대부분은 랭킹 자체가 아니라 컷 위치에서 옵니다: 기준선의 `roc_auc` 0.9101 → 0.9423 은 개선이지만, 기준선의 `balanced_accuracy_at_best_cut` 0.84 와 최고 시도의 0.8865 차이(0.0465)에 비해 실제 `balanced_accuracy` 는 0.662 → 0.8787 로 0.2167 이 움직였습니다. `logreg` 는 `recall` 0.3478 로 컷이 다수 클래스 쪽에 심하게 치우쳐 있었고, `class_weight={0:1, 1:5}` 가 그 컷을 `recall` 0.8733 / `specificity` 0.8840 로 거의 최적점에 놓은 것이 이번 실행의 본질적 이득입니다.
- **운영점 축은 사실상 소진되었습니다.** iteration 1 의 `cut_headroom` 은 0.0078 — 즉 어떤 컷을 골라도 최대 0.0078 밖에 더 얻을 수 없고, 이는 이 슬라이스의 CI 폭 0.0211 의 절반도 안 되어 측정 불가능한 크기입니다. critic 이 두 차례 `failure_type: hyperparam` 을 낸 근거가 여기에 있습니다.
- **랭킹 축은 시도한 범위 안에서 평평합니다.** iteration 1 vs 3 은 `balanced_accuracy` 0.8787 vs 0.8757 이고 CI(0.8676~0.8887 vs 0.8647~0.8857)가 거의 전면적으로 겹칩니다 — **이 데이터로 구분되지 않는 두 시도**이므로, capacity 를 `max_leaf_nodes` 31→63, `learning_rate` 0.06→0.03, `max_iter` 400→800 으로 올린 것이 "성능을 떨어뜨렸다"고 서술할 근거는 없습니다. `roc_auc` 도 0.9423 vs 0.9417 (차이 0.0006) 로 쌍대 분해능(0.003~0.006) 이내입니다. 세 시도의 `balanced_accuracy_at_best_cut` 은 0.8865 / 0.8857 / 0.8841 로 총 산포가 0.0024 에 불과합니다: 두 tree family, 두 가지 capacity 설정 모두 같은 순서를 만들어 냈습니다.
- **해석 가능한 유일한 시도 간 차이는 iteration 2 의 퇴행이며, 그 원인은 랭킹이 아니라 컷입니다.** 쌍대 Δ -0.0152 (CI -0.0235~-0.0067) 로 0을 포함하지 않습니다. 그런데 best-cut 은 0.8857 로 iteration 1 과 동일 수준이고, `cut_headroom` 만 0.0078 → 0.0222 로 커졌으며 `recall` 0.8233 이 `specificity` 0.9037 보다 0.0804 낮아졌습니다. `scale_pos_weight=5.0` 은 `hist_gbdt` 의 `class_weight={0:1,1:5}` 와 같은 컷을 만들지 않는다 — 손실의 거의 전부가 이 한 가지에서 옵니다.
- **과적합은 원인이 아닙니다.** `train_val_gap` 은 0.0293 / 0.0777 / 0.0383 이고, 가장 큰 iteration 2 조차 검증 점수가 통과선보다 0.10 이상 높습니다.
- **숨어 있는 약점은 확률 품질입니다.** 최고 구성의 `calibration_error` 0.0987, `brier` 0.0820 — 예측 확률이 평균 약 9.9%p 어긋나 있습니다. `roc_auc` 는 캘리브레이션과 무관하므로 이 문제는 어떤 목표 지표에도 나타나지 않지만, 기본 `predict()` 컷이 그 miscalibration 위에 서 있다는 뜻입니다. 실행기에는 recalibration 레버가 없어 루프 안에서는 손댈 수 없었습니다.
- **`unsupported_claims` 에 대해:** iteration 1·2 의 계획에 `feature_engineering` 플래그가 붙었습니다. iteration 1 의 계획 본문을 읽으면 파생 변수를 *쓰겠다*는 것이 아니라 "이 파일에는 실제 NaN 이 없고 `unknown` 은 one-hot 레벨이라 트리가 직접 분기할 수 있으므로 median 유지, 결측 열 추가 안 함"이라고 **제약을 명시한 것**이므로, 이 시도가 잃은 것은 없습니다. iteration 2 의 계획 본문은 이 리포트에 제공되지 않아 같은 판단을 확인할 수 없습니다 — 근거 없이 손실을 귀속하지 않겠습니다. 어느 쪽이든 파생 변수는 실행기가 하지 않는 일이므로, 그것은 "실패한 것"이 아니라 "아직 만들지 않은 것"으로 다음 단계에 속합니다.

## 다음 단계 제안

1. **`class_weight` / `scale_pos_weight` 재조정에는 더 이상 예산을 쓰지 마십시오.** 최고 구성의 `cut_headroom` 0.0078 은 CI 폭 0.0211 의 절반 미만이므로, 컷을 완벽히 맞춘다 해도 이 슬라이스에서는 관측 불가능한 이득입니다. 같은 이유로 `hist_gbdt` 내부의 capacity 재탐색도 우선순위가 낮습니다 — iteration 1 과 3 이 이미 구분되지 않았습니다.
2. **`unknown` 마커를 실제 NaN 으로 바꾼 데이터 카드를 다시 만드는 것이, 지금 가장 정직하게 남은 레버입니다.** 카드 caveat 이 명시하듯 V2(0.6%), V4(4.1%), V9(28.8%), V16(81.8%) 의 `unknown` 은 변환되지 않은 상태이고 `missing.overall_rate` 는 0.0 입니다. 따라서 **현재 파일에서 `impute: none`, `missing_indicator`, `missing_count` 는 모두 무의미**합니다(상수이거나 효과 없음). 변환은 실행기가 할 수 없으므로 원본 파일/카드 단계에서 처리해야 하며, 그것이 되면 `hist_gbdt`·`xgboost` 의 `impute: none` 을 통한 native NaN 분기와 `missing_indicator` 의 비교가 처음으로 측정 가능한 실험이 됩니다.
3. **다만 그 실험은 "결측이 정보인가"를 함께 검증하는 설계로만 신뢰하십시오.** 실행기 문서가 기록한 대로, 랜덤 분할에서는 결측 열의 이득(0.0097 / 0.0065 / 0.0151 roc_auc)이 연속(contiguous) 분할에서 전부 0을 포함하는 크기로 붕괴한 사례가 있습니다. V16 은 81.8% 가 `unknown` 이라 "언제/어떻게 기록된 행인가"를 식별할 위험이 특히 큽니다. 현재 프로토콜은 stratified random 고정이므로, 이 검증은 별도 실행(다른 분할 정의)으로 해야 하고 그 결과 없이는 결측 열로 얻은 이득을 성능으로 보고하지 않는 것이 안전합니다.
4. **확률을 그대로 쓰는 용도가 있다면 루프 밖에서 캘리브레이션을 붙이십시오.** `calibration_error` 0.0987 / `brier` 0.0820 은 랭킹은 좋지만 확률이 체계적으로 과확신임을 뜻합니다. 루프에는 recalibration 레버가 없으므로, 선정된 iteration 1 구성을 그대로 재학습한 뒤 별도 홀드아웃에서 isotonic/Platt 보정을 적용하는 것은 파이프라인 바깥 작업입니다. 이는 목표 지표(`balanced_accuracy`)를 바꾸지는 않지만, 임계값을 운영에서 직접 고를 때의 0.0078~0.0084 여유를 실제로 회수할 수 있게 해 줍니다.
5. **선택 편향 크기(검증 0.8787 vs 테스트 0.8582, +0.0204)를 다른 seed 로 한 번 더 재보십시오.** 두 구간(0.8676~0.8887, 0.8474~0.8692)이 거의 접해 있으므로 이 0.02 가 분할 우연인지 선택 효과인지 단일 실행으로는 분리되지 않습니다. seed 를 바꾼 전체 루프 재실행(설정 변경이므로 루프 내부에서는 불가) 2~3회로, 보고할 성능 구간을 "테스트 0.858 ± 분할 산포" 형태로 확정할 수 있습니다.