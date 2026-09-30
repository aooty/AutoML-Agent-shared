# AutoML 실행 보고서 — `nollm-spambase-seed44`

## 요약

목표 `balanced_accuracy >= 0.9294`를 **달성**했습니다. 최고 성능은 iteration 1의 `hist_gbdt`이며 balanced_accuracy=0.9347 (95% CI 0.9169~0.9494, 폭 0.0325)입니다. 다만 목표 0.9294가 이 구간 안에 있어, 이 검증 슬라이스는 달성 여부를 판정할 만큼 두 값을 구분하지 못합니다 — 아래 최종 검증 점수를 보십시오. 총 1회 시도했고(최대 5회), 종료 사유는 **목표 지표 달성**입니다.

목표 기준: auto 모드 — balanced_accuracy 0.9294 이상 ← 기준선 0.9058 + 남은 여유의 25% (chance 0.5) — 이 바는 기준선 랭킹의 상한 0.9233를 넘습니다 (KS 0.8466). 이 랭킹으로는 어떤 임계값을 골라도 닿지 않으니 랭킹 자체를 올려야 합니다 — 모델 family나 특성. 크기로 말하면 이 바가 요구하는 KS가 0.8588이고 기준선은 0.8466이므로 랭킹이 0.0122만큼 올라야 합니다 — 계열 교체로 그만큼 움직인 기록이 있는지 먼저 보십시오. 지금 기준선에서 이 상한 안에 드는 margin은 0.185 이하입니다

최종 검증: 최종 테스트(20%, 반복 중 한 번도 쓰이지 않은 행): iteration 1의 hist_gbdt → balanced_accuracy=0.9490 (95% CI 0.9346~0.9631) (검증 0.9347 대비 -0.0143 — 이 차이가 선택 편향의 크기입니다, 다만 검증 점수가 위 CI 안에 있어 이 행들로는 0과 구분되지 않습니다)

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
| --- | --- | --- | --- | --- |
| 1 | `hist_gbdt` | `learning_rate`=0.1, `max_iter`=150 | balanced_accuracy=0.9347 | — |

## 최고 성능 구성

- model: `hist_gbdt`
- hyperparams: `learning_rate`=0.1, `max_iter`=150
- preprocessing: `impute`=median, `missing_count`=False, `missing_indicator`=False, `scale`=False
- balanced_accuracy: 0.9347
- 전체 지표: `accuracy`=0.9402, `average_precision`=0.9675, `balanced_accuracy`=0.9347, `balanced_accuracy_at_best_cut`=0.9434, `balanced_accuracy_ci_high`=0.9494, `balanced_accuracy_ci_low`=0.9169, `brier`=0.0501, `calibration_error`=0.0433, `cut_headroom`=0.0087, `f1`=0.9229, `pr_auc`=0.9675, `precision`=0.9373, `recall`=0.9088, `roc_auc`=0.9823, `specificity`=0.9606, `train_accuracy`=0.9993, `train_balanced_accuracy`=0.9992, `train_f1`=0.9991, `train_val_gap`=0.0645
- 학습 시간: 2.676초
- 재현: `seed=44`, iteration 1

## 원인 분석

Critic이 개입하기 전에 종료되어 축적된 진단이 없습니다.

## 다음 단계 제안

- 동일 구성으로 seed를 바꿔 3회 재현 실험해 분산을 확인한다.
- 홀드아웃 세트가 아닌 교차검증으로 성능을 재검증한다.
- 학습 시간과 지표의 트레이드오프를 보고 더 가벼운 구성으로 축소 가능한지 확인한다.

---

*이 보고서는 실제 실행 결과입니다.*
