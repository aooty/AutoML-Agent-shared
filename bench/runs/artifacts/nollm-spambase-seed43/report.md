# AutoML 실행 보고서 — `nollm-spambase-seed43`

## 요약

목표 `balanced_accuracy >= 0.938`를 **달성**했습니다. 최고 성능은 iteration 1의 `hist_gbdt`이며 balanced_accuracy=0.9575 (95% CI 0.9429~0.9699, 폭 0.0270)입니다. 총 1회 시도했고(최대 5회), 종료 사유는 **목표 지표 달성**입니다.

목표 기준: auto 모드 — balanced_accuracy 0.938 이상 ← 기준선 0.9173 + 남은 여유의 25% (chance 0.5) — 이 바는 기준선 랭킹의 상한 0.9304를 넘습니다 (KS 0.8609). 이 랭킹으로는 어떤 임계값을 골라도 닿지 않으니 랭킹 자체를 올려야 합니다 — 모델 family나 특성. 크기로 말하면 이 바가 요구하는 KS가 0.876이고 기준선은 0.8609이므로 랭킹이 0.0151만큼 올라야 합니다 — 계열 교체로 그만큼 움직인 기록이 있는지 먼저 보십시오. 지금 기준선에서 이 상한 안에 드는 margin은 0.158 이하입니다

최종 검증: 최종 테스트(20%, 반복 중 한 번도 쓰이지 않은 행): iteration 1의 hist_gbdt → balanced_accuracy=0.9527 (95% CI 0.9385~0.9656) (검증 0.9575 대비 +0.0048 — 이 차이가 선택 편향의 크기입니다, 다만 검증 점수가 위 CI 안에 있어 이 행들로는 0과 구분되지 않습니다)

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
| --- | --- | --- | --- | --- |
| 1 | `hist_gbdt` | `learning_rate`=0.1, `max_iter`=150 | balanced_accuracy=0.9575 | — |

## 최고 성능 구성

- model: `hist_gbdt`
- hyperparams: `learning_rate`=0.1, `max_iter`=150
- preprocessing: `impute`=median, `missing_count`=False, `missing_indicator`=False, `scale`=False
- balanced_accuracy: 0.9575
- 전체 지표: `accuracy`=0.962, `average_precision`=0.9767, `balanced_accuracy`=0.9575, `balanced_accuracy_at_best_cut`=0.965, `balanced_accuracy_ci_high`=0.9699, `balanced_accuracy_ci_low`=0.9429, `brier`=0.0317, `calibration_error`=0.0259, `cut_headroom`=0.0075, `f1`=0.9509, `pr_auc`=0.9767, `precision`=0.9658, `recall`=0.9365, `roc_auc`=0.9861, `specificity`=0.9785, `train_accuracy`=1.0, `train_balanced_accuracy`=1.0, `train_f1`=1.0, `train_val_gap`=0.0425
- 학습 시간: 2.696초
- 재현: `seed=43`, iteration 1

## 원인 분석

Critic이 개입하기 전에 종료되어 축적된 진단이 없습니다.

## 다음 단계 제안

- 동일 구성으로 seed를 바꿔 3회 재현 실험해 분산을 확인한다.
- 홀드아웃 세트가 아닌 교차검증으로 성능을 재검증한다.
- 학습 시간과 지표의 트레이드오프를 보고 더 가벼운 구성으로 축소 가능한지 확인한다.

---

*이 보고서는 실제 실행 결과입니다.*
