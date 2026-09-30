# AutoML 실행 보고서 — `nollm-bank-marketing-seed44`

## 요약

목표 `balanced_accuracy >= 0.7452`를 **달성**했습니다. 최고 성능은 iteration 2의 `hist_gbdt`이며 balanced_accuracy=0.7767 (95% CI 0.7607~0.7917, 폭 0.0310)입니다. 총 2회 시도했고(최대 5회), 종료 사유는 **목표 지표 달성**입니다.

목표 기준: auto 모드 — balanced_accuracy 0.7452 이상 ← 기준선 0.6603 + 남은 여유의 25% (chance 0.5)

최종 검증: 최종 테스트(20%, 반복 중 한 번도 쓰이지 않은 행): iteration 2의 hist_gbdt → balanced_accuracy=0.7694 (95% CI 0.7541~0.7824) (검증 0.7767 대비 +0.0073 — 이 차이가 선택 편향의 크기입니다, 다만 검증 점수가 위 CI 안에 있어 이 행들로는 0과 구분되지 않습니다)

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
| --- | --- | --- | --- | --- |
| 1 | `hist_gbdt` | `learning_rate`=0.1, `max_iter`=150 | balanced_accuracy=0.7278 | `hyperparam` |
| 2 | `hist_gbdt` | `class_weight`={'0': 1.0, '1': 1.5}, `learning_rate`=0.15, `max_depth`=12, `max_iter`=400 | balanced_accuracy=0.7767 | — |

## 최고 성능 구성

- model: `hist_gbdt`
- hyperparams: `class_weight`={'0': 1.0, '1': 1.5}, `learning_rate`=0.15, `max_depth`=12, `max_iter`=400
- preprocessing: `impute`=median, `missing_count`=False, `missing_indicator`=False, `scale`=False
- balanced_accuracy: 0.7767
- 전체 지표: `accuracy`=0.9069, `average_precision`=0.6138, `balanced_accuracy`=0.7767, `balanced_accuracy_at_best_cut`=0.8653, `balanced_accuracy_ci_high`=0.7917, `balanced_accuracy_ci_low`=0.7607, `brier`=0.0642, `calibration_error`=0.0205, `cut_headroom`=0.0886, `f1`=0.604, `pr_auc`=0.6138, `precision`=0.6011, `recall`=0.6068, `roc_auc`=0.9309, `specificity`=0.9466, `train_accuracy`=0.925, `train_balanced_accuracy`=0.8277, `train_f1`=0.686, `train_val_gap`=0.0509
- 학습 시간: 4.65초
- 재현: `seed=44`, iteration 2

## 원인 분석

Critic 진단 분포: {'hyperparam': 1}. 가장 빈번한 원인은 `hyperparam`이며, 모델/데이터 적합도 쪽 문제로 수렴했습니다.

마지막 Critic 방향: balanced_accuracy는 recall과 specificity의 평균이므로 둘이 같아지는 지점이 최적이다. recall(0.4915)가 specificity(0.9641)보다 낮으니 양성 클래스 가중치를 올린다: 1 → 1.5. train과 validation의 격차는 용량에 대한 증거이므로 가중치 방향의 근거가 되지 못한다.

## 다음 단계 제안

- 동일 구성으로 seed를 바꿔 3회 재현 실험해 분산을 확인한다.
- 홀드아웃 세트가 아닌 교차검증으로 성능을 재검증한다.
- 학습 시간과 지표의 트레이드오프를 보고 더 가벼운 구성으로 축소 가능한지 확인한다.

---

*이 보고서는 실제 실행 결과입니다.*
