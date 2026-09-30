# AutoML 실행 보고서 — `nollm2-bank-marketing-seed42`

## 요약

목표 `balanced_accuracy >= 0.84`를 **달성**했습니다. 최고 성능은 iteration 4의 `hist_gbdt`이며 balanced_accuracy=0.8564 (95% CI 0.8429~0.8667, 폭 0.0238)입니다. 총 4회 시도했고(최대 5회), 종료 사유는 **목표 지표 달성**입니다. 시간 예산 사용: 41초 / 3,600초 (1%).

재계획: critic이 3회 실행되어 그만큼 재계획했습니다 (시도 4회 — 마지막 시도는 평가 직후 루프가 끝나므로 진단 대상이 아닙니다).

목표 기준: auto 모드 — balanced_accuracy 0.84 이상 ← 기준선 0.662 + 남은 여유의 25%, 기준선 랭킹의 최적 컷을 넘도록 상향 (chance 0.5) — margin이 낸 바는 0.7465였지만, 같은 기준선의 랭킹을 최적 컷에서 자르면 0.84 (KS 0.6801)입니다: 기준선을 다시 자른 것이 이미 넘는 점수는 목표가 아니므로 그 값으로 올렸습니다. 이 카드에서 --margin 0.526 이하는 모두 같은 바를 냅니다 — 더 어려운 목표를 원하면 그보다 크게 주십시오

최종 검증: 최종 테스트(20%, 반복 중 한 번도 쓰이지 않은 행): iteration 4의 hist_gbdt → balanced_accuracy=0.8477 (95% CI 0.8351~0.8599) (검증 0.8564 대비 +0.0087 — 이 차이가 선택 편향의 크기입니다, 다만 검증 점수가 위 CI 안에 있어 이 행들로는 0과 구분되지 않습니다)

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
| --- | --- | --- | --- | --- |
| 1 | `hist_gbdt` | `learning_rate`=0.1, `max_iter`=150 | balanced_accuracy=0.7316 | `hyperparam` |
| 2 | `hist_gbdt` | `class_weight`={'0': 1.0, '1': 1.5}, `learning_rate`=0.15, `max_depth`=12, `max_iter`=400 | balanced_accuracy=0.7833 | `hyperparam` |
| 3 | `hist_gbdt` | `class_weight`={'0': 1.0, '1': 2.25}, `learning_rate`=0.03, `max_depth`=6, `max_iter`=400 | balanced_accuracy=0.8162 | `hyperparam` |
| 4 | `hist_gbdt` | `class_weight`={'0': 1.0, '1': 3.375}, `learning_rate`=0.05, `max_depth`=8, `max_iter`=400 | balanced_accuracy=0.8564 | — |

## 최고 성능 구성

- model: `hist_gbdt`
- hyperparams: `class_weight`={'0': 1.0, '1': 3.375}, `learning_rate`=0.05, `max_depth`=8, `max_iter`=400
- preprocessing: `impute`=median, `missing_count`=False, `missing_indicator`=False, `scale`=False
- balanced_accuracy: 0.8564
- 전체 지표: `accuracy`=0.8956, `average_precision`=0.6476, `balanced_accuracy`=0.8564, `balanced_accuracy_at_best_cut`=0.8849, `balanced_accuracy_ci_high`=0.8667, `balanced_accuracy_ci_low`=0.8429, `brier`=0.0718, `calibration_error`=0.0738, `cut_headroom`=0.0285, `f1`=0.6435, `pr_auc`=0.6476, `precision`=0.5358, `recall`=0.8053, `roc_auc`=0.9416, `specificity`=0.9076, `train_accuracy`=0.9117, `train_balanced_accuracy`=0.8937, `train_f1`=0.6974, `train_val_gap`=0.0372
- 학습 시간: 6.901초
- 재현: `seed=42`, iteration 4

## 원인 분석

Critic 진단 분포: {'hyperparam': 3}. 가장 빈번한 원인은 `hyperparam`이며, 모델/데이터 적합도 쪽 문제로 수렴했습니다.

마지막 Critic 방향: balanced_accuracy는 recall과 specificity의 평균이므로 둘이 같아지는 지점이 최적이다. recall(0.7013)가 specificity(0.9311)보다 낮으니 양성 클래스 가중치를 올린다: 2.25 → 3.375. train과 validation의 격차는 용량에 대한 증거이므로 가중치 방향의 근거가 되지 못한다.

## 다음 단계 제안

- 동일 구성으로 seed를 바꿔 3회 재현 실험해 분산을 확인한다.
- 홀드아웃 세트가 아닌 교차검증으로 성능을 재검증한다.
- 학습 시간과 지표의 트레이드오프를 보고 더 가벼운 구성으로 축소 가능한지 확인한다.

---

*이 보고서는 실제 실행 결과입니다.*
