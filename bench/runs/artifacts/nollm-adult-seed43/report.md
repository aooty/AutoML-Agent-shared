# AutoML 실행 보고서 — `nollm-adult-seed43`

## 요약

목표 `balanced_accuracy >= 0.8274`를 **달성**했습니다. 최고 성능은 iteration 3의 `hist_gbdt`이며 balanced_accuracy=0.8420 (95% CI 0.8328~0.8506, 폭 0.0178)입니다. 총 3회 시도했고(최대 5회), 종료 사유는 **목표 지표 달성**입니다.

목표 기준: auto 모드 — balanced_accuracy 0.8274 이상 ← 기준선 0.7698 + 남은 여유의 25% (chance 0.5) — 이 바는 기준선 랭킹의 상한 0.8266를 넘습니다 (KS 0.6531). 이 랭킹으로는 어떤 임계값을 골라도 닿지 않으니 랭킹 자체를 올려야 합니다 — 모델 family나 특성. 크기로 말하면 이 바가 요구하는 KS가 0.6548이고 기준선은 0.6531이므로 랭킹이 0.0017만큼 올라야 합니다 — 계열 교체로 그만큼 움직인 기록이 있는지 먼저 보십시오. 지금 기준선에서 이 상한 안에 드는 margin은 0.246 이하입니다

최종 검증: 최종 테스트(20%, 반복 중 한 번도 쓰이지 않은 행): iteration 3의 hist_gbdt → balanced_accuracy=0.8476 (95% CI 0.8389~0.8555) (검증 0.8420 대비 -0.0056 — 이 차이가 선택 편향의 크기입니다, 다만 검증 점수가 위 CI 안에 있어 이 행들로는 0과 구분되지 않습니다)

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
| --- | --- | --- | --- | --- |
| 1 | `hist_gbdt` | `learning_rate`=0.1, `max_iter`=150 | balanced_accuracy=0.8017 | `hyperparam` |
| 2 | `hist_gbdt` | `class_weight`={'0': 1.0, '1': 1.5}, `learning_rate`=0.15, `max_depth`=12, `max_iter`=400 | balanced_accuracy=0.8260 | `hyperparam` |
| 3 | `hist_gbdt` | `class_weight`={'0': 1.0, '1': 2.25}, `learning_rate`=0.03, `max_depth`=6, `max_iter`=400 | balanced_accuracy=0.8420 | — |

## 최고 성능 구성

- model: `hist_gbdt`
- hyperparams: `class_weight`={'0': 1.0, '1': 2.25}, `learning_rate`=0.03, `max_depth`=6, `max_iter`=400
- preprocessing: `impute`=median, `missing_count`=False, `missing_indicator`=False, `scale`=False
- balanced_accuracy: 0.8420
- 전체 지표: `accuracy`=0.8507, `average_precision`=0.8325, `balanced_accuracy`=0.842, `balanced_accuracy_at_best_cut`=0.8469, `balanced_accuracy_ci_high`=0.8506, `balanced_accuracy_ci_low`=0.8328, `brier`=0.0989, `calibration_error`=0.0791, `cut_headroom`=0.0049, `f1`=0.7256, `pr_auc`=0.8325, `precision`=0.6473, `recall`=0.8254, `roc_auc`=0.9289, `specificity`=0.8586, `train_accuracy`=0.8613, `train_balanced_accuracy`=0.8566, `train_f1`=0.7452, `train_val_gap`=0.0146
- 학습 시간: 16.145초
- 재현: `seed=43`, iteration 3

## 원인 분석

Critic 진단 분포: {'hyperparam': 2}. 가장 빈번한 원인은 `hyperparam`이며, 모델/데이터 적합도 쪽 문제로 수렴했습니다.

마지막 Critic 방향: balanced_accuracy는 recall과 specificity의 평균이므로 둘이 같아지는 지점이 최적이다. recall(0.7437)가 specificity(0.9084)보다 낮으니 양성 클래스 가중치를 올린다: 1.5 → 2.25. train과 validation의 격차는 용량에 대한 증거이므로 가중치 방향의 근거가 되지 못한다.

## 다음 단계 제안

- 동일 구성으로 seed를 바꿔 3회 재현 실험해 분산을 확인한다.
- 홀드아웃 세트가 아닌 교차검증으로 성능을 재검증한다.
- 학습 시간과 지표의 트레이드오프를 보고 더 가벼운 구성으로 축소 가능한지 확인한다.

---

*이 보고서는 실제 실행 결과입니다.*
