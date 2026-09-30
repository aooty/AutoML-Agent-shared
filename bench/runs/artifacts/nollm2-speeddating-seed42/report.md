# AutoML 실행 보고서 — `nollm2-speeddating-seed42`

## 요약

목표 `balanced_accuracy >= 0.7712`를 **달성하지 못했습니다**. 최고 성능은 iteration 5의 `logreg`이며 balanced_accuracy=0.6998 (95% CI 0.6708~0.7293, 폭 0.0585)입니다. 총 5회 시도했고(최대 5회), 종료 사유는 **최대 반복 횟수 도달**입니다. 시간 예산 사용: 48초 / 3,600초 (1%).

재계획: critic이 4회 실행되어 그만큼 재계획했습니다 (시도 5회 — 마지막 시도는 평가 직후 루프가 끝나므로 진단 대상이 아닙니다).

목표 기준: auto 모드 — balanced_accuracy 0.7712 이상 ← 기준선 0.6685 + 남은 여유의 25%, 기준선 랭킹의 최적 컷을 넘도록 상향 (chance 0.5) — margin이 낸 바는 0.7514였지만, 같은 기준선의 랭킹을 최적 컷에서 자르면 0.7712 (KS 0.5423)입니다: 기준선을 다시 자른 것이 이미 넘는 점수는 목표가 아니므로 그 값으로 올렸습니다. 이 카드에서 --margin 0.309 이하는 모두 같은 바를 냅니다 — 더 어려운 목표를 원하면 그보다 크게 주십시오

최종 검증: 최종 테스트(20%, 반복 중 한 번도 쓰이지 않은 행): iteration 5의 logreg → balanced_accuracy=0.7158 (95% CI 0.6869~0.7496) (검증 0.6998 대비 -0.0159 — 이 차이가 선택 편향의 크기입니다, 다만 검증 점수가 위 CI 안에 있어 이 행들로는 0과 구분되지 않습니다)

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
| --- | --- | --- | --- | --- |
| 1 | `hist_gbdt` | `learning_rate`=0.1, `max_iter`=150 | balanced_accuracy=0.6586 | `overfitting` |
| 2 | `hist_gbdt` | `l2_regularization`=1.0, `learning_rate`=0.05, `max_depth`=4, `max_iter`=150 | balanced_accuracy=0.6455 | `hyperparam` |
| 3 | `hist_gbdt` | `class_weight`={'0': 1.0, '1': 1.5}, `l2_regularization`=1.0, `learning_rate`=0.03, `max_depth`=6, `max_iter`=400 | balanced_accuracy=0.6899 | `overfitting` |
| 4 | `hist_gbdt` | `class_weight`={'0': 1.0, '1': 1.5}, `l2_regularization`=1.0, `learning_rate`=0.05, `max_depth`=4, `max_iter`=400 | balanced_accuracy=0.6852 | `overfitting` |
| 5 | `logreg` | `class_weight`={'0': 1.0, '1': 1.5}, `max_iter`=400 | balanced_accuracy=0.6998 | — |

## 최고 성능 구성

- model: `logreg`
- hyperparams: `class_weight`={'0': 1.0, '1': 1.5}, `max_iter`=400
- preprocessing: `impute`=median, `missing_count`=False, `missing_indicator`=False, `scale`=True
- balanced_accuracy: 0.6998
- 전체 지표: `accuracy`=0.8484, `average_precision`=0.5496, `balanced_accuracy`=0.6998, `balanced_accuracy_at_best_cut`=0.7701, `balanced_accuracy_ci_high`=0.7293, `balanced_accuracy_ci_low`=0.6708, `brier`=0.1069, `calibration_error`=0.0388, `cut_headroom`=0.0703, `f1`=0.5097, `pr_auc`=0.5496, `precision`=0.5455, `recall`=0.4783, `roc_auc`=0.8504, `specificity`=0.9214, `train_accuracy`=0.8741, `train_balanced_accuracy`=0.7515, `train_f1`=0.5981, `train_val_gap`=0.0517
- 학습 시간: 3.508초
- 재현: `seed=42`, iteration 5

## 원인 분석

Critic 진단 분포: {'overfitting': 3, 'hyperparam': 1}. 가장 빈번한 원인은 `overfitting`이며, 모델/데이터 적합도 쪽 문제로 수렴했습니다.

마지막 Critic 방향: 정규화를 강화하고 용량을 줄인다: l2 증가, 깊이 축소, learning_rate 하향.

## 다음 단계 제안

- `--max-iterations`를 늘려 탐색 예산을 확대한다 (현재 예산 안에서는 balanced_accuracy 0.7712에 도달하지 못했다).
- 데이터셋 카드의 `n_rows`/`class_balance`가 실제와 일치하는지 확인한다 — 계획 품질은 카드 정확도에 종속된다.
- 목표 임계값이 이 데이터에서 달성 가능한 수준인지 베이스라인 대비 재검토한다.

---

*이 보고서는 실제 실행 결과입니다.*
