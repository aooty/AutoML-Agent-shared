# AutoML 실행 보고서 — `nollm-speeddating-seed44`

## 요약

목표 `balanced_accuracy >= 0.7605`를 **달성하지 못했습니다**. 최고 성능은 iteration 4의 `hist_gbdt`이며 balanced_accuracy=0.7080 (95% CI 0.6784~0.7376, 폭 0.0592)입니다. 총 5회 시도했고(최대 5회), 종료 사유는 **최대 반복 횟수 도달**입니다.

목표 기준: auto 모드 — balanced_accuracy 0.7605 이상 ← 기준선 0.6807 + 남은 여유의 25% (chance 0.5)

최종 검증: 최종 테스트(20%, 반복 중 한 번도 쓰이지 않은 행): iteration 4의 hist_gbdt → balanced_accuracy=0.7174 (95% CI 0.6858~0.7468) (검증 0.7080 대비 -0.0094 — 이 차이가 선택 편향의 크기입니다, 다만 검증 점수가 위 CI 안에 있어 이 행들로는 0과 구분되지 않습니다)

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
| --- | --- | --- | --- | --- |
| 1 | `hist_gbdt` | `learning_rate`=0.1, `max_iter`=150 | balanced_accuracy=0.6583 | `overfitting` |
| 2 | `hist_gbdt` | `l2_regularization`=1.0, `learning_rate`=0.05, `max_depth`=4, `max_iter`=150 | balanced_accuracy=0.6608 | `hyperparam` |
| 3 | `hist_gbdt` | `class_weight`={'0': 1.0, '1': 1.5}, `l2_regularization`=1.0, `learning_rate`=0.03, `max_depth`=6, `max_iter`=400 | balanced_accuracy=0.6866 | `overfitting` |
| 4 | `hist_gbdt` | `class_weight`={'0': 1.0, '1': 1.5}, `l2_regularization`=1.0, `learning_rate`=0.05, `max_depth`=4, `max_iter`=400 | balanced_accuracy=0.7080 | `overfitting` |
| 5 | `logreg` | `class_weight`={'0': 1.0, '1': 1.5}, `max_iter`=400 | balanced_accuracy=0.7048 | — |

## 최고 성능 구성

- model: `hist_gbdt`
- hyperparams: `class_weight`={'0': 1.0, '1': 1.5}, `l2_regularization`=1.0, `learning_rate`=0.05, `max_depth`=4, `max_iter`=400
- preprocessing: `impute`=median, `missing_count`=False, `missing_indicator`=False, `scale`=False
- balanced_accuracy: 0.7080
- 전체 지표: `accuracy`=0.8669, `average_precision`=0.6011, `balanced_accuracy`=0.708, `balanced_accuracy_at_best_cut`=0.7842, `balanced_accuracy_ci_high`=0.7376, `balanced_accuracy_ci_low`=0.6784, `brier`=0.0961, `calibration_error`=0.0244, `cut_headroom`=0.0762, `f1`=0.5383, `pr_auc`=0.6011, `precision`=0.628, `recall`=0.471, `roc_auc`=0.8603, `specificity`=0.945, `train_accuracy`=0.9568, `train_balanced_accuracy`=0.915, `train_f1`=0.8668, `train_val_gap`=0.207
- 학습 시간: 6.355초
- 재현: `seed=44`, iteration 4

## 원인 분석

Critic 진단 분포: {'overfitting': 3, 'hyperparam': 1}. 가장 빈번한 원인은 `overfitting`이며, 모델/데이터 적합도 쪽 문제로 수렴했습니다.

마지막 Critic 방향: 정규화를 강화하고 용량을 줄인다: l2 증가, 깊이 축소, learning_rate 하향.

## 다음 단계 제안

- `--max-iterations`를 늘려 탐색 예산을 확대한다 (현재 예산 안에서는 balanced_accuracy 0.7605에 도달하지 못했다).
- 데이터셋 카드의 `n_rows`/`class_balance`가 실제와 일치하는지 확인한다 — 계획 품질은 카드 정확도에 종속된다.
- 목표 임계값이 이 데이터에서 달성 가능한 수준인지 베이스라인 대비 재검토한다.

---

*이 보고서는 실제 실행 결과입니다.*
