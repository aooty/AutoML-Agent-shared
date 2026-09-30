# AutoML 실행 보고서 — `bar65nollm-adult-seed43`

## 요약

목표 `balanced_accuracy >= 0.9194`를 **달성하지 못했습니다**. 최고 성능은 iteration 3의 `hist_gbdt`이며 balanced_accuracy=0.8420 (95% CI 0.8328~0.8506, 폭 0.0178)입니다. 총 5회 시도했고(최대 5회), 종료 사유는 **최대 반복 횟수 도달**입니다. 시간 예산 사용: 58초 / 3,600초 (2%).

재계획: critic이 4회 실행되어 그만큼 재계획했습니다 (시도 5회 — 마지막 시도는 평가 직후 루프가 끝나므로 진단 대상이 아닙니다).

목표 기준: auto 모드 — balanced_accuracy 0.9194 이상 ← 기준선 0.7698 + 남은 여유의 65% (chance 0.5) — 이 바는 기준선 랭킹의 상한 0.8266를 넘습니다 (KS 0.6531). 이 랭킹으로는 어떤 임계값을 골라도 닿지 않으니 랭킹 자체를 올려야 합니다 — 모델 family나 특성. 크기로 말하면 이 바가 요구하는 KS가 0.8388이고 기준선은 0.6531이므로 랭킹이 0.1857만큼 올라야 합니다 — 계열 교체로 그만큼 움직인 기록이 있는지 먼저 보십시오. 지금 기준선에서 이 상한 안에 드는 margin은 0.246 이하입니다

최종 검증: 최종 테스트(20%, 반복 중 한 번도 쓰이지 않은 행): iteration 3의 hist_gbdt → balanced_accuracy=0.8476 (95% CI 0.8389~0.8555) (검증 0.8420 대비 -0.0056 — 이 차이가 선택 편향의 크기입니다, 다만 검증 점수가 위 CI 안에 있어 이 행들로는 0과 구분되지 않습니다)

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
| --- | --- | --- | --- | --- |
| 1 | `hist_gbdt` | `learning_rate`=0.1, `max_iter`=150 | balanced_accuracy=0.8017 | `hyperparam` |
| 2 | `hist_gbdt` | `class_weight`={'0': 1.0, '1': 1.5}, `learning_rate`=0.15, `max_depth`=12, `max_iter`=400 | balanced_accuracy=0.8260 | `hyperparam` |
| 3 | `hist_gbdt` | `class_weight`={'0': 1.0, '1': 2.25}, `learning_rate`=0.03, `max_depth`=6, `max_iter`=400 | balanced_accuracy=0.8420 | `underfitting` |
| 4 | `hist_gbdt` | `class_weight`={'0': 1.0, '1': 2.25}, `learning_rate`=0.08, `max_depth`=6, `max_iter`=1200, `max_leaf_nodes`=63 | balanced_accuracy=0.8409 | `underfitting` |
| 5 | `logreg` | `class_weight`={'0': 1.0, '1': 2.25}, `max_iter`=1200 | balanced_accuracy=0.8197 | — |

## 최고 성능 구성

- model: `hist_gbdt`
- hyperparams: `class_weight`={'0': 1.0, '1': 2.25}, `learning_rate`=0.03, `max_depth`=6, `max_iter`=400
- preprocessing: `impute`=median, `missing_count`=False, `missing_indicator`=False, `scale`=False
- balanced_accuracy: 0.8420
- 전체 지표: `accuracy`=0.8507, `average_precision`=0.8325, `balanced_accuracy`=0.842, `balanced_accuracy_at_best_cut`=0.8469, `balanced_accuracy_ci_high`=0.8506, `balanced_accuracy_ci_low`=0.8328, `brier`=0.0989, `calibration_error`=0.0791, `cut_headroom`=0.0049, `f1`=0.7256, `pr_auc`=0.8325, `precision`=0.6473, `recall`=0.8254, `roc_auc`=0.9289, `specificity`=0.8586, `train_accuracy`=0.8613, `train_balanced_accuracy`=0.8566, `train_f1`=0.7452, `train_val_gap`=0.0146
- 학습 시간: 12.754초
- 재현: `seed=43`, iteration 3

## 원인 분석

Critic 진단 분포: {'hyperparam': 2, 'underfitting': 2}. 가장 빈번한 원인은 `hyperparam`이며, 모델/데이터 적합도 쪽 문제로 수렴했습니다.

마지막 Critic 방향: 모델 용량과 학습량을 늘린다: 반복 수 증가, 트리 깊이·리프 수 확대, 정규화 완화.

## 다음 단계 제안

- `--max-iterations`를 늘려 탐색 예산을 확대한다 (현재 예산 안에서는 balanced_accuracy 0.9194에 도달하지 못했다).
- 데이터셋 카드의 `n_rows`/`class_balance`가 실제와 일치하는지 확인한다 — 계획 품질은 카드 정확도에 종속된다.
- 목표 임계값이 이 데이터에서 달성 가능한 수준인지 베이스라인 대비 재검토한다.

---

*이 보고서는 실제 실행 결과입니다.*
