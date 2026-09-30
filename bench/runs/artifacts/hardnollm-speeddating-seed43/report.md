# AutoML 실행 보고서 — `hardnollm-speeddating-seed43`

## 요약

목표 `balanced_accuracy >= 0.8399`를 **달성하지 못했습니다**. 최고 성능은 iteration 4의 `hist_gbdt`이며 balanced_accuracy=0.7327 (95% CI 0.7069~0.7649, 폭 0.0580)입니다. 총 5회 시도했고(최대 5회), 종료 사유는 **최대 반복 횟수 도달**입니다.

재계획: critic이 4회 실행되어 그만큼 재계획했습니다 (시도 5회 — 마지막 시도는 평가 직후 루프가 끝나므로 진단 대상이 아닙니다).

목표 기준: auto 모드 — balanced_accuracy 0.8399 이상 ← 기준선 0.6798 + 남은 여유의 50% (chance 0.5) — 이 바는 기준선 랭킹의 상한 0.7558를 넘습니다 (KS 0.5116). 이 랭킹으로는 어떤 임계값을 골라도 닿지 않으니 랭킹 자체를 올려야 합니다 — 모델 family나 특성. 크기로 말하면 이 바가 요구하는 KS가 0.6798이고 기준선은 0.5116이므로 랭킹이 0.1682만큼 올라야 합니다 — 계열 교체로 그만큼 움직인 기록이 있는지 먼저 보십시오. 지금 기준선에서 이 상한 안에 드는 margin은 0.237 이하입니다

최종 검증: 최종 테스트(20%, 반복 중 한 번도 쓰이지 않은 행): iteration 4의 hist_gbdt → balanced_accuracy=0.7163 (95% CI 0.6834~0.7475) (검증 0.7327 대비 +0.0163 — 이 차이가 선택 편향의 크기입니다, 다만 검증 점수가 위 CI 안에 있어 이 행들로는 0과 구분되지 않습니다)

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
| --- | --- | --- | --- | --- |
| 1 | `hist_gbdt` | `learning_rate`=0.1, `max_iter`=150 | balanced_accuracy=0.6960 | `overfitting` |
| 2 | `hist_gbdt` | `l2_regularization`=1.0, `learning_rate`=0.05, `max_depth`=4, `max_iter`=150 | balanced_accuracy=0.6633 | `hyperparam` |
| 3 | `hist_gbdt` | `class_weight`={'0': 1.0, '1': 1.5}, `l2_regularization`=1.0, `learning_rate`=0.03, `max_depth`=6, `max_iter`=400 | balanced_accuracy=0.7102 | `overfitting` |
| 4 | `hist_gbdt` | `class_weight`={'0': 1.0, '1': 1.5}, `l2_regularization`=1.0, `learning_rate`=0.05, `max_depth`=4, `max_iter`=400 | balanced_accuracy=0.7327 | `overfitting` |
| 5 | `logreg` | `class_weight`={'0': 1.0, '1': 1.5}, `max_iter`=400 | balanced_accuracy=0.7068 | — |

## 최고 성능 구성

- model: `hist_gbdt`
- hyperparams: `class_weight`={'0': 1.0, '1': 1.5}, `l2_regularization`=1.0, `learning_rate`=0.05, `max_depth`=4, `max_iter`=400
- preprocessing: `impute`=median, `missing_count`=False, `missing_indicator`=False, `scale`=False
- balanced_accuracy: 0.7327
- 전체 지표: `accuracy`=0.8741, `average_precision`=0.6134, `balanced_accuracy`=0.7327, `balanced_accuracy_at_best_cut`=0.7905, `balanced_accuracy_ci_high`=0.7649, `balanced_accuracy_ci_low`=0.7069, `brier`=0.0957, `calibration_error`=0.0334, `cut_headroom`=0.0578, `f1`=0.5772, `pr_auc`=0.6134, `precision`=0.6457, `recall`=0.5217, `roc_auc`=0.8632, `specificity`=0.9436, `train_accuracy`=0.9618, `train_balanced_accuracy`=0.9209, `train_f1`=0.8812, `train_val_gap`=0.1882
- 학습 시간: 6.768초
- 재현: `seed=43`, iteration 4

## 원인 분석

Critic 진단 분포: {'overfitting': 3, 'hyperparam': 1}. 가장 빈번한 원인은 `overfitting`이며, 모델/데이터 적합도 쪽 문제로 수렴했습니다.

마지막 Critic 방향: 정규화를 강화하고 용량을 줄인다: l2 증가, 깊이 축소, learning_rate 하향.

## 다음 단계 제안

- `--max-iterations`를 늘려 탐색 예산을 확대한다 (현재 예산 안에서는 balanced_accuracy 0.8399에 도달하지 못했다).
- 데이터셋 카드의 `n_rows`/`class_balance`가 실제와 일치하는지 확인한다 — 계획 품질은 카드 정확도에 종속된다.
- 목표 임계값이 이 데이터에서 달성 가능한 수준인지 베이스라인 대비 재검토한다.

---

*이 보고서는 실제 실행 결과입니다.*
