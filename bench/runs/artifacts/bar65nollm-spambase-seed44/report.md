# AutoML 실행 보고서 — `bar65nollm-spambase-seed44`

## 요약

목표 `balanced_accuracy >= 0.967`를 **달성하지 못했습니다**. 최고 성능은 iteration 2의 `hist_gbdt`이며 balanced_accuracy=0.9361 (95% CI 0.9179~0.9526, 폭 0.0348)입니다. 총 4회 시도했고(최대 5회), 종료 사유는 **연속 미개선(정체)으로 조기 종료**입니다. 시간 예산 사용: 31초 / 3,600초 (1%).

재계획: critic이 3회 실행되어 그만큼 재계획했습니다 (시도 4회 — 마지막 시도는 평가 직후 루프가 끝나므로 진단 대상이 아닙니다).

목표 기준: auto 모드 — balanced_accuracy 0.967 이상 ← 기준선 0.9058 + 남은 여유의 65% (chance 0.5) — 이 바는 기준선 랭킹의 상한 0.9233를 넘습니다 (KS 0.8466). 이 랭킹으로는 어떤 임계값을 골라도 닿지 않으니 랭킹 자체를 올려야 합니다 — 모델 family나 특성. 크기로 말하면 이 바가 요구하는 KS가 0.934이고 기준선은 0.8466이므로 랭킹이 0.0874만큼 올라야 합니다 — 계열 교체로 그만큼 움직인 기록이 있는지 먼저 보십시오. 지금 기준선에서 이 상한 안에 드는 margin은 0.185 이하입니다

최종 검증: 최종 테스트(20%, 반복 중 한 번도 쓰이지 않은 행): iteration 2의 hist_gbdt → balanced_accuracy=0.9599 (95% CI 0.9455~0.9740) (검증 0.9361 대비 -0.0238 — 이 차이가 선택 편향의 크기입니다)

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
| --- | --- | --- | --- | --- |
| 1 | `hist_gbdt` | `learning_rate`=0.1, `max_iter`=150 | balanced_accuracy=0.9347 | `hyperparam` |
| 2 | `hist_gbdt` | `learning_rate`=0.05, `max_depth`=8, `max_iter`=400 | balanced_accuracy=0.9361 | `hyperparam` |
| 3 | `hist_gbdt` | `learning_rate`=0.02, `max_depth`=5, `max_iter`=800 | balanced_accuracy=0.9329 | `wrong_model_family` |
| 4 | `random_forest` | `n_estimators`=300 | balanced_accuracy=0.9329 | — |

## 최고 성능 구성

- model: `hist_gbdt`
- hyperparams: `learning_rate`=0.05, `max_depth`=8, `max_iter`=400
- preprocessing: `impute`=median, `missing_count`=False, `missing_indicator`=False, `scale`=False
- balanced_accuracy: 0.9361
- 전체 지표: `accuracy`=0.9413, `average_precision`=0.9684, `balanced_accuracy`=0.9361, `balanced_accuracy_at_best_cut`=0.9416, `balanced_accuracy_ci_high`=0.9526, `balanced_accuracy_ci_low`=0.9179, `brier`=0.0497, `calibration_error`=0.0412, `cut_headroom`=0.0055, `f1`=0.9244, `pr_auc`=0.9684, `precision`=0.9375, `recall`=0.9116, `roc_auc`=0.9833, `specificity`=0.9606, `train_accuracy`=0.9989, `train_balanced_accuracy`=0.9988, `train_f1`=0.9986, `train_val_gap`=0.0627
- 학습 시간: 4.321초
- 재현: `seed=44`, iteration 2

## 원인 분석

Critic 진단 분포: {'hyperparam': 2, 'wrong_model_family': 1}. 가장 빈번한 원인은 `hyperparam`이며, 모델/데이터 적합도 쪽 문제로 수렴했습니다.

마지막 Critic 방향: 모델 계열 자체를 바꾼다.

## 다음 단계 제안

- `--max-iterations`를 늘려 탐색 예산을 확대한다 (현재 예산 안에서는 balanced_accuracy 0.967에 도달하지 못했다).
- 동일 계열 안에서의 미세 조정이 정체됐으므로 특성 공학 또는 데이터 품질 개선을 먼저 시도한다.
- 데이터셋 카드의 `n_rows`/`class_balance`가 실제와 일치하는지 확인한다 — 계획 품질은 카드 정확도에 종속된다.
- 목표 임계값이 이 데이터에서 달성 가능한 수준인지 베이스라인 대비 재검토한다.

---

*이 보고서는 실제 실행 결과입니다.*
