# AutoML 실행 보고서 — `nollm2-jungle-chess-seed42`

## 요약

목표 `balanced_accuracy >= 0.633`를 **달성**했습니다. 최고 성능은 iteration 1의 `hist_gbdt`이며 balanced_accuracy=0.8248 (95% CI 0.8128~0.8365, 폭 0.0237)입니다. 총 1회 시도했고(최대 5회), 종료 사유는 **목표 지표 달성**입니다. 시간 예산 사용: 12초 / 3,600초 (0%).

재계획: critic이 한 번도 실행되지 않았습니다 (시도 1회, 진단 0회). 진단·재계획 경로는 이 결과에 기여하지 않았습니다 — 점수는 첫 계획 하나가 낸 것입니다. 그 경로가 도움이 된다는 증거도, 해가 된다는 증거도 이 실행에는 없습니다.

목표 기준: auto 모드 — balanced_accuracy 0.633 이상 ← 기준선 0.5106 + 남은 여유의 25% (chance 0.3333)

최종 검증: 최종 테스트(20%, 반복 중 한 번도 쓰이지 않은 행): iteration 1의 hist_gbdt → balanced_accuracy=0.8089 (95% CI 0.7962~0.8200) (검증 0.8248 대비 +0.0159 — 이 차이가 선택 편향의 크기입니다)

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
| --- | --- | --- | --- | --- |
| 1 | `hist_gbdt` | `learning_rate`=0.1, `max_iter`=150 | balanced_accuracy=0.8248 | — |

## 최고 성능 구성

- model: `hist_gbdt`
- hyperparams: `learning_rate`=0.1, `max_iter`=150
- preprocessing: `impute`=median, `missing_count`=False, `missing_indicator`=False, `scale`=False
- balanced_accuracy: 0.8248
- 전체 지표: `accuracy`=0.8739, `balanced_accuracy`=0.8248, `balanced_accuracy_ci_high`=0.8365, `balanced_accuracy_ci_low`=0.8128, `f1`=0.828, `precision`=0.8313, `recall`=0.8248, `train_accuracy`=0.9058, `train_balanced_accuracy`=0.871, `train_f1`=0.8765, `train_val_gap`=0.0462
- 학습 시간: 4.493초
- 재현: `seed=42`, iteration 1

## 원인 분석

Critic이 실행되지 않아 축적된 진단이 없습니다. 이 실행이 잰 것은 첫 계획의 품질이고, 진단·재계획 경로는 여기서 평가되지 않았습니다 — 이 결과를 그 경로의 근거로 인용할 수 없습니다.

## 다음 단계 제안

- 동일 구성으로 seed를 바꿔 3회 재현 실험해 분산을 확인한다.
- 홀드아웃 세트가 아닌 교차검증으로 성능을 재검증한다.
- 학습 시간과 지표의 트레이드오프를 보고 더 가벼운 구성으로 축소 가능한지 확인한다.

---

*이 보고서는 실제 실행 결과입니다.*
