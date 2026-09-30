# AutoML 실행 보고서 — `nollm-house_sales-seed44`

## 요약

목표 `r2 >= 0.785`를 **달성**했습니다. 최고 성능은 iteration 1의 `hist_gbdt`이며 r2=0.9013 (95% CI 0.8882~0.9125, 폭 0.0244)입니다. 총 1회 시도했고(최대 5회), 종료 사유는 **목표 지표 달성**입니다.

목표 기준: auto 모드 — r2 0.785 이상 ← 기준선 0.7133 + 남은 여유의 25% (chance -0.0)

최종 검증: 최종 테스트(20%, 반복 중 한 번도 쓰이지 않은 행): iteration 1의 hist_gbdt → r2=0.8953 (95% CI 0.8810~0.9068) (검증 0.9013 대비 +0.0060 — 이 차이가 선택 편향의 크기입니다, 다만 검증 점수가 위 CI 안에 있어 이 행들로는 0과 구분되지 않습니다)

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
| --- | --- | --- | --- | --- |
| 1 | `hist_gbdt` | `learning_rate`=0.1, `max_iter`=150 | r2=0.9013 | — |

## 최고 성능 구성

- model: `hist_gbdt`
- hyperparams: `learning_rate`=0.1, `max_iter`=150
- preprocessing: `impute`=median, `missing_count`=False, `missing_indicator`=False, `scale`=False
- r2: 0.9013
- 전체 지표: `mae`=64964.1769, `mean_absolute_error`=64964.1769, `r2`=0.9013, `r2_ci_high`=0.9125, `r2_ci_low`=0.8882, `r2_score`=0.9013, `rmse`=113928.1886, `root_mean_squared_error`=113928.1886, `train_mae`=48679.4288, `train_r2`=0.9372, `train_val_gap`=0.0358
- 학습 시간: 2.362초
- 재현: `seed=44`, iteration 1

## 원인 분석

Critic이 개입하기 전에 종료되어 축적된 진단이 없습니다.

## 다음 단계 제안

- 동일 구성으로 seed를 바꿔 3회 재현 실험해 분산을 확인한다.
- 홀드아웃 세트가 아닌 교차검증으로 성능을 재검증한다.
- 학습 시간과 지표의 트레이드오프를 보고 더 가벼운 구성으로 축소 가능한지 확인한다.

---

*이 보고서는 실제 실행 결과입니다.*
