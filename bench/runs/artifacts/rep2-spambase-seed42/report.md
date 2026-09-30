# AutoML 실행 보고서 — `spambase` (binary_classification, 목표: `balanced_accuracy` 최대화)

## 요약

목표는 달성되었습니다. 임계값 `balanced_accuracy` 0.9421에 대해, 1회차 시도의 `hist_gbdt`가 검증 슬라이스에서 0.9553 (95% CI 0.9403~0.9676)을 기록했고, 한 번도 사용되지 않은 최종 테스트 20%에서 0.9473 (95% CI 0.9305~0.9611)을 기록했습니다. 5회 예산 중 1회만 사용하고 첫 계획에서 바로 기준선(`logreg` 0.9228)과 임계값을 모두 넘겨 루프가 종료되었습니다. 따라서 이 실행에는 critic 진단이 한 건도 존재하지 않으며, 점수는 전적으로 첫 계획 하나의 결과입니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=500`, `learning_rate=0.06`, `max_depth=6`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `early_stopping=False` | status `ok`, 검증 `balanced_accuracy`=0.9552763421057843 (CI 0.9403~0.9676), `roc_auc`=0.9931, `train_time_sec`=3.144 → 임계값 0.9421 초과, 루프 종료 | 없음 (목표 달성으로 critic 미실행) |

`dropped_hyperparams`는 비어 있습니다 — 제안한 모든 키가 estimator에 그대로 적용되었습니다. `unsupported_claims`도 비어 있습니다.

## 최고 성능 구성

**iteration 1 / `hist_gbdt`** (실제 적용된 값 기준)

```
model: hist_gbdt
hyperparams:
  max_iter: 500
  learning_rate: 0.06
  max_depth: 6
  max_leaf_nodes: 31
  l2_regularization: 1.0
  early_stopping: false
preprocessing (applied):
  impute: median
  scale: false
  missing_indicator: false
  missing_count: false
protocol: stratified 60/20/20, seed 42 (카드 baseline과 동일 분할)
```

| 지표 | 검증(20%) | 최종 테스트(20%, 결정에 한 번도 사용되지 않은 행) |
|---|---|---|
| `balanced_accuracy` | **0.9553** (95% CI 0.9403~0.9676) | **0.9473** (95% CI 0.9305~0.9611) |

검증과 테스트의 차이는 +0.0080이며, 이것이 **선택 편향의 크기**입니다 — 루프는 검증 숫자만 보고 최종 구성을 골랐기 때문입니다. 다만 검증 점수 0.9553이 테스트 CI(0.9305~0.9611) 안에 들어 있으므로, 이 행들만으로는 그 차이를 0과 구분할 수 없습니다. 이 실행이 실제로 입증한 값은 테스트의 0.9473이며, 이는 임계값 0.9421을 0.0052 상회합니다.

검증 슬라이스의 나머지 지표(모두 검증 기준): `f1`=0.9471, `accuracy`=0.9587, `precision`=0.9551, `recall`=0.9392, `specificity`=0.971326, `roc_auc`=0.9931, `pr_auc`/`average_precision`=0.9901, `balanced_accuracy_at_best_cut`=0.9606, `cut_headroom`=0.005324, `brier`=0.032241, `calibration_error`=0.018949, `train_balanced_accuracy`=0.9987818392907402, `train_val_gap`=0.043505. 학습 시간 3.144초 (제한 600초).

## 원인 분석

이 실행에는 진단이 존재하지 않습니다. `재계획` 라인이 명시하듯 critic은 한 번도 실행되지 않았고(시도 1회, 진단 0회), 첫 시도가 임계값을 넘겨 루프가 즉시 종료되었습니다. 따라서 **verdict들 사이의 패턴이라고 부를 것이 없으며**, 이 점수는 첫 계획 하나가 낸 것입니다. 진단·재계획 경로가 도움이 된다는 증거도, 해가 된다는 증거도 이 실행에는 없습니다.

시도가 하나뿐이므로 시도 간 비교도 성립하지 않습니다. 무엇이 성능을 "제한했는지"에 대해 이 실행이 근거를 가지고 말할 수 있는 것은 사실 관계 두 가지뿐입니다.

- 기준선(`logreg`, 검증 `balanced_accuracy` 0.9228, CI 0.9030~0.9410)과 최고 구성(0.9553, CI 0.9403~0.9676)의 CI는 거의 겹치지 않으며, 차이 0.0325는 두 구간 폭보다 크지 않다고 보기 어려운 수준입니다. 목표 산출 근거에서 기준선의 `balanced_accuracy_at_best_cut`(= ranking_ceiling) 0.9415가 임계값 0.9421보다 낮았다는 점, 즉 선형 기준선의 랭킹 자체로는 어떤 컷을 골라도 임계값에 닿지 못했다는 점이 계획의 출발점이었고, `hist_gbdt`의 `roc_auc` 0.9931 / `balanced_accuracy_at_best_cut` 0.9606은 그 상한이 실제로 올라갔음을 보여줍니다.
- `cut_headroom`이 0.005324로 매우 작습니다. 즉 기본 결정 규칙이 이미 이 모델 랭킹에서 거의 최적 지점을 자르고 있고, 남은 여지는 운영 지점(`class_weight`, `scale_pos_weight`)이 아니라 랭킹 축에 있습니다. 이는 관측된 진단이 아니라 이번 시도 자체의 지표를 읽은 것이며, 진단으로 제시하는 것이 아닙니다.

`train_balanced_accuracy` 0.9988 / `train_val_gap` 0.043505는 학습 데이터를 거의 완전히 맞춘 상태를 뜻하지만, 비교 대상 시도가 없어 이것이 성능을 깎았는지는 이 실행 데이터로 판정할 수 없습니다.

## 다음 단계 제안

1. **테스트 마진이 얇다는 점을 다른 seed의 별도 실행으로 확인할 것.** 최종 테스트 0.9473의 CI 하한(0.9305)은 임계값 0.9421 아래에 있고, 초과 마진은 0.0052에 불과합니다. 분할·seed는 실행 내부에서 고정 설정이므로 루프 안에서는 건드릴 수 없습니다 — 동일 구성(`hist_gbdt`, 위 hyperparams/preprocessing)을 seed만 바꾼 여러 실행으로 재현해 보는 것이 유일하게 가능한 확인 경로이며, 통과가 분할 운(運)에 얹혀 있는지 아닌지를 가려 줍니다.
2. **남은 4회 예산은 운영 지점이 아니라 랭킹 축에 쓸 것.** `cut_headroom`=0.005324이므로 `class_weight='balanced'`류로 얻을 수 있는 최대치는 검증 기준 0.9606이 상한이고, 실질 개선 폭은 CI 폭(약 ±0.014)보다 작습니다. 대신 파이프라인을 고정한 채 `xgboost` 또는 `extra_trees`로 계열을 바꿔 `roc_auc` / `balanced_accuracy_at_best_cut`를 비교하는 편이 낫습니다. 다만 실행 지침에 기록된 계열 교체의 관측 폭은 0.0022~0.0077 `roc_auc`이고 페어드 분해 해상도가 0.003~0.006이므로, **이 크기의 개선은 이 검증 슬라이스에서 유의하게 보이지 않을 가능성이 큽니다** — 개선을 기대하기보다 현재 랭킹이 계열 선택에 얼마나 민감한지를 확인하는 용도로 계획하십시오.
3. **용량(capacity) 축을 한 번 눌러 볼 것.** `train_balanced_accuracy` 0.9988 / `train_val_gap` 0.043505 상태에서 `max_leaf_nodes`를 낮추거나 `l2_regularization`을 1.0 이상으로 올린 변형 1~2개를 돌려, 일반화 손실 없이 학습 적합을 낮출 수 있는지 봅니다. 판단은 반드시 CI를 겹쳐 읽고, 0.014 미만의 차이는 개선으로 보고하지 마십시오.
4. **확률값을 그대로 쓸 계획이라면 별도 캘리브레이션 단계를 파이프라인 밖에 만들 것.** `brier`=0.032241, `calibration_error`=0.018949로 확률은 평균 약 1.9%p 어긋나 있고, 이 executor에는 재캘리브레이션 레버가 없습니다(`CalibratedClassifierCV`, 컷 이동 모두 불가). 스팸 필터처럼 임계값을 운영상 조정해야 하는 용도라면, 이 항목은 루프에 제안할 실험이 아니라 **루프 밖에서 구축해야 하는 기능**입니다.

데이터 카드에는 별도로 기록된 주의사항이 없으며(missing rate 0.0, 비수치 열 0개, 타깃 결측 0), 위 제안 중 어느 것도 특정 열이나 분할의 신뢰성에 의존하지 않습니다. `missing_indicator` / `missing_count`는 결측이 전혀 없는 이 데이터에서는 아무것도 추가하지 못하므로 후보에서 제외했습니다.