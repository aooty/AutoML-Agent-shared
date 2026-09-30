# AutoML 최종 리포트 — `jungle-chess` (multiclass_classification, 3 classes)

## 요약

목표는 달성되었습니다. 목표 지표인 `balanced_accuracy` 임계값 0.633에 대해, 첫 번째 시도에서 `hist_gbdt`가 검증 분할에서 **0.7995 (95% CI 0.78912~0.811543)** 를 기록했고, 한 번도 학습·선택에 쓰이지 않은 최종 테스트 20%에서 **0.7978 (95% CI 0.7864~0.8087)** 를 기록했습니다. 사용한 반복은 5회 중 **1회**이며, 첫 계획이 바로 임계값을 넘겨 루프가 종료되었습니다 (`stop_reason: goal_reached`). 참고로 카드의 baseline인 `logreg`는 동일 분할·동일 시드에서 0.5106 (CI 0.5024~0.5177)이었습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=600`, `learning_rate=0.1`, `max_leaf_nodes=255`, `min_samples_leaf=20`, `l2_regularization=1.0`, `class_weight='balanced'`, `early_stopping=False` | `status: ok` — 검증 `balanced_accuracy=0.7995` (CI 0.78912~0.811543), `accuracy=0.8513`, `f1=0.7942`, `train_val_gap=0.199716`, 학습 시간 28.146s | 없음 (goal_reached로 critic 미실행) |

`dropped_hyperparams`는 비어 있어, 제안한 모든 하이퍼파라미터가 그대로 estimator에 적용되었습니다. `unsupported_claims`도 비어 있습니다.

## 최고 성능 구성

iteration 1, model `hist_gbdt`. 아래 값은 history에 기록된 **적용된** 값입니다.

**hyperparams**
```json
{
  "max_iter": 600,
  "learning_rate": 0.1,
  "max_leaf_nodes": 255,
  "min_samples_leaf": 20,
  "l2_regularization": 1.0,
  "class_weight": "balanced",
  "early_stopping": false
}
```

**preprocessing** (해당 시도의 적용값)
```json
{
  "impute": "median",
  "scale": false,
  "missing_indicator": false,
  "missing_count": false
}
```

분할 프로토콜: stratified 60/20/20, seed 42 (카드 baseline과 동일 분할).

| 지표 | 검증(20%) | 최종 테스트(20%, 한 번도 사용되지 않은 행) |
|---|---|---|
| `balanced_accuracy` | **0.7995** (CI 0.78912~0.811543) | **0.7978** (CI 0.7864~0.8087) |
| `accuracy` | 0.8513 | — |
| `f1` | 0.7942 | — |
| `precision` | 0.7895 | — |
| `recall` | 0.7995 | — |
| `train_balanced_accuracy` | 0.999239 | — |
| `train_val_gap` | 0.199716 | — |

검증 0.7995와 테스트 0.7978의 차이는 **+0.0018**이며, 이것이 이 실행의 선택 편향 크기입니다 — 루프는 검증 숫자를 보고 선택했고, 실제로 이 모델에 대해 입증된 값은 테스트 쪽 0.7978입니다. 다만 검증 점수가 테스트 CI(0.7864~0.8087) 안에 들어오므로, 이 행 수로는 그 차이를 0과 구분할 수 없습니다. 어느 쪽 숫자를 쓰더라도 임계값 0.633을 CI 하한(0.7864)까지 포함해 여유 있게 상회합니다.

## 원인 분석

- **critic 진단은 0건입니다.** 시도 1회, 진단 0회로, 첫 계획이 임계값을 넘긴 시점에 루프가 종료되었습니다. 따라서 여러 verdict에 걸친 "패턴"은 존재하지 않으며, 설명할 시도 간 이동도 없습니다. 이 점수는 **첫 계획 하나가 낸 결과**이고, 이 실행은 진단·재계획 경로가 도움이 되는지 해가 되는지에 대해 **어떤 증거도 제시하지 않습니다**.
- 계획이 세운 가설 — "선형 baseline의 랭킹을 버리고 고용량 트리 앙상블로 6개 이산 보드-상태 컬럼의 비선형 상호작용을 학습한다" — 은 baseline 0.5106 (CI 0.5024~0.5177)과 이번 0.7995 (CI 0.78912~0.811543)의 CI가 전혀 겹치지 않는다는 점에서, 이 데이터가 구분해 주는 유일한 크기입니다. 그 외에는 비교 대상 시도가 없어 어떤 레버의 기여도 분리할 수 없습니다.
- 기록에서 눈에 띄는 관찰 하나는 **`train_val_gap = 0.199716`** (학습 `balanced_accuracy` 0.999239 대 검증 0.7995)입니다. 이는 diagnosis가 아니라 단일 시도의 계측값입니다 — 어떤 critic도 이것을 원인으로 판정하지 않았고, 규제를 조인 대조 시도가 없으므로 이 갭이 점수를 깎고 있는지 여부는 이 실행으로 알 수 없습니다.
- 계획은 후속 판단을 위해 `cut_headroom`을 읽겠다고 적었지만, 이번 시도의 `metrics`에는 `balanced_accuracy_at_best_cut`/`cut_headroom`이 기록되어 있지 않습니다(3-class 타깃). 즉 남은 격차가 **랭킹 축**인지 **작동점 축**인지 갈라 읽을 수 있는 숫자가 이 실행에는 없습니다. 이는 계획의 실패가 아니라, 이 설정에서 그 진단값이 제공되지 않았다는 사실입니다.

## 다음 단계 제안

이 데이터셋에는 별도로 기록된 caveat이 없어, 아래 제안을 무효화하는 항목은 없습니다. 다만 **판정 해상도**를 먼저 정하십시오: 검증 CI 폭이 약 ±0.011이므로, 대략 **0.02 미만의 `balanced_accuracy` 차이는 이 분할에서 개선으로 보고할 수 없습니다.**

1. **규제 축을 한 번만 대조해 `train_val_gap=0.199716`의 대가를 실측하기.** 동일 `hist_gbdt`에서 `max_leaf_nodes`를 크게 낮추고 (`31`~`63`), `min_samples_leaf`를 올리고 (`50`~`100`), `l2_regularization`을 키운 (`10.0`) 구성을 1회 적합. 목적은 점수 상승이 아니라 "완전 암기 상태가 손해였는가"를 갈라 읽는 것 — 차이가 0.02 미만이면 갭은 무해했다고 결론 내리고 이 축을 닫으십시오.
2. **`class_weight` 기여를 분리하기.** 소수 클래스 비중이 0.0967(imbalance_ratio 5.32)이고 `balanced_accuracy`는 세 클래스를 동일 가중하므로, `class_weight='balanced'`가 점수의 어느 정도를 만들었는지 알 필요가 있습니다. 동일 구성에서 `class_weight`만 제거한 1회 적합이 그것을 직접 답합니다. 이 설정에서는 `cut_headroom`이 보고되지 않으므로, 작동점 축을 볼 유일한 방법이 이 대조입니다.
3. **패밀리 스왑(`xgboost`)은 "확인용"으로만, 기대치를 낮춰 실행.** 문서화된 패밀리 스왑 효과 크기는 roc_auc 기준 0.0022~0.0077로 이 분할의 판정 해상도와 겹치거나 그 이하입니다. 즉 개선 수단으로는 근거가 약하고, `hist_gbdt`가 이 6컬럼 문제에서 특별히 불리하지 않음을 확인하는 용도로만 값이 있습니다. 예산이 남을 때 마지막에 두십시오.
4. **파생 특성은 executor가 아니라 데이터 카드 쪽에서 만들기.** jungle-chess의 클래스는 두 말의 **상대** 강도·상대 file/rank의 함수일 가능성이 높지만, executor는 컬럼 간 차이·비율·상호작용을 만들 수 없습니다("Derive, encode or drop features" 금지). 따라서 `white_piece0_strength - black_piece0_strength`, `white_piece0_file - black_piece0_file`, `white_piece0_rank - black_piece0_rank` 같은 컬럼을 **CSV/카드 단계에서 추가**한 뒤 다시 실행해야 합니다. 이것이 열리면, 트리가 축 정렬 분할로 근사해야 했던 관계를 직접 표현하게 되어 랭킹 축에서 가장 큰 여지가 남은 후보가 됩니다.