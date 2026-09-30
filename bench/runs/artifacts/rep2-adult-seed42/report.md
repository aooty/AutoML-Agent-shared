# AutoML 최종 리포트 — `adult` (binary_classification, balanced_accuracy)

## 요약

목표를 달성했습니다. 1회차 시도에서 `hist_gbdt`가 검증 `balanced_accuracy` **0.8371 (95% CI 0.8283~0.8455)** 을 기록해 목표 임계값 0.8248을 넘겼고, 루프는 5회 예산 중 1회만 사용하고 종료했습니다. 한 번도 학습·선택에 쓰이지 않은 최종 테스트 20%에서도 **0.8335 (95% CI 0.8240~0.8413)** 로 임계값 위를 유지했습니다. baseline(`logreg`, median impute + standard scale) 검증 점수 0.7664 (CI 0.7575~0.7762) 대비 상당한 개선이며, 개선은 운영점 이동뿐 아니라 순위(ranking) 축에서도 확인됩니다(roc_auc 0.9075 → 0.9297, `balanced_accuracy_at_best_cut` 0.8238 → 0.8498).

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `learning_rate=0.06`, `max_iter=600`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=30`, `class_weight={"0":1.0,"1":2.2}` / preprocessing `impute="none"`, `scale=False` | status `ok` — `balanced_accuracy=0.8371` (CI 0.8283~0.8455), `roc_auc=0.9297`, `recall=0.8032`, `specificity=0.8710`, `cut_headroom=0.0127`, `train_val_gap=0.0324`, 학습 8.716초, early stopping이 243/600 iter에서 정지 | 없음 (목표 달성으로 critic 미실행) |

## 최고 성능 구성

- **model**: `hist_gbdt` (iteration 1)
- **hyperparams** (실제 적용값, `dropped_hyperparams`는 없음):
  ```json
  {
    "learning_rate": 0.06,
    "max_iter": 600,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": true,
    "validation_fraction": 0.1,
    "n_iter_no_change": 30,
    "class_weight": {"0": 1.0, "1": 2.2}
  }
  ```
- **preprocessing** (해당 attempt에 기록된 적용값):
  ```json
  {"impute": "none", "scale": false, "missing_indicator": false, "missing_count": false}
  ```
  즉 결측 대치를 하지 않고 트리가 NaN을 직접 분기했습니다(카드의 `impute: median, scale: true`는 요청값일 뿐 이 실행에는 적용되지 않았습니다).
- **내부 early stopping**: `fit_rows=26373`, `held_out_rows=2931`, `stopped_at_iter=243` (`max_iter=600`)
- **프로토콜**: stratified 3-way split, seed 42, train 60% / validation 20% / test 20% (baseline과 동일 split)

| 지표 | 검증(validation) | 최종 테스트(held-back) |
|---|---|---|
| `balanced_accuracy` | **0.8371** (CI 0.8283~0.8455) | **0.8335** (CI 0.8240~0.8413) |
| `roc_auc` | 0.9297 | — |
| `pr_auc` / `average_precision` | 0.8306 | — |
| `f1` | 0.7257 | — |
| `accuracy` | 0.8547 | — |
| `precision` / `recall` | 0.6618 / 0.8032 | — |
| `specificity` | 0.8710 | — |
| `balanced_accuracy_at_best_cut` / `cut_headroom` | 0.8498 / 0.0127 | — |
| `brier` / `calibration_error` | 0.09602 / 0.064848 | — |
| `train_balanced_accuracy` / `train_val_gap` | 0.8695 / 0.0324 | — |

검증 0.8371과 테스트 0.8335의 차이 **+0.0036** 은 선택 편향(selection effect)의 크기입니다 — 루프는 검증 숫자만 보고 최종 구성을 골랐습니다. 다만 검증 점수가 테스트 CI(0.8240~0.8413) 안에 들어오므로, 이 행들만으로는 그 차이를 0과 구분할 수 없습니다. 이 실행이 실제로 입증하는 값은 테스트 점수 0.8335입니다.

## 원인 분석

- **critic은 한 번도 실행되지 않았습니다** (시도 1회, 진단 0회). 따라서 여러 시도에 걸친 판정 패턴은 존재하지 않으며, 이 점수는 **첫 계획 하나가 낸 결과**입니다. 이 실행은 진단·재계획 경로가 유용하다는 증거도, 해롭다는 증거도 제공하지 않습니다.
- 시도가 하나뿐이므로 시도 간 이동으로 원인을 구성할 수 없고, 아래는 진단이 아니라 이 attempt에 기록된 사실의 확인일 뿐입니다: baseline의 순위 상한(`balanced_accuracy_at_best_cut=0.8238`)은 목표 0.8248보다 낮았고, 실제로 넘긴 구성은 순위 축도 함께 올렸습니다(`roc_auc` 0.9075→0.9297, best-cut 0.8238→0.8498). 남은 `cut_headroom`은 0.0127로 검증 CI 폭(약 0.017)보다 작습니다.
- `plan.unsupported_claims`에 `feature_engineering`이 기록되어 있으나, 1회차 계획 본문은 "same imputed matrix", `impute: 'none'`, `missing_indicator`가 `impute: none`과 함께는 중복이라는 점 등을 언급하는 수준이고 파생 피처 생성에 의존하지 않습니다. 즉 이 플래그는 문구 검사에 걸린 것이며, 이 시도가 피처 엔지니어링 때문에 무엇을 잃은 것은 아닙니다. 해당 역량은 `## 다음 단계 제안`의 "만들어야 할 것" 항목으로 다룹니다.
- 성능을 제약한 요인을 데이터 쪽에서 지목할 근거도 이 실행에는 없습니다. `Data caveats`는 비어 있고, 별도 진단도 수행되지 않았습니다.

## 다음 단계 제안

1. **테스트 점수를 최종 수치로 보고하고, 예산 4회는 "이 구성이 재현되는가"에 쓰기.** 목표는 이미 충족됐으므로 추가 탐색의 목적은 점수 경쟁이 아니라 확인입니다. 단, 검증 CI 폭이 약 0.017이고 남은 `cut_headroom`이 0.0127이므로, 이보다 작은 변화는 이 데이터로 구분되지 않습니다 — 0.005 수준의 개선을 "개선"으로 보고하지 마십시오.
2. **순위 축만 겨냥한 비교 1~2회** (`xgboost` 패밀리 교체, 또는 `hist_gbdt` 안에서 `learning_rate`/`max_leaf_nodes` 재조정). 실행기 문서가 기록한 크기는 패밀리 교체 0.0022~0.0077 roc_auc, 동일 패밀리 내 재조정 0.0032이고 roc_auc 차이의 해상도는 0.003~0.006이므로, 판단은 `balanced_accuracy`가 아니라 `roc_auc` / `balanced_accuracy_at_best_cut`로 하고 겹치는 구간은 "구분 불가"로 남기십시오. 현재 early stopping이 243/600에서 멈췄으므로 `max_iter`를 더 늘리는 것은 무의미합니다.
3. **운영점(class_weight)을 더 만지는 데 예산을 쓰지 마십시오.** `cut_headroom=0.0127`이며 `recall=0.8032` / `specificity=0.8710`은 이미 근접 균형입니다 — 최적 컷으로 옮겨도 상한이 0.8498이고, 그 이득은 CI 폭보다 작습니다. 남은 개선 여지는 순위(모델 패밀리·피처)에 있습니다.
4. **결측 관련 레버는 건드리지 말고, 실제로 부족한 두 가지를 도구로 만들어 두기.**
   - `impute: none`과 `missing_indicator`의 조합은 예측이 비트 단위로 동일해지는 것이 확인된 중복이므로 시도 금지. `missing_count`도 랜덤 split에서 0.0003/0.0000였습니다. 게다가 이 카드에는 `workclass`(0.0573), `occupation`(0.0575), `native-country`(0.0175)의 **결측 원인에 대한 caveat이 하나도 기록되어 있지 않아**, 결측 컬럼이 기록 체제를 학습하는지 판별할 근거가 없습니다 — 원인 문서화가 선행 조건입니다.
   - 확률값을 그대로 쓸 계획이라면 `calibration_error=0.0648`, `brier=0.09602`은 무시할 수 없는 크기입니다(평균 약 6.5%p 오차). 이 루프에는 재보정 레버가 없으므로, 필요하면 루프 밖 별도 파이프라인에서 보정하십시오.
   - 파생 피처(예: `capital-gain`/`capital-loss` 변환, 컬럼 간 상호작용)는 실행기가 제공하지 않는 기능입니다. 시도해 볼 가치가 있다면 플랜이 아니라 **데이터 카드/전처리 파이프라인 쪽에 추가**해야 하며, 그것이 순위 축을 더 올릴 수 있는 유일한 미탐색 방향입니다.