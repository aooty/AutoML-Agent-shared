# AutoML 실행 보고서 — bank-marketing (balanced_accuracy)

## 요약

목표는 달성되었습니다. 목표 임계값 `balanced_accuracy` 0.7462에 대해, 첫 번째 시도인 `hist_gbdt`가 검증 슬라이스에서 0.8696 (95% CI 0.8572~0.8807)을 기록했고, 한 번도 사용되지 않은 최종 테스트 20%에서 0.8661 (95% CI 0.8555~0.8771)을 기록했습니다. 사용된 반복은 5회 중 1회이며, 첫 계획이 임계값을 넘겨 루프가 즉시 종료되었습니다. 기준선(logreg, balanced_accuracy 0.6616, CI 0.6456~0.6757)과 비교하면 검증·테스트 양쪽에서 구간이 겹치지 않는 명확한 개선입니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `learning_rate=0.06`, `max_iter=500`, `max_leaf_nodes=31`, `min_samples_leaf=30`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=30`, `class_weight={"0":1.0,"1":8.0}` | `status=ok` — `balanced_accuracy=0.8696` (CI 0.8572~0.8807), `roc_auc=0.9318`, `pr_auc=0.6341`, `recall=0.8809`, `specificity=0.8583`, `train_val_gap=0.0417`, 학습 5.05초, 내부 early stopping은 172/500에서 정지 | 없음 (목표 달성으로 critic 미실행) |

## 최고 성능 구성

**iteration 1 — `hist_gbdt`** (실제로 빌드된 값 기준)

- `hyperparams` (적용값): `learning_rate=0.06`, `max_iter=500`, `max_leaf_nodes=31`, `min_samples_leaf=30`, `l2_regularization=1.0`, `early_stopping=true`, `validation_fraction=0.1`, `n_iter_no_change=30`, `class_weight={"0": 1.0, "1": 8.0}`
- `preprocessing` (적용값): `impute="median"`, `scale=false`, `missing_indicator=false`, `missing_count=false`
- `dropped_hyperparams`: 없음 (요청한 모든 키가 그대로 적용됨)
- `internal_validation`: `fit_rows=24413`, `held_out_rows=2713`, `validation_fraction=0.1`, `stopped_at_iter=172`
- 프로토콜: 층화 3분할(train 60% / val 20% / test 20%), seed 43 — 카드의 기준선과 동일한 분할

검증 슬라이스 지표:

| metric | 값 |
|---|---|
| `balanced_accuracy` | **0.8696** (95% CI 0.8572~0.8807) |
| `balanced_accuracy_at_best_cut` | 0.8711 |
| `cut_headroom` | 0.001475 |
| `roc_auc` | 0.9318 |
| `pr_auc` / `average_precision` | 0.6341 |
| `recall` | 0.8809 |
| `specificity` | 0.858342 |
| `precision` | 0.4518 |
| `f1` | 0.5972 |
| `accuracy` | 0.8610 |
| `brier` | 0.098036 |
| `calibration_error` | 0.125103 |
| `train_balanced_accuracy` | 0.9113 |
| `train_val_gap` | 0.041692 |

**최종 held-back 테스트(20%, 루프 중 한 번도 사용되지 않은 행): `balanced_accuracy=0.8661` (95% CI 0.8555~0.8771).**

검증 0.8696 대비 테스트 0.8661로 −0.0035 차이가 있으며, 이 차이가 바로 선택 편향의 크기입니다 — 루프의 모든 선택은 검증 숫자로 이루어졌고, 모델에 대해 이 실행이 실제로 입증하는 값은 테스트 숫자입니다. 다만 검증 점수가 테스트 CI(0.8555~0.8771) 안에 들어가므로, 이 행들만으로는 그 차이를 0과 구분할 수 없습니다.

## 원인 분석

이 실행에서는 critic이 **한 번도 실행되지 않았습니다** (시도 1회, 진단 0회). 첫 계획이 곧바로 임계값을 넘겨 루프가 종료되었기 때문입니다. 따라서 verdict의 패턴이라 부를 만한 것이 존재하지 않으며, 이 점수는 **첫 번째 계획 하나가 낸 결과**입니다. 진단·재계획 경로가 이 결과에 기여했다는 증거도, 해가 되었다는 증거도 이 실행에는 없습니다 — 그 경로에 대해 이 실행은 어느 방향으로도 말해주지 않습니다.

또한 시도가 1건뿐이므로 시도 간 이동을 근거로 한 인과 설명 자체가 성립하지 않습니다. 비교 가능한 것은 기준선과의 차이뿐이며, 그것은 구간 폭보다 충분히 큽니다: `balanced_accuracy` 0.6616 (0.6456~0.6757) → 0.8696 (0.8572~0.8807), `roc_auc` 0.9064 → 0.9318, `pr_auc` 0.5628 → 0.6341. 즉 개선은 순위(ranking) 축과 작동점(operating point) 축 양쪽에서 일어났습니다: 기준선의 `balanced_accuracy_at_best_cut`이 이미 0.8415였던 것에 비해 이번 모델은 0.8711로 순위 자체가 올라갔고, 동시에 `class_weight={"0":1,"1":8}`가 기본 0.5 컷을 소수 클래스 쪽으로 옮겨 `recall=0.8809` / `specificity=0.8583`의 거의 균형 잡힌 지점에 착지했습니다. `cut_headroom=0.001475`는 이 컷이 이미 최적 근처라는 뜻이며, 남은 여지는 작동점이 아니라 순위(모델 계열·특징) 쪽에 있습니다. 다만 이것들은 이 시도 자체의 지표를 읽은 것이지, 어떤 진단이 내려진 결과가 아닙니다.

기록상 유의할 점 두 가지:

- `dropped_hyperparams`는 비어 있으므로, 계획이 요청한 설정 중 실행기가 거부한 것은 없습니다.
- `unsupported_claims`에 `feature_engineering`이 표시되어 있으나, 해당 계획 본문은 "the one-hot block of 44 levels goes in as the card encodes it"처럼 **카드가 이미 인코딩한 형태를 그대로 쓴다**는 서술과 "no added missingness columns, no scaling"이라는 서술만 담고 있습니다. 즉 이 표시는 문자열 검사에 걸린 것으로, 계획이 사용할 수 없는 기능에 의존한 흔적은 없습니다. 이 시도가 파생 특징 때문에 무언가를 잃었다고 볼 근거는 없으며, 특징 생성은 "실패한 것"이 아니라 "아직 만들어지지 않은 것"으로 다음 단계에 속합니다.

## 다음 단계 제안

1. **센티널 값 `unknown`을 데이터 준비 단계에서 결측으로 정규화한 카드를 다시 만들 것.** 카드의 caveat는 V16의 81.8%, V9의 28.8%, V4의 4.1%, V2의 0.6%가 결측 표기로 흔히 쓰이는 `unknown`이며 **변환되지 않은 채** 실제 측정치로 집계·학습되었다고 명시합니다. 현재 이 열들의 `target_corr`·통계와 이번 점수는 그 미변환 상태에서 나온 값입니다. 실행기는 열을 재인코딩·삭제할 수 없으므로(“Derive, encode or drop features”는 불가) 이 작업은 카드/원본 쪽에서 해야 합니다. 이것이 해결되면 비로소 `impute: none`(hist_gbdt가 NaN을 직접 분기)이나 `missing_count` 같은 결측 레버를 의미 있게 시험할 수 있습니다 — 지금 상태에서는 결측 자체가 0.0%로 보고되므로 그 레버들은 애초에 아무것도 바꾸지 못합니다.
2. **확률 보정이 필요한 용도라면 별도 단계로 설계할 것.** `brier=0.0980`, `calibration_error=0.1251`은 예측 확률이 평균적으로 12.5%p 어긋나 있음을 뜻합니다. 순위 지표(`roc_auc=0.9318`)와 목표 지표는 이 문제에 영향을 받지 않지만, 확률값을 그대로 쓰는 다운스트림이 있다면 그대로는 부적합합니다. 이 실행기에는 재보정 레버가 없으므로(`CalibratedClassifierCV`·컷 이동 불가) 파이프라인 밖의 후처리 단계로 만들어야 합니다.
3. **더 큰 예산이 있다면 순위 축에만 투자할 것.** `cut_headroom=0.001475`이므로 `class_weight`를 더 만지는 것은 최대 0.0015 수준의 여지밖에 없습니다. 남은 개선은 `balanced_accuracy_at_best_cut`(현재 0.8711)을 올리는 일, 즉 다른 트리 계열(`xgboost`, `random_forest`) 교체나 하이퍼파라미터 재탐색입니다. 단, 알려진 크기(계열 교체 0.0022~0.0077 roc_auc, 동일 계열 재탐색 0.0032 roc_auc)는 이 검증 슬라이스의 CI 폭(0.8572~0.8807, 약 ±0.012 balanced_accuracy)보다 작으므로, **한 번의 검증 측정으로는 구분되지 않을 가능성이 높다**는 점을 미리 감안하고 시도해야 합니다.
4. **V12를 프로덕션에서 쓸 수 있는지 데이터 소유자와 확인할 것.** 카드는 V12의 `target_corr`를 유일하게 `strong`으로 표기하고 있고, 이번 모델의 성능은 상당 부분 이 열에 의존할 가능성이 있습니다. 이 값이 예측 시점에 실제로 관측 가능한지(사후에만 알 수 있는 값이 아닌지)를 확인하지 않으면, 0.8661이라는 테스트 점수가 운영 환경에서 재현될 것이라고 말할 수 없습니다. 이 확인은 모델링이 아니라 데이터 정의 확인 작업이며, 결과에 따라 위 1~3번의 우선순위가 바뀝니다.