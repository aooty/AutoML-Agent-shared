# AutoML 실행 최종 보고서 — `speeddating` / `balanced_accuracy`

## 요약

목표는 달성되었습니다. 목표 기준선 `balanced_accuracy` 0.7514에 대해, 2회차 시도의 `hist_gbdt`가 검증 슬라이스에서 0.7856 (95% CI 0.7572~0.8148)을 기록했고, 루프는 5회 예산 중 2회만 사용하고 `goal_reached`로 종료되었습니다. 한 번도 사용되지 않은 최종 테스트 20%에서의 점수는 0.7780 (95% CI 0.7491~0.8066)으로, 이 값이 이 실행이 실제로 입증한 성능입니다. 두 시도 사이의 개선은 랭킹 성능(roc_auc / pr_auc)이 아니라 결정 지점(operating point)이 `class_weight`로 이동한 결과이며, 그 방향은 1회차 critic이 사전에 지목한 것과 일치합니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.15`, `n_iter_no_change=25`, `class_weight={0:1,1:4}`, `random_state=42` / `impute='none'`, `scale=False` | `balanced_accuracy` 0.7290 (CI 0.6997~0.7597), recall 0.5616 / specificity 0.8964, roc_auc 0.8672, pr_auc 0.5858, `balanced_accuracy_at_best_cut` 0.8017, `cut_headroom` 0.0727, `train_val_gap` 0.2173 — 기준선 0.7514 미달 | `data_issue`: 랭킹 축은 이미 baseline 상한(roc_auc 0.8505, best_cut 0.7712)을 넘겼고 병목은 결정 지점. `cut_headroom` 0.0727이 남은 격차 0.0224의 325%이며 recall≪specificity → 양성 가중치를 클래스 빈도 5.07 위로(`class_weight={0:1,1:7}`) 올리고 나머지는 고정 |
| 2 | `hist_gbdt` | 1회차와 동일, `class_weight={0:1,1:7}`만 변경 / `impute='none'`, `scale=False` | **`balanced_accuracy` 0.7856 (CI 0.7572~0.8148)** — 목표 달성. recall 0.7319 / specificity 0.8393, roc_auc 0.8730, pr_auc 0.5776, `balanced_accuracy_at_best_cut` 0.7996, `cut_headroom` 0.0140, brier 0.1223, `calibration_error` 0.1306, `train_val_gap` 0.1366 | (목표 달성으로 critic 미실행) |

두 시도 모두 `status='ok'`, `dropped_hyperparams` 없음, `unsupported_claims` 없음, 학습 시간 4.5~4.8초(예산 600초).

## 최고 성능 구성

**iteration 2 · `hist_gbdt`** (아래 값은 히스토리에 기록된 *적용된* 하이퍼파라미터와 전처리입니다)

```json
{
  "model": "hist_gbdt",
  "hyperparams": {
    "max_iter": 400,
    "learning_rate": 0.06,
    "max_leaf_nodes": 31,
    "min_samples_leaf": 20,
    "l2_regularization": 1.0,
    "early_stopping": true,
    "validation_fraction": 0.15,
    "n_iter_no_change": 25,
    "class_weight": {"0": 1.0, "1": 7.0},
    "random_state": 42
  },
  "preprocessing": {
    "impute": "none",
    "scale": false,
    "missing_indicator": false,
    "missing_count": false
  }
}
```

- 분할 프로토콜: stratified 60/20/20, seed 42 (카드의 baseline과 동일한 분할).
- `impute='none'`이므로 결측은 대치되지 않고 `hist_gbdt`가 NaN 자체를 분기합니다. 스케일링도 적용되지 않았습니다 — 카드의 `preprocessing`(median + scale)은 요청값일 뿐이며, 이 실행에서는 사용되지 않았습니다.

| 지표 | 검증 (20%) | 최종 테스트 (20%, 미사용 행) |
|---|---|---|
| `balanced_accuracy` | **0.7856** (CI 0.7572~0.8148) | **0.7780** (CI 0.7491~0.8066) |
| `recall` / `specificity` | 0.7319 / 0.8393 | — |
| `roc_auc` / `pr_auc` | 0.8730 / 0.5776 | — |
| `f1` / `accuracy` / `precision` | 0.5747 / 0.8216 / 0.4731 | — |
| `brier` / `calibration_error` | 0.1223 / 0.1306 | — |
| `train_balanced_accuracy` / `train_val_gap` | 0.9222 / 0.1366 | — |

검증 0.7856과 테스트 0.7780의 차이 0.0076이 **선택 편향의 크기**입니다 — 루프는 검증 숫자만 보고 최고 시도를 골랐고, 테스트는 그 선택 이후 단 한 번 채점되었습니다. 검증 점수가 테스트 CI(0.7491~0.8066) 안에 들어오므로 이 행 수로는 그 차이를 0과 구분할 수 없습니다. 다만 테스트 CI의 하단 0.7491은 목표 기준선 0.7514를 근소하게 밑돕니다: 점 추정치로는 기준선을 넘겼지만, 이 슬라이스만으로 "기준선 초과"를 여유 있게 단정할 수는 없습니다.

## 원인 분석

- **랭킹 축은 1회차에 이미 해결되어 있었습니다.** 1회차의 roc_auc 0.8672 / pr_auc 0.5858 / `balanced_accuracy_at_best_cut` 0.8017은 baseline의 0.8505 / 0.5507 / 0.7712를 모두 상회했습니다. 즉 모델이 행을 정렬하는 능력 자체는 첫 시도부터 기준선을 통과할 수준이었고, 부족했던 것은 정렬을 자르는 위치였습니다.
- **실제 병목은 결정 지점이었습니다.** 1회차에서 recall 0.5616 대 specificity 0.8964 — `balanced_accuracy`가 두 값의 평균이므로 이 비대칭이 곧 손실이고, `cut_headroom` 0.0727이 남은 격차 0.0224보다 3배 이상 컸습니다. `class_weight`를 4 → 7(클래스 빈도 5.07보다 높게, `'balanced'`로는 도달할 수 없는 지점)로 올린 뒤 recall 0.7319 / specificity 0.8393으로 이동하고 `cut_headroom`이 0.0140으로 축소되었습니다. 이는 사전에 방향까지 지정된 개입이 예측대로 작동한 사례입니다.
- **두 시도의 CI는 0.7572~0.7597 구간에서 아주 좁게 겹칩니다** (1회차 0.6997~0.7597, 2회차 0.7572~0.8148). 따라서 `balanced_accuracy` 0.7290 → 0.7856이라는 숫자 자체는 이 슬라이스에서 완전히 분리된 차이라고 주장할 수 없습니다. 개선을 신뢰하는 근거는 점수 차이가 아니라 메커니즘입니다: 랭킹 지표는 실질적으로 움직이지 않았고(roc_auc +0.0058은 이 환경에서 알려진 paired 해상도 0.003~0.006의 경계, pr_auc −0.0082, best_cut 0.8017 → 0.7996으로 오히려 소폭 하락), 이동한 것은 recall/specificity의 위치뿐입니다. 즉 두 시도는 **같은 랭킹의 서로 다른 절단점**이며, 리샘플 노이즈가 아니라 지렛대 하나의 방향성 효과로 설명됩니다.
- **용량(capacity)은 제약이 아니었습니다.** 1회차 `train_val_gap` 0.2173, 2회차 0.1366 — 훈련 성능은 충분히 높고(0.9464 / 0.9222) 학습 시간은 예산의 1% 미만입니다. 부족했던 것은 모델의 표현력이 아닙니다.
- **부작용: 확률의 신뢰도가 나빠졌습니다.** 가중치를 올리면서 brier 0.1056 → 0.1223, `calibration_error` 0.0666 → 0.1306으로 악화되었습니다. 2회차 모델의 출력은 순위(랭킹)로는 유효하지만 평균 13%p 어긋난 값이므로 **확률로 읽어서는 안 됩니다**. 이 실행에는 재보정 지렛대가 없으므로 이는 측정된 사실로만 남습니다.
- `dropped_hyperparams`는 두 시도 모두 비어 있고 `unsupported_claims`도 없으므로, 실행기가 거부해서 검증되지 않은 설정은 없습니다. 보고된 숫자는 계획된 구성 그대로가 돌아간 결과입니다.

## 다음 단계 제안

이 데이터셋에는 별도로 기록된 caveat이 없습니다. 따라서 아래 제안은 카드의 집계값과 실행기의 실제 능력 범위만을 근거로 합니다.

1. **양성 가중치를 7보다 조금 더 올려 남은 `cut_headroom` 0.0140을 회수한다** (`class_weight={0:1,1:8.5}` 또는 `{0:1,1:10}`, 나머지 파이프라인·하이퍼파라미터 전부 고정). 근거: 2회차에서도 recall 0.7319 < specificity 0.8393로 여전히 특이도 쪽으로 치우쳐 있고, 같은 랭킹의 최적 절단은 `balanced_accuracy_at_best_cut` 0.7996이므로 순수하게 절단 위치만으로 최대 +0.0140이 남아 있습니다. 이 축은 1회차 → 2회차에서 이미 예측대로 반응한 축이므로 기대 효과가 가장 명확합니다.
2. **랭킹 축을 한 번 별도로 검증한다** — `impute='none'`을 유지한 채 `xgboost`(`scale_pos_weight≈7`)로 계열만 교체. 근거: 남은 여지의 대부분(best_cut 0.7996 이상)은 절단이 아니라 정렬을 개선해야 얻어집니다. 단, 이 환경에서 계열 교체가 움직인 roc_auc 폭은 0.0022~0.0077, paired 해상도는 0.003~0.006이므로 **한 번의 시도로 유의한 차이가 나오지 않을 가능성이 높다는 전제**로 계획해야 하며, 결과는 roc_auc/pr_auc/best_cut으로만 판정하고 `balanced_accuracy` 변동으로 계열 우열을 매기지 말아야 합니다.
3. **`missing_indicator` / `missing_count`에는 시도를 쓰지 않는다.** `impute='none'`에서 NaN 분기를 이미 하는 계열에 indicator를 붙이는 것은 예측이 비트 단위로 동일해질 만큼 중복이며(사전 측정으로 확인된 사항), `missing_count`도 이전 측정에서 0.0003 / 0.0000 수준이었습니다. 결측률 78.5%인 `expected_num_interested_in_me` 같은 열도 `hist_gbdt`가 직접 분기로 처리하고 있으므로 추가 열의 여지는 없습니다.
4. **평가 폭을 넓히는 인프라를 우선한다.** 테스트 CI 하단 0.7491이 목표 0.7514를 근소하게 밑돌고, 두 시도의 검증 CI도 겹칩니다. 즉 지금 남은 불확실성의 상당 부분은 모델이 아니라 **1개의 20% 슬라이스라는 측정 단위**에서 옵니다. 현재 실행기는 CV/out-of-fold를 지원하지 않고 split·seed도 고정이므로, 이는 "다음 하이퍼파라미터"가 아니라 **여러 seed에서 동일 프로토콜을 반복 실행하거나 교차검증을 지원 기능으로 추가**해야 해결됩니다. 그때까지는 0.7780을 점 추정치로 보고하고 ±0.03 수준의 폭을 함께 명시하는 것이 정직한 표현입니다.
5. **(운영 전 확인 사항, 모델 작업 아님)** 확률 출력이 필요한 용도라면 현 모델은 부적합합니다(`calibration_error` 0.1306). 재보정은 이 실행기에 없는 기능이므로, 필요하다면 재보정 단계를 별도로 구축한 뒤 다시 측정해야 합니다. 또한 카드에서 `like`(target_corr `strong`)와 `guess_prob_liked`(`moderate`)처럼 상관이 강한 열들이 **실제 예측 시점에 동일하게 관측되는 값인지**를 운영자에게 확인해 두는 것이 좋습니다 — 이는 caveat으로 기록된 사항은 아니지만, 만약 사후 응답이라면 위 점수 전체의 의미가 달라지므로 모델 개선보다 먼저 답이 나와야 하는 질문입니다.