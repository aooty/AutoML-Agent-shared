# AutoML 실행 리포트 — `speeddating` / `balanced_accuracy`

## 요약

목표는 달성되었습니다. 목표 기준선은 `balanced_accuracy` 0.7514였고, iteration 2의 `hist_gbdt`가 검증 슬라이스에서 0.7904 (95% CI 0.7604~0.8149)를 기록했습니다. 한 번도 사용되지 않은 최종 테스트 20%에서는 같은 모델이 0.7810 (95% CI 0.7515~0.8064)으로, 이 값이 이번 실행이 실제로 입증한 성능입니다. 5회 예산 중 2회만 사용했고, 두 번째 시도에서 목표를 넘겨 루프가 `goal_reached`로 종료되었습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `min_samples_leaf=20`, `max_iter=400`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=30`, `class_weight={0:1, 1:3}`; `impute='none'`, `scale=false` | `balanced_accuracy` 0.7187 (CI 0.6894~0.7488) — 목표 대비 -0.0327. `roc_auc` 0.8712, `balanced_accuracy_at_best_cut` 0.8039, `cut_headroom` 0.0852, `recall` 0.5217 vs `specificity` 0.9157, `train_val_gap` 0.2416 | `data_issue`. 랭킹 축은 이미 충분(`roc_auc` 0.8712 > baseline 0.8505, best-cut 0.8039은 기준선보다 0.0525 위)이며, 부족분 0.0327은 전부 operating point에 있음(`cut_headroom`이 부족분의 261%). recall이 specificity보다 0.394 낮아 양성 클래스 가중치 부족. 처방: 파이프라인 그대로 두고 `class_weight`만 `'balanced'`(≈5.07) 수준으로 올릴 것 |
| 2 | `hist_gbdt` | `learning_rate=0.05`, `max_leaf_nodes=15`, `l2_regularization=3.0`, `min_samples_leaf=40`, `max_iter=400`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=30`, `class_weight={0:1, 1:6}`; `impute='none'`, `scale=false` | **`balanced_accuracy` 0.7904 (CI 0.7604~0.8149)** — 목표 초과. `recall` 0.7536, `specificity` 0.827143, `roc_auc` 0.8726, `pr_auc` 0.5920, `cut_headroom` 0.0109, `train_val_gap` 0.1154 | (목표 달성으로 critic 미실행) |

두 시도 모두 `status='ok'`, `dropped_hyperparams`는 비어 있고 `unsupported_claims`도 없습니다. 학습 시간은 각각 4.554s / 4.873s로 `max_train_time_sec=600` 대비 여유가 컸습니다.

## 최고 성능 구성

**iteration 2 / `hist_gbdt`** — 아래는 히스토리에 기록된 *적용된* 값입니다.

적용된 `hyperparams`:

```json
{
  "max_iter": 400,
  "learning_rate": 0.05,
  "max_leaf_nodes": 15,
  "l2_regularization": 3.0,
  "min_samples_leaf": 40,
  "early_stopping": true,
  "validation_fraction": 0.1,
  "n_iter_no_change": 30,
  "class_weight": {"0": 1.0, "1": 6.0},
  "random_state": 42
}
```

적용된 `preprocessing`:

```json
{
  "impute": "none",
  "scale": false,
  "missing_indicator": false,
  "missing_count": false
}
```

즉 결측치는 대치하지 않고 `hist_gbdt`가 NaN 분기로 직접 처리했으며, 스케일링과 결측 파생 열은 사용하지 않았습니다. 분할은 실행 프로토콜 고정값(stratified 60/20/20, seed 42)입니다.

| 지표 | 검증(20%) | 최종 테스트(20%, 미사용 행) |
|---|---|---|
| `balanced_accuracy` | **0.7904** (CI 0.7604~0.8149) | **0.7810** (CI 0.7515~0.8064) |
| `recall` | 0.7536 | — |
| `specificity` | 0.827143 | — |
| `roc_auc` | 0.8726 | — |
| `pr_auc` / `average_precision` | 0.5920 | — |
| `f1` | 0.5730 | — |
| `precision` | 0.4622 | — |
| `accuracy` | 0.8150 | — |
| `brier` | 0.128358 | — |
| `calibration_error` | 0.146973 | — |
| `balanced_accuracy_at_best_cut` / `cut_headroom` | 0.8013 / 0.0109 | — |
| `train_balanced_accuracy` / `train_val_gap` | 0.905743 / 0.11536 | — |

검증 0.7904와 테스트 0.7810의 차이 **+0.0094**는 선택 편향(selection effect)의 크기입니다 — 루프는 검증 숫자만 보고 최종 모델을 골랐기 때문입니다. 다만 검증 점수가 테스트 CI(0.7515~0.8064) 안에 들어오므로, 이 테스트 행들만으로는 그 차이를 0과 구분할 수 없습니다. 보고해야 하는 수치는 테스트 0.7810이며, 이 값은 목표 0.7514를 넘습니다(단, 테스트 CI 하단 0.7515가 기준선에 거의 붙어 있음).

## 원인 분석

- **부족분의 위치는 처음부터 랭킹이 아니라 컷이었습니다.** iteration 1은 `roc_auc` 0.8712, `balanced_accuracy_at_best_cut` 0.8039로 이미 목표(0.7514)를 훨씬 넘는 순위 품질을 갖고 있었지만, 기본 `predict()` 규칙이 자른 지점에서의 점수는 0.7187에 불과했습니다. `cut_headroom` 0.0852는 남은 거리 0.0327의 261%였고, `recall` 0.5217 vs `specificity` 0.9157이라는 0.394의 비대칭이 방향까지 지목했습니다. `balanced_accuracy`는 이 두 항의 평균이므로 낮은 항(recall)을 올리는 것이 정확히 필요한 조치였습니다.
- **두 시도의 차이는 리샘플 노이즈가 아닙니다.** iteration 1 CI 상단 0.7488과 iteration 2 CI 하단 0.7604는 겹치지 않으므로, 0.0717의 상승은 이 데이터가 구분해 주는 실제 개선입니다.
- **개선은 전부 operating point 축에서 나왔습니다.** 랭킹 축 지표는 `roc_auc` 0.8712 → 0.8726(+0.0014), `balanced_accuracy_at_best_cut` 0.8039 → 0.8013(-0.0026)으로, 실행 환경에서 알려진 `roc_auc` 짝지음 해상도(0.003~0.006)보다 작습니다. 즉 모델의 순위 능력은 사실상 그대로이고, `class_weight`를 `{0:1, 1:3}` → `{0:1, 1:6}`으로 올려 컷을 옮긴 것이 점수를 만들었습니다: `recall` 0.5217 → 0.7536, `specificity` 0.9157 → 0.827143, `cut_headroom` 0.0852 → 0.0109.
- **동시에 줄인 용량(capacity)은 과적합 지표를 정리했지만, 점수의 주인은 아닙니다.** `train_val_gap` 0.2416 → 0.11536, `train_balanced_accuracy` 0.9603 → 0.9057로 내려갔습니다. 그러나 `learning_rate`/`max_leaf_nodes`/`l2_regularization`/`min_samples_leaf`가 `class_weight`와 한 번에 바뀌었기 때문에, 이 정규화가 랭킹에 얼마를 기여했는지는 이번 실행으로 분리되지 않습니다 — 랭킹 지표가 해상도 이내로 움직였다는 사실만 확인됩니다.
- **남은 여유는 컷 축에 거의 없습니다.** iteration 2의 `cut_headroom`은 0.0109로, 어떤 임계값을 골라도 이 순위에서 얻을 수 있는 최대치(0.8013)까지 0.011뿐입니다. 앞으로의 개선은 랭킹 축(모델 패밀리·피처)에서만 나옵니다.
- **확률의 품질은 나빠졌습니다.** `brier` 0.101756 → 0.128358, `calibration_error` 0.050265 → 0.146973. 즉 예측 확률은 평균적으로 약 14.7%p 어긋나 있으며, 이번 승리 구성은 "잘 정렬되지만 과신하는" 모델입니다. 실행 환경에는 재보정 레버가 없으므로 이는 이번 실행이 고칠 수 없었던 항목이고, 확률값 자체를 쓸 계획이라면 그대로 배포하면 안 됩니다.
- 두 시도 모두 `dropped_hyperparams`가 비어 있어 실행기가 거부한 설정 때문에 잃은 것은 없습니다. `unsupported_claims`도 비어 있어 계획이 불가능한 기능에 의존한 흔적은 없습니다. 자원 제약(4.9s / 600s)도 병목이 아니었습니다.

## 다음 단계 제안

1. **테스트 CI 하단이 기준선에 붙어 있다는 점을 먼저 해소하십시오(추가 학습 없음).** 테스트 `balanced_accuracy` 0.7810의 CI는 0.7515~0.8064이고 목표는 0.7514입니다 — 목표 초과는 사실이지만 여유가 CI 폭 대비 매우 작습니다. 같은 구성(iteration 2의 `hyperparams`/`preprocessing`, seed 42)을 다른 seed의 고정 프로토콜로 재실행해 검증/테스트 점수의 seed 분산을 측정하는 것이, 지금 하이퍼파라미터를 더 만지는 것보다 정보량이 큽니다. (seed·분할 변경은 이번 실행 설정 밖이므로 새 런으로 잡아야 합니다.)
2. **남은 개선은 랭킹 축에서만 찾으십시오 — `xgboost` + `scale_pos_weight` 스왑 1회.** `cut_headroom`이 0.0109로 사실상 소진되었으므로 `class_weight` 미세조정(3~6 사이 탐색)의 기대 이득은 0.011 이하이고, 이는 검증 CI 폭(약 ±0.027)보다 작아 측정으로 구분되지 않습니다. 대신 파이프라인(`impute='none'`, `scale=false`)을 고정한 채 패밀리만 `xgboost`로 바꾸고 `scale_pos_weight`를 5~6 근처로 두어 `roc_auc`/`balanced_accuracy_at_best_cut`이 0.8726/0.8013 위로 움직이는지 확인하십시오. 이 환경에서 패밀리 스왑의 관측된 크기는 0.0022~0.0077 `roc_auc`로 보고되어 있어, 해상도(0.003~0.006) 경계에 걸친 도박임을 감안해 1회만 쓰는 것이 맞습니다.
3. **`missing_indicator` / `missing_count`에는 예산을 쓰지 마십시오.** `impute='none'`으로 NaN을 직접 분기하는 `hist_gbdt`/`xgboost`에서는 `missing_indicator`가 중복이라는 점이 이미 확정 사실로 기록되어 있고, `missing_count`도 랜덤 분할에서 단독 0.0003, 지시자 위에서 0.0000이었습니다. 결측률이 0.7852인 `expected_num_interested_in_me` 같은 열은 지금 방식(NaN 분기)으로 이미 표현되고 있습니다.
4. **확률을 쓰려면 별도 재보정 단계를 이 루프 밖에 만드십시오.** `calibration_error` 0.146973, `brier` 0.128358은 `class_weight={0:1,1:6}`로 컷을 밀어낸 대가입니다. 실행기에는 `CalibratedClassifierCV`나 임계값 이동 레버가 없으므로, (a) 라벨/점수만 필요한 용도라면 현재 구성 그대로 쓰고, (b) 확률 해석이 필요하면 `class_weight`를 낮춘 잘 보정된 변형(iteration 1은 `calibration_error` 0.050265였습니다)을 랭킹 모델로 쓰고 보정·임계값 결정은 파이프라인 밖 별도 단계로 구현하십시오. 이는 "다음 시도"가 아니라 구축해야 할 기능입니다.
5. **피처 축을 열려면 카드부터 바꿔야 합니다.** `field`는 고카디널리티로 드롭되어 있고(`dropped_high_cardinality`), 실행기는 파생·재인코딩·드롭을 하지 않습니다. `field`를 소수 범주로 묶은 열을 입력 파일 단계에서 추가해 카드에 포함시키는 것이 랭킹 축에서 남은 유일한 구조적 레버이며, 이것이 2번의 패밀리 스왑이 실패했을 때의 다음 목적지입니다. (이 데이터에는 별도 기록된 caveat이 없어, 특정 열·값·분할을 신뢰할 수 없다는 제약은 위 제안들에 걸리지 않습니다.)