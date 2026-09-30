# AutoML 실행 보고서 — `speeddating` / `balanced_accuracy`

## 요약

목표는 달성되었습니다. 목표 기준선 `balanced_accuracy` = 0.7514에 대해, 단 1회 시도(예산 1회 중 1회 사용)로 나온 `hist_gbdt` 모델이 검증 슬라이스에서 0.7578 (95% CI 0.7283~0.7872)을 기록했고, 한 번도 사용되지 않은 최종 테스트 20%에서는 0.7781 (95% CI 0.7503~0.8073)을 기록했습니다. 카드의 `logreg` 베이스라인 0.6685 (CI 0.6425~0.6950) 대비 확실한 개선이며, 베이스라인의 최적 컷 상한(`balanced_accuracy_at_best_cut` = 0.7712)도 테스트 점수 기준으로 넘어섰습니다. 첫 계획이 곧바로 기준을 통과했기 때문에 critic(진단·재계획) 경로는 한 번도 실행되지 않았습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=25`, `class_weight='balanced'`, `random_state=42` | status `ok` — 검증 `balanced_accuracy`=0.7578 (CI 0.7283~0.7872), `roc_auc`=0.8726, `pr_auc`=0.5939, 학습 4.278초, 내부 조기중단 95/400 iter | 없음 (목표 달성으로 루프 종료 — critic 미실행) |

## 최고 성능 구성

iteration 1에서 실제로 실행된(= 실행기가 만든 추정기에서 읽어낸) 구성입니다.

- **model**: `hist_gbdt`
- **hyperparams** (적용값, `dropped_hyperparams` 없음):
  `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=True`, `validation_fraction=0.1`, `n_iter_no_change=25`, `class_weight='balanced'`, `random_state=42`
- **preprocessing** (적용값): `impute: none`(NaN을 모델이 직접 분기), `scale: false`, `missing_indicator: false`, `missing_count: false`
- **프로토콜**: seed 42, stratified 60/20/20. 검증 20%에서 모든 선택이 이루어졌고, 테스트 20%는 루프 종료 후 단 한 번만 채점되었습니다.
- **내부 조기중단**: fit_rows 4523 / held_out_rows 503, 95번째 iteration에서 중단.

| 지표 | 검증(20%) |
|---|---|
| `balanced_accuracy` | **0.7578** (CI 0.7283~0.7872) |
| `roc_auc` | 0.8726 |
| `pr_auc` / `average_precision` | 0.5939 |
| `f1` | 0.5628 |
| `accuracy` | 0.8359 |
| `precision` / `recall` | 0.5014 / 0.6413 |
| `specificity` | 0.8743 |
| `balanced_accuracy_at_best_cut` / `cut_headroom` | 0.8030 / 0.0452 |
| `brier` / `calibration_error` | 0.1124 / 0.0989 |
| `train_balanced_accuracy` / `train_val_gap` | 0.9440 / 0.1862 |

**최종 held-back 측정**: 테스트 20%에서 `balanced_accuracy` = **0.7781** (CI 0.7503~0.8073). 검증 0.7578과 0.0203 차이가 나며, 이 차이가 곧 선택 편향의 크기입니다 — 이 실행의 모든 선택은 검증 숫자로 이루어졌고, 테스트 숫자만이 모델에 대해 실제로 입증된 값입니다. 다만 검증 점수가 테스트 CI(0.7503~0.8073) 안에 들어 있으므로, 이 행 수로는 그 차이를 0과 구분할 수 없습니다.

## 원인 분석

이 실행에는 critic 판정이 **0건**입니다(시도 1회, 진단 0회). 첫 계획이 곧바로 목표 기준선을 넘겨 루프가 종료되었기 때문에, 진단·재계획 경로는 이 점수에 전혀 기여하지 않았습니다. 따라서 이 실행은 그 경로가 유용하다는 증거도, 해롭다는 증거도 제공하지 않습니다. 여러 판정 사이의 "패턴"이라 부를 것도 존재하지 않으며, 시도가 하나뿐이므로 시도 간 변화로 이야기를 만들 수도 없습니다.

또한 아래 관찰들은 **진단이 아니라 단일 시도의 계측값**임을 분명히 해둡니다. 어떤 진단도 내려지지 않았습니다.

- 계획이 명시한 전략은 "선형 베이스라인을 부스팅 트리로 교체해 랭킹 축을 개선하고, 동시에 `class_weight='balanced'`로 기본 컷을 recall 쪽으로 옮긴다"였고, 실행된 값은 그 계획과 일치합니다. 결과적으로 `roc_auc`는 베이스라인 0.8505에서 0.8726으로, `balanced_accuracy_at_best_cut`은 0.7712에서 0.8030으로 이동했습니다.
- `plan.unsupported_claims`에 `feature_engineering`이 기록되어 있으나, 해당 계획 본문은 열 구성을 서술하는 문맥에서 표현이 겹친 것이고 파생 피처 생성에 의존하지 않습니다(실행기는 피처 생성을 하지 않으며, 실제 적용 전처리도 `impute: none` / `scale: false`뿐입니다). 따라서 이 항목 때문에 시도가 무엇을 잃었다고 볼 근거는 없습니다 — 피처 엔지니어링은 "실패한 것"이 아니라 "이 실행기에 없는 기능"이며, 다음 단계에서 다룰 사안입니다.
- 남아 있는 여유는 두 축으로 나뉘어 계측되어 있습니다: `cut_headroom` = 0.0452(운영점 축, recall 0.6413 대 specificity 0.8743의 비대칭)와, 최적 컷을 쓰더라도 0.8030에서 멈추는 랭킹 축. 어느 쪽이 다음에 더 크게 움직일지는 이 한 번의 측정으로 결정되지 않습니다.

## 다음 단계 제안

(데이터 카드에 기록된 caveat는 없으므로, 특정 열·값·split을 배제해야 할 제약은 없습니다.)

1. **운영점 축을 `'balanced'` 한 점에서 벗어나 탐색한다.** `cut_headroom` = 0.0452는 작지 않고, recall 0.6413 / specificity 0.8743의 비대칭은 기본 컷이 아직 음성 쪽에 치우쳐 있다는 뜻입니다. 실행기에서 컷 자체는 조정할 수 없으므로, 명시적 가중치 맵(예: `class_weight={"0": 1, "1": 8}` 등 클래스 빈도비 5.07보다 무겁게)을 몇 점 두고 `recall`/`specificity`/`cut_headroom`을 함께 읽는 것이 가장 직접적인 다음 시도입니다. 단, `balanced_accuracy` 차이가 검증 CI 폭(약 ±0.03)보다 작으면 개선으로 보고하지 않아야 합니다.
2. **랭킹 축은 기대치를 작게 잡고 시도한다.** `xgboost`를 `impute: none` + `early_stopping_rounds`로 한 번, 그리고 `hist_gbdt` 재튜닝(`train_val_gap` = 0.1862를 감안해 `min_samples_leaf` 상향 / `max_leaf_nodes` 하향 / `l2_regularization` 강화)을 한 번 시도할 가치가 있습니다. 다만 사전 계측에 따르면 동일 계열 재튜닝은 `roc_auc` 0.0032 폭, 계열 교체는 0.0022~0.0077 폭이며 paired 해상도가 0.003~0.006이므로, 이 축에서의 이득은 통계적으로 구분되지 않을 가능성이 높습니다 — 예산을 크게 걸 대상은 아닙니다.
3. **`missing_indicator`에는 iteration을 쓰지 않는다.** 이번 승자는 `impute: none`으로 NaN을 직접 분기하므로 지표 열은 중복이며(비트 단위로 동일한 예측이 재현된 사례가 있음), `expected_num_interested_in_me`(결측률 0.7852)처럼 결측이 큰 열도 이미 트리가 표현합니다. 검증할 가치가 있는 것은 `missing_count`(행 단위 집계)이지만, 기존 계측에서 0.0003 수준이었으므로 우선순위는 낮습니다.
4. **확률값을 쓰려면 별도 작업이 필요하다.** `calibration_error` = 0.0989, `brier` = 0.1124 — 랭킹은 좋지만 확률은 평균 약 10%p 어긋나 있습니다. 이 실행기에는 recalibration 레버가 없으므로, 확률 기반 의사결정이 필요하다면 파이프라인 밖에서 보정 단계를 만드는 것이 선행 과제입니다. 마찬가지로 피처 파생(선호도-평가 차이, 쌍 단위 상호작용 등)과 임계값 탐색도 실행기 밖에서 구축해야 할 항목이며, 그것이 열리면 랭킹 축의 상한(현재 최적 컷 0.8030) 자체를 밀어올릴 여지가 생깁니다.
5. **재계획 경로는 여전히 미검증 상태**이므로, 다음 실행에서 iteration 예산을 2회 이상 주어 critic이 최소 한 번 판정을 내리게 하는 것을 권합니다. 이번 결과는 그 경로에 대해 어떤 판단도 뒷받침하지 않습니다.