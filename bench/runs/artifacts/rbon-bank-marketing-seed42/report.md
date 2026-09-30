# 최종 보고서 — bank-marketing (balanced_accuracy)

## 요약

목표를 달성했습니다. 단 1회 시도(예산 1회 중 1회 사용)에서 `hist_gbdt`가 검증 `balanced_accuracy` **0.8705** (95% CI 0.8593~0.8815)를 기록해 목표 임계값 0.7465와 카드 baseline(`logreg` 0.662, CI 0.6496~0.6765)을 모두 크게 넘었습니다. 루프 종료 후 한 번도 사용되지 않은 held-back 테스트 20%에서의 점수는 **0.8537** (95% CI 0.8405~0.8657)로, 이것이 이 실행이 실제로 입증한 수치입니다. 첫 계획이 곧바로 기준을 통과했기 때문에 critic은 한 번도 실행되지 않았습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=63`, `max_depth=8`, `l2_regularization=1.0`, `min_samples_leaf=40`, `early_stopping=false`, `class_weight={"0":1.0,"1":9.0}` | status `ok` — validation `balanced_accuracy`=0.8704942967219884 (CI 0.8593~0.8815), `roc_auc`=0.9391, `pr_auc`=0.6417, `recall`=0.8582, `specificity`=0.8828, `train_val_gap`=0.0809, `train_time_sec`=8.354 | 없음 (목표 달성으로 루프 종료 — critic 미실행) |

## 최고 성능 구성

iteration 1, model `hist_gbdt`. 아래 값은 히스토리의 해당 시도에서 읽은 **적용된** 하이퍼파라미터와 전처리입니다 (`dropped_hyperparams`는 비어 있음 — 거부된 키 없음).

**hyperparams**
```json
{
  "max_iter": 400,
  "learning_rate": 0.06,
  "max_leaf_nodes": 63,
  "max_depth": 8,
  "l2_regularization": 1.0,
  "min_samples_leaf": 40,
  "early_stopping": false,
  "class_weight": {"0": 1.0, "1": 9.0}
}
```

**preprocessing**
```json
{"impute": "median", "scale": false, "missing_indicator": false, "missing_count": false}
```

프로토콜: stratified 3-way split, seed 42, train 60% / validation 20% / test 20% (카드 baseline과 동일한 split).

| 지표 | validation | 비고 |
|---|---|---|
| `balanced_accuracy` | **0.8705** (CI 0.8593~0.8815) | 목표 지표 |
| `balanced_accuracy` (held-back test) | **0.8537** (CI 0.8405~0.8657) | 루프 중 한 번도 쓰이지 않은 20% |
| `roc_auc` | 0.9391 | |
| `pr_auc` / `average_precision` | 0.6417 | |
| `f1` | 0.6258 | |
| `accuracy` | 0.8799 | |
| `precision` | 0.4924 | |
| `recall` | 0.8582 | |
| `specificity` | 0.8828 | |
| `balanced_accuracy_at_best_cut` | 0.8784 | 진단용 (목표로 지정 불가) |
| `cut_headroom` | 0.0079 | 진단용 |
| `brier` | 0.0863 | 진단용 |
| `calibration_error` | 0.0970 | 진단용 |
| `train_balanced_accuracy` | 0.9514 | |
| `train_val_gap` | 0.0809 | |
| `train_time_sec` | 8.354 s | |

검증 0.8705와 테스트 0.8537의 차이 **+0.0168**은 선택 편향의 크기입니다. 루프는 검증 숫자만 보고 최고 시도를 골랐으므로, 외부에 인용할 수치는 테스트 0.8537 (CI 0.8405~0.8657)입니다. 이 값 역시 목표 임계값 0.7465와 참조 `ranking_ceiling` 0.84를 상회합니다.

## 원인 분석

`Outcome`의 재계획 라인이 명시하듯 **critic 진단은 0건**입니다. 첫 시도가 곧바로 임계값을 통과해 루프가 종료되었으므로, 이 점수는 **첫 계획 하나가 낸 결과**이며 진단·재계획 경로는 이 결과에 전혀 기여하지 않았습니다. 따라서 이 실행은 diagnose-and-replan 루프가 도움이 된다는 증거도, 해가 된다는 증거도 제시하지 않습니다. 여러 시도 사이의 움직임도 존재하지 않으므로(시도 1회), 시도 간 비교로 성능을 제한한 요인을 특정할 근거 자체가 없습니다. 아래는 진단이 아니라, 향후 판단에 쓰이는 관측된 사실의 정리입니다.

- 이 시도의 `cut_headroom`은 0.0079이고 `balanced_accuracy_at_best_cut`은 0.8784입니다. 즉 기본 결정 규칙이 이미 이 모델 ranking에서 거의 최적 지점에 놓여 있고, `class_weight={"0":1,"1":9}`로 만든 operating point는 `recall` 0.8582 / `specificity` 0.8828로 두 축이 균형에 가깝습니다. 남은 여지는 operating point가 아니라 ranking 축(모델 계열·피처)에 있다는 뜻이지만, 이는 진단으로 검증된 것이 아니라 단일 시도의 진단 지표 판독일 뿐입니다.
- `train_val_gap` 0.0809 (train 0.9514 vs val 0.8705)는 이 설정이 훈련 분할에 상당히 적합했음을 보여줍니다. 다만 held-back 테스트가 0.8537로 검증 CI 하한 근처에 들어왔으므로, 과적합이 일반화를 무너뜨리는 수준은 아닙니다.
- `calibration_error` 0.0970, `brier` 0.0863: 예측 확률은 평균적으로 약 9.7%p 어긋나 있습니다. 소수 클래스를 9배로 가중한 결과로 확률이 위로 밀린 형태이며, 이 실행 환경에는 재보정 레버가 없습니다. `balanced_accuracy`·`roc_auc`·`pr_auc`는 이 왜곡의 영향을 받지 않지만, 이 모델의 출력값을 "확률"로 그대로 쓰는 것은 부적절합니다.
- `plan.unsupported_claims`에 `feature_engineering`이 기록되어 있습니다. 해당 계획 문구를 읽어보면 missingness 컬럼을 **쓰지 않겠다는 이유를 설명**한 대목("no missingness columns … the 'unknown' text markers are already one-hot levels the trees can split on")에서 문자열 검사가 걸린 것으로, 계획이 사용 불가능한 기능에 의존한 흔적은 없습니다. 따라서 이 시도가 이 항목 때문에 잃은 것은 없으며, 파생 피처 생성은 "실패한 것"이 아니라 아직 만들어지지 않은 역량으로 아래 제안에 둡니다.

## 다음 단계 제안

1. **`unknown` 센티넬을 카드 단계에서 해소한 뒤 `impute: none`을 시험한다.** 카드는 `missing.overall_rate`=0.0으로 보고하지만, caveats는 V16 81.8%, V9 28.8%, V4 4.1%, V2 0.6%가 결측 표기로 흔히 쓰이는 `'unknown'` 문자열이며 **변환되지 않은 상태로 실측치처럼 집계되었다**고 명시합니다. 이 상태에서는 `missing_indicator`/`missing_count`가 전부 0이 되어 아무 정보도 추가하지 못합니다(실행기 문서도 NaN 분기가 있는 계열에서는 중복이라고 못박습니다). 따라서 권고는 인디케이터 추가가 아니라 **caveat 자체의 해소** — 파일/카드 수준에서 `'unknown'`을 실제 NaN으로 변환하는 것입니다. 이것이 완료되면 `hist_gbdt`/`xgboost`의 `impute: none`(NaN 직접 분기)과 missingness 컬럼이 처음으로 유의미하게 검증 가능해집니다. 단, 그 레버의 효과 크기는 문서상 안정적으로 확립되어 있지 않으므로(랜덤 split 0.0097 roc_auc vs 연속 split +0.0011, 구간이 0을 포함) 기대값을 미리 못박지 말고 측정 대상으로만 다뤄야 합니다.
2. **V16(그리고 V9)을 카드 수준에서 재검토한다.** V16은 81.8%가 `'unknown'` 단일 값이므로 사실상 거의 상수에 가깝고, 현재 one-hot 레벨로 모델에 들어가 있습니다. 실행기는 컬럼을 드롭할 수 없으므로(“a column you want out has to leave the card, not the plan”), 이 결정은 데이터 준비 쪽에서 해야 합니다. 이 컬럼을 실제 결측으로 재해석할지 제거할지 확정하는 것이 1번의 전제이며, 동시에 현재 baseline·target_corr 수치의 해석 신뢰도를 회복시킵니다.
3. **추가 예산은 ranking 축에 쓰고, 해상도 이하의 차이를 개선으로 읽지 않는다.** `cut_headroom`이 0.0079뿐이므로 `class_weight`/`scale_pos_weight` 스윕에서 얻을 수 있는 최대치는 이미 매우 작습니다. 남은 여지가 있는 축은 ranking이며, 실행기 문서의 관측된 크기는 동일 계열 내 하이퍼파라미터 재조정 0.0032 roc_auc, 계열 교체 0.0022~0.0077 roc_auc이고 paired 해상도는 0.003~0.006입니다. 즉 `xgboost`로의 계열 교체 1~2회 정도가 합리적인 시도이지만, 검증 CI 폭(0.8593~0.8815, 약 ±0.011)보다 작은 `balanced_accuracy` 변화는 개선으로 보고하지 말아야 합니다.
4. **확률값을 소비하는 용도가 있다면 루프 밖에서 처리한다.** `calibration_error` 0.0970은 이 모델의 확률이 그대로 쓰기에 부적합함을 보여주지만, 이 실행 환경에는 재보정도 임계값 탐색도 없습니다(`CalibratedClassifierCV`·임계값 스윕은 실행기가 수행하지 않음). 확률 기반 의사결정이 필요하다면 보정을 파이프라인 외부 단계로 별도 설계하고, 그 전까지는 이 모델을 순위·이진 판정용으로만 사용하십시오.