# AutoML 최종 리포트 — `speeddating` / `balanced_accuracy`

## 요약

목표는 달성되었습니다. 목표 기준선은 `balanced_accuracy` ≥ 0.7712였고, 2회 시도 중 iteration 2의 `hist_gbdt`가 검증 `balanced_accuracy` **0.7878** (95% CI 0.7593~0.8111)로 이를 넘었습니다. 루프 시작 전에 떼어 두고 한 번도 사용하지 않은 최종 테스트 20%에서의 점수는 **0.8008** (95% CI 0.7753~0.8273)로, 검증 점수보다 오히려 높았습니다. 총 5회 예산 중 2회를 사용했고, critic 진단은 1회 발생했습니다 (iteration 1에 대한 진단).

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `class_weight='balanced'`, `early_stopping=False` / `impute: none`, `scale: False`, `missing_count: True` | `balanced_accuracy`=0.6991 (CI 0.6679~0.7278) — 기준선 0.7712 미달 (-0.0721). `roc_auc`=0.8755, `balanced_accuracy_at_best_cut`=0.7984, `cut_headroom`=0.0993, `recall`=0.4710 vs `specificity`=0.9271, `train_val_gap`=0.3007 | `failure_type: data_issue`. 순위(ranking) 축은 이미 충분(best cut 0.7984 > 0.7712)하며 부족분 0.0721의 138%가 `cut_headroom`에 남아 있음 → 문제는 operating point. `class_weight='balanced'`(5.07:1)를 명시적 `{"0":1,"1":12}`로 교체 권고 |
| 2 | `hist_gbdt` | `max_iter=220`, `learning_rate=0.06`, `max_leaf_nodes=15`, `min_samples_leaf=40`, `l2_regularization=3.0`, `class_weight={"0":1.0,"1":12.0}`, `early_stopping=False` / `impute: none`, `scale: False`, `missing_count: True` | **`balanced_accuracy`=0.7878 (CI 0.7593~0.8111) — 기준선 통과.** `recall`=0.7862 / `specificity`=0.7893, `roc_auc`=0.8720, `pr_auc`=0.6012, `cut_headroom`=0.0111, `train_val_gap`=0.1302 | — (목표 달성으로 루프 종료, 진단 대상 아님) |

## 최고 성능 구성

**iteration 2 · `hist_gbdt`** — 아래 값은 실행된(applied) 설정을 히스토리에서 그대로 인용한 것입니다.

`hyperparams`:
```json
{
  "max_iter": 220,
  "learning_rate": 0.06,
  "max_leaf_nodes": 15,
  "min_samples_leaf": 40,
  "l2_regularization": 3.0,
  "class_weight": {"0": 1.0, "1": 12.0},
  "early_stopping": false
}
```
`preprocessing` (실제 적용됨):
```json
{"impute": "none", "scale": false, "missing_indicator": false, "missing_count": true}
```
`dropped_hyperparams`는 비어 있고 `status: ok`, `train_time_sec`=6.095입니다. 즉 결측치는 임퓨테이션 없이 `hist_gbdt`가 NaN 분기로 직접 처리했고(결측 열 60개, 최대 결측률 0.7852), 스케일링은 없으며 행 단위 결측 개수 열(`missing_count`) 하나만 추가되었습니다.

검증 성능 (validation 20%):

| metric | 값 |
|---|---|
| `balanced_accuracy` | **0.7878** (95% CI 0.7593~0.8111) |
| `recall` / `specificity` | 0.7862 / 0.7893 |
| `precision` / `f1` | 0.4238 / 0.5508 |
| `accuracy` | 0.7888 |
| `roc_auc` | 0.8720 |
| `pr_auc` / `average_precision` | 0.6012 |
| `balanced_accuracy_at_best_cut` / `cut_headroom` | 0.7989 / 0.0111 |
| `brier` / `calibration_error` | 0.1428 / 0.1645 |
| `train_balanced_accuracy` / `train_val_gap` | 0.9179 / 0.1302 |

**최종 held-back 측정:** 같은 모델을 첫 fit 이전에 분리해 두고 어떤 결정에도 쓰지 않은 테스트 20%에서 한 번 채점한 결과 `balanced_accuracy` = **0.8008** (95% CI 0.7753~0.8273). 검증 0.7878 대비 **-0.0131**의 차이가 있는데, 이 차이가 이 루프의 선택 편향(selection bias) 크기입니다 — 루프는 검증 숫자만 보고 최선의 시도를 골랐습니다. 다만 검증 점수가 테스트 CI(0.7753~0.8273) 안에 들어 있으므로, 이 행 수로는 그 차이를 0과 구분할 수 없습니다. 이 실행이 실제로 입증한 숫자는 테스트의 0.8008이며, 그 값 역시 기준선 0.7712를 넘습니다 (CI 하한 0.7753도 기준선 위).

## 원인 분석

critic 진단은 1회 존재하며, 그 진단의 내용과 이후 결과가 정확히 일치합니다.

- **두 시도는 이 데이터로 구분 가능합니다.** iteration 1의 CI(0.6679~0.7278)와 iteration 2의 CI(0.7593~0.8111)는 겹치지 않으므로, +0.0887의 개선은 리샘플 노이즈가 아닙니다.
- **개선은 전부 operating point 축에서 나왔습니다.** 순위 축 지표는 사실상 움직이지 않았습니다: `roc_auc` 0.8755 → 0.8720, `balanced_accuracy_at_best_cut` 0.7984 → 0.7989. 이 크기(0.0035, 0.0005)는 이 하네스에서 언급된 roc_auc 차이의 해상도(약 0.003~0.006)보다 작거나 그 수준이므로, 순위 품질은 두 시도가 동일하다고 읽어야 합니다. 반면 `cut_headroom`은 0.0993 → 0.0111로 줄었고, `recall`/`specificity`는 0.4710/0.9271 → 0.7862/0.7893으로 대칭에 가까워졌습니다. 즉 iteration 1의 부족분은 모델이 행을 잘못 정렬해서가 아니라, `predict()`의 기본 컷이 `class_weight='balanced'`(클래스 빈도 5.07:1에 고정된 한 점)에서 다수 클래스 쪽에 치우쳐 있었기 때문이며, 이를 `{"0":1,"1":12}`로 옮긴 것이 곧 점수였습니다. critic이 "부족분 0.0721의 138%가 `cut_headroom`에 있다"고 계량한 그대로입니다.
- **용량(capacity) 조정은 부수적이었습니다.** 같은 시도에서 `max_iter` 400→220, `max_leaf_nodes` 31→15, `min_samples_leaf`=40, `l2_regularization` 1.0→3.0이 함께 들어가 `train_val_gap`이 0.3007→0.1302로 줄었습니다. 그러나 이 변화가 순위를 개선했다는 증거는 없습니다 (`roc_auc`는 오히려 0.0035 낮고 이는 구분 불가 범위). 두 변경이 한 시도에 묶여 있어 용량 축의 기여를 따로 값 매길 수는 없으며, 관측 가능한 효과는 과적합 지표의 감소뿐입니다.
- **남은 제약은 확률의 품질입니다.** 양성 가중치를 12로 올린 대가로 `brier` 0.1036→0.1428, `calibration_error` 0.0547→0.1645로 확률 보정이 크게 나빠졌습니다. 순위 지표(`roc_auc`, `pr_auc`)는 보정과 무관하므로 목표 지표에는 영향이 없지만, 이 모델의 출력 확률을 그대로 "매치 확률"로 읽으면 평균 16pp 어긋납니다. 이 하네스에는 재보정 레버가 없습니다.
- **`unsupported_claims`에 대해:** iteration 1의 계획에 `feature_engineering`이 표시되어 있으나, 해당 iteration의 계획 원문은 이 기록에 남아 있지 않아 계획이 그 기능에 *의존*했는지 단지 언급했는지 확인할 수 없습니다. 확인 가능한 사실은 실제 적용된 `preprocessing`이 `impute: none` + `missing_count`뿐이고 파생 열은 하나도 생성되지 않았다는 것 — 즉 iteration 1의 0.6991은 파생 피처 없이 카드의 열만으로 얻은 숫자입니다. 따라서 이를 "무엇을 잃었다"로 해석하지 않고, 아래 제안에서 "만들어야 할 것"으로 다룹니다.
- 남은 상한선: 이 실행에서 순위 축의 최선은 `balanced_accuracy_at_best_cut` 0.7989이고 현재 컷은 그 0.0111 아래입니다. 즉 **operating point 축은 거의 소진되었고**, 추가 개선은 순위 축(모델 패밀리 또는 피처)에서만 나올 수 있습니다.

## 다음 단계 제안

이 데이터셋에는 별도로 기록된 caveat이 없으므로(“없음”), 아래 제안을 제약하는 것은 executor의 기능 한계뿐입니다.

1. **남은 예산은 순위(ranking) 축에만 쓰십시오 — `class_weight` 미세조정은 하지 마십시오.** `cut_headroom`이 0.0111로, 검증 CI 폭(0.7593~0.8111, 약 ±0.026)보다 작습니다. 가중치를 8이나 10으로 바꿔 얻을 최대 이득은 이 슬라이스에서 측정 불가한 크기이므로 한 iteration을 쓸 가치가 없습니다. 대신 목표는 `roc_auc` 0.8720 / `balanced_accuracy_at_best_cut` 0.7989를 올리는 것입니다.
2. **패밀리 교체를 1회 시도: `xgboost` + `impute: none`, `scale_pos_weight`를 12 부근에서 시작.** 파이프라인을 고정한 상태의 트리 패밀리 교체는 이 하네스의 관측에서 `roc_auc` 0.0022~0.0077 범위를 움직였습니다 — 확실한 이득은 아니지만, 현재 유일하게 남은 순위 축 레버입니다. `impute: none`을 유지해 결측률 0.7852인 `expected_num_interested_in_me` 등 60개 열을 NaN 분기로 그대로 다루고, `missing_indicator`는 **넣지 마십시오** (`impute: none`과 함께라면 예측이 비트 단위로 동일하다는 것이 이미 확인된 사항입니다).
3. **파생 피처 능력은 executor 밖에서 만들어야 합니다.** 이 도메인에서 가장 유망한 신호는 열 간 조합(예: 자기 평가 vs 상대 평가 차이, `like`/`guess_prob_liked` 상호작용, `attractive` 계열의 self-partner 격차)인데, executor는 비율·차이·상호작용을 만들 수 없고 열 삭제도 못 합니다. 이런 열은 **카드(입력 CSV) 단계에서 미리 계산해 넣어야** 하며, 그때 순위 축이 0.7989를 넘어설 여지가 생깁니다. 고카디널리티로 드롭된 `field`를 저카디널리티 그룹으로 사전 인코딩해 다시 넣는 것도 같은 경로입니다.
4. **확률을 소비할 계획이라면 보정을 루프 밖에서 처리하십시오.** 최고 구성의 `calibration_error`=0.1645, `brier`=0.1428은 무거운 양성 가중치의 직접적 대가이고, 이 하네스에는 재보정 레버가 없습니다. 순위만 필요하다면 무시해도 되지만, "매치 확률" 자체를 보고할 용도라면 (a) 별도 홀드아웃에서의 사후 보정 단계를 파이프라인 밖에 추가하거나 (b) 가중치를 낮춘 구성을 별도로 학습해 확률 전용으로 쓰는 두 갈래 중 하나를 명시적으로 선택해야 합니다.
5. **(선택) 단일 분할 의존도 확인.** 모든 점수는 seed 42의 고정 20% 검증/20% 테스트 한 번에서 나왔고 교차검증은 없습니다. 최종 테스트 0.8008(CI 0.7753~0.8273)은 기준선 위에 안전하게 있지만, 배포 판단이 필요하다면 다른 seed로 동일 구성을 재학습해 점수 분포의 폭을 확인하는 것이 가장 값싼 검증입니다.