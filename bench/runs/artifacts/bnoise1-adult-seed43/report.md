## 요약

목표(balanced_accuracy ≥ 0.9194)는 **달성하지 못했습니다**. 3회 시도(예산 5회) 중 최고는 iteration 1의 `hist_gbdt`로 검증 balanced_accuracy = **0.8442** (95% CI 0.8362~0.8517)이며, 목표선까지 0.0752 부족합니다. 한 번도 사용되지 않은 최종 테스트 20%에서 같은 모델은 **0.8432** (95% CI 0.8354~0.8515)로, 이것이 이 실행이 실제로 입증한 수치입니다. 연속 미개선으로 루프는 3회에서 조기 종료되었고, critic 진단은 두 번 모두 동일한 결론(`wrong_model_family`, 즉 랭킹 축의 한계)이었습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `class_weight='balanced'`, `early_stopping=False` / preprocessing: `impute='none'`, `scale=False` | balanced_accuracy **0.8442** (CI 0.8362~0.8517), roc_auc 0.9272, cut_headroom 0.000936, train_val_gap 0.0444 — **최고** | `wrong_model_family`: 운영점은 이미 최적(headroom이 남은 거리의 1.2%), 남은 부족분은 랭킹(KS 0.6902 vs 요구 0.8388). 다음은 계열을 `xgboost`로 교체해 상한을 확정 |
| 2 | `xgboost` | `n_estimators=800`, `learning_rate=0.05`, `max_depth=8`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_lambda=2.0`, `scale_pos_weight=3.18` / preprocessing: `impute='none'`, `scale=False` | balanced_accuracy 0.8383 (CI 0.8299~0.8463), roc_auc 0.9226, cut_headroom 0.002225, train_val_gap 0.0886 | `wrong_model_family`: 계열 교체로 랭킹이 오르지 않음(짝지은 Δ = -0.0059, CI -0.0114~-0.0005). 남은 미측정 축은 컬럼 정보(impute/missing_count) 하나뿐 — 상한 확정 측정으로 사용하라 |
| 3 | `hist_gbdt` | iteration 1과 동일한 하이퍼파라미터 / preprocessing: `impute='median'`, `scale=False`, `missing_count=True` | balanced_accuracy 0.8442 (CI 0.8362~0.8517) — iteration 1과 **모든 지표가 자리수까지 동일** | (없음 — 평가 직후 루프 종료) |

## 최고 성능 구성

**iteration 1 — `hist_gbdt`** (아래 값은 히스토리에 기록된, 실제로 빌드된 설정입니다)

- `hyperparams`: `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `l2_regularization=1.0`, `class_weight='balanced'`, `early_stopping=False`
- `preprocessing`: `impute='none'`, `scale=false`, `missing_indicator=false`, `missing_count=false`
- `dropped_hyperparams`: 없음 (요청한 설정이 모두 그대로 적용됨)
- 프로토콜: stratified 3-way split, seed 43, train 60% / val 20% / test 20%. 학습 시간 13.243초.

| 지표 | 값 |
|---|---|
| balanced_accuracy (검증) | **0.8442** (95% CI 0.8362~0.8517) |
| balanced_accuracy (**최종 테스트, 한 번도 쓰이지 않은 20%**) | **0.8432** (95% CI 0.8354~0.8515) |
| recall / specificity | 0.8567 / 0.831674 |
| roc_auc / pr_auc | 0.9272 / 0.8288 |
| accuracy / f1 / precision | 0.8376 / 0.7163 / 0.6154 |
| balanced_accuracy_at_best_cut / cut_headroom | 0.8451 / 0.000936 |
| brier / calibration_error | 0.108126 / 0.098609 |
| train_balanced_accuracy / train_val_gap | 0.8886 / 0.044433 |

검증 0.8442 → 테스트 0.8432의 차이 **+0.0010**이 선택 편향(루프가 검증 숫자만 보고 최고 시도를 골랐기 때문에 생기는 낙관 편향)의 크기입니다. 다만 검증값이 테스트 CI(0.8354~0.8515) 안에 들어가므로, 이 행들만으로는 이 차이를 0과 구분할 수 없습니다. 어느 쪽을 쓰더라도 목표선 0.9194에는 0.075 이상 부족합니다.

## 원인 분석

**1) 부족분은 전부 랭킹 축에 있습니다.** critic은 2회 모두 `wrong_model_family`로 같은 진단을 냈고, 근거도 같은 숫자였습니다. iteration 1에서 `cut_headroom=0.000936`, `balanced_accuracy_at_best_cut=0.8451` — 즉 어떤 임계값을 골라도 남은 0.0752의 1.2%만 살 수 있습니다. recall 0.8567 vs specificity 0.831674는 이미 거의 대칭이므로 `class_weight`/`scale_pos_weight`를 더 밀어도 두 값의 평균은 오르지 않습니다. 운영점 레버는 소진되었습니다.

**2) 목표선 자체가 이 랭킹으로는 도달 불가 영역입니다.** 목표 메타데이터가 이미 `exceeds_ranking_ceiling: true`를 기록하고 있습니다: baseline의 `ranking_ceiling`은 0.8266, 목표 0.9194가 요구하는 KS는 0.8388인데, 이번 실행의 최고 KS는 2×0.8451−1 = 0.6902로 0.1486 미달(`ks_shortfall` 0.1857 대비 개선은 있었지만 여전히 큰 격차)입니다. 최적 컷에서조차 0.8451이므로 남은 0.0743은 순전히 "행을 더 잘 정렬해야" 메워집니다.

**3) 계열 교체로 살 수 있는 폭은 부족분의 한 자릿수 %입니다.** iteration 1(0.8442, CI 0.8362~0.8517)과 iteration 2(0.8383, CI 0.8299~0.8463)의 구간은 겹치므로 두 시도는 이 데이터가 (짝짓지 않은 비교로는) 분리하지 못하는 시도입니다. critic이 계산한 짝지은 Δ = −0.0059 (CI −0.0114~−0.0005)는 소폭 하락 쪽을 가리키지만, 방향이 어느 쪽이든 크기가 0.006 수준이라 남은 0.075와는 자릿수가 다릅니다. 두 계열의 `balanced_accuracy_at_best_cut`은 0.8405~0.8451(폭 0.0046)에 모여 있습니다 — 부족분이 이 폭의 약 16배입니다. `dropped_hyperparams`는 두 시도 모두 비어 있고, 학습 시간은 10~15초로 600초 한도에서 한참 여유가 있으므로 용량·자원·설정 문제는 아닙니다. xgboost에서 train_val_gap이 0.0444→0.0886으로 커지면서 검증은 내려간 것은, 추가 용량이 분산으로만 갔다는 것 — 즉 더 큰 트리로는 이 랭킹 상한을 넘지 못한다는 신호입니다.

**4) 컬럼 정보 레버는 여기서 정확히 0을 벌었습니다.** iteration 3은 iteration 1과 하이퍼파라미터가 동일하고 preprocessing만 `impute='median'` + `missing_count=True`로 바꿨는데, balanced_accuracy·roc_auc·brier까지 **모든 지표가 자리수까지 동일**했습니다. 결측은 3개 컬럼(`workclass` 0.0573, `occupation` 0.0575, `native-country` 0.0175)에만 있고 전체 결측률은 0.0095, 그리고 이들은 모두 one-hot 인코딩되는 범주형입니다 — 이 조건에서 median 대체와 결측 개수 컬럼은 모델에 새 정보를 주지 못했습니다. 실행기 문서가 경고한 대로 `missing_count`는 이 파일에서 살 것이 없었고, 이 결과가 그것을 확인했습니다.

**5) `unsupported_claims`에 대해.** iteration 1과 3의 계획에 `feature_engineering` 플래그가 떴지만, 확인 가능한 iteration 1의 계획 본문은 one-hot 인코딩 경로를 *언급*했을 뿐 파생 피처에 의존하지 않았고, 요청한 하이퍼파라미터·전처리가 모두 그대로 적용되었습니다(`dropped_hyperparams` 비어 있음). 따라서 이 플래그 때문에 시도가 무언가를 잃은 것은 아닙니다 — 이 점수는 이 실행기가 낼 수 있는 값 그대로이고, 없는 능력(피처 생성)은 아래 "다음 단계"에서 **만들어야 할 것**으로 다룹니다. (iteration 3의 계획 본문은 기록에 남아 있지 않아 그 문장까지는 확인할 수 없습니다.)

또한 `calibration_error=0.0986`은 확률이 평균 약 10%p 어긋나 있음을 말합니다. balanced_accuracy는 랭킹+컷의 함수라 여기에 직접 영향받지 않지만, 이 모델의 확률값을 그대로 의사결정에 쓰기에는 부적절합니다(실행기에는 재보정 레버가 없습니다).

## 다음 단계 제안

기록된 데이터 주의사항은 없으므로(“없음”), 아래 제안을 무효화하는 caveat은 없습니다. 다만 위 4)에서 확인된 대로 결측 관련 레버는 이 파일에서 이미 0으로 측정되었으니 재시도 대상이 아닙니다.

1. **목표선 0.9194를 먼저 재검토하십시오 (최우선).** 목표 메타데이터가 스스로 `exceeds_ranking_ceiling: true`, `required_ks=0.8388`을 기록하고 있고, 두 계열의 최적 컷 성능이 0.8405~0.8451에 갇혀 있습니다. 현재 피처 집합·현재 실행기 레버 범위(계열 폭 0.0046, cut_headroom 0.0022) 안에서는 남은 0.075를 살 수 있는 레버가 없습니다. 파생된 threshold를 그대로 두고 예산을 더 태우기보다, (a) 목표를 랭킹 상한을 반영한 값으로 재도출하거나 (b) 아래 2번처럼 피처 축을 열어야 합니다.
2. **피처 생성을 실행기 밖에서 열어 카드에 넣으십시오.** 부족분이 랭킹 축에 있고 계열/하이퍼파라미터/결측 레버가 모두 소진된 상태에서, 아직 한 번도 측정되지 않은 축은 "입력 컬럼 자체"입니다. 실행기는 파생·인코딩·삭제를 하지 못하므로(`capital-gain`/`capital-loss`의 high skew 변환, `age`×`education-num` 같은 상호작용, `native-country`/`occupation`의 목표 기반 재그룹화 등) 이런 컬럼은 **카드 생성 단계에서 미리 만들어 넣어야** 합니다. 이것이 열리면 roc_auc 0.9272 위쪽이 처음으로 측정 가능한 대상이 됩니다. 단, 이 실행에서 계열 교체가 산 폭(roc_auc 0.0022~0.0077)을 근거로 피처가 0.075를 살 것이라고 미리 약속할 수는 없습니다 — 이것은 아직 **미측정 축**이라는 뜻이며, 크기 주장이 아닙니다.
3. **남은 예산으로는 "상한 확정" 측정만 하십시오.** 예산이 더 있다면 같은 파이프라인(`impute='none'`, `scale=False`)에서 `random_forest` 등 아직 안 본 트리 계열 1회, 그리고 `hist_gbdt`의 용량을 반대 방향으로(예: `max_leaf_nodes` 축소 + `learning_rate` 하향 + `max_iter` 상향) 1회 돌려 `roc_auc`와 `balanced_accuracy_at_best_cut`이 0.9226~0.9272 / 0.8405~0.8451 밖으로 나가는지만 확인하십시오. 목적은 목표 달성이 아니라 "이 피처 집합의 랭킹 상한"을 숫자로 못 박아 2번의 필요성을 확정하는 것입니다.
4. **다음 두 레버에는 예산을 쓰지 마십시오.** (i) 임계값/`class_weight` 재조정 — 최고 시도의 `cut_headroom`이 0.000936이고 recall 0.8567 vs specificity 0.831674로 이미 대칭입니다. (ii) 결측 관련 컬럼(`missing_indicator`, `missing_count`)과 대체 전략 변경 — iteration 3이 iteration 1과 비트 단위로 동일한 결과를 냈고, 결측은 범주형 3개 컬럼·전체 0.95%에 불과합니다. 확률값을 실제로 써야 한다면 재보정은 실행기 밖 작업으로 분리하십시오(`calibration_error=0.0986`, `brier=0.1081`).