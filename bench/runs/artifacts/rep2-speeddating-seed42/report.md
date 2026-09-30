## 요약

목표(`balanced_accuracy` ≥ 0.7514)는 **달성**했습니다. 3회 시도 중 마지막 iteration 3의 `hist_gbdt`가 검증 슬라이스에서 `balanced_accuracy` **0.7570** (95% CI 0.7277~0.7861)을 기록해 임계값을 넘겼고, 한 번도 사용되지 않은 최종 테스트 20%에서도 **0.7592** (95% CI 0.7315~0.7916)로 재확인됐습니다. 5회 예산 중 3회를 썼고, 세 시도 모두 동일한 모델 패밀리·동일한 파이프라인에서 `class_weight` 하나만 움직인 결과입니다(1:3.5 → 1:6.0 → 1:9.0). 참고로 카드의 baseline(logreg, median impute + scale)은 0.6685 (CI 0.6425~0.6950)였습니다.

## 시도별 경과

| iteration | model | 주요 하이퍼파라미터 | 결과 | critic 진단 |
|---|---|---|---|---|
| 1 | `hist_gbdt` | `max_iter=400`, `learning_rate=0.06`, `max_leaf_nodes=31`, `min_samples_leaf=20`, `l2_regularization=1.0`, `early_stopping=false`, `class_weight={"0":1.0,"1":3.5}` | `balanced_accuracy` 0.6932 (CI 0.6619~0.7211), 목표 미달(-0.0582). `recall` 0.4493 vs `specificity` 0.9371, `roc_auc` 0.8719, `cut_headroom` 0.1020, `train_val_gap` 0.3067 | `data_issue` — 랭킹 축은 이미 충분(`balanced_accuracy_at_best_cut` 0.7952). 남은 격차는 전부 operating point 문제이며 positive class에 **더** 큰 가중치가 필요. 파이프라인·패밀리는 그대로 두고 `class_weight`만 `balanced`(≈5.07) 방향으로 올릴 것 |
| 2 | `hist_gbdt` | `max_iter=300`, `learning_rate=0.06`, `max_leaf_nodes=24`, `min_samples_leaf=40`, `l2_regularization=3.0`, `early_stopping=false`, `class_weight={"0":1.0,"1":6.0}` | `balanced_accuracy` 0.7402 (CI 0.7068~0.7707), 목표 미달(-0.0112, 임계값이 CI 안에 위치). `recall` 0.5833 vs `specificity` 0.8971, `roc_auc` 0.8748, `pr_auc` 0.5983, `cut_headroom` 0.0632, `train_val_gap` 0.2504 | `data_issue` — iteration 1 대비 paired Δ +0.0470 (CI +0.0274~+0.0686)로 가중치 레버가 실제로 작동했고 아직 소진되지 않음. `recall`이 여전히 `specificity`보다 낮아 교차점을 지나지 않았으므로 `class_weight`를 1:9.0으로만 올리고 나머지는 전부 고정 |
| 3 | `hist_gbdt` | `max_iter=300`, `learning_rate=0.06`, `max_leaf_nodes=24`, `min_samples_leaf=40`, `l2_regularization=3.0`, `early_stopping=false`, `class_weight={"0":1.0,"1":9.0}` | `balanced_accuracy` **0.7570** (CI 0.7277~0.7861) — 목표 달성. `recall` 0.6304, `specificity` 0.8836, `roc_auc` 0.8730, `pr_auc` 0.6026, `cut_headroom` 0.0425, `train_val_gap` 0.2281 | 없음(목표 달성으로 루프 종료 — 이 시도는 진단 대상이 아님) |

## 최고 성능 구성

**iteration 3 / `hist_gbdt`** — 아래 값은 히스토리에 기록된 적용값(executor가 실제로 만든 estimator·파이프라인)입니다.

```
model: hist_gbdt
hyperparams:
  max_iter: 300
  learning_rate: 0.06
  max_leaf_nodes: 24
  min_samples_leaf: 40
  l2_regularization: 3.0
  early_stopping: false
  class_weight: {"0": 1.0, "1": 9.0}
  random_state: 42
preprocessing (applied):
  impute: none          # 트리가 NaN을 직접 분기 (결측 컬럼 60개, 최대 결측률 0.7852)
  scale: false
  missing_indicator: false
  missing_count: true
dropped_hyperparams: []
train_time_sec: 7.903
```

| 구분 | `balanced_accuracy` | 95% CI |
|---|---|---|
| 검증 20% (선택에 사용된 점수) | 0.7570 | 0.7277~0.7861 |
| **최종 테스트 20% (한 번도 쓰이지 않은 행)** | **0.7592** | 0.7315~0.7916 |

두 값의 차이는 −0.0022(검증 대비 테스트)이며, 이것이 **선택 편향의 크기**입니다. 루프는 검증 숫자만 보고 최적 시도를 골랐고, 이 모델에 대해 실제로 입증된 것은 테스트 점수 쪽입니다. 다만 검증 점수가 테스트 CI 안에 들어 있어, 이 행 수로는 두 값의 차이를 0과 구분할 수 없습니다. 임계값 0.7514는 테스트 점추정(0.7592)보다 낮지만 테스트 CI 하단(0.7315)보다는 높다는 점도 함께 읽어야 합니다 — 목표는 규정된 방식(검증 슬라이스 기준)으로 달성됐고 테스트에서도 점추정이 임계값을 넘었지만, 이 크기의 슬라이스에서 여유가 통계적으로 넉넉하다고 말할 수는 없습니다.

검증 슬라이스의 그 외 지표: `f1` 0.5677, `accuracy` 0.8419, `precision` 0.5163, `recall` 0.6304, `specificity` 0.8836, `roc_auc` 0.8730, `pr_auc`/`average_precision` 0.6026, `brier` 0.1117, `calibration_error` 0.0716, `balanced_accuracy_at_best_cut` 0.7995, `cut_headroom` 0.0425, `train_val_gap` 0.2281.

## 원인 분석

critic 판정은 2건 존재하며 두 판정이 완전히 같은 결론을 냈습니다: 성능을 제한한 것은 **랭킹 품질이 아니라 결정 경계(operating point)의 위치**였습니다.

- 랭킹 축은 세 시도 내내 거의 움직이지 않았습니다: `roc_auc` 0.8719 → 0.8748 → 0.8730, `pr_auc` 0.5918 → 0.5983 → 0.6026. 이 폭은 문서가 밝힌 랭킹 축의 paired 해상도(약 0.003~0.006)와 같은 규모라, 세 시도의 랭킹은 이 데이터로 서로 구분되지 않습니다. 반면 `balanced_accuracy`는 0.6932 → 0.7570으로 0.0638 움직였고, 유일하게 바뀐 레버는 `class_weight`였습니다. 즉 관측된 개선은 사실상 전부 operating point 축에서 나왔습니다.
- 그 방향이 명확했던 근거는 `recall`과 `specificity`의 비대칭입니다: 1:3.5에서 0.4493 vs 0.9371(격차 0.4879), 1:6.0에서 0.5833 vs 0.8971(0.3138), 1:9.0에서 0.6304 vs 0.8836(0.2532). `balanced_accuracy`가 이 둘의 평균이므로, `recall`이 계속 낮은 쪽에 머무는 동안에는 positive 가중치를 더 올리는 것이 항상 옳은 방향이었고 세 점 모두 교차점의 같은 쪽에 있습니다. 동시에 `cut_headroom`이 각 시도에서 남은 격차보다 훨씬 컸다는 점(iteration 1: 0.1020 vs 필요 0.0582, iteration 2: 0.0632 vs 필요 0.0112)이 "격차가 랭킹이 아니라 컷에 있다"는 판정을 수치로 뒷받침했습니다.
- **구간이 겹치는 이동은 이야기로 만들지 않습니다.** iteration 2(CI 0.7068~0.7707)와 iteration 3(CI 0.7277~0.7861)은 구간이 크게 겹치므로 이 슬라이스만으로는 서로 구분되지 않습니다 — 0.7402 → 0.7570의 +0.0168을 "확실한 개선"으로 읽을 근거는 없습니다. 반대로 iteration 1(CI 0.6619~0.7211)과 iteration 3은 구간이 겹치지 않고, critic이 계산한 iteration 1→2 paired Δ +0.0470(CI +0.0274~+0.0686)도 0을 포함하지 않으므로, **확실하게 말할 수 있는 것은 1:3.5에서 벗어난 이동이 실제 개선이었다는 점까지**입니다. 1:6.0과 1:9.0 사이의 미세 조정은 이 데이터로는 판정 불가입니다.
- `train_val_gap`은 0.3067 → 0.2504 → 0.2281로 계속 컸고 `train_balanced_accuracy`는 0.9851까지도 높았지만, 이 과적합 신호는 랭킹을 훼손하지 않았고(`roc_auc` 유지, `brier` 0.1037~0.1117) critic도 명시했듯 컷을 어느 방향으로 옮길지에 대해 아무 정보도 주지 않습니다. 즉 규제 강화는 이번 격차의 원인 후보가 아니었습니다.
- 비용은 제약이 아니었습니다: 학습 시간 최대 11.3초(예산 600초), `dropped_hyperparams`는 세 시도 모두 비어 있어 실행기가 거부한 설정도 없습니다. `unsupported_claims`도 모두 비어 있으므로, 사용 불가능한 기능에 의존해 실패한 시도는 없습니다.

한계로 남은 것: `cut_headroom`이 마지막에도 0.0425였고 `balanced_accuracy_at_best_cut`은 0.7995입니다. 이 모델의 랭킹은 최적 컷에서 약 0.80까지 허용하지만, 이 실행기에는 임계값 탐색 레버가 없어 `class_weight`로만 그 지점에 접근할 수 있었습니다. 또 `calibration_error`가 0.0577 → 0.0542 → 0.0716으로 마지막 시도에서 다시 커졌습니다 — 가중치를 강하게 밀면 확률의 절대 수준은 나빠진다는 뜻이며, 확률값 자체를 쓰려면 별도 고려가 필요합니다.

## 다음 단계 제안

1. **`class_weight`를 1:9.0 주변에서 한두 점 더 브래킷해 교차점을 확인.** 1:9.0에서도 `recall` 0.6304 < `specificity` 0.8836이라 교차점을 아직 지나지 않았고, `cut_headroom` 0.0425가 남아 있습니다. 다른 모든 값을 iteration 3과 동일하게 고정한 채 1:12, 1:15를 추가로 재보면 `balanced_accuracy`의 정점을 실제로 넘겼는지 확인할 수 있습니다. 단, iteration 2와 3의 CI가 겹친 사실이 보여주듯 이 슬라이스에서 0.02 미만의 차이는 판정 불가이므로, **비대칭(`recall` vs `specificity`)이 뒤집히는지**를 판단 기준으로 삼고 소수점 셋째 자리 개선을 성과로 읽지 마십시오.
2. **랭킹 축을 한 번 별도로 검증(패밀리 스왑).** 남은 개선 여지는 `balanced_accuracy_at_best_cut` 0.7995가 상한을 시사하는데, 이는 랭킹 축의 문제입니다. 파이프라인(`impute: none`, `scale: false`, `missing_count: true`)과 `class_weight`를 iteration 3에 고정한 채 `xgboost`를 한 번 돌려 `roc_auc`/`pr_auc`가 움직이는지만 보십시오. 문서가 기록한 패밀리 스왑의 실측 폭은 0.0022~0.0077 `roc_auc`이고 paired 해상도가 0.003~0.006이므로, **기대치는 낮게** 잡고 "구분되지 않으면 랭킹 축은 이 데이터에서 포화"라는 결론을 얻는 용도로 쓰는 것이 맞습니다.
3. **확률 캘리브레이션 악화를 의사결정 요구사항과 대조.** iteration 3에서 `calibration_error` 0.0716, `brier` 0.1117로 iteration 2보다 나빠졌습니다. 이 실행기에는 recalibration 레버가 없으므로 여기서 고칠 수는 없고, 만약 하위 시스템이 `match` 확률 값을 그대로 소비한다면 상대적으로 캘리브레이션이 나은 iteration 2 구성(1:6.0, `calibration_error` 0.0542)을 별도 후보로 남겨두고 두 목적(균형정확도 vs 확률 품질)을 분리해 결정하십시오.
4. **정지 규칙과 슬라이스 크기 자체를 보강.** 최종 테스트 CI 하단(0.7315)이 임계값 0.7514보다 낮습니다. 현 프로토콜은 고정된 단일 20% 검증 분할이라 이보다 좁은 구간을 얻을 방법이 실행기 안에는 없으므로, 이 결과를 배포 판정으로 쓰려면 더 많은 평가 행(또는 반복 측정)을 확보하는 것이 다음 예산의 최우선 항목입니다. 이것이 해결되면 1번의 미세 가중치 조정도 "판정 불가"에서 벗어나 실제로 비교 가능해집니다.

> 참고: 데이터 카드에 별도로 기록된 주의사항(`Data caveats`)은 없었습니다. 다만 결측 컬럼이 60개, 최악 결측률이 0.7852(`expected_num_interested_in_me`)이고 이 실행에서는 `impute: none` + `missing_count: true`로만 다뤘습니다. `missing_indicator`는 `impute: none`과 함께 쓰면 중복(예측이 비트 단위로 동일)이라는 것이 이미 확인된 사항이므로, 위 제안에 포함하지 않았습니다. 또한 결측이 "언제/어디서 기록되었는지"를 반영하는 경우 무작위 분할은 그것을 이득으로 오독할 수 있다는 점이 문서에 명시돼 있는데, 이 데이터에는 그 이유를 확인할 카드 주의사항이 없습니다 — `missing_count`가 기여했는지 여부를 근거로 삼는 주장은 하지 않았습니다.