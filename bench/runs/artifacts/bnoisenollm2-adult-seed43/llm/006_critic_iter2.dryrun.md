You are the Result Critic.

The most recent training attempt did not reach the goal. Diagnose *why*, citing the numbers, and name one concrete change for the next attempt.

## Goal

{
  "metric": "balanced_accuracy",
  "direction": "maximize",
  "mode": "auto",
  "threshold": 0.9194,
  "source": "derived",
  "reference": {
    "source": "baseline",
    "metric": "balanced_accuracy",
    "baseline": 0.7698,
    "margin": 0.65,
    "ks": 0.6531,
    "ranking_ceiling": 0.8266,
    "exceeds_ranking_ceiling": true,
    "required_ks": 0.8388,
    "ks_shortfall": 0.1857,
    "passable_margin": 0.246,
    "baseline_ci": [
      0.7586,
      0.7798
    ],
    "baseline_ci_unit": "row",
    "chance": 0.5
  }
}

auto 모드 — balanced_accuracy 0.9194 이상 ← 기준선 0.7698 + 남은 여유의 65% (chance 0.5) — 이 바는 기준선 랭킹의 상한 0.8266를 넘습니다 (KS 0.6531). 이 랭킹으로는 어떤 임계값을 골라도 닿지 않으니 랭킹 자체를 올려야 합니다 — 모델 family나 특성. 크기로 말하면 이 바가 요구하는 KS가 0.8388이고 기준선은 0.6531이므로 랭킹이 0.1857만큼 올라야 합니다 — 계열 교체로 그만큼 움직인 기록이 있는지 먼저 보십시오. 지금 기준선에서 이 상한 안에 드는 margin은 0.246 이하입니다

The line above is the same goal in prose, and two of the things it can say are
obligations rather than context.

- *"이 바는 기준선 랭킹의 상한 X를 넘습니다"* — the bar is above the ceiling of the
  ranking the baseline had. That ceiling belongs to the baseline, so a better ranking
  generator can pass it and one already has on other runs; what cannot pass it is another
  threshold, another class weight, or another point in the same family's hyperparameter
  space. Read it as: the remaining distance is in the ranking, so spend attempts on the
  ranking — a different family, or more information in the columns.
- *"기준선의 신뢰구간 안에 있습니다"* — clearing this bar would not be distinguishable
  from where the run started.

## The attempt being judged

Plan:

{
  "strategy": "하이퍼파라미터 재탐색: learning_rate와 깊이 조합을 이전과 다른 지점에서 시도한다",
  "model_family": "gbdt",
  "candidate_models": [
    "hist_gbdt",
    "xgboost",
    "gradient_boosting"
  ],
  "hyperparams": {
    "max_iter": 400,
    "learning_rate": 0.15,
    "max_depth": 12,
    "class_weight": {
      "0": 1.0,
      "1": 1.5
    }
  },
  "preprocessing": {
    "impute": "median"
  },
  "changes_from_last": "hyperparam 진단 — learning_rate<path> 조합 변경",
  "rationale": "critic.failure_type=hyperparam / direction=balanced_accuracy는 recall과 specificity의 평균이므로 둘이 같아지는 지점이 최적이다. recall(0.6594)가 specificity(0.9440)보다 낮으니 양성 클래스 가중치를 올린다: 1 → 1.5. train과 validation의 격차는 용량에 대한 증거이므로 가중치 방향의 근거가 되지 못한다.",
  "unsupported_claims": [],
  "source": "heuristic",
  "iteration": 2
}

Model: `hist_gbdt`

Hyperparameters:

{
  "max_iter": 400,
  "learning_rate": 0.15,
  "max_depth": 12,
  "class_weight": {
    "0": 1.0,
    "1": 1.5
  }
}

Result:

{
  "metrics": {
    "f1": 0.7308662741799832,
    "accuracy": 0.8689732828334528,
    "balanced_accuracy": 0.8260288518626504,
    "precision": 0.7184787102108309,
    "recall": 0.7436884895164741,
    "roc_auc": 0.9290268567662163,
    "pr_auc": 0.8325261120304375,
    "average_precision": 0.8325261120304375,
    "specificity": 0.908369,
    "balanced_accuracy_at_best_cut": 0.8458,
    "cut_headroom": 0.019771,
    "brier": 0.089952,
    "calibration_error": 0.037673,
    "balanced_accuracy_ci_low": 0.816178,
    "balanced_accuracy_ci_high": 0.835365,
    "train_f1": 0.7647182359117956,
    "train_accuracy": 0.8853057603057604,
    "train_balanced_accuracy": 0.8488552320499712,
    "train_val_gap": 0.022826
  },
  "applied_hyperparams": {
    "max_iter": 400,
    "learning_rate": 0.15,
    "max_depth": 12,
    "class_weight": {
      "0": 1.0,
      "1": 1.5
    }
  },
  "applied_preprocessing": {
    "impute": "median",
    "scale": false,
    "missing_indicator": false,
    "missing_count": false
  },
  "internal_validation": {
    "held_out_rows": 2931,
    "fit_rows": 26373,
    "validation_fraction": 0.1,
    "stopped_at_iter": 96,
    "max_iter": 400
  },
  "paired": {
    "status": "measured",
    "unit": "row",
    "metric": "balanced_accuracy",
    "delta_vs_best": 0.02432,
    "delta_ci_low": 0.018042,
    "delta_ci_high": 0.030482,
    "p_better": 1.0,
    "baseline_iteration": 1,
    "resamples": 400,
    "threads_changed": false
  },
  "status": "ok",
  "split": "val",
  "error_type": null,
  "train_time_sec": 6.467,
  "wall_time_sec": 9.026,
  "returncode": 0,
  "dropped_hyperparams": []
}

## All previous attempts

[
  {
    "iteration": 1,
    "model": "hist_gbdt",
    "hyperparams": {
      "max_iter": 150,
      "learning_rate": 0.1
    },
    "dropped_hyperparams": [],
    "preprocessing": {
      "impute": "median",
      "scale": false,
      "missing_indicator": false,
      "missing_count": false
    },
    "internal_validation": {
      "held_out_rows": 2931,
      "fit_rows": 26373,
      "validation_fraction": 0.1,
      "stopped_at_iter": 105,
      "max_iter": 150
    },
    "unsupported_claims": [],
    "status": "ok",
    "error_type": null,
    "metrics": {
      "f1": 0.7177456916627852,
      "accuracy": 0.8759340771829256,
      "balanced_accuracy": 0.8017091088139368,
      "precision": 0.7874297393970363,
      "recall": 0.6593923833975182,
      "roc_auc": 0.928478711908812,
      "pr_auc": 0.8310797566504815,
      "average_precision": 0.8310797566504815,
      "specificity": 0.944026,
      "balanced_accuracy_at_best_cut": 0.8471,
      "cut_headroom": 0.045391,
      "brier": 0.087374,
      "calibration_error": 0.007454,
      "balanced_accuracy_ci_low": 0.791397,
      "balanced_accuracy_ci_high": 0.811475,
      "train_f1": 0.7412598302577279,
      "train_accuracy": 0.8866025116025116,
      "train_balanced_accuracy": 0.815396069632853,
      "train_val_gap": 0.013687
    },
    "train_time_sec": 5.899,
    "critic": {
      "failure_type": "hyperparam",
      "evidence": "balanced_accuracy=0.8017는 recall=0.6594와 specificity=0.9440의 평균이고 둘의 차이가 0.2846다 — recall가 낮아 운영점이 한쪽으로 기울어 있다.",
      "direction": "balanced_accuracy는 recall과 specificity의 평균이므로 둘이 같아지는 지점이 최적이다. recall(0.6594)가 specificity(0.9440)보다 낮으니 양성 클래스 가중치를 올린다: 1 → 1.5. train과 validation의 격차는 용량에 대한 증거이므로 가중치 방향의 근거가 되지 못한다.",
      "concrete_changes": {
        "class_weight": {
          "0": 1,
          "1": 1.5
        }
      },
      "unsupported_claims": [],
      "source": "heuristic"
    }
  }
]

## Best result so far

{
  "iteration": 2,
  "model": "hist_gbdt",
  "hyperparams": {
    "max_iter": 400,
    "learning_rate": 0.15,
    "max_depth": 12,
    "class_weight": {
      "0": 1.0,
      "1": 1.5
    }
  },
  "preprocessing": {
    "impute": "median",
    "scale": false,
    "missing_indicator": false,
    "missing_count": false
  },
  "metric": "balanced_accuracy",
  "score": 0.8260288518626504,
  "metrics": {
    "f1": 0.7308662741799832,
    "accuracy": 0.8689732828334528,
    "balanced_accuracy": 0.8260288518626504,
    "precision": 0.7184787102108309,
    "recall": 0.7436884895164741,
    "roc_auc": 0.9290268567662163,
    "pr_auc": 0.8325261120304375,
    "average_precision": 0.8325261120304375,
    "specificity": 0.908369,
    "balanced_accuracy_at_best_cut": 0.8458,
    "cut_headroom": 0.019771,
    "brier": 0.089952,
    "calibration_error": 0.037673,
    "balanced_accuracy_ci_low": 0.816178,
    "balanced_accuracy_ci_high": 0.835365,
    "train_f1": 0.7647182359117956,
    "train_accuracy": 0.8853057603057604,
    "train_balanced_accuracy": 0.8488552320499712,
    "train_val_gap": 0.022826
  },
  "train_time_sec": 6.467,
  "plan_strategy": "하이퍼파라미터 재탐색: learning_rate와 깊이 조합을 이전과 다른 지점에서 시도한다"
}

## What each prescription was worth

One row per attempt, joined to the verdict that produced it, so you can see what your own
earlier diagnoses bought. The last row is the attempt being judged; the verdict you are
about to write is what the next row will be attributed to. The score column is the goal
metric against the best score that existed *before* that attempt ran, and "최고 갱신" there
is arithmetic on two point estimates and nothing more.

A row beginning `— 다만` is one where the attempt is not the prescription: the Planner is
allowed to overrule you, and when it did, that row's score is not what your diagnosis bought.
`처방은 … 였고 계획이 … 로 바꿨다` is the family; `처방의 전처리가 이 시도에 없다` is a
preprocessing setting that never reached the pipeline that ran, so the row is not evidence
about it either way. Do not read such a row as the prescription confirmed or refuted, and if
you still want the thing you asked for, prescribe it again rather than treating it as tried.

What qualifies it is `짝지은 Δ` on the same row, where the row has one: the two attempts were
scored on the same validation rows, so that Δ is the difference resampled over those pairs
rather than two separate scores subtracted. It is the sharper test of "did this change move
the score", and `이 행들로는 0과 구분되지 않음` on a row means the movement it reports is not
evidence, however large the subtraction beside it looks. A row saying `짝지은 검정 없음` was
not compared at all — that is not the same as having been compared and found nothing. Use
this to decide what to prescribe next; it says nothing about what the run has demonstrated,
which is measured once after the loop ends on rows you never see.

  iteration 1  hist_gbdt      balanced_accuracy=0.8017  첫 측정
  iteration 2  hist_gbdt      balanced_accuracy=0.8260  직전 최고 대비 +0.0243 — 최고 갱신  [짝지은 Δ(iteration 1 대비) +0.0243, 95% CI +0.0180~+0.0305, P(개선) 1.000 — 0과 구분됨]  (hyperparam 처방의 결과)

  써 본 계열: hist_gbdt 2회(최고 0.8260)
  운영점 레버의 크기: cut_headroom 0.0198 대 목표까지 남은 거리 0.0934 — 임계값과 클래스 가중치로 살 수 있는 최대치는 남은 거리의 21%이고, 나머지 79%는 랭킹에 있습니다.
  남은 iteration: 3

## What these rows can resolve

Every score above is one measurement on one validation slice, so part of any difference
between two of them is the slice rather than the models. The line below is the 95%
bootstrap interval of the attempt being judged, and it names the numbers that fall inside
it. Those are the numbers this slice does **not** separate from the attempt's score.

balanced_accuracy=0.8260 (95% CI 0.8162~0.8354, 폭 0.0192). 비교 대상 중 이 구간 안에 들어오는 값은 없으므로, 위 점수와의 차이는 이 슬라이스가 실제로 구분해 낸 차이입니다.

## Data caveats

Facts about this dataset that no metric above shows — from the profiler's own checks and
from the operator, who has seen the raw file. They constrain the diagnosis as much as the
prescription: a caveat can be the reason a score is where it is, and it rules out any
`direction` that depends on what it invalidates.

(없음 — 이 데이터에 대해 별도로 기록된 주의사항이 없습니다)

## What the executor can and cannot do

Your `direction` and `concrete_changes` are carried out by a fixed, verified script.
Prescribing something outside this list costs the next attempt. Write `direction` as
what to do next; there is no need to enumerate the unavailable items to show you read
the list.

### The executor will do

- Fit exactly one estimator from the model list per attempt.
- Apply proposed `hyperparams` through `set_params`, under sklearn's own names (a small alias map covers `n_estimators`→`max_iter` for hist_gbdt and similar). Keys the estimator does not accept are dropped and recorded in `dropped_hyperparams`.
- Rebalance classes through the estimator's own `class_weight` — `'balanced'`, or an explicit weight per class such as `{"0": 1, "1": 10}`, whose keys are class codes 0..n_classes-1 in the order of the card's `class_balance` — or xgboost's `scale_pos_weight`, where that estimator supports it. This is the only imbalance lever available, and `'balanced'` is one point on it, not the best one: it pins the ratio at the class frequency. A map that does not cover every class exactly once is dropped into `dropped_hyperparams`.
- Cut the training set with `train_subsample` (a float strictly between 0 and 1).
- Impute and scale as the plan's `preprocessing` block asks, falling back to the dataset card's: `impute` as one of `median`, `mean` or `most_frequent`, applied to every column with one strategy, and `scale` (standardisation) for scale-sensitive families only. `impute` as `none` removes the imputer so the model splits on NaN itself — available for `hist_gbdt` and `xgboost` only, and any other family is put back on `median`. Each attempt reports the pipeline that was really built in `applied_preprocessing`, so a downgrade is visible in the history rather than only in the training log.
- Turn missingness into columns, both booleans and both applied before imputation: `missing_indicator` appends one 0 or 1 column per input column, and `missing_count` appends a single column holding how many of that row's fields were not measured. `missing_indicator` covers every column or none — there is no naming a subset of the high-missing ones. `missing_count` is the one a NaN-splitting family cannot already express — its splits are per column, and this aggregates across them, so on clinical rows it stands in for how much workup a patient received. One thing about `missing_indicator` is settled. Handed to a family that splits on NaN natively under `impute: none` it is *redundant*, not merely weak — appending 14 indicator columns returned predictions identical bit for bit, because the indicator's only split is the NaN branch the tree already had. That reproduced under both of the splits below, so do not spend an iteration on it alongside `impute: none`. What it is worth when the plan *does* impute is not settled, and the reason is worth reading before spending an iteration on it. Under a stratified random split of one clinical sample the sizes were large and mutually consistent: imputing cost 0.0097 roc_auc, the indicator recovered 0.0065 of that in the same tree family, and on `logreg`, which cannot be handed a NaN at all, it gained 0.0151 — the largest of the three, which fits a linear model having no way to express *this value was invented* while a tree can by splitting at the imputed value. Refitting the same configurations on the same file under a contiguous split, so that training and validation rows fall on either side of a change in what the file records, all three collapsed to +0.0011, -0.0022 and +0.0017, every interval spanning 0. Absolute roc_auc barely moved (0.8709 to 0.8671), so what failed to carry over is the missingness lever specifically and not the models. The likeliest reading is that under the random split those columns were partly identifying which recording regime a row came from, which is not a subject's state; the contiguous split does not cleanly separate that from the weaker reading, since the column with the most missingness is observed in 83% of its training rows and 8% of its validation rows and an indicator on it is nearly constant there. Either way the instruction is the same: on that sample the three sizes are not established, so do not cite them as what an iteration here will buy. `missing_count` bought nothing under the random split — 0.0003 on its own, 0.0000 on top of the indicator in either family — and was not refitted under the contiguous one. Being representable is not the same as being worth a column. What generalises is the check rather than the sizes: read the card's caveats for *why* values are missing, and where missingness tracks when or where a row was recorded rather than the subject's state, these columns let the model learn the recording regime — a random split scores that as a gain instead of showing it.
- Read `batch_size` and `precision: fp16` as *memory-budget hints only* — they change the pre-flight memory estimate, not how the model is fitted.
- Score on a fixed protocol: a stratified three-way split at the run's seed — train 60%, validation 20%, test 20% — where every score you are shown, and the choice of the run's best attempt, comes from the validation 20%. The test 20% is carved off first, is scored exactly once after the loop ends, and is never visible to a plan or a diagnosis. Reported on the validation split: every metric in the registry, plus `train_f1`/`train_accuracy`/`train_<goal metric>` and `train_val_gap` measured on the goal metric. On a binary target it also reports `specificity`, which no goal may target but which names the direction the imbalance lever has to move: `balanced_accuracy` is the mean of `recall` and `specificity`.
- Report what choosing a decision threshold would have been worth, without choosing one: `balanced_accuracy_at_best_cut` is the best `balanced_accuracy` any cut of this model's ranking allows, and `cut_headroom` is the distance from the score actually achieved. Both are diagnostics no goal may target. Read `cut_headroom` before reaching for the operating point — when it is small the cut is near-optimal already and the remaining gap is in the ranking, which means the model family or the features, not `class_weight`.
- Report whether the predicted probabilities are worth reading as probabilities, without changing them: `brier` is the mean squared error of the positive-class probability, and `calibration_error` is the count-weighted distance between predicted and observed rate over ten bins — so 0.04 means the probabilities are off by four percentage points on average. Both are diagnostics no goal may target, and `calibration_error` is omitted under 50 rows because that estimator is biased upward on few rows; its absence means not measured, not fine. There is no recalibration lever here — nothing refits the probabilities, so do not propose `CalibratedClassifierCV` or a shifted cut. What the two numbers are for is reading a shortfall: a model can rank well and still be systematically overconfident, and no metric in the registry would show it.
- Price a lever on the axis it moves, because two of them move independently and `balanced_accuracy` is their sum. The *ranking* axis is what `roc_auc`, `pr_auc` and `balanced_accuracy_at_best_cut` measure — how well the model orders rows, which no decision rule can improve. The *operating point* axis is where the default rule happens to cut that ranking, which is what `class_weight` and `scale_pos_weight` move and what `cut_headroom` measures the remaining size of. A `balanced_accuracy` difference on its own does not say which of the two moved, so it cannot rank levers. Measured on one clinical sample, the best and the worst of five attempts sat 0.0526 of `balanced_accuracy` apart (0.7334 to 0.7860), and that gap splits exactly in two: 0.0066 of it was their `balanced_accuracy_at_best_cut` (0.7823 to 0.7889) and the other 0.0460 was the cut landing somewhere else. 87% of the apparent swing was the operating point, and the worst attempt carried `cut_headroom` 0.0489 with recall 0.5197 against specificity 0.9471. On the ranking axis in the same measurements, retuning hyperparameters inside one family spanned 0.0032 of roc_auc, and swapping tree family under an unchanged pipeline moved 0.0077 for one family and 0.0022 for another. Read those as sizes, not as a ranking: the paired resolution of a roc_auc difference in these measurements is somewhere between 0.003 and 0.006, so the smaller family swap is not distinguishable from retuning, and there is no single size for 'swap the family' to budget against either — it spanned 0.0022 to 0.0077 depending on which family. The imputation lever is deliberately given no size at all: on this file it reads 0.0097 of roc_auc under a random split and +0.0011 with the interval spanning 0 under a contiguous one, so there is nothing stable to rank it by, and the missing-data entry above is where that is set out. No number here licenses calling a lever the largest available. One verdict read a size that way, prescribed the imputation change together with a family it had not tried, and the attempt lost 0.0572 of `balanced_accuracy` with two owners for the loss. What survives is not a lever ranking: the two axes have to be read apart, the order the `balanced_accuracy` column suggests is not the order the ranking axis has, and `cut_headroom` against the distance still to go is the number that says which axis a shortfall is on.

### The executor will not do

- **Choose or search a decision threshold / probability cut-off.** Instead: the executor always calls `predict()`, so the operating point is whatever the estimator's default rule gives. Move it with `class_weight='balanced'` or `scale_pos_weight`, and read `cut_headroom` first — it reports what the sweep you are about to propose would have been worth.
- **Run cross-validation or produce out-of-fold predictions.** Instead: there is one 20% validation split, fixed by the seed and shared with the card's baseline so the two numbers are comparable.
- **Derive, encode or drop features. The two missingness columns are the whole exception, and the CAN list above states them.** Instead: every numeric column of the card goes in as it is, and the only columns the executor will add are `missing_indicator` and `missing_count`. Nothing is combined: no ratio or difference between two columns, no interaction, no polynomial, no re-encoding, and no dropping — a column you want out has to leave the card, not the plan.
- **Resample the training set (SMOTE, over-sampling, under-sampling).** Instead: every training row goes in exactly once; `train_subsample` only shrinks.
- **Calibrate predicted probabilities.** Instead: read `brier` and `calibration_error`, which are reported on every binary attempt: the miscalibration is measured for you, it just is not corrected. Ranking metrics (`roc_auc`, `pr_auc`) are calibration-free anyway, and the threshold that calibration would inform is not tunable here either.
- **Combine several models: stacking, blending, voting, seed averaging.** Instead: one estimator per attempt. A tree *ensemble* like hist_gbdt or random_forest is a single estimator and is fine — combining separate attempts is not.
- **Treat columns differently in preprocessing.** Instead: one imputer and one scaler for the whole matrix.
- **Name your own `eval_set`, or pass `callbacks`, to xgboost.** Instead: `early_stopping_rounds` works and needs nothing else from you — the executor holds 10% of train back and supplies the eval set itself, the way hist_gbdt's `early_stopping=True` + `validation_fraction` does, and reports both counts in `internal_validation`. Those 10% are rows the fit does not see, so on a small training split the loss can outweigh what the stop buys. What the executor cannot forward is an eval set naming rows it did not split, or a callback list.
- **Change the split, the seed, the goal metric, or add a separate test set.** Instead: those are the run's configuration, fixed before the loop starts.

Do not build a plan on anything in the second list. A plan that assumes it spends an attempt and then reports a configuration that never ran.

## Diagnosis rules

Pick exactly one `failure_type` from: underfitting, overfitting, data_issue, hyperparam, oom, too_slow, wrong_model_family, unknown

- `oom` — the result's `error_type` is `oom`, or the log shows an allocation failure.
- `too_slow` — the run exceeded its time budget (`error_type` is `too_slow`).
- `underfitting` — train and validation scores are both short of the bar and close
  together. "Short" follows the goal's own direction: below the bar for a score to
  maximise, above it for an error to minimise.
- `overfitting` — training is good while validation lags (large `train_val_gap`). The gap
  is always *how much worse validation is than training*, so a large positive value means
  this whichever way the goal metric runs.
- `hyperparam` — the family looks right and the gap is unremarkable, but the settings
  are off (learning rate, depth, regularisation strength).
- `wrong_model_family` — several tuning attempts within one family have plateaued well
  short of the goal.
- `data_issue` — the failure is about the data itself: class imbalance, label noise,
  too few rows, a broken column, `error_type` of `data_issue`. On a continuous target the
  data shapes that belong here are a heavy-tailed or skewed target and outliers in it —
  there is no class imbalance and no weight lever to prescribe.
- `unknown` — the evidence genuinely does not support any of the above.

## Instructions

1. `evidence` must quote the actual numbers you reasoned from (scores, gap, timing,
   error type). No vague statements.
2. `direction` is one sentence telling the Planner what to change next, and why.
3. `concrete_changes` is a small dict of specific settings, for example
   `{"model": "smaller", "batch_size": 16, "precision": "fp16"}` or
   `{"max_iter": 400, "learning_rate": 0.05}`.
4. If earlier attempts already tried your suggestion and it did not help, suggest
   something else — read `What each prescription was worth` before answering, not the raw
   history. A `failure_type` listed there as having paid nothing is a diagnosis you have
   already spent an iteration on: either name in `evidence` what is different this time,
   or diagnose something else. A run that issued `wrong_model_family` twice tried three
   families in five iterations and ended where iteration 1 had already been. Note the
   remaining iteration count too — with one left, prescribe the change with the largest
   expected effect, not the cheapest.
   Where that section ends, up to two lines size a lever against the distance still to go:
   `운영점 레버의 크기` for the decision threshold, and `랭킹 상한의 산포` for the family
   swap. Read the second one before prescribing another family. When the shortfall is several
   times the span the families tried have actually covered, one more family off the same list
   is not a plan that reaches the bar, and `evidence` should quote those two numbers as the
   reason. That span is a range over per-family maxima, not a paired test, so it is never
   evidence that two families *do* differ — only a measure of how little the swap has bought.
5. Check `dropped_hyperparams` in the result before you prescribe. A key listed there
   was never applied, so the attempt does not tell you whether that setting would have
   helped — and re-proposing it will be dropped again. Prescribe only what the section
   above says the executor will do.
Instructions 6, 7 and 8 are about a binary decision rule, so they apply only when the
target is one. On a continuous target there is no operating point to move: `recall`,
`specificity`, `cut_headroom` and `balanced_accuracy_at_best_cut` are not in
`result.metrics`, and no weight lever exists to prescribe. Skip all three there — do not
substitute an analogy for them — and diagnose from `train_val_gap`, the train and
validation scores against the bar, the history, and the interval below.

6. Read the weight direction off `recall` and `specificity`, never off the `train_val_gap`.
   On a binary target `balanced_accuracy = (recall + specificity) / 2`, so its optimum
   is where the two are *equal*: if `recall` is the lower of the two the positive class
   needs **more** weight, and if `specificity` is lower it needs **less**. The gap is
   evidence about capacity and says nothing about which way the operating point should
   move — a run that prescribed "dial the positive weight back from 8 to 5" from a large
   gap, while recall 0.680 sat well under specificity 0.864, prescribed the wrong
   direction and the Planner had to overrule it. Both numbers are in `result.metrics`.
7. Before you reach for the operating point at all, read `cut_headroom`. It is
   `balanced_accuracy_at_best_cut` minus the `balanced_accuracy` the attempt achieved —
   the exact amount that choosing a decision threshold would have been worth. When it is
   small the cut is already near-optimal and the imbalance lever has nothing left to
   give, so the remaining gap is in the *ranking* and `wrong_model_family` is the honest
   diagnosis. `balanced_accuracy_at_best_cut` is `(1 + KS) / 2`, a property of this
   model's ranking rather than of the data, so a better family raises it. Never quote it
   as a score the attempt reached — the executor does not apply that cut.
8. If two earlier attempts of the same model straddle the crossing — one with `recall`
   above `specificity` and one below — interpolate between their weights instead of
   taking another step outward. The sign flip brackets the optimum, so a step past either
   observation returns to a weight already measured as too far.

9. A difference the measurement cannot resolve is not evidence — and which measurement
   answers that depends on what the difference is between. One attempt against another: read
   that row's `짝지은 Δ`, and a CI spanning 0 means unresolved. A row without one: fall back
   to the interval in `What these rows can resolve`, which for this comparison is the wider
   of the two and so errs toward refusing a difference rather than granting it. An attempt
   against the bar: that section is the only measurement there is.
   Either way, do not diagnose a cause for an unresolved difference and do not call it an
   improvement or a regression. Name it as movement the measurement cannot resolve, and base
   the diagnosis on something else — `train_val_gap`, an error type, or (on a binary target
   only) `recall` against `specificity` and `cut_headroom`.
   If the only thing left to say is that the attempts are indistinguishable, `direction`
   should be a change big enough to produce a difference wider than that width, and
   `evidence` should quote the width as the reason.

10. A transition that moved the family *and* the preprocessing has two owners for one
    number, and `한 행에 레버가 둘인 전이` in the ledger names those rows. The delta there is
    the sum, so do not attribute it to either lever — not in `evidence`, and not as the
    premise of the next prescription. Moving both at once is allowed and sometimes forced
    (`logreg` cannot run on native NaN), so the fix is not to forbid it: prescribe a next
    transition that moves one of them, or say in `evidence` why both have to move again.
    That applies to the row you are about to create, not only to the ones above. If
    `concrete_changes` moves a preprocessing key *and* the family *or* a hyperparameter, the
    next attempt's score will not be attributable either, and `전처리 레버 단독 전이` in the
    ledger reports whether any transition so far has been. Three runs of one sample produced
    none between them — every attempt that touched the pipeline moved something else with it,
    so none of those histories says anything about that lever, in either direction. "Both have
    to move again" is a real exception, but it is the exception of a family that *cannot* take
    the other setting, not of two changes that both look worth trying. The one case where an
    unattributable row costs nothing is when `남은 iteration` is 0: no further verdict will
    read it, so there is nothing to preserve attribution for.

Respond with a single JSON object matching the schema. No prose outside it.
