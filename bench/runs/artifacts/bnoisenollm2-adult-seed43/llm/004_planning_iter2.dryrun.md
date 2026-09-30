You are the Planning Agent of an AutoML system. You never see the raw data — only
the dataset card below. Produce the plan for the next training attempt.

## Dataset card

{
  "name": "adult",
  "description": "csv 데이터에서 자동 생성된 카드입니다. 원본 행은 이 카드에 포함되지 않습니다 — 모든 수치는 열 전체에 대한 집계입니다.",
  "task": "binary_classification",
  "target_column": "class",
  "n_rows": 48842,
  "n_features": 14,
  "n_features_dropped_non_numeric": 0,
  "encoding": {
    "numeric": 6,
    "one_hot_columns": 8,
    "one_hot_levels": 102,
    "dropped_high_cardinality": [],
    "dropped_unsupported_dtype": [],
    "max_cardinality": 50
  },
  "n_classes": 2,
  "class_balance": [
    0.7607,
    0.2393
  ],
  "imbalance_ratio": 3.18,
  "missing": {
    "overall_rate": 0.0095,
    "columns_with_missing": 3,
    "worst_rate": 0.0575
  },
  "features": [
    {
      "name": "age",
      "dtype": "int64",
      "missing_rate": 0.0,
      "distinct": "medium",
      "usable_by_executor": true,
      "kind": "continuous",
      "magnitude": "tens",
      "skew": "moderate",
      "outlier_rate": 0.0044,
      "target_corr": "moderate"
    },
    {
      "name": "workclass",
      "dtype": "str",
      "missing_rate": 0.0573,
      "distinct": "low",
      "usable_by_executor": true,
      "kind": "categorical"
    },
    {
      "name": "fnlwgt",
      "dtype": "int64",
      "missing_rate": 0.0,
      "distinct": "high",
      "usable_by_executor": true,
      "kind": "continuous",
      "magnitude": "thousands+",
      "skew": "moderate",
      "outlier_rate": 0.0297,
      "target_corr": "none"
    },
    {
      "name": "education",
      "dtype": "str",
      "missing_rate": 0.0,
      "distinct": "medium",
      "usable_by_executor": true,
      "kind": "categorical"
    },
    {
      "name": "education-num",
      "dtype": "int64",
      "missing_rate": 0.0,
      "distinct": "medium",
      "usable_by_executor": true,
      "kind": "discrete",
      "magnitude": "unit",
      "skew": "low",
      "outlier_rate": 0.0367,
      "target_corr": "strong"
    },
    {
      "name": "marital-status",
      "dtype": "str",
      "missing_rate": 0.0,
      "distinct": "low",
      "usable_by_executor": true,
      "kind": "categorical"
    },
    {
      "name": "occupation",
      "dtype": "str",
      "missing_rate": 0.0575,
      "distinct": "medium",
      "usable_by_executor": true,
      "kind": "categorical"
    },
    {
      "name": "relationship",
      "dtype": "str",
      "missing_rate": 0.0,
      "distinct": "low",
      "usable_by_executor": true,
      "kind": "categorical"
    },
    {
      "name": "race",
      "dtype": "str",
      "missing_rate": 0.0,
      "distinct": "low",
      "usable_by_executor": true,
      "kind": "categorical"
    },
    {
      "name": "sex",
      "dtype": "str",
      "missing_rate": 0.0,
      "distinct": "binary",
      "usable_by_executor": true,
      "kind": "binary"
    },
    {
      "name": "capital-gain",
      "dtype": "int64",
      "missing_rate": 0.0,
      "distinct": "high",
      "usable_by_executor": true,
      "kind": "continuous",
      "magnitude": "sub_unit",
      "skew": "high",
      "target_corr": "moderate"
    },
    {
      "name": "capital-loss",
      "dtype": "int64",
      "missing_rate": 0.0,
      "distinct": "medium",
      "usable_by_executor": true,
      "kind": "continuous",
      "magnitude": "sub_unit",
      "skew": "high",
      "target_corr": "weak"
    },
    {
      "name": "hours-per-week",
      "dtype": "int64",
      "missing_rate": 0.0,
      "distinct": "medium",
      "usable_by_executor": true,
      "kind": "continuous",
      "magnitude": "tens",
      "skew": "low",
      "outlier_rate": 0.2763,
      "target_corr": "moderate"
    },
    {
      "name": "native-country",
      "dtype": "str",
      "missing_rate": 0.0175,
      "distinct": "medium",
      "usable_by_executor": true,
      "kind": "categorical"
    }
  ],
  "target_missing": {
    "policy": "reject",
    "n_dropped": 0
  },
  "preprocessing": {
    "impute": "median",
    "scale": true
  },
  "constraints": {
    "memory_limit_mb": 2048,
    "max_train_time_sec": 600
  },
  "profile": {
    "generated_by": "automl_agent.scripts.profile",
    "policy": "aggregates only — no cell values, no min<path>, no class labels"
  },
  "baseline": {
    "model": "logreg (median impute + standard scale)",
    "note": "LogisticRegression fitted on the training split and scored on the validation split of automl_agent.scoring.splits' protocol, seed 43 — the same split scripts<path> uses, so the numbers are directly comparable. The test slice is not read here.",
    "n_rows_used": 48842,
    "protocol": {
      "train_fraction": 0.6,
      "val_fraction": 0.2,
      "test_fraction": 0.2,
      "stratified": true,
      "grouped_by": null,
      "seed": 43
    },
    "scores": {
      "f1": 0.6665,
      "accuracy": 0.8546,
      "balanced_accuracy": 0.7698,
      "precision": 0.7387,
      "recall": 0.6072,
      "roc_auc": 0.908,
      "pr_auc": 0.7723,
      "average_precision": 0.7723
    },
    "chance": {
      "f1": 0.0,
      "accuracy": 0.7608,
      "balanced_accuracy": 0.5,
      "precision": 0.0,
      "recall": 0.0,
      "roc_auc": 0.5,
      "pr_auc": 0.2392,
      "average_precision": 0.2392
    },
    "ks": 0.6531,
    "balanced_accuracy_at_best_cut": 0.8266,
    "ci": {
      "level": 0.95,
      "unit": "row",
      "resamples": 400,
      "scores": {
        "f1": {
          "low": 0.6492,
          "high": 0.6821
        },
        "accuracy": {
          "low": 0.847,
          "high": 0.8618
        },
        "balanced_accuracy": {
          "low": 0.7586,
          "high": 0.7798
        },
        "precision": {
          "low": 0.7189,
          "high": 0.756
        },
        "recall": {
          "low": 0.5847,
          "high": 0.6263
        },
        "roc_auc": {
          "low": 0.9015,
          "high": 0.9147
        },
        "pr_auc": {
          "low": 0.7567,
          "high": 0.7883
        },
        "average_precision": {
          "low": 0.7567,
          "high": 0.7883
        }
      }
    }
  }
}

## Data caveats

Facts about this dataset that the aggregates above cannot show — from the profiler's own
checks and from the operator, who has seen the raw file. Treat them as constraints, not
suggestions: a plan built on a column a caveat invalidates spends its attempt producing a
number nobody can trust.

(없음 — 이 데이터에 대해 별도로 기록된 주의사항이 없습니다)

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

This is attempt 2 of at most 5.

## Models this system can actually run

Only these identifiers exist. Anything else will be rejected by the executor.

[
  {
    "id": "logreg",
    "family": "linear",
    "cost": 1,
    "params": [
      "C",
      "max_iter",
      "class_weight"
    ],
    "notes": "fast, low-capacity baseline; features are scaled automatically"
  },
  {
    "id": "decision_tree",
    "family": "tree",
    "cost": 1,
    "params": [
      "max_depth",
      "min_samples_leaf",
      "class_weight"
    ],
    "notes": "interpretable, overfits easily"
  },
  {
    "id": "knn",
    "family": "instance",
    "cost": 2,
    "params": [
      "n_neighbors",
      "weights"
    ],
    "notes": "no training cost, slow at prediction, sensitive to dimensionality"
  },
  {
    "id": "hist_gbdt",
    "family": "gbdt",
    "cost": 3,
    "params": [
      "max_iter",
      "n_estimators",
      "learning_rate",
      "max_depth",
      "max_leaf_nodes",
      "l2_regularization",
      "class_weight",
      "early_stopping"
    ],
    "notes": "strong default for tabular data; also accepts n_estimators as an alias of max_iter. `early_stopping` defaults to 'auto', which is *on* above 10k rows, so saying nothing about it does not mean fitting on every train row"
  },
  {
    "id": "random_forest",
    "family": "bagging",
    "cost": 4,
    "params": [
      "n_estimators",
      "max_depth",
      "min_samples_leaf",
      "class_weight"
    ],
    "notes": "robust, memory-hungry with many trees"
  },
  {
    "id": "extra_trees",
    "family": "bagging",
    "cost": 4,
    "params": [
      "n_estimators",
      "max_depth",
      "min_samples_leaf",
      "class_weight"
    ],
    "notes": "more randomised than random_forest, often better with noisy labels"
  },
  {
    "id": "gradient_boosting",
    "family": "gbdt",
    "cost": 5,
    "params": [
      "n_estimators",
      "learning_rate",
      "max_depth",
      "subsample"
    ],
    "notes": "sequential and slow; prefer hist_gbdt unless small data"
  },
  {
    "id": "xgboost",
    "family": "gbdt",
    "cost": 4,
    "params": [
      "n_estimators",
      "learning_rate",
      "max_depth",
      "reg_lambda",
      "subsample",
      "scale_pos_weight",
      "early_stopping_rounds"
    ],
    "notes": "requires the xgboost package. `early_stopping_rounds` needs nothing else from the plan — the executor holds its own stopping slice back and supplies the eval set; `eval_set` and `callbacks` are the two keys it refuses. `scale_pos_weight` is **binary only** — xgboost ignores it on a multiclass objective, so above two classes the executor drops it and the attempt records that in `dropped_hyperparams`"
  },
  {
    "id": "mlp",
    "family": "neural",
    "cost": 5,
    "params": [
      "hidden_layer_sizes",
      "alpha",
      "learning_rate_init",
      "batch_size",
      "max_iter",
      "early_stopping"
    ],
    "notes": "highest capacity available here, and the most likely to hit the memory budget. Its `early_stopping` is a boolean only — the `'auto'` spelling belongs to hist_gbdt, and the executor drops it here rather than letting `fit` raise on it"
  },
  {
    "id": "svc",
    "family": "kernel",
    "cost": 5,
    "params": [
      "C",
      "kernel",
      "gamma",
      "class_weight"
    ],
    "notes": "quadratic in rows; unusable above roughly 20k rows"
  }
]

## What the executor can and cannot do

The executor is a fixed, verified script. It is not a notebook: it will not run
whatever the plan describes, only what is listed here.

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

## How many rows the fit will see

The card above gives the row total and the split fractions separately; this is their
product, plus the one executor default whose value depends on it.

- **train 29,304 rows** / val 9,769 / test 9,769, out of the card's 48,842. The card gives the fractions and the total but not this product, and capacity and early-stopping choices are made against the product.
- `hist_gbdt` and `mlp` early-stop on a slice of the **training** rows, not on the validation set above — so that slice is subtracted from the number in the first bullet, and an attempt that pays it is being compared on score against attempts that did not.
- Their default `early_stopping='auto'` is **on** here (29,304 > 10,000). Saying nothing about early stopping therefore does not mean running without it: at the default `validation_fraction` of 0.1 the fit sees 26,373 rows, not 29,304. Setting `early_stopping: false` is what turns it off; the executor reports what really happened in `internal_validation`.

## Previous attempts

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
  "iteration": 1,
  "model": "hist_gbdt",
  "hyperparams": {
    "max_iter": 150,
    "learning_rate": 0.1
  },
  "preprocessing": {
    "impute": "median",
    "scale": false,
    "missing_indicator": false,
    "missing_count": false
  },
  "metric": "balanced_accuracy",
  "score": 0.8017091088139368,
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
  "plan_strategy": "baseline: tabular 데이터에 대한 검증된 기본값(hist_gbdt)으로 기준선을 만든다"
}

## The Critic's verdict on the most recent attempt

{
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

## Instructions

1. If there is any history, your plan **must differ materially** from every previous
   attempt. Repeating a plan wastes the remaining budget. State the difference
   explicitly in `changes_from_last`.
2. If the Critic reported `oom` or `too_slow`, the plan must reduce the resource
   footprint: choose a smaller model family, lower the estimator or iteration count,
   or subsample the training set (`"train_subsample": 0.5`). `"batch_size"` and
   `"precision": "fp16"` also lower the pre-flight memory estimate, so they help clear
   an `oom` guard — but they do not change how the model is fitted, so do not count on
   them for speed or for score. Do not propose a *larger* configuration in response to
   a resource failure.
3. If the Critic reported `underfitting`, increase capacity or training length. If it
   reported `overfitting`, add regularisation, reduce capacity, or address the data.
   If it reported `wrong_model_family`, switch to a different family entirely.
4. Keep `hyperparams` to keys that plausibly apply to the chosen family. The executor
   drops unknown keys, so inventing them just wastes the attempt.
5. Be concrete and numeric. "Tune the hyperparameters" is not a plan.
6. `strategy`, `changes_from_last` and `rationale` describe what *this executor* will
   do. Do not describe a step it cannot take — a plan whose reasoning depends on
   threshold tuning, cross-validation or engineered features spends its attempt on a
   configuration that will run without any of them. Write it as what you *will* do:
   there is no need to enumerate the unavailable items to show you read the list.

Respond with a single JSON object matching the schema. No prose outside it.
