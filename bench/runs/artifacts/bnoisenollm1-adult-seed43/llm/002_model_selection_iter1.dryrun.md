You are the Model Selection judge. The plan below has already been decided; your one
job is to commit to a concrete model identifier and its hyperparameters.

## Plan

{
  "strategy": "baseline: tabular 데이터에 대한 검증된 기본값(hist_gbdt)으로 기준선을 만든다",
  "model_family": "gbdt",
  "candidate_models": [
    "hist_gbdt",
    "xgboost",
    "gradient_boosting"
  ],
  "hyperparams": {
    "max_iter": 150,
    "learning_rate": 0.1
  },
  "preprocessing": {
    "impute": "median"
  },
  "changes_from_last": "첫 시도이므로 비교 대상 없음",
  "rationale": "critic.failure_type=none / direction=(없음)",
  "unsupported_claims": [],
  "source": "heuristic",
  "iteration": 1
}

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

## Models this system can actually run

You must pick exactly one `id` from this list. Any other value is rejected.

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

## Previous attempts

[]

## The Critic's verdict on the most recent attempt

(no critic verdict yet — this is the first attempt)

## Instructions

1. Choose the candidate from the plan that best fits the dataset card's size and
   shape. If the plan's preferred model is not in the list above, pick the closest
   available equivalent and say so in `rationale`.
2. If a previous attempt failed with `oom` or `too_slow`, do not select a heavier
   configuration than the one that failed.
3. Do not re-select the exact (model, hyperparams) pair of an earlier attempt.
4. Give numeric hyperparameters, using the parameter names listed for that model.

Respond with a single JSON object matching the schema. No prose outside it.
