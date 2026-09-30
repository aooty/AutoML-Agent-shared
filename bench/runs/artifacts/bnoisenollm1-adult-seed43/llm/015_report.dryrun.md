You are the Report Agent. The AutoML loop has finished. Write the final report.

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

## Outcome

- Goal reached: no
- Iterations used: 5 of 5
- Stop reason: max_iterations (최대 반복 횟수 도달)
- 재계획: critic이 4회 실행되어 그만큼 재계획했습니다 (시도 5회 — 마지막 시도는 평가 직후 루프가 끝나므로 진단 대상이 아닙니다).

## Best result

{
  "iteration": 3,
  "model": "hist_gbdt",
  "hyperparams": {
    "max_iter": 400,
    "learning_rate": 0.03,
    "max_depth": 6,
    "class_weight": {
      "0": 1.0,
      "1": 2.25
    }
  },
  "preprocessing": {
    "impute": "median",
    "scale": false,
    "missing_indicator": false,
    "missing_count": false
  },
  "metric": "balanced_accuracy",
  "score": 0.8420008505011116,
  "metrics": {
    "f1": 0.725597141245063,
    "accuracy": 0.8506500153546934,
    "balanced_accuracy": 0.8420008505011116,
    "precision": 0.6473154362416107,
    "recall": 0.8254172015404364,
    "roc_auc": 0.9288910368283333,
    "pr_auc": 0.8324792992613164,
    "average_precision": 0.8324792992613164,
    "specificity": 0.858584,
    "balanced_accuracy_at_best_cut": 0.8469,
    "cut_headroom": 0.004899,
    "brier": 0.098926,
    "calibration_error": 0.079096,
    "balanced_accuracy_ci_low": 0.832822,
    "balanced_accuracy_ci_high": 0.850641,
    "train_f1": 0.7452357071213641,
    "train_accuracy": 0.8613158613158614,
    "train_balanced_accuracy": 0.8566458422663503,
    "train_val_gap": 0.014645
  },
  "train_time_sec": 14.23,
  "plan_strategy": "하이퍼파라미터 재탐색: learning_rate와 깊이 조합을 이전과 다른 지점에서 시도한다"
}

## Final held-back measurement

최종 테스트(20%, 반복 중 한 번도 쓰이지 않은 행): iteration 3의 hist_gbdt → balanced_accuracy=0.8476 (95% CI 0.8389~0.8555) (검증 0.8420 대비 -0.0056 — 이 차이가 선택 편향의 크기입니다, 다만 검증 점수가 위 CI 안에 있어 이 행들로는 0과 구분되지 않습니다)

Every number in `Best result` and in the history below is a *validation* score, and the
loop chose the best of them by comparing exactly those numbers. The line above is the
same model scored once on rows that were held back before the first fit and never used
for any decision — so it, not the validation score, is what this run actually
demonstrates about the model. If it says the scoring was skipped, say so; do not treat
the validation score as if it had been held back.

## Full attempt history

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
    "train_time_sec": 7.786,
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
  },
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
      "stopped_at_iter": 96,
      "max_iter": 400
    },
    "unsupported_claims": [],
    "status": "ok",
    "error_type": null,
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
    "train_time_sec": 6.753,
    "critic": {
      "failure_type": "hyperparam",
      "evidence": "balanced_accuracy=0.8260는 recall=0.7437와 specificity=0.9084의 평균이고 둘의 차이가 0.1647다 — recall가 낮아 운영점이 한쪽으로 기울어 있다.",
      "direction": "balanced_accuracy는 recall과 specificity의 평균이므로 둘이 같아지는 지점이 최적이다. recall(0.7437)가 specificity(0.9084)보다 낮으니 양성 클래스 가중치를 올린다: 1.5 → 2.25. train과 validation의 격차는 용량에 대한 증거이므로 가중치 방향의 근거가 되지 못한다.",
      "concrete_changes": {
        "class_weight": {
          "0": 1,
          "1": 2.25
        }
      },
      "unsupported_claims": [],
      "source": "heuristic"
    }
  },
  {
    "iteration": 3,
    "model": "hist_gbdt",
    "hyperparams": {
      "max_iter": 400,
      "learning_rate": 0.03,
      "max_depth": 6,
      "class_weight": {
        "0": 1.0,
        "1": 2.25
      }
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
      "stopped_at_iter": 400,
      "max_iter": 400
    },
    "unsupported_claims": [],
    "status": "ok",
    "error_type": null,
    "metrics": {
      "f1": 0.725597141245063,
      "accuracy": 0.8506500153546934,
      "balanced_accuracy": 0.8420008505011116,
      "precision": 0.6473154362416107,
      "recall": 0.8254172015404364,
      "roc_auc": 0.9288910368283333,
      "pr_auc": 0.8324792992613164,
      "average_precision": 0.8324792992613164,
      "specificity": 0.858584,
      "balanced_accuracy_at_best_cut": 0.8469,
      "cut_headroom": 0.004899,
      "brier": 0.098926,
      "calibration_error": 0.079096,
      "balanced_accuracy_ci_low": 0.832822,
      "balanced_accuracy_ci_high": 0.850641,
      "train_f1": 0.7452357071213641,
      "train_accuracy": 0.8613158613158614,
      "train_balanced_accuracy": 0.8566458422663503,
      "train_val_gap": 0.014645
    },
    "train_time_sec": 14.23,
    "critic": {
      "failure_type": "underfitting",
      "evidence": "train_balanced_accuracy=0.8566, balanced_accuracy=0.8420 모두 목표 0.9194에 닿지 못하고 격차도 작음 — 용량 부족.",
      "direction": "모델 용량과 학습량을 늘린다: 반복 수 증가, 트리 깊이·리프 수 확대, 정규화 완화.",
      "concrete_changes": {
        "max_iter": 1200,
        "max_leaf_nodes": 63,
        "learning_rate": 0.08
      },
      "unsupported_claims": [],
      "source": "heuristic"
    }
  },
  {
    "iteration": 4,
    "model": "hist_gbdt",
    "hyperparams": {
      "max_iter": 1200,
      "learning_rate": 0.08,
      "max_depth": 6,
      "class_weight": {
        "0": 1.0,
        "1": 2.25
      },
      "max_leaf_nodes": 63
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
      "stopped_at_iter": 244,
      "max_iter": 1200
    },
    "unsupported_claims": [],
    "status": "ok",
    "error_type": null,
    "metrics": {
      "f1": 0.7254160363086233,
      "accuracy": 0.8513665677141979,
      "balanced_accuracy": 0.8408583854619351,
      "precision": 0.6499491697729584,
      "recall": 0.8207103123662816,
      "roc_auc": 0.9293028723585066,
      "pr_auc": 0.8334625885009739,
      "average_precision": 0.8334625885009739,
      "specificity": 0.861006,
      "balanced_accuracy_at_best_cut": 0.845,
      "cut_headroom": 0.004142,
      "brier": 0.098153,
      "calibration_error": 0.074648,
      "balanced_accuracy_ci_low": 0.83217,
      "balanced_accuracy_ci_high": 0.849411,
      "train_f1": 0.7586772687113467,
      "train_accuracy": 0.8695058695058695,
      "train_balanced_accuracy": 0.8653036823188651,
      "train_val_gap": 0.024445
    },
    "train_time_sec": 9.679,
    "critic": {
      "failure_type": "underfitting",
      "evidence": "train_balanced_accuracy=0.8653, balanced_accuracy=0.8409 모두 목표 0.9194에 닿지 못하고 격차도 작음 — 용량 부족.",
      "direction": "모델 용량과 학습량을 늘린다: 반복 수 증가, 트리 깊이·리프 수 확대, 정규화 완화.",
      "concrete_changes": {
        "max_iter": 1200,
        "max_leaf_nodes": 63,
        "learning_rate": 0.08
      },
      "unsupported_claims": [],
      "source": "heuristic"
    }
  },
  {
    "iteration": 5,
    "model": "logreg",
    "hyperparams": {
      "max_iter": 1200,
      "class_weight": {
        "0": 1.0,
        "1": 2.25
      }
    },
    "dropped_hyperparams": [
      "learning_rate",
      "max_depth",
      "max_leaf_nodes"
    ],
    "preprocessing": {
      "impute": "median",
      "scale": true,
      "missing_indicator": false,
      "missing_count": false
    },
    "internal_validation": {},
    "unsupported_claims": [],
    "status": "ok",
    "error_type": null,
    "metrics": {
      "f1": 0.6951791408741325,
      "accuracy": 0.8336574879721568,
      "balanced_accuracy": 0.8196857901599808,
      "precision": 0.6189044756179025,
      "recall": 0.7928968763371844,
      "roc_auc": 0.9078943050279747,
      "pr_auc": 0.7706933786024506,
      "average_precision": 0.7706933786024506,
      "specificity": 0.846475,
      "balanced_accuracy_at_best_cut": 0.8272,
      "cut_headroom": 0.007514,
      "brier": 0.113621,
      "calibration_error": 0.090755,
      "balanced_accuracy_ci_low": 0.81076,
      "balanced_accuracy_ci_high": 0.829381,
      "train_f1": 0.6845974989002702,
      "train_accuracy": 0.8287264537264537,
      "train_balanced_accuracy": 0.8109338516760737,
      "train_val_gap": -0.008752
    },
    "train_time_sec": 4.154,
    "critic": null
  }
]

An attempt's `metrics` may carry `<metric>_ci_low` and `<metric>_ci_high`: the 95%
bootstrap interval of that attempt's goal-metric score on its validation slice. It is the
spread of *one* measurement over *those* rows — not the selection effect above, which
points one way and is what the held-back score measures. Two attempts whose intervals
overlap are attempts this data does not separate, so a difference smaller than the
interval's width is not an improvement to report as one.

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
checks and from the operator, who has seen the raw file.

(없음 — 이 데이터에 대해 별도로 기록된 주의사항이 없습니다)

## What the executor can and cannot do

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

## Instructions

Write the report in Korean, as GitHub-flavoured Markdown. Keep identifiers, model
names, metric names, hyperparameter keys and error types in English exactly as they
appear in the data. Do not invent numbers that are not above.

Required structure:

1. `## 요약` — three or four sentences: was the goal met, the best score, how many
   attempts it took.
2. `## 시도별 경과` — a table with columns `iteration | model | 주요 하이퍼파라미터 |
   결과 | critic 진단`. One row per attempt, in order.
3. `## 최고 성능 구성` — the best configuration and its metrics, reproducibly stated.
   State the held-back test score here next to the validation score, and if the two
   differ, say that the difference is the size of the selection effect — the run made its
   choices on the validation number.
4. `## 원인 분석` — what actually limited performance, citing the numbers and the
   pattern across the Critic's verdicts. The `재계획` line under `Outcome` says how many
   verdicts exist. **If it says zero, there is no pattern to describe** — a first attempt
   that cleared the bar ends the run before the Critic ever runs. Say that the score is the
   first plan's, and that this run therefore says nothing either way about the
   diagnose-and-replan loop. Do not assemble a cause from the attempt's own metrics and
   present it as a diagnosis: no diagnosis was made.
5. `## 다음 단계 제안` — two to four concrete next actions. If the goal was not
   reached, this section carries the weight: say what you would try with more budget
   and why the evidence points there.

If the goal was not reached, say so plainly in the first sentence. Do not present the
best result as a success when it fell short of the threshold.

Every recommendation in `## 다음 단계 제안` has to survive the `Data caveats` section. A
caveat that names a column, a value or a split as untrustworthy rules out the actions that
depend on it — recommend the caveat's own resolution instead, and say what it would unblock.
A recommendation a caveat contradicts is worse than no recommendation: it reads as measured
advice, and the next person spends a week on it.

Where the write-up states the best score, state its interval next to it if the history
carries one. And in `## 원인 분석`, do not build a story out of movement between attempts
whose intervals overlap: say that those attempts are indistinguishable on this data, and
reason from the differences that are larger than the interval. A report that explains a
0.009 gain on a slice with a ±0.03 interval has explained the resample.

`## 최고 성능 구성` describes what ran, not what was planned. Quote `hyperparams` and
`preprocessing` from that attempt in the history — both are the applied values, read off
the estimator the executor built. The plan's `preprocessing` and the card's are requests:
the executor downgrades a strategy the model family cannot take, so quoting either can
announce imputation the run did not do. Only if the attempt carries no `preprocessing` at
all (an older run) fall back to the card's block, and say that is what you are quoting. An
attempt's `dropped_hyperparams` lists keys the executor refused, so that attempt does
not tell you whether those settings would have helped — that belongs in `## 원인 분석`,
never in `## 최고 성능 구성` as part of the winning configuration.

`plan.unsupported_claims` is weaker evidence: a substring check over the plan's prose,
so it can fire on a plan that merely *mentioned* an unavailable capability. Before you
attribute anything to it, read that iteration's `plan` text. If the plan relied on the
capability, say so and say what the attempt therefore did not test. If the plan only
noted the limitation, do not write that the attempt lost anything to it — the number is
simply what this executor achieves, and the missing capability belongs in
`## 다음 단계 제안` as something to build, not in `## 원인 분석` as something that failed.
