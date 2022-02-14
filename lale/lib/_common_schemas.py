# Copyright 2021 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any, Dict

JSON_TYPE = Dict[str, Any]

schema_estimator: JSON_TYPE = {
    "description": "Planned Lale individual operator or pipeline.",
    "anyOf": [
        {"laleType": "operator"},
        {
            "enum": [None],
            "description": "lale.lib.sklearn.LogisticRegression",
        },
    ],
    "default": None,
}

# schemas used by many optimizers
schema_scoring_item: JSON_TYPE = {
    "description": "Scorer object, or known scorer named by string.",
    "anyOf": [
        {
            "description": """Callable with signature ``scoring(estimator, X, y)`` as documented in `sklearn scoring`_.

The callable has to return a scalar value, such that a higher score is better.
This may be created from one of the `sklearn metrics`_ using `make_scorer`_.
Or it can be one of the scoring callables returned by the factory
functions in `lale.lib.aif360 metrics`_, for example,
``symmetric_disparate_impact(**fairness_info)``.
Or it can be a completely custom user-written Python callable.

.. _`sklearn scoring`: https://scikit-learn.org/stable/modules/model_evaluation.html#the-scoring-parameter-defining-model-evaluation-rules
.. _`make_scorer`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer.
.. _`sklearn metrics`: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
.. _`lale.lib.aif360 metrics`: https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.html#metrics
""",
            "laleType": "callable",
        },
        {
            "description": "Known scorer for classification task.",
            "enum": [
                "accuracy",
                "explained_variance",
                "max_error",
                "roc_auc",
                "roc_auc_ovr",
                "roc_auc_ovo",
                "roc_auc_ovr_weighted",
                "roc_auc_ovo_weighted",
                "balanced_accuracy",
                "average_precision",
                "neg_log_loss",
                "neg_brier_score",
            ],
        },
        {
            "description": "Known scorer for regression task.",
            "enum": [
                "r2",
                "neg_mean_squared_error",
                "neg_mean_absolute_error",
                "neg_root_mean_squared_error",
                "neg_mean_squared_log_error",
                "neg_median_absolute_error",
            ],
        },
    ],
}

schema_scoring_single: JSON_TYPE = {
    "description": "Scorer object, or known scorer named by string.",
    "anyOf": [
        {
            "enum": [None],
            "description": "When not specified, use `accuracy` for classification tasks and `r2` for regression.",
        },
        schema_scoring_item,
    ],
}

schema_scoring_list: JSON_TYPE = {
    "description": "A list of Scorer objects, or known scorers named by string.  The optimizer may take the order into account.",
    "type": "array",
    "items": schema_scoring_item,
}

schema_scoring: JSON_TYPE = {
    "description": "Either a single or a list of (Scorer objects, or known scorers named by string).",
    "anyOf": [schema_scoring_single, schema_scoring_list],
}

schema_best_score_single: JSON_TYPE = {
    "description": """The best score for the specified scorer.

Given that higher scores are better, passing ``(best_score - score)``
as a loss to the minimizing optimizer will maximize the score.
By specifying best_score, the loss can be >=0, where 0 is the best loss.""",
    "type": "number",
    "default": 0.0,
}

schema_best_score: JSON_TYPE = {
    "description": """The best score for the specified scorer.

Given that higher scores are better, passing ``(best_score - score)``
as a loss to the minimizing optimizer will maximize the score.
By specifying best_score, the loss can be >=0, where 0 is the best loss.""",
    "default": 0.0,
    "anyOf": [
        schema_best_score_single,
        {
            "description": """The best score for each specified scorer.
            If not enough are specified, the remainder are assumed to be the default.

            Given that higher scores are better, passing ``(best_score - score)``
            as a loss to the minimizing optimizer will maximize the score.
            By specifying best_score, the loss can be >=0, where 0 is the best loss.""",
            "type": "array",
            "items": schema_best_score_single,
        },
    ],
}


def check_scoring_best_score_constraint(scoring=None, best_score=0) -> None:
    if isinstance(best_score, list):
        if isinstance(scoring, list):
            if len(scoring) < len(best_score):
                raise ValueError(
                    f"Error: {len(best_score)} best scores were specified, but there are only {len(scoring)} scorers."
                )
        else:
            raise ValueError(
                f"Error: {len(best_score)} best scores were specified, but there is only one scorer."
            )


schema_simple_cv: JSON_TYPE = {
    "description": "Number of folds for cross-validation.",
    "type": "integer",
    "minimum": 2,
    "default": 5,
    "minimumForOptimizer": 3,
    "maximumForOptimizer": 4,
    "distribution": "uniform",
}

schema_cv: JSON_TYPE = {
    "description": """Cross-validation as integer or as object that has a split function.

The fit method performs cross validation on the input dataset for per
trial, and uses the mean cross validation performance for optimization.
This behavior is also impacted by the handle_cv_failure flag.
""",
    "anyOf": [
        schema_simple_cv,
        {
            "not": {"type": "integer"},
            "forOptimizer": False,
            "description": "Object with split function: generator yielding (train, test) splits as arrays of indices. Can use any of the iterators from https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators",
        },
    ],
    "default": 5,
}

schema_max_opt_time: JSON_TYPE = {
    "description": "Maximum amount of time in seconds for the optimization.",
    "anyOf": [
        {"type": "number", "minimum": 0.0},
        {"description": "No runtime bound.", "enum": [None]},
    ],
    "default": None,
}
