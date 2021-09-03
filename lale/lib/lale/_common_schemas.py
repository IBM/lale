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

schema_estimator = {
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

schema_scoring = {
    "description": "Scorer object, or known scorer named by string.",
    "anyOf": [
        {
            "enum": [None],
            "description": "When not specified, use `accuracy` for classification tasks and `r2` for regression.",
        },
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
    "default": None,
}

schema_best_score = {
    "description": """The best score for the specified scorer.

Given that higher scores are better, passing ``(best_score - score)``
as a loss to the minimizing optimizer will maximize the score.
By specifying best_score, the loss can be >=0, where 0 is the best loss.""",
    "type": "number",
    "default": 0.0,
}
