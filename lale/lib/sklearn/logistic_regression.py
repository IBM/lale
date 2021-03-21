# Copyright 2019 IBM Corporation
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
import typing

import sklearn
import sklearn.linear_model

import lale.docstrings
import lale.operators
from lale.schemas import AnyOf, Enum, Float, Null

_input_fit_schema = {
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        "y": {
            "description": "Target class labels; the array is over samples.",
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
                {"type": "array", "items": {"type": "boolean"}},
            ],
        },
    },
}

_input_predict_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        }
    },
}

_output_predict_schema = {
    "description": "Predicted class label per sample.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "string"}},
        {"type": "array", "items": {"type": "boolean"}},
    ],
}

_input_predict_proba_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        }
    },
}

_output_predict_proba_schema = {
    "description": "Probability of the sample for each class in the model.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_input_decision_function_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        }
    },
}

_output_decision_function_schema = {
    "description": "Confidence scores for samples for each class in the model.",
    "anyOf": [
        {
            "description": "In the multi-way case, score per (sample, class) combination.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        {
            "description": "In the binary case, score for `self._classes[1]`.",
            "type": "array",
            "items": {"type": "number"},
        },
    ],
}

_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "additionalProperties": False,
            "required": [
                "penalty",
                "dual",
                "tol",
                "C",
                "fit_intercept",
                "intercept_scaling",
                "class_weight",
                "random_state",
                "solver",
                "max_iter",
                "multi_class",
                "verbose",
                "warm_start",
                "n_jobs",
            ],
            "relevantToOptimizer": [
                "dual",
                "tol",
                "fit_intercept",
                "solver",
                "multi_class",
                "intercept_scaling",
                "max_iter",
            ],
            "properties": {
                "solver": {
                    "description": """Algorithm for optimization problem.
- For small datasets, 'liblinear' is a good choice, whereas 'sag' and
  'saga' are faster for large ones.
- For multiclass problems, only 'newton-cg', 'sag', 'saga' and 'lbfgs'
  handle multinomial loss; 'liblinear' is limited to one-versus-rest
  schemes.
- 'newton-cg', 'lbfgs', 'sag' and 'saga' handle L2 or no penalty
- 'liblinear' and 'saga' also handle L1 penalty
- 'saga' also supports 'elasticnet' penalty
- 'liblinear' does not support setting penalty='none'
Note that 'sag' and 'saga' fast convergence is only guaranteed on
features with approximately the same scale. You can
preprocess the data with a scaler from sklearn.preprocessing.""",
                    "enum": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                    "default": "liblinear",
                },
                "penalty": {
                    "description": "Norm used in the penalization.",
                    "enum": ["l1", "l2"],
                    "default": "l2",
                },
                "dual": {
                    "description": """Dual or primal formulation.
Dual formulation is only implemented for l2 penalty with liblinear solver. Prefer dual=False when n_samples > n_features.""",
                    "type": "boolean",
                    "default": False,
                },
                "C": {
                    "description": "Inverse regularization strength. Smaller values specify "
                    "stronger regularization.",
                    "type": "number",
                    "distribution": "loguniform",
                    "minimum": 0.0,
                    "exclusiveMinimum": True,
                    "default": 1.0,
                    "minimumForOptimizer": 0.03125,
                    "maximumForOptimizer": 32768,
                },
                "tol": {
                    "description": "Tolerance for stopping criteria.",
                    "type": "number",
                    "distribution": "loguniform",
                    "minimum": 0.0,
                    "exclusiveMinimum": True,
                    "default": 0.0001,
                    "minimumForOptimizer": 1e-08,
                    "maximumForOptimizer": 0.01,
                },
                "fit_intercept": {
                    "description": "Specifies whether a constant (bias or intercept) should be "
                    "added to the decision function.",
                    "type": "boolean",
                    "default": True,
                },
                "intercept_scaling": {
                    "description": """Useful only when the solver 'liblinear' is used
and self.fit_intercept is set to True. In this case, X becomes
[X, self.intercept_scaling],
i.e. a "synthetic" feature with constant value equal to
intercept_scaling is appended to the instance vector.
The intercept becomes "intercept_scaling * synthetic_feature_weight".
Note! the synthetic feature weight is subject to l1/l2 regularization
as all other features.
To lessen the effect of regularization on synthetic feature weight
(and therefore on the intercept) intercept_scaling has to be increased.""",
                    "type": "number",
                    "distribution": "uniform",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 1.0,
                },
                "class_weight": {
                    "anyOf": [
                        {
                            "description": "By default, all classes have weight 1.",
                            "enum": [None],
                        },
                        {
                            "description": """Uses the values of y to automatically adjust weights inversely
proportional to class frequencies in the input data as "n_samples / (n_classes * np.bincount(y))".""",
                            "enum": ["balanced"],
                        },
                        {
                            "description": 'Weights associated with classes in the form "{class_label: weight}".',
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                            "forOptimizer": False,
                        },
                    ],
                    "default": None,
                },
                "random_state": {
                    "description": "Seed of pseudo-random number generator for shuffling data when solver == ‘sag’, ‘saga’ or ‘liblinear’.",
                    "anyOf": [
                        {
                            "description": "RandomState used by np.random",
                            "enum": [None],
                        },
                        {
                            "description": "Use the provided random state, only affecting other users of that same random state instance.",
                            "laleType": "numpy.random.RandomState",
                        },
                        {"description": "Explicit seed.", "type": "integer"},
                    ],
                    "default": None,
                },
                "max_iter": {
                    "description": "Maximum number of iterations for solvers to converge.",
                    "type": "integer",
                    "distribution": "uniform",
                    "minimum": 1,
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 1000,
                    "default": 100,
                },
                "multi_class": {
                    "description": """Approach for handling a multi-class problem.
If the option chosen is 'ovr', then a binary problem is fit for each
label. For 'multinomial' the loss minimised is the multinomial loss fit
across the entire probability distribution, *even when the data is
binary*. 'multinomial' is unavailable when solver='liblinear'.
'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
and otherwise selects 'multinomial'.""",
                    "enum": ["ovr", "multinomial", "auto"],
                    "default": "ovr",
                },
                "verbose": {
                    "description": "For the liblinear and lbfgs solvers set verbose to any positive "
                    "number for verbosity.",
                    "type": "integer",
                    "default": 0,
                },
                "warm_start": {
                    "description": """When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.
Useless for liblinear solver.""",
                    "type": "boolean",
                    "default": False,
                },
                "n_jobs": {
                    "description": """Number of CPU cores when parallelizing over classes if
multi_class is ovr.  This parameter is ignored when the "solver" is
set to 'liblinear' regardless of whether 'multi_class' is specified or
not.""",
                    "anyOf": [
                        {
                            "description": "1 unless in joblib.parallel_backend context.",
                            "enum": [None],
                        },
                        {"description": "Use all processors.", "enum": [-1]},
                        {
                            "description": "Number of CPU cores.",
                            "type": "integer",
                            "minimum": 1,
                        },
                    ],
                    "default": None,
                },
            },
        },
        {
            "description": "The newton-cg, sag, and lbfgs solvers support only l2 or no penalties.",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "solver": {"not": {"enum": ["newton-cg", "sag", "lbfgs"]}}
                    },
                },
                {"type": "object", "properties": {"penalty": {"enum": ["l2", "none"]}}},
            ],
        },
        {
            "description": "The dual formulation is only implemented for l2 "
            "penalty with the liblinear solver.",
            "anyOf": [
                {"type": "object", "properties": {"dual": {"enum": [False]}}},
                {
                    "type": "object",
                    "properties": {
                        "penalty": {"enum": ["l2"]},
                        "solver": {"enum": ["liblinear"]},
                    },
                },
            ],
        },
        {
            "description": "The multi_class multinomial option is unavailable when the solver is liblinear.",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"multi_class": {"not": {"enum": ["multinomial"]}}},
                },
                {
                    "type": "object",
                    "properties": {"solver": {"not": {"enum": ["liblinear"]}}},
                },
            ],
        },
        # {
        #     "description": "Penalty elasticnet is only supported by the saga solver.",
        #     "anyOf": [
        #         {
        #             "type": "object",
        #             "properties": {"penalty": {"not": {"enum": ["elasticnet"]}}},
        #         },
        #         {"type": "object", "properties": {"solver": {"enum": ["saga"]}}},
        #     ],
        # },
        # {
        #     "description": "When penalty is elasticnet, l1_ratio must be between 0 and 1.",
        #     "anyOf": [
        #         {
        #             "type": "object",
        #             "properties": {"penalty": {"not": {"enum": ["elasticnet"]}}},
        #         },
        #         {
        #             "type": "object",
        #             "properties": {
        #                 "l1_ratio": {
        #                     "type": "number",
        #                     "minimum": 0.0,
        #                     "exclusiveMinimum": True,
        #                     "maximum": 1.0,
        #                 }
        #             },
        #         },
        #     ],
        # },
        # {
        #     "description": "Solver liblinear does not support penalty none.",
        #     "anyOf": [
        #         {
        #             "type": "object",
        #             "properties": {"solver": {"not": {"enum": ["liblinear"]}}},
        #         },
        #         {
        #             "type": "object",
        #             "properties": {"penalty": {"enum": ["l1", "l2", "elasticnet"]}},
        #         },
        #     ],
        # },
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Logistic regression`_ linear model from scikit-learn for classification.

.. _`Logistic regression`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.logistic_regression.html",
    "import_from": "sklearn.linear_model",
    "type": "object",
    "tags": {
        "pre": ["~categoricals"],
        "op": ["estimator", "classifier", "interpretable"],
        "post": ["probabilities"],
    },
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
        "input_decision_function": _input_decision_function_schema,
        "output_decision_function": _output_decision_function_schema,
    },
}

LogisticRegression = lale.operators.make_operator(
    sklearn.linear_model.LogisticRegression, _combined_schemas
)

if sklearn.__version__ >= "0.21":
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.linear_model.LogisticRegression.html
    # new: https://scikit-learn.org/0.21/modules/generated/sklearn.linear_model.LogisticRegression.html
    LogisticRegression = typing.cast(
        lale.operators.PlannedIndividualOp,
        LogisticRegression.customize_schema(
            penalty=Enum(
                values=["l1", "l2", "elasticnet", "none"],
                desc="Norm used in the penalization.",
                default="l2",
            ),
        ),
    )

if sklearn.__version__ >= "0.22":
    # old: https://scikit-learn.org/0.21/modules/generated/sklearn.linear_model.LogisticRegression.html
    # new: https://scikit-learn.org/0.23/modules/generated/sklearn.linear_model.LogisticRegression.html
    LogisticRegression = typing.cast(
        lale.operators.PlannedIndividualOp,
        LogisticRegression.customize_schema(
            solver=Enum(
                values=["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                desc="Algorithm for optimization problem.",
                default="lbfgs",
            ),
            multi_class=Enum(
                values=["auto", "ovr", "multinomial"],
                desc="If the option chosen is `ovr`, then a binary problem is fit for each label. For `multinomial` the loss minimised is the multinomial loss fit across the entire probability distribution, even when the data is binary. `multinomial` is unavailable when solver=`liblinear`. `auto` selects `ovr` if the data is binary, or if solver=`liblinear`, and otherwise selects `multinomial`.",
                default="auto",
            ),
            l1_ratio=AnyOf(
                types=[Float(minimum=0.0, maximum=1.0), Null()],
                desc="The Elastic-Net mixing parameter.",
                default=None,
            ),
        ),
    )

lale.docstrings.set_docstrings(LogisticRegression)
