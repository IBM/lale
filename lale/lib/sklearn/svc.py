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

import sklearn
import sklearn.svm

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "kernel",
                "degree",
                "gamma",
                "shrinking",
                "tol",
                "cache_size",
                "max_iter",
                "decision_function_shape",
            ],
            "relevantToOptimizer": [
                "kernel",
                "degree",
                "gamma",
                "shrinking",
                "probability",
                "tol",
            ],
            "properties": {
                "C": {
                    "description": "Penalty parameter C of the error term.",
                    "type": "number",
                    "distribution": "loguniform",
                    "minimum": 0.0,
                    "exclusiveMinimum": True,
                    "default": 1.0,
                    "minimumForOptimizer": 0.03125,
                    "maximumForOptimizer": 32768,
                },
                "kernel": {
                    "anyOf": [
                        {"enum": ["precomputed"], "forOptimizer": False},
                        {"enum": ["linear", "poly", "rbf", "sigmoid"]},
                        {"laleType": "callable", "forOptimizer": False},
                    ],
                    "default": "rbf",
                    "description": "Specifies the kernel type to be used in the algorithm.",
                },
                "degree": {
                    "type": "integer",
                    "minimum": 0,
                    "minimumForOptimizer": 2,
                    "maximumForOptimizer": 5,
                    "default": 3,
                    "description": "Degree of the polynomial kernel function ('poly').",
                },
                "gamma": {
                    "anyOf": [
                        {
                            "type": "number",
                            "minimum": 0.0,
                            "exclusiveMinimum": True,
                            "minimumForOptimizer": 3.0517578125e-05,
                            "maximumForOptimizer": 8,
                            "distribution": "loguniform",
                        },
                        {"enum": ["auto", "auto_deprecated", "scale"]},
                    ],
                    "default": "auto_deprecated",  # going to change to 'scale' from sklearn 0.22.
                    "description": "Kernel coefficient for 'rbf', 'poly', and 'sigmoid'.",
                },
                "coef0": {
                    "type": "number",
                    "minimumForOptimizer": -1,
                    "maximumForOptimizer": 1,
                    "default": 0.0,
                    "description": "Independent term in kernel function.",
                },
                "shrinking": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to use the shrinking heuristic.",
                },
                "probability": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to enable probability estimates.",
                },
                "tol": {
                    "type": "number",
                    "distribution": "loguniform",
                    "minimum": 0.0,
                    "exclusiveMinimum": True,
                    "maximumForOptimizer": 0.01,
                    "default": 0.001,
                    "description": "Tolerance for stopping criteria.",
                },
                "cache_size": {
                    "type": "integer",
                    "minimum": 0,
                    "maximumForOptimizer": 1000,
                    "default": 200,
                    "description": "Specify the size of the kernel cache (in MB).",
                },
                "class_weight": {
                    "anyOf": [
                        {
                            "description": "By default, all classes have weight 1.",
                            "enum": [None],
                        },
                        {
                            "description": "Adjust weights by inverse frequency.",
                            "enum": ["balanced"],
                        },
                        {
                            "description": "Dictionary mapping class labels to weights.",
                            "type": "object",
                            "additionalProperties": {"type": "number"},
                            "forOptimizer": False,
                        },
                    ],
                    "default": None,
                },
                "verbose": {
                    "type": "boolean",
                    "default": False,
                    "description": "Enable verbose output.",
                },
                "max_iter": {
                    "type": "integer",
                    "minimumForOptimizer": 1,
                    "maximumForOptimizer": 1000,
                    "default": -1,
                    "description": "Hard limit on iterations within solver, or -1 for no limit.",
                },
                "decision_function_shape": {
                    "enum": ["ovo", "ovr"],
                    "default": "ovr",
                    "description": "Whether to return a one-vs-rest ('ovr') decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one (‘ovo’) decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2).",
                },
                "random_state": {
                    "description": "Seed of pseudo-random number generator.",
                    "anyOf": [
                        {"laleType": "numpy.random.RandomState"},
                        {
                            "description": "RandomState used by np.random",
                            "enum": [None],
                        },
                        {"description": "Explicit seed.", "type": "integer"},
                    ],
                    "default": None,
                },
            },
        },
        {
            "description": "coef0 only significant in kernel ‘poly’ and ‘sigmoid’.",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"kernel": {"enum": ["poly", "sigmoid"]}},
                },
                {"type": "object", "properties": {"coef0": {"enum": [0.0]}}},
            ],
        },
    ]
}

_input_fit_schema = {
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "description": "The outer array is over samples aka rows.",
            "items": {
                "type": "array",
                "description": "The inner array is over features aka columns.",
                "items": {"type": "number"},
            },
        },
        "y": {
            "description": "The predicted classes.",
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
                {"type": "array", "items": {"type": "boolean"}},
            ],
        },
        "sample_weight": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"enum": [None], "description": "Samples are equally weighted."},
            ],
            "description": "Sample weights.",
        },
    },
}

_input_predict_schema = {
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "description": "The outer array is over samples aka rows.",
            "items": {
                "type": "array",
                "description": "The inner array is over features aka columns.",
                "items": {"type": "number"},
            },
        }
    },
}

_output_predict_schema = {
    "description": "The predicted classes.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "string"}},
        {"type": "array", "items": {"type": "boolean"}},
    ],
}

_input_predict_proba_schema = {
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "description": "The outer array is over samples aka rows.",
            "items": {
                "type": "array",
                "description": "The inner array is over features aka columns.",
                "items": {"type": "number"},
            },
        }
    },
}

_output_predict_proba_schema = {
    "type": "array",
    "description": "The outer array is over samples aka rows.",
    "items": {
        "type": "array",
        "description": "The inner array has items corresponding to each class.",
        "items": {"type": "number"},
    },
}

_input_decision_function_schema = {
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "description": "The outer array is over samples aka rows.",
            "items": {
                "type": "array",
                "description": "The inner array is over features aka columns.",
                "items": {"type": "number"},
            },
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

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Support Vector Classification`_ from scikit-learn.

.. _`Support Vector Classification`: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.svc.html",
    "import_from": "sklearn.svm",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
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

SVC: lale.operators.PlannedIndividualOp
SVC = lale.operators.make_operator(sklearn.svm.SVC, _combined_schemas)

if sklearn.__version__ >= "0.22":
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.svm.SVC.html
    # new: https://scikit-learn.org/0.23/modules/generated/sklearn.svm.SVC.html
    from lale.schemas import AnyOf, Bool, Enum, Float

    SVC = SVC.customize_schema(
        gamma=AnyOf(
            types=[
                Enum(["scale", "auto"]),
                Float(
                    minimum=0.0,
                    exclusiveMinimum=True,
                    minimumForOptimizer=3.0517578125e-05,
                    maximumForOptimizer=8,
                    distribution="loguniform",
                ),
            ],
            desc="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.",
            default="scale",
        ),
        break_ties=Bool(
            desc="If true, decision_function_shape='ovr', and number of classes > 2, predict will break ties according to the confidence values of decision_function; otherwise the first class among the tied classes is returned.",
            default=False,
        ),
    )


lale.docstrings.set_docstrings(SVC)
