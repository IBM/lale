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

import sklearn.svm

import lale.docstrings
import lale.operators

_hyperparams_schema = {
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "penalty",
                "loss",
                "dual",
                "tol",
                "C",
                "multi_class",
                "fit_intercept",
                "intercept_scaling",
                "class_weight",
                "verbose",
                "random_state",
                "max_iter",
            ],
            "relevantToOptimizer": [
                "penalty",
                "loss",
                "dual",
                "tol",
                "C",
                "multi_class",
                "fit_intercept",
            ],
            "properties": {
                "penalty": {
                    "description": "Norm used in the penalization.",
                    "enum": ["l1", "l2"],
                    "default": "l2",
                },
                "loss": {
                    "description": "Loss function.",
                    "enum": ["hinge", "squared_hinge"],
                    "default": "squared_hinge",
                },
                "dual": {
                    "type": "boolean",
                    "default": True,
                    "description": "Select the algorithm to either solve the dual or primal optimization problem.",
                },
                "tol": {
                    "type": "number",
                    "distribution": "loguniform",
                    "minimum": 0.0,
                    "exclusiveMinimum": True,
                    "maximumForOptimizer": 0.01,
                    "default": 0.0001,
                    "description": "Tolerance for stopping criteria.",
                },
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
                "multi_class": {
                    "description": "Determines the multi-class strategy if `y` contains more than two classes.",
                    "enum": ["ovr", "crammer_singer"],
                    "default": "ovr",
                },
                "fit_intercept": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to calculate the intercept for this model.",
                },
                "intercept_scaling": {
                    "type": "number",
                    "description": "Append a constant feature with constant value "
                    "intercept_scaling to the instance vector.",
                    "minimum": 0.0,
                    "exclusiveMinimum": True,
                    "maximumForOptimizer": 1.0,
                    "default": 1.0,
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
                    "type": "integer",
                    "default": 0,
                    "description": "Enable verbose output.",
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
                "max_iter": {
                    "type": "integer",
                    "minimum": 1,
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 1000,
                    "default": 1000,
                    "description": "The maximum number of iterations to be run.",
                },
            },
        },
        {
            "description": "The combination of penalty=`l1` and loss=`hinge` is not supported. "
            "If multi_class='crammer_singer', the options loss, penalty and dual will be ignored.",
            "anyOf": [
                {"type": "object", "properties": {"penalty": {"enum": ["l2"]}}},
                {"type": "object", "properties": {"loss": {"enum": ["squared_hinge"]}}},
                {
                    "type": "object",
                    "properties": {"multi_class": {"enum": ["crammer_singer"]}},
                },
            ],
        },
        {
            "description": "The combination of penalty=`l2` and loss=`hinge` "
            "is not supported when dual=False. If multi_class='crammer_singer', the options loss, "
            "penalty and dual will be ignored.",
            "anyOf": [
                {"type": "object", "properties": {"penalty": {"enum": ["l1"]}}},
                {"type": "object", "properties": {"loss": {"enum": ["squared_hinge"]}}},
                {"type": "object", "properties": {"dual": {"enum": [True]}}},
                {
                    "type": "object",
                    "properties": {"multi_class": {"enum": ["crammer_singer"]}},
                },
            ],
        },
        {
            "description": "The combination of penalty=`l1` and "
            "loss=`squared_hinge` is not supported when dual=True.  If multi_class='crammer_singer', "
            "the options loss, penalty and dual will be ignored.",
            "anyOf": [
                {"type": "object", "properties": {"penalty": {"enum": ["l2"]}}},
                {"type": "object", "properties": {"loss": {"enum": ["hinge"]}}},
                {"type": "object", "properties": {"dual": {"enum": [False]}}},
                {
                    "type": "object",
                    "properties": {"multi_class": {"enum": ["crammer_singer"]}},
                },
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
    "description": "Predict class labels for samples in X.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "string"}},
        {"type": "array", "items": {"type": "boolean"}},
    ],
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
    "description": """`Linear Support Vector Classification`_ from scikit-learn.

.. _`Linear Support Vector Classification`: https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.linear_svc.html",
    "import_from": "sklearn.svm",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_decision_function": _input_decision_function_schema,
        "output_decision_function": _output_decision_function_schema,
    },
}


LinearSVC = lale.operators.make_operator(sklearn.svm.LinearSVC, _combined_schemas)

lale.docstrings.set_docstrings(LinearSVC)
