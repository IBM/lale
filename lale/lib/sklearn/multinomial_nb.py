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

import sklearn.naive_bayes

import lale.docstrings
import lale.operators


class MultinomialNBImpl:
    def __init__(self, alpha=1.0, fit_prior=True, class_prior=None):
        self._hyperparams = {
            "alpha": alpha,
            "fit_prior": fit_prior,
            "class_prior": class_prior,
        }
        self._wrapped_model = sklearn.naive_bayes.MultinomialNB(**self._hyperparams)

    def fit(self, X, y=None):
        self._wrapped_model.fit(X, y)
        return self

    def predict(self, X):
        return self._wrapped_model.predict(X)

    def predict_proba(self, X):
        return self._wrapped_model.predict_proba(X)


_hyperparams_schema = {
    "description": "Naive Bayes classifier for multinomial models",
    "allOf": [
        {
            "type": "object",
            "required": ["alpha", "fit_prior"],
            "relevantToOptimizer": ["alpha", "fit_prior"],
            "properties": {
                "alpha": {
                    "type": "number",
                    "distribution": "loguniform",
                    "minimumForOptimizer": 1e-10,
                    "maximumForOptimizer": 1.0,
                    "default": 1.0,
                    "description": "Additive (Laplace/Lidstone) smoothing parameter",
                },
                "fit_prior": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to learn class prior probabilities or not.",
                },
                "class_prior": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "number"}},
                        {"enum": [None]},
                    ],
                    "default": None,
                    "description": "Prior probabilities of the classes. If specified the priors are not",
                },
            },
        }
    ],
}
_input_fit_schema = {
    "description": "Fit Naive Bayes classifier according to X, y",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
            "description": "Training vectors, where n_samples is the number of samples and n_features is the number of features.",
        },
        "y": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
                {"type": "array", "items": {"type": "boolean"}},
            ],
            "description": "Target values.",
        },
        "sample_weight": {
            "anyOf": [{"type": "array", "items": {"type": "number"}}, {"enum": [None]}],
            "default": None,
            "description": "Weights applied to individual samples (1. for unweighted).",
        },
    },
}
_input_predict_schema = {
    "description": "Perform classification on an array of test vectors X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
    },
}
_output_predict_schema = {
    "description": "Perform classification on an array of test vectors X.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "string"}},
        {"type": "array", "items": {"type": "boolean"}},
    ],
}

_input_predict_proba_schema = {
    "description": "Perform classification on an array of test vectors X.",
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}},
    },
}
_output_predict_proba_schema = {
    "description": "Returns the probability of the samples for each class in the model. The columns correspond to the classes in sorted order, as they appear in the attribute classes_.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Multinomial Naive Bayes`_ classifier from scikit-learn.

.. _`Multinomial Naive Bayes`: https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.multinomial_naive_bayes.html",
    "import_from": "sklearn.naive_bayes",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
    },
}

lale.docstrings.set_docstrings(MultinomialNBImpl, _combined_schemas)

MultinomialNB = lale.operators.make_operator(MultinomialNBImpl, _combined_schemas)
