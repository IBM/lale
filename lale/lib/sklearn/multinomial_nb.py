# Copyright 2019-2022 IBM Corporation
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

from ._common_schemas import (
    schema_1D_cats,
    schema_2D_numbers,
    schema_sample_weight,
    schema_X_numbers,
)

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
                    "description": "Additive (Laplace/Lidstone) smoothing parameter (0 for no smoothing).",
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
                    "description": "Prior probabilities of the classes. If specified the priors are not adjusted according to the data.",
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
        "X": schema_2D_numbers,
        "y": schema_1D_cats,
        "sample_weight": schema_sample_weight,
    },
}

_input_partial_fit_schema = {
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": schema_2D_numbers,
        "y": schema_1D_cats,
        "classes": schema_1D_cats,
        "sample_weight": schema_sample_weight,
    },
}

_output_predict_proba_schema = {
    "description": "Returns the probability of the samples for each class in the model. The columns correspond to the classes in sorted order, as they appear in the attribute `classes_`.",
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
        "input_partial_fit": _input_partial_fit_schema,
        "input_predict": schema_X_numbers,
        "output_predict": schema_1D_cats,
        "input_predict_proba": schema_X_numbers,
        "output_predict_proba": _output_predict_proba_schema,
    },
}


MultinomialNB = lale.operators.make_operator(
    sklearn.naive_bayes.MultinomialNB, _combined_schemas
)

lale.docstrings.set_docstrings(MultinomialNB)
