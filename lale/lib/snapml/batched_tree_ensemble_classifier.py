# Copyright 2022 IBM Corporation
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
try:
    import snapml  # type: ignore

    snapml_installed = True
except ImportError:
    snapml_installed = False

import pandas as pd

import lale.datasets.data_schemas
import lale.docstrings
import lale.operators


def _ensure_numpy(data):
    if isinstance(data, (pd.DataFrame, pd.Series)):
        return data.to_numpy()
    return lale.datasets.data_schemas.strip_schema(data)


class _BatchedTreeEnsembleClassifierImpl:
    def __init__(self, **hyperparams):
        assert (
            snapml_installed
        ), """Your Python environment does not have snapml installed. Install using: pip install snapml"""
        if hyperparams.get("base_ensemble", None) is None:
            from snapml import SnapBoostingMachineClassifier

            hyperparams["base_ensemble"] = SnapBoostingMachineClassifier()
        self._wrapped_model = snapml.BatchedTreeEnsembleClassifier(**hyperparams)

    def fit(self, X, y, **fit_params):
        X = _ensure_numpy(X)
        y = _ensure_numpy(y)
        self._wrapped_model.fit(X, y, **fit_params)
        return self

    def predict(self, X, **predict_params):
        X = _ensure_numpy(X)
        return self._wrapped_model.predict(X, **predict_params)

    def predict_proba(self, X, **predict_proba_params):
        X = _ensure_numpy(X)
        return self._wrapped_model.predict_proba(X, **predict_proba_params)

    def partial_fit(self, X, y, **fit_params):
        X = _ensure_numpy(X)
        y = _ensure_numpy(y)
        self._wrapped_model.partial_fit(X, y, **fit_params)
        return self


_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
            "type": "object",
            "additionalProperties": True,
            "relevantToOptimizer": [],
            "properties": {},
        }
    ],
}

_input_fit_schema = {
    "description": "Fit the base ensemble without batching.",
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
            "description": "The classes.",
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
            "default": None,
        },
    },
}

_input_predict_schema = {
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
        },
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
        },
    },
}

_output_predict_proba_schema = {
    "type": "array",
    "description": "The outer array is over samples aka rows.",
    "items": {
        "type": "array",
        "description": "The inner array contains probabilities corresponding to each class.",
        "items": {"type": "number"},
    },
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Batched Tree Ensemble Classifier`_ from `Snap ML`_.

.. _`Batched Tree Ensemble Classifier`: https://snapml.readthedocs.io/en/latest/batched_tree_ensembles.html
.. _`Snap ML`: https://www.zurich.ibm.com/snapml/
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.snapml.batched_tree_ensemble_classifier.html",
    "import_from": "snapml",
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


BatchedTreeEnsembleClassifier = lale.operators.make_operator(
    _BatchedTreeEnsembleClassifierImpl, _combined_schemas
)

lale.docstrings.set_docstrings(BatchedTreeEnsembleClassifier)
