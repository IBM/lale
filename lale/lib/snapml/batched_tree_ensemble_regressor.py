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

import lale.datasets.data_schemas
import lale.docstrings
import lale.operators


class _BatchedTreeEnsembleRegressorImpl:
    def __init__(
        self,
        base_ensemble=None,
        max_sub_ensembles=10,
        inner_lr_scaling=0.5,
        outer_lr_scaling=0.5,
    ):
        assert (
            snapml_installed
        ), """Your Python environment does not have snapml installed. Install using: pip install snapml"""
        if base_ensemble is None:
            from snapml import SnapBoostingMachineRegressor

            base_ensemble = SnapBoostingMachineRegressor()
        self._hyperparams = {
            "base_ensemble": base_ensemble,
            "max_sub_ensembles": max_sub_ensembles,
            "inner_lr_scaling": inner_lr_scaling,
            "outer_lr_scaling": outer_lr_scaling,
        }
        self._wrapped_model = snapml.BatchedTreeEnsembleRegressor(**self._hyperparams)

    def fit(self, X, y, **fit_params):
        X = lale.datasets.data_schemas.strip_schema(X)
        y = lale.datasets.data_schemas.strip_schema(y)
        self._wrapped_model.fit(X, y, **fit_params)
        return self

    def predict(self, X, **predict_params):
        X = lale.datasets.data_schemas.strip_schema(X)
        return self._wrapped_model.predict(X, **predict_params)

    def predict_proba(self, X, **predict_proba_params):
        X = lale.datasets.data_schemas.strip_schema(X)
        return self._wrapped_model.predict_proba(X, **predict_proba_params)

    def partial_fit(self, X, y, **fit_params):
        X = lale.datasets.data_schemas.strip_schema(X)
        y = lale.datasets.data_schemas.strip_schema(y)
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
    "description": """`Batched Tree Ensemble Regressor`_ from `Snap ML`_.

.. _`Batched Tree Ensemble Regressor`: https://snapml.readthedocs.io/en/latest/batched_tree_ensembles.html
.. _`Snap ML`: https://www.zurich.ibm.com/snapml/
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.snapml.batched_tree_ensemble_regressor.html",
    "import_from": "snapml",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "regressor"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
    },
}


BatchedTreeEnsembleRegressor = lale.operators.make_operator(
    _BatchedTreeEnsembleRegressorImpl, _combined_schemas
)

lale.docstrings.set_docstrings(BatchedTreeEnsembleRegressor)
