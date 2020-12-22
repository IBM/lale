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

import aif360.algorithms.preprocessing
import aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools
import aif360.datasets
import numpy as np
import pandas as pd

import lale.docstrings
import lale.operators

from .protected_attributes_encoder import ProtectedAttributesEncoder
from .util import (
    _categorical_fairness_properties,
    _group_flag,
    _ndarray_to_series,
    _PandasToDatasetConverter,
    dataset_to_pandas,
)


class OptimPreprocImpl:
    def __init__(
        self,
        favorable_labels,
        protected_attributes,
        optimizer,
        optim_options,
        verbose=0,
        seed=None,
    ):
        if optimizer is None:
            optimizer = (
                aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools.OptTools
            )
        self._hyperparams = {
            "favorable_labels": favorable_labels,
            "protected_attributes": protected_attributes,
            "optimizer": optimizer,
            "optim_options": optim_options,
            "verbose": verbose,
            "seed": seed,
        }
        prot_attr_names = [pa["feature"] for pa in protected_attributes]
        self._unprivileged_groups = [{name: 0 for name in prot_attr_names}]
        self._privileged_groups = [{name: 1 for name in prot_attr_names}]
        self._prot_attr_enc = ProtectedAttributesEncoder(
            protected_attributes=protected_attributes
        )
        self._pandas_to_dataset = _PandasToDatasetConverter(
            favorable_label=1,
            unfavorable_label=0,
            protected_attribute_names=prot_attr_names,
        )

    def _encode(self, X, y=None):
        encoded_X = self._prot_attr_enc.transform(X)
        if y is None:
            encoded_y = pd.Series(
                data=0.0, index=X.index, dtype=np.float64, name=self._class_attr,
            )
        else:
            if isinstance(y, np.ndarray):
                encoded_y = _ndarray_to_series(y, X.shape[1])
            else:
                encoded_y = y
            favorable_labels = self._hyperparams["favorable_labels"]
            encoded_y = encoded_y.apply(lambda v: _group_flag(v, favorable_labels))
        result = self._pandas_to_dataset(encoded_X, encoded_y)
        return result

    def fit(self, X, y):
        self._class_attr = y.name
        self._wrapped_model = aif360.algorithms.preprocessing.OptimPreproc(
            optimizer=self._hyperparams["optimizer"],
            optim_options=self._hyperparams["optim_options"],
            unprivileged_groups=self._unprivileged_groups,
            privileged_groups=self._privileged_groups,
            verbose=self._hyperparams["verbose"],
            seed=self._hyperparams["seed"],
        )
        encoded_data = self._encode(X, y)
        self._wrapped_model.fit(encoded_data)
        return self

    def transform(self, X):
        encoded_data = self._encode(X)
        result_data = self._wrapped_model.transform(encoded_data)
        result_X, _ = dataset_to_pandas(result_data, return_only="X")
        return result_X


_input_fit_schema = {
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        },
        "y": {
            "description": "Target class labels; the array is over samples.",
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
            ],
        },
    },
}

_input_transform_schema = {
    "description": "Input data schema for transform.",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        }
    },
}

_output_transform_schema = {
    "description": "Output data schema for transform.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}

_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "favorable_labels",
                "protected_attributes",
                "optimizer",
                "optim_options",
                "verbose",
                "seed",
            ],
            "relevantToOptimizer": [],
            "properties": {
                "favorable_labels": _categorical_fairness_properties[
                    "favorable_labels"
                ],
                "protected_attributes": _categorical_fairness_properties[
                    "protected_attributes"
                ],
                "optimizer": {
                    "description": "Optimizer class.",
                    "anyOf": [
                        {"description": "User-provided.", "laleType": "Any",},
                        {
                            "description": "Use `aif360.algorithms.preprocessing.optim_preproc_helpers.opt_tools.OptTools`.",
                            "enum": [None],
                        },
                    ],
                    "default": None,
                },
                "optim_options": {
                    "description": "Options for optimization to estimate the transformation.",
                    "type": "object",
                    "patternProperties": {"^[A-Za-z_][A-Za-z_0-9]*$": {}},
                    "default": {},
                },
                "verbose": {
                    "description": "If zero, then no output.",
                    "type": "integer",
                    "default": 0,
                },
                "seed": {
                    "description": "Seed to make `transform` repeatable.",
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": None,
                },
            },
        }
    ],
}

_combined_schemas = {
    "description": """Work-in-progress, not covered in successful test yet: `Optimized Preprocessing`_ for fairness mitigation.

.. _`Optimized Preprocessing`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.preprocessing.OptimPreproc.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.disparate_impact_remover.html",
    "import_from": "aif360.algorithms.preprocessing",
    "type": "object",
    "tags": {"pre": ["~categoricals"], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

lale.docstrings.set_docstrings(OptimPreprocImpl, _combined_schemas)

OptimPreproc = lale.operators.make_operator(OptimPreprocImpl, _combined_schemas)
