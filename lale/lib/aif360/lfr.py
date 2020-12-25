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
import aif360.datasets

import lale.docstrings
import lale.operators

from .protected_attributes_encoder import ProtectedAttributesEncoder
from .redacting import Redacting
from .util import (
    _categorical_fairness_properties,
    _PandasToDatasetConverter,
    dataset_to_pandas,
)


class LFRImpl:
    def __init__(
        self,
        favorable_labels,
        protected_attributes,
        k=5,
        Ax=0.01,
        Az=1.0,
        Ay=50.0,
        print_interval=250,
        verbose=0,
        seed=None,
    ):
        self._hyperparams = {
            "favorable_labels": favorable_labels,
            "protected_attributes": protected_attributes,
            "k": k,
            "Ax": Ax,
            "Az": Az,
            "Ay": Ay,
            "print_interval": print_interval,
            "verbose": verbose,
            "seed": seed,
        }
        prot_attr_names = [pa["feature"] for pa in protected_attributes]
        self._unprivileged_groups = [{name: 0 for name in prot_attr_names}]
        self._privileged_groups = [{name: 1 for name in prot_attr_names}]
        self._prot_attr_enc = ProtectedAttributesEncoder(
            favorable_labels=favorable_labels,
            protected_attributes=protected_attributes,
            remainder="passthrough",
            return_X_y=True,
        )
        self._pandas_to_dataset = _PandasToDatasetConverter(
            favorable_label=1,
            unfavorable_label=0,
            protected_attribute_names=prot_attr_names,
        )
        self._redacting = Redacting(protected_attribute_names=prot_attr_names)

    def _encode(self, X, y=None):
        encoded_X, encoded_y = self._prot_attr_enc.transform(X, y)
        result = self._pandas_to_dataset.convert(encoded_X, encoded_y)
        return result

    def fit(self, X, y):
        self._wrapped_model = aif360.algorithms.preprocessing.LFR(
            unprivileged_groups=self._unprivileged_groups,
            privileged_groups=self._privileged_groups,
            k=self._hyperparams["k"],
            Ax=self._hyperparams["Ax"],
            Az=self._hyperparams["Az"],
            Ay=self._hyperparams["Ay"],
            print_interval=self._hyperparams["print_interval"],
            verbose=self._hyperparams["verbose"],
            seed=self._hyperparams["seed"],
        )
        encoded_data = self._encode(X, y)
        self._wrapped_model.fit(encoded_data)
        self._redacting = self._redacting.fit(X)
        return self

    def transform(self, X):
        encoded_data = self._encode(X)
        remediated_data = self._wrapped_model.transform(encoded_data)
        remediated_X, _ = dataset_to_pandas(remediated_data, return_only="X")
        redacted_X = self._redacting.transform(remediated_X)
        return redacted_X


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
            "items": {"type": "array", "items": {"type": "number"}},
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
                "k",
                "Ax",
                "Az",
                "Ay",
                "print_interval",
                "verbose",
                "seed",
            ],
            "relevantToOptimizer": ["k", "Ax", "Az", "Ay"],
            "properties": {
                "favorable_labels": _categorical_fairness_properties[
                    "favorable_labels"
                ],
                "protected_attributes": {
                    **_categorical_fairness_properties["protected_attributes"],
                    "minItems": 1,
                    "maxItems": 1,
                },
                "k": {
                    "description": "Number of prototypes.",
                    "type": "integer",
                    "minimum": 1,
                    "default": 5,
                    "maximumForOptimizer": 20,
                },
                "Ax": {
                    "description": "Input recontruction quality term weight.",
                    "type": "number",
                    "minimum": 0.0,
                    "default": 0.01,
                    "maximumForOptimizer": 100.0,
                },
                "Az": {
                    "description": "Fairness constraint term weight.",
                    "type": "number",
                    "minimum": 0.0,
                    "default": 1.0,
                    "maximumForOptimizer": 100.0,
                },
                "Ay": {
                    "description": "Output prediction error.",
                    "type": "number",
                    "minimum": 0.0,
                    "default": 50.0,
                    "maximumForOptimizer": 100.0,
                },
                "print_interval": {
                    "description": "Print optimization objective value every print_interval iterations.",
                    "type": "integer",
                    "minimum": 1,
                    "default": 250,
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
    "description": """`LFR`_ (learning fair representations) preprocessor for fairness mitigation.

.. _`LFR`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.preprocessing.LFR.html
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

lale.docstrings.set_docstrings(LFRImpl, _combined_schemas)

LFR = lale.operators.make_operator(LFRImpl, _combined_schemas)
