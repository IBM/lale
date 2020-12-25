# Copyright 2020 IBM Corporation
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

import aif360.algorithms.inprocessing

import lale.docstrings
import lale.operators

from .protected_attributes_encoder import ProtectedAttributesEncoder
from .util import (
    _categorical_fairness_properties,
    _PandasToDatasetConverter,
    dataset_to_pandas,
)


class MetaFairClassifierImpl:
    def __init__(
        self, favorable_labels, protected_attributes, tau=0.8, type="fdr",
    ):
        self._hyperparams = {
            "favorable_labels": favorable_labels,
            "protected_attributes": protected_attributes,
            "tau": tau,
            "type": type,
        }
        prot_attr_names = [pa["feature"] for pa in protected_attributes]
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

    def _encode(self, X, y=None):
        encoded_X, encoded_y = self._prot_attr_enc.transform(X, y)
        result = self._pandas_to_dataset.convert(encoded_X, encoded_y)
        return result

    def fit(self, X, y):
        pans = [pa["feature"] for pa in self._hyperparams["protected_attributes"]]
        self._wrapped_model = aif360.algorithms.inprocessing.MetaFairClassifier(
            tau=self._hyperparams["tau"],
            sensitive_attr=pans[0],
            type=self._hyperparams["type"],
        )
        encoded_data = self._encode(X, y)
        self._wrapped_model.fit(encoded_data)
        return self

    def predict(self, X):
        encoded_data = self._encode(X)
        result_data = self._wrapped_model.predict(encoded_data)
        _, result_y = dataset_to_pandas(result_data, return_only="y")
        return result_y


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

_input_predict_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
    },
}

_output_predict_schema = {
    "description": "Predicted class label per sample.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "string"}},
    ],
}

_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "additionalProperties": False,
            "required": ["favorable_labels", "protected_attributes", "tau", "type",],
            "relevantToOptimizer": ["tau", "type"],
            "properties": {
                "favorable_labels": _categorical_fairness_properties[
                    "favorable_labels"
                ],
                "protected_attributes": {
                    **_categorical_fairness_properties["protected_attributes"],
                    "minItems": 1,
                    "maxItems": 1,
                },
                "tau": {
                    "description": "Fairness penalty parameter.",
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.8,
                },
                "type": {
                    "description": "The type of fairness metric to be used.",
                    "anyOf": [
                        {
                            "description": "False discovery rate ratio.",
                            "enum": ["fdr"],
                        },
                        {
                            "description": "Statistical rate / disparate impact.",
                            "enum": ["sr"],
                        },
                    ],
                    "default": "fdr",
                },
            },
        },
    ],
}

_combined_schemas = {
    "description": """Work-in-progress, not covered in successful test yet: `MetaFairClassifier`_ in-processing operator for fairness mitigation.

.. _`MetaFairClassifier`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.inprocessing.MetaFairClassifier.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.prejudice_remover.html",
    "import_from": "aif360.sklearn.inprocessing",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}

lale.docstrings.set_docstrings(MetaFairClassifierImpl, _combined_schemas)

MetaFairClassifier = lale.operators.make_operator(
    MetaFairClassifierImpl, _combined_schemas
)
