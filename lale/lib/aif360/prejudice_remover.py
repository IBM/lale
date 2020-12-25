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
import numpy as np
import pandas as pd

import lale.docstrings
import lale.operators

from .util import (
    _categorical_fairness_properties,
    _group_flag,
    _ndarray_to_series,
    _PandasToDatasetConverter,
    dataset_to_pandas,
)


class PrejudiceRemoverImpl:
    def __init__(self, eta=1.0, sensitive_attr="", favorable_labels=[1.0]):
        self.eta = eta
        self.sensitive_attr = sensitive_attr
        self.favorable_labels = favorable_labels
        self.pandas_to_dataset = _PandasToDatasetConverter(
            favorable_label=1,
            unfavorable_label=0,
            protected_attribute_names=[sensitive_attr],
        )

    def _encode(self, X, y=None):
        if y is None:
            encoded_y = pd.Series(
                data=0.0, index=X.index, dtype=np.float64, name=self.class_attr,
            )
        else:
            if isinstance(y, np.ndarray):
                encoded_y = _ndarray_to_series(y, X.shape[1])
            else:
                encoded_y = y
            encoded_y = encoded_y.apply(lambda v: _group_flag(v, self.favorable_labels))
        result = self.pandas_to_dataset.convert(X, encoded_y)
        return result

    def fit(self, X, y):
        self.class_attr = y.name
        self._wrapped_model = aif360.algorithms.inprocessing.PrejudiceRemover(
            eta=self.eta,
            sensitive_attr=self.sensitive_attr,
            class_attr=self.class_attr,
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
        }
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
            "required": ["eta", "sensitive_attr", "favorable_labels"],
            "relevantToOptimizer": ["eta"],
            "properties": {
                "eta": {
                    "description": "Fairness penalty parameter.",
                    "type": "number",
                    "minimum": 0.0,
                    "exclusiveMinimum": True,
                    "default": 1.0,
                    "minimumForOptimizer": 0.03125,
                    "maximumForOptimizer": 32768,
                },
                "sensitive_attr": {
                    "description": "Name of protected attribute.",
                    "type": "string",
                },
                "favorable_labels": _categorical_fairness_properties[
                    "favorable_labels"
                ],
            },
        },
    ],
}

_combined_schemas = {
    "description": """`PrejudiceRemover`_ in-processing operator for fairness mitigation.

.. _`PrejudiceRemover`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.inprocessing.PrejudiceRemover.html
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

lale.docstrings.set_docstrings(PrejudiceRemoverImpl, _combined_schemas)

PrejudiceRemover = lale.operators.make_operator(PrejudiceRemoverImpl, _combined_schemas)
