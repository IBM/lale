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

try:
    import tensorflow as tf

    tensorflow_installed = True
except ImportError:
    tensorflow_installed = False

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


class AdversarialDebiasingImpl:
    def __init__(
        self,
        favorable_labels,
        protected_attributes,
        scope_name="adversarial_debiasing",
        sess=None,
        seed=None,
        adversary_loss_weight=0.1,
        num_epochs=50,
        batch_size=128,
        classifier_num_hidden_units=200,
        debias=True,
    ):
        assert tensorflow_installed, """Your Python environment does not have tensorflow installed. You can install it with
    pip install tensorflow
or with
    pip install 'lale[full]'"""
        assert "1.13.1" <= tf.__version__ <= "2", tf.__version__
        if sess is None:
            sess = tf.Session()
        self._hyperparams = {
            "favorable_labels": favorable_labels,
            "protected_attributes": protected_attributes,
            "scope_name": scope_name,
            "sess": sess,
            "seed": seed,
            "adversary_loss_weight": adversary_loss_weight,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "classifier_num_hidden_units": classifier_num_hidden_units,
            "debias": debias,
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
        self._wrapped_model = aif360.algorithms.inprocessing.AdversarialDebiasing(
            unprivileged_groups=self._unprivileged_groups,
            privileged_groups=self._privileged_groups,
            scope_name=self._hyperparams["scope_name"],
            sess=self._hyperparams["sess"],
            seed=self._hyperparams["seed"],
            adversary_loss_weight=self._hyperparams["adversary_loss_weight"],
            num_epochs=self._hyperparams["num_epochs"],
            batch_size=self._hyperparams["batch_size"],
            classifier_num_hidden_units=self._hyperparams[
                "classifier_num_hidden_units"
            ],
            debias=self._hyperparams["debias"],
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
            "required": [
                "favorable_labels",
                "protected_attributes",
                "scope_name",
                "sess",
                "seed",
                "adversary_loss_weight",
                "num_epochs",
                "batch_size",
                "classifier_num_hidden_units",
                "debias",
            ],
            "relevantToOptimizer": [
                "adversary_loss_weight",
                "num_epochs",
                "batch_size",
                "classifier_num_hidden_units",
            ],
            "properties": {
                "favorable_labels": _categorical_fairness_properties[
                    "favorable_labels"
                ],
                "protected_attributes": {
                    **_categorical_fairness_properties["protected_attributes"],
                    "minItems": 1,
                    "maxItems": 1,
                },
                "scope_name": {
                    "description": "Scope name for the tenforflow variables.",
                    "type": "string",
                    "default": "adversarial_debiasing",
                },
                "sess": {
                    "description": "TensorFlow session.",
                    "anyOf": [
                        {
                            "description": "User-provided session object.",
                            "laleType": "Any",
                        },
                        {
                            "description": "Create a session for the user.",
                            "enum": [None],
                        },
                    ],
                    "default": None,
                },
                "seed": {
                    "description": "Seed to make `predict` repeatable.",
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": None,
                },
                "adversary_loss_weight": {
                    "description": "Hyperparameter that chooses the strength of the adversarial loss.",
                    "type": "number",
                    "default": 0.1,
                    "distribution": "loguniform",
                    "minimumForOptimizer": 0.03125,
                    "maximumForOptimizer": 32768,
                },
                "num_epochs": {
                    "description": "Number of training epochs.",
                    "type": "integer",
                    "minimum": 1,
                    "default": 50,
                    "distribution": "loguniform",
                    "minimumForOptimizer": 5,
                    "maximumForOptimizer": 500,
                },
                "batch_size": {
                    "description": "Batch size.",
                    "type": "integer",
                    "minimum": 1,
                    "default": 128,
                    "distribution": "loguniform",
                    "minimumForOptimizer": 4,
                    "maximumForOptimizer": 512,
                },
                "classifier_num_hidden_units": {
                    "description": "Number of hidden units in the classifier model.",
                    "type": "integer",
                    "minimum": 1,
                    "default": 200,
                    "distribution": "loguniform",
                    "minimumForOptimizer": 16,
                    "maximumForOptimizer": 1024,
                },
                "debias": {
                    "description": "Learn a classifier with or without debiasing.",
                    "type": "boolean",
                    "default": True,
                },
            },
        },
    ],
}

_combined_schemas = {
    "description": """`AdversarialDebiasing`_ in-processing operator for fairness mitigation.

.. _`AdversarialDebiasing`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.inprocessing.AdversarialDebiasing.html
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

lale.docstrings.set_docstrings(AdversarialDebiasingImpl, _combined_schemas)

AdversarialDebiasing = lale.operators.make_operator(
    AdversarialDebiasingImpl, _combined_schemas
)
