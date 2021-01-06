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

import aif360.algorithms.preprocessing

import lale.docstrings
import lale.operators

from .protected_attributes_encoder import ProtectedAttributesEncoder
from .redacting import Redacting
from .util import (
    _categorical_fairness_properties,
    _categorical_input_predict_schema,
    _categorical_output_predict_schema,
    _categorical_supervised_input_fit_schema,
    _PandasToDatasetConverter,
)


class ReweighingImpl:
    def __init__(
        self, favorable_labels, protected_attributes, estimator,
    ):
        self.favorable_labels = favorable_labels
        self.protected_attributes = protected_attributes
        self.estimator = estimator

    def fit(self, X, y):
        fairness_info = {
            "favorable_labels": self.favorable_labels,
            "protected_attributes": self.protected_attributes,
        }
        prot_attr_enc = ProtectedAttributesEncoder(
            **fairness_info, remainder="drop", return_X_y=True,
        )
        encoded_X, encoded_y = prot_attr_enc.transform(X, y)
        prot_attr_names = [pa["feature"] for pa in self.protected_attributes]
        pandas_to_dataset = _PandasToDatasetConverter(
            favorable_label=1,
            unfavorable_label=0,
            protected_attribute_names=prot_attr_names,
        )
        encoded_data = pandas_to_dataset.convert(encoded_X, encoded_y)
        unpriv_groups = [{name: 0 for name in prot_attr_names}]
        priv_groups = [{name: 1 for name in prot_attr_names}]
        reweighing_trainable = aif360.algorithms.preprocessing.Reweighing(
            unprivileged_groups=unpriv_groups, privileged_groups=priv_groups,
        )
        reweighing_trained = reweighing_trainable.fit(encoded_data)
        reweighted_data = reweighing_trained.transform(encoded_data)
        sample_weight = reweighted_data.instance_weights
        redacting_trainable = Redacting(**fairness_info)
        self.redacting = redacting_trainable.fit(X)
        redacted_X = self.redacting.transform(X)
        if isinstance(self.estimator, lale.operators.TrainablePipeline):
            trainable_prefix = self.estimator.remove_last()
            trainable_suffix = self.estimator.get_last()
            trained_prefix = trainable_prefix.fit(X, y)
            transformed_X = trained_prefix.transform(redacted_X)
            trained_suffix = trainable_suffix.fit(
                transformed_X, y, sample_weight=sample_weight
            )
            self.estimator = trained_prefix >> trained_suffix
        else:
            self.estimator = self.estimator.fit(
                redacted_X, y, sample_weight=sample_weight
            )
        return self

    def predict(self, X):
        redacted_X = self.redacting.transform(X)
        result = self.estimator.predict(redacted_X)
        return result


_input_fit_schema = _categorical_supervised_input_fit_schema
_input_predict_schema = _categorical_input_predict_schema
_output_predict_schema = _categorical_output_predict_schema

_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": [*_categorical_fairness_properties.keys(), "estimator"],
            "relevantToOptimizer": [],
            "properties": {
                **_categorical_fairness_properties,
                "estimator": {
                    "description": "Nested classifier, fit method must support sample_weight.",
                    "laleType": "operator",
                },
            },
        }
    ],
}

_combined_schemas = {
    "description": """`Reweighing`_ preprocessor for fairness mitigation.

.. _`Reweighing`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.sklearn.preprocessing.Reweighing.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.reweighing.html",
    "import_from": "aif360.sklearn.preprocessing",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}

lale.docstrings.set_docstrings(ReweighingImpl, _combined_schemas)

Reweighing = lale.operators.make_operator(ReweighingImpl, _combined_schemas)
