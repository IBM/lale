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
import pandas as pd

import lale.docstrings
import lale.lib.lale
import lale.operators

from .protected_attributes_encoder import ProtectedAttributesEncoder
from .redacting import Redacting
from .util import (
    _categorical_fairness_properties,
    _categorical_input_transform_schema,
    _categorical_supervised_input_fit_schema,
    _numeric_output_transform_schema,
    _PandasToDatasetConverter,
    dataset_to_pandas,
)


class _LFRImpl:
    def __init__(
        self,
        *,
        favorable_labels,
        protected_attributes,
        redact=True,
        preparation=None,
        k=5,
        Ax=0.01,
        Az=1.0,
        Ay=50.0,
        print_interval=250,
        verbose=0,
        seed=None,
    ):
        self.favorable_labels = favorable_labels
        self.protected_attributes = protected_attributes
        self.redact = redact
        if preparation is None:
            preparation = lale.lib.lale.NoOp
        self.preparation = preparation
        prot_attr_names = [pa["feature"] for pa in protected_attributes]
        unprivileged_groups = [{name: 0 for name in prot_attr_names}]
        privileged_groups = [{name: 1 for name in prot_attr_names}]
        self.mitigator = aif360.algorithms.preprocessing.LFR(
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
            k=k,
            Ax=Ax,
            Az=Az,
            Ay=Ay,
            print_interval=print_interval,
            verbose=verbose,
            seed=seed,
        )

    def _prep_and_encode(self, X, y=None):
        prepared_X = self.redact1_and_prep.transform(X, y)
        encoded_X, encoded_y = self.prot_attr_enc.transform(X, y)
        combined_attribute_names = list(prepared_X.columns) + [
            name for name in encoded_X.columns if name not in prepared_X.columns
        ]
        combined_columns = [
            encoded_X[name] if name in encoded_X else prepared_X[name]
            for name in combined_attribute_names
        ]
        combined_X = pd.concat(combined_columns, axis=1)
        result = self.pandas_to_dataset.convert(combined_X, encoded_y)
        return result

    def _mitigate(self, encoded_data):
        mitigated_data = self.mitigator.transform(encoded_data)
        mitigated_X, _ = dataset_to_pandas(mitigated_data, return_only="X")
        return mitigated_X

    def fit(self, X, y):
        fairness_info = {
            "favorable_labels": self.favorable_labels,
            "protected_attributes": self.protected_attributes,
        }
        redacting = Redacting(**fairness_info) if self.redact else lale.lib.lale.NoOp
        preparation = self.preparation
        trainable_redact1_and_prep = redacting >> preparation
        assert isinstance(trainable_redact1_and_prep, lale.operators.TrainablePipeline)
        self.redact1_and_prep = trainable_redact1_and_prep.fit(X, y)
        self.prot_attr_enc = ProtectedAttributesEncoder(
            **fairness_info,
            remainder="drop",
            return_X_y=True,
        )
        prot_attr_names = [pa["feature"] for pa in self.protected_attributes]
        self.pandas_to_dataset = _PandasToDatasetConverter(
            favorable_label=1,
            unfavorable_label=0,
            protected_attribute_names=prot_attr_names,
        )
        encoded_data = self._prep_and_encode(X, y)
        self.mitigator.fit(encoded_data)
        mitigated_X = self._mitigate(encoded_data)
        self.redact2 = redacting.fit(mitigated_X)
        return self

    def transform(self, X):
        encoded_data = self._prep_and_encode(X)
        mitigated_X = self._mitigate(encoded_data)
        redacted_X = self.redact2.transform(mitigated_X)
        return redacted_X


_input_fit_schema = _categorical_supervised_input_fit_schema
_input_transform_schema = _categorical_input_transform_schema
_output_transform_schema = _numeric_output_transform_schema

_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                *_categorical_fairness_properties.keys(),
                "redact",
                "preparation",
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
                **_categorical_fairness_properties,
                "redact": {
                    "description": "Whether to redact protected attributes before data preparation (recommended) or not.",
                    "type": "boolean",
                    "default": True,
                },
                "preparation": {
                    "description": "Transformer, which may be an individual operator or a sub-pipeline.",
                    "anyOf": [
                        {"laleType": "operator"},
                        {"description": "lale.lib.lale.NoOp", "enum": [None]},
                    ],
                    "default": None,
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
    "description": """`LFR`_ (learning fair representations) pre-estimator fairness mitigator. Finds a latent representation that encodes the data well but obfuscates information about protected attributes (`Zemel et al. 2013`_).

.. _`LFR`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.preprocessing.LFR.html
.. _`Zemel et al. 2013`: http://proceedings.mlr.press/v28/zemel13.html
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.lfr.html#lale.lib.aif360.lfr.LFR",
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


LFR = lale.operators.make_operator(_LFRImpl, _combined_schemas)

lale.docstrings.set_docstrings(LFR)
