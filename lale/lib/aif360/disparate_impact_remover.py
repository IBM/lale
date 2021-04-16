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
import numpy as np
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
)


class _DisparateImpactRemoverImpl:
    def __init__(
        self,
        *,
        favorable_labels,
        protected_attributes,
        redact=True,
        preparation=None,
        repair_level=1.0,
    ):
        self.favorable_labels = favorable_labels
        self.protected_attributes = protected_attributes
        self.redact = redact
        if preparation is None:
            preparation = lale.lib.lale.NoOp
        self.preparation = preparation
        self.repair_level = repair_level

    def _prep_and_encode(self, X, y=None):
        prepared_X = self.redact_and_prep.transform(X, y)
        encoded_X, encoded_y = self.prot_attr_enc.transform(X, y)
        assert isinstance(encoded_X, pd.DataFrame), type(encoded_X)
        assert encoded_X.shape[1] == 1, encoded_X.columns
        if isinstance(prepared_X, pd.DataFrame):
            combined_attribute_names = list(prepared_X.columns) + [
                name for name in encoded_X.columns if name not in prepared_X.columns
            ]
            combined_columns = [
                encoded_X[name] if name in encoded_X else prepared_X[name]
                for name in combined_attribute_names
            ]
            combined_X = pd.concat(combined_columns, axis=1)
            sensitive_attribute = list(encoded_X.columns)[0]
        else:
            if isinstance(prepared_X, pd.DataFrame):
                prepared_X = prepared_X.to_numpy()
            assert isinstance(prepared_X, np.ndarray)
            encoded_X = encoded_X.to_numpy()
            assert isinstance(encoded_X, np.ndarray)
            combined_X = np.concatenate([prepared_X, encoded_X], axis=1)
            sensitive_attribute = combined_X.shape[1] - 1
        return combined_X, sensitive_attribute

    def fit(self, X, y=None):
        fairness_info = {
            "favorable_labels": self.favorable_labels,
            "protected_attributes": self.protected_attributes,
        }
        redacting = Redacting(**fairness_info) if self.redact else lale.lib.lale.NoOp
        preparation = self.preparation
        trainable_redact_and_prep = redacting >> preparation
        assert isinstance(trainable_redact_and_prep, lale.operators.TrainablePipeline)
        self.redact_and_prep = trainable_redact_and_prep.fit(X, y)
        self.prot_attr_enc = ProtectedAttributesEncoder(
            **fairness_info, remainder="drop", return_X_y=True, combine="and"
        )
        encoded_X, sensitive_attribute = self._prep_and_encode(X, y)
        if isinstance(sensitive_attribute, str):
            assert isinstance(encoded_X, pd.DataFrame)
            features = encoded_X.to_numpy().tolist()
            index = encoded_X.columns.to_list().index(sensitive_attribute)
        else:
            assert isinstance(encoded_X, np.ndarray)
            features = encoded_X.tolist()
            index = sensitive_attribute
        # since DisparateImpactRemover does not have separate fit and transform
        di_remover = aif360.algorithms.preprocessing.DisparateImpactRemover(
            repair_level=self.repair_level, sensitive_attribute=sensitive_attribute
        )
        self.mitigator = di_remover.Repairer(features, index, self.repair_level, False)
        return self

    def transform(self, X):
        encoded_X, _ = self._prep_and_encode(X)
        if isinstance(encoded_X, pd.DataFrame):
            features = encoded_X.to_numpy().tolist()
        else:
            assert isinstance(encoded_X, np.ndarray)
            features = encoded_X.tolist()
        mitigated_X = self.mitigator.repair(features)
        if isinstance(X, pd.DataFrame):
            result = pd.DataFrame(mitigated_X, index=X.index, columns=encoded_X.columns)
        else:
            result = np.array(mitigated_X)
        return result


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
                "repair_level",
            ],
            "relevantToOptimizer": ["repair_level"],
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
                "repair_level": {
                    "description": "Repair amount from 0 = none to 1 = full.",
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 1,
                },
            },
        }
    ],
}

_combined_schemas = {
    "description": """`Disparate impact remover`_ pre-estimator fairness mitigator. Edits feature values to increase group fairness while preserving rank-ordering within groups (`Feldman et al. 2015`_). In the case of multiple protected attributes, the combined reference group is the intersection of the reference groups for each attribute.

.. _`Disparate impact remover`: https://aif360.readthedocs.io/en/latest/modules/generated/aif360.algorithms.preprocessing.DisparateImpactRemover.html
.. _`Feldman et al. 2015`: https://doi.org/10.1145/2783258.2783311
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.disparate_impact_remover.html#lale.lib.aif360.disparate_impact_remover.DisparateImpactRemover",
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

DisparateImpactRemover = lale.operators.make_operator(
    _DisparateImpactRemoverImpl, _combined_schemas
)

lale.docstrings.set_docstrings(DisparateImpactRemover)
