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


import numpy as np
import pandas as pd

import lale.datasets.data_schemas
import lale.docstrings
import lale.operators
import lale.type_checking

from .util import (
    _categorical_fairness_properties,
    _dataframe_replace,
    _group_flag,
    _ndarray_to_dataframe,
)

_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
            "type": "object",
            "additionalProperties": False,
            "required": ["protected_attributes"],
            "relevantToOptimizer": [],
            "properties": {
                "protected_attributes": _categorical_fairness_properties[
                    "protected_attributes"
                ],
                "remainder": {
                    "description": "Transformation for columns that were not specified in protected_attributes.",
                    "enum": ["passthrough", "drop"],
                    "default": "drop",
                },
            },
        },
    ],
}

_input_transform_schema = {
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
    "type": "array",
    "items": {
        "description": "This operator encodes protected attributes as `0` or `1`. So if the remainder (non-protected attributes) is dropped, the output is numeric. Otherwise, the output may still contain non-numeric values.",
        "type": "array",
        "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
    },
}


_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Protected attributes encoder.

The `protected_attributes` argument describes each sensitive column by
a `feature` name or index and a `privileged_groups` list of values or
ranges. This transformer encodes protected attributes with values of
`0` or `1` to indicate group membership. That encoding makes the
protected attributes suitable as input for downstream fairness
mitigation operators. This operator does not encode the remaining
(non-protected) attributes. A common usage is to encode non-protected
attributes with a separate preprocessing pipeline and to perform a
feature union before piping the transformed data to downstream
operators that require numeric data.
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.protected_attributes_encoder.html",
    "import_from": "lale.lib.aif360",
    "type": "object",
    "tags": {"pre": ["categoricals"], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


class ProtectedAttributesEncoderImpl:
    def __init__(self, protected_attributes=None, remainder="drop"):
        self.protected_attributes = protected_attributes
        self.remainder = remainder

    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = _ndarray_to_dataframe(X)
        assert isinstance(X, pd.DataFrame), type(X)
        protected = {}
        for prot_attr in self.protected_attributes:
            feature = prot_attr["feature"]
            groups = prot_attr["privileged_groups"]
            if isinstance(feature, str):
                column = X[feature]
            else:
                column = X.iloc[:, feature]
            series = column.apply(lambda v: _group_flag(v, groups))
            protected[feature] = series
        if self.remainder == "drop":
            result = pd.concat([protected[f] for f in protected], axis=1)
        else:
            result = _dataframe_replace(X, protected)
        s_X = lale.datasets.data_schemas.to_schema(X)
        s_result = self.transform_schema(s_X)
        result = lale.datasets.data_schemas.add_schema(result, s_result)
        return result

    def transform_schema(self, s_X):
        """Used internally by Lale for type-checking downstream operators."""
        s_X = lale.datasets.data_schemas.to_schema(s_X)
        if self.remainder == "drop":
            out_names = [pa["feature"] for pa in self.protected_attributes]
            if all([isinstance(n, str) for n in out_names]):
                result = {
                    **s_X,
                    "items": {
                        "type": "array",
                        "minItems": len(out_names),
                        "maxItems": len(out_names),
                        "items": [
                            {"description": n, "enum": [0, 1]} for n in out_names
                        ],
                    },
                }
            else:
                result = {
                    **s_X,
                    "items": {
                        "type": "array",
                        "minItems": len(out_names),
                        "maxItems": len(out_names),
                        "items": {"enum": [0, 1]},
                    },
                }
        else:
            result = {
                "type": "array",
                "items": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "number"}},
                        {"type": "array", "items": {"type": "string"}},
                    ],
                },
            }
        return result


lale.docstrings.set_docstrings(ProtectedAttributesEncoderImpl, _combined_schemas)

ProtectedAttributesEncoder = lale.operators.make_operator(
    ProtectedAttributesEncoderImpl, _combined_schemas,
)
