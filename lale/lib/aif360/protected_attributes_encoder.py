# Copyright 2020, 2021 IBM Corporation
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
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd

import lale.datasets.data_schemas
import lale.docstrings
import lale.operators
import lale.type_checking

from .util import (
    _categorical_fairness_properties,
    _ensure_str,
    _ndarray_to_dataframe,
    _ndarray_to_series,
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
                "favorable_labels": {
                    "anyOf": [
                        _categorical_fairness_properties["favorable_labels"],
                        {"enum": [None]},
                    ],
                    "default": None,
                },
                "protected_attributes": _categorical_fairness_properties[
                    "protected_attributes"
                ],
                "unfavorable_labels": _categorical_fairness_properties[
                    "unfavorable_labels"
                ],
                "remainder": {
                    "description": "Transformation for columns that were not specified in protected_attributes.",
                    "enum": ["passthrough", "drop"],
                    "default": "drop",
                },
                "return_X_y": {
                    "description": "If True, return tuple with X and y; otherwise, return only X, not as a tuple.",
                    "type": "boolean",
                    "default": False,
                },
                "combine": {
                    "description": "How to handle the case when there is more than one protected attribute.",
                    "enum": ["keep_separate", "and", "or", "error"],
                    "default": "keep_separate",
                },
            },
        },
        {
            "description": "If returning y, need to know how to encode it.",
            "anyOf": [
                {"type": "object", "properties": {"return_X_y": {"enum": [False]}}},
                {
                    "type": "object",
                    "properties": {"favorable_labels": {"not": {"enum": [None]}}},
                },
            ],
        },
        {
            "description": "If combine is error, must have only one protected attribute.",
            "anyOf": [
                {
                    "type": "object",
                    "properties": {"combine": {"not": {"enum": ["error"]}}},
                },
                {
                    "type": "object",
                    "properties": {
                        "protected_attributes": {"type": "array", "maxItems": 1}
                    },
                },
            ],
        },
    ],
}

_input_transform_schema = {
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
            "description": "Target labels.",
            "anyOf": [
                {
                    "type": "array",
                    "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
                },
                {"enum": [None]},
            ],
            "default": None,
        },
    },
}

_output_transform_schema = {
    "anyOf": [
        {
            "description": "If return_X_y is False, return X.",
            "type": "array",
            "items": {
                "description": "This operator encodes protected attributes as `0`, `0.5`, or `1`. So if the remainder (non-protected attributes) is dropped, the output is numeric. Otherwise, the output may still contain non-numeric values.",
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        },
        {
            "description": "If return_X_y is True, return tuple of X and y.",
            "type": "array",
            "laleType": "tuple",
            "items": [
                {
                    "description": "X",
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
                    },
                },
                {
                    "description": "y",
                    "type": "array",
                    "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
                },
            ],
        },
    ],
}


_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Protected attributes encoder.

The `protected_attributes` argument describes each sensitive column by
a `feature` name or index and a `reference_group` list of values or
ranges. This transformer encodes protected attributes with values of
`0`, `0.5`, or `1` to indicate membership in the unprivileged, neither,
or privileged group, respectively. That encoding makes the
protected attributes suitable as input for downstream fairness
mitigation operators. This operator does not encode the remaining
(non-protected) attributes. A common usage is to encode non-protected
attributes with a separate data preparation pipeline and to perform a
feature union before piping the transformed data to downstream
operators that require numeric data.
This operator is used internally by various lale.lib.aif360 metrics
and mitigators, so you often do not need to use it directly yourself.
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.aif360.protected_attributes_encoder.html#lale.lib.aif360.protected_attributes_encoder.ProtectedAttributesEncoder",
    "import_from": "lale.lib.aif360",
    "type": "object",
    "tags": {"pre": ["categoricals"], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


def _dataframe_replace(dataframe, subst):
    new_columns = [
        subst.get(i, subst.get(name, dataframe.iloc[:, i]))
        for i, name in enumerate(dataframe.columns)
    ]
    result = pd.concat(new_columns, axis=1)
    return result


def _group_flag(value, pos_groups, other_groups):
    for group in pos_groups:
        if value == group:
            return 1
        if isinstance(group, list) and group[0] <= value <= group[1]:
            return 1
    if other_groups is None:
        return 0
    for group in other_groups:
        if value == group:
            return 0
        if isinstance(group, list) and group[0] <= value <= group[1]:
            return 0
    return 0.5  # neither positive nor other


class _ProtectedAttributesEncoderImpl:
    y_name: str
    protected_attributes: List[Dict[str, Any]]

    def __init__(
        self,
        *,
        favorable_labels=None,
        protected_attributes=[],
        unfavorable_labels=None,
        remainder="drop",
        return_X_y=False,
        combine="keep_separate",
    ):
        self.favorable_labels = favorable_labels
        self.protected_attributes = protected_attributes
        self.unfavorable_labels = unfavorable_labels
        self.remainder = remainder
        self.return_X_y = return_X_y
        self.combine = combine

    def transform(self, X: Union[np.ndarray, pd.DataFrame], y=None):
        X_pd: pd.DataFrame
        if isinstance(X, np.ndarray):
            X_pd = _ndarray_to_dataframe(X)
        else:
            X_pd = X
        assert isinstance(X_pd, pd.DataFrame), type(X_pd)
        protected = {}
        for prot_attr in self.protected_attributes:
            feature = prot_attr["feature"]
            pos_groups = prot_attr["reference_group"]
            other_groups = prot_attr.get("monitored_group", None)
            if isinstance(feature, str):
                column = X_pd[feature]
            else:
                column = X_pd.iloc[:, feature]
            series = column.apply(lambda v: _group_flag(v, pos_groups, other_groups))
            protected[feature] = series
        if self.combine in ["and", "or"]:
            prot_attr_names = [
                _ensure_str(pa["feature"]) for pa in self.protected_attributes
            ]
            comb_name = f"_{self.combine}_".join(prot_attr_names)
            if comb_name in X_pd.columns:
                suffix = 0
                while f"{comb_name}_{suffix}" in X_pd.columns:
                    suffix += 1
                comb_name = f"{comb_name}_{suffix}"
            comb_df = pd.concat([protected[f] for f in protected], axis=1)
            if self.combine == "and":
                comb_series = comb_df.min(axis=1)
            elif self.combine == "or":
                comb_series = comb_df.max(axis=1)
            else:
                assert False, self.combine
            comb_series.name = comb_name
            protected = {comb_name: comb_series}
        if self.remainder == "drop":
            result_X = pd.concat([protected[f] for f in protected], axis=1)
        else:
            result_X = _dataframe_replace(X_pd, protected)
        s_X = lale.datasets.data_schemas.to_schema(X_pd)
        s_result = self.transform_schema(s_X)
        result_X = lale.datasets.data_schemas.add_schema(result_X, s_result)
        if not self.return_X_y:
            return result_X
        assert self.favorable_labels is not None
        if y is None:
            assert hasattr(self, "y_name"), "must call transform with non-None y first"
            result_y = pd.Series(
                data=0.0, index=X_pd.index, dtype=np.float64, name=self.y_name
            )
        else:
            if isinstance(y, np.ndarray):
                self.y_name = _ensure_str(X_pd.shape[1])
                series_y = _ndarray_to_series(y, self.y_name, X_pd.index, y.dtype)
            else:
                series_y = y.squeeze() if isinstance(y, pd.DataFrame) else y
                assert isinstance(series_y, pd.Series), type(series_y)
                self.y_name = series_y.name
            result_y = series_y.apply(
                lambda v: _group_flag(v, self.favorable_labels, self.unfavorable_labels)
            )
        return result_X, result_y

    def transform_schema(self, s_X):
        """Used internally by Lale for type-checking downstream operators."""
        s_X = lale.datasets.data_schemas.to_schema(s_X)
        if self.remainder == "drop" and self.combine == "keep_separate":
            out_names = [pa["feature"] for pa in self.protected_attributes]
            if all([isinstance(n, str) for n in out_names]):
                result = {
                    **s_X,
                    "items": {
                        "type": "array",
                        "minItems": len(out_names),
                        "maxItems": len(out_names),
                        "items": [
                            {"description": n, "enum": [0, 0.5, 1]} for n in out_names
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
                        "items": {"enum": [0, 0.5, 1]},
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


ProtectedAttributesEncoder = lale.operators.make_operator(
    _ProtectedAttributesEncoderImpl,
    _combined_schemas,
)

lale.docstrings.set_docstrings(ProtectedAttributesEncoder)
