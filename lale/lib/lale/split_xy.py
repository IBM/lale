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

import pandas as pd

import lale.docstrings
import lale.operators
from lale.helpers import _is_pandas_df, _is_spark_df


class _SplitXyImpl:
    def __init__(self, operator=None, label_name="y"):
        self.operator = operator
        self.label_name = label_name

    def split_df(self, X):
        if self.label_name not in X.columns:
            return X, None
        if _is_pandas_df(X):
            y = pd.DataFrame(X[self.label_name])
            X = X.drop(self.label_name, axis=1)
        elif _is_spark_df(X):
            y = X.select(X[self.label_name])
            X = X.drop(self.label_name)
        else:
            raise ValueError(
                "Only Pandas or Spark dataframe are supported as inputs. Please check that pyspark is installed if you see this error for a Spark dataframe."
            )
        return X, y

    def fit(self, X, y=None):
        X, y = self.split_df(X)
        op = self.operator
        assert op is not None
        self.trained_operator = op.fit(X, y)
        return self

    def transform(self, X):
        X, y = self.split_df(X)
        return self.trained_operator.transform(X, y)

    def predict(self, X):
        X, _ = self.split_df(X)
        return self.trained_operator.predict(X)

    def predict_proba(self, X):
        X, _ = self.split_df(X)
        return self.trained_operator.predict_proba(X)

    def decision_function(self, X):
        X, _ = self.split_df(X)
        return self.trained_operator.decision_function(X)


_hyperparams_schema = {
    "description": "SplitXy operator separates the label field/column from the input dataframe X.",
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters",
            "type": "object",
            "additionalProperties": False,
            "relevantToOptimizer": [],
            "properties": {
                "label_name": {
                    "description": "The name of the label column in the input dataframe X.",
                    "default": "y",
                    "type": "string",
                },
                "operator": {
                    "description": "The operator with which has to be applied over the separated dataframe X.",
                    "laleType": "operator",
                },
            },
        }
    ],
}

_input_fit_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": True,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
    },
}

_input_transform_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": True,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
    },
}

_output_transform_schema = {
    "description": "Output data schema for transformed data.",
    "laleType": "Any",
}

_input_predict_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": True,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        }
    },
}

_output_predict_schema = {
    "description": "Output data schema for predictions.",
    "laleType": "Any",
}

_input_predict_proba_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": True,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        }
    },
}

_output_predict_proba_schema = {
    "description": "Output data schema for predictions.",
    "laleType": "Any",
}

_input_decision_function_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": True,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        }
    },
}

_output_decision_function_schema = {
    "description": "Output data schema for predictions.",
    "laleType": "Any",
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Relational algebra SplitXy operator.",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.split_xy.html",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
        "input_decision_function": _input_decision_function_schema,
        "output_decision_function": _output_decision_function_schema,
    },
}

SplitXy = lale.operators.make_operator(_SplitXyImpl, _combined_schemas)

lale.docstrings.set_docstrings(SplitXy)
