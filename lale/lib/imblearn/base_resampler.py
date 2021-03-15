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


class _BaseResamplerImpl:
    def __init__(self, operator=None, resampler=None):
        self.operator = operator
        self.resampler = resampler

    def fit(self, X, y=None):
        X, y = self.resampler.fit_resample(X, y)
        self.trained_operator = self.operator.fit(X, y)
        return self

    def transform(self, X, y=None):
        return self.trained_operator.transform(X, y)

    def predict(self, X):
        return self.trained_operator.predict(X)

    def predict_proba(self, X):
        return self.trained_operator.predict_proba(X)

    def decision_function(self, X):
        return self.trained_operator.decision_function(X)


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
                {"enum": [None]},
            ],
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
    "description": "Output data schema for predictions.",
    "laleType": "Any",
}

_input_predict_proba_schema = {
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

_output_predict_proba_schema = {
    "description": "Output data schema for predictions.",
    "laleType": "Any",
}

_input_decision_function_schema = {
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

_output_decision_function_schema = {
    "description": "Output data schema for predictions.",
    "laleType": "Any",
}
