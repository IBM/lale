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

import numpy as np
import pandas as pd

import lale.docstrings
import lale.operators


class SampleBasedVotingImpl:
    def __init__(self, hyperparams=None):
        self._hyperparams = hyperparams
        self.end_index_list = None

    def set_meta_data(self, meta_data_dict):
        if "end_index_list" in meta_data_dict.keys():
            self.end_index_list = meta_data_dict["end_index_list"]

    def transform(self, X, end_index_list=None):
        if end_index_list is None:
            end_index_list = (
                self.end_index_list
            )  # in case the end_index_list was set as meta_data

        if end_index_list is None:
            return X
        else:
            voted_labels = []
            prev_index = 0
            if not isinstance(X, np.ndarray):
                if isinstance(X, list):
                    X = np.array(X)
                elif isinstance(X, pd.dataframe):
                    X = X.as_matrix()
            for index in end_index_list:
                labels = X[prev_index:index]
                (values, counts) = np.unique(labels, return_counts=True)
                ind = np.argmax(
                    counts
                )  # If two labels are in majority, this will pick the first one.
                voted_labels.append(ind)
            return np.array(voted_labels)


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters",
            "type": "object",
            "additionalProperties": False,
            "relevantToOptimizer": [],
            "properties": {},
        }
    ]
}

_input_transform_schema = {
    "description": "Input data schema for transformations using NoOp.",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Labels from the previous component in a pipeline.",
            "type": "array",
            "items": {"laleType": "Any"},
        },
        "end_index_list": {
            "laleType": "Any",
            "description": "For each output label to be produced, end_index_list is supposed to contain the index of the last element corresponding to the original input.",
        },
    },
}

_output_transform_schema = {"type": "array", "items": {"laleType": "Any"}}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Treat the input as labels and use the end_index_list to produce labels using voting. Note that here, X contains the label and no y is accepted.""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.sample_based_voting.html",
    "import_from": "lale.lib.lale",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

lale.docstrings.set_docstrings(SampleBasedVotingImpl, _combined_schemas)

SampleBasedVoting = lale.operators.make_operator(
    SampleBasedVotingImpl, _combined_schemas
)
