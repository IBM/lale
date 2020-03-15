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

import sklearn.preprocessing.data
import lale.helpers
import lale.operators

class StandardScalerImpl():

    def __init__(self, copy=True, with_mean=True, with_std=True):
        self._hyperparams = {
            'copy': copy,
            'with_mean': with_mean,
            'with_std': with_std}
        self._sklearn_model = sklearn.preprocessing.data.StandardScaler(**self._hyperparams)

    def fit(self, X, y=None):
        self._sklearn_model.fit(X, y)
        return self

    def transform(self, X, copy=None):
        return self._sklearn_model.transform(X, copy)

_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Standardize features by removing the mean and scaling to unit variance',
    'allOf': [{
        'type': 'object',
        'required': ['copy', 'with_mean', 'with_std'],
        'relevantToOptimizer': ['copy', 'with_mean', 'with_std'],
        'additionalProperties': False,
        'properties': {
            'copy': {
                'type': 'boolean',
                'default': True,
                'description': 'If False, try to avoid a copy and do inplace scaling instead.'},
            'with_mean': {
                'type': 'boolean',
                'default': True,
                'description': 'If True, center the data before scaling.'},
            'with_std': {
                'type': 'boolean',
                'default': True,
                'description': 'If True, scale the data to unit variance (or equivalently,'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Compute the mean and std to be used for later scaling.',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'The data used to compute the mean and standard deviation'},
        'y': {'description': 'Ignored'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Perform standardization by centering and scaling',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'The data used to scale along the features axis.'},
        'copy': {
            'anyOf': [{
                'type': 'boolean'}, {
                'enum': [None]}],
            'default': None,
            'description': 'Copy the input X or not.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Perform standardization by centering and scaling',
    'type': 'array',
    'items': {'type': 'array', 'items': {'type': 'number'}}
}
_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html',
    'type': 'object',
    'tags': {
        'pre': [],
        'op': ['transformer'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_transform': _input_transform_schema,
        'output_transform': _output_transform_schema}}

if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
StandardScaler = lale.operators.make_operator(StandardScalerImpl, _combined_schemas)
