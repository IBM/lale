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

import lale.helpers
import lale.operators
import pandas as pd

class NoOpImpl():
    def __init__(self, hyperparams=None):
        self._hyperparams = hyperparams

    # def fit(self, X, y = None):
    #     result = NoOpImpl(self._hyperparams)
    #     if isinstance(X, pd.DataFrame):
    #         result._feature_names = list(X.columns)
    #     else:
    #         #This assumes X is a 2d array which is consistent with its schema.
    #         result._feature_names = ['x%d' % i for i in range(X.shape[1])] 
    #     return result

    def transform(self, X, y = None):
        return X

    def transform_schema(self, s_X):
        return s_X

    # def get_feature_names(self, input_features=None):
    #     if input_features is not None:
    #         return list(input_features)
    #     elif self._feature_names is not None:
    #         return self._feature_names
    #     else:
    #         raise ValueError('Can only call get_feature_names on a trained operator. Please call fit to get a trained operator.')

_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Hyperparameter schema for the NoOp, which is a place-holder for no operation.',
    'allOf': [
    {   'description': 'This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters',
        'type': 'object',
        'additionalProperties': False,
        'relevantToOptimizer': [],
        'properties': {}}]}

_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Input data schema for training NoOp.',
    'type': 'object',
    'required': ['X'],
    'additionalProperties': False,
    'properties': {
        'X': {}}}

_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Input data schema for transformations using NoOp.',
    'type': 'object',
    'required': ['X', 'y'],
    'additionalProperties': False,
    'properties': {
        'X': {},
        'y': {}}}

_output_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Output data schema for transformations using NoOp.'}

_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.no_op.html',
    'type': 'object',
    'tags': {
        'pre': [],
        'op': ['transformer'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_predict': _input_predict_schema,
        'output': _output_schema }}

if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)

NoOp = lale.operators.make_operator(NoOpImpl, _combined_schemas)
