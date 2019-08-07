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

import lale.datasets.data_schemas
import lale.operators
import numpy as np
import pandas as pd

class KeepNonNumbersImpl:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        s_all = lale.datasets.data_schemas.to_schema(X)
        s_row = s_all['items']
        n_columns = s_row['minItems']
        assert n_columns == s_row['maxItems']
        s_cols = s_row['items']
        def is_numeric(schema):
            return 'type' in schema and schema['type'] in ['number', 'integer']
        if isinstance(s_cols, dict):
            if is_numeric(s_cols):
                self._keep_cols = []
            else:
                self._keep_cols = [*range(n_columns)]
        else:
            assert isinstance(s_cols, list)
            self._keep_cols = [i for i in range(n_columns)
                               if not is_numeric(s_cols[i])]
        return self

    def transform(self, X, y=None):
        if isinstance(X, np.ndarray):
            result = X[:, self._keep_cols]
        elif isinstance(X, pd.DataFrame):
            result = X.iloc[:, self._keep_cols]
        else:
            assert False, f'case for type {type(X)} value {X} not implemented'
        s_X = lale.datasets.data_schemas.to_schema(X)
        s_result = self.transform_schema(s_X)
        return lale.datasets.data_schemas.add_schema(result, s_result)

    def transform_schema(self, s_X):
        s_row = s_X['items']
        s_cols = s_row['items']
        n_columns = len(self._keep_cols)
        if isinstance(s_cols, dict):
            s_cols_result = s_cols
        else:
            s_cols_result = [s_cols[i] for i in self._keep_cols]
        s_result = {
            **s_X,
            'items': {
                **s_row,
                'minItems': n_columns, 'maxItems': n_columns,
                'items': s_cols_result}}
        return s_result

_hyperparams_schema = {
  'description': 'Hyperparameter schema for KeepNonNumbers transformer.',
  'allOf': [
    { 'description':
        'This first sub-object lists all constructor arguments with their '
        'types, one at a time, omitting cross-argument constraints.',
      'type': 'object',
      'additionalProperties': False,
      'relevantToOptimizer': [],
      'properties': {}}]}

_input_fit_schema = {
  'description': 'Input data schema for training KeepNonNumbers.',
  'type': 'object',
  'required': ['X'],
  'additionalProperties': False,
  'properties': {
    'X': {
      'description': 'Features; the outer array is over samples.',
      'type': 'array',
      'items': {
        'type': 'array',
        'items': {
          'anyOf':[{'type': 'number'}, {'type':'string'}]}}},
    'y': {
      'description': 'Target class labels; the array is over samples.'}}}

_input_predict_schema = {
  'description': 'Input data schema for transformation using KeepNonNumbers.',
  'type': 'object',
  'required': ['X'],
  'additionalProperties': False,
  'properties': {
    'X': {
      'description': 'Features; the outer array is over samples.',
      'type': 'array',
      'items': {
        'type': 'array',
        'items': {
           'anyOf':[{'type': 'number'}, {'type':'string'}]}}}}}

_output_schema = {
  'description': 'Output data schema for transformed data using KeepNonNumbers.',
  'type': 'array',
  'items': {
    'type': 'array',
    'items': {
       'anyOf':[{'type': 'number'}, {'type':'string'}]}}}

_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'documentation_url': 'https://github.ibm.com/aimodels/lale',
    'type': 'object',
    'tags': {
        'pre': ['categoricals'],
        'op': ['transformer'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_predict': _input_predict_schema,
        'output': _output_schema }}

if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)

KeepNonNumbers = lale.operators.make_operator(KeepNonNumbersImpl, _combined_schemas)
