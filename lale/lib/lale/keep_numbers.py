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

import lale.operators
import numpy as np
import pandas as pd

class KeepNumbersImpl:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        def is_numeric(i):
            return np.issubdtype(X.dtypes[i], np.number)
        self._keep_cols = [i for i in X.columns if is_numeric(i)]
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)
        return X.iloc[:, self._keep_cols]

_hyperparams_schema = {
  'description': 'Hyperparameter schema for KeepNumbers transformer.',
  'allOf': [
    { 'description':
        'This first sub-object lists all constructor arguments with their '
        'types, one at a time, omitting cross-argument constraints.',
      'type': 'object',
      'additionalProperties': False,
      'relevantToOptimizer': [],
      'properties': {}}]}

_input_fit_schema = {
  'description': 'Input data schema for training KeepNumbers.',
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
  'description': 'Input data schema for transformation using KeepNumbers.',
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
  'description': 'Output data schema for transformed data using KeepNumbers.',
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

KeepNumbers = lale.operators.make_operator(KeepNumbersImpl, _combined_schemas)
