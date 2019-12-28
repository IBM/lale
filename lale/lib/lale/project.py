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

import jsonsubschema
import lale.datasets.data_schemas
import lale.helpers
import lale.operators
import numpy as np
import pandas as pd
import sklearn.compose
import sys

def isSubschema(sub, sup):
    try:
        return jsonsubschema.isSubschema(sub, sup)
    except Exception as e:
        raise ValueError(f'problem checking ({sub} <: {sup})') from e

class ProjectImpl:
    def __init__(self, columns=None):
        self._hyperparams = { 'columns': columns }

    def fit(self, X, y=None):
        columns = self._hyperparams['columns']
        if lale.helpers.is_schema(columns):
            s_all = lale.datasets.data_schemas.to_schema(X)
            s_row = s_all['items']
            n_columns = s_row['minItems']
            assert n_columns == s_row['maxItems']
            s_cols = s_row['items']
            if isinstance(s_cols, dict):
                if isSubschema(s_cols, columns):
                    columns = [*range(n_columns)]
                else:
                    columns = []
            else:
                assert isinstance(s_cols, list)
                columns = [
                    i for i in range(n_columns)
                    if isSubschema(s_cols[i], columns)]
        self._col_tfm = sklearn.compose.ColumnTransformer(
            transformers=[('keep', 'passthrough', columns)])
        self._col_tfm.fit(X)
        return self

    def transform(self, X, y=None):
        result = self._col_tfm.transform(X)
        s_X = lale.datasets.data_schemas.to_schema(X)
        s_result = self.transform_schema(s_X)
        return lale.datasets.data_schemas.add_schema(result, s_result)

    def transform_schema(self, s_X):
        if hasattr(self, '_col_tfm'):
            return self._transform_schema_col_tfm(s_X, self._col_tfm)
        columns = self._hyperparams['columns']
        if lale.helpers.is_schema(columns):
            return self._transform_schema_schema(s_X, columns)
        if not lale.helpers.is_schema(s_X):
            X = lale.datasets.data_schemas.add_schema(s_X)
            self.fit(X)
            return self._transform_schema_col_tfm(X.json_schema, self._col_tfm)
        return s_X

    def _transform_schema_col_tfm(self, s_X, col_tfm):
        s_X = lale.datasets.data_schemas.to_schema(s_X)
        s_row = s_X['items']
        s_cols = s_row['items']
        keep_cols = [col
                     for name, tfm, cols in col_tfm.transformers_
                     if tfm == 'passthrough'
                     for col in cols]
        n_columns = len(keep_cols)
        if isinstance(s_cols, dict):
            s_cols_result = s_cols
        else:
            name2i = {s_cols[i]['description']: i for i in range(len(s_cols))}
            keep_cols_i = [name2i[col] if isinstance(col, str) else col
                           for col in keep_cols]
            s_cols_result = [s_cols[i] for i in keep_cols_i]
        s_result = {
            **s_X,
            'items': {
                **s_row,
                'minItems': n_columns, 'maxItems': n_columns,
                'items': s_cols_result}}
        return s_result

    def _transform_schema_schema(self, s_X, schema):
        s_X = lale.datasets.data_schemas.to_schema(s_X)
        s_row = s_X['items']
        s_cols = s_row['items']
        if isinstance(s_cols, dict):
            if isSubschema(s_cols, schema):
                s_row_result = s_row
            else:
                s_row_result = {'type': 'array', 'minItems': 0, 'maxItems': 0}
        else:
            assert isinstance(s_cols, list)
            s_cols_result = [s for s in s_cols if isSubschema(s, schema)]
            n_columns = len(s_cols_result)
            s_row_result = {
                'type': 'array',
                'minItems': n_columns, 'maxItems': n_columns,
                'items': s_cols_result}
        return {'type': 'array', 'items': s_row_result}

_hyperparams_schema = {
  'description': 'Hyperparameter schema for Project transformer.',
  'allOf': [
    { 'description':
        'This first sub-object lists all constructor arguments with their '
        'types, one at a time, omitting cross-argument constraints.',
      'type': 'object',
      'additionalProperties': False,
      'required': ['columns'],
      'relevantToOptimizer': [],
      'properties': {
          'columns': {
              'description': 'string or int, array-like of string or int, slice, boolean mask array or callable.',
              'anyOf': [
                  {'type': 'string'},
                  {'type': 'integer'},
                  {'type': 'array', 'items': {'type': 'string'}},
                  {'type': 'array', 'items': {'type': 'integer'}},
                  {'type': 'array', 'items': {'type': 'boolean'}},
                  {'type': 'object'}]}}}]}

_input_fit_schema = {
  'description': 'Input data schema for training Project.',
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
  'description': 'Input data schema for transformation using Project.',
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
  'description': 'Output data schema for transformed data using Project.',
  'type': 'array',
  'items': {
    'type': 'array',
    'items': {
       'anyOf':[{'type': 'number'}, {'type':'string'}]}}}

_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.project.html',
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

Project = lale.operators.make_operator(ProjectImpl, _combined_schemas)
