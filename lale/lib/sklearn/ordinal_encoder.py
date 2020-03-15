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
import sklearn.preprocessing

class OrdinalEncoderImpl():
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        self._sklearn_model = sklearn.preprocessing.OrdinalEncoder(**self._hyperparams)

    def fit(self, X, y=None):
        self._sklearn_model.fit(X, y)
        return self

    def transform(self, X):
        return self._sklearn_model.transform(X)
        

_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Hyperparameter schema for the OrdinalEncoder model from scikit-learn.',
    'allOf': [
    {   'type': 'object',
        'additionalProperties': False,
        'required': ['categories', 'dtype'],
        'relevantToOptimizer': [],
        'properties': {
            'categories': {
                'anyOf': [
                {   'description':
                        'Determine categories automatically from training data.',
                    'enum': ['auto', None]},
                {   'description': 'The ith list element holds the categories expected in the ith column.',
                    'type': 'array',
                    'items': {
                        'anyOf': [
                        {   'type': 'array',
                            'items': {
                                'type': 'string'},
                        }, {
                            'type': 'array',
                            'items': {
                                'type': 'number'},
                            'description': 'Should be sorted.'}]}}],
                'default': 'auto'},
            'dtype': {
                'description': 'Desired dtype of output, must be number. See https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.scalars.html#arrays-scalars-built-in',
                'enum': ['byte', 'short', 'intc', 'int_', 'longlong', 'intp', 'int8', 'int16', 'int32', 'int64', 'ubyte', 'ushort', 'uintc', 'uint', 'ulonglong', 'uintp', 'uint16', 'uint32', 'uint64', 'half', 'single', 'double', 'float_', 'longfloat', 'float16', 'float32', 'float64', 'float96', 'float128'],
                'default': 'float64'},
        }}]}

_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Input data schema for training the OrdinalEncoder model from scikit-learn.',
    'type': 'object',
    'required': ['X'],
    'additionalProperties': False,
    'properties': {
        'X': {
            'description': 'Features; the outer array is over samples.',
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {'type': 'number'},
            }},
        'y': {
            'description': 'Target class labels; the array is over samples.'}}}

_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Input data schema for predictions using the OrdinalEncoder model from scikit-learn.',
    'type': 'object',
    'required': ['X'],
    'additionalProperties': False,
    'properties': {
        'X': {
            'description': 'Features; the outer array is over samples.',
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {'type': 'number'}}}}}

_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Output data schema for predictions (projected data) using the OrdinalEncoder model from scikit-learn.',
    'type': 'array',
    'items': {
        'type': 'array',
        'items': {
            'type': 'number'}}}

_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'type': 'object',
    'tags': {
        'pre': ['categoricals'],
        'op': ['transformer'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_transform': _input_transform_schema,
        'output_transform': _output_transform_schema }}

if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)

OrdinalEncoder = lale.operators.make_operator(OrdinalEncoderImpl, _combined_schemas)
