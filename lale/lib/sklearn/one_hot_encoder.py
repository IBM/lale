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

class OneHotEncoderImpl():
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams

    def fit(self, X, y=None):
        self._sklearn_model = sklearn.preprocessing.OneHotEncoder(**self._hyperparams)
        self._sklearn_model.fit(X, y)
        if isinstance(X, pd.DataFrame):
            cols_X = [str(c) for c in X.columns]
            self._feature_names = self._sklearn_model.get_feature_names(cols_X)
        else:
            self._feature_names = self._sklearn_model.get_feature_names()
        return self

    def transform(self, X):
        return self._sklearn_model.transform(X)

    def get_feature_names(self, input_features=None):
        """Return feature names for output features after this transformation. 
        This uses the output features obtained from scikit-learn's
        OneHotEncoder's get_feature_names method.
        https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder.get_feature_names
        The precedence of input feature names used is as follows:
        1. input_features passed as an argument to this function.
           This is a list string of length n_features.
        2. feature_names obtained at the time of training.
           They are input data's column names if it was a Pandas dataframe.
        3. feature_names obtained at the time of training.
           If training data was not a Pandas DataFrame, this method returns
           the output of scikit's get_feature_names(None).
        Returns
        -------
        output_feature_names : array of string, length n_output_features
        """
        try:
            trained_sklearn_model = self._sklearn_model
        except AttributeError:
            raise ValueError('Can only call get_feature_names on a trained operator. Please call fit to get a trained operator.')
        if input_features is not None:
            return trained_sklearn_model.get_feature_names(input_features)
        elif self._feature_names is not None:
            return self._feature_names

_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Hyperparameter schema for the OneHotEncoder model from scikit-learn.',
    'allOf': [
    {   'description': 'This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.',
        'type': 'object',
        'additionalProperties': False,
        'required': ['categories', 'sparse', 'dtype', 'handle_unknown'],
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
                'default': None},
            'sparse': {
                'description':
                    'Will return sparse matrix if set true, else array.',
                'type': 'boolean',
                'default': True},
            'dtype': {
                'description': 'Desired dtype of output, must be number. See https://docs.scipy.org/doc/numpy-1.14.0/reference/arrays.scalars.html#arrays-scalars-built-in',
                'enum': ['byte', 'short', 'intc', 'int_', 'longlong', 'intp', 'int8', 'int16', 'int32', 'int64', 'ubyte', 'ushort', 'uintc', 'uint', 'ulonglong', 'uintp', 'uint16', 'uint32', 'uint64', 'half', 'single', 'double', 'float_', 'longfloat', 'float16', 'float32', 'float64', 'float96', 'float128'],
                'default': 'float64'},
            'handle_unknown': {
                'description': 'Whether to raise an error or ignore if an unknown categorical feature is present during transform.',
                'enum': ['error', 'ignore'],
                'default': 'error'},
        }}]}

_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Input data schema for training the OneHotEncoder model from scikit-learn.',
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
                    'anyOf':[{'type': 'number'}, {'type':'string'}]},
            }},
        'y': {
            'description': 'Target class labels; the array is over samples.'}}}

_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Input data schema for predictions using the OneHotEncoder model from scikit-learn.',
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
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Output data schema for predictions (projected data) using the OneHotEncoder model from scikit-learn. See the official documentation for details: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html\n',
    'type': 'array',
    'items': {
        'type': 'array',
        'items': {
            'type': 'number'}}}

_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html',
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

OneHotEncoder = lale.operators.make_operator(OneHotEncoderImpl, _combined_schemas)
