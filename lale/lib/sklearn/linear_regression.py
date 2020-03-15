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

import sklearn.linear_model.base
import lale.helpers
import lale.operators

class LinearRegressionImpl():

    def __init__(self, fit_intercept=True, normalize=False, copy_X=True, n_jobs=None):
        self._hyperparams = {
            'fit_intercept': fit_intercept,
            'normalize': normalize,
            'copy_X': copy_X,
            'n_jobs': n_jobs}
        self._sklearn_model = sklearn.linear_model.base.LinearRegression(**self._hyperparams)

    def fit(self, X, y, **fit_params):
        if fit_params is None:
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self._sklearn_model.predict(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Ordinary least squares Linear Regression.',
    'allOf': [{
        'type': 'object',
        'required': ['fit_intercept', 'normalize', 'copy_X'],
        'relevantToOptimizer': ['fit_intercept', 'normalize'],
        'additionalProperties': False,
        'properties': {
            'fit_intercept': {
                'type': 'boolean',
                'default': True,
                'description': 'whether to calculate the intercept for this model. If set'},
            'normalize': {
                'type': 'boolean',
                'default': False,
                'description': 'This parameter is ignored when ``fit_intercept`` is set to False.'},
            'copy_X': {
                'type': 'boolean',
                'default': True,
                'description': 'If True, X will be copied; else, it may be overwritten.'},
            'n_jobs': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The number of jobs to use for the computation. This will only provide'},
        }}, {
        'description': 'Normalize is ignored when fit_intercept is set to False.',
        'anyOf': [
        {   'type': 'object',
            'properties': {
                'fit_intercept': {
                    'enum': [True]},
            }},
        {   'type': 'object',
            'properties': {
                'normalize': {
                    'enum': [False]},
            }}]}]}

_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit linear model.',
    'type': 'object',
    'required': ['X', 'y'],
    'properties': {
        'X': {
            'description': 'Features; the outer array is over samples.',
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            }},
        'y': {
            'anyOf': [
            {   'type': 'array',
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }},
            {   'type': 'array',
                'items': {
                    'type': 'number'},
            }],
            'description': "Target values. Will be cast to X's dtype if necessary"},
        'sample_weight': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'type': 'number'},
            }, {
                'enum': [None]}],
            'description': 'Individual weights for each sample'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict using the linear model',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'Samples.'},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Returns predicted values.',
    'anyOf': [
    {   'type': 'array',
        'items': {
            'type': 'array',
            'items': {
                'type': 'number'},
        }},
    {   'type': 'array',
        'items': {
            'type': 'number'}}]}

_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html',
    'type': 'object',
    'tags': {
        'pre': [],
        'op': ['estimator', 'regressor'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_predict': _input_predict_schema,
        'output_predict': _output_predict_schema}}

if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
LinearRegression = lale.operators.make_operator(LinearRegressionImpl, _combined_schemas)
