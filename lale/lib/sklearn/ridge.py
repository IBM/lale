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

import sklearn.linear_model.ridge
import lale.helpers
import lale.operators

class RidgeImpl():

    def __init__(self, alpha=None, fit_intercept=None, normalize=False, copy_X=True, max_iter=None, tol=0.001, solver=None, random_state=None):
        self._hyperparams = {
            'alpha': alpha,
            'fit_intercept': fit_intercept,
            'normalize': normalize,
            'copy_X': copy_X,
            'max_iter': max_iter,
            'tol': tol,
            'solver': solver,
            'random_state': random_state}

    def fit(self, X, y, **fit_params):
        self._sklearn_model = sklearn.linear_model.ridge.Ridge(**self._hyperparams)
        if fit_params is None:
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self._sklearn_model.predict(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Linear least squares with l2 regularization.',
    'allOf': [{
        'type': 'object',
        'required': ['alpha', 'fit_intercept', 'solver'],
        'relevantToOptimizer': ['alpha', 'fit_intercept', 'normalize', 'copy_X', 'max_iter', 'tol', 'solver'],
        'additionalProperties': False,
        'properties': {
            'alpha': {
                'anyOf': [{
                    'type': 'number',
                    'minimum': 0.0,
                    'exclusiveMinimum': True,
                    'minimumForOptimizer': 1e-05,
                    'maximumForOptimizer': 10.0,
                    'distribution': 'loguniform'                   
                    }, {
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                    'forOptimizer': False}],
                'default': 1.0,
                'description': 'Regularization strength; must be a positive float. Regularization'},
            'fit_intercept': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether to calculate the intercept for this model. If set'},
            'normalize': {
                'type': 'boolean',
                'default': False,
                'description': 'This parameter is ignored when ``fit_intercept`` is set to False.'},
            'copy_X': {
                'type': 'boolean',
                'default': True,
                'description': 'If True, X will be copied; else, it may be overwritten.'},
            'max_iter': {
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 10,
                    'maximumForOptimizer': 1000}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Maximum number of iterations for conjugate gradient solver.'},
            'tol': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'distribution': 'loguniform',
                'default': 0.001,
                'description': 'Precision of the solution.'},
            'solver': {
                'enum': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
                'default': 'auto',
                'description': 'Solver to use in the computational routines:'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The seed of the pseudo random number generator to use when shuffling'},
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
            }}]},
        {
        'description': 'random_state is used when solver == ‘sag’',
        'anyOf': [
        {   'type': 'object',
            'properties': {
                'solver': {'enum': ['sag']},
            }},
        {   'type': 'object',
            'properties': {
                'random_state': {
                    'enum': [None]},
            }}]},
        {'description': 'Maximum number of iterations for conjugate gradient solver',
        'anyOf': [
        {   'type': 'object',
            'properties': {
                'solver': {'enum': ['sparse_cg', 'lsqr', 'sag', 'saga']},
            }},
        {   'type': 'object',
            'properties': {
                'max_iter': {
                    'enum': [None]},
            }}]}]}

_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit Ridge regression model',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'Training data'},
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
            'description': 'Target values'},
        'sample_weight': {
            'anyOf': [{
                'type': 'number'}, {
                'type': 'array',
                'items': {
                    'type': 'number'},
            }, {'enum': [None]}],
            'description': 'Individual weights for each sample'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict using the linear model',
    'type': 'object',
    'properties': {
        'X': {
            'anyOf': [{
                'type': 'array',
                'items': {'type': 'number'}}, {
                'type': 'array',
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }}],
            'description': 'Samples.'},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Returns predicted values.',
    'type': 'array',
    'items': {
        'type': 'number'},
}
_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html',
    'type': 'object',
    'tags': {
        'pre': [],
        'op': ['estimator'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_predict': _input_predict_schema,
        'output': _output_predict_schema},
}
if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
Ridge = lale.operators.make_operator(RidgeImpl, _combined_schemas)
