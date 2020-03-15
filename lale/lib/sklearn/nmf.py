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

from sklearn.decomposition import NMF as SKLModel
import lale.helpers
import lale.operators

class NMFImpl():
    def __init__(self, n_components=None, init=None, solver='cd', beta_loss='frobenius', tol=0.0001, max_iter=200, random_state=None, alpha=0.0, l1_ratio=0.0, verbose=0, shuffle=False):
        self._hyperparams = {
            'n_components': n_components,
            'init': init,
            'solver': solver,
            'beta_loss': beta_loss,
            'tol': tol,
            'max_iter': max_iter,
            'random_state': random_state,
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'verbose': verbose,
            'shuffle': shuffle}
        self._sklearn_model = SKLModel(**self._hyperparams)

    def fit(self, X, y=None):
        if (y is not None):
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X)
        return self

    def transform(self, X):
        return self._sklearn_model.transform(X)

_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Non-Negative Matrix Factorization (NMF)',
    'allOf': [{
        'type': 'object',
        'required': ['n_components', 'init', 'solver', 'beta_loss', 'tol', 'max_iter', 'random_state', 'alpha', 'l1_ratio', 'verbose', 'shuffle'],
        'relevantToOptimizer': ['n_components', 'tol', 'max_iter', 'alpha', 'shuffle'],
        'additionalProperties': False,
        'properties': {
            'n_components': {
                'anyOf': [
                {   'type': 'integer',
                    'minimum': 1,
                    'minimumForOptimizer': 2,
                    'maximumForOptimizer': 256,
                    'distribution': 'uniform'},
                {   'enum': [None]}],
                'default': None,
                'description': 'Number of components, if n_components is not set all features'},
            'init': {
                'enum': ['custom', 'nndsvd', 'nndsvda', 'nndsvdar', 'random', None],
                'default': None,
                'description': 'Method used to initialize the procedure.'},
            'solver': {
                'enum': ['cd', 'mu'],
                'default': 'cd',
                'description': 'Numerical solver to use:'},
            'beta_loss': {
                'description': 'Beta divergence to be minimized, measuring the distance between X and the dot product WH.',
                'anyOf': [
                {   'type': 'number'},
                {   'enum': ['frobenius', 'kullback-leibler', 'itakura-saito']}],
                'default': 'frobenius' },
            'tol': {
                'type': 'number',
                'minimum': 0.0,
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'distribution': 'loguniform',
                'default': 0.0001,
                'description': 'Tolerance of the stopping condition.'},
            'max_iter': {
                'type': 'integer',
                'minimum': 1,
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 200,
                'description': 'Maximum number of iterations before timing out.'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'If int, random_state is the seed used by the random number generator;'},
            'alpha': {
                'type': 'number',
                'minimumForOptimizer': 1e-10,
                'maximumForOptimizer': 1.0,
                'distribution': 'loguniform',
                'default': 0.0,
                'description': 'Constant that multiplies the regularization terms. Set it to zero to have no regularization.'},
            'l1_ratio': {
                'type': 'number',
                'default': 0.0,
                'minimum': 0.0,
                'maximum': 1.0,
                'description': 'The regularization mixing parameter.'},
            'verbose': {
                'anyOf': [{
                    'type': 'boolean'}, {
                    'type': 'integer'}],
                'default': 0,
                'description': 'Whether to be verbose.'},
            'shuffle': {
                'type': 'boolean',
                'default': False,
                'description': 'If true, randomize the order of coordinates in the CD solver.'},
        }}, {
        'description': "beta_loss, only in 'mu' solver",
        'anyOf': [{
            'type': 'object',
            'properties': {
                'beta_loss': {
                    'enum': ['frobenius']},
            }}, {
            'type': 'object',
            'properties': {
                'solver': {
                    'enum': ['mu']},
            }}]}],
}

_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'type': 'object',
    'required': ['X'],
    'additionalProperties': False,
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number', 'minimum': 0.0},
            }},
        'y': {}}}

_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number', 'minimum': 0.0}}}}}

_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transformed data',
    'type': 'array',
    'items': {
        'type': 'array',
        'items': {
            'type': 'number'},
    },
}

_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.NMF.html',
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

NMF = lale.operators.make_operator(NMFImpl, _combined_schemas)
