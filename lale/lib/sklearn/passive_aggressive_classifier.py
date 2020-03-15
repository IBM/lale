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


import sklearn.linear_model.passive_aggressive
import lale.helpers
import lale.operators

class PassiveAggressiveClassifierImpl():

    def __init__(self, C=1.0, fit_intercept=True, max_iter=None, tol=None, early_stopping=False, 
    validation_fraction=0.1, n_iter_no_change=5, shuffle=True, verbose=0, loss='hinge', 
    n_jobs=None, random_state=None, warm_start=False, class_weight=None, average=False):
    #The wrapper does not support n_iter as it is deprecated and will be removed in sklearn 0.21.
        self._hyperparams = {
            'C': C,
            'fit_intercept': fit_intercept,
            'max_iter': max_iter,
            'tol': tol,
            'early_stopping': early_stopping,
            'validation_fraction': validation_fraction,
            'n_iter_no_change': n_iter_no_change,
            'shuffle': shuffle,
            'verbose': verbose,
            'loss': loss,
            'n_jobs': n_jobs,
            'random_state': random_state,
            'warm_start': warm_start,
            'class_weight': class_weight,
            'average': average}
        self._sklearn_model = sklearn.linear_model.passive_aggressive.PassiveAggressiveClassifier(**self._hyperparams)

    def fit(self, X, y=None):
        self._sklearn_model.fit(X, y)
        return self

    def predict(self, X):
        return self._sklearn_model.predict(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Passive Aggressive Classifier',
    'allOf': [{
        'type': 'object',
        'additionalProperties': False,
        'required': ['C', 'fit_intercept', 'max_iter', 'tol', 'early_stopping', 
            'shuffle', 'loss', 'average'],
        'relevantToOptimizer': ['C', 'fit_intercept', 'max_iter', 'tol', 'early_stopping', 
            'shuffle', 'loss', 'average'],         
        'properties': {
            'C': {
                'type': 'number',
                'description': 'Maximum step size (regularization). Defaults to 1.0.',
                'default': 1.0,
                'distribution': 'loguniform',
                'minimumForOptimizer': 1e-5,
                'maximumForOptimizer': 10},
            'fit_intercept': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether the intercept should be estimated or not. If False, the'
                'the data is assumed to be already centered.'},
            'max_iter': {
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 5,
                    'maximumForOptimizer': 1000,
                    'distribution': 'uniform',
                    'default': 5}, #default value is 1000 for sklearn 0.21.
                    {'enum': [None]}],
                'default': None,
                'description': 'The maximum number of passes over the training data (aka epochs).'},
            'tol': {
                'anyOf': [{
                    'type': 'number',
                    'minimumForOptimizer': 1e-08,
                    'maximumForOptimizer': 0.01,
                    'distribution': 'loguniform'}, {
                    'enum': [None]}],
                'default': None, #default value is 1e-3 from sklearn 0.21.
                'description': 'The stopping criterion. If it is not None, the iterations will stop'},
            'early_stopping': {
                'type': 'boolean',
                'default': False,
                'description': 'Whether to use early stopping to terminate training when validation.'},
            'validation_fraction': {
                'type': 'number',
                'default': 0.1,
                'description': 'The proportion of training data to set aside as validation set for'},
            'n_iter_no_change': {
                'type': 'integer',
                'minimumForOptimizer': 5,
                'maximumForOptimizer': 10,
                'default': 5,
                'description': 'Number of iterations with no improvement to wait before early stopping.'},
            'shuffle': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether or not the training data should be shuffled after each epoch.'},
            'verbose': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': 0,
                'description': 'The verbosity level'},
            'loss': {
                'enum': ['hinge', 'squared_hinge'],
                'default': 'hinge',
                'description': 'The loss function to be used:'},
            'n_jobs': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The number of CPUs to use to do the OVA (One Versus All, for'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The seed of the pseudo random number generator to use when shuffling'},
            'warm_start': {
                'type': 'boolean',
                'default': False,
                'description': 'When set to True, reuse the solution of the previous call to' 
                ' fit as initialization, otherwise, just erase the previous solution.'},
            'class_weight': {
                'anyOf': [{
                    'type': 'object'}, {
                    'enum': ['balanced', None]}],
                'default': None,
                'description': 'Preset for the class_weight fit parameter.'},
            'average': {
                'anyOf': [{
                    'type': 'boolean'}, {
                    'type': 'integer',
                    'forOptimizer': False}],
                'default': False,
                'description': 'When set to True, computes the averaged SGD weights and stores the'}
        }},
        {'description': 'validation_fraction, only used if early_stopping is true',
        'anyOf': [{
            'type': 'object',
            'properties': {
                'early_stopping': {
                    'enum': [True]},
            }}, {
            'type': 'object',
            'properties': {
                'validation_fraction': {
                    'enum': [0.1]}, #i.e. it should not have a value other than its default.
            }}]}]}

_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit linear model with Passive Aggressive algorithm.',
    'type': 'object',
    'required': ['X', 'y'],
    'properties': {
        'X': {
            'description': 'Training data',
            'type': 'array',
            'items': {
                'type': 'array',
                'items': { 'type': 'number'}}},
        'y': {
            'description': 'Target values',
            'anyOf': [
                {'type': 'array', 'items': {'type': 'number'}},
                {'type': 'array', 'items': {'type': 'string'}}]
            },
        'coef_init': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': { 'type': 'number'}},
            'description': 'The initial coefficients to warm-start the optimization.'},
        'intercept_init': {
            'type': 'array',
            'items': {
                    'type': 'number'},
            'description': 'The initial intercept to warm-start the optimization.'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict class labels for samples in X.',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'description': 'Test data',
            'type': 'array',
            'items': {
                'type': 'array',
                'items': { 'type': 'number'}}},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict class labels for samples in X.',
    'anyOf': [
        {'type': 'array', 'items': {'type': 'number'}},
        {'type': 'array', 'items': {'type': 'string'}}]}

_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveClassifier.html',
    'type': 'object',
    'tags': {
        'pre': [],
        'op': ['estimator', 'classifier'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_predict': _input_predict_schema,
        'output_predict': _output_predict_schema}}

if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
PassiveAggressiveClassifier = lale.operators.make_operator(PassiveAggressiveClassifierImpl, _combined_schemas)

