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

from sklearn.ensemble.weight_boosting import AdaBoostClassifier as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class AdaBoostClassifierImpl():

    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None):
        if isinstance(base_estimator, lale.operators.Operator):
            if isinstance(base_estimator, lale.operators.IndividualOp):
                base_estimator = base_estimator._impl_instance()._sklearn_model
            else:
                raise ValueError("If base_estimator is a Lale operator, it needs to be an individual operator. ")
        self._hyperparams = {
            'base_estimator': base_estimator,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'algorithm': algorithm,
            'random_state': random_state}
        self._sklearn_model = SKLModel(**self._hyperparams)

    def fit(self, X, y=None):
        if (y is not None):
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X)
        return self

    def predict(self, X):
        return self._sklearn_model.predict(X)

    def predict_proba(self, X):
        return self._sklearn_model.predict_proba(X)

_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Hyperparameter schema.',
    'allOf': [{
        'type': 'object',
        'required': ['base_estimator', 'n_estimators', 'learning_rate', 'algorithm', 'random_state'],
        'relevantToOptimizer': ['n_estimators', 'learning_rate', 'algorithm'],
        'additionalProperties': False,
        'properties': {
            'base_estimator': {
                'anyOf': [{'laleType' : 'operator'},
                {'enum': [None]}],
                'default': None,
                'description': 'The base estimator from which the boosted ensemble is built.'},
            'n_estimators': {
                'type': 'integer',
                'minimumForOptimizer': 50,
                'maximumForOptimizer': 500,
                'distribution': 'uniform',
                'default': 50,
                'description': 'The maximum number of estimators at which boosting is terminated.'},
            'learning_rate': {
                'type': 'number',
                'minimumForOptimizer': 0.01,
                'maximumForOptimizer': 1.0,
                'distribution': 'loguniform',
                'default': 1.0,
                'description': 'Learning rate shrinks the contribution of each classifier by'},
            'algorithm': {
                'enum': ['SAMME', 'SAMME.R'],
                'default': 'SAMME.R',
                'description': "If 'SAMME.R' then use the SAMME.R real boosting algorithm."},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'If int, random_state is the seed used by the random number generator;'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Build a boosted classifier from the training set (X, y).',
    'type': 'object',    
    'required': ['X', 'y'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'The training input samples. Sparse matrix can be CSC, CSR, COO,'},
        'y': {
            'anyOf': [
                {'type': 'array', 'items': {'type': 'number'}},
                {'type': 'array', 'items': {'type': 'string'}}],
            'description': 'The target values (class labels).'},
        'sample_weight': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'type': 'number'},
            }, {
                'enum': [None]}],
            'default': None,
            'description': 'Sample weights. If None, the sample weights are initialized to'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict classes for X.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'The training input samples. Sparse matrix can be CSC, CSR, COO,'},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'The predicted classes.',
    'anyOf': [
        {'type': 'array', 'items': {'type': 'number'}},
        {'type': 'array', 'items': {'type': 'string'}}]}

_input_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict class probabilities for X.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'The training input samples. Sparse matrix can be CSC, CSR, COO,'},
    },
}
_output_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'The class probabilities of the input samples. The order of',
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
    'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html',
    'type': 'object',
    'tags': {
        'pre': [],
        'op': ['estimator', 'classifier'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_predict': _input_predict_schema,
        'output_predict': _output_predict_schema,
        'input_predict_proba': _input_predict_proba_schema,
        'output_predict_proba': _output_predict_proba_schema}}

if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
AdaBoostClassifier = lale.operators.make_operator(AdaBoostClassifierImpl, _combined_schemas)

