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

import sklearn.discriminant_analysis
import lale.helpers
import lale.operators

class QuadraticDiscriminantAnalysisImpl():

    def __init__(self, priors=None, reg_param=None, store_covariance=None, tol=0.0001, store_covariances=None):
        self._hyperparams = {
            'priors': priors,
            'reg_param': reg_param,
            'store_covariance': store_covariance,
            'tol': tol,
            'store_covariances': store_covariances}
        self._sklearn_model = sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis(**self._hyperparams)

    def fit(self, X, y=None):
        self._sklearn_model.fit(X, y)
        return self

    def predict(self, X):
        return self._sklearn_model.predict(X)

    def predict_proba(self, X):
        return self._sklearn_model.predict_proba(X)

_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Quadratic Discriminant Analysis',
    'allOf': [{
        'type': 'object',
        'required': ['priors', 'store_covariance'],
        'relevantToOptimizer': ['reg_param', 'tol'],
        'additionalProperties': False,
        'properties': {
            'priors': {
                'anyOf': [{
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }, {
                    'enum': [None]}],
                'default': None,
                'description': 'Priors on classes'},
            'reg_param': {
                'type': 'number',
                'minimumForOptimizer': 0.0,
                'maximumForOptimizer': 1.0,
                'distribution': 'uniform',
                'default': 0.0,
                'description': 'Regularizes the covariance estimate as'},
            'store_covariance': {
                'type': 'boolean',
                'default': False,
                'description': 'If True the covariance matrices are computed and stored in the'},
            'tol': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'distribution': 'loguniform',
                'default': 0.0001,
                'description': 'Threshold used for rank estimation.'}
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the model according to the given training data and parameters.',
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
            'description': 'Training vector, where n_samples is the number of samples and'},
        'y': {
            'anyOf': [
                {'type': 'array', 'items': {'type': 'number'}},
                {'type': 'array', 'items': {'type': 'string'}}],
            'description': 'Target values (integers)'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Perform classification on an array of test vectors X.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            }},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Perform classification on an array of test vectors X.',
    'anyOf': [
        {'type': 'array', 'items': {'type': 'number'}},
        {'type': 'array', 'items': {'type': 'string'}}]}

_input_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Return posterior probabilities of classification.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'Array of samples/test vectors.'},
    },
}
_output_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Posterior probabilities of classification per class.',
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
    'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis.html',
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
QuadraticDiscriminantAnalysis = lale.operators.make_operator(QuadraticDiscriminantAnalysisImpl, _combined_schemas)
