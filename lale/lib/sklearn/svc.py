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


import sklearn.svm.classes
import lale.helpers
import lale.operators


class SVCImpl():

    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None):
        self._hyperparams = {
            'C': C,
            'kernel': kernel,
            'degree': degree,
            'gamma': gamma,
            'coef0': coef0,
            'shrinking': shrinking,
            'probability': probability,
            'tol': tol,
            'cache_size': cache_size,
            'class_weight': class_weight,
            'verbose': verbose,
            'max_iter': max_iter,
            'decision_function_shape': decision_function_shape,
            'random_state': random_state}
        self._sklearn_model = sklearn.svm.classes.SVC(**self._hyperparams)

    def fit(self, X, y=None, sample_weight=None):
        self._sklearn_model.fit(X, y, sample_weight)
        return self

    def predict(self, X):
        return self._sklearn_model.predict(X)

    def predict_proba(self, X):
        return self._sklearn_model.predict_proba(X)


_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'C-Support Vector Classification.',
    'allOf': [{
        'type': 'object',
        'additionalProperties': False,
        'required': ['kernel', 'degree', 'gamma', 'shrinking', 'tol', 'cache_size', 'max_iter', 'decision_function_shape'],
        'relevantToOptimizer': ['kernel', 'degree', 'gamma', 'shrinking', 'probability', 'tol'],
        'properties': {
            'C': {
                'type': 'number',
                'description': 'Penalty parameter C of the error term.',
                'distribution': 'loguniform',
                'default': 1.0,
                'minimumForOptimizer': 0.03125,
                'maximumForOptimizer': 32768},
            'kernel': {
                'anyOf': [
                    {'enum':['precomputed'], 'forOptimizer': False}, 
                    {'enum': ['linear', 'poly', 'rbf', 'sigmoid']}],
                    #support for callable is missing as of now.               
                'default': 'rbf',
                'description': 'Specifies the kernel type to be used in the algorithm.'},
            'degree': {
                'type': 'integer',
                'minimumForOptimizer': 2,
                'maximumForOptimizer': 5,
                'default': 3,
                'description': "Degree of the polynomial kernel function ('poly')."},
            'gamma': {
                'anyOf': [{
                    'type': 'number',
                    'minimumForOptimizer': 3.0517578125e-05,
                    'maximumForOptimizer': 8,
                    'distribution': 'loguniform'},
                    {'enum': ['auto', 'auto_deprecated', 'scale']}
                    ],
                'default': 'auto_deprecated', #going to change to 'scale' from sklearn 0.22.
                'description': "Kernel coefficient for 'rbf', 'poly' and 'sigmoid'."},
            'coef0': {
                'type': 'number',
                'default': 0.0,
                'description': 'Independent term in kernel function.'},
            'shrinking': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether to use the shrinking heuristic.'},
            'probability': {
                'type': 'boolean',
                'default': False,
                'description': 'Whether to enable probability estimates.'},
            'tol': {
                'type': 'number',
                'distribution':'loguniform',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'default': 0.001,
                'description': 'Tolerance for stopping criterion.'},
            'cache_size': {
                'type': 'integer',
                'default': 200,
                'description': 'Specify the size of the kernel cache (in MB).'},
            'class_weight': {
                'anyOf': [{
                    'type': 'object'}, {
                    'enum': ['balanced', None]}],
                'default': None,
                'description': 'Set the parameter C of class i to class_weight[i]*C for SVC'},
            'verbose': {
                'type': 'boolean',
                'default': False,
                'description': 'Enable verbose output.'},
            'max_iter': {
                'type': 'integer',
                'default': -1,
                'description': 'Hard limit on iterations within solver, or -1 for no limit.'},
            'decision_function_shape': {
                'enum': ['ovo', 'ovr'],
                'default': 'ovr',
                'description': "Whether to return a one-vs-rest ('ovr') decision function of shape"},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The seed of the pseudo random number generator used when shuffling'},
        }},
        {'description': 'gamma only used when kernel is ‘rbf’, ‘poly’ or ‘sigmoid’',
         'anyOf': [{
             'type': 'object',
             'properties': {
                 'kernel': {
                     'enum': ['rbf', 'poly', 'sigmoid']},
             }}, {
             'type': 'object',
             'properties': {
                 'gamma': {
                     'enum': ['auto_deprecated']}, #this needs to be the default value of gamma, changes with sklearn versions.
             }}]},
        {'description': 'coef0 only significant in kernel ‘poly’ and ‘sigmoid’.',
         'anyOf': [{
             'type': 'object',
             'properties': {
                 'kernel': {
                     'enum': ['poly', 'sigmoid']},
             }}, {
             'type': 'object',
             'properties': {
                 'coef0': {
                     'enum': [0.0]},
             }}]}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the SVM model according to the given training data.',
    'type': 'object',
    'required': ['X', 'y'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {'type': 'array', 'items': {'type': 'number'}},
            'description': 'Training vectors, where n_samples is the number of samples and n_features is the number of features.'},
        'y': {
            'anyOf': [
                {'type': 'array', 'items': {'type': 'number'}},
                {'type': 'array', 'items': {'type': 'string'}}],
            'description': 'Target values (class labels in classification, real numbers in regression)'},
        'sample_weight': {
            'anyOf': [{
                    'type': 'array',
                    'items': {'type': 'number'},
                    'description': 'Per-sample weights. Rescale C per sample.'},
                {'enum': [None]}],
            'default': None,
        },
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Perform classification on samples in X.',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {'type': 'array', 'items': {'type': 'number'}}
        },
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Class labels for samples in X.',
    'anyOf': [
        {'type': 'array', 'items': {'type': 'number'}},
        {'type': 'array', 'items': {'type': 'string'}}]}

_input_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {'type': 'array', 'items': {'type': 'number'}},
        },
    },
}
_output_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'type': 'array',
    'items': {'type': 'array', 'items': {'type': 'number'}},
    'description': 'Returns the probability of the sample for each class in the model.'

}
_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html',
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
SVC = lale.operators.make_operator(SVCImpl, _combined_schemas)
