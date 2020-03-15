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
import sklearn.kernel_approximation

class NystroemImpl():
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        self._sklearn_model = sklearn.kernel_approximation.Nystroem(**self._hyperparams)

    def fit(self, X, y=None):
        self._sklearn_model.fit(X, y)
        return self

    def transform(self, X):
        return self._sklearn_model.transform(X)

_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Hyperparameter schema for the Nystroem model from scikit-learn.',
    'allOf': [{
        'description': 'This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.',
        'type': 'object',
        'additionalProperties': False,
        'required': ['kernel', 'gamma', 'coef0', 'degree', 'n_components', 'random_state'],
        'relevantToOptimizer': ['kernel', 'gamma', 'coef0', 'degree', 'n_components'],
        'properties': {
            'kernel': {
                'description': 'Kernel map to be approximated. In the scikit learn version, this can be a string or a callable. To keep arguments as plain JSON documents, the wrapper only allows an enum of the keys of sklearn.metrics.pairwise.KERNEL_PARAMS.',
                'enum': ['additive_chi2', 'chi2', 'cosine', 'linear', 'poly', 'polynomial', 'rbf', 'laplacian', 'sigmoid'],
                'default': 'rbf'},
            'gamma': {
                'description': 'Gamma parameter.',
                'anyOf': [{
                    'enum': [None]}, {
                    'type': 'number',
                    'distribution': 'loguniform',
                    'minimumForOptimizer': 3.0517578125e-05,
                    'maximumForOptimizer': 8}],
                'default': None},
            'coef0': {
                'description': 'Zero coefficient.',
                'anyOf': [{
                    'enum': [None]}, {
                    'type': 'number',
                    'minimum': (- 1),
                    'distribution': 'uniform',
                    'maximumForOptimizer': 1}],
                'default': None},
            'degree': {
                'description': 'Degree of the polynomial kernel.',
                'anyOf': [{
                    'enum': [None]}, {
                    'type': 'integer',
                    'minimumForOptimizer': 2,
                    'maximumForOptimizer': 5}],
                'default': None},
            'kernel_params':{
                'description': 'Additional parameters (keyword arguments) for kernel '
                'function passed as callable object.',
                'anyOf':[
                {'type':'object'},
                {'enum':[None]}],
                'default': None
            },
            'n_components': {
                'description': 'Number of features to construct. How many data points will be used to construct the mapping.',
                'type': 'integer',
                'default': 100,
                'minimum': 1,
                'distribution': 'loguniform',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 256},
            'random_state': {
                'description': 'Seed of pseudo-random number generator.',
                'anyOf': [{
                    'description': 'RandomState used by np.random',
                    'enum': [None]}, {
                    'description': 'Explicit seed.',
                    'type': 'integer'}],
                'default': None},
        }},
    {   'description': 'Gamma is ignored by other kernels.',
        'anyOf': [{
            'type': 'object',
            'properties': {
                'gamma': {
                    'enum': [None]},
            }}, {
            'type': 'object',
            'properties': {
                'kernel': {
                    'enum': ['rbf', 'laplacian', 'polynomial', 'additive_chi2', 'sigmoid']},
            }}]},
    {   'description': 'Zero coefficient ignored by other kernels.',
        'anyOf': [{
            'type': 'object',
            'properties': {
                'coef0': {
                    'enum': [None]},
            }}, {
            'type': 'object',
            'properties': {
                'kernel': {
                    'enum': ['polynomial', 'sigmoid']},
            }}]},
    {   'description': 'Degree ignored by other kernels.',
        'anyOf': [{
            'type': 'object',
            'properties': {
                'degree': {
                    'enum': [None]},
            }}, {
            'type': 'object',
            'properties': {
                'kernel': {
                    'enum': ['polynomial']},
            }}]}]}

_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Input data schema for training the Nystroem model from scikit-learn.',
    'type': 'object',
    'required': ['X'],
    'additionalProperties': False,
    'properties': {
        'X': {
            'description': 'Features; the outer array is over samples.',
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {'type': 'number'}}},
        'y': {
            'description': 'Target class labels; the array is over samples.'}}}

_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Input data schema for predictions using the Nystroem model from scikit-learn.',
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
    'description': 'Output data schema for predictions (projected data) using the Nystroem model from scikit-learn.',
    'type': 'array',
    'items': {
        'type': 'array',
        'items': {'type': 'number'}}}

_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.kernel_approximation.Nystroem.html',
    'type': 'object',
    'tags': {
        'pre': ['~categoricals'],
        'op': ['transformer'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_transform': _input_transform_schema,
        'output_transform': _output_transform_schema}}

if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)

Nystroem = lale.operators.make_operator(NystroemImpl, _combined_schemas)
