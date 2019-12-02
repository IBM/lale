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
import sklearn.decomposition

class PCAImpl():
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams

    def fit(self, X, y=None):
        self._sklearn_model = sklearn.decomposition.PCA(**self._hyperparams)
        self._sklearn_model.fit(X, y)
        return self

    def transform(self, X):
        return self._sklearn_model.transform(X)

_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Hyperparameter schema for the PCA model from scikit-learn.',
    'allOf': [{
        'description': 'This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.\n',
        'type': 'object',
        'additionalProperties': False,
        'required': ['n_components', 'copy', 'whiten', 'svd_solver', 'tol', 'iterated_power', 'random_state'],
        'relevantToOptimizer': ['n_components', 'whiten', 'svd_solver'],
        'properties': {
            'n_components': {
                'anyOf': [{
                    'description': 'If not set, keep all components.',
                    'enum': [None]},
                {   'description': "Use Minka's MLE to guess the dimension.",
                    'enum': ['mle']},
                {   'description': 'Select the number of components such that the amount of variance that needs to be explained is greater than the specified percentage.',
                    'type': 'number',
                    'minimum': 0.0,
                    'exclusiveMinimum': True,
                    'maximum': 1.0,
                    'exclusiveMaximum': True},
                {   'description': 'Number of components to keep.',
                    'type': 'integer',
                    'minimum': 1,
                    'forOptimizer': False}],
                'default': None},
            'copy': {
                'description': 'If false, overwrite data passed to fit.',
                'default': True},
            'whiten': {
                'description': 'When true, multiply the components_ vectors by the square root of n_samples and then divide by the singular values to ensure uncorrelated outputs with unit component-wise variances.',
                'type': 'boolean',
                'default': False},
            'svd_solver': {
                'description': 'Algorithm to use.',
                'enum': ['auto', 'full', 'arpack', 'randomized'],
                'default': 'auto'},
            'tol': {
                'description': 'Tolerance for singular values computed by svd_solver arpack.',
                'type': 'number',
                'minimum': 0.0,
                'default': 0.0},
            'iterated_power': {
                'anyOf': [{
                    'description': 'Number of iterations for the power method computed by svd_solver randomized.',
                    'type': 'integer'}, {
                    'description': 'Pick automatically.',
                    'enum': ['auto']}],
                'default': 'auto'},
            'random_state': {
                'description': 'Seed of pseudo-random number generator for shuffling data.',
                'anyOf': [{
                    'description': 'RandomState used by np.random',
                    'enum': [None]}, {
                    'description': 'Explicit seed.',
                    'type': 'integer'}],
                'default': None},
        }},
    {   'description': 'Option n_components mle can only be set for svd_solver full or auto.',
        'anyOf': [
        {   'type': 'object',
            'properties': {
                'n_components': {
                    'not': {
                        'enum': ['mle']},
                }},
        },
        {   'type': 'object',
            'properties': {
                'svd_solver': {
                    'enum': ['full', 'auto']},
            }}]},
    {   'description': 'Setting 0 < n_components < 1 only works for svd_solver full.',
        'anyOf': [
        {   'type': 'object',
            'properties': {
                'n_components': {
                    'not': {
                        'description': 'Select the number of components such that the amount of variance that needs to be explained is greater than the specified percentage.',
                        'type': 'number',
                        'minimum': 0.0,
                        'exclusiveMinimum': True,
                        'maximum': 1.0,
                        'exclusiveMaximum': True},
                }}},
        {   'type': 'object',
            'properties': {
                'svd_solver': {
                    'enum': ['full']},
            }}]},
    {   'description': 'Option tol can be set for svd_solver arpack.',
        'anyOf': [
        {   'type': 'object',
            'properties': {
                'tol': {
                    'enum': [0.0]}}},
        {   'type': 'object',
            'properties': {
                'svd_solver': {
                    'enum': ['arpack']},
            }}]},
    {   'description': 'Option iterated_power can be set for svd_solver randomized.',
        'anyOf': [
        {   'type': 'object',
            'properties': {
                'iterated_power': {
                    'enum': ['auto']},
            }},
        {   'type': 'object',
            'properties': {
                'svd_solver': {
                    'enum': ['randomized']},
            }}]},
    {   'description': 'Option random_state can be set for svd_solver arpack or randomized.',
        'anyOf': [
        {   'type': 'object',
            'properties': {
                'random_state': {
                    'enum': [None]},
            }},
        {   'type': 'object',
            'properties': {
                'svd_solver': {
                    'enum': ['arpack', 'randomized']},
            }}]}]}

_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Input data schema for training the PCA model from scikit-learn.',
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

_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Input data schema for predictions using the PCA model from scikit-learn.',
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

_output_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Output data schema for predictions (projected data) using the PCA model from scikit-learn.',
    'type': 'array',
    'items': {
        'type': 'array',
        'items': {'type': 'number'}}}

_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html',
    'type': 'object',
    'tags': {
        'pre': ['~categoricals'],
        'op': ['transformer'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_predict': _input_predict_schema,
        'output': _output_schema }}

if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)

PCA = lale.operators.make_operator(PCAImpl, _combined_schemas)
