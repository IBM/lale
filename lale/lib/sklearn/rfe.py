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

from sklearn.feature_selection import RFE as SKLModel
import lale.helpers
import lale.operators

class RFEImpl():
    def __init__(self, estimator, n_features_to_select=None, step=1, verbose=0):
        self._hyperparams = {
            'estimator': estimator,
            'n_features_to_select': n_features_to_select,
            'step': step,
            'verbose': verbose}
        self._sklearn_model = SKLModel(**self._hyperparams)

    def fit(self, X, y):
        self._sklearn_model.fit(X, y)
        return self

    def transform(self, X):
        return self._sklearn_model.transform(X)

_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Feature ranking with recursive feature elimination.',
    'allOf': [{
        'type': 'object',
        'required': ['estimator', 'n_features_to_select', 'step', 'verbose'],
        'relevantToOptimizer': [],
        'additionalProperties': False,
        'properties': {
            'estimator': {
                'description': 'A supervised learning estimator with a fit method that provides information about feature importance either through a coef_ attribute or through a feature_importances_ attribute.',
                'laleType': 'estimator'},
            'n_features_to_select': {
                'description': 'The number of features to select. If None, half of the features are selected.',
                'anyOf': [
                {   'type': 'integer',
                    'minimum': 1},
                {   'enum': [None]}],
                'default': None},
            'step': {
                'description': 'If greater than or equal to 1, then step corresponds to the (integer) number of features to remove at each iteration. If within (0.0, 1.0), then step corresponds to the percentage (rounded down) of features to remove at each iteration.',
                'anyOf': [
                {   'type': 'integer',
                    'minimum': 1},
                {   'type': 'number',
                    'minimum': 0, 'exclusiveMinimum': True,
                    'maximum': 1, 'exclusiveMaximum': True}],
                'default': 1},
            'verbose': {
                'anyOf': [{
                    'type': 'boolean'}, {
                    'type': 'integer'}],
                'default': 0,
                'description': 'Controls verbosity of output.'}}}]}

_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the model to data matrix X and target(s) y.',
    'type': 'object',
    'required': ['X', 'y'],
    'additionalProperties': False,
    'properties': {
        'X': {
            'description': 'Features; the outer array is over samples.',
            'type': 'array',
            'items': {
                'type': 'array',
                'items': { 'type': 'number'}}},
        'y': {
            'description': 'Target class labels; the array is over samples.',
            'type': 'array',
            'items': { 'type': 'number'}}}}

_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Reduce X to the selected features.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': { 'type': 'number'},
            },
            'description': 'The input samples.'}}}

_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'type': 'array',
    'items': {
        'type': 'array',
        'items': {
            'type': 'number'}}}

_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFE.html',
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

RFE = lale.operators.make_operator(RFEImpl, _combined_schemas)
