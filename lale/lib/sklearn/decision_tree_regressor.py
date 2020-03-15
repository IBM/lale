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

import sklearn.tree.tree
import lale.helpers
import lale.operators

class DecisionTreeRegressorImpl():

    def __init__(self, criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False):
        self._hyperparams = {
            'criterion': criterion,
            'splitter': splitter,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'min_weight_fraction_leaf': min_weight_fraction_leaf,
            'max_features': max_features,
            'random_state': random_state,
            'max_leaf_nodes': max_leaf_nodes,
            'min_impurity_decrease': min_impurity_decrease,
            'min_impurity_split': min_impurity_split,
            'presort': presort}
        self._sklearn_model = sklearn.tree.tree.DecisionTreeRegressor(**self._hyperparams)

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
    'description': 'A decision tree regressor.',
    'allOf': [
    {   'type': 'object',
        'required': ['criterion', 'splitter', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'],
        'relevantToOptimizer': ['criterion', 'splitter', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'],
        'additionalProperties': False,
        'properties': {
            'criterion': {
                'description': 'Function to measure the quality of a split.',
                'enum': ['mse', 'friedman_mse', 'mae'],
                'default': 'mse'},
            'splitter': {
                'enum': ['best', 'random'],
                'default': 'best',
                'description': 'Strategy to choose the split at each node.'},
            'max_depth': {
                'description': 'Maximum depth of the tree.',
                'default': None,
                'anyOf': [
                {   'type': 'integer',
                    'minimum': 1,
                    'minimumForOptimizer': 3,
                    'maximumForOptimizer': 5},
                {   'enum': [None],
                    'description': 'If None, then nodes are expanded until all leaves are pure, or until all leaves contain less than min_samples_split samples.'}]},
            'min_samples_split': {
                'description': 'Minimum number of samples required to split an internal node.',
                'anyOf': [
                {   'type': 'integer',
                    'minimumForOptimizer': 2,
                    'maximumForOptimizer': 20,
                    'distribution': 'uniform'},
                {   'type': 'number',
                    'minimumForOptimizer': 0.01,
                    'maximumForOptimizer': 0.5}],
                'default': 2},
            'min_samples_leaf': {
                'description': 'Minimum number of samples required to be at a leaf node.',
                'anyOf': [
                {   'type': 'integer',
                    'minimumForOptimizer': 1,
                    'maximumForOptimizer': 20,
                    'distribution': 'uniform'},
                {   'type': 'number',
                    'minimumForOptimizer': 0.01,
                    'maximumForOptimizer': 0.5}],
                'default': 1},
            'min_weight_fraction_leaf': {
                'type': 'number',
                'default': 0.0,
                'description': 'Minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node.'},
            'max_features': {
                'description': 'The number of features to consider when looking for the best split.',
                'anyOf': [
                {   'type': 'integer',
                    'forOptimizer': False},
                {   'type': 'number',
                    'minimum': 0.0,
                    'exclusiveMinimum': True,
                    'minimumForOptimizer': 0.0,
                    'maximumForOptimizer': 1.0,
                    'distribution': 'uniform'},
                {   'enum': ['auto', 'sqrt', 'log2', None]}],
                'default': None},
            'random_state': {
                'anyOf': [
                {   'type': 'integer'},
                {   'type': 'object'},
                {   'enum': [None]}],
                'default': None},
            'max_leaf_nodes': {
                'anyOf': [
                {   'type': 'integer'},
                {   'enum': [None]}],
                'default': None,
                'description': 'Grow a tree with `max_leaf_nodes` in best-first fashion.'},
            'min_impurity_decrease': {
                'type': 'number',
                'default': 0.0,
                'description': 'A node will be split if this split induces a decrease of the impurity greater than or equal to this value.'},
            'min_impurity_split': {
                'anyOf':[
                {'type': 'number'},{
                    'enum': [None]
                }],
                'default': None,
                'description': 'Threshold for early stopping in tree growth.'},
            'presort': {
                'type': 'boolean',
                'default': False,
                'description': 'Whether to presort the data to speed up the finding of best splits in fitting.'},
        }}]}

_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Build a decision tree regressor from the training set (X, y).',
    'type': 'object',
    'required': ['X', 'y'],
    'properties': {
        'X': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }}],
            'description': 'The training input samples. Internally, it will be converted to'},
        'y': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'The target values (real numbers). Use ``dtype=np.float64`` and'},
        'sample_weight': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'type': 'number'},
            }, {
                'enum': [None]}],
            'description': 'Sample weights. If None, then samples are equally weighted. Splits'},
        'check_input': {
            'type': 'boolean',
            'default': True,
            'description': 'Allow to bypass several input checking.'},
        'X_idx_sorted': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }}, {
                'enum': [None]}],
            'default': None,
            'description': 'The indexes of the sorted training input samples. If many tree'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict class or regression value for X.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'The input samples. Internally, its dtype will be converted to'},
        'check_input': {
            'type': 'boolean',
            'default': True,
            'description': 'Allow to bypass several input checking.'},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'The predicted classes, or the predict values.',
    'type': 'array',
    'items': {
        'type': 'number'},
}
_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html',
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
DecisionTreeRegressor = lale.operators.make_operator(DecisionTreeRegressorImpl, _combined_schemas)
