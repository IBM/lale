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

import sklearn.ensemble
import lale.docstrings
import lale.operators

class ExtraTreesClassifierImpl():
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        self._wrapped_model = sklearn.ensemble.ExtraTreesClassifier(**self._hyperparams)

    def fit(self, X, y, **fit_params):
        self._wrapped_model.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self._wrapped_model.predict(X)

    def predict_proba(self, X):
        return self._wrapped_model.predict_proba(X)

_hyperparams_schema = {
    'description': 'An extra-trees classifier.',
    'allOf': [{
        'type': 'object',
        'required': ['class_weight'],
        'relevantToOptimizer': ['n_estimators', 'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'bootstrap'],
        'additionalProperties': False,
        'properties': {
            'n_estimators': {
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 100,
                'default': 10,
                'description': 'The number of trees in the forest.'},
            'criterion': {
                'enum': ['gini', 'entropy'],
                'default': 'gini',
                'description': 'The function to measure the quality of a split. Supported criteria are'},
            'max_depth': {
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 3,
                    'maximumForOptimizer': 5}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The maximum depth of the tree. If None, then nodes are expanded until'},
            'min_samples_split': {
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 2,
                    'maximumForOptimizer': 20,
                    'distribution': 'uniform'}, {
                    'type': 'number',
                    'minimumForOptimizer': 0.01,
                    'maximumForOptimizer': 0.5}],
                'default': 2,
                'description': 'The minimum number of samples required to split an internal node:'},
            'min_samples_leaf': {
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 1,
                    'maximumForOptimizer': 20,
                    'distribution': 'uniform'}, {
                    'type': 'number',
                    'minimumForOptimizer': 0.01,
                    'maximumForOptimizer': 0.5}],
                'default': 1,
                'description': 'The minimum number of samples required to be at a leaf node.'},
            'min_weight_fraction_leaf': {
                'type': 'number',
                'default': 0.0,
                'description': 'The minimum weighted fraction of the sum total of weights (of all'},
            'max_features': {
                'anyOf': [{
                    'type': 'integer',
                    'forOptimizer': False}, {
                    'type': 'number',
                    'minimum': 0.0,
                    'exclusiveMinimum': True,
                    'minimumForOptimizer': 0.0,
                    'maximumForOptimizer': 1.0,
                    'distribution': 'uniform'}, {
                    'enum': ['auto', 'sqrt', 'log2', None]}],
                'default': 'auto',
                'description': 'The number of features to consider when looking for the best split:'},
            'max_leaf_nodes': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Grow trees with ``max_leaf_nodes`` in best-first fashion.'},
            'min_impurity_decrease': {
                'type': 'number',
                'default': 0.0,
                'description': 'A node will be split if this split induces a decrease of the impurity'},
            'min_impurity_split': {
                'anyOf':[
                {'type': 'number'},{
                    'enum': [None]
                }],
                'default': None,
                'description': 'Threshold for early stopping in tree growth. A node will split'},
            'bootstrap': {
                'type': 'boolean',
                'default': False,
                'description': 'Whether bootstrap samples are used when building trees. If False, the'},
            'oob_score': {
                'type': 'boolean',
                'default': False,
                'description': 'Whether to use out-of-bag samples to estimate'},
            'n_jobs': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The number of jobs to run in parallel for both `fit` and `predict`.'},
            'random_state': {
                'anyOf': [
                {   'type': 'integer'},
                {   'laleType': 'numpy.random.RandomState'},
                {   'enum': [None]}],
                'default': None,
                'description': 'If int, random_state is the seed used by the random number generator;'},
            'verbose': {
                'type': 'integer',
                'default': 0,
                'description': 'Controls the verbosity when fitting and predicting.'},
            'warm_start': {
                'type': 'boolean',
                'default': False,
                'description': 'When set to ``True``, reuse the solution of the previous call to fit'},
            'class_weight': {
                'anyOf': [{
                    'type': 'object'}, #dict, list of dicts, 
                    {'enum': ['balanced', 'balanced_subsample', None]}],
                'description': 'Weights associated with classes in the form ``{class_label: weight}``.',
                'default': None},
        }}]
}
_input_fit_schema = {
    'description': 'Build a forest of trees from the training set (X, y).',
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
            'description': 'The training input samples. Internally, its dtype will be converted'},
        'y': {
            'anyOf': [
                {'type': 'array', 'items': {'type': 'number'}},
                {'type': 'array', 'items': {'type': 'string'}},
                {'type': 'array', 'items': {'type': 'boolean'}}],
            'description': 'The target values (class labels in classification, real numbers in'},
        'sample_weight': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'type': 'number'},
            }, {
                'enum': [None]}],
            'description': 'Sample weights. If None, then samples are equally weighted. Splits'},
    },
}
_input_predict_schema = {
    'description': 'Predict class for X.',
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
    },
}
_output_predict_schema = {
    'description': 'The predicted classes.',
    'anyOf': [
        {'type': 'array', 'items': {'type': 'number'}},
        {'type': 'array', 'items': {'type': 'string'}},
        {'type': 'array', 'items': {'type': 'boolean'}}]}

_input_predict_proba_schema = {
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
            'description': 'The input samples. Internally, its dtype will be converted to'},
    },
}
_output_predict_proba_schema = {
    'description': 'such arrays if n_outputs > 1.',
    'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            }    
}
_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': """`Extra trees classifier`_ random forest from scikit-learn.

.. _`Extra trees classifier`: https://scikit-learn.org/0.20/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html#sklearn-ensemble-extratreesclassifier
""",
    'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.extra_trees_classifier.html',
    'import_from': 'sklearn.ensemble',
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

ExtraTreesClassifier : lale.operators.IndividualOp
ExtraTreesClassifier = lale.operators.make_operator(ExtraTreesClassifierImpl, _combined_schemas)

if sklearn.__version__ >= '0.22':
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    # new: https://scikit-learn.org/0.23/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
    from lale.schemas import AnyOf, Float, Int, Null
    ExtraTreesClassifier = ExtraTreesClassifier.customize_schema(
        n_estimators=Int(
            desc='The number of trees in the forest.',
            default=100,
            forOptimizer=True,
            minForOptimizer=10,
            maxForOptimizer=100),
        ccp_alpha=Float(
            desc='Complexity parameter used for Minimal Cost-Complexity Pruning. The subtree with the largest cost complexity that is smaller than ccp_alpha will be chosen. By default, no pruning is performed.',
            default=0.0,
            forOptimizer=True,
            min=0.0,
            maxForOptimizer=0.1),
        max_samples=AnyOf(
            types=[
                Null(desc='Draw X.shape[0] samples.'),
                Int(desc='Draw max_samples samples.', min=1),
                Float(desc='Draw max_samples * X.shape[0] samples.',
                      min=0.0, exclusiveMin=True, max=1.0, exclusiveMax=True)],
            desc='If bootstrap is True, the number of samples to draw from X to train each base estimator.',
            default=None))

lale.docstrings.set_docstrings(ExtraTreesClassifierImpl, ExtraTreesClassifier._schemas)
