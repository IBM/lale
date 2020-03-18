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

import lale.docstrings
import lale.helpers
import lale.operators
import sklearn.linear_model

_input_fit_schema = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'type': 'object',
  'required': ['X', 'y'],
  'additionalProperties': False,
  'properties': {
    'X': {
      'description': 'Features; the outer array is over samples.',
      'type': 'array',
      'items': {'type': 'array', 'items': {'type': 'number'}}},
    'y': {
      'description': 'Target class labels; the array is over samples.',
        'anyOf': [
            {'type': 'array', 'items': {'type': 'number'}},
            {'type': 'array', 'items': {'type': 'string'}}]}}}

_input_predict_schema = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'type': 'object',
  'required': ['X'],
  'additionalProperties': False,
  'properties': {
    'X': {
      'description': 'Features; the outer array is over samples.',
      'type': 'array',
      'items': {'type': 'array', 'items': {'type': 'number'}}}}}

_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predicted class label per sample.',
    'anyOf': [
        {'type': 'array', 'items': {'type': 'number'}},
        {'type': 'array', 'items': {'type': 'string'}}]}

_input_predict_proba_schema = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'type': 'object',
  'required': ['X'],
  'additionalProperties': False,
  'properties': {
    'X': {
      'description': 'Features; the outer array is over samples.',
      'type': 'array',
      'items': {'type': 'array', 'items': {'type': 'number'}}}}}

_output_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Probability of the sample for each class in the model.',
    'type': 'array',
    'items': {
        'type': 'array',
        'items': {
            'type': 'number'}}}

_hyperparams_schema = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'description': 'Hyperparameter schema.',
  'allOf': [
    { 'description':
        'This first sub-object lists all constructor arguments with their '
        'types, one at a time, omitting cross-argument constraints.',
      'type': 'object',
      'additionalProperties': False,
      'required': [
        'penalty', 'dual', 'tol', 'C', 'fit_intercept', 'intercept_scaling',
        'class_weight', 'random_state', 'solver', 'max_iter', 'multi_class',
        'verbose', 'warm_start', 'n_jobs'],
      'relevantToOptimizer': [
        'penalty', 'dual', 'tol', 'C', 'fit_intercept', 'class_weight',
        'solver', 'multi_class'],
      'properties': {
        'solver': {
          'description': 'Algorithm for optimization problem.',
          'enum': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
          'default': 'liblinear'},
        'penalty': {
          'description': 'Norm used in the penalization.',
          'enum': ['l1', 'l2'],
          'default': 'l2'},
        'dual': {
          'description': 'Dual or primal formulation.',
          'type': 'boolean',
          'default': False},
        'C': {
          'description':
            'Inverse regularization strength. Smaller values specify '
            'stronger regularization.',
          'type': 'number',
          'distribution': 'loguniform',
          'minimum': 0.0,
          'exclusiveMinimum': True,
          'default': 1.0,
          'minimumForOptimizer': 0.03125,
          'maximumForOptimizer': 32768},
        'tol': {
          'description': 'Tolerance for stopping criteria.',
          'type': 'number',
          'distribution': 'loguniform',
          'minimum': 0.0,
          'exclusiveMinimum': True,
          'default': 0.0001,
          'minimumForOptimizer': 1e-05,
          'maximumForOptimizer': 0.1},
        'fit_intercept': {
          'description':
            'Specifies whether a constant (bias or intercept) should be '
            'added to the decision function.',
          'type': 'boolean',
          'default': True},
        'intercept_scaling': {
          'description':
            'Append a constant feature with constant value '
            'intercept_scaling to the instance vector.',
          'type': 'number',
          'distribution': 'loguniform',
          'minimum': 0.0,
          'exclusiveMinimum': True,
          'default': 1.0},
        'class_weight': {
          'anyOf': [
            { 'description': 'By default, all classes have weight 1.',
              'enum': [None]},
            { 'description': 'Adjust weights by inverse frequency.',
              'enum': ['balanced']},
            { 'description': 'Dictionary mapping class labels to weights.',
              'type': 'object',
              'propertyNames': {'pattern': '^.+$', 'type': 'number'},
              'forOptimizer': False}],
          'default': None},
        'random_state': {
          'description':
            'Seed of pseudo-random number generator for shuffling data.',
          'anyOf': [
            { 'description': 'RandomState used by np.random',
              'enum': [None]},
            { 'description': 'Explicit seed.',
              'type': 'integer'}],
          'default': None},
        'max_iter': {
          'description':
            'Maximum number of iterations for solvers to converge.',
          'type': 'integer',
          'distribution': 'loguniform',
          'minimum': 1,
          'default': 100},
        'multi_class': {
          'description':
            'Approach for more than two classes (not binary classifier).',
          'enum': ['ovr', 'multinomial', 'auto'],
          'default': 'ovr'},
        'verbose': {
          'description':
            'For the liblinear and lbfgs solvers set verbose to any positive '
            'number for verbosity.',
          'type': 'integer',
          'default': 0},
        'warm_start': {
          'description':
            'If true, initialize with solution of previous call to fit.',
          'type': 'boolean',
          'default': False},
        'n_jobs': {
          'description':
            'Number of CPU cores when parallelizing over classes if '
            'multi_class is ovr.',
          'anyOf': [
            { 'description': '1 unless in joblib.parallel_backend context.',
              'enum': [None]},
            { 'description': 'Use all processors.',
              'enum': [-1]},
            { 'description': 'Number of CPU cores.',
              'type': 'integer',
              'minimum': 1}],
            'default': None}}},
      { 'description':
          'The newton-cg, sag, and lbfgs solvers support only l2 penalties.',
        'anyOf': [
          { 'type': 'object',
            'properties': {
              'solver': {'not': {'enum': ['newton-cg', 'sag', 'lbfgs']}}}},
          { 'type': 'object',
            'properties': {'penalty': {'enum': ['l2']}}}]},
      { 'description':
          'The dual formulation is only implemented for l2 '
          'penalty with the liblinear solver.',
        'anyOf': [
          { 'type': 'object',
            'properties': {'dual': {'enum': [False]}}},
          { 'type': 'object',
            'properties': {
              'penalty': {'enum': ['l2']},
              'solver': {'enum': ['liblinear']}}}]},
      { 'description':
          'Setting intercept_scaling is useful only when the solver is '
          'liblinear and fit_intercept is true.',
        'anyOf': [
          { 'type': 'object',
            'properties': {'intercept_scaling': {'enum': [1.0]}}},
          { 'type': 'object',
            'properties': {
              'fit_intercept': {'enum': [True]},
              'solver': {'enum': ['liblinear']}}}]},
      { 'description':
          'Setting max_iter is only useful for the newton-cg, sag, '
          'lbfgs solvers.',
        'anyOf': [
          { 'type': 'object',
            'properties': {'max_iter': {'enum': [100]}}},
          { 'type': 'object',
            'properties': {
              'solver': {'enum': ['newton-cg', 'sag', 'lbfgs']}}}]},
      { 'description':
          'The multi_class multinomial option is unavailable when the '
          'solver is liblinear.',
        'anyOf': [
          { 'type': 'object',
            'properties': {
              'multi_class': {'not': {'enum': ['multinomial']}}}},
          { 'type': 'object',
            'properties': {
              'solver': {'not': {'enum': ['liblinear']}}}}]}]}

_combined_schemas = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'description': """`Logistic regression`_ linear model for classification.

.. _`Logistic regression`: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
""",
  'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.logistic_regression.html',
  'type': 'object',
  'tags': {
    'pre': ['~categoricals'],
    'op': ['estimator', 'classifier', 'interpretable'],
    'post': ['probabilities']},
  'properties': {
    'hyperparams': _hyperparams_schema,
    'input_fit': _input_fit_schema,
    'input_predict': _input_predict_schema,
    'output_predict': _output_predict_schema,
    'input_predict_proba': _input_predict_proba_schema,
    'output_predict_proba': _output_predict_proba_schema}}

if __name__ == "__main__":
    lale.helpers.validate_is_schema(_combined_schemas)

class LogisticRegressionImpl:
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        self._sklearn_model = sklearn.linear_model.LogisticRegression(
            **self._hyperparams)

    def fit(self, X, y, **fit_params):
        if fit_params is None:
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X, y, **fit_params)
        return self

    def predict(self, X):
        return self._sklearn_model.predict(X)

    def predict_proba(self, X):
        return self._sklearn_model.predict_proba(X)

lale.docstrings.set_docstrings(LogisticRegressionImpl, _combined_schemas)

LogisticRegression = lale.operators.make_operator(LogisticRegressionImpl, _combined_schemas)
