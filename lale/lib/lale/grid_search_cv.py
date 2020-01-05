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

import lale.lib.sklearn
import lale.search.lale_grid_search_cv
import lale.operators
import lale.sklearn_compat

class GridSearchCVImpl:
    def __init__(self, estimator=None, cv=5, scoring='accuracy', n_jobs=None, lale_num_samples=None, lale_num_grids=None, param_grid=None, pgo=None):
        self._hyperparams = {
            'estimator': estimator,
            'cv': cv,
            'scoring': scoring,
            'n_jobs': n_jobs,
            'lale_num_samples': lale_num_samples,
            'lale_num_grids': lale_num_grids,
            'pgo': pgo,
            'hp_grid': param_grid }

    def fit(self, X, y):
        if self._hyperparams['estimator'] is None:
            op = lale.lib.sklearn.LogisticRegression
        else:
            op = self._hyperparams['estimator']
        hp_grid = self._hyperparams['hp_grid']
        if hp_grid is None:
            hp_grid = lale.search.lale_grid_search_cv.get_parameter_grids(
                op,
                num_samples=self._hyperparams['lale_num_samples'],
                num_grids=self._hyperparams['lale_num_grids'],
                pgo=self._hyperparams['pgo'])
        if not hp_grid and isinstance(op, lale.operators.IndividualOp):
            hp_grid = [
                lale.search.lale_grid_search_cv.get_defaults_as_param_grid(op)]
        self.grid = lale.search.lale_grid_search_cv.get_lale_gridsearchcv_op(
            lale.sklearn_compat.make_sklearn_compat(op),
            hp_grid,
            cv=self._hyperparams['cv'],
            scoring=self._hyperparams['scoring'],
            n_jobs=self._hyperparams['n_jobs'])
        self.grid.fit(X, y)
        self.best_estimator = self.grid.best_estimator_.to_lale()
        return self

    def predict(self, X):
        return self.best_estimator.predict(X)

_hyperparams_schema = {
    'allOf': [
    {   'type': 'object',
        'required': [
            'estimator', 'cv', 'scoring', 'n_jobs', 'lale_num_samples',
            'lale_num_grids', 'pgo'],
        'relevantToOptimizer': ['estimator'],
        'additionalProperties': False,
        'properties': {
            'estimator': {
                'anyOf': [
                {   'typeForOptimizer': 'operator',
                    'not': {'enum': [None]}},
                {   'enum': [None],
                    'description': 'lale.lib.sklearn.LogisticRegression'}],
                'default': None},
            'cv': {
                'type': 'integer',
                'minimum': 1,
                'default': 5},
            'scoring': {
                'anyOf': [
                {   'description': 'Custom scorer object, see https://scikit-learn.org/stable/modules/model_evaluation.html',
                    'not': {'type': 'string'}},
                {   'enum': [
                        'accuracy', 'explained_variance', 'max_error',
                        'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo',
                        'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted',
                        'balanced_accuracy', 'average_precision',
                        'neg_log_loss', 'neg_brier_score']},
                {   'enum': [
                        'r2', 'neg_mean_squared_error',
                        'neg_mean_absolute_error',
                        'neg_root_mean_squared_error',
                        'neg_mean_squared_log_error',
                        'neg_median_absolute_error']}],
                'default': 'accuracy'},
            'n_jobs': {
                'anyOf': [
                    {   'description':
                            '1 unless in joblib.parallel_backend context.',
                        'enum': [None]},
                    {   'description': 'Use all processors.',
                        'enum': [-1]},
                    {   'description': 'Number of jobs to run in parallel.',
                        'type': 'integer',
                        'minimum': 1}],
                'default': None},
            'lale_num_samples': {
                'anyOf': [
                    {   'description': 'How many samples to draw when discretizing a continuous hyperparameter.',
                        'type': 'integer',
                        'minimum': 1},
                    {   'enum': [None]}],
                'default': None},
            'lale_num_grids': {
                'anyOf': [
                {   'description': 'If not set, keep all grids.',
                    'enum': [None]},
                {   'description': 'Fraction of grids to keep.',
                    'type': 'number',
                    'minimum': 0.0,
                    'exclusiveMinimum': True,
                    'maximum': 1.0,
                    'exclusiveMaximum': True},
                {   'description': 'Number of grids to keep.',
                    'type': 'integer',
                    'minimum': 1}],
                'default': None},
            'param_grid': {
                'anyOf':[
                {
                'enum': [None],
                'description': 'Generated automatically.'},
                {'description': 'Dictionary of hyperparameter ranges in the grid.'}],
                'default': None},
            'pgo': {
                'anyOf': [
                {   'description': 'lale.search.PGO'},
                {   'enum': [None]}],
                'default': None}}}]}

_input_fit_schema = {
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {'type': ['number', 'string']}}},
        'y': {
            'type': 'array', 'items': {'type': 'number'}}}}

_input_predict_schema = {
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {'type': ['number', 'string']}}}}}

_output_predict_schema = {
    'type': 'array', 'items': {'type': 'number'}}

_combined_schemas = {
    'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.hyperopt_classifier.html',
    'type': 'object',
    'tags': {
        'pre': [],
        'op': ['estimator'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_predict': _input_predict_schema,
        'output': _output_predict_schema}}

GridSearchCV = lale.operators.make_operator(GridSearchCVImpl, _combined_schemas)
