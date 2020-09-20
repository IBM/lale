# Copyright 2020 IBM Corporation
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

import hyperopt
import lale.docstrings
import lale.operators

def auto_prep(X):
    from lale.lib.lale import ConcatFeatures
    from lale.lib.lale import Project
    from lale.lib.lale import categorical
    from lale.lib.sklearn import OneHotEncoder
    from lale.lib.sklearn import SimpleImputer
    n_cols = X.shape[1]
    n_cats = len(categorical()(X))
    prep_num = SimpleImputer(strategy='mean')
    prep_cat = (SimpleImputer(strategy='most_frequent')
                >> OneHotEncoder(handle_unknown='ignore'))
    if n_cats == 0:
        result = prep_num
    elif n_cats == n_cols:
        result = prep_cat
    else:
        result = (
            (Project(columns={'type': 'number'}, drop_columns=categorical())
             >> prep_num)
            & (Project(columns=categorical()) >> prep_cat)
        ) >> ConcatFeatures
    return result

class AutoPipelineImpl:
    def __init__(self, prediction_type='classification', scoring=None,
                 max_evals=100, max_opt_time=600.0, max_eval_time=120.0):
        self.prediction_type = prediction_type
        if scoring is None:
            scoring = 'r2' if prediction_type=='regression' else 'accuracy'
        self.scoring = scoring
        self.max_evals = max_evals
        self.max_opt_time = max_opt_time
        self.max_eval_time = max_eval_time

    def fit(self, X, y):
        from lale.lib.lale import Hyperopt
        from lale.lib.sklearn import KNeighborsClassifier
        from lale.lib.sklearn import KNeighborsRegressor
        from lale.lib.sklearn import LinearRegression
        from lale.lib.sklearn import LogisticRegression
        from lale.lib.sklearn import RandomForestClassifier
        from lale.lib.sklearn import RandomForestRegressor
        prep = auto_prep(X)
        if self.prediction_type == 'regression':
            estimator = (LinearRegression | RandomForestRegressor
                         | KNeighborsRegressor)
        else:
            estimator = (LogisticRegression | RandomForestClassifier
                         | KNeighborsClassifier)
        planned_pipeline = prep >> estimator
        self.hyperopt = Hyperopt(
            estimator=planned_pipeline,
            max_evals=self.max_evals,
            scoring=self.scoring,
            max_opt_time=self.max_opt_time,
            max_eval_time=self.max_eval_time,
            verbose=True)
        self.hyperopt.fit(X, y)
        return self

    def predict(self, X):
        result = self.hyperopt.predict(X)
        return result

    def summary(self):
        """Table summarizing the trial results (name, tid, loss, time, log_loss, status).
Returns
-------
result : DataFrame"""
        result = self.hyperopt.summary()
        return result

    def get_pipeline(self, pipeline_name=None, astype='lale'):
        """Retrieve one of the trials.
Parameters
----------
pipeline_name : union type, default None
    - string
        Key for table returned by summary(), return a trainable pipeline.
    - None
        When not specified, return the best trained pipeline found.
astype : 'lale' or 'sklearn', default 'lale'
    Type of resulting pipeline.
Returns
-------
result : Trained operator if best, trainable operator otherwise.
"""
        result = self.hyperopt.get_pipeline(pipeline_name, astype)
        return result

_hyperparams_schema = {
    'allOf': [
    {   'type': 'object',
        'required': [
            'prediction_type', 'scoring',
            'max_evals', 'max_opt_time', 'max_eval_time'],
        'relevantToOptimizer': [],
        'additionalProperties': False,
        'properties': {
            'prediction_type': {
                'description': 'The kind of learning problem.',
                'enum': [
                    'binary', 'multiclass', 'classification', 'regression'],
                'default': 'classification'},
            'scoring': {
                'description': 'Scorer object or known scorer named by string.',
                'anyOf': [
                {    'description': 'If None, use accuracy for classification and r2 for regression.',
                     'enum': [None]},                     
                {    'description': """Custom scorer object created with `make_scorer`_.

The argument to make_scorer can be one of scikit-learn's metrics_,
or it can be a user-written Python function to create a completely
custom scorer object, following the `model_evaluation`_ example.
The metric has to return a scalar value. Note that scikit-learns's
scorer object always returns values such that higher score is
better.

.. _`make_scorer`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer.
.. _metrics: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
.. _`model_evaluation`: https://scikit-learn.org/stable/modules/model_evaluation.html
""",
                     'not': {'type': ['string', 'null']}},
                {   'description': 'Known scorer for classification task.',
                    'enum': [
                        'accuracy', 'explained_variance', 'max_error',
                        'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo',
                        'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted',
                        'balanced_accuracy', 'average_precision',
                        'neg_log_loss', 'neg_brier_score']},
                {   'description': 'Known scorer for regression task.',
                    'enum': [
                        'r2', 'neg_mean_squared_error',
                        'neg_mean_absolute_error',
                        'neg_root_mean_squared_error',
                        'neg_mean_squared_log_error',
                        'neg_median_absolute_error']}],
                'default': None},
            'max_evals': {
                'description': 'Number of trials of Hyperopt search.',
                'type': 'integer',
                'minimum': 1,
                'default': 100},
            'max_opt_time': {
                'description': 'Maximum time in seconds for the optimization.',
                'anyOf': [
                {   'type': 'number',
                    'minimum': 0.0, 'exclusiveMinimum': True},
                {   'description': 'No runtime bound.',
                    'enum': [None]}],
                'default': 600.0},
            'max_eval_time': {
                'description': 'Maximum time in seconds for each evaluation.',
                'anyOf': [
                {   'type': 'number',
                    'minimum': 0.0, 'exclusiveMinimum': True},
                {   'description': 'No runtime bound.',
                    'enum': [None]}],
                'default': 120.0}}}]}

_input_fit_schema = {
    'type': 'object',
    'required': ['X', 'y'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {'laleType': 'Any'}}},
        'y': {
            'anyOf': [
            {   'type': 'array', 'items': {'type': 'number'}},
            {   'type': 'array', 'items': {'type': 'string'}},
            {   'type': 'array', 'items': {'type': 'boolean'}}]}}}

_input_predict_schema = {
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {'laleType': 'Any'}}}}}

_output_predict_schema = {
    'anyOf': [
    {   'type': 'array', 'items': {'type': 'number'}},
    {   'type': 'array', 'items': {'type': 'string'}},
    {   'type': 'array', 'items': {'type': 'boolean'}}]}

_combined_schemas = {
    'description': """Automatically find a pipeline for a dataset.

This is a high-level entry point to get an initial trained pipeline
without having to specify your own planned pipeline first. It is
designed to be simple at the expense of not offering much control.""",
    'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.auto_pipelines.html',
    'import_from': 'lale.lib.lale',
    'type': 'object',
    'tags': {
        'pre': [],
        'op': ['estimator'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_predict': _input_predict_schema,
        'output_predict': _output_predict_schema}}

lale.docstrings.set_docstrings(AutoPipelineImpl, _combined_schemas)

AutoPipeline = lale.operators.make_operator(AutoPipelineImpl, _combined_schemas)
