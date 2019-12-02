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

from lale.lib.sklearn import RandomForestRegressor
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from lale.helpers import cross_val_score_track_trials, create_instance_from_hyperopt_search_space
from lale.search.op2hp import hyperopt_search_space
from lale.search.PGO import PGO
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import warnings
import numpy as np
import sys
import time
import logging
import traceback
from typing import Optional
import lale.operators

SEED=42
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class HyperoptRegressorImpl:

    def __init__(self, estimator = None, max_evals=50, cv=5, handle_cv_failure = False, max_opt_time=None, pgo:Optional[PGO]=None):
        self.max_evals = max_evals
        if estimator is None:
            self.estimator = RandomForestRegressor
        else:
            self.estimator = estimator
        self.search_space = hp.choice('meta_model', [hyperopt_search_space(self.estimator, pgo=pgo)])
        self.handle_cv_failure = handle_cv_failure
        self.cv = cv
        self.trials = Trials()
        self.max_opt_time = max_opt_time


    def fit(self, X_train, y_train):
        opt_start_time = time.time()

        def hyperopt_train_test(params, X_train, y_train):
            warnings.filterwarnings("ignore")

            reg = create_instance_from_hyperopt_search_space(self.estimator, params)
            try:
                cv_score, _, execution_time = cross_val_score_track_trials(reg, X_train, y_train, cv=KFold(self.cv), scoring = 'r2')
                logger.debug("Successful trial of hyperopt")
            except BaseException as e:
                #If there is any error in cross validation, use the accuracy based on a random train-test split as the evaluation criterion
                if self.handle_cv_failure:
                    X_train_part, X_validation, y_train_part, y_validation = train_test_split(X_train, y_train, test_size=0.20)
                    start = time.time()
                    reg_trained = reg.fit(X_train_part, y_train_part)
                    predictions = reg_trained.predict(X_validation)
                    execution_time = time.time() - start
                    cv_score = r2_score(y_validation, predictions)
                else:
                    logger.debug(e)
                    logger.debug("Error {} with pipeline:{}".format(e, reg.to_json()))
                    raise e

            return cv_score, execution_time

        def get_final_trained_reg(params, X_train, y_train):
            warnings.filterwarnings("ignore")
            reg = create_instance_from_hyperopt_search_space(self.estimator, params)
            reg = reg.fit(X_train, y_train)
            return reg

        def f(params):
            current_time = time.time()
            if (self.max_opt_time is not None) and ((current_time - opt_start_time) > self.max_opt_time) :
                # if max optimization time set, and we have crossed it, exit optimization completely
                sys.exit(0)

            try:
                r_squared, execution_time = hyperopt_train_test(params, X_train=X_train, y_train=y_train)
            except BaseException as e:
                logger.warning(f'Exception caught in HyperoptRegressor: {type(e)}, {traceback.format_exc()} with hyperparams: {params}, setting loss to zero')
                r_squared = 0
                execution_time = 0
            return {'loss': -r_squared, 'time': execution_time, 'status': STATUS_OK}


        try :
            fmin(f, self.search_space, algo=tpe.suggest, max_evals=self.max_evals, trials=self.trials, rstate=np.random.RandomState(SEED))
        except SystemExit :
            logger.warning('Maximum alloted optimization time exceeded. Optimization exited prematurely')

        try :
            best_params = space_eval(self.search_space, self.trials.argmin)
            logger.info('best accuracy: {:.1%}\nbest hyperparams found using {} hyperopt trials: {}'.format(-1*self.trials.average_best_error(), self.max_evals, best_params))
            trained_reg = get_final_trained_reg(best_params, X_train, y_train)
            self.best_estimator = trained_reg
        except BaseException as e :
            logger.warning('Unable to extract the best parameters from optimization, the error: {}'.format(e))
            trained_reg = None


        return trained_reg

    def predict(self, X_eval):
        import warnings
        warnings.filterwarnings("ignore")
        reg = self.best_estimator
        try:
            predictions = reg.predict(X_eval)
        except ValueError as e:
            logger.warning("ValueError in predicting using regressor:{}, the error is:{}".format(reg, e))
            predictions = None

        return predictions

    def get_trials(self):
        return self.trials

_hyperparams_schema = {
    'allOf': [
    {   'type': 'object',
        'required': [
            'estimator', 'max_evals', 'cv', 'handle_cv_failure',
            'max_opt_time', 'pgo'],
        'relevantToOptimizer': ['estimator'],
        'additionalProperties': False,
        'properties': {
            'estimator': {
                'anyOf': [
                {   'typeForOptimizer': 'operator'},
                {   'enum': [None],
                    'description':
                        'lale.lib.sklearn.RandomForestRegressor'}],
                'default': None},
            'max_evals': {
                'type': 'integer',
                'minimum': 1,
                'default': 50},
            'cv': {
                'type': 'integer',
                'minimum': 1,
                'default': 5},
            'handle_cv_failure': {
                'type': 'boolean',
                'default': False},
            'max_opt_time': {
                'anyOf': [
                {   'type': 'number',
                    'minimum': 0.0},
                {   'enum': [None]}],
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
    'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.hyperopt_regressor.html',
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

HyperoptRegressor = lale.operators.make_operator(HyperoptRegressorImpl, _combined_schemas)

if __name__ == '__main__':
    from lale.lib.lale import ConcatFeatures
    from lale.lib.sklearn import Nystroem, PCA, RandomForestRegressor
    from sklearn.metrics import r2_score
    pca = PCA(n_components=3)
    nys = Nystroem(n_components=3)
    concat = ConcatFeatures()
    rf = RandomForestRegressor()

    trainable = (pca & nys) >> concat >> rf
    #trainable = nys >>rf
    import sklearn.datasets
    from lale.helpers import cross_val_score
    diabetes = sklearn.datasets.load_diabetes()
    X, y = sklearn.utils.shuffle(diabetes.data, diabetes.target, random_state=42)

    hp_n = HyperoptRegressor(estimator=trainable, max_evals=20)

    hp_n_trained = hp_n.fit(X, y)
    predictions = hp_n_trained.predict(X)
    mse = r2_score(y, [round(pred) for pred in predictions])
    print(mse)
