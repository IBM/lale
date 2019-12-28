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

from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval
from lale.helpers import cross_val_score_track_trials, create_instance_from_hyperopt_search_space
from lale.search.op2hp import hyperopt_search_space
from lale.search.PGO import PGO
from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import check_cv
from sklearn.metrics import log_loss
from sklearn.metrics.scorer import check_scoring
import warnings
import numpy as np
import traceback

import time
import logging
from typing import Optional
import json
import datetime
import copy
import sys
import lale.operators
from lale.lib.sklearn import LogisticRegression

SEED=42
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class HyperoptCVImpl:

    def __init__(self, estimator=None, max_evals=50, cv=5, handle_cv_failure=False, scoring='accuracy', best_score=0.0, max_opt_time=None, pgo:Optional[PGO]=None):
        """ Instantiate the HyperoptCV that will use the given estimator and other parameters to select the 
        best performing trainable instantiation of the estimator. 

        Parameters
        ----------
        estimator : lale.operators.IndividualOp or lale.operators.Pipeline, optional
            A valid Lale individual operator or pipeline, by default LogisticRegression
        max_evals : int, optional
            Number of trials of Hyperopt search, by default 50
        cv : an integer or an object that has a split function as a generator yielding (train, test) splits as arrays of indices.
            Integer value is used as number of folds in sklearn.model_selection.StratifiedKFold, default is 5.
            Note that any of the iterators from https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators can be used here.
            The fit method performs cross validation on the input dataset for per trial, 
            and uses the mean cross validation performance for optimization. This behavior is also impacted by handle_cv_failure flag, 
            by default 5
        handle_cv_failure : bool, optional
            A boolean flag to indicating how to deal with cross validation failure for a trial.
            If True, the trial is continued by doing a 80-20 percent train-validation split of the dataset input to fit
            and reporting the score on the validation part.
            If False, the trial is terminated by assigning status to FAIL.
            , by default False
        scoring: string or a scorer object created using 
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer.
            A string from sklearn.metrics.SCORERS.keys() can be used or a scorer created from one of 
            sklearn.metrics (https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics).
            A completely custom scorer object can be created from a python function following the example at 
            https://scikit-learn.org/stable/modules/model_evaluation.html
            The metric has to return a scalar value, and note that scikit-learns's scorer object always returns values such that
            higher score is better. Since Hyperopt solves a minimization problem, we pass (best_score - score) to Hyperopt.
            by default 'accuracy'.
        best_score : float, optional
            The best score for the specified scorer. This allows us to return a loss to hyperopt that is
            greater than equal to zero, where zero is the best loss. By default, zero.
        max_opt_time : float, optional
            Maximum amout of time in seconds for the optimization. By default, None, implying no runtime
            bound.
        pgo : Optional[PGO], optional
            [description], by default None
        

        Examples
        --------
        >>> from sklearn.metrics import make_scorer, f1_score, accuracy_score
        >>> lr = LogisticRegression()
        >>> clf = HyperoptCV(estimator=lr, scoring='accuracy', cv=5, max_evals=2)
        >>> from sklearn import datasets
        >>> diabetes = datasets.load_diabetes()
        >>> X = diabetes.data[:150]
        >>> y = diabetes.target[:150]
        >>> trained = clf.fit(X, y)
        >>> predictions = trained.predict(X)

        Other scoring metrics:

        >>> clf = HyperoptCV(estimator=lr, scoring=make_scorer(f1_score, average='macro'), cv=3, max_evals=2)

        """
        self.max_evals = max_evals
        if estimator is None:
            self.estimator = LogisticRegression()
        else:
            self.estimator = estimator
        self.search_space = hp.choice('meta_model', [hyperopt_search_space(self.estimator, pgo=pgo)])
        self.scoring = scoring
        self.best_score = best_score
        self.handle_cv_failure = handle_cv_failure
        self.cv = cv
        self.trials = Trials()
        self.max_opt_time = max_opt_time


    def fit(self, X_train, y_train):
        opt_start_time = time.time()
        self.cv = check_cv(self.cv, y = y_train, classifier=True) #TODO: Replace the classifier flag value by using tags?
        def hyperopt_train_test(params, X_train, y_train):
            warnings.filterwarnings("ignore")

            trainable = create_instance_from_hyperopt_search_space(self.estimator, params)
            try:
                cv_score, logloss, execution_time = cross_val_score_track_trials(trainable, X_train, y_train, cv=self.cv, scoring=self.scoring)
                logger.debug("Successful trial of hyperopt")
            except BaseException as e:
                #If there is any error in cross validation, use the score based on a random train-test split as the evaluation criterion
                if self.handle_cv_failure:
                    X_train_part, X_validation, y_train_part, y_validation = train_test_split(X_train, y_train, test_size=0.20)
                    start = time.time()
                    trained = trainable.fit(X_train_part, y_train_part)
                    scorer = check_scoring(trainable, scoring=self.scoring)
                    cv_score  = scorer(trained, X_validation, y_validation)
                    execution_time = time.time() - start
                    y_pred_proba = trained.predict_proba(X_validation)
                    try:
                        logloss = log_loss(y_true=y_validation, y_pred=y_pred_proba)
                    except BaseException:
                        logloss = 0
                        logger.debug("Warning, log loss cannot be computed")
                else:
                    logger.debug(e)
                    logger.debug("Error {} with pipeline:{}".format(e, trainable.to_json()))
                    raise e
            return cv_score, logloss, execution_time
        def get_final_trained_estimator(params, X_train, y_train):
            warnings.filterwarnings("ignore")
            trainable = create_instance_from_hyperopt_search_space(self.estimator, params)
            trained = trainable.fit(X_train, y_train)
            return trained

        def f(params):
            current_time = time.time()
            if (self.max_opt_time is not None) and ((current_time - opt_start_time) > self.max_opt_time) :
                # if max optimization time set, and we have crossed it, exit optimization completely
                sys.exit(0)

            params_to_save = copy.deepcopy(params)
            return_dict = {}
            try:
                score, logloss, execution_time = hyperopt_train_test(params, X_train=X_train, y_train=y_train)
                return_dict = {
                    'loss': self.best_score - score,
                    'time': execution_time,
                    'log_loss': logloss,
                    'status': STATUS_OK,
                    'params': params_to_save
                }
            except BaseException as e:
                logger.warning(f"Exception caught in HyperoptCV:{type(e)}, {traceback.format_exc()} with hyperparams: {params}, setting status to FAIL")
                return_dict = {'status': STATUS_FAIL}
            return return_dict

        try :
            fmin(f, self.search_space, algo=tpe.suggest, max_evals=self.max_evals, trials=self.trials, rstate=np.random.RandomState(SEED))
        except SystemExit :
            logger.warning('Maximum alloted optimization time exceeded. Optimization exited prematurely')

        try :
            best_params = space_eval(self.search_space, self.trials.argmin)
            logger.info(
                'best score: {:.1%}\nbest hyperparams found using {} hyperopt trials: {}'.format(
                    self.best_score - self.trials.average_best_error(), self.max_evals, best_params
                )
            )
            trained = get_final_trained_estimator(best_params, X_train, y_train)
            self.best_estimator = trained
        except BaseException as e :
            logger.warning('Unable to extract the best parameters from optimization, the error: {}'.format(e))
            self.best_estimator = None

        return self

    def predict(self, X_eval):
        import warnings
        warnings.filterwarnings("ignore")
        trained = self.best_estimator
        try:
            predictions = trained.predict(X_eval)
        except ValueError as e:
            logger.warning("ValueError in predicting using HyperoptCV:{}, the error is:{}".format(trained, e))
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
                {   'typeForOptimizer': 'operator',
                    'not': {'enum': [None]}},
                {   'enum': [None]}],
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
            'scoring': {
                'anyOf': [
                {    'description': 'Custom scorer object, see https://scikit-learn.org/stable/modules/model_evaluation.html',
                     'not': {'type': 'string'}},
                {    'enum': [
                        'accuracy', 'explained_variance', 'max_error',
                        'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo',
                        'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted',
                        'balanced_accuracy', 'average_precision',
                        'neg_log_loss', 'neg_brier_score', 'r2', 'neg_mean_squared_error', 'neg_mean_absolute_error',
                         'neg_root_mean_squared_error', 'neg_mean_squared_log_error',
                         'neg_median_absolute_error']}],
                'default': 'accuracy'},
            'best_score': {
                'type': 'number',
                'default': 0.0},
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
                'anyOf': [
                {   'type': 'array', 'items': {'type': ['number', 'string']}},
                {   'type': 'string'}]}},
        'y': {
            'type': 'array', 'items': {'type': 'number'}}}}

_input_predict_schema = {
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'anyOf': [
                {   'type': 'array', 'items': {'type': ['number', 'string']}},
                {   'type': 'string'}]}}}}

_output_predict_schema = {
    'type': 'array', 'items': {'type': 'number'}}

_combined_schemas = {
    'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.hyperopt_cv.html',
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

HyperoptCV = lale.operators.make_operator(HyperoptCVImpl, _combined_schemas)

if __name__ == '__main__':
    from lale.lib.lale import ConcatFeatures
    from lale.lib.sklearn import Nystroem
    from lale.lib.sklearn import PCA
    pca = PCA(n_components=10)
    nys = Nystroem(n_components=10)
    concat = ConcatFeatures()
    lr = LogisticRegression(random_state=42, C=0.1)

    trainable = (pca & nys) >> concat >> lr

    import sklearn.datasets
    from lale.helpers import cross_val_score
    digits = sklearn.datasets.load_iris()
    X, y = sklearn.utils.shuffle(digits.data, digits.target, random_state=42)

    hp_n = HyperoptCV(estimator=trainable, max_evals=2)

    hp_n_trained = hp_n.fit(X, y)
    predictions = hp_n_trained.predict(X)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y, [round(pred) for pred in predictions])
    print(accuracy)
