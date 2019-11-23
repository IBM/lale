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

from lale.lib.sklearn import LogisticRegression
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials, space_eval
from lale.helpers import cross_val_score_track_trials, create_instance_from_hyperopt_search_space
from lale.search.op2hp import hyperopt_search_space
from lale.search.PGO import PGO
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, make_scorer
from sklearn.metrics.scorer import check_scoring
import warnings
import numpy as np
import time
import logging
from typing import Optional
import json
import datetime
import copy
import sys

SEED=42
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class HyperoptClassifier():

    def __init__(self, model = None, max_evals=50, cv=5, handle_cv_failure = False, scoring='accuracy', max_opt_time=None, pgo:Optional[PGO]=None):
        """ Instantiate the HyperoptClassifier that will use the given model and other parameters to select the 
        best performing trainable instantiation of the model. This optimizer uses negation of accuracy_score 
        as the performance metric to be minimized by Hyperopt.

        Parameters
        ----------
        model : lale.operators.IndividualOp or lale.operators.Pipeline, optional
            A valid Lale individual operator or pipeline, by default None
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
            If False, the trial is terminated by assigning accuracy to zero.
            , by default False
        scoring: string or a scorer object created using 
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer.
            A string from sklearn.metrics.SCORERS.keys() can be used or a scorer created from one of 
            sklearn.metrics (https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics).
            A completely custom scorer object can be created from a python function following the example at 
            https://scikit-learn.org/stable/modules/model_evaluation.html
            The metric has to return a scalar value, and note that scikit-learns's scorer object always returns values such that
            higher score is better. Since Hyperopt solves a minimization problem, we negate the score value to pass to Hyperopt.
            by default 'accuracy'.
        max_opt_time : float, optional
            Maximum amout of time in seconds for the optimization. By default, None, implying no runtime
            bound.
        pgo : Optional[PGO], optional
            [description], by default None
        
        Raises
        ------
        e
            [description]

        Examples
        --------
        >>> from sklearn.metrics import make_scorer, f1_score, accuracy_score
        >>> lr = LogisticRegression()
        >>> clf = HyperoptClassifier(lr, scoring='accuracy', cv = 5, max_evals = 2)
        >>> from sklearn import datasets
        >>> diabetes = datasets.load_diabetes()
        >>> X = diabetes.data[:150]
        >>> y = diabetes.target[:150]
        >>> trained = clf.fit(X, y)
        >>> predictions = trained.predict(X)

        Other scoring metrics:

        >>> clf = HyperoptClassifier(lr, scoring=make_scorer(f1_score, average='macro'), cv = 3, max_evals = 2)

        """
        self.max_evals = max_evals
        if model is None:
            self.model = LogisticRegression
        else:
            self.model = model
        self.search_space = hp.choice('meta_model', [hyperopt_search_space(self.model, pgo=pgo)])
        self.scoring = scoring
        self.handle_cv_failure = handle_cv_failure
        self.cv = cv
        self.trials = Trials()
        self.max_opt_time = max_opt_time


    def fit(self, X_train, y_train):
        opt_start_time = time.time()

        def hyperopt_train_test(params, X_train, y_train):
            warnings.filterwarnings("ignore")

            clf = create_instance_from_hyperopt_search_space(self.model, params)
            try:
                cv_score, logloss, execution_time = cross_val_score_track_trials(clf, X_train, y_train, cv=self.cv, scoring=self.scoring)
                logger.debug("Successful trial of hyperopt")
            except BaseException as e:
                #If there is any error in cross validation, use the accuracy based on a random train-test split as the evaluation criterion
                if self.handle_cv_failure:
                    X_train_part, X_validation, y_train_part, y_validation = train_test_split(X_train, y_train, test_size=0.20)
                    start = time.time()
                    clf_trained = clf.fit(X_train_part, y_train_part)
                    #predictions = clf_trained.predict(X_validation)
                    scorer = check_scoring(clf, scoring=self.scoring)
                    cv_score  = scorer(clf_trained, X_validation, y_validation)
                    execution_time = time.time() - start
                    y_pred_proba = clf_trained.predict_proba(X_validation)
                    try:
                        logloss = log_loss(y_true=y_validation, y_pred=y_pred_proba)
                    except BaseException:
                        logloss = 0
                        logger.debug("Warning, log loss cannot be computed")
                else:
                    logger.debug(e)
                    logger.debug("Error {} with pipeline:{}".format(e, clf.to_json()))
                    raise e
            return cv_score, logloss, execution_time
        def get_final_trained_clf(params, X_train, y_train):
            warnings.filterwarnings("ignore")
            clf = create_instance_from_hyperopt_search_space(self.model, params)
            clf = clf.fit(X_train, y_train)
            return clf

        def f(params):
            current_time = time.time()
            if (self.max_opt_time is not None) and ((current_time - opt_start_time) > self.max_opt_time) :
                # if max optimization time set, and we have crossed it, exit optimization completely
                sys.exit(0)

            params_to_save = copy.deepcopy(params)
            return_dict = {}
            try:
                acc, logloss, execution_time = hyperopt_train_test(params, X_train=X_train, y_train=y_train)
                return_dict = {'loss': -acc, 'time': execution_time, 'log_loss': logloss, 'status': STATUS_OK, 'params': params_to_save}
            except BaseException as e:
                logger.warning("Exception caught in HyperoptClassifer:{}, setting status to FAIL".format(e))
                return_dict = {'status': STATUS_FAIL}
            return return_dict

        try :
            fmin(f, self.search_space, algo=tpe.suggest, max_evals=self.max_evals, trials=self.trials, rstate=np.random.RandomState(SEED))
        except SystemExit :
            logger.warning('Maximum alloted optimization time exceeded. Optimization exited prematurely')

        try :
            best_params = space_eval(self.search_space, self.trials.argmin)
            logger.info('best accuracy: {:.1%}\nbest hyperparams found using {} hyperopt trials: {}'.format(-1*self.trials.average_best_error(), self.max_evals, best_params))
            trained_clf = get_final_trained_clf(best_params, X_train, y_train)
            self.best_model = trained_clf
        except BaseException as e :
            logger.warning('Unable to extract the best parameters from optimization, the error: {}'.format(e))
            trained_clf = None

        return trained_clf

    def predict(self, X_eval):
        import warnings
        warnings.filterwarnings("ignore")
        clf = self.best_model
        try:
            predictions = clf.predict(X_eval)
        except ValueError as e:
            logger.warning("ValueError in predicting using classifier:{}, the error is:{}".format(clf, e))
            predictions = None

        return predictions

    def get_trials(self):
        return self.trials


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

    hp_n = HyperoptClassifier(model=trainable, max_evals=2)

    hp_n_trained = hp_n.fit(X, y)
    predictions = hp_n_trained.predict(X)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y, [round(pred) for pred in predictions])
    print(accuracy)
