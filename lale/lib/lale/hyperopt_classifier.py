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
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials, space_eval
from lale.helpers import cross_val_score_track_trials, create_instance_from_hyperopt_search_space
from lale.search.op2hp import hyperopt_search_space
from lale.search.PGO import PGO
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import warnings
import numpy as np
import time
import logging
from typing import Optional
import json
import datetime
import copy

SEED=42
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class HyperoptClassifier():

    def __init__(self, model = None, max_evals=50, cv=5, handle_cv_failure = False, pgo:Optional[PGO]=None):
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
            and reporting the accuracy on the validation part.
            If False, the trial is terminated by assigning accuracy to zero.
            , by default False
        pgo : Optional[PGO], optional
            [description], by default None
        
        Raises
        ------
        e
            [description]
        """
        self.max_evals = max_evals
        if model is None:
            self.model = LogisticRegression
        else:
            self.model = model
        self.search_space = hp.choice('meta_model', [hyperopt_search_space(self.model, pgo=pgo)])
        self.handle_cv_failure = handle_cv_failure
        self.cv = cv
        self.trials = Trials()


    def fit(self, X_train, y_train):

        def hyperopt_train_test(params, X_train, y_train):
            warnings.filterwarnings("ignore")

            clf = create_instance_from_hyperopt_search_space(self.model, params)
            try:
                cv_score, logloss, execution_time = cross_val_score_track_trials(clf, X_train, y_train, cv=self.cv)
                logger.debug("Successful trial of hyperopt")
            except BaseException as e:
                #If there is any error in cross validation, use the accuracy based on a random train-test split as the evaluation criterion
                if self.handle_cv_failure:
                    X_train_part, X_validation, y_train_part, y_validation = train_test_split(X_train, y_train, test_size=0.20)
                    start = time.time()
                    clf_trained = clf.fit(X_train_part, y_train_part)
                    predictions = clf_trained.predict(X_validation)
                    execution_time = time.time() - start
                    y_pred_proba = clf_trained.predict_proba(X_validation)
                    try:
                        logloss = log_loss(y_true=y_validation, y_pred=y_pred_proba)
                    except BaseException:
                        logloss = 0
                        logger.debug("Warning, log loss cannot be computed")
                    cv_score = accuracy_score(y_validation, [round(pred) for pred in predictions])
                else:
                    logger.debug(e)
                    logger.debug("Error {} with pipeline:{}".format(e, clf.to_json()))
                    raise e
            #print("TRIALS")
            #print(json.dumps(self.get_trials().trials, default = myconverter, indent=4))
            return cv_score, logloss, execution_time
        def get_final_trained_clf(params, X_train, y_train):
            warnings.filterwarnings("ignore")
            clf = create_instance_from_hyperopt_search_space(self.model, params)
            clf = clf.fit(X_train, y_train)
            return clf

        def f(params):
            params_to_save = copy.deepcopy(params)
            try:
                acc, logloss, execution_time = hyperopt_train_test(params, X_train=X_train, y_train=y_train)
            except BaseException as e:
                logger.warning("Exception caught in HyperoptClassifer:{}, setting accuracy to zero".format(e))
                acc = 0
                execution_time = 0
                logloss = 0
            return {'loss': -acc, 'time': execution_time, 'log_loss': logloss, 'status': STATUS_OK, 'params': params_to_save}


        fmin(f, self.search_space, algo=tpe.suggest, max_evals=self.max_evals, trials=self.trials, rstate=np.random.RandomState(SEED))
        best_params = space_eval(self.search_space, self.trials.argmin)
        logger.info('best accuracy: {:.1%}\nbest hyperparams found using {} hyperopt trials: {}'.format(-1*self.trials.average_best_error(), self.max_evals, best_params))
        trained_clf = get_final_trained_clf(best_params, X_train, y_train)

        return trained_clf

    def predict(self, X_eval):
        import warnings
        warnings.filterwarnings("ignore")
        clf = self.model
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
    accuracy = accuracy_score(y, [round(pred) for pred in predictions])
    print(accuracy)
