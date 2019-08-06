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
from sklearn.metrics import r2_score, log_loss
from sklearn.model_selection import KFold
import warnings
import numpy as np
import time
import logging
from typing import Optional

SEED=42
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class HyperoptRegressor():

    def __init__(self, model = None, max_evals=50, handle_cv_failure = False, pgo:Optional[PGO]=None):
        self.max_evals = max_evals
        if model is None:
            self.model = RandomForestRegressor
        else:
            self.model = model
        self.search_space = hp.choice('meta_model', [hyperopt_search_space(self.model, pgo=pgo)])
        self.handle_cv_failure = handle_cv_failure
        self.trials = Trials()


    def fit(self, X_train, y_train):

        def hyperopt_train_test(params, X_train, y_train):
            warnings.filterwarnings("ignore")

            reg = create_instance_from_hyperopt_search_space(self.model, params)
            try:
                cv_score, logloss, execution_time = cross_val_score_track_trials(reg, X_train, y_train, cv=KFold(10), scoring = r2_score)
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

            return cv_score, logloss, execution_time

        def get_final_trained_reg(params, X_train, y_train):
            warnings.filterwarnings("ignore")
            reg = create_instance_from_hyperopt_search_space(self.model, params)
            reg = reg.fit(X_train, y_train)
            return reg

        def f(params):
            try:
                r_squared, logloss, execution_time = hyperopt_train_test(params, X_train=X_train, y_train=y_train)
            except BaseException as e:
                logger.warning("Exception caught in HyperoptClassifer:{} with hyperparams:{}, setting accuracy to zero".format(e, params))
                r_squared = 0
                execution_time = 0
                logloss = 0
            return {'loss': -r_squared, 'time': execution_time, 'log_loss': logloss, 'status': STATUS_OK}


        fmin(f, self.search_space, algo=tpe.suggest, max_evals=self.max_evals, trials=self.trials, rstate=np.random.RandomState(SEED))
        best_params = space_eval(self.search_space, self.trials.argmin)
        logger.info('best accuracy: {:.1%}\nbest hyperparams found using {} hyperopt trials: {}'.format(-1*self.trials.average_best_error(), self.max_evals, best_params))
        trained_reg = get_final_trained_reg(best_params, X_train, y_train)

        return trained_reg

    def predict(self, X_eval):
        import warnings
        warnings.filterwarnings("ignore")
        reg = self.model
        try:
            predictions = reg.predict(X_eval)
        except ValueError as e:
            logger.warning("ValueError in predicting using classifier:{}, the error is:{}".format(reg, e))
            predictions = None

        return predictions

    def get_trials(self):
        return self.trials


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

    hp_n = HyperoptRegressor(model=trainable, max_evals=20)

    hp_n_trained = hp_n.fit(X, y)
    predictions = hp_n_trained.predict(X)
    mse = r2_score(y, [round(pred) for pred in predictions])
    print(mse)
