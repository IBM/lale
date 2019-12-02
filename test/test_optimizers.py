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

import unittest
import jsonschema
import warnings
from lale.lib.lale import ConcatFeatures
from lale.lib.lale import NoOp
from lale.lib.sklearn import KNeighborsClassifier
from lale.lib.sklearn import LinearSVC
from lale.lib.sklearn import LogisticRegression
from lale.lib.sklearn import LinearRegression
from lale.lib.sklearn import RandomForestRegressor
from lale.lib.sklearn import MinMaxScaler
from lale.lib.sklearn import Normalizer
from lale.lib.sklearn import MLPClassifier
from lale.lib.sklearn import Nystroem
from lale.lib.sklearn import OneHotEncoder
from lale.lib.sklearn import PCA
from lale.lib.sklearn import TfidfVectorizer
from lale.lib.sklearn import MultinomialNB
from lale.lib.sklearn import SimpleImputer
from lale.lib.sklearn import SVC
from lale.lib.xgboost import XGBClassifier
from lale.lib.sklearn import PassiveAggressiveClassifier
from lale.lib.sklearn import StandardScaler
from lale.lib.sklearn import FeatureAgglomeration

from lale.search.SMAC import get_smac_space, lale_trainable_op_from_config
from lale.lib.lale import HyperoptClassifier, HyperoptRegressor

import numpy as np
from typing import List

def test_f_min(op, X, y, num_folds=5):
    from sklearn import datasets
    from lale.helpers import cross_val_score
    import numpy as np

    # try:
    scores = cross_val_score(op, X, y, cv = num_folds)

    return 1-np.mean(scores)  # Minimize!
    # except BaseException as e:
    #     print(e)
    #     return 

def test_iris_f_min(op, num_folds=5):
    from sklearn import datasets

    iris = datasets.load_iris()
    return test_f_min(op, iris.data, iris.target, num_folds = num_folds)

def test_iris_f_min_for_folds(num_folds=5):
    return lambda op: test_iris_f_min(op, num_folds=num_folds)
    
from lale.search.SMAC import lale_op_smac_tae

def test_iris_fmin_tae(op, num_folds=5):
    return lale_op_smac_tae(op, test_iris_f_min_for_folds(num_folds=num_folds))

class TestSMAC(unittest.TestCase):
    def test_smac(self):

        import numpy as np
        from sklearn import svm, datasets
        from sklearn.model_selection import cross_val_score

        # Import ConfigSpace and different types of parameters
        from smac.configspace import ConfigurationSpace

        # Import SMAC-utilities
        from smac.tae.execute_func import ExecuteTAFuncDict
        from smac.scenario.scenario import Scenario
        from smac.facade.smac_facade import SMAC

        from lale.search.SMAC import get_smac_space

        lr = LogisticRegression()

        cs:ConfigurationSpace = get_smac_space(lr)

        # Scenario object
        scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                            "runcount-limit": 1,  # maximum function evaluations
                            "cs": cs,               # configuration space
                            "deterministic": "true",
                            "abort_on_first_run_crash": False
                            })

        # Optimize, using a SMAC-object
        tae = test_iris_fmin_tae(lr, num_folds=2)
        print("Optimizing! Depending on your machine, this might take a few minutes.")
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
                tae_runner=tae)

        incumbent = smac.optimize()

        inc_value = tae(incumbent)

        print("Optimized Value: %.2f" % (inc_value))

    def dont_test_smac_choice(self):

        import numpy as np
        from sklearn import svm, datasets
        from sklearn.model_selection import cross_val_score

        # Import ConfigSpace and different types of parameters
        from smac.configspace import ConfigurationSpace

        # Import SMAC-utilities
        from smac.tae.execute_func import ExecuteTAFuncDict
        from smac.scenario.scenario import Scenario
        from smac.facade.smac_facade import SMAC


        tfm = PCA() | Nystroem() | NoOp()
        planned_pipeline1 = (OneHotEncoder(handle_unknown = 'ignore',  sparse = False) | NoOp()) >> tfm >> (LogisticRegression() | KNeighborsClassifier())

        cs:ConfigurationSpace = get_smac_space(planned_pipeline1, lale_num_grids=5)

        # Scenario object
        scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                            "runcount-limit": 1,  # maximum function evaluations
                            "cs": cs,               # configuration space
                            "deterministic": "true"
                            })

        # Optimize, using a SMAC-object
        tae = test_iris_fmin_tae(planned_pipeline1, num_folds=2)
        print("Optimizing! Depending on your machine, this might take a few minutes.")
        smac = SMAC(scenario=scenario, rng=np.random.RandomState(42),
                tae_runner=tae)

        incumbent = smac.optimize()

        inc_value = tae(incumbent)

        print("Optimized Value: %.2f" % (inc_value))
        
def run_hyperopt_on_planned_pipeline(planned_pipeline, max_iters=1) :
    # data
    from sklearn.datasets import load_iris
    features, labels = load_iris(return_X_y=True)
    # set up optimizer
    from lale.lib.lale.hyperopt_classifier import HyperoptClassifier
    opt = HyperoptClassifier(model=planned_pipeline, max_evals=max_iters)
    # run optimizer
    res = opt.fit(features, labels)    

class TestHyperoptOperatorDuplication(unittest.TestCase) :
    def test_planned_pipeline_1(self) :
        plan = (
            ( PCA & ( MinMaxScaler | Normalizer ) ) >> ConcatFeatures() >>
            ( MinMaxScaler | Normalizer ) >>
            ( LogisticRegression | KNeighborsClassifier)
        )
        run_hyperopt_on_planned_pipeline(plan)

    def test_planned_pipeline_2(self) :
        plan = (
            ( MinMaxScaler() & NoOp() ) >> ConcatFeatures() >>
            ( NoOp() & MinMaxScaler() ) >> ConcatFeatures() >>
            ( LogisticRegression | KNeighborsClassifier )
        )
        run_hyperopt_on_planned_pipeline(plan)

    def test_planned_pipeline_3(self) :
        plan = (
            ( MinMaxScaler() & NoOp() ) >> ConcatFeatures() >>
            ( StandardScaler & ( NoOp() | MinMaxScaler() ) ) >> ConcatFeatures() >>
            ( LogisticRegression | KNeighborsClassifier )
        )
        run_hyperopt_on_planned_pipeline(plan)

class TestHyperoptClassifier(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(X, y)    

    def test_using_scoring(self):
        from sklearn.metrics import hinge_loss, make_scorer, f1_score, accuracy_score
        lr = LogisticRegression()
        clf = HyperoptClassifier(lr, scoring='accuracy', cv = 5, max_evals = 2)
        trained = clf.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)
        predictions_1 = clf.predict(self.X_test)
        assert np.array_equal(predictions_1, predictions)

    def test_custom_scoring(self):
        from sklearn.metrics import f1_score, make_scorer
        lr = LogisticRegression()
        clf = HyperoptClassifier(lr, scoring=make_scorer(f1_score, average='macro'), cv = 5, max_evals=2)
        trained = clf.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)
        predictions_1 = clf.predict(self.X_test)
        assert np.array_equal(predictions_1, predictions)

    def test_runtime_limit_hoc(self):
        import time
        planned_pipeline = (MinMaxScaler | Normalizer) >> (LogisticRegression | KNeighborsClassifier)
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True)
        
        max_opt_time = 2.0
        hoc = HyperoptClassifier(
            model=planned_pipeline,
            max_evals=100,
            cv=3,
            scoring='accuracy',
            max_opt_time=max_opt_time
        )
        start = time.time()
        best_trained = hoc.fit(X, y)
        end = time.time()
        opt_time = end - start
        rel_diff = (opt_time - max_opt_time) / max_opt_time
        assert rel_diff < 0.2, (
            'Max time: {}, Actual time: {}, relative diff: {}'.format(max_opt_time, opt_time, rel_diff)
        )
        
    def test_runtime_limit_zero_time_hoc(self):
        planned_pipeline = (MinMaxScaler | Normalizer) >> (LogisticRegression | KNeighborsClassifier)
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True)
        
        hoc = HyperoptClassifier(
            model=planned_pipeline,
            max_evals=100,
            cv=3,
            scoring='accuracy',
            max_opt_time=0.0
        )
        best_trained = hoc.fit(X, y)
        assert best_trained is None

    def test_runtime_limit_hor(self):
        import time
        planned_pipeline = (MinMaxScaler | Normalizer) >> LinearRegression
        from sklearn.datasets import load_boston
        X, y = load_boston(return_X_y=True)
        
        max_opt_time = 3.0
        hor = HyperoptRegressor(
            estimator=planned_pipeline,
            max_evals=100,
            cv=3,
            max_opt_time=max_opt_time
        )
        start = time.time()
        best_trained = hor.fit(X[:500,:], y[:500])
        end = time.time()
        opt_time = end - start
        rel_diff = (opt_time - max_opt_time) / max_opt_time
        assert rel_diff < 0.2, (
            'Max time: {}, Actual time: {}, relative diff: {}'.format(max_opt_time, opt_time, rel_diff)
        )
        
    def test_runtime_limit_zero_time_hor(self):
        planned_pipeline = (MinMaxScaler | Normalizer) >> LinearRegression
        from sklearn.datasets import load_boston
        X, y = load_boston(return_X_y=True)
        
        hor = HyperoptRegressor(
            estimator=planned_pipeline,
            max_evals=100,
            cv=3,
            max_opt_time=0.0
        )
        best_trained = hor.fit(X, y)
        assert best_trained._impl is None

