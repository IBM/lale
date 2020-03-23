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
import warnings

import sklearn.datasets
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

from lale.search.lale_smac import get_smac_space, lale_trainable_op_from_config
from lale.lib.lale import Hyperopt
from lale.search.op2hp import hyperopt_search_space


import numpy as np
from typing import List

def f_min(op, X, y, num_folds=5):
    from sklearn import datasets
    from lale.helpers import cross_val_score
    import numpy as np

    # try:
    scores = cross_val_score(op, X, y, cv = num_folds)

    return 1-np.mean(scores)  # Minimize!
    # except BaseException as e:
    #     print(e)
    #     return 

def iris_f_min(op, num_folds=5):
    from sklearn import datasets

    iris = datasets.load_iris()
    return f_min(op, iris.data, iris.target, num_folds = num_folds)

def iris_f_min_for_folds(num_folds=5):
    return lambda op: iris_f_min(op, num_folds=num_folds)
    
from lale.search.lale_smac import lale_op_smac_tae

def iris_fmin_tae(op, num_folds=5):
    return lale_op_smac_tae(op, iris_f_min_for_folds(num_folds=num_folds))

class TestSMAC(unittest.TestCase):

    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        X, y = load_iris(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(X, y)    

    def test_smac(self):

        import numpy as np
        from sklearn import svm, datasets
        from sklearn.model_selection import cross_val_score

        # Import ConfigSpace and different types of parameters
        from smac.configspace import ConfigurationSpace

        # Import SMAC-utilities
        from smac.tae.execute_func import ExecuteTAFuncDict
        from smac.scenario.scenario import Scenario
        from smac.facade.smac_facade import SMAC as orig_SMAC

        from lale.search.lale_smac import get_smac_space

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
        tae = iris_fmin_tae(lr, num_folds=2)
        print("Optimizing! Depending on your machine, this might take a few minutes.")
        smac = orig_SMAC(scenario=scenario, rng=np.random.RandomState(42),
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
        from smac.facade.smac_facade import SMAC as orig_SMAC


        tfm = PCA() | Nystroem() | NoOp()
        planned_pipeline1 = (OneHotEncoder(handle_unknown = 'ignore',  sparse = False) | NoOp()) >> tfm >> (LogisticRegression() | KNeighborsClassifier())

        cs:ConfigurationSpace = get_smac_space(planned_pipeline1, lale_num_grids=1)

        # Scenario object
        scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                            "runcount-limit": 1,  # maximum function evaluations
                            "cs": cs,               # configuration space
                            "deterministic": "true"
                            })

        # Optimize, using a SMAC-object
        tae = iris_fmin_tae(planned_pipeline1, num_folds=2)
        print("Optimizing! Depending on your machine, this might take a few minutes.")
        smac = orig_SMAC(scenario=scenario, rng=np.random.RandomState(42),
                tae_runner=tae)

        incumbent = smac.optimize()

        inc_value = tae(incumbent)

        print("Optimized Value: %.2f" % (inc_value))

    def test_smac1(self):
        from sklearn.metrics import accuracy_score
        from lale.lib.lale import SMAC
        planned_pipeline = (PCA | NoOp) >> LogisticRegression        
        opt = SMAC(estimator=planned_pipeline, max_evals=1)
        # run optimizer
        res = opt.fit(self.X_train, self.y_train)
        predictions = res.predict(self.X_test)

    def test_smac2(self):
        from sklearn.metrics import accuracy_score
        from lale.lib.lale import SMAC
        planned_pipeline = (PCA | NoOp) >> KNeighborsClassifier(n_neighbors = 10000)
        opt = SMAC(estimator=planned_pipeline, max_evals=1)
        # run optimizer
        res = opt.fit(self.X_train, self.y_train)
        # Get the trials object and make sure that SMAC assigned cost_for_crash which is MAXINT by default to 
        #at least one trial (correspond to KNN).
        trials = res._impl.get_trials()
        assert 2147483647.0 in trials.cost_per_config.values()

    def test_smac_timeout_zero_classification(self):
        from lale.lib.lale import SMAC
        planned_pipeline = (MinMaxScaler | Normalizer) >> (LogisticRegression | KNeighborsClassifier)
        opt = SMAC(estimator=planned_pipeline, max_evals=1, max_opt_time=0.0)
        # run optimizer
        res = opt.fit(self.X_train, self.y_train)
        assert res.get_pipeline() is None

    def test_smac_timeout_zero_regression(self):
        from lale.lib.lale import SMAC
        planned_pipeline = (MinMaxScaler | Normalizer) >> LinearRegression
        from sklearn.datasets import load_boston
        X, y = load_boston(return_X_y=True)
        opt = SMAC(estimator=planned_pipeline, scoring = 'r2', max_evals=1, max_opt_time=0.0)
        # run optimizer
        res = opt.fit(X[:500,:], y[:500])
        assert res.get_pipeline() is None

    def test_smac_timeout_classification(self):
        from lale.lib.lale import SMAC
        import time
        planned_pipeline = (MinMaxScaler | Normalizer) >> (LogisticRegression | KNeighborsClassifier)
        max_opt_time = 4.0
        opt = SMAC(estimator=planned_pipeline, max_evals=1, max_opt_time=max_opt_time)

        start = time.time()
        res = opt.fit(self.X_train, self.y_train)
        end = time.time()
        opt_time = end - start
        rel_diff = (opt_time - max_opt_time) / max_opt_time
        assert rel_diff < 1.2, (
            'Max time: {}, Actual time: {}, relative diff: {}'.format(max_opt_time, opt_time, rel_diff)
        )

    def test_smac_timeout_regression(self):
        from lale.lib.lale import SMAC
        from sklearn.datasets import load_boston
        import time
        planned_pipeline = (MinMaxScaler | Normalizer) >> LinearRegression
        X, y = load_boston(return_X_y=True)
        max_opt_time = 2.0
        opt = SMAC(estimator=planned_pipeline, scoring = 'r2', max_evals=1, max_opt_time=max_opt_time)

        start = time.time()
        res = opt.fit(X[:500,:], y[:500])
        end = time.time()
        opt_time = end - start
        rel_diff = (opt_time - max_opt_time) / max_opt_time
        assert rel_diff < 0.5, (
            'Max time: {}, Actual time: {}, relative diff: {}'.format(max_opt_time, opt_time, rel_diff)
        )

def run_hyperopt_on_planned_pipeline(planned_pipeline, max_iters=1) :
    # data
    from sklearn.datasets import load_iris
    features, labels = load_iris(return_X_y=True)
    # set up optimizer
    from lale.lib.lale.hyperopt import Hyperopt
    opt = Hyperopt(estimator=planned_pipeline, max_evals=max_iters)
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

class TestHyperopt(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(X, y)    

    def test_using_scoring(self):
        from sklearn.metrics import hinge_loss, make_scorer, f1_score, accuracy_score
        lr = LogisticRegression()
        clf = Hyperopt(estimator=lr, scoring='accuracy', cv=5, max_evals=1)
        trained = clf.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)
        predictions_1 = clf.predict(self.X_test)
        assert np.array_equal(predictions_1, predictions)

    def test_custom_scoring(self):
        from sklearn.metrics import f1_score, make_scorer
        lr = LogisticRegression()
        clf = Hyperopt(estimator=lr, scoring=make_scorer(f1_score, average='macro'), cv = 5, max_evals=1)
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
        hoc = Hyperopt(
            estimator=planned_pipeline,
            max_evals=1,
            cv=3,
            scoring='accuracy',
            max_opt_time=max_opt_time
        )
        start = time.time()
        best_trained = hoc.fit(X, y)
        end = time.time()
        opt_time = end - start
        rel_diff = (opt_time - max_opt_time) / max_opt_time
        assert rel_diff < 0.5, (
            'Max time: {}, Actual time: {}, relative diff: {}'.format(max_opt_time, opt_time, rel_diff)
        )
        
    def test_runtime_limit_zero_time_hoc(self):
        planned_pipeline = (MinMaxScaler | Normalizer) >> (LogisticRegression | KNeighborsClassifier)
        from sklearn.datasets import load_iris
        X, y = load_iris(return_X_y=True)
        
        hoc = Hyperopt(
            estimator=planned_pipeline,
            max_evals=1,
            cv=3,
            scoring='accuracy',
            max_opt_time=0.0
        )
        hoc_fitted = hoc.fit(X, y)
        assert hoc_fitted.get_pipeline() is None

    def test_runtime_limit_hor(self):
        import time
        planned_pipeline = (MinMaxScaler | Normalizer) >> LinearRegression
        from sklearn.datasets import load_boston
        X, y = load_boston(return_X_y=True)
        
        max_opt_time = 3.0
        hor = Hyperopt(
            estimator=planned_pipeline,
            max_evals=1,
            cv=3,
            max_opt_time=max_opt_time,
            scoring='r2'
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
        
        hor = Hyperopt(
            estimator=planned_pipeline,
            max_evals=1,
            cv=3,
            max_opt_time=0.0,
            scoring='r2'
        )
        hor_fitted = hor.fit(X, y)
        assert hor_fitted.get_pipeline() is None

    def test_hyperparam_overriding_with_hyperopt(self):
        pca1 = PCA(n_components = 3)
        pca2 = PCA()
        search_space1 = hyperopt_search_space(pca1)
        search_space2 = hyperopt_search_space(pca2)
        self.assertNotEqual(search_space1, search_space2)

    def test_nested_pipeline1(self):
        from sklearn.datasets import load_iris
        from lale.lib.lale import Hyperopt
        from sklearn.metrics import accuracy_score
        data = load_iris()
        X, y = data.data, data.target
        #pipeline = KNeighborsClassifier() | (OneHotEncoder(handle_unknown = 'ignore') >> LogisticRegression())
        pipeline = KNeighborsClassifier() | (SimpleImputer() >> LogisticRegression())
        clf = Hyperopt(estimator=pipeline, max_evals=1)
        trained = clf.fit(X, y)
        predictions = trained.predict(X)
        print(accuracy_score(y, predictions))

    def test_with_concat_features1(self):
        import warnings
        warnings.filterwarnings("ignore")
        import logging
        logging.basicConfig(level=logging.DEBUG)

        from sklearn.datasets import load_iris
        from lale.lib.lale import Hyperopt
        from sklearn.metrics import accuracy_score
        data = load_iris()
        X, y = data.data, data.target
        pca = PCA(n_components=3)
        nys = Nystroem(n_components=10)
        concat = ConcatFeatures()
        lr = LogisticRegression(random_state=42, C=0.1)
        pipeline = ((pca & nys) >> concat >> lr) | KNeighborsClassifier()
        clf = Hyperopt(estimator=pipeline, max_evals=1)
        trained = clf.fit(X, y)
        predictions = trained.predict(X)
        print(accuracy_score(y, predictions))
        warnings.resetwarnings()

    def test_with_concat_features2(self):
        import warnings
        warnings.filterwarnings("ignore")
        import logging
        logging.basicConfig(level=logging.DEBUG)

        from sklearn.datasets import load_iris
        from lale.lib.lale import Hyperopt
        from sklearn.metrics import accuracy_score
        data = load_iris()
        X, y = data.data, data.target
        pca = PCA(n_components=3)
        nys = Nystroem(n_components=10)
        concat = ConcatFeatures()
        lr = LogisticRegression(random_state=42, C=0.1)
        from lale.operators import make_pipeline
        pipeline = make_pipeline(((((SimpleImputer() | NoOp()) >> pca) & nys) >> concat >> lr) | KNeighborsClassifier())
        clf = Hyperopt(estimator=pipeline, max_evals=1, handle_cv_failure=True)
        trained = clf.fit(X, y)
        predictions = trained.predict(X)
        print(accuracy_score(y, predictions))
        warnings.resetwarnings()

    def test_preprocessing_union(self):
        from lale.datasets import openml
        (train_X, train_y), (test_X, test_y) = openml.fetch(
            'credit-g', 'classification', preprocess=False)
        from lale.lib.lale import Project
        from lale.lib.sklearn import Normalizer, OneHotEncoder
        from lale.lib.lale import ConcatFeatures as Concat
        from lale.lib.sklearn import RandomForestClassifier as Forest
        prep_num = Project(columns={'type': 'number'}) >> Normalizer
        prep_cat = Project(columns={'not': {'type': 'number'}}) >> OneHotEncoder(sparse=False)
        planned = (prep_num & prep_cat) >> Concat >> Forest
        from lale.lib.lale import Hyperopt
        hyperopt_classifier = Hyperopt(estimator=planned, max_evals=1)
        best_found = hyperopt_classifier.fit(train_X, train_y)

    def test_text_and_structured(self):
        from lale.datasets.uci.uci_datasets import fetch_drugscom
        from sklearn.model_selection import train_test_split
        train_X_all, train_y_all, test_X, test_y = fetch_drugscom()
        #subset to speed up debugging
        train_X, train_X_ignore, train_y, train_y_ignore = train_test_split(
            train_X_all, train_y_all, train_size=0.01, random_state=42)
        from lale.lib.lale import Project
        from lale.lib.lale import ConcatFeatures as Cat
        from lale.lib.sklearn import TfidfVectorizer as Tfidf
        from lale.lib.sklearn import LinearRegression as LinReg
        from lale.lib.sklearn import RandomForestRegressor as Forest
        prep_text = Project(columns=['review']) >> Tfidf(max_features=100)
        prep_nums = Project(columns={'type': 'number'})
        planned = (prep_text & prep_nums) >> Cat >> (LinReg | Forest)
        from lale.lib.lale import Hyperopt
        hyperopt_classifier = Hyperopt(estimator=planned, max_evals=1, scoring='r2')
        best_found = hyperopt_classifier.fit(train_X, train_y)

    def test_custom_scorer(self):
        from sklearn.metrics import f1_score, make_scorer
        pipeline = PCA() >> LogisticRegression()
        def custom_scorer(estimator, X, y, factor=0.1):
            #This is a custom scorer for demonstrating the use of kwargs
            #Just applies some factor to the accuracy
            from sklearn.metrics import accuracy_score
            predictions = estimator.predict(X)
            self.assertEqual(factor, 0.5)
            return factor*accuracy_score(y, predictions)
        clf = Hyperopt(estimator=pipeline, scoring=custom_scorer, cv = 5, max_evals=1, args_to_scorer={'factor':0.5})
        trained = clf.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)
        predictions_1 = clf.predict(self.X_test)
        assert np.array_equal(predictions_1, predictions)

class TestAutoConfigureClassification(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(X, y)    

    def test_with_Hyperopt(self):
        from lale.lib.sklearn import PCA, LogisticRegression
        from lale.lib.lale import NoOp, Hyperopt

        planned_pipeline = (PCA | NoOp) >> LogisticRegression
        best_pipeline = planned_pipeline.auto_configure(self.X_train, self.y_train, optimizer = Hyperopt, cv = 3, 
            scoring='accuracy', max_evals=1)
        predictions = best_pipeline.predict(self.X_test)
        from lale.operators import TrainedPipeline
        assert isinstance(best_pipeline, TrainedPipeline)

    def test_with_Hyperopt_2(self):
        from lale.lib.sklearn import LogisticRegression as LR
        from lale.lib.sklearn import KNeighborsClassifier as KNN
        from lale.lib.lale import Hyperopt
        choice = LR | KNN
        best = choice.auto_configure(self.X_train, self.y_train,
                                     optimizer=Hyperopt, cv=3, max_evals=3)
        predictions = best.predict(self.X_test)

    def test_with_gridsearchcv(self):
        from lale.lib.sklearn import PCA, LogisticRegression
        from lale.lib.lale import NoOp, GridSearchCV
        warnings.simplefilter("ignore")
        planned_pipeline = (PCA | NoOp)  >> LogisticRegression
        best_pipeline = planned_pipeline.auto_configure(self.X_train, self.y_train, optimizer = GridSearchCV, cv = 3, 
            scoring='accuracy', lale_num_samples=1, lale_num_grids=1)
        predictions = best_pipeline.predict(self.X_test)
        assert best_pipeline is not None
        
    def test_with_smaccv(self):
        from lale.lib.sklearn import PCA, LogisticRegression
        from lale.lib.lale import NoOp, SMAC

        planned_pipeline = (PCA | NoOp) >> LogisticRegression
        best_pipeline = planned_pipeline.auto_configure(self.X_train, self.y_train, optimizer = SMAC, cv = 3, 
            scoring='accuracy', max_evals=1)
        predictions = best_pipeline.predict(self.X_test)
        from lale.operators import TrainedPipeline
        assert isinstance(best_pipeline, TrainedPipeline)

class TestAutoConfigureRegression(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_boston
        from sklearn.model_selection import train_test_split
        X, y = load_boston(return_X_y=True)
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(X, y)    

    def test_with_Hyperopt(self):
        from lale.lib.sklearn import PCA, LogisticRegression
        from lale.lib.lale import NoOp, Hyperopt

        planned_pipeline = (MinMaxScaler | Normalizer) >> LinearRegression
        best_pipeline = planned_pipeline.auto_configure(self.X_train, self.y_train, optimizer = Hyperopt, cv = 3, 
            scoring='r2', max_evals=1)
        predictions = best_pipeline.predict(self.X_test)
        from lale.operators import TrainedPipeline
        assert isinstance(best_pipeline, TrainedPipeline)

    def test_with_gridsearchcv(self):
        from lale.lib.sklearn import PCA, LogisticRegression
        from lale.lib.lale import NoOp, GridSearchCV
        warnings.simplefilter("ignore")
        planned_pipeline = (MinMaxScaler | Normalizer) >> LinearRegression
        best_pipeline = planned_pipeline.auto_configure(self.X_train, self.y_train, optimizer = GridSearchCV, cv = 3, 
            scoring='r2', lale_num_samples=1, lale_num_grids=1)
        predictions = best_pipeline.predict(self.X_test)
        assert best_pipeline is not None

class TestGridSearchCV(unittest.TestCase):
    def test_manual_grid(self):
        from lale.lib.sklearn import SVC
        from sklearn.datasets import load_iris
        from lale.lib.lale import GridSearchCV
        warnings.simplefilter("ignore")

        from lale import wrap_imported_operators
        wrap_imported_operators()
        iris = load_iris()
        parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
        svc = SVC()
        clf = GridSearchCV(estimator=svc, param_grid=parameters)
        clf.fit(iris.data, iris.target)
        clf.predict(iris.data)

    def test_with_gridsearchcv_auto_wrapped_pipe1(self):
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
  
        lr = LogisticRegression()
        pca = PCA()
        trainable = pca >> lr

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from lale.lib.lale import GridSearchCV
            clf = GridSearchCV(
                estimator=trainable, lale_num_samples=1, lale_num_grids=1,
                cv=2, scoring=make_scorer(accuracy_score))
            iris = load_iris()
            clf.fit(iris.data, iris.target)
    
    def test_with_gridsearchcv_auto_wrapped_pipe2(self):
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
  
        lr = LogisticRegression()
        pca1 = PCA()
        pca1._name = "PCA1"
        pca2 = PCA()
        pca2._name = "PCA2"
        trainable = (pca1 | pca2) >> lr

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from lale.lib.lale import GridSearchCV
            clf = GridSearchCV(
                estimator=trainable, lale_num_samples=1, lale_num_grids=1,
                cv=2, scoring=make_scorer(accuracy_score))
            iris = load_iris()
            clf.fit(iris.data, iris.target)

class TestCrossValidation(unittest.TestCase):
    def test_cv_folds(self):
        trainable_lr = LogisticRegression(n_jobs=1)
        iris = sklearn.datasets.load_iris()
        from lale.helpers import cross_val_score
        from sklearn.model_selection import KFold
        cv_results = cross_val_score(trainable_lr, iris.data, iris.target, cv = KFold(2))
        self.assertEqual(len(cv_results), 2)

    def test_cv_scoring(self):
        trainable_lr = LogisticRegression(n_jobs=1)
        iris = sklearn.datasets.load_iris()
        from lale.helpers import cross_val_score
        from sklearn.metrics import confusion_matrix
        cv_results = cross_val_score(trainable_lr, iris.data, iris.target, scoring=confusion_matrix)
        self.assertEqual(len(cv_results), 5)

    def test_cv_folds_scikit(self):
        trainable_lr = LogisticRegression(n_jobs=1)
        iris = sklearn.datasets.load_iris()
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.model_selection import KFold
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_results = cross_val_score(
                trainable_lr, iris.data, iris.target,
                cv = KFold(2), scoring=make_scorer(accuracy_score))
        self.assertEqual(len(cv_results), 2)

class TestHigherOrderOperators(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(X, y)    

    def test_ada_boost(self):
        from lale.lib.sklearn import AdaBoostClassifier, DecisionTreeClassifier
        clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier())
        trained = clf.auto_configure(self.X_train, self.y_train, optimizer=Hyperopt, max_evals=1)
        #Checking that the inner decision tree does not get the default value for min_samples_leaf, not sure if this will always pass
        self.assertNotEqual(trained.hyperparams()['base_estimator'].hyperparams()['min_samples_leaf'], 1)

    def test_ada_boost1(self):
        from lale.lib.sklearn import AdaBoostClassifier
        from sklearn.tree import DecisionTreeClassifier
        clf = AdaBoostClassifier(base_estimator = DecisionTreeClassifier())
        clf.fit(self.X_train, self.y_train)

    def test_ada_boost_regressor(self):
        from sklearn.datasets import load_boston
        from sklearn.model_selection import train_test_split
        X, y = load_boston(return_X_y=True)
        X_train, X_test, y_train, y_test =  train_test_split(X, y)    
        from lale.lib.sklearn import AdaBoostRegressor, DecisionTreeRegressor
        reg = AdaBoostRegressor(base_estimator = DecisionTreeRegressor())
        trained = reg.auto_configure(X_train, y_train, optimizer=Hyperopt, max_evals=1, scoring='r2')
        #Checking that the inner decision tree does not get the default value for min_samples_leaf, not sure if this will always pass
        self.assertNotEqual(trained.hyperparams()['base_estimator'].hyperparams()['min_samples_leaf'], 1)
