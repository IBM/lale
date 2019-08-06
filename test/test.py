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
import random
import jsonschema
import sys
import lale.operators as Ops
from lale.lib.lale import ConcatFeatures
from lale.lib.lale import NoOp
from lale.lib.sklearn import KNeighborsClassifier
from lale.lib.sklearn import LinearSVC
from lale.lib.sklearn import LogisticRegression
from lale.lib.sklearn import MinMaxScaler
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
from typing import List

import sklearn.datasets

from lale.sklearn_compat import make_sklearn_compat
from lale.search.GridSearchCV import LaleGridSearchCV, get_grid_search_parameter_grids
from lale.search.SMAC import get_smac_space, lale_trainable_op_from_config
from lale.search.op2hp import hyperopt_search_space


class TestHyperparamRanges(unittest.TestCase):
    def validate_get_param_ranges(self, operator):
        # there are ranges for exactly the relevantToOptimizer properties
        def sorted(l):
            l_copy = [*l]
            l_copy.sort()
            return l_copy
        ranges, cat_idx = operator.get_param_ranges()
        keys1 = ranges.keys()
        keys2 = operator.hyperparam_schema()['allOf'][0]['relevantToOptimizer']
        self.assertEqual(sorted(keys1), sorted(keys2))
        # all defaults are in-range
        hp_defaults = operator.hyperparam_defaults()
        for hp, r in ranges.items():
            if type(r) == tuple:
                minimum, maximum, default = r
                if minimum != None and maximum != None and default != None:
                    assert minimum <= default and default <= maximum
            else:
                minimum, maximum, default = cat_idx[hp]
                assert minimum == 0 and len(r) - 1 == maximum
    def test_get_param_ranges(self):
        for op in [ConcatFeatures, KNeighborsClassifier, LogisticRegression,
                   MLPClassifier, Nystroem, OneHotEncoder, PCA]:
            self.validate_get_param_ranges(op)

class TestCrossValidation(unittest.TestCase):
    def test_cv_classification(self):
        trainable_lr = LogisticRegression(n_jobs=1)
        iris = sklearn.datasets.load_iris()
        from lale.helpers import cross_val_score
        num_folds=5
        cv_results = cross_val_score(trainable_lr, iris.data, iris.target, cv = num_folds)
        self.assertEqual(len(cv_results), num_folds)

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



class TestGetAvailableOps(unittest.TestCase):
    def test_estimators(self):
        ops = Ops.get_available_estimators()
        ops_names = [op.name() for op in ops]
        self.assertIn('LogisticRegression', ops_names)
        self.assertIn('MLPClassifier', ops_names)
        self.assertNotIn('PCA', ops_names)
    def test_interpretable_estimators(self):
        ops = Ops.get_available_estimators({'interpretable'})
        ops_names = [op.name() for op in ops]
        self.assertIn('KNeighborsClassifier', ops_names)
        self.assertNotIn('MLPClassifier', ops_names)
        self.assertNotIn('PCA', ops_names)
    def test_transformers(self):
        ops = Ops.get_available_transformers()
        ops_names = [op.name() for op in ops]
        self.assertIn('PCA', ops_names)
        self.assertNotIn('LogisticRegression', ops_names)
        self.assertNotIn('MLPClassifier', ops_names)


class TestKNeighborsClassifier(unittest.TestCase):
    def test_with_multioutput_targets(self):
        from sklearn.datasets import make_classification, load_iris
        import numpy as np
        from sklearn.utils import shuffle

        X, y1 = make_classification(n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1)
        y2 = shuffle(y1, random_state=1)
        y3 = shuffle(y1, random_state=2)
        Y = np.vstack((y1, y2, y3)).T
        trainable = KNeighborsClassifier()
        trained = trainable.fit(X, Y)
        predictions = trained.predict(X)
    def test_predict_proba(self):
        trainable = KNeighborsClassifier()
        iris = sklearn.datasets.load_iris()
        trained = trainable.fit(iris.data, iris.target)
        with self.assertWarns(DeprecationWarning):
            predicted = trainable.predict_proba(iris.data)
        predicted = trained.predict_proba(iris.data)

class TestLogisticRegression(unittest.TestCase):
    def test_hyperparam_keyword_enum(self):
        lr = LogisticRegression(LogisticRegression.penalty.l1, C=0.1, solver=LogisticRegression.solver.saga)
    def test_hyperparam_exclusive_min(self):
        with self.assertRaises(jsonschema.ValidationError):
            lr = LogisticRegression(LogisticRegression.penalty.l1, C=0.0)
    def test_hyperparam_penalty_solver_dependence(self):
        with self.assertRaises(jsonschema.ValidationError):
            lr = LogisticRegression(LogisticRegression.penalty.l1, LogisticRegression.solver.newton_cg)
    def test_hyperparam_dual_penalty_solver_dependence(self):
        with self.assertRaises(jsonschema.ValidationError):
            lr = LogisticRegression(LogisticRegression.penalty.l2, LogisticRegression.solver.sag, dual=True)
    def test_sample_weight(self):
        import numpy as np
        trainable_lr = LogisticRegression(n_jobs=1)
        iris = sklearn.datasets.load_iris()
        trained_lr = trainable_lr.fit(iris.data, iris.target, sample_weight = np.arange(len(iris.target)))
        predicted = trained_lr.predict(iris.data)
    def test_predict_proba(self):
        import numpy as np
        trainable_lr = LogisticRegression(n_jobs=1)
        iris = sklearn.datasets.load_iris()
        trained_lr = trainable_lr.fit(iris.data, iris.target, sample_weight = np.arange(len(iris.target)))
        with self.assertWarns(DeprecationWarning):
            predicted = trainable_lr.predict_proba(iris.data)
        predicted = trained_lr.predict_proba(iris.data)
    def test_clone_with_scikit(self):
        lr = LogisticRegression()
        lr.get_params()
        from sklearn.base import clone
        lr_clone = clone(lr)
        self.assertNotEqual(lr, lr_clone)
        self.assertNotEqual(lr._impl, lr_clone._impl)
        iris = sklearn.datasets.load_iris()
        trained_lr = lr.fit(iris.data, iris.target)
        predicted = trained_lr.predict(iris.data)
        cloned_trained_lr = clone(trained_lr)
        self.assertNotEqual(trained_lr._impl, cloned_trained_lr._impl)
        predicted_clone = cloned_trained_lr.predict(iris.data)
        for i in range(len(iris.target)):
            self.assertEqual(predicted[i], predicted_clone[i])
        # Testing clone with pipelines having OperatorChoice
    def test_clone_operator_choice(self):
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.base import clone
        from sklearn.datasets import load_iris
        iris = load_iris()
        X, y = iris.data, iris.target

        lr = LogisticRegression()
        trainable = PCA() >> lr 
        trainable_wrapper = make_sklearn_compat(trainable)
        trainable2 = clone(trainable_wrapper)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cross_val_score(trainable_wrapper, X, y,
                                     scoring=make_scorer(accuracy_score), cv=2)
            result2 = cross_val_score(trainable2, X, y,
                                      scoring=make_scorer(accuracy_score), cv=2)
        for i in range(len(result)):
            self.assertEqual(result[i], result2[i])

    def test_with_gridsearchcv(self):
        from sklearn.model_selection import GridSearchCV
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
        lr = LogisticRegression()
        parameters = {'solver':('liblinear', 'lbfgs'), 'penalty':['l2']}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = GridSearchCV(lr, parameters, cv=5,
                               scoring=make_scorer(accuracy_score))
            iris = load_iris()
            clf.fit(iris.data, iris.target)

    def test_with_gridsearchcv_auto(self):
        from sklearn.model_selection import GridSearchCV
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
        lr = LogisticRegression()
        parameters = get_grid_search_parameter_grids(lr,num_samples=2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = GridSearchCV(lr, parameters, cv=5,
                               scoring=make_scorer(accuracy_score))
            iris = load_iris()
            clf.fit(iris.data, iris.target)

    def test_with_gridsearchcv_auto_wrapped_pipe1(self):
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
  
        lr = LogisticRegression()
        pca = PCA()
        trainable = pca >> lr

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LaleGridSearchCV(trainable, lale_num_samples=2, lale_num_grids=3, cv=5,
                               scoring=make_scorer(accuracy_score))
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
            clf = LaleGridSearchCV(trainable, lale_num_samples=1, lale_num_grids=3, cv=5,
                               scoring=make_scorer(accuracy_score))
            iris = load_iris()
            clf.fit(iris.data, iris.target)

    def test_with_randomizedsearchcv(self):
        from sklearn.model_selection import RandomizedSearchCV
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
        from scipy.stats.distributions import uniform
        import numpy as np
        lr = LogisticRegression()
        parameters = {'solver':('liblinear', 'lbfgs'), 'penalty':['l2']}
        ranges, cat_idx = lr.get_param_ranges()
        min_C, max_C, default_C = ranges['C']
        # specify parameters and distributions to sample from
        #the loguniform distribution needs to be taken care of properly
        param_dist = {"solver": ranges['solver'],
                      "C": uniform(min_C, np.log(max_C))}
        # run randomized search
        n_iter_search = 5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            random_search = RandomizedSearchCV(
                lr, param_distributions=param_dist, n_iter=n_iter_search, cv=5,
                scoring=make_scorer(accuracy_score))
            iris = load_iris()
            random_search.fit(iris.data, iris.target)
    def test_clone_of_trained(self):
        from sklearn.base import clone
        lr = LogisticRegression()
        from sklearn.datasets import load_iris
        iris = load_iris()
        X, y = iris.data, iris.target
        trained = lr.fit(X, y)
        trained2 = clone(trained)
    def test_grid_search_on_trained(self):
        from sklearn.model_selection import GridSearchCV
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.base import clone
        iris = load_iris()
        X, y = iris.data, iris.target
        lr = LogisticRegression()
        trained = lr.fit(X, y)
        trained2 = clone(trained)
        trained3 = trained2.fit(X, y)
        #trained2.predict(X)
        parameters = {'solver':('liblinear', 'lbfgs'), 'penalty':['l2']}

        clf = GridSearchCV(trained, parameters, cv=5, scoring=make_scorer(accuracy_score))
        #clf.fit(X, y)
    def test_grid_search_on_trained_auto(self):
        from sklearn.model_selection import GridSearchCV
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.base import clone
        iris = load_iris()
        X, y = iris.data, iris.target
        lr = LogisticRegression()
        trained = lr.fit(X, y)
        trained2 = clone(trained)
        trained3 = trained2.fit(X, y)
        #trained2.predict(X)
        parameters = get_grid_search_parameter_grids(lr, num_samples=2)

        clf = GridSearchCV(trained, parameters, cv=5, scoring=make_scorer(accuracy_score))
        #clf.fit(X, y)
    def test_doc(self):
        import sklearn.datasets
        import sklearn.utils
        from test.test_custom_operators import MyLR
        iris = sklearn.datasets.load_iris()
        X_all, y_all = sklearn.utils.shuffle(iris.data, iris.target, random_state=42)
        X_train, y_train = X_all[10:], y_all[10:]
        X_test, y_test = X_all[:10], y_all[:10]
        print('expected {}'.format(y_test))
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        trainable = MyLR(solver = 'lbfgs', C=0.1)
        trained = trainable.fit(X_train, y_train)
        predictions = trained.predict(X_test)
        print('actual {}'.format(predictions))

class TestMetaModel(unittest.TestCase):
    def test_make_pipeline(self):
        from lale.operators import make_pipeline
        tfm = PCA(n_components=10)
        clf = LogisticRegression(random_state=42)
        trainable = make_pipeline(tfm, clf)
        digits = sklearn.datasets.load_digits()
        trained = trainable.fit(digits.data, digits.target)
        predicted = trained.predict(digits.data)
    def test_compose2(self):
        from lale.operators import make_pipeline
        tfm = PCA(n_components=10)
        clf = LogisticRegression(random_state=42)
        trainable = tfm >> clf
        digits = sklearn.datasets.load_digits()
        trained = trainable.fit(digits.data, digits.target)
        predicted = trained.predict(digits.data)
    def test_compose3(self):
        from lale.operators import make_pipeline
        nys = Nystroem(n_components=15)
        pca = PCA(n_components=10)
        lr = LogisticRegression(random_state=42)
        trainable = nys >> pca >> lr
        digits = sklearn.datasets.load_digits()
        trained = trainable.fit(digits.data, digits.target)
        predicted = trained.predict(digits.data)
    def test_pca_nys_lr(self):
        from lale.operators import make_union
        nys = Nystroem(n_components=15)
        pca = PCA(n_components=10)
        lr = LogisticRegression(random_state=42)
        trainable = make_union(nys, pca) >> lr
        digits = sklearn.datasets.load_digits()
        trained = trainable.fit(digits.data, digits.target)
        predicted = trained.predict(digits.data)
    def test_compose4(self):
        from lale.operators import make_choice
        digits = sklearn.datasets.load_digits()
        ohe = OneHotEncoder(handle_unknown=OneHotEncoder.handle_unknown.ignore)
        ohe.get_params()
        no_op = NoOp()
        pca = PCA()
        nys = Nystroem()
        lr = LogisticRegression()
        knn = KNeighborsClassifier()
        step1 = ohe | no_op
        step2 = pca | nys
        step3 = lr | knn
        model_plan = step1 >> step2 >> step3
        #TODO: optimize on this plan and then fit and predict
    def test_compose5(self):
        ohe = OneHotEncoder(handle_unknown=OneHotEncoder.handle_unknown.ignore)
        digits = sklearn.datasets.load_digits()
        lr = LogisticRegression()
        lr_trained = lr.fit(digits.data, digits.target)
        lr_trained.predict(digits.data)
        pipeline1 = ohe >> lr
        pipeline1_trained = pipeline1.fit(digits.data, digits.target)
        pipeline1_trained.predict(digits.data)

    def test_concat_with_hyperopt(self):
        from lale.lib.lale import HyperoptClassifier
        pca = PCA(n_components=3)
        nys = Nystroem(n_components=10)
        concat = ConcatFeatures()
        lr = LogisticRegression(random_state=42, C=0.1)

        trainable = (pca & nys) >> concat >> lr
        clf = HyperoptClassifier(trainable, max_evals = 2)
        from sklearn.datasets import load_iris
        iris_data = load_iris()
        clf.fit(iris_data.data, iris_data.target)
        clf.predict(iris_data.data)

    def test_concat_with_hyperopt2(self):
        from lale.operators import make_pipeline, make_union
        from lale.lib.lale import HyperoptClassifier
        pca = PCA(n_components=3)
        nys = Nystroem(n_components=10)
        concat = ConcatFeatures()
        lr = LogisticRegression(random_state=42, C=0.1)

        trainable = make_pipeline(make_union(pca, nys), lr)
        clf = HyperoptClassifier(trainable, max_evals = 2)
        from sklearn.datasets import load_iris
        iris_data = load_iris()
        clf.fit(iris_data.data, iris_data.target)
        clf.predict(iris_data.data)

    def test_clone_with_scikit(self):
        lr = LogisticRegression()
        from sklearn.model_selection import cross_val_score
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.datasets import load_iris
        pca = PCA()
        trainable = pca >> lr
        from sklearn.base import clone
        iris = load_iris()
        X, y = iris.data, iris.target
        trainable2 = clone(trainable)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cross_val_score(trainable, X, y,
                                     scoring=make_scorer(accuracy_score), cv=2)
            result2 = cross_val_score(trainable2, X, y,
                                      scoring=make_scorer(accuracy_score), cv=2)
        for i in range(len(result)):
            self.assertEqual(result[i], result2[i])
        # Testing clone with nested linear pipelines
        trainable = PCA() >> trainable
        trainable2 = clone(trainable)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cross_val_score(trainable, X, y,
                                     scoring=make_scorer(accuracy_score), cv=2)
            result2 = cross_val_score(trainable2, X, y,
                                      scoring=make_scorer(accuracy_score), cv=2)
        for i in range(len(result)):
            self.assertEqual(result[i], result2[i])

    def test_with_voting_classifier1(self):
        lr = LogisticRegression()
        pca = PCA()
        from sklearn.ensemble import VotingClassifier
        vclf = VotingClassifier(estimators = [('lr', lr), ('pca', pca)])
        from sklearn.datasets import load_iris
        iris = load_iris()
        X, y = iris.data, iris.target
        vclf.fit(X, y)

    def test_with_voting_classifier2(self):
        lr = LogisticRegression()
        pca = PCA()
        trainable = pca >> lr

        from sklearn.ensemble import VotingClassifier
        vclf = VotingClassifier(estimators = [('lr', lr), ('pipe', trainable)])
        from sklearn.datasets import load_iris
        iris = load_iris()
        X, y = iris.data, iris.target
        vclf.fit(X, y)

class TestMLPClassifier(unittest.TestCase):
    def test_with_multioutput_targets(self):
        from sklearn.datasets import make_classification, load_iris
        import numpy as np
        from sklearn.utils import shuffle

        X, y1 = make_classification(n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1)
        y2 = shuffle(y1, random_state=1)
        y3 = shuffle(y1, random_state=2)
        Y = np.vstack((y1, y2, y3)).T
        trainable = KNeighborsClassifier()
        trained = trainable.fit(X, Y)
        predictions = trained.predict(X)
    def test_predict_proba(self):
        trainable = MLPClassifier()
        iris = sklearn.datasets.load_iris()
        trained = trainable.fit(iris.data, iris.target)
        with self.assertWarns(DeprecationWarning):
            predicted = trainable.predict_proba(iris.data)
        predicted = trained.predict_proba(iris.data)

class TestPCA(unittest.TestCase):
    def test_hyperparam_overriding_with_hyperopt(self):
        pca1 = PCA(n_components = 3)
        pca2 = PCA()
        search_space1 = hyperopt_search_space(pca1)
        search_space2 = hyperopt_search_space(pca2)
        self.assertNotEqual(search_space1, search_space2)

class TestToJson(unittest.TestCase):
    def test_with_operator_choice(self):
        from lale.operators import make_union, make_choice, make_pipeline
        kernel_tfm_or_not =  NoOp | Nystroem
        tfm = PCA
        clf = make_choice(LogisticRegression, KNeighborsClassifier)
        clf.to_json()
        optimizable = kernel_tfm_or_not >> tfm >> clf
        optimizable.to_json()

class TestToGraphviz(unittest.TestCase):
    def test_with_operator_choice(self):
        from lale.operators import make_union, make_choice, make_pipeline
        from lale.helpers import to_graphviz
        kernel_tfm_or_not =  NoOp | Nystroem
        tfm = PCA
        clf = make_choice(LogisticRegression, KNeighborsClassifier)
        to_graphviz(clf)
        optimizable = kernel_tfm_or_not >> tfm >> clf
        to_graphviz(optimizable)

    def test_invalid_input(self):
        from sklearn.linear_model import LogisticRegression as SklearnLR
        scikit_lr = SklearnLR()
        from lale.helpers import to_graphviz
        with self.assertRaises(ValueError):
            to_graphviz(scikit_lr)

class TestOperatorChoice(unittest.TestCase):
    def test_make_choice_with_instance(self):
        from lale.operators import make_union, make_choice, make_pipeline
        from sklearn.datasets import load_iris
        iris = load_iris()
        X, y = iris.data, iris.target
        tfm = PCA() | Nystroem() | NoOp()
        with self.assertRaises(AttributeError):
            trained = tfm.fit(X, y)
        planned_pipeline1 = (OneHotEncoder | NoOp) >> tfm >> (LogisticRegression | KNeighborsClassifier)
        planned_pipeline2 = (OneHotEncoder | NoOp) >> (PCA | Nystroem) >> (LogisticRegression | KNeighborsClassifier)
        planned_pipeline3 = make_choice(OneHotEncoder, NoOp) >> make_choice(PCA, Nystroem) >> make_choice(LogisticRegression, KNeighborsClassifier)

class TestPipeline(unittest.TestCase):
    def test_new_pipeline(self):
        from lale.operators import make_pipeline
        tfm = PCA()
        clf = LogisticRegression(LogisticRegression.solver.lbfgs, LogisticRegression.multi_class.auto)
        trainable = make_pipeline(tfm, clf)
        digits = sklearn.datasets.load_digits()
        trained = trainable.fit(digits.data, digits.target)
        predicted = trained.predict(digits.data)
        from sklearn.pipeline import make_pipeline as scikit_make_pipeline
        from sklearn.decomposition import PCA as SklearnPCA
        from sklearn.linear_model import LogisticRegression as SklearnLR
        sklearn_pipeline = scikit_make_pipeline(SklearnPCA(), SklearnLR(solver="lbfgs", multi_class="auto"))
        sklearn_pipeline.fit(digits.data, digits.target)
        predicted_sklearn = sklearn_pipeline.predict(digits.data)

        from sklearn.metrics import accuracy_score
        lale_score = accuracy_score(digits.target, predicted)
        scikit_score = accuracy_score(digits.target, predicted_sklearn)
        self.assertEqual(lale_score, scikit_score)

    def test_two_estimators(self):
        tfm = PCA()
        clf = LogisticRegression(LogisticRegression.solver.lbfgs, LogisticRegression.multi_class.auto)
        clf1 = LogisticRegression(LogisticRegression.solver.lbfgs, LogisticRegression.multi_class.auto)
        trainable = (tfm & clf) >> ConcatFeatures() >> clf1
        digits = sklearn.datasets.load_digits()
        trained = trainable.fit(digits.data, digits.target)
        predicted = trained.predict(digits.data)

    def test_two_transformers(self):
        tfm1 = PCA()
        tfm2 = Nystroem()
        trainable = tfm1 >> tfm2
        digits = sklearn.datasets.load_digits()
        trained = trainable.fit(digits.data, digits.target)
        predicted = trained.transform(digits.data)

    def test_predict_proba(self):
        from lale.operators import make_pipeline
        tfm = PCA()
        clf = LogisticRegression(LogisticRegression.solver.lbfgs, LogisticRegression.multi_class.auto)
        trainable = make_pipeline(tfm, clf)
        digits = sklearn.datasets.load_digits()
        trained = trainable.fit(digits.data, digits.target)
        predicted = trained.predict_proba(digits.data)

    def test_duplicate_instances(self):
        from lale.operators import make_pipeline
        tfm = PCA()
        clf = LogisticRegression(LogisticRegression.solver.lbfgs, LogisticRegression.multi_class.auto)
        with self.assertRaises(ValueError):
            trainable = make_pipeline(tfm, tfm, clf)

    def test_with_gridsearchcv2_auto(self):
        from sklearn.model_selection import GridSearchCV
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
        lr = LogisticRegression(random_state = 42)
        pca = PCA(random_state = 42, svd_solver = 'arpack')
        trainable = pca >> lr
        from sklearn.pipeline import Pipeline
        scikit_pipeline = Pipeline([(pca.name(), PCA(random_state = 42, svd_solver = 'arpack')), (lr.name(), LogisticRegression(random_state = 42))])
        all_parameters = get_grid_search_parameter_grids(trainable, num_samples=1)
        # otherwise the test takes too long
        parameters = random.sample(all_parameters, 10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = GridSearchCV(scikit_pipeline, parameters, cv=5, scoring=make_scorer(accuracy_score))
            iris = load_iris()
            clf.fit(iris.data, iris.target)
            predicted = clf.predict(iris.data)
            accuracy_with_lale_operators = accuracy_score(iris.target, predicted)

        from sklearn.pipeline import Pipeline
        from sklearn.decomposition import PCA as SklearnPCA
        from sklearn.linear_model import LogisticRegression as SklearnLR
        scikit_pipeline = Pipeline([(pca.name(), SklearnPCA(random_state = 42, svd_solver = 'arpack')), (lr.name(), SklearnLR(random_state = 42))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = GridSearchCV(scikit_pipeline, parameters, cv=5, scoring=make_scorer(accuracy_score))
            iris = load_iris()
            clf.fit(iris.data, iris.target)
            predicted = clf.predict(iris.data)
            accuracy_with_scikit_operators = accuracy_score(iris.target, predicted)
        self.assertEqual(accuracy_with_lale_operators, accuracy_with_scikit_operators)

    def test_with_gridsearchcv3(self):
        from sklearn.model_selection import GridSearchCV
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
        lr = LogisticRegression()
        from sklearn.pipeline import Pipeline
        scikit_pipeline = Pipeline([("nystroem", Nystroem()), ("lr", LogisticRegression())])
        parameters = {'lr__solver':('liblinear', 'lbfgs'), 'lr__penalty':['l2']}
        clf = GridSearchCV(scikit_pipeline, parameters, cv=5, scoring=make_scorer(accuracy_score))
        iris = load_iris()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(iris.data, iris.target)
        predicted = clf.predict(iris.data)

    def test_with_gridsearchcv3_auto(self):
        from sklearn.model_selection import GridSearchCV
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
        lr = LogisticRegression()
        from sklearn.pipeline import Pipeline
        scikit_pipeline = Pipeline([(Nystroem().name(), Nystroem()), (lr.name(), LogisticRegression())])
        all_parameters = get_grid_search_parameter_grids(Nystroem()>>lr, num_samples=1)
        # otherwise the test takes too long
        parameters = random.sample(all_parameters, 10)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            clf = GridSearchCV(scikit_pipeline, parameters, cv=3, scoring=make_scorer(accuracy_score))
            iris = load_iris()
            clf.fit(iris.data, iris.target)
            predicted = clf.predict(iris.data)

    def test_with_gridsearchcv3_auto_wrapped(self):
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer

        pipeline = Nystroem() >> LogisticRegression()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            clf = LaleGridSearchCV(pipeline, lale_num_samples=1, lale_num_grids=10, cv=3, scoring=make_scorer(accuracy_score))
            iris = load_iris()
            clf.fit(iris.data, iris.target)
            predicted = clf.predict(iris.data)

    # def test_with_gridsearchcv_choice(self):
    #     from sklearn.datasets import load_iris
    #     from sklearn.metrics import accuracy_score, make_scorer
    #     iris = load_iris()
    #     X, y = iris.data, iris.target
    #     tfm = PCA() | Nystroem() | NoOp()
    #     planned_pipeline1 = (OneHotEncoder(handle_unknown = 'ignore',  sparse = False) | NoOp()) >> tfm >> (LogisticRegression() | KNeighborsClassifier())

        # with warnings.catch_warnings():
        #     warnings.simplefilter("ignore")

        #     param_search = LaleGridSearchCV(planned_pipeline1, lale_num_samples=1, lale_num_grids=3, scoring=make_scorer(accuracy_score))
        #     best_pipeline = param_search.fit(X, y)
        #     print(accuracy_score(y, best_pipeline.predict(X)))

    def test_increase_num_rows(self):
        from test.test_custom_operators import IncreaseRows
        increase_rows = IncreaseRows()
        trainable = increase_rows >> NoOp()
        iris = sklearn.datasets.load_iris()
        X, y = iris.data[0:10], iris.target[0:10]

        trained = trainable.fit(X, y)
        predicted = trained.transform(X, y)

class TestTfidfVectorizer(unittest.TestCase):
    def test_more_hyperparam_values(self):
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            tf_idf = TfidfVectorizer(max_df=2.5, min_df=2,
                                    max_features=1000,
                                    stop_words='english')
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            tf_idf = TfidfVectorizer(max_df=2, min_df=2,
                                    max_features=1000,
                                    stop_words=['I', 'we', 'not', 'this', 'that'],
                                    analyzer = 'char')

    def test_non_null_tokenizer(self):
        # tokenize the doc and lemmatize its tokens
        def my_tokenizer():
            return 'abc'
        with self.assertRaises(jsonschema.exceptions.ValidationError):
            tf_idf = TfidfVectorizer(max_df=2, min_df=2,
                                    max_features=1000,
                                    stop_words='english',
                                    tokenizer = my_tokenizer,
                                    analyzer = 'char')

class TestHyperoptClassifier(unittest.TestCase):
    def test_nested_pipeline1(self):
        from sklearn.datasets import load_iris
        from lale.lib.lale import HyperoptClassifier
        from sklearn.metrics import accuracy_score
        data = load_iris()
        X, y = data.data, data.target
        #pipeline = KNeighborsClassifier() | (OneHotEncoder(handle_unknown = 'ignore') >> LogisticRegression())
        pipeline = KNeighborsClassifier() | (SimpleImputer() >> LogisticRegression())
        clf = HyperoptClassifier(model = pipeline, max_evals=1)
        trained = clf.fit(X, y)
        predictions = trained.predict(X)
        print(accuracy_score(y, predictions))

    def test_with_concat_features1(self):
        import warnings
        warnings.filterwarnings("ignore")
        import logging
        logging.basicConfig(level=logging.DEBUG)

        from sklearn.datasets import load_iris
        from lale.lib.lale import HyperoptClassifier
        from sklearn.metrics import accuracy_score
        data = load_iris()
        X, y = data.data, data.target
        pca = PCA(n_components=3)
        nys = Nystroem(n_components=10)
        concat = ConcatFeatures()
        lr = LogisticRegression(random_state=42, C=0.1)
        pipeline = ((pca & nys) >> concat >> lr) | KNeighborsClassifier()
        clf = HyperoptClassifier(model = pipeline, max_evals=1)
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
        from lale.lib.lale import HyperoptClassifier
        from sklearn.metrics import accuracy_score
        data = load_iris()
        X, y = data.data, data.target
        pca = PCA(n_components=3)
        nys = Nystroem(n_components=10)
        concat = ConcatFeatures()
        lr = LogisticRegression(random_state=42, C=0.1)
        from lale.operators import make_pipeline
        pipeline = make_pipeline(((((SimpleImputer() | NoOp()) >> pca) & nys) >> concat >> lr) | KNeighborsClassifier())
        clf = HyperoptClassifier(model = pipeline, max_evals=100, handle_cv_failure=True)
        trained = clf.fit(X, y)
        predictions = trained.predict(X)
        print(accuracy_score(y, predictions))
        warnings.resetwarnings()


# class TestGetFeatureNames(unittest.TestCase):
#     def test_gfn_ohe(self):
#         from sklearn.datasets import load_iris
#         import pandas as pd
#         trainable_ohe = OneHotEncoder()
#         iris = load_iris()
#         X_train = iris.data
#         y_train = iris.target
#         df = pd.DataFrame(X_train, columns = iris.feature_names)
#         trained_ohe = trainable_ohe.fit(df)
#         trained_ohe.get_feature_names()
#         trained_ohe = trainable_ohe.fit(X_train)
#         trained_ohe.get_feature_names()
#         trained_ohe.get_feature_names(df.columns)
#     def test_gfn_no_op(self):
#         from sklearn.datasets import load_iris
#         import pandas as pd
#         trainable_ohe = NoOp()
#         iris = load_iris()
#         X_train = iris.data
#         y_train = iris.target
#         df = pd.DataFrame(X_train, columns = iris.feature_names)
#         trained_ohe = trainable_ohe.fit(df)
#         trained_ohe.get_feature_names()
#         trained_ohe = trainable_ohe.fit(X_train)
#         trained_ohe.get_feature_names()
#         trained_ohe.get_feature_names(df.columns)

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

    iris = sklearn.datasets.load_iris()
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
        from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
            UniformFloatHyperparameter, UniformIntegerHyperparameter
        from ConfigSpace.conditions import InCondition

        # Import SMAC-utilities
        from smac.tae.execute_func import ExecuteTAFuncDict
        from smac.scenario.scenario import Scenario
        from smac.facade.smac_facade import SMAC

        from lale.search.SMAC import get_smac_space

        lr = LogisticRegression()

        cs:ConfigurationSpace = get_smac_space(lr)

        # Scenario object
        scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                            "runcount-limit": 200,  # maximum function evaluations
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
        from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
            UniformFloatHyperparameter, UniformIntegerHyperparameter
        from ConfigSpace.conditions import InCondition

        # Import SMAC-utilities
        from smac.tae.execute_func import ExecuteTAFuncDict
        from smac.scenario.scenario import Scenario
        from smac.facade.smac_facade import SMAC


        tfm = PCA() | Nystroem() | NoOp()
        planned_pipeline1 = (OneHotEncoder(handle_unknown = 'ignore',  sparse = False) | NoOp()) >> tfm >> (LogisticRegression() | KNeighborsClassifier())

        cs:ConfigurationSpace = get_smac_space(planned_pipeline1, lale_num_grids=5)

        # Scenario object
        scenario = Scenario({"run_obj": "quality",   # we optimize quality (alternatively runtime)
                            "runcount-limit": 200,  # maximum function evaluations
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
        
class TestOperatorWithoutSchema(unittest.TestCase):
    def test_trainable_pipe_left(self):
        from lale.lib.lale import NoOp
        from lale.lib.sklearn import LogisticRegression
        from sklearn.decomposition import PCA
        iris = sklearn.datasets.load_iris()
        pipeline = PCA() >> LogisticRegression(random_state=42)
        pipeline.fit(iris.data, iris.target)
    
    def test_trainable_pipe_right(self):
        from lale.lib.lale import NoOp
        from lale.lib.sklearn import LogisticRegression
        from sklearn.decomposition import PCA
        iris = sklearn.datasets.load_iris()
        pipeline = NoOp() >> PCA() >> LogisticRegression(random_state=42)
        pipeline.fit(iris.data, iris.target)

    def test_planned_pipe_left(self):
        from lale.lib.lale import NoOp
        from lale.lib.sklearn import LogisticRegression
        from sklearn.decomposition import PCA
        from lale.lib.lale import HyperoptClassifier
        iris = sklearn.datasets.load_iris()
        pipeline = NoOp() >> PCA >> LogisticRegression
        clf = HyperoptClassifier(model=pipeline, max_evals=1)
        clf.fit(iris.data, iris.target)
        
    def test_planned_pipe_right(self):
        from lale.lib.lale import NoOp
        from lale.lib.sklearn import LogisticRegression
        from sklearn.decomposition import PCA
        from lale.lib.lale import HyperoptClassifier
        iris = sklearn.datasets.load_iris()
        pipeline = PCA >> LogisticRegression
        clf = HyperoptClassifier(model=pipeline, max_evals=1)
        clf.fit(iris.data, iris.target)

class TestPrettyPrint(unittest.TestCase):
    def round_trip(self, string1):
        globals1 = {}
        exec(string1, globals1)
        pipeline1 = globals1['pipeline']
        from lale.pretty_print import pipeline_to_string
        string2 = pipeline_to_string(pipeline1)
        self.maxDiff = None
        self.assertEqual(string1, string2)
        globals2 = {}
        exec(string2, globals2)
        pipeline2 = globals2['pipeline']

    def test_reducible(self):
        string1 = \
"""from lale.lib.sklearn import MinMaxScaler
from lale.lib.lale import NoOp
from lale.lib.sklearn import PCA
from lale.lib.sklearn import Nystroem
from lale.lib.lale import ConcatFeatures
from lale.lib.sklearn import KNeighborsClassifier
from lale.lib.sklearn import LogisticRegression
pca = PCA(copy=False)
logistic_regression = LogisticRegression(solver='saga', C=0.9)
pipeline = (MinMaxScaler | NoOp) >> (pca & Nystroem) >> ConcatFeatures >> (KNeighborsClassifier | logistic_regression)"""
        self.round_trip(string1)

    def test_import_as(self):
        #code to reproduce in printing
        from lale.lib.sklearn import MinMaxScaler as Scaler
        from lale.lib.lale import NoOp
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import Nystroem
        from lale.lib.lale import ConcatFeatures as Concat
        from lale.lib.sklearn import KNeighborsClassifier as KNN
        from lale.lib.sklearn import LogisticRegression as LR
        pca = PCA(copy=False)
        lr = LR(solver='saga', C=0.9)
        pipeline = (Scaler | NoOp) >> (pca & Nystroem) >> Concat >> (KNN | lr)
        #expected string
        string1 = \
"""from lale.lib.sklearn import MinMaxScaler as Scaler
from lale.lib.lale import NoOp
from lale.lib.sklearn import PCA
from lale.lib.sklearn import Nystroem
from lale.lib.lale import ConcatFeatures as Concat
from lale.lib.sklearn import KNeighborsClassifier as KNN
from lale.lib.sklearn import LogisticRegression as LR
pca = PCA(copy=False)
lr = LR(solver='saga', C=0.9)
pipeline = (Scaler | NoOp) >> (pca & Nystroem) >> Concat >> (KNN | lr)"""
        #testing harness
        from lale.pretty_print import pipeline_to_string
        string2 = pipeline_to_string(pipeline)
        self.maxDiff = None
        self.assertEqual(string1, string2)
        globals2 = {}
        exec(string2, globals2)
        pipeline2 = globals2['pipeline']

    def test_irreducible(self):
        string1 = \
"""from lale.lib.sklearn import PCA
from lale.lib.sklearn import Nystroem
from lale.lib.sklearn import MinMaxScaler
from lale.lib.sklearn import LogisticRegression
from lale.lib.sklearn import KNeighborsClassifier
from lale.operators import get_pipeline_of_applicable_type
step_1 = PCA | Nystroem
pipeline = get_pipeline_of_applicable_type(
    steps=[step_1, MinMaxScaler, LogisticRegression, KNeighborsClassifier],
    edges=[(step_1,LogisticRegression), (MinMaxScaler,LogisticRegression), (MinMaxScaler,KNeighborsClassifier)])"""
        self.round_trip(string1)

    @unittest.skip("TODO: didn't yet implement what's needed for this to work")
    def test_preprocessing(self):
        string1 = \
"""from ai4ml.transformers.small_data_preprocessing_transformers import CatEncoder
from ai4ml.transformers.small_data_preprocessing_transformers import CatImputer
from ai4ml.transformers.small_data_preprocessing_transformers import CompressStrings
from ai4ml.transformers.small_data_preprocessing_transformers import FloatStr2Float
from ai4ml.transformers.small_data_preprocessing_transformers import NumImputer
from ai4ml.transformers.small_data_preprocessing_transformers import NumpyColumnSelector
from ai4ml.transformers.small_data_preprocessing_transformers import NumpyPermuteArray
from ai4ml.transformers.small_data_preprocessing_transformers import NumpyReplaceMissingValues
from ai4ml.transformers.small_data_preprocessing_transformers import NumpyReplaceUnknownValues
from ai4ml.transformers.small_data_preprocessing_transformers import OptStandardScaler
from ai4ml.transformers.small_data_preprocessing_transformers import boolean2float
from ai4ml.transformers.small_data_preprocessing_transformers import float32_transform
from lale.lib.lale import ConcatFeatures
from lale.lib.xgboost import XGBClassifier
import numpy as np
numpy_column_selector = NumpyColumnSelector(columns=[0, 2, 3])
compress_strings = CompressStrings(activate_flag=True, dtypes_list=['char_str', 'char_str', 'char_str'], missing_values_reference_list=['', np.nan, '-', '?'], misslist_list: [[], [], []])
numpy_replace_missing_values = NumpyReplaceMissingValues(filling_values=np.nan, missing_values=[])
numpy_replace_unknown_values = NumpyReplaceUnknownValues(filling_values=np.nan, filling_values_list=[np.nan, np.nan, np.nan], known_values_list=[['CampingEquipment', 'GolfEquipment', 'MountaineeringEquipment', 'OutdoorProtection', 'PersonalAccessories'], ['Married', 'Single', 'Unspecified'], ['Executive', 'Hospitality', 'Other', 'Professional', 'Retail', 'Retired', 'Sales', 'Student', 'Trades']], missing_values_reference_list=['', np.nan, '-', '?'])
boolean2float_1 = boolean2float(activate_flag=True)
cat_imputer = CatImputer(activate_flag=True, missing_values=np.nan, sklearn_version_family=20, strategy='most_frequent')
cat_encoder = CatEncoder(activate_flag=True, categories='auto', dtype=np.float64, encoding='ordinal', handle_unknown='error', sklearn_version_family=20)
float32_transform_1 = float32_transform(activate_flag=True)
numpy_column_selector_1 = NumpyColumnSelector(columns=[1])
float_str2float = FloatStr2Float(activate_flag=True, dtypes_list=['int_num'], missing_values_reference_list=[])
numpy_replace_missing_values_2 = NumpyReplaceMissingValues(filling_values=np.nan, missing_values=[])
num_imputer = NumImputer(activate_flag=True, missing_values=np.nan, strategy='median')
opt_standard_scaler = OptStandardScaler(num_scaler_copy=None, num_scaler_with_mean=None, num_scaler_with_std=None, use_scaler_flag=False)
float32_transform_2 = float32_transform(activate_flag=True)
numpy_permute_array = NumpyPermuteArray(axis=0, permutation_indices=[0, 2, 3, 1])
xgb_classifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1, colsample_bytree=1, gamma=0, learning_rate=0.1, max_delta_step=0, max_depth=3, min_child_weight=1, missing: None, n_estimators=100, n_jobs=1, nthread=None, objective='binary:logistic', random_state=33, reg_alpha=0, reg_lambda=1, scale_pos_weight=1.0834726777719228, seed=None, silent=True, subsample=1, kwargs={})
pipeline = ((numpy_column_selector >> compress_strings >> numpy_replace_missing_values >> numpy_replace_unknown_values >> boolean2float_1 >> cat_imputer >> cat_encoder >> float32_transform_1) & (numpy_column_selector_1 >> float_str2float >> numpy_replace_missing_values_2 >> num_imputer >> opt_standard_scaler >> float32_transform_2)) >> numpy_permute_array >> xgb_classifier"""
        self.round_trip(string1)

if __name__ == '__main__':
    unittest.main()
