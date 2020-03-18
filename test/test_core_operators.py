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
import sklearn.datasets
import inspect

import lale.lib.lale
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
from lale.lib.sklearn import RidgeClassifier
from lale.search.lale_grid_search_cv import get_grid_search_parameter_grids
from lale.sklearn_compat import make_sklearn_compat

import lale.operators as Ops
class TestClassification(unittest.TestCase):

    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(X, y)    

def create_function_test_classifier(clf_name):
    def test_classifier(self):
        X_train, y_train = self.X_train, self.y_train
        X_test, y_test = self.X_test, self.y_test
        import importlib
        module_name = ".".join(clf_name.split('.')[0:-1])
        class_name = clf_name.split('.')[-1]
        module = importlib.import_module(module_name)

        class_ = getattr(module, class_name)
        clf = class_()

        #test_schemas_are_schemas
        from lale.helpers import validate_is_schema
        validate_is_schema(clf.input_schema_fit())
        validate_is_schema(clf.input_schema_predict())
        validate_is_schema(clf.output_schema_predict())
        validate_is_schema(clf.hyperparam_schema())

        #test_init_fit_predict
        trained = clf.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)

        #test_with_hyperopt
        from lale.lib.lale import Hyperopt
        hyperopt = Hyperopt(estimator=clf, max_evals=1)
        trained = hyperopt.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)

        #test_cross_validation
        from lale.helpers import cross_val_score
        cv_results = cross_val_score(clf, X_train, y_train, cv = 2)
        self.assertEqual(len(cv_results), 2)

        #test_with_gridsearchcv_auto_wrapped
        from sklearn.metrics import accuracy_score, make_scorer
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from lale.lib.sklearn.gradient_boosting_classifier import GradientBoostingClassifierImpl
            from lale.lib.sklearn.mlp_classifier import MLPClassifierImpl
            if clf._impl_class() == GradientBoostingClassifierImpl:
                #because exponential loss does not work with iris dataset as it is not binary classification
                import lale.schemas as schemas
                clf = clf.customize_schema(loss=schemas.Enum(default='deviance', values=['deviance']))
            grid_search = lale.lib.lale.GridSearchCV(
                estimator=clf, lale_num_samples=1, lale_num_grids=1,
                cv=2, scoring=make_scorer(accuracy_score))
            grid_search.fit(X_train, y_train)

        #test_predict_on_trainable
        trained = clf.fit(X_train, y_train)
        clf.predict(X_train)

        #test_to_json
        clf.to_json()

        #test_in_a_pipeline
        pipeline = NoOp() >> clf
        trained = pipeline.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)


    test_classifier.__name__ = 'test_{0}'.format(clf.split('.')[-1])
    return test_classifier

classifiers = ['lale.lib.sklearn.RandomForestClassifier',
               'lale.lib.sklearn.DecisionTreeClassifier',
               'lale.lib.sklearn.ExtraTreesClassifier',
               'lale.lib.sklearn.GradientBoostingClassifier',
               'lale.lib.sklearn.GaussianNB',
               'lale.lib.sklearn.QuadraticDiscriminantAnalysis',
               'lale.lib.lightgbm.LGBMClassifier',
               'lale.lib.xgboost.XGBClassifier',
               'lale.lib.sklearn.KNeighborsClassifier',
               'lale.lib.sklearn.LinearSVC',
               'lale.lib.sklearn.LogisticRegression',
               'lale.lib.sklearn.MLPClassifier',
               'lale.lib.sklearn.SVC',
               'lale.lib.sklearn.PassiveAggressiveClassifier',
               'lale.lib.sklearn.MultinomialNB',
               'lale.lib.sklearn.AdaBoostClassifier',
               'lale.lib.sklearn.SGDClassifier',
               'lale.lib.sklearn.RidgeClassifier']
for clf in classifiers:
    setattr(
        TestClassification,
        'test_{0}'.format(clf.split('.')[-1]),
        create_function_test_classifier(clf)
    )

class TestRegression(unittest.TestCase):

    def setUp(self):
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split
        X, y = make_regression(n_features=4, n_informative=2,
                               random_state=0, shuffle=False)
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(X, y)    

def create_function_test_regressor(clf_name):
    def test_regressor(self):
        X_train, y_train = self.X_train, self.y_train
        X_test, y_test = self.X_test, self.y_test
        import importlib
        module_name = ".".join(clf_name.split('.')[0:-1])
        class_name = clf_name.split('.')[-1]
        module = importlib.import_module(module_name)

        class_ = getattr(module, class_name)
        regr = class_()

        #test_schemas_are_schemas
        from lale.helpers import validate_is_schema
        validate_is_schema(regr.input_schema_fit())
        validate_is_schema(regr.input_schema_predict())
        validate_is_schema(regr.output_schema_predict())
        validate_is_schema(regr.hyperparam_schema())

        #test_init_fit_predict
        trained = regr.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)

        #test_predict_on_trainable
        trained = regr.fit(X_train, y_train)
        regr.predict(X_train)

        #test_to_json
        regr.to_json()

        #test_in_a_pipeline
        pipeline = NoOp() >> regr
        trained = pipeline.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)

        #test_with_hyperopt
        from lale.lib.sklearn.ridge import RidgeImpl
        if regr._impl_class() != RidgeImpl:
            from lale.lib.lale import Hyperopt
            hyperopt = Hyperopt(estimator=pipeline, max_evals=1, scoring='r2')
            trained = hyperopt.fit(self.X_train, self.y_train)
            predictions = trained.predict(self.X_test)

    test_regressor.__name__ = 'test_{0}'.format(clf_name.split('.')[-1])
    return test_regressor

regressors = ['lale.lib.sklearn.RandomForestRegressor',
              'lale.lib.sklearn.DecisionTreeRegressor',
              'lale.lib.sklearn.ExtraTreesRegressor',
              'lale.lib.sklearn.GradientBoostingRegressor',
              'lale.lib.sklearn.LinearRegression',
              'lale.lib.sklearn.Ridge',
              'lale.lib.lightgbm.LGBMRegressor',
              'lale.lib.xgboost.XGBRegressor',
              'lale.lib.sklearn.AdaBoostRegressor',
              'lale.lib.sklearn.SGDRegressor']
for clf in regressors:
    setattr(
        TestRegression,
        'test_{0}'.format(clf.split('.')[-1]),
        create_function_test_regressor(clf)
    )

class TestFeaturePreprocessing(unittest.TestCase):

    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(X, y)    

def create_function_test_feature_preprocessor(fproc_name):
    def test_feature_preprocessor(self):
        X_train, y_train = self.X_train, self.y_train
        X_test, y_test = self.X_test, self.y_test
        import importlib
        module_name = ".".join(fproc_name.split('.')[0:-1])
        class_name = fproc_name.split('.')[-1]
        module = importlib.import_module(module_name)

        class_ = getattr(module, class_name)
        fproc = class_()

        from lale.lib.sklearn.one_hot_encoder import OneHotEncoderImpl
        if fproc._impl_class() == OneHotEncoderImpl:
            #fproc = OneHotEncoder(handle_unknown = 'ignore')
            #remove the hack when this is fixed
            fproc = PCA()
        #test_schemas_are_schemas
        from lale.helpers import validate_is_schema
        validate_is_schema(fproc.input_schema_fit())
        validate_is_schema(fproc.input_schema_transform())
        validate_is_schema(fproc.output_schema_transform())
        validate_is_schema(fproc.hyperparam_schema())

        #test_init_fit_transform
        trained = fproc.fit(self.X_train, self.y_train)
        predictions = trained.transform(self.X_test)

        #test_predict_on_trainable
        trained = fproc.fit(X_train, y_train)
        fproc.transform(X_train)

        #test_to_json
        fproc.to_json()

        #test_in_a_pipeline
        #This test assumes that the output of feature processing is compatible with LogisticRegression
        from lale.lib.sklearn import LogisticRegression
        pipeline = fproc >> LogisticRegression()
        trained = pipeline.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)

        #Tune the pipeline with LR using Hyperopt
        from lale.lib.lale import Hyperopt
        hyperopt = Hyperopt(estimator=pipeline, max_evals=1)
        trained = hyperopt.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)

    test_feature_preprocessor.__name__ = 'test_{0}'.format(fproc_name.split('.')[-1])
    return test_feature_preprocessor

feature_preprocessors = ['lale.lib.sklearn.PolynomialFeatures',
                         'lale.lib.sklearn.PCA',
                         'lale.lib.sklearn.Nystroem',
                         'lale.lib.sklearn.Normalizer',
                         'lale.lib.sklearn.MinMaxScaler',
                         'lale.lib.sklearn.OneHotEncoder',
                         'lale.lib.sklearn.SimpleImputer',
                         'lale.lib.sklearn.StandardScaler',
                         'lale.lib.sklearn.FeatureAgglomeration',
                         'lale.lib.sklearn.RobustScaler',
                         'lale.lib.sklearn.QuantileTransformer'
                         ]
for fproc in feature_preprocessors:
    setattr(
        TestFeaturePreprocessing,
        'test_{0}'.format(fproc.split('.')[-1]),
        create_function_test_feature_preprocessor(fproc)
    )

class TestConcatFeatures(unittest.TestCase):
    def test_hyperparam_defaults(self):
        cf = ConcatFeatures()
    def test_init_fit_predict(self):
        trainable_cf = ConcatFeatures()
        A = [ [11, 12, 13],
              [21, 22, 23],
              [31, 32, 33] ]
        B = [ [14, 15],
              [24, 25],
              [34, 35] ]

        trained_cf = trainable_cf.fit(X = [A, B])
        transformed = trained_cf.transform([A, B])
        expected = [ [11, 12, 13, 14, 15],
                     [21, 22, 23, 24, 25],
                     [31, 32, 33, 34, 35] ]
        for i_sample in range(len(transformed)):
            for i_feature in range(len(transformed[i_sample])):
                self.assertEqual(transformed[i_sample][i_feature],
                                 expected[i_sample][i_feature])
    def test_comparison_with_scikit(self):
        import warnings
        warnings.filterwarnings("ignore")
        from lale.lib.sklearn import PCA
        import sklearn.datasets
        from lale.helpers import cross_val_score
        pca = PCA(n_components=3, random_state=42, svd_solver='arpack')
        nys = Nystroem(n_components=10, random_state=42)
        concat = ConcatFeatures()
        lr = LogisticRegression(random_state=42, C=0.1)
        trainable = (pca & nys) >> concat >> lr
        digits = sklearn.datasets.load_digits()
        X, y = sklearn.utils.shuffle(digits.data, digits.target, random_state=42)

        cv_results = cross_val_score(trainable, X, y)
        cv_results = ['{0:.1%}'.format(score) for score in cv_results]

        from sklearn.pipeline import make_pipeline, FeatureUnion
        from sklearn.decomposition import PCA as SklearnPCA
        from sklearn.kernel_approximation import Nystroem as SklearnNystroem
        from sklearn.linear_model import LogisticRegression as SklearnLR
        from sklearn.model_selection import cross_val_score
        union = FeatureUnion([("pca", SklearnPCA(n_components=3, random_state=42, svd_solver='arpack')),
                            ("nys", SklearnNystroem(n_components=10, random_state=42))])
        lr = SklearnLR(random_state=42, C=0.1)
        pipeline = make_pipeline(union, lr)

        scikit_cv_results = cross_val_score(pipeline, X, y, cv = 5)
        scikit_cv_results = ['{0:.1%}'.format(score) for score in scikit_cv_results]
        self.assertEqual(cv_results, scikit_cv_results)
        warnings.resetwarnings()
    def test_with_pandas(self):
        from lale.datasets import load_iris_df
        import warnings
        warnings.filterwarnings("ignore")
        pca = PCA(n_components=3)
        nys = Nystroem(n_components=10)
        concat = ConcatFeatures()
        lr = LogisticRegression(random_state=42, C=0.1)
        trainable = (pca & nys) >> concat >> lr

        (X_train, y_train), (X_test, y_test) = load_iris_df()
        trained = trainable.fit(X_train, y_train)
        predicted = trained.predict(X_test)
    def test_concat_with_hyperopt(self):
        from lale.lib.lale import Hyperopt
        pca = PCA(n_components=3)
        nys = Nystroem(n_components=10)
        concat = ConcatFeatures()
        lr = LogisticRegression(random_state=42, C=0.1)

        trainable = (pca & nys) >> concat >> lr
        clf = Hyperopt(estimator=trainable, max_evals=2)
        from sklearn.datasets import load_iris
        iris_data = load_iris()
        clf.fit(iris_data.data, iris_data.target)
        clf.predict(iris_data.data)

    def test_concat_with_hyperopt2(self):
        from lale.operators import make_pipeline, make_union
        from lale.lib.lale import Hyperopt
        pca = PCA(n_components=3)
        nys = Nystroem(n_components=10)
        concat = ConcatFeatures()
        lr = LogisticRegression(random_state=42, C=0.1)

        trainable = make_pipeline(make_union(pca, nys), lr)
        clf = Hyperopt(estimator=trainable, max_evals=2)
        from sklearn.datasets import load_iris
        iris_data = load_iris()
        clf.fit(iris_data.data, iris_data.target)
        clf.predict(iris_data.data)

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
        #with self.assertWarns(DeprecationWarning):
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
        #with self.assertWarns(DeprecationWarning):
        predicted = trainable_lr.predict_proba(iris.data)
        predicted = trained_lr.predict_proba(iris.data)

    def test_with_sklearn_gridsearchcv(self):
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
    def test_grid_search_on_trained(self):
        from sklearn.model_selection import GridSearchCV
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
        iris = load_iris()
        X, y = iris.data, iris.target
        lr = LogisticRegression()
        trained = lr.fit(X, y)
        parameters = {'solver':('liblinear', 'lbfgs'), 'penalty':['l2']}

        clf = GridSearchCV(trained, parameters, cv=5, scoring=make_scorer(accuracy_score))
    def test_grid_search_on_trained_auto(self):
        from sklearn.model_selection import GridSearchCV
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
        iris = load_iris()
        X, y = iris.data, iris.target
        lr = LogisticRegression()
        trained = lr.fit(X, y)
        parameters = get_grid_search_parameter_grids(lr, num_samples=2)

        clf = GridSearchCV(trained, parameters, cv=5, scoring=make_scorer(accuracy_score))
    def test_doc(self):
        import sklearn.datasets
        import sklearn.utils
        from test.mock_custom_operators import MyLR
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

class TestClone(unittest.TestCase):
    def test_clone_with_scikit1(self):
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

    def test_clone_with_scikit2(self):
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
    def test_clone_of_trained(self):
        from sklearn.base import clone
        lr = LogisticRegression()
        from sklearn.datasets import load_iris
        iris = load_iris()
        X, y = iris.data, iris.target
        trained = lr.fit(X, y)
        trained2 = clone(trained)
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
#        with self.assertWarns(DeprecationWarning):
        predicted = trainable.predict_proba(iris.data)
        predicted = trained.predict_proba(iris.data)

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

class TestTfidfVectorizer(unittest.TestCase):
    def test_more_hyperparam_values(self):
        with self.assertRaises(jsonschema.ValidationError):
            tf_idf = TfidfVectorizer(max_df=2.5, min_df=2,
                                    max_features=1000,
                                    stop_words='english')
        with self.assertRaises(jsonschema.ValidationError):
            tf_idf = TfidfVectorizer(max_df=2, min_df=2,
                                    max_features=1000,
                                    stop_words=['I', 'we', 'not', 'this', 'that'],
                                    analyzer = 'char')

    def test_non_null_tokenizer(self):
        # tokenize the doc and lemmatize its tokens
        def my_tokenizer():
            return 'abc'
        with self.assertRaises(jsonschema.ValidationError):
            tf_idf = TfidfVectorizer(max_df=2, min_df=2,
                                    max_features=1000,
                                    stop_words='english',
                                    tokenizer = my_tokenizer,
                                    analyzer = 'char')

class TestTags(unittest.TestCase):
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

    def dont_test_planned_pipe_left(self):
        from lale.lib.lale import NoOp
        from lale.lib.sklearn import LogisticRegression
        from sklearn.decomposition import PCA
        from lale.lib.lale import Hyperopt
        iris = sklearn.datasets.load_iris()
        pipeline = NoOp() >> PCA >> LogisticRegression
        clf = Hyperopt(estimator=pipeline, max_evals=1)
        clf.fit(iris.data, iris.target)
        
    def dont_test_planned_pipe_right(self):
        from lale.lib.lale import NoOp
        from lale.lib.sklearn import LogisticRegression
        from sklearn.decomposition import PCA
        from lale.lib.lale import Hyperopt
        iris = sklearn.datasets.load_iris()
        pipeline = PCA >> LogisticRegression
        clf = Hyperopt(estimator=pipeline, max_evals=1)
        clf.fit(iris.data, iris.target)

class TestVotingClassifier(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(X, y)    

    def test_with_lale_classifiers(self):
        from lale.lib.sklearn import VotingClassifier
        clf = VotingClassifier(estimators=[('knn', KNeighborsClassifier()), ('lr', LogisticRegression())])
        trained = clf.fit(self.X_train, self.y_train)
        trained.predict(self.X_test)
        
    def test_with_lale_pipeline(self):
        from lale.lib.sklearn import VotingClassifier
        clf = VotingClassifier(estimators=[('knn', KNeighborsClassifier()), ('pca_lr', PCA() >> LogisticRegression())])
        trained = clf.fit(self.X_train, self.y_train)
        trained.predict(self.X_test)

    def test_with_hyperopt(self):
        from lale.lib.sklearn import VotingClassifier
        from lale.lib.lale import Hyperopt
        clf = VotingClassifier(estimators=[('knn', KNeighborsClassifier()), ('lr', LogisticRegression())])
        trained = clf.auto_configure(self.X_train, self.y_train, Hyperopt, max_evals=1)

    def test_with_gridsearch(self):
        from lale.lib.sklearn import VotingClassifier
        from lale.lib.lale import GridSearchCV
        from sklearn.metrics import accuracy_score, make_scorer
        clf = VotingClassifier(estimators=[('knn', KNeighborsClassifier()), ('lr', LogisticRegression())])
        trained = clf.auto_configure(self.X_train, self.y_train, GridSearchCV, lale_num_samples=1, lale_num_grids=1, cv=2, scoring=make_scorer(accuracy_score))

class TestBaggingClassifier(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(X, y)    

    def test_with_lale_classifiers(self):
        from lale.lib.sklearn import BaggingClassifier
        from lale.sklearn_compat import make_sklearn_compat
        clf = BaggingClassifier(base_estimator=LogisticRegression())
        trained = clf.fit(self.X_train, self.y_train)
        trained.predict(self.X_test)

    def test_with_lale_pipeline(self):
        from lale.lib.sklearn import BaggingClassifier
        clf = BaggingClassifier(base_estimator = PCA() >> LogisticRegression())
        trained = clf.fit(self.X_train, self.y_train)
        trained.predict(self.X_test)

    def test_with_hyperopt(self):
        from lale.lib.sklearn import BaggingClassifier
        from lale.lib.lale import Hyperopt
        clf = BaggingClassifier(base_estimator=LogisticRegression())
        trained = clf.auto_configure(self.X_train, self.y_train, Hyperopt, max_evals=1)
        print(trained.to_json())

    def test_pipeline_with_hyperopt(self):
        from lale.lib.sklearn import BaggingClassifier
        from lale.lib.lale import Hyperopt
        clf = BaggingClassifier(base_estimator=PCA() >> LogisticRegression())
        trained = clf.auto_configure(self.X_train, self.y_train, Hyperopt, max_evals=1)

    def test_pipeline_choice_with_hyperopt(self):
        from lale.lib.sklearn import BaggingClassifier
        from lale.lib.lale import Hyperopt
        clf = BaggingClassifier(base_estimator=PCA() >> (LogisticRegression() | KNeighborsClassifier()))
        trained = clf.auto_configure(self.X_train, self.y_train, Hyperopt, max_evals=1)

class TestLazyImpl(unittest.TestCase):
    def test_lazy_impl(self):
        from lale.lib.lale import Hyperopt
        impl = Hyperopt._impl
        self.assertTrue(inspect.isclass(impl))
