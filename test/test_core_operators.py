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
        validate_is_schema(clf.output_schema())
        validate_is_schema(clf.hyperparam_schema())

        #test_init_fit_predict
        trained = clf.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)

        #test_with_hyperopt
        from lale.lib.lale import HyperoptCV
        hyperopt = HyperoptCV(estimator=clf, max_evals=1)
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
            if isinstance(clf._impl, GradientBoostingClassifierImpl):
                #because exponential loss does not work with iris dataset as it is not binary classification
                import lale.schemas as schemas
                clf = clf.customize_schema(loss=schemas.Enum(default='deviance', values=['deviance']))
            if not isinstance(clf._impl, MLPClassifierImpl):
                #mlp fails due to issue #164.
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
               'lale.lib.sklearn.MultinomialNB']
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
        validate_is_schema(regr.output_schema())
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
        if not isinstance(regr._impl, RidgeImpl):
            from lale.lib.lale import HyperoptCV
            hyperopt = HyperoptCV(estimator=pipeline, max_evals=1, scoring='r2')
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
              'lale.lib.xgboost.XGBRegressor']
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
        if isinstance(fproc._impl, OneHotEncoderImpl):
            #fproc = OneHotEncoder(handle_unknown = 'ignore')
            #remove the hack when this is fixed
            fproc = PCA()
        #test_schemas_are_schemas
        from lale.helpers import validate_is_schema
        validate_is_schema(fproc.input_schema_fit())
        validate_is_schema(fproc.input_schema_transform())
        validate_is_schema(fproc.output_schema())
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

        #Tune the pipeline with LR using HyperoptCV
        from lale.lib.lale import HyperoptCV
        hyperopt = HyperoptCV(estimator=pipeline, max_evals=1)
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
