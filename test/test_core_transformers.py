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
from test import EnableSchemaValidation
from typing import Any

import jsonschema

import lale.lib.lale
import lale.lib.sklearn
import lale.type_checking
from lale.lib.lale import ConcatFeatures
from lale.lib.sklearn import (
    NMF,
    PCA,
    RFE,
    FunctionTransformer,
    LogisticRegression,
    MissingIndicator,
    Nystroem,
    TfidfVectorizer,
)


class TestFeaturePreprocessing(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)


def create_function_test_feature_preprocessor(fproc_name):
    def test_feature_preprocessor(self):
        X_train, y_train = self.X_train, self.y_train
        import importlib

        module_name = ".".join(fproc_name.split(".")[0:-1])
        class_name = fproc_name.split(".")[-1]
        module = importlib.import_module(module_name)

        class_ = getattr(module, class_name)
        fproc = class_()

        from lale.lib.sklearn.one_hot_encoder import OneHotEncoder

        if isinstance(fproc, OneHotEncoder):  # type: ignore
            # fproc = OneHotEncoder(handle_unknown = 'ignore')
            # remove the hack when this is fixed
            fproc = PCA()
        # test_schemas_are_schemas
        lale.type_checking.validate_is_schema(fproc.input_schema_fit())
        lale.type_checking.validate_is_schema(fproc.input_schema_transform())
        lale.type_checking.validate_is_schema(fproc.output_schema_transform())
        lale.type_checking.validate_is_schema(fproc.hyperparam_schema())

        # test_init_fit_transform
        trained = fproc.fit(self.X_train, self.y_train)
        _ = trained.transform(self.X_test)

        # test_predict_on_trainable
        trained = fproc.fit(X_train, y_train)
        fproc.transform(X_train)

        # test_to_json
        fproc.to_json()

        # test_in_a_pipeline
        # This test assumes that the output of feature processing is compatible with LogisticRegression
        from lale.lib.sklearn import LogisticRegression

        pipeline = fproc >> LogisticRegression()
        trained = pipeline.fit(self.X_train, self.y_train)
        _ = trained.predict(self.X_test)

        # Tune the pipeline with LR using Hyperopt
        from lale.lib.lale import Hyperopt

        hyperopt = Hyperopt(estimator=pipeline, max_evals=1, verbose=True, cv=3)
        trained = hyperopt.fit(self.X_train, self.y_train)
        _ = trained.predict(self.X_test)

    test_feature_preprocessor.__name__ = "test_{0}".format(fproc_name.split(".")[-1])
    return test_feature_preprocessor


feature_preprocessors = [
    "lale.lib.sklearn.PolynomialFeatures",
    "lale.lib.sklearn.PCA",
    "lale.lib.sklearn.Nystroem",
    "lale.lib.sklearn.Normalizer",
    "lale.lib.sklearn.MinMaxScaler",
    "lale.lib.sklearn.OneHotEncoder",
    "lale.lib.sklearn.SimpleImputer",
    "lale.lib.sklearn.StandardScaler",
    "lale.lib.sklearn.FeatureAgglomeration",
    "lale.lib.sklearn.RobustScaler",
    "lale.lib.sklearn.QuantileTransformer",
    "lale.lib.sklearn.VarianceThreshold",
]
for fproc in feature_preprocessors:
    setattr(
        TestFeaturePreprocessing,
        "test_{0}".format(fproc.split(".")[-1]),
        create_function_test_feature_preprocessor(fproc),
    )


class TestNMF(unittest.TestCase):
    def test_init_fit_predict(self):
        import lale.datasets

        nmf = NMF()
        lr = LogisticRegression()
        trainable = nmf >> lr
        (train_X, train_y), (test_X, test_y) = lale.datasets.digits_df()
        trained = trainable.fit(train_X, train_y)
        _ = trained.predict(test_X)

    def test_not_randome_state(self):
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                _ = NMF(random_state='"not RandomState"')


class TestFunctionTransformer(unittest.TestCase):
    def test_init_fit_predict(self):
        import numpy as np

        import lale.datasets

        ft = FunctionTransformer(func=np.log1p)
        lr = LogisticRegression()
        trainable = ft >> lr
        (train_X, train_y), (test_X, test_y) = lale.datasets.digits_df()
        trained = trainable.fit(train_X, train_y)
        _ = trained.predict(test_X)

    def test_not_callable(self):
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                _ = FunctionTransformer(func='"not callable"')


class TestMissingIndicator(unittest.TestCase):
    def test_init_fit_transform(self):
        import numpy as np

        X1 = np.array([[np.nan, 1, 3], [4, 0, np.nan], [8, 1, 0]])
        X2 = np.array([[5, 1, np.nan], [np.nan, 2, 3], [2, 4, 0]])
        trainable = MissingIndicator()
        trained = trainable.fit(X1)
        transformed = trained.transform(X2)
        expected = np.array([[False, True], [True, False], [False, False]])
        self.assertTrue((transformed == expected).all())


class TestRFE(unittest.TestCase):
    def test_init_fit_predict(self):
        import sklearn.datasets
        import sklearn.svm

        svm = lale.lib.sklearn.SVR(kernel="linear")
        rfe = RFE(estimator=svm, n_features_to_select=2)
        lr = LogisticRegression()
        trainable = rfe >> lr
        data = sklearn.datasets.load_iris()
        X, y = data.data, data.target
        trained = trainable.fit(X, y)
        _ = trained.predict(X)

    def test_init_fit_predict_sklearn(self):
        import sklearn.datasets
        import sklearn.svm

        svm = sklearn.svm.SVR(kernel="linear")
        rfe = RFE(estimator=svm, n_features_to_select=2)
        lr = LogisticRegression()
        trainable = rfe >> lr
        data = sklearn.datasets.load_iris()
        X, y = data.data, data.target
        trained = trainable.fit(X, y)
        _ = trained.predict(X)

    def test_not_operator(self):
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                _ = RFE(estimator='"not an operator"', n_features_to_select=2)

    def test_attrib_sklearn(self):
        import sklearn.datasets
        import sklearn.svm

        from lale.lib.sklearn import RFE, LogisticRegression

        svm = sklearn.svm.SVR(kernel="linear")
        rfe = RFE(estimator=svm, n_features_to_select=2)
        lr = LogisticRegression()
        trainable = rfe >> lr
        data = sklearn.datasets.load_iris()
        X, y = data.data, data.target
        trained = trainable.fit(X, y)
        _ = trained.predict(X)
        from lale.lib.lale import Hyperopt

        opt = Hyperopt(estimator=trainable, max_evals=2, verbose=True)
        opt.fit(X, y)

    def test_attrib(self):
        import sklearn.datasets

        from lale.lib.sklearn import RFE, LogisticRegression

        svm = lale.lib.sklearn.SVR(kernel="linear")
        rfe = RFE(estimator=svm, n_features_to_select=2)
        lr = LogisticRegression()
        trainable = rfe >> lr
        data = sklearn.datasets.load_iris()
        X, y = data.data, data.target
        trained = trainable.fit(X, y)
        _ = trained.predict(X)
        from lale.lib.lale import Hyperopt

        opt = Hyperopt(estimator=trainable, max_evals=2, verbose=True)
        opt.fit(X, y)


class TestOrdinalEncoder(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_with_hyperopt(self):
        from lale.lib.sklearn import OrdinalEncoder

        fproc = OrdinalEncoder(handle_unknown="ignore")
        from lale.lib.sklearn import LogisticRegression

        pipeline = fproc >> LogisticRegression()

        # Tune the pipeline with LR using Hyperopt
        from lale.lib.lale import Hyperopt

        hyperopt = Hyperopt(estimator=pipeline, max_evals=1)
        trained = hyperopt.fit(self.X_train, self.y_train)
        _ = trained.predict(self.X_test)

    def test_inverse_transform(self):
        from lale.lib.sklearn import OneHotEncoder, OrdinalEncoder

        fproc_ohe = OneHotEncoder(handle_unknown="ignore")
        # test_init_fit_transform
        trained_ohe = fproc_ohe.fit(self.X_train, self.y_train)
        transformed_X = trained_ohe.transform(self.X_test)
        orig_X_ohe = trained_ohe._impl._wrapped_model.inverse_transform(transformed_X)

        fproc_oe = OrdinalEncoder(handle_unknown="ignore")
        # test_init_fit_transform
        trained_oe = fproc_oe.fit(self.X_train, self.y_train)
        transformed_X = trained_oe.transform(self.X_test)
        orig_X_oe = trained_oe._impl.inverse_transform(transformed_X)
        self.assertEqual(orig_X_ohe.all(), orig_X_oe.all())

    def test_handle_unknown_error(self):
        from lale.lib.sklearn import OrdinalEncoder

        fproc_oe = OrdinalEncoder(handle_unknown="error")
        # test_init_fit_transform
        trained_oe = fproc_oe.fit(self.X_train, self.y_train)
        with self.assertRaises(
            ValueError
        ):  # This is repying on the train_test_split, so may fail randomly
            _ = trained_oe.transform(self.X_test)

    def test_encode_unknown_with(self):
        from lale.lib.sklearn import OrdinalEncoder

        fproc_oe = OrdinalEncoder(handle_unknown="ignore", encode_unknown_with=1000)
        # test_init_fit_transform
        trained_oe = fproc_oe.fit(self.X_train, self.y_train)
        transformed_X = trained_oe.transform(self.X_test)
        # This is repying on the train_test_split, so may fail randomly
        self.assertTrue(1000 in transformed_X)
        # Testing that inverse_transform works even for encode_unknown_with=1000
        _ = trained_oe._impl.inverse_transform(transformed_X)


class TestConcatFeatures(unittest.TestCase):
    def test_hyperparam_defaults(self):
        _ = ConcatFeatures()

    def test_init_fit_predict(self):
        trainable_cf = ConcatFeatures()
        A = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
        B = [[14, 15], [24, 25], [34, 35]]

        trained_cf = trainable_cf.fit(X=[A, B])
        transformed: Any = trained_cf.transform([A, B])
        expected = [[11, 12, 13, 14, 15], [21, 22, 23, 24, 25], [31, 32, 33, 34, 35]]
        for i_sample in range(len(transformed)):
            for i_feature in range(len(transformed[i_sample])):
                self.assertEqual(
                    transformed[i_sample][i_feature], expected[i_sample][i_feature]
                )

    def test_comparison_with_scikit(self):
        import warnings

        warnings.filterwarnings("ignore")
        import sklearn.datasets
        import sklearn.utils

        from lale.helpers import cross_val_score
        from lale.lib.sklearn import PCA

        pca = PCA(n_components=3, random_state=42, svd_solver="arpack")
        nys = Nystroem(n_components=10, random_state=42)
        concat = ConcatFeatures()
        lr = LogisticRegression(random_state=42, C=0.1)
        trainable = (pca & nys) >> concat >> lr
        digits = sklearn.datasets.load_digits()
        X, y = sklearn.utils.shuffle(digits.data, digits.target, random_state=42)

        cv_results = cross_val_score(trainable, X, y)
        cv_results = ["{0:.1%}".format(score) for score in cv_results]

        from sklearn.decomposition import PCA as SklearnPCA
        from sklearn.kernel_approximation import Nystroem as SklearnNystroem
        from sklearn.linear_model import LogisticRegression as SklearnLR
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import FeatureUnion, make_pipeline

        union = FeatureUnion(
            [
                (
                    "pca",
                    SklearnPCA(n_components=3, random_state=42, svd_solver="arpack"),
                ),
                ("nys", SklearnNystroem(n_components=10, random_state=42)),
            ]
        )
        lr = SklearnLR(random_state=42, C=0.1)
        pipeline = make_pipeline(union, lr)

        scikit_cv_results = cross_val_score(pipeline, X, y, cv=5)
        scikit_cv_results = ["{0:.1%}".format(score) for score in scikit_cv_results]
        self.assertEqual(cv_results, scikit_cv_results)
        warnings.resetwarnings()

    def test_with_pandas(self):
        import warnings

        from lale.datasets import load_iris_df

        warnings.filterwarnings("ignore")
        pca = PCA(n_components=3)
        nys = Nystroem(n_components=10)
        concat = ConcatFeatures()
        lr = LogisticRegression(random_state=42, C=0.1)
        trainable = (pca & nys) >> concat >> lr

        (X_train, y_train), (X_test, y_test) = load_iris_df()
        trained = trainable.fit(X_train, y_train)
        _ = trained.predict(X_test)

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
        from lale.lib.lale import Hyperopt
        from lale.operators import make_pipeline, make_union

        pca = PCA(n_components=3)
        nys = Nystroem(n_components=10)
        lr = LogisticRegression(random_state=42, C=0.1)

        trainable = make_pipeline(make_union(pca, nys), lr)
        clf = Hyperopt(estimator=trainable, max_evals=2)
        from sklearn.datasets import load_iris

        iris_data = load_iris()
        clf.fit(iris_data.data, iris_data.target)
        clf.predict(iris_data.data)


class TestTfidfVectorizer(unittest.TestCase):
    def test_more_hyperparam_values(self):
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                _ = TfidfVectorizer(
                    max_df=2.5, min_df=2, max_features=1000, stop_words="english"
                )
            with self.assertRaises(jsonschema.ValidationError):
                _ = TfidfVectorizer(
                    max_df=2,
                    min_df=2,
                    max_features=1000,
                    stop_words=["I", "we", "not", "this", "that"],
                    analyzer="char",
                )

    def test_non_null_tokenizer(self):
        # tokenize the doc and lemmatize its tokens
        def my_tokenizer():
            return "abc"

        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                _ = TfidfVectorizer(
                    max_df=2,
                    min_df=2,
                    max_features=1000,
                    stop_words="english",
                    tokenizer=my_tokenizer,
                    analyzer="char",
                )
