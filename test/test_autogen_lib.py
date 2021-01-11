import inspect
import logging
from random import choice

import numpy as np
import pytest
from sklearn import datasets

import lale.lib.autogen as autogen
from lale.lib.lale import Hyperopt
from lale.lib.lale.hyperopt import logger
from lale.lib.sklearn import LogisticRegression
from lale.operators import Operator, make_choice

logger.setLevel(logging.ERROR)


def load_iris():
    iris = datasets.load_iris()
    return iris.data, iris.target


def load_regression():
    return datasets.make_regression(
        n_features=4, n_informative=2, random_state=0, shuffle=False
    )


def base_test(name, pipeline, data_loader, max_evals=250, scoring="accuracy"):
    def test(i):
        if i > max_evals:
            assert False
        try:
            X, y = data_loader()
            clf = Hyperopt(estimator=pipeline, max_evals=i, scoring=scoring)
            trained_pipeline = clf.fit(X, y)
            trained_pipeline.predict(X)
            return True
        except Exception:
            test(3 * i)

    test(1)


kls = inspect.getmembers(autogen, lambda m: isinstance(m, Operator))
LR = LogisticRegression.customize_schema(relevantToOptimizer=[])


classifiers = [
    "AdaBoostClassifier",
    "BernoulliNB",
    "CalibratedClassifierCV",
    "ComplementNB",
    "DecisionTreeClassifier",
    "ExtraTreesClassifier",
    "GaussianNB",
    "GaussianProcessClassifier",
    "GradientBoostingClassifier",
    "KNeighborsClassifier",
    "LGBMClassifier",
    "LabelPropagation",
    "LabelSpreading",
    "LinearSVC",
    "LogisticRegression",
    "LogisticRegressionCV",
    "MLPClassifier",
    "MultinomialNB",
    "NearestCentroid",
    "NuSVC",
    "PassiveAggressiveClassifier",
    "Perceptron",
    "QuadraticDiscriminantAnalysis",
    "RadiusNeighborsClassifier",
    "RandomForestClassifier",
    "RidgeClassifier",
    "RidgeClassifierCV",
    "SGDClassifier",
    "SVC",
    "SVR",
]

failed_classifiers = [
    ("LinearRegression", "Input predict type (matrix with one column)"),
]


@pytest.mark.parametrize("name, Op", [(n, Op) for (n, Op) in kls if n in classifiers])
def test_classifier(name, Op):
    base_test(name, Op, load_iris)


@pytest.mark.parametrize("name, reason", [(n, r) for (n, r) in failed_classifiers])
def test_failed_classifier(name, reason):
    pytest.xfail(reason)


multi = [
    "MultiTaskElasticNet",
    "MultiTaskElasticNetCV",
    "MultiTaskLasso",
    "MultiTaskLassoCV",
]


@pytest.mark.parametrize("name, Op", [(n, Op) for (n, Op) in kls if n in multi])
def test_multi(name, Op):
    def load_multi():
        X_multi = [[i, i] for i in range(100)]
        return X_multi, X_multi

    pytest.xfail(reason="Documentation error predict output type is 2D")
    base_test(name, Op, load_multi)


regressors = [
    "ARDRegression",
    "AdaBoostRegressor",
    "BayesianRidge",
    "DecisionTreeRegressor",
    "ElasticNet",
    "ElasticNetCV",
    "ExtraTreesRegressor",
    "GaussianProcessRegressor",
    "GradientBoostingRegressor",
    "HuberRegressor",
    "KNeighborsRegressor",
    "Lars",
    "LarsCV",
    "Lasso",
    "LassoCV",
    "LassoLars",
    "LassoLarsCV",
    "LassoLarsIC",
    "LGBMRegressor",
    "LinearSVR",
    "NuSVR",
    "OrthogonalMatchingPursuit",
    "OrthogonalMatchingPursuitCV",
    "PassiveAggressiveRegressor",
    "RANSACRegressor",
    "RandomForestRegressor",
    "KernelRidge",
    "Ridge",
    "RidgeCV",
    "SGDRegressor",
    "TheilSenRegressor",
    "TransformedTargetRegressor",
]

failed_regressors = [
    ("MLPRegressor", "Input predict type (matrix with one column)"),
    ("RadiusNeighborsRegressor", "Radius argument is data dependent"),
]


@pytest.mark.parametrize("name, Op", [(n, Op) for (n, Op) in kls if n in regressors])
def test_regressors(name, Op):
    base_test(name, Op, load_regression, scoring="r2")


@pytest.mark.parametrize("name, reason", [(n, r) for (n, r) in failed_regressors])
def test_failed_regressor(name, reason):
    pytest.xfail(reason)


transformers = [
    "AdditiveChi2Sampler",
    "BernoulliRBM",
    "Binarizer",
    "Birch",
    "DictionaryLearning",
    "FactorAnalysis",
    "FastICA",
    "FunctionTransformer",
    "GaussianRandomProjection",
    "IncrementalPCA",
    "Isomap",
    "KBinsDiscretizer",
    "KMeans",
    "KernelPCA",
    "LinearDiscriminantAnalysis",
    "LocallyLinearEmbedding",
    "MaxAbsScaler",
    "MinMaxScaler",
    "MiniBatchDictionaryLearning",
    "MiniBatchKMeans",
    "MiniBatchSparsePCA",
    "NMF",
    "Normalizer",
    "Nystroem",
    "PCA",
    "PolynomialFeatures",
    "PowerTransformer",
    "QuantileTransformer",
    "RandomTreesEmbedding",
    "RBFSampler",
    "RobustScaler",
    "SimpleImputer",
    "SkewedChi2Sampler",
    "SparsePCA",
    "SparseRandomProjection",
    "StandardScaler",
    "TruncatedSVD",
]

failed_transformers = [
    ("CCA", "Fit required Y (not y)"),
    ("LabelBinarizer", "operates on labels (not supported by lale yet)"),
    ("LabelEncoder", "operates on labels (not supported by lale yet)"),
    ("LatentDirichletAllocation", "Failed 2D array output"),
    ("MultiLabelBinarizer", "operates on labels (not supported by lale yet)"),
    ("MissingIndicator", "Output boolean (wrong pipeline)"),
    ("OneHotEncoder", "Unknown categories during transform"),
    ("PLSCanonical", "Fit required Y (not y)"),
    ("PLSRegression", "Fit required Y (not y)"),
    ("PLSSVD", "Fit required Y (not y)"),
]


@pytest.mark.parametrize("name, Op", [(n, Op) for (n, Op) in kls if n in transformers])
def test_transformer(name, Op):
    base_test(name, Op >> LR, load_iris)


def test_ordinal_encoder():
    from lale.lib.autogen import OrdinalEncoder as Op

    def data_loader():
        X = np.array([[choice([0, 2]), choice([1, 2, 3])] for _ in range(100)])
        y = np.array([choice([0, 1]) for _ in X])
        return X, y

    base_test("OrdinalEncoder", Op >> LR, data_loader)


@pytest.mark.xfail(reason="Output boolean array (pipeline?)")
def test_missing_indicator():
    from lale.lib.autogen import MissingIndicator as Op

    def data_loader():
        X = np.array(
            [[choice([-1, 1, 2, 3]), choice([-1, 1, 2, 3])] for _ in range(100)]
        )
        y = np.array([choice([0, 1]) for _ in X])
        return X, y

    base_test("MissingIndicator", Op >> LR, data_loader)


@pytest.mark.parametrize("name, reason", [(n, r) for (n, r) in failed_transformers])
def test_failed_transformer(name, reason):
    pytest.xfail(reason)


def test_2_steps_classifier():
    T = make_choice(*[Op for (n, Op) in kls if n in transformers])
    C = make_choice(*[Op for (n, Op) in kls if n in classifiers])
    base_test("transformer_classifier", T >> C, load_iris)


def test_2_steps_regressor():
    T = make_choice(*[Op for (n, Op) in kls if n in transformers])
    R = make_choice(*[Op for (n, Op) in kls if n in regressors])
    base_test("transformer_regressor", T >> R, load_regression, scoring="r2")
