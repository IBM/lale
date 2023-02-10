import inspect
import logging

import pytest
from sklearn import datasets

from lale.lib import autogen
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
            return
        except Exception:
            test(3 * i)

    test(1)


kls = inspect.getmembers(autogen, lambda m: isinstance(m, Operator))
LR = LogisticRegression.customize_schema(relevantToOptimizer=[])


classifiers = [
    "BernoulliNB",
    "CalibratedClassifierCV",
    "ComplementNB",
    "GaussianProcessClassifier",
    "LGBMClassifier",
    "LabelPropagation",
    "LabelSpreading",
    "LogisticRegressionCV",
    "NearestCentroid",
    "NuSVC",
    "Perceptron",
    "RadiusNeighborsClassifier",
    "RidgeClassifierCV",
]


@pytest.mark.parametrize("name, Op", [(n, Op) for (n, Op) in kls if n in classifiers])
def test_classifier(name, Op):
    base_test(name, Op, load_iris)


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
    "BayesianRidge",
    "ElasticNet",
    "ElasticNetCV",
    "GaussianProcessRegressor",
    "HuberRegressor",
    "Lars",
    "LarsCV",
    "Lasso",
    "LassoCV",
    "LassoLars",
    "LassoLarsCV",
    "LassoLarsIC",
    "LGBMRegressor",
    "NuSVR",
    "OrthogonalMatchingPursuit",
    "OrthogonalMatchingPursuitCV",
    "PassiveAggressiveRegressor",
    "RANSACRegressor",
    "KernelRidge",
    "RidgeCV",
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


@pytest.mark.parametrize("name, reason", failed_regressors)
def test_failed_regressor(name, reason):
    pytest.xfail(reason)


transformers = [
    "AdditiveChi2Sampler",
    "BernoulliRBM",
    "Binarizer",
    "Birch",
    "DictionaryLearning",
    # "FactorAnalysis",
    "FastICA",
    "GaussianRandomProjection",
    "IncrementalPCA",
    "KBinsDiscretizer",
    "KernelPCA",
    "LinearDiscriminantAnalysis",
    "LocallyLinearEmbedding",
    "MaxAbsScaler",
    "MiniBatchDictionaryLearning",
    "MiniBatchKMeans",
    "MiniBatchSparsePCA",
    "PowerTransformer",
    # "RandomTreesEmbedding",
    "RBFSampler",
    "SkewedChi2Sampler",
    "SparsePCA",
    "SparseRandomProjection",
    "TruncatedSVD",
]

failed_transformers = [
    ("CCA", "Fit required Y (not y)"),
    ("LabelBinarizer", "operates on labels (not supported by lale yet)"),
    ("LabelEncoder", "operates on labels (not supported by lale yet)"),
    ("LatentDirichletAllocation", "Failed 2D array output"),
    ("MultiLabelBinarizer", "operates on labels (not supported by lale yet)"),
    ("PLSCanonical", "Fit required Y (not y)"),
    ("PLSRegression", "Fit required Y (not y)"),
    ("PLSSVD", "Fit required Y (not y)"),
]


@pytest.mark.parametrize("name, Op", [(n, Op) for (n, Op) in kls if n in transformers])
def test_transformer(name, Op):
    base_test(name, Op >> LR, load_iris)


@pytest.mark.parametrize("name, reason", failed_transformers)
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
