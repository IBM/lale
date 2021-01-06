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

# Test cases for miscellaneous functionality of Lale that is also part of the
# core behavior but does not fall into other test_core* modules.
import inspect
import io
import logging
import unittest
import warnings

from sklearn.datasets import load_iris

import lale.operators as Ops
import lale.type_checking
from lale.lib.lale import ConcatFeatures, NoOp
from lale.lib.sklearn import (
    NMF,
    PCA,
    KNeighborsClassifier,
    LogisticRegression,
    MLPClassifier,
    Nystroem,
    OneHotEncoder,
    RandomForestClassifier,
)
from lale.sklearn_compat import make_sklearn_compat


class TestTags(unittest.TestCase):
    def test_estimators(self):
        ops = Ops.get_available_estimators()
        ops_names = [op.name() for op in ops]
        self.assertIn("LogisticRegression", ops_names)
        self.assertIn("MLPClassifier", ops_names)
        self.assertNotIn("PCA", ops_names)

    def test_interpretable_estimators(self):
        ops = Ops.get_available_estimators({"interpretable"})
        ops_names = [op.name() for op in ops]
        self.assertIn("KNeighborsClassifier", ops_names)
        self.assertNotIn("MLPClassifier", ops_names)
        self.assertNotIn("PCA", ops_names)

    def test_transformers(self):
        ops = Ops.get_available_transformers()
        ops_names = [op.name() for op in ops]
        self.assertIn("PCA", ops_names)
        self.assertNotIn("LogisticRegression", ops_names)
        self.assertNotIn("MLPClassifier", ops_names)


class TestOperatorWithoutSchema(unittest.TestCase):
    def test_trainable_pipe_left(self):
        from sklearn.decomposition import PCA

        from lale.lib.sklearn import LogisticRegression

        iris = load_iris()
        pipeline = PCA() >> LogisticRegression(random_state=42)
        pipeline.fit(iris.data, iris.target)

    def test_trainable_pipe_right(self):
        from sklearn.decomposition import PCA

        from lale.lib.lale import NoOp
        from lale.lib.sklearn import LogisticRegression

        iris = load_iris()
        pipeline = NoOp() >> PCA() >> LogisticRegression(random_state=42)
        pipeline.fit(iris.data, iris.target)

    def dont_test_planned_pipe_left(self):
        from sklearn.decomposition import PCA

        from lale.lib.lale import Hyperopt, NoOp
        from lale.lib.sklearn import LogisticRegression

        iris = load_iris()
        pipeline = NoOp() >> PCA >> LogisticRegression
        clf = Hyperopt(estimator=pipeline, max_evals=1)
        clf.fit(iris.data, iris.target)

    def dont_test_planned_pipe_right(self):
        from sklearn.decomposition import PCA

        from lale.lib.lale import Hyperopt
        from lale.lib.sklearn import LogisticRegression

        iris = load_iris()
        pipeline = PCA >> LogisticRegression
        clf = Hyperopt(estimator=pipeline, max_evals=1)
        clf.fit(iris.data, iris.target)


class TestLazyImpl(unittest.TestCase):
    def test_lazy_impl(self):
        from lale.lib.lale import Hyperopt

        impl = Hyperopt._impl
        self.assertTrue(inspect.isclass(impl))


class TestOperatorErrors(unittest.TestCase):
    def test_trainable_get_pipeline_fail(self):
        try:
            _ = LogisticRegression().get_pipeline
            self.fail("get_pipeline did not fail")
        except AttributeError as e:
            msg: str = str(e)
            self.assertRegex(msg, "TrainableOperator is deprecated")
            self.assertRegex(msg, "meant to train")

    def test_trained_get_pipeline_fail(self):
        try:
            _ = NoOp().get_pipeline
            self.fail("get_pipeline did not fail")
        except AttributeError as e:
            msg: str = str(e)
            self.assertRegex(msg, "underlying operator")

    def test_trained_get_pipeline_success(self):
        from lale.lib.lale import Hyperopt

        iris_data = load_iris()
        op = Hyperopt(estimator=LogisticRegression(), max_evals=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            op2 = op.fit(iris_data.data[10:], iris_data.target[10:])
            _ = op2.get_pipeline

    def test_trainable_summary_fail(self):
        try:
            _ = LogisticRegression().summary
            self.fail("summary did not fail")
        except AttributeError as e:
            msg: str = str(e)
            self.assertRegex(msg, "TrainableOperator is deprecated")
            self.assertRegex(msg, "meant to train")

    def test_trained_summary_fail(self):
        try:
            _ = NoOp().summary
            self.fail("summary did not fail")
        except AttributeError as e:
            msg: str = str(e)
            self.assertRegex(msg, "underlying operator")

    def test_trained_summary_success(self):
        from lale.lib.lale import Hyperopt

        iris_data = load_iris()
        op = Hyperopt(
            estimator=LogisticRegression(), max_evals=1, show_progressbar=False
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            op2 = op.fit(iris_data.data[10:], iris_data.target[10:])
            _ = op2.summary


class TestLaleVersion(unittest.TestCase):
    def test_version_exists(self):
        import lale

        self.assertIsNot(lale.__version__, None)


class TestOperatorLogging(unittest.TestCase):
    def setUp(self):
        self.old_level = Ops.logger.level
        Ops.logger.setLevel(logging.INFO)
        self.stream = io.StringIO()
        self.handler = logging.StreamHandler(self.stream)
        Ops.logger.addHandler(self.handler)

    @unittest.skip("Turned off the logging for now")
    def test_log_fit_predict(self):
        import lale.datasets

        trainable = LogisticRegression()
        (X_train, y_train), (X_test, y_test) = lale.datasets.load_iris_df()
        trained = trainable.fit(X_train, y_train)
        _ = trained.predict(X_test)
        self.handler.flush()
        s1, s2, s3, s4 = self.stream.getvalue().strip().split("\n")
        self.assertTrue(s1.endswith("enter fit LogisticRegression"))
        self.assertTrue(s2.endswith("exit  fit LogisticRegression"))
        self.assertTrue(s3.endswith("enter predict LogisticRegression"))
        self.assertTrue(s4.endswith("exit  predict LogisticRegression"))

    def tearDown(self):
        Ops.logger.removeHandler(self.handler)
        Ops.logger.setLevel(self.old_level)
        self.handler.close()


class TestBoth(unittest.TestCase):
    def test_init_fit_transform(self):
        import lale.datasets
        from lale.lib.lale import Both

        nmf = NMF()
        pca = PCA()
        trainable = Both(op1=nmf, op2=pca)
        (train_X, train_y), (test_X, test_y) = lale.datasets.digits_df()
        trained = trainable.fit(train_X, train_y)
        _ = trained.transform(test_X)


class TestClone(unittest.TestCase):
    def test_clone_with_scikit1(self):
        lr = LogisticRegression()
        lr.get_params()
        from sklearn.base import clone

        lr_clone = clone(lr)
        self.assertNotEqual(lr, lr_clone)
        self.assertNotEqual(lr._impl, lr_clone._impl)
        iris = load_iris()
        trained_lr = lr.fit(iris.data, iris.target)
        predicted = trained_lr.predict(iris.data)
        cloned_trained_lr = clone(trained_lr)
        self.assertNotEqual(trained_lr._impl, cloned_trained_lr._impl)
        predicted_clone = cloned_trained_lr.predict(iris.data)
        for i in range(len(iris.target)):
            self.assertEqual(predicted[i], predicted_clone[i])
        # Testing clone with pipelines having OperatorChoice

    def test_clone_operator_choice(self):
        from sklearn.base import clone
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.model_selection import cross_val_score

        iris = load_iris()
        X, y = iris.data, iris.target

        lr = LogisticRegression()
        trainable = PCA() >> lr
        trainable_wrapper = make_sklearn_compat(trainable)
        trainable2 = clone(trainable_wrapper)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cross_val_score(
                trainable_wrapper, X, y, scoring=make_scorer(accuracy_score), cv=2
            )
            result2 = cross_val_score(
                trainable2, X, y, scoring=make_scorer(accuracy_score), cv=2
            )
        for i in range(len(result)):
            self.assertEqual(result[i], result2[i])

    def test_clone_with_scikit2(self):
        lr = LogisticRegression()
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.model_selection import cross_val_score

        pca = PCA()
        trainable = pca >> lr
        from sklearn.base import clone

        iris = load_iris()
        X, y = iris.data, iris.target
        trainable2 = clone(trainable)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cross_val_score(
                trainable, X, y, scoring=make_scorer(accuracy_score), cv=2
            )
            result2 = cross_val_score(
                trainable2, X, y, scoring=make_scorer(accuracy_score), cv=2
            )
        for i in range(len(result)):
            self.assertEqual(result[i], result2[i])
        # Testing clone with nested linear pipelines
        trainable = PCA() >> trainable
        trainable2 = clone(trainable)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = cross_val_score(
                trainable, X, y, scoring=make_scorer(accuracy_score), cv=2
            )
            result2 = cross_val_score(
                trainable2, X, y, scoring=make_scorer(accuracy_score), cv=2
            )
        for i in range(len(result)):
            self.assertEqual(result[i], result2[i])

    def test_clone_of_trained(self):
        from sklearn.base import clone

        lr = LogisticRegression()

        iris = load_iris()
        X, y = iris.data, iris.target
        trained = lr.fit(X, y)
        _ = clone(trained)

    def test_with_voting_classifier1(self):
        lr = LogisticRegression()
        knn = KNeighborsClassifier()
        from sklearn.ensemble import VotingClassifier

        vclf = VotingClassifier(estimators=[("lr", lr), ("knn", knn)])

        iris = load_iris()
        X, y = iris.data, iris.target
        vclf.fit(X, y)

    def test_with_voting_classifier2(self):
        lr = LogisticRegression()
        pca = PCA()
        trainable = pca >> lr

        from sklearn.ensemble import VotingClassifier

        vclf = VotingClassifier(estimators=[("lr", lr), ("pipe", trainable)])

        iris = load_iris()
        X, y = iris.data, iris.target
        vclf.fit(X, y)

    def test_fit_clones_impl(self):

        lr_trainable = LogisticRegression()
        iris = load_iris()
        X, y = iris.data, iris.target
        lr_trained = lr_trainable.fit(X, y)
        self.assertIsNot(lr_trainable._impl, lr_trained._impl)


class TestHyperparamRanges(unittest.TestCase):
    def exactly_relevant_properties(self, keys1, operator):
        def sorted(ll):
            l_copy = [*ll]
            l_copy.sort()
            return l_copy

        keys2 = operator.hyperparam_schema()["allOf"][0]["relevantToOptimizer"]
        self.assertEqual(sorted(keys1), sorted(keys2))

    def validate_get_param_ranges(self, operator):
        ranges, cat_idx = operator.get_param_ranges()
        self.exactly_relevant_properties(ranges.keys(), operator)
        # all defaults are in-range
        for hp, r in ranges.items():
            if isinstance(r, tuple):
                minimum, maximum, default = r
                if minimum is not None and maximum is not None and default is not None:
                    assert minimum <= default and default <= maximum
            else:
                minimum, maximum, default = cat_idx[hp]
                assert minimum == 0 and len(r) - 1 == maximum

    def validate_get_param_dist(self, operator):
        size = 5
        dist = operator.get_param_dist(size)
        self.exactly_relevant_properties(dist.keys(), operator)
        for hp, d in dist.items():
            self.assertTrue(len(d) > 0)
            if isinstance(d[0], int):
                self.assertTrue(len(d) <= size)
            elif isinstance(d[0], float):
                self.assertTrue(len(d) == size)
            schema = operator.hyperparam_schema(hp)
            for v in d:
                lale.type_checking.validate_schema(v, schema)

    def test_get_param_ranges_and_dist(self):
        for op in [
            ConcatFeatures,
            KNeighborsClassifier,
            LogisticRegression,
            MLPClassifier,
            Nystroem,
            OneHotEncoder,
            PCA,
            RandomForestClassifier,
        ]:
            self.validate_get_param_ranges(op)
            self.validate_get_param_dist(op)

    def test_sklearn_get_param_ranges_and_dist(self):
        for op in [
            ConcatFeatures,
            KNeighborsClassifier,
            LogisticRegression,
            MLPClassifier,
            Nystroem,
            OneHotEncoder,
            PCA,
            RandomForestClassifier,
        ]:
            skop = make_sklearn_compat(op)
            self.validate_get_param_ranges(skop)
            self.validate_get_param_dist(skop)

    def test_random_forest_classifier(self):
        ranges, dists = RandomForestClassifier.get_param_ranges()
        expected_ranges = {
            "n_estimators": (10, 100, 10),
            "criterion": ["entropy", "gini"],
            "max_depth": (3, 5, None),
            "min_samples_split": (2, 5, 2),
            "min_samples_leaf": (1, 5, 1),
            "max_features": (0.01, 1.0, 0.5),
        }
        self.maxDiff = None
        self.assertEqual(ranges, expected_ranges)
