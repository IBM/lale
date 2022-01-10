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
from typing import Any, Dict

from sklearn.datasets import load_iris

import lale.operators as Ops
import lale.type_checking
from lale.helpers import nest_HPparams
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


class TestUnparseExpr(unittest.TestCase):
    def test_unparse_const38(self):
        import lale.expressions
        from lale.expressions import it

        test_expr = it.hello["hi"]
        # This fails on 3.8 with some versions of the library
        # which is why we use the fixed version
        # import astunparse
        # astunparse.unparse(he._expr)
        str(lale.expressions.fixedUnparse(test_expr._expr))


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


class _TestLazyImpl(unittest.TestCase):
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
            self.assertRegex(msg, "is not trained.")
            self.assertRegex(msg, "the result of fit is a new trained")

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
            self.assertRegex(msg, "is not trained.")
            self.assertRegex(msg, "the result of fit is a new trained")

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


class TestMethodParameters(unittest.TestCase):
    def test_fit_predict_params_individual(self):
        from test.mock_custom_operators import CustomParamsCheckerOp

        trainable = CustomParamsCheckerOp()

        trained = trainable.fit([[3, 4], [5, 6]], fit_version=5)
        _ = trained.predict([3, 4], predict_version=6)

        self.assertEqual(trained.impl._fit_params.get("fit_version", None), 5)
        self.assertEqual(trained.impl._predict_params.get("predict_version", None), 6)

    def test_predict_params_pipeline(self):
        from test.mock_custom_operators import CustomParamsCheckerOp

        trainable = CustomParamsCheckerOp() >> CustomParamsCheckerOp()

        trained = trainable.fit([[3, 4], [5, 6]], y=[3], fit_version=5)
        _ = trained.predict([3, 4], predict_version=6)

        self.assertEqual(
            trained.steps_list()[1].impl._predict_params.get("predict_version", None), 6
        )


#        self.assertEqual(trained.steps()[1].impl._fit_params.get("fit_version", None), 5)


class TestWrappedImpl(unittest.TestCase):
    def test_impl(self):
        import sklearn.preprocessing._encoders as skohe

        ohe = OneHotEncoder()
        self.assertIsInstance(ohe.impl, skohe.OneHotEncoder)

    def test_shallow_impl(self):
        import lale.lib.sklearn.one_hot_encoder as lohe

        ohe = OneHotEncoder()
        self.assertIsInstance(ohe.shallow_impl, lohe._OneHotEncoderImpl)


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


class TestTee(unittest.TestCase):
    def test_tee_None(self):
        import lale.datasets
        from lale.lib.lale import Tee

        pca = PCA()
        trainable = Tee() >> pca
        (train_X, train_y), (test_X, test_y) = lale.datasets.digits_df()
        trained = trainable.fit(train_X, train_y)
        _ = trained.transform(test_X)

    def test_tee_lambda(self):
        import numpy as np

        import lale.datasets
        from lale.lib.lale import Tee

        def check_data(X, y):
            self.assertEqual(X.dtypes["x1"], np.float64)

        pca = PCA()
        trainable = Tee(listener=lambda df, y: check_data(df, y)) >> pca
        (train_X, train_y), (test_X, test_y) = lale.datasets.digits_df()
        trained = trainable.fit(train_X, train_y)
        _ = trained.transform(test_X)

    def test_tee_def(self):
        import numpy as np

        import lale.datasets
        from lale.lib.lale import Tee

        def check_data(X, y):
            self.assertEqual(X.dtypes["x1"], np.float64)

        pca = PCA()
        trainable = Tee(listener=check_data) >> pca
        (train_X, train_y), (test_X, test_y) = lale.datasets.digits_df()
        trained = trainable.fit(train_X, train_y)
        _ = trained.transform(test_X)

    def test_tee_obj(self):
        import numpy as np

        import lale.datasets
        from lale.lib.lale import Tee

        class check_data:
            def __init__(self, outerSelf):
                self._outerSelf = outerSelf

            def __call__(self, X, y):
                self._outerSelf.assertEqual(X.dtypes["x1"], np.float64)

        pca = PCA()
        trainable = Tee(listener=check_data(self)) >> pca
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
        _ = trained_lr.predict(iris.data)
        cloned_trained_lr = clone(trained_lr)
        self.assertNotEqual(trained_lr._impl, cloned_trained_lr._impl)
        # Testing clone with pipelines having OperatorChoice

    def test_clone_operator_pipeline(self):
        from sklearn.base import clone
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.model_selection import cross_val_score

        iris = load_iris()
        X, y = iris.data, iris.target

        lr = LogisticRegression()
        trainable = PCA() >> lr
        trainable_wrapper = trainable
        trainable2 = clone(trainable_wrapper)
        _ = clone(trainable)
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

    def test_clone_operator_choice(self):
        from sklearn.base import clone

        lr = LogisticRegression()
        trainable = (PCA() | NoOp) >> lr
        trainable_wrapper = trainable
        _ = clone(trainable_wrapper)
        _ = clone(trainable)

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


class TestGetParams(unittest.TestCase):
    @classmethod
    def remove_lale_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for (k, v) in params.items() if not k.startswith("_lale_")}

    def test_shallow_planned_individual_operator(self):
        op: Ops.PlannedIndividualOp = LogisticRegression
        params = op.get_params(deep=False)
        filtered_params = self.remove_lale_params(params)

        self.assertIn("_lale_schemas", params)
        expected = LogisticRegression.get_defaults()

        self.assertEqual(filtered_params, expected)

    def test_shallow0_planned_individual_operator(self):
        op: Ops.PlannedIndividualOp = LogisticRegression
        params = op.get_params(deep=0)

        self.assertNotIn("_lale_schemas", params)
        expected = LogisticRegression.get_defaults()

        self.assertEqual(params, expected)

    def test_deep_planned_individual_operator(self):
        op: Ops.PlannedIndividualOp = LogisticRegression
        params = op.get_params(deep=True)
        filtered_params = self.remove_lale_params(params)

        expected = LogisticRegression.get_defaults()

        self.assertEqual(filtered_params, expected)

    def test_shallow_trainable_individual_operator_defaults(self):
        op: Ops.TrainableIndividualOp = LogisticRegression()
        params = op.get_params(deep=False)
        filtered_params = self.remove_lale_params(params)

        expected = LogisticRegression.get_defaults()

        self.assertEqual(filtered_params, expected)

    def test_shallow_trainable_individual_operator_configured(self):
        op: Ops.TrainableIndividualOp = LogisticRegression(
            LogisticRegression.enum.solver.saga
        )
        params = op.get_params(deep=False)
        filtered_params = self.remove_lale_params(params)

        expected = dict(LogisticRegression.get_defaults())
        expected["solver"] = "saga"

        self.assertEqual(filtered_params, expected)

    def test_shallow_trained_individual_operator_defaults(self):
        op1: Ops.TrainableIndividualOp = LogisticRegression()
        iris = load_iris()
        op: Ops.TrainedIndividualOp = op1.fit(iris.data, iris.target)

        params = op.get_params(deep=False)
        filtered_params = self.remove_lale_params(params)

        expected = LogisticRegression.get_defaults()

        self.assertEqual(filtered_params, expected)

    def test_shallow_trained_individual_operator_configured(self):
        op1: Ops.TrainableIndividualOp = LogisticRegression(
            LogisticRegression.enum.solver.saga
        )
        iris = load_iris()
        op: Ops.TrainedIndividualOp = op1.fit(iris.data, iris.target)

        params = op.get_params(deep=False)
        filtered_params = self.remove_lale_params(params)

        expected = dict(LogisticRegression.get_defaults())
        expected["solver"] = "saga"

        self.assertEqual(filtered_params, expected)

    def test_shallow_planned_pipeline(self):
        op: Ops.PlannedPipeline = PCA >> LogisticRegression

        params = op.get_params(deep=False)
        assert "steps" in params
        assert "_lale_preds" in params
        pca = params["steps"][0]
        lr = params["steps"][1]
        assert isinstance(pca, Ops.PlannedIndividualOp)
        assert isinstance(lr, Ops.PlannedIndividualOp)
        lr_params = lr.get_params()
        lr_filtered_params = self.remove_lale_params(lr_params)

        lr_expected = LogisticRegression.get_defaults()

        self.assertEqual(lr_filtered_params, lr_expected)

    def test_shallow_planned_pipeline_with_trainable_default(self):
        op: Ops.PlannedPipeline = PCA >> LogisticRegression()

        params = op.get_params(deep=False)
        assert "steps" in params
        assert "_lale_preds" in params
        pca = params["steps"][0]
        lr = params["steps"][1]
        assert isinstance(pca, Ops.PlannedIndividualOp)
        assert isinstance(lr, Ops.TrainableIndividualOp)
        lr_params = lr.get_params()
        lr_filtered_params = self.remove_lale_params(lr_params)

        lr_expected = LogisticRegression.get_defaults()

        self.assertEqual(lr_filtered_params, lr_expected)

    def test_shallow_planned_pipeline_with_trainable_configured(self):
        op: Ops.PlannedPipeline = PCA >> LogisticRegression(
            LogisticRegression.enum.solver.saga
        )

        params = op.get_params(deep=False)
        assert "steps" in params
        assert "_lale_preds" in params
        pca = params["steps"][0]
        lr = params["steps"][1]
        assert isinstance(pca, Ops.PlannedIndividualOp)
        assert isinstance(lr, Ops.TrainableIndividualOp)
        lr_params = lr.get_params()
        lr_filtered_params = self.remove_lale_params(lr_params)

        lr_expected = dict(LogisticRegression.get_defaults())
        lr_expected["solver"] = "saga"

        self.assertEqual(lr_filtered_params, lr_expected)

    def test_shallow_trainable_pipeline_default(self):
        op: Ops.TrainablePipeline = PCA() >> LogisticRegression()

        params = op.get_params(deep=False)
        assert "steps" in params
        assert "_lale_preds" in params
        pca = params["steps"][0]
        lr = params["steps"][1]
        assert isinstance(pca, Ops.TrainableIndividualOp)
        assert isinstance(lr, Ops.TrainableIndividualOp)
        lr_params = lr.get_params()
        lr_filtered_params = self.remove_lale_params(lr_params)

        lr_expected = LogisticRegression.get_defaults()

        self.assertEqual(lr_filtered_params, lr_expected)

    def test_shallow_trainable_pipeline_configured(self):
        op: Ops.TrainablePipeline = PCA() >> LogisticRegression(
            LogisticRegression.enum.solver.saga
        )

        params = op.get_params(deep=False)
        assert "steps" in params
        assert "_lale_preds" in params
        pca = params["steps"][0]
        lr = params["steps"][1]
        assert isinstance(pca, Ops.TrainableIndividualOp)
        assert isinstance(lr, Ops.TrainableIndividualOp)
        lr_params = lr.get_params()
        lr_filtered_params = self.remove_lale_params(lr_params)

        lr_expected = dict(LogisticRegression.get_defaults())
        lr_expected["solver"] = "saga"

        self.assertEqual(lr_filtered_params, lr_expected)

    def test_shallow0_trainable_pipeline_configured(self):
        op: Ops.TrainablePipeline = PCA() >> LogisticRegression(
            LogisticRegression.enum.solver.saga
        )

        params = op.get_params(deep=0)
        assert "steps" in params
        assert "_lale_preds" not in params
        pca = params["steps"][0]
        lr = params["steps"][1]
        assert isinstance(pca, Ops.TrainableIndividualOp)
        assert isinstance(lr, Ops.TrainableIndividualOp)
        lr_params = lr.get_params()

        lr_expected = dict(LogisticRegression.get_defaults())
        lr_expected["solver"] = "saga"

        self.assertEqual(lr_params, lr_expected)

    def test_shallow_planned_nested_indiv_operator(self):
        from lale.lib.sklearn import BaggingClassifier, DecisionTreeClassifier

        clf = BaggingClassifier(base_estimator=DecisionTreeClassifier())
        params = clf.get_params(deep=False)
        filtered_params = self.remove_lale_params(params)
        assert filtered_params["bootstrap"]

    def test_shallow_planned_nested_list_indiv_operator(self):
        from lale.lib.sklearn import DecisionTreeClassifier, VotingClassifier

        clf = VotingClassifier(estimators=[("dtc", DecisionTreeClassifier())])
        params = clf.get_params(deep=False)
        filtered_params = self.remove_lale_params(params)
        filtered_params["voting"] == "hard"

    def test_deep_planned_pipeline(self):
        op: Ops.PlannedPipeline = PCA >> LogisticRegression

        params = op.get_params(deep=True)
        assert "steps" in params
        assert "_lale_preds" not in params
        pca = params["steps"][0]
        lr = params["steps"][1]
        assert isinstance(pca, Ops.PlannedIndividualOp)
        assert isinstance(lr, Ops.PlannedIndividualOp)
        assert "LogisticRegression__fit_intercept" in params
        lr_params = lr.get_params()
        lr_filtered_params = self.remove_lale_params(lr_params)

        lr_expected = LogisticRegression.get_defaults()

        self.assertEqual(lr_filtered_params, lr_expected)

    def test_deep_planned_choice(self):
        op: Ops.PlannedPipeline = (PCA | NoOp) >> LogisticRegression

        params = op.get_params(deep=True)
        assert "steps" in params
        choice = params["steps"][0]
        assert isinstance(choice, Ops.OperatorChoice)
        choice_name = choice.name()
        self.assertTrue(params[choice_name + "__PCA__copy"])

    def test_deep_planned_nested_indiv_operator(self):
        from lale.lib.sklearn import BaggingClassifier, DecisionTreeClassifier

        dtc = DecisionTreeClassifier()
        clf = BaggingClassifier(base_estimator=dtc)
        params = clf.get_params(deep=True)
        filtered_params = self.remove_lale_params(params)

        # expected = LogisticRegression.get_defaults()
        base = filtered_params["base_estimator"]
        base_params = self.remove_lale_params(base.get_params(deep=True))
        nested_base_params = nest_HPparams("base_estimator", base_params)
        self.assertDictEqual(
            {
                k: v
                for k, v in filtered_params.items()
                if k.startswith("base_estimator__")
                and not k.startswith("base_estimator___lale")
            },
            nested_base_params,
        )

    def test_deep_grammar(self):
        from lale.grammar import Grammar
        from lale.lib.sklearn import BaggingClassifier, DecisionTreeClassifier
        from lale.lib.sklearn import KNeighborsClassifier as KNN
        from lale.lib.sklearn import LogisticRegression as LR
        from lale.lib.sklearn import StandardScaler as Scaler

        dtc = DecisionTreeClassifier()
        clf = BaggingClassifier(base_estimator=dtc)
        params = clf.get_params(deep=True)
        filtered_params = self.remove_lale_params(params)

        g = Grammar()
        g.start = g.estimator
        g.estimator = (NoOp | g.transformer) >> g.prim_est
        g.transformer = (NoOp | g.transformer) >> g.prim_tfm

        g.prim_est = LR | KNN
        g.prim_tfm = PCA | Scaler

        params = g.get_params(deep=True)
        filtered_params = self.remove_lale_params(params)
        assert filtered_params["start__name"] == "estimator"
        assert filtered_params["prim_est__LogisticRegression__penalty"] == "l2"

    # TODO: design question.
    # def test_deep_planned_nested_list_indiv_operator(self):
    #     from lale.lib.sklearn import VotingClassifier, DecisionTreeClassifier
    #
    #     clf = VotingClassifier(estimators=[("dtc", DecisionTreeClassifier())])
    #     params = clf.get_params(deep=True)
    #     filtered_params = self.remove_lale_params(params)
    #
    #     # expected = LogisticRegression.get_defaults()
    #     base = filtered_params['base_estimator']
    #     base_params = self.remove_lale_params(base.get_params(deep=True))
    #     nested_base_params = nest_HPparams('base_esimator', base_params)
    #
    #     self.assertLess(nested_base_params, filtered_params)


class TestWithParams(unittest.TestCase):
    @classmethod
    def remove_lale_params(cls, params: Dict[str, Any]) -> Dict[str, Any]:
        return {k: v for (k, v) in params.items() if not k.startswith("_lale_")}

    def test_shallow_copied_trainable_individual_operator(self):
        from lale.lib.lightgbm import LGBMClassifier as LGBM

        op: Ops.PlannedIndividualOp = LGBM()
        op2 = op.clone()
        new_param_dict = {"learning_rate": 0.8}
        op3 = op2.with_params(**new_param_dict)
        params = op3.get_params(deep=False)

        self.assertEqual(params["learning_rate"], 0.8)


class UserValidatorImpl:
    @classmethod
    def validate_hyperparams(cls, **hyperparams):
        assert "validate" in hyperparams
        v = hyperparams["validate"]
        if not v:
            raise ValueError("validate set to False!")

    def __init__(self, validate=True):
        pass

    def fit(self, X, y=None):
        return self


_user_validate_hyperparam_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "additionalProperties": False,
            "relevantToOptimizer": [],
            "required": ["validate"],
            "properties": {
                "validate": {
                    "type": "boolean",
                    "default": True,
                    "description": "Should we pass validation?",
                }
            },
        }
    ],
}

_user_validate_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "type": "object",
    "tags": {"pre": [], "op": [""], "post": []},
    "properties": {
        "hyperparams": _user_validate_hyperparam_schema,
        # "input_fit": None,
        # "input_transform": ,
        # "output_transform": _output_transform_schema,
    },
}


UserValidatorOp = Ops.make_operator(UserValidatorImpl, _user_validate_combined_schemas)


class TestUserValidator(unittest.TestCase):
    def test_validate_none(self):
        import re

        import jsonschema

        self.assertRaisesRegex(
            jsonschema.ValidationError,
            re.compile(
                r"invalid value validate=None.*boolean", re.MULTILINE | re.DOTALL
            ),
            UserValidatorOp,
            validate=None,
        )

    def test_validate_true(self):
        UserValidatorOp(validate=True)

    def test_validate_false(self):
        self.assertRaisesRegex(
            ValueError, "validate set to False!", UserValidatorOp, validate=False
        )


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
                lale.type_checking.validate_schema_directly(v, schema)

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
            skop = op
            self.validate_get_param_ranges(skop)
            self.validate_get_param_dist(skop)

    def test_random_forest_classifier(self):
        ranges, dists = RandomForestClassifier.get_param_ranges()
        expected_ranges = {
            "n_estimators": (10, 100, 100),
            "criterion": ["entropy", "gini"],
            "max_depth": (3, 5, None),
            "min_samples_split": (2, 5, 2),
            "min_samples_leaf": (1, 5, 1),
            "max_features": (0.01, 1.0, 0.5),
        }
        self.maxDiff = None
        self.assertEqual(ranges, expected_ranges)

    def test_lgbclassifier(self):
        from lale.lib.lightgbm import LGBMClassifier

        ranges, dists = LGBMClassifier.get_param_ranges()
        expected_ranges = {
            "boosting_type": ["dart", "gbdt"],
            "num_leaves": [4, 8, 32, 64, 128, 16, 2],
            "learning_rate": (0.02, 1.0, 0.1),
            "n_estimators": (50, 1000, 200),
            "min_child_weight": (0.0001, 0.01, 0.001),
            "min_child_samples": (5, 30, 20),
            "subsample": (0.01, 1.0, 1.0),
            "subsample_freq": (0, 5, 0),
            "colsample_bytree": (0.01, 1.0, 1.0),
            "reg_alpha": (0.0, 1.0, 0.0),
            "reg_lambda": (0.0, 1.0, 0.0),
        }
        self.maxDiff = None
        self.assertEqual(ranges, expected_ranges)

    def test_bool_enum(self):
        from lale.lib.sklearn import SVR
        from lale.schemas import AnyOf, Bool, Null

        SVR = SVR.customize_schema(
            shrinking=AnyOf(
                types=[Bool(), Null()],
                default=None,
                desc="Whether to use the shrinking heuristic.",
            )
        )

        ranges, dists = SVR.get_param_ranges()
        expected_ranges = {
            "kernel": ["poly", "rbf", "sigmoid", "linear"],
            "degree": (2, 5, 3),
            "gamma": (3.0517578125e-05, 8, None),
            "tol": (0.0, 0.01, 0.001),
            "C": (0.03125, 32768, 1.0),
            "shrinking": [False, True, None],
        }

        self.maxDiff = None
        self.assertEqual(ranges, expected_ranges)


class TestScoreIndividualOp(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_score_planned_op(self):
        from lale.lib.sklearn import LogisticRegression

        with self.assertRaises(AttributeError):
            LogisticRegression.score(self.X_test, self.y_test)

    def test_score_trainable_op(self):
        from lale.lib.sklearn import LogisticRegression

        trainable = LogisticRegression()
        _ = trainable.fit(self.X_train, self.y_train)
        trainable.score(self.X_test, self.y_test)

    def test_score_trained_op(self):
        from sklearn.metrics import accuracy_score

        from lale.lib.sklearn import LogisticRegression

        trainable = LogisticRegression()
        trained_lr = trainable.fit(self.X_train, self.y_train)
        score = trained_lr.score(self.X_test, self.y_test)
        predictions = trained_lr.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions)
        self.assertEqual(score, accuracy)

    def test_score_trained_op_sample_wt(self):
        import numpy as np
        from sklearn.metrics import accuracy_score

        from lale.lib.sklearn import LogisticRegression

        trainable = LogisticRegression()
        trained_lr = trainable.fit(self.X_train, self.y_train)
        rng = np.random.RandomState(0)
        iris_weights = rng.randint(10, size=self.y_test.shape)
        score = trained_lr.score(self.X_test, self.y_test, sample_weight=iris_weights)
        predictions = trained_lr.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, predictions, sample_weight=iris_weights)
        self.assertEqual(score, accuracy)


class TestEmptyY(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris

        data = load_iris()
        self.X, self.y = data.data, data.target

    def test_PCA(self):
        op = PCA()
        op.fit(self.X, [])


class TestFitPlannedOp(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris

        data = load_iris()
        self.X, self.y = data.data, data.target

    def test_planned_individual_op(self):
        planned = LogisticRegression
        try:
            planned.fit(self.X, self.y)
        except AttributeError as e:
            self.assertEqual(
                e.__str__(),
                """Please use `LogisticRegression()` instead of `LogisticRegression` to make it trainable.
Alternatively, you could use `auto_configure(X, y, Hyperopt, max_evals=5)` on the operator to use Hyperopt for
`max_evals` iterations for hyperparameter tuning. `Hyperopt` can be imported as `from lale.lib.lale import Hyperopt`.""",
            )

    def test_planned_pipeline_with_choice(self):
        planned = PCA() >> (LogisticRegression() | KNeighborsClassifier())
        try:
            planned.fit(self.X, self.y)
        except AttributeError as e:
            self.assertEqual(
                e.__str__(),
                """The pipeline is not trainable, which means you can not call fit on it.

Suggested fixes:
Fix [A]: You can make the following changes in the pipeline in order to make it trainable:
[A.1] Please remove the operator choice `|` from `LogisticRegression | KNeighborsClassifier` and keep only one of those operators.

Fix [B]: Alternatively, you could use `auto_configure(X, y, Hyperopt, max_evals=5)` on the pipeline
to use Hyperopt for `max_evals` iterations for hyperparameter tuning. `Hyperopt` can be imported as `from lale.lib.lale import Hyperopt`.""",
            )

    def test_planned_pipeline_with_choice_1(self):
        planned = PCA >> (LogisticRegression() | KNeighborsClassifier())
        try:
            planned.fit(self.X, self.y)
        except AttributeError as e:
            self.maxDiff = None
            self.assertEqual(
                e.__str__(),
                """The pipeline is not trainable, which means you can not call fit on it.

Suggested fixes:
Fix [A]: You can make the following changes in the pipeline in order to make it trainable:
[A.1] Please use `PCA()` instead of `PCA.`
[A.2] Please remove the operator choice `|` from `LogisticRegression | KNeighborsClassifier` and keep only one of those operators.

Fix [B]: Alternatively, you could use `auto_configure(X, y, Hyperopt, max_evals=5)` on the pipeline
to use Hyperopt for `max_evals` iterations for hyperparameter tuning. `Hyperopt` can be imported as `from lale.lib.lale import Hyperopt`.""",
            )

    def test_choice(self):
        planned = LogisticRegression() | KNeighborsClassifier()
        try:
            planned.fit(self.X, self.y)
        except AttributeError as e:
            self.assertEqual(
                e.__str__(),
                """The pipeline is not trainable, which means you can not call fit on it.

Suggested fixes:
Fix [A]: You can make the following changes in the pipeline in order to make it trainable:
[A.1] Please remove the operator choice `|` from `LogisticRegression | KNeighborsClassifier` and keep only one of those operators.

Fix [B]: Alternatively, you could use `auto_configure(X, y, Hyperopt, max_evals=5)` on the pipeline
to use Hyperopt for `max_evals` iterations for hyperparameter tuning. `Hyperopt` can be imported as `from lale.lib.lale import Hyperopt`.""",
            )


class _OperatorForwardingTestWrappedImpl:
    def __init__(self):
        pass

    def fshadow(self):
        return False

    def finner(self):
        return True


class _OperatorForwardingTestImpl:
    def __init__(self):
        self._wrapped_model = _OperatorForwardingTestWrappedImpl()

        self.prop_ = True

    def fit(self, X, y=None):
        return self

    def f(self):
        return True

    def fshadow(self):
        return True

    def fnotforward(self):
        return True

    @property
    def p(self):
        return True

    def auto_(self):
        return True


_operator_forwarding_test_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "allOf": [{}],
}

_operator_forwarding_test_combined_schema = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "type": "object",
    "tags": {"pre": [], "op": [""], "post": []},
    "forwards": ["f", "p", "fshadow", "finner"],
    "properties": {
        "hyperparams": _operator_forwarding_test_schema,
        # "input_fit": None,
        # "input_transform": ,
        # "output_transform": _output_transform_schema,
    },
}


_OperatorForwardingTest = Ops.make_operator(
    _OperatorForwardingTestImpl, _operator_forwarding_test_combined_schema
)


class TestOperatorFowarding(unittest.TestCase):
    def test_fowards_method_list(self):
        self.assertEquals(
            _OperatorForwardingTest.get_forwards(),
            _operator_forwarding_test_combined_schema["forwards"],
        )

    def test_fowards_method_succeeds(self):
        self.assertTrue(_OperatorForwardingTest.f())

    def test_fowards_underscore_method_succeeds(self):
        self.assertTrue(_OperatorForwardingTest.auto_())

    def test_fowards_underscore_prop_succeeds(self):
        self.assertTrue(_OperatorForwardingTest.prop_)

    # test that the outer impl method is given priority over the inner impl method
    def test_fowards_method_shadow_succeeds(self):
        self.assertTrue(_OperatorForwardingTest.fshadow())

    def test_fowards_method_wrapped_succeeds(self):
        self.assertTrue(_OperatorForwardingTest.finner())

    def test_fowards_property_succeeds(self):
        self.assertTrue(_OperatorForwardingTest.p)

    def test_not_fowards_method(self):
        with self.assertRaises(AttributeError):
            self.assertTrue(_OperatorForwardingTest.fnotforward())

    def test_bad_forwards_decl(self):
        from test import EnableSchemaValidation

        _operator_forwarding_test_combined_schema2 = {
            "$schema": "http://json-schema.org/draft-04/schema#",
            "description": "Combined schema for expected data and hyperparameters.",
            "type": "object",
            "tags": {"pre": [], "op": [""], "post": []},
            "forwards": ["f", "p", "fshadow", "finner", "predict"],
            "properties": {
                "hyperparams": _operator_forwarding_test_schema,
                # "input_fit": None,
                # "input_transform": ,
                # "output_transform": _output_transform_schema,
            },
        }

        with self.assertRaises(AssertionError):
            with EnableSchemaValidation():
                Ops.make_operator(
                    _OperatorForwardingTestImpl,
                    _operator_forwarding_test_combined_schema2,
                )

    def test_bad_forwards_false_decl(self):

        _operator_forwarding_test_combined_schema2 = {
            "$schema": "http://json-schema.org/draft-04/schema#",
            "description": "Combined schema for expected data and hyperparameters.",
            "type": "object",
            "tags": {"pre": [], "op": [""], "post": []},
            "properties": {
                "hyperparams": _operator_forwarding_test_schema,
                # "input_fit": None,
                # "input_transform": ,
                # "output_transform": _output_transform_schema,
            },
        }
        _OperatorForwardingTest2 = Ops.make_operator(
            _OperatorForwardingTestImpl, _operator_forwarding_test_combined_schema2
        )
        with self.assertRaises(AttributeError):
            _OperatorForwardingTest2.f()

    def test_bad_forwards_true_decl(self):

        _operator_forwarding_test_combined_schema2 = {
            "$schema": "http://json-schema.org/draft-04/schema#",
            "description": "Combined schema for expected data and hyperparameters.",
            "type": "object",
            "tags": {"pre": [], "op": [""], "post": []},
            "forwards": True,
            "properties": {
                "hyperparams": _operator_forwarding_test_schema,
                # "input_fit": None,
                # "input_transform": ,
                # "output_transform": _output_transform_schema,
            },
        }
        _OperatorForwardingTest2 = Ops.make_operator(
            _OperatorForwardingTestImpl, _operator_forwarding_test_combined_schema2
        )
        _OperatorForwardingTest2.f()
        _OperatorForwardingTest2.fnotforward()

        with self.assertRaises(AttributeError):
            _OperatorForwardingTest2.unknown()

    def test_customize_schema_forward_success(self):
        Op = _OperatorForwardingTest.customize_schema(forwards=["fnotforward"])
        self.assertTrue(Op.fnotforward())

    def test_customize_schema_forward_failure(self):
        Op = _OperatorForwardingTest.customize_schema(forwards=["fnotforward"])
        with self.assertRaises(AttributeError):
            self.assertTrue(Op.f()())


class TestSteps(unittest.TestCase):
    def test_pipeline(self):
        pca = PCA()
        op: Ops.PlannedPipeline = pca >> LogisticRegression

        self.assertEqual(len(op.steps), 2)
        self.assertEquals(op.steps[0][0], "PCA")
        self.assertEquals(op.steps[0][1], pca)
