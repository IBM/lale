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

import pickle
import traceback
import typing
import unittest

import sklearn.datasets
import sklearn.pipeline
from sklearn.metrics import accuracy_score

import lale.datasets.openml
import lale.helpers
import lale.operators
from lale.helpers import import_from_sklearn_pipeline
from lale.lib.autogen import SGDClassifier
from lale.lib.lale import ConcatFeatures, NoOp
from lale.lib.sklearn import (
    PCA,
    GaussianNB,
    KNeighborsClassifier,
    LinearRegression,
    LinearSVC,
    LogisticRegression,
    Nystroem,
    OneHotEncoder,
    PassiveAggressiveClassifier,
    StandardScaler,
)
from lale.lib.xgboost import XGBClassifier


class TestCreation(unittest.TestCase):
    def setUp(self):
        from sklearn.model_selection import train_test_split

        data = sklearn.datasets.load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_pipeline_create(self):
        from lale.operators import Pipeline

        pipeline = Pipeline(([("pca1", PCA()), ("lr1", LogisticRegression())]))
        trained = pipeline.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)
        accuracy_score(self.y_test, predictions)

    def test_pipeline_create_trainable(self):
        import lale.lib.sklearn
        import lale.operators

        pipeline = lale.lib.sklearn.Pipeline(
            steps=[("pca1", PCA()), ("lr1", LogisticRegression())]
        )
        self.assertIsInstance(pipeline, lale.operators.TrainableIndividualOp)
        trained = pipeline.fit(self.X_train, self.y_train)
        pca_trained, lr_trained = [op for _, op in trained.hyperparams()["steps"]]
        self.assertIsInstance(pca_trained, lale.operators.TrainedIndividualOp)
        self.assertIsInstance(lr_trained, lale.operators.TrainedIndividualOp)
        predictions = trained.predict(self.X_test)
        accuracy_score(self.y_test, predictions)

    def test_pipeline_create_trained(self):
        import lale.lib.sklearn
        import lale.operators

        orig_trainable = PCA() >> LogisticRegression()
        orig_trained = orig_trainable.fit(self.X_train, self.y_train)
        self.assertIsInstance(orig_trained, lale.operators.TrainedPipeline)
        pca_trained, lr_trained = orig_trained.steps()
        pre_trained = lale.lib.sklearn.Pipeline(
            steps=[("pca1", pca_trained), ("lr1", lr_trained)]
        )
        self.assertIsInstance(pre_trained, lale.operators.TrainedIndividualOp)
        predictions = pre_trained.predict(self.X_test)
        accuracy_score(self.y_test, predictions)

    def test_pipeline_clone(self):
        from sklearn.base import clone

        from lale.operators import Pipeline

        pipeline = Pipeline(([("pca1", PCA()), ("lr1", LogisticRegression())]))
        trained = pipeline.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)
        orig_acc = accuracy_score(self.y_test, predictions)

        cloned_pipeline = clone(pipeline)
        trained = cloned_pipeline.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)
        cloned_acc = accuracy_score(self.y_test, predictions)
        self.assertEqual(orig_acc, cloned_acc)

    def test_make_pipeline(self):
        tfm = PCA(n_components=10)
        clf = LogisticRegression(random_state=42)
        trainable = lale.operators.make_pipeline(tfm, clf)
        digits = sklearn.datasets.load_digits()
        trained = trainable.fit(digits.data, digits.target)
        _ = trained.predict(digits.data)

    def test_compose2(self):
        tfm = PCA(n_components=10)
        clf = LogisticRegression(random_state=42)
        trainable = tfm >> clf
        digits = sklearn.datasets.load_digits()
        trained = trainable.fit(digits.data, digits.target)
        _ = trained.predict(digits.data)

    def test_compose3(self):
        nys = Nystroem(n_components=15)
        pca = PCA(n_components=10)
        lr = LogisticRegression(random_state=42)
        trainable = nys >> pca >> lr
        digits = sklearn.datasets.load_digits()
        trained = trainable.fit(digits.data, digits.target)
        _ = trained.predict(digits.data)

    def test_pca_nys_lr(self):
        from lale.operators import make_union

        nys = Nystroem(n_components=15)
        pca = PCA(n_components=10)
        lr = LogisticRegression(random_state=42)
        trainable = make_union(nys, pca) >> lr
        digits = sklearn.datasets.load_digits()
        trained = trainable.fit(digits.data, digits.target)
        _ = trained.predict(digits.data)

    def test_compose4(self):

        digits = sklearn.datasets.load_digits()
        _ = digits
        ohe = OneHotEncoder(handle_unknown=OneHotEncoder.enum.handle_unknown.ignore)
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
        _ = model_plan
        # TODO: optimize on this plan and then fit and predict

    def test_compose5(self):
        ohe = OneHotEncoder(handle_unknown=OneHotEncoder.enum.handle_unknown.ignore)
        digits = sklearn.datasets.load_digits()
        lr = LogisticRegression()
        lr_trained = lr.fit(digits.data, digits.target)
        lr_trained.predict(digits.data)
        pipeline1 = ohe >> lr
        pipeline1_trained = pipeline1.fit(digits.data, digits.target)
        pipeline1_trained.predict(digits.data)

    def test_compare_with_sklearn(self):
        tfm = PCA()
        clf = LogisticRegression(
            LogisticRegression.enum.solver.lbfgs,
            LogisticRegression.enum.multi_class.auto,
        )
        trainable = lale.operators.make_pipeline(tfm, clf)
        digits = sklearn.datasets.load_digits()
        trained = trainable.fit(digits.data, digits.target)
        predicted = trained.predict(digits.data)
        from sklearn.decomposition import PCA as SklearnPCA
        from sklearn.linear_model import LogisticRegression as SklearnLR

        sklearn_pipeline = sklearn.pipeline.make_pipeline(
            SklearnPCA(), SklearnLR(solver="lbfgs", multi_class="auto")
        )
        sklearn_pipeline.fit(digits.data, digits.target)
        predicted_sklearn = sklearn_pipeline.predict(digits.data)

        lale_score = accuracy_score(digits.target, predicted)
        scikit_score = accuracy_score(digits.target, predicted_sklearn)
        self.assertEqual(lale_score, scikit_score)


class TestImportExport(unittest.TestCase):
    def setUp(self):
        from sklearn.model_selection import train_test_split

        data = sklearn.datasets.load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    @classmethod
    def get_sklearn_params(cls, op):
        lale_sklearn_impl = op._impl_instance()
        wrapped_model = getattr(lale_sklearn_impl, "_wrapped_model", None)
        if wrapped_model is not None:
            lale_sklearn_impl = wrapped_model
        return lale_sklearn_impl.get_params()

    def assert_equal_predictions(self, pipeline1, pipeline2):
        trained = pipeline1.fit(self.X_train, self.y_train)
        predictions1 = trained.predict(self.X_test)

        trained = pipeline2.fit(self.X_train, self.y_train)
        predictions2 = trained.predict(self.X_test)
        [self.assertEqual(p1, predictions2[i]) for i, p1 in enumerate(predictions1)]

    def test_import_from_sklearn_pipeline(self):
        from sklearn.feature_selection import SelectKBest, f_regression
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC as SklearnSVC

        anova_filter = SelectKBest(f_regression, k=3)
        clf = SklearnSVC(kernel="linear")
        sklearn_pipeline = Pipeline([("anova", anova_filter), ("svc", clf)])
        lale_pipeline = typing.cast(
            lale.operators.TrainablePipeline,
            import_from_sklearn_pipeline(sklearn_pipeline),
        )
        for i, pipeline_step in enumerate(sklearn_pipeline.named_steps):
            sklearn_step_params = sklearn_pipeline.named_steps[
                pipeline_step
            ].get_params()
            lale_sklearn_params = self.get_sklearn_params(lale_pipeline.steps()[i])
            self.assertEqual(sklearn_step_params, lale_sklearn_params)
        self.assert_equal_predictions(sklearn_pipeline, lale_pipeline)

    def test_import_from_sklearn_pipeline1(self):
        from sklearn.decomposition import PCA as SklearnPCA
        from sklearn.neighbors import KNeighborsClassifier as SklearnKNN

        sklearn_pipeline = sklearn.pipeline.make_pipeline(
            SklearnPCA(n_components=3), SklearnKNN()
        )
        lale_pipeline = typing.cast(
            lale.operators.TrainablePipeline,
            import_from_sklearn_pipeline(sklearn_pipeline),
        )
        for i, pipeline_step in enumerate(sklearn_pipeline.named_steps):
            sklearn_step_params = sklearn_pipeline.named_steps[
                pipeline_step
            ].get_params()
            lale_sklearn_params = self.get_sklearn_params(lale_pipeline.steps()[i])
            self.assertEqual(sklearn_step_params, lale_sklearn_params)
        self.assert_equal_predictions(sklearn_pipeline, lale_pipeline)

    def test_import_from_sklearn_pipeline_feature_union(self):
        from sklearn.decomposition import PCA as SklearnPCA
        from sklearn.kernel_approximation import Nystroem as SklearnNystroem
        from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
        from sklearn.pipeline import FeatureUnion

        union = FeatureUnion(
            [
                ("pca", SklearnPCA(n_components=1)),
                ("nys", SklearnNystroem(n_components=2, random_state=42)),
            ]
        )
        sklearn_pipeline = sklearn.pipeline.make_pipeline(union, SklearnKNN())
        lale_pipeline = typing.cast(
            lale.operators.TrainablePipeline,
            import_from_sklearn_pipeline(sklearn_pipeline),
        )
        self.assertEqual(len(lale_pipeline.edges()), 3)
        from lale.lib.lale.concat_features import ConcatFeatures
        from lale.lib.sklearn.k_neighbors_classifier import KNeighborsClassifier
        from lale.lib.sklearn.nystroem import Nystroem
        from lale.lib.sklearn.pca import PCA

        self.assertIsInstance(lale_pipeline.edges()[0][0], PCA)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[0][1], ConcatFeatures)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[1][0], Nystroem)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[1][1], ConcatFeatures)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[2][0], ConcatFeatures)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[2][1], KNeighborsClassifier)  # type: ignore
        self.assert_equal_predictions(sklearn_pipeline, lale_pipeline)

    def test_import_from_sklearn_pipeline_nested_pipeline(self):
        from sklearn.decomposition import PCA as SklearnPCA
        from sklearn.feature_selection import SelectKBest
        from sklearn.kernel_approximation import Nystroem as SklearnNystroem
        from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
        from sklearn.pipeline import FeatureUnion

        union = FeatureUnion(
            [
                (
                    "selectkbest_pca",
                    sklearn.pipeline.make_pipeline(
                        SelectKBest(k=3), SklearnPCA(n_components=1)
                    ),
                ),
                ("nys", SklearnNystroem(n_components=2, random_state=42)),
            ]
        )
        sklearn_pipeline = sklearn.pipeline.make_pipeline(union, SklearnKNN())
        lale_pipeline = typing.cast(
            lale.operators.TrainablePipeline,
            import_from_sklearn_pipeline(sklearn_pipeline),
        )
        self.assertEqual(len(lale_pipeline.edges()), 4)
        from lale.lib.lale.concat_features import ConcatFeatures
        from lale.lib.sklearn.k_neighbors_classifier import KNeighborsClassifier
        from lale.lib.sklearn.nystroem import Nystroem
        from lale.lib.sklearn.pca import PCA
        from lale.lib.sklearn.select_k_best import SelectKBest

        # These assertions assume topological sort
        self.assertIsInstance(lale_pipeline.edges()[0][0], SelectKBest)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[0][1], PCA)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[1][0], PCA)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[1][1], ConcatFeatures)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[2][0], Nystroem)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[2][1], ConcatFeatures)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[3][0], ConcatFeatures)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[3][1], KNeighborsClassifier)  # type: ignore
        self.assert_equal_predictions(sklearn_pipeline, lale_pipeline)

    def test_import_from_sklearn_pipeline_nested_pipeline1(self):
        from sklearn.decomposition import PCA as SklearnPCA
        from sklearn.feature_selection import SelectKBest
        from sklearn.kernel_approximation import Nystroem as SklearnNystroem
        from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
        from sklearn.pipeline import FeatureUnion

        union = FeatureUnion(
            [
                (
                    "selectkbest_pca",
                    sklearn.pipeline.make_pipeline(
                        SelectKBest(k=3),
                        FeatureUnion(
                            [
                                ("pca", SklearnPCA(n_components=1)),
                                (
                                    "nested_pipeline",
                                    sklearn.pipeline.make_pipeline(
                                        SelectKBest(k=2), SklearnNystroem()
                                    ),
                                ),
                            ]
                        ),
                    ),
                ),
                ("nys", SklearnNystroem(n_components=2, random_state=42)),
            ]
        )
        sklearn_pipeline = sklearn.pipeline.make_pipeline(union, SklearnKNN())
        lale_pipeline = typing.cast(
            lale.operators.TrainablePipeline,
            import_from_sklearn_pipeline(sklearn_pipeline),
        )
        self.assertEqual(len(lale_pipeline.edges()), 8)
        # These assertions assume topological sort, which may not be unique. So the assertions are brittle.
        from lale.lib.lale.concat_features import ConcatFeatures
        from lale.lib.sklearn.k_neighbors_classifier import KNeighborsClassifier
        from lale.lib.sklearn.nystroem import Nystroem
        from lale.lib.sklearn.pca import PCA
        from lale.lib.sklearn.select_k_best import SelectKBest

        self.assertIsInstance(lale_pipeline.edges()[0][0], SelectKBest)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[0][1], PCA)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[1][0], SelectKBest)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[1][1], SelectKBest)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[2][0], SelectKBest)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[2][1], Nystroem)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[3][0], PCA)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[3][1], ConcatFeatures)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[4][0], Nystroem)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[4][1], ConcatFeatures)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[5][0], ConcatFeatures)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[5][1], ConcatFeatures)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[6][0], Nystroem)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[6][1], ConcatFeatures)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[7][0], ConcatFeatures)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[7][1], KNeighborsClassifier)  # type: ignore
        self.assert_equal_predictions(sklearn_pipeline, lale_pipeline)

    def test_import_from_sklearn_pipeline_nested_pipeline2(self):
        from sklearn.decomposition import PCA as SklearnPCA
        from sklearn.feature_selection import SelectKBest
        from sklearn.kernel_approximation import Nystroem as SklearnNystroem
        from sklearn.neighbors import KNeighborsClassifier as SklearnKNN
        from sklearn.pipeline import FeatureUnion

        union = FeatureUnion(
            [
                (
                    "selectkbest_pca",
                    sklearn.pipeline.make_pipeline(
                        SelectKBest(k=3),
                        sklearn.pipeline.make_pipeline(SelectKBest(k=2), SklearnPCA()),
                    ),
                ),
                ("nys", SklearnNystroem(n_components=2, random_state=42)),
            ]
        )
        sklearn_pipeline = sklearn.pipeline.make_pipeline(union, SklearnKNN())
        lale_pipeline = typing.cast(
            lale.operators.TrainablePipeline,
            import_from_sklearn_pipeline(sklearn_pipeline),
        )
        self.assertEqual(len(lale_pipeline.edges()), 5)
        from lale.lib.lale.concat_features import ConcatFeatures
        from lale.lib.sklearn.k_neighbors_classifier import KNeighborsClassifier
        from lale.lib.sklearn.nystroem import Nystroem
        from lale.lib.sklearn.pca import PCA
        from lale.lib.sklearn.select_k_best import SelectKBest

        self.assertIsInstance(lale_pipeline.edges()[0][0], SelectKBest)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[0][1], SelectKBest)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[1][0], SelectKBest)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[1][1], PCA)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[2][0], PCA)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[2][1], ConcatFeatures)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[3][0], Nystroem)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[3][1], ConcatFeatures)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[4][0], ConcatFeatures)  # type: ignore
        self.assertIsInstance(lale_pipeline.edges()[4][1], KNeighborsClassifier)  # type: ignore

        self.assert_equal_predictions(sklearn_pipeline, lale_pipeline)

    def test_import_from_sklearn_pipeline_noop(self):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.pipeline import Pipeline

        from lale.helpers import import_from_sklearn_pipeline

        pipe = Pipeline([("noop", None), ("gbc", GradientBoostingClassifier())])
        with self.assertRaises(ValueError):
            _ = import_from_sklearn_pipeline(pipe)

    def test_import_from_sklearn_pipeline_noop1(self):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.pipeline import Pipeline

        from lale.helpers import import_from_sklearn_pipeline

        pipe = Pipeline([("noop", NoOp()), ("gbc", GradientBoostingClassifier())])
        _ = import_from_sklearn_pipeline(pipe)

    def test_export_to_sklearn_pipeline(self):
        lale_pipeline = PCA(n_components=3) >> KNeighborsClassifier()
        trained_lale_pipeline = lale_pipeline.fit(self.X_train, self.y_train)
        sklearn_pipeline = trained_lale_pipeline.export_to_sklearn_pipeline()
        for i, pipeline_step in enumerate(sklearn_pipeline.named_steps):
            sklearn_step_params = sklearn_pipeline.named_steps[
                pipeline_step
            ].get_params()
            lale_sklearn_params = self.get_sklearn_params(
                trained_lale_pipeline.steps()[i]
            )
            self.assertEqual(sklearn_step_params, lale_sklearn_params)
        self.assert_equal_predictions(sklearn_pipeline, trained_lale_pipeline)

    def test_export_to_sklearn_pipeline1(self):
        from sklearn.feature_selection import SelectKBest

        lale_pipeline = SelectKBest(k=3) >> KNeighborsClassifier()
        trained_lale_pipeline = lale_pipeline.fit(self.X_train, self.y_train)
        sklearn_pipeline = trained_lale_pipeline.export_to_sklearn_pipeline()
        for i, pipeline_step in enumerate(sklearn_pipeline.named_steps):
            sklearn_step_params = type(sklearn_pipeline.named_steps[pipeline_step])
            lale_sklearn_params = (
                type(trained_lale_pipeline.steps()[i]._impl._wrapped_model)
                if hasattr(trained_lale_pipeline.steps()[i]._impl, "_wrapped_model")
                else type(trained_lale_pipeline.steps()[i]._impl)
            )
            self.assertEqual(sklearn_step_params, lale_sklearn_params)
        self.assert_equal_predictions(sklearn_pipeline, trained_lale_pipeline)

    def test_export_to_sklearn_pipeline2(self):
        from sklearn.feature_selection import SelectKBest
        from sklearn.pipeline import FeatureUnion

        lale_pipeline = (
            (
                (
                    (PCA(svd_solver="randomized", random_state=42) & SelectKBest(k=3))
                    >> ConcatFeatures()
                )
                & Nystroem(random_state=42)
            )
            >> ConcatFeatures()
            >> KNeighborsClassifier()
        )
        trained_lale_pipeline = lale_pipeline.fit(self.X_train, self.y_train)
        sklearn_pipeline = trained_lale_pipeline.export_to_sklearn_pipeline()
        self.assertIsInstance(
            sklearn_pipeline.named_steps["featureunion"], FeatureUnion
        )
        from sklearn.neighbors import KNeighborsClassifier as SklearnKNN

        self.assertIsInstance(
            sklearn_pipeline.named_steps["kneighborsclassifier"], SklearnKNN
        )
        self.assert_equal_predictions(sklearn_pipeline, trained_lale_pipeline)

    def test_export_to_sklearn_pipeline3(self):
        from sklearn.feature_selection import SelectKBest
        from sklearn.pipeline import FeatureUnion

        lale_pipeline = (
            (
                (PCA() >> SelectKBest(k=2))
                & (Nystroem(random_state=42) >> SelectKBest(k=3))
                & (SelectKBest(k=3))
            )
            >> ConcatFeatures()
            >> SelectKBest(k=2)
            >> LogisticRegression()
        )
        trained_lale_pipeline = lale_pipeline.fit(self.X_train, self.y_train)
        sklearn_pipeline = trained_lale_pipeline.export_to_sklearn_pipeline()
        self.assertIsInstance(
            sklearn_pipeline.named_steps["featureunion"], FeatureUnion
        )
        self.assertIsInstance(sklearn_pipeline.named_steps["selectkbest"], SelectKBest)
        from sklearn.linear_model import LogisticRegression as SklearnLR

        self.assertIsInstance(
            sklearn_pipeline.named_steps["logisticregression"], SklearnLR
        )
        self.assert_equal_predictions(sklearn_pipeline, trained_lale_pipeline)

    def test_export_to_sklearn_pipeline4(self):
        lale_pipeline = lale.operators.make_pipeline(LogisticRegression())
        trained_lale_pipeline = lale_pipeline.fit(self.X_train, self.y_train)
        sklearn_pipeline = trained_lale_pipeline.export_to_sklearn_pipeline()
        from sklearn.linear_model import LogisticRegression as SklearnLR

        self.assertIsInstance(
            sklearn_pipeline.named_steps["logisticregression"], SklearnLR
        )
        self.assert_equal_predictions(sklearn_pipeline, trained_lale_pipeline)

    def test_export_to_sklearn_pipeline5(self):
        lale_pipeline = PCA() >> (XGBClassifier() | SGDClassifier())
        with self.assertRaises(ValueError):
            _ = lale_pipeline.export_to_sklearn_pipeline()

    def test_export_to_pickle(self):
        lale_pipeline = lale.operators.make_pipeline(LogisticRegression())
        trained_lale_pipeline = lale_pipeline.fit(self.X_train, self.y_train)
        pickle.dumps(lale_pipeline)
        pickle.dumps(trained_lale_pipeline)

    def test_import_from_sklearn_pipeline2(self):
        from sklearn.feature_selection import SelectKBest, f_regression
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC as SklearnSVC

        anova_filter = SelectKBest(f_regression, k=3)
        clf = SklearnSVC(kernel="linear")
        sklearn_pipeline = Pipeline([("anova", anova_filter), ("svc", clf)])
        sklearn_pipeline.fit(self.X_train, self.y_train)
        lale_pipeline = typing.cast(
            lale.operators.TrainedPipeline,
            import_from_sklearn_pipeline(sklearn_pipeline),
        )
        lale_pipeline.predict(self.X_test)

    def test_import_from_sklearn_pipeline3(self):
        from sklearn.feature_selection import SelectKBest, f_regression
        from sklearn.pipeline import Pipeline
        from sklearn.svm import SVC as SklearnSVC

        anova_filter = SelectKBest(f_regression, k=3)
        clf = SklearnSVC(kernel="linear")
        sklearn_pipeline = Pipeline([("anova", anova_filter), ("svc", clf)])
        lale_pipeline = typing.cast(
            lale.operators.TrainablePipeline,
            import_from_sklearn_pipeline(sklearn_pipeline, fitted=False),
        )
        with self.assertRaises(
            ValueError
        ):  # fitted=False returns a Trainable, so calling predict is invalid.
            lale_pipeline.predict(self.X_test)

    def test_export_to_sklearn_pipeline_with_noop_1(self):
        lale_pipeline = NoOp() >> PCA(n_components=3) >> KNeighborsClassifier()
        trained_lale_pipeline = lale_pipeline.fit(self.X_train, self.y_train)
        sklearn_pipeline = trained_lale_pipeline.export_to_sklearn_pipeline()
        self.assert_equal_predictions(sklearn_pipeline, trained_lale_pipeline)

    def test_export_to_sklearn_pipeline_with_noop_2(self):
        lale_pipeline = PCA(n_components=3) >> NoOp() >> KNeighborsClassifier()
        trained_lale_pipeline = lale_pipeline.fit(self.X_train, self.y_train)
        sklearn_pipeline = trained_lale_pipeline.export_to_sklearn_pipeline()
        self.assert_equal_predictions(sklearn_pipeline, trained_lale_pipeline)

    def test_export_to_sklearn_pipeline_with_noop_3(self):
        # This test is probably unnecessary, but doesn't harm at this point
        lale_pipeline = PCA(n_components=3) >> KNeighborsClassifier() >> NoOp()
        trained_lale_pipeline = lale_pipeline.fit(self.X_train, self.y_train)
        _ = trained_lale_pipeline.export_to_sklearn_pipeline()

    def test_export_to_sklearn_pipeline_with_noop_4(self):
        lale_pipeline = NoOp() >> KNeighborsClassifier()
        trained_lale_pipeline = lale_pipeline.fit(self.X_train, self.y_train)
        sklearn_pipeline = trained_lale_pipeline.export_to_sklearn_pipeline()
        self.assert_equal_predictions(sklearn_pipeline, trained_lale_pipeline)


class TestComposition(unittest.TestCase):
    def setUp(self):
        from sklearn.model_selection import train_test_split

        data = sklearn.datasets.load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_two_estimators_predict(self):
        pipeline = (
            StandardScaler()
            >> (PCA() & Nystroem() & LogisticRegression())
            >> ConcatFeatures()
            >> NoOp()
            >> LogisticRegression()
        )
        trained = pipeline.fit(self.X_train, self.y_train)
        trained.predict(self.X_test)

    def test_two_estimators_predict1(self):
        pipeline = (
            StandardScaler()
            >> (PCA() & Nystroem() & PassiveAggressiveClassifier())
            >> ConcatFeatures()
            >> NoOp()
            >> PassiveAggressiveClassifier()
        )
        trained = pipeline.fit(self.X_train, self.y_train)
        trained.predict(self.X_test)

    def test_two_estimators_predict_proba(self):
        pipeline = (
            StandardScaler()
            >> (PCA() & Nystroem() & LogisticRegression())
            >> ConcatFeatures()
            >> NoOp()
            >> LogisticRegression()
        )
        trained = pipeline.fit(self.X_train, self.y_train)
        trained.predict_proba(self.X_test)

    def test_two_estimators_predict_proba1(self):
        pipeline = (
            StandardScaler()
            >> (PCA() & Nystroem() & GaussianNB())
            >> ConcatFeatures()
            >> NoOp()
            >> GaussianNB()
        )
        pipeline.fit(self.X_train, self.y_train)
        pipeline.predict_proba(self.X_test)

    def test_multiple_estimators_predict_predict_proba(self):
        pipeline = (
            StandardScaler()
            >> (LogisticRegression() & PCA())
            >> ConcatFeatures()
            >> (NoOp() & LinearSVC())
            >> ConcatFeatures()
            >> KNeighborsClassifier()
        )
        pipeline.fit(self.X_train, self.y_train)
        _ = pipeline.predict_proba(self.X_test)
        _ = pipeline.predict(self.X_test)

    def test_two_transformers(self):
        tfm1 = PCA()
        tfm2 = Nystroem()
        trainable = tfm1 >> tfm2
        digits = sklearn.datasets.load_digits()
        trained = trainable.fit(digits.data, digits.target)
        _ = trained.transform(digits.data)

    def test_duplicate_instances(self):
        tfm = PCA()
        clf = LogisticRegression(
            LogisticRegression.enum.solver.lbfgs,
            LogisticRegression.enum.multi_class.auto,
        )
        with self.assertRaises(ValueError):
            _ = lale.operators.make_pipeline(tfm, tfm, clf)

    def test_increase_num_rows(self):
        from test.mock_custom_operators import IncreaseRows

        increase_rows = IncreaseRows()
        trainable = increase_rows >> LogisticRegression()
        iris = sklearn.datasets.load_iris()
        X, y = iris.data, iris.target

        trained = trainable.fit(X, y)
        _ = trained.predict(X)

    def test_remove_last1(self):
        pipeline = (
            StandardScaler()
            >> (PCA() & Nystroem() & PassiveAggressiveClassifier())
            >> ConcatFeatures()
            >> NoOp()
            >> PassiveAggressiveClassifier()
        )
        new_pipeline = pipeline.remove_last()
        self.assertEqual(len(new_pipeline._steps), 6)
        self.assertEqual(len(pipeline._steps), 7)

    def test_remove_last2(self):
        pipeline = (
            StandardScaler()
            >> (PCA() & Nystroem() & PassiveAggressiveClassifier())
            >> ConcatFeatures()
            >> NoOp()
            >> (PassiveAggressiveClassifier() & LogisticRegression())
        )
        with self.assertRaises(ValueError):
            pipeline.remove_last()

    def test_remove_last3(self):
        pipeline = (
            StandardScaler()
            >> (PCA() & Nystroem() & PassiveAggressiveClassifier())
            >> ConcatFeatures()
            >> NoOp()
            >> PassiveAggressiveClassifier()
        )
        pipeline.remove_last().freeze_trainable()

    def test_remove_last4(self):
        pipeline = (
            StandardScaler()
            >> (PCA() & Nystroem() & PassiveAggressiveClassifier())
            >> ConcatFeatures()
            >> NoOp()
            >> PassiveAggressiveClassifier()
        )
        new_pipeline = pipeline.remove_last(inplace=True)
        self.assertEqual(len(new_pipeline._steps), 6)
        self.assertEqual(len(pipeline._steps), 6)

    def test_remove_last5(self):
        pipeline = (
            StandardScaler()
            >> (PCA() & Nystroem() & PassiveAggressiveClassifier())
            >> ConcatFeatures()
            >> NoOp()
            >> PassiveAggressiveClassifier()
        )
        pipeline.remove_last(inplace=True).freeze_trainable()


class TestAutoPipeline(unittest.TestCase):
    def _fit_predict(self, prediction_type, all_X, all_y, verbose=True):
        import sklearn.metrics
        import sklearn.model_selection

        if verbose:
            file_name, line, fn_name, text = traceback.extract_stack()[-2]
            print(f"--- TestAutoPipeline.{fn_name}() ---")
        from lale.lib.lale import AutoPipeline

        train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(
            all_X, all_y
        )
        trainable = AutoPipeline(
            prediction_type=prediction_type, max_evals=10, verbose=verbose
        )
        trained = trainable.fit(train_X, train_y)
        predicted = trained.predict(test_X)
        if prediction_type == "regression":
            score = f"r2 score {sklearn.metrics.r2_score(test_y, predicted):.2f}"
        else:
            score = f"accuracy {sklearn.metrics.accuracy_score(test_y, predicted):.1%}"
        if verbose:
            print(score)
            print(trained.get_pipeline().pretty_print(show_imports=False))

    def test_sklearn_iris(self):
        # classification, only numbers, no missing values
        all_X, all_y = sklearn.datasets.load_iris(return_X_y=True)
        self._fit_predict("classification", all_X, all_y)

    def test_sklearn_digits(self):
        # classification, numbers but some appear categorical, no missing values
        all_X, all_y = sklearn.datasets.load_digits(return_X_y=True)
        self._fit_predict("classification", all_X, all_y)

    def test_sklearn_boston(self):
        # regression, categoricals+numbers, no missing values
        all_X, all_y = sklearn.datasets.load_boston(return_X_y=True)
        self._fit_predict("regression", all_X, all_y)

    def test_sklearn_diabetes(self):
        # regression, categoricals+numbers, no missing values
        all_X, all_y = sklearn.datasets.load_diabetes(return_X_y=True)
        self._fit_predict("regression", all_X, all_y)

    def test_openml_creditg(self):
        import sklearn.model_selection

        # classification, categoricals+numbers incl. string, no missing values
        (orig_train_X, orig_train_y), _ = lale.datasets.openml.fetch(
            "credit-g", "classification", preprocess=False
        )
        subsample_X, _, subsample_y, _ = sklearn.model_selection.train_test_split(
            orig_train_X, orig_train_y, train_size=0.05
        )
        self._fit_predict("classification", subsample_X, subsample_y)

    def test_missing_iris(self):
        # classification, only numbers, synthetically added missing values
        all_X, all_y = sklearn.datasets.load_iris(return_X_y=True)
        with_missing_X = lale.helpers.add_missing_values(all_X)
        with self.assertRaisesRegex(ValueError, "Input contains NaN"):
            lr_trainable = LogisticRegression()
            _ = lr_trainable.fit(with_missing_X, all_y)
        self._fit_predict("classification", with_missing_X, all_y)

    def test_missing_boston(self):
        # regression, categoricals+numbers, synthetically added missing values
        all_X, all_y = sklearn.datasets.load_boston(return_X_y=True)
        with_missing_X = lale.helpers.add_missing_values(all_X)
        with self.assertRaisesRegex(ValueError, "Input contains NaN"):
            lr_trainable = LinearRegression()
            _ = lr_trainable.fit(with_missing_X, all_y)
        self._fit_predict("regression", with_missing_X, all_y)

    def test_missing_creditg(self):
        import sklearn.model_selection

        # classification, categoricals+numbers incl. string, synth. missing
        (orig_train_X, orig_train_y), _ = lale.datasets.openml.fetch(
            "credit-g", "classification", preprocess=False
        )
        subsample_X, _, subsample_y, _ = sklearn.model_selection.train_test_split(
            orig_train_X, orig_train_y, train_size=0.05
        )
        with_missing_X = lale.helpers.add_missing_values(subsample_X)
        self._fit_predict("classification", with_missing_X, subsample_y)


class TestOperatorChoice(unittest.TestCase):
    def test_make_choice_with_instance(self):
        from sklearn.datasets import load_iris

        from lale.operators import make_choice

        iris = load_iris()
        X, y = iris.data, iris.target
        tfm = PCA() | Nystroem() | NoOp()
        with self.assertRaises(AttributeError):
            # we are trying to trigger a runtime error here, so we ignore the static warning
            _ = tfm.fit(X, y)  # type: ignore
        _ = (OneHotEncoder | NoOp) >> tfm >> (LogisticRegression | KNeighborsClassifier)
        _ = (
            (OneHotEncoder | NoOp)
            >> (PCA | Nystroem)
            >> (LogisticRegression | KNeighborsClassifier)
        )
        _ = (
            make_choice(OneHotEncoder, NoOp)
            >> make_choice(PCA, Nystroem)
            >> make_choice(LogisticRegression, KNeighborsClassifier)
        )


class TestScore(unittest.TestCase):
    def setUp(self):
        from sklearn.model_selection import train_test_split

        data = sklearn.datasets.load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_trained_pipeline(self):
        trainable_pipeline = StandardScaler() >> LogisticRegression()
        trained_pipeline = trainable_pipeline.fit(self.X_train, self.y_train)
        score = trained_pipeline.score(self.X_test, self.y_test)
        predictions = trained_pipeline.predict(self.X_test)
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(self.y_test, predictions)
        self.assertEqual(accuracy, score)

    def test_trainable_pipeline(self):
        trainable_pipeline = StandardScaler() >> LogisticRegression()
        trainable_pipeline.fit(self.X_train, self.y_train)
        score = trainable_pipeline.score(self.X_test, self.y_test)
        predictions = trainable_pipeline.predict(self.X_test)
        from sklearn.metrics import accuracy_score

        accuracy = accuracy_score(self.y_test, predictions)
        self.assertEqual(accuracy, score)

    def test_planned_pipeline(self):
        planned_pipeline = StandardScaler >> LogisticRegression
        with self.assertRaises(AttributeError):
            planned_pipeline.score(self.X_test, self.y_test)  # type: ignore
