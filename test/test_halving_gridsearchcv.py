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

from lale.lib.lale import HalvingGridSearchCV
from lale.lib.sklearn import (
    PCA,
    KNeighborsClassifier,
    KNeighborsRegressor,
    LinearRegression,
    LogisticRegression,
    MinMaxScaler,
    Normalizer,
    StandardScaler,
)


class TestAutoConfigureClassification(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_with_halving_gridsearchcv(self):
        from lale.lib.lale import HalvingGridSearchCV, NoOp
        from lale.lib.sklearn import PCA, LogisticRegression

        warnings.simplefilter("ignore")
        planned_pipeline = (PCA | NoOp) >> LogisticRegression
        best_pipeline = planned_pipeline.auto_configure(
            self.X_train,
            self.y_train,
            optimizer=HalvingGridSearchCV,
            cv=3,
            scoring="accuracy",
            lale_num_samples=1,
            lale_num_grids=1,
        )
        _ = best_pipeline.predict(self.X_test)
        assert best_pipeline is not None

    def test_runtime_limit_hoc(self):
        import time

        planned_pipeline = (MinMaxScaler | Normalizer) >> (
            LogisticRegression | KNeighborsClassifier
        )
        from sklearn.datasets import load_iris

        X, y = load_iris(return_X_y=True)

        max_opt_time = 2.0
        hoc = HalvingGridSearchCV(
            estimator=planned_pipeline,
            cv=3,
            scoring="accuracy",
            max_opt_time=max_opt_time,
        )
        start = time.time()
        with self.assertRaises(BaseException):
            _ = hoc.fit(X, y)
        end = time.time()
        opt_time = end - start
        rel_diff = (opt_time - max_opt_time) / max_opt_time
        assert (
            rel_diff < 0.7
        ), "Max time: {}, Actual time: {}, relative diff: {}".format(
            max_opt_time, opt_time, rel_diff
        )

    def test_runtime_limit_hor(self):
        import time

        planned_pipeline = (MinMaxScaler | Normalizer) >> LinearRegression
        from sklearn.datasets import load_boston

        X, y = load_boston(return_X_y=True)

        max_opt_time = 3.0
        hor = HalvingGridSearchCV(
            estimator=planned_pipeline,
            cv=3,
            max_opt_time=max_opt_time,
            scoring="r2",
        )
        start = time.time()
        with self.assertRaises(BaseException):
            _ = hor.fit(X[:500, :], y[:500])
        end = time.time()
        opt_time = end - start
        rel_diff = (opt_time - max_opt_time) / max_opt_time
        assert (
            rel_diff < 0.2
        ), "Max time: {}, Actual time: {}, relative diff: {}".format(
            max_opt_time, opt_time, rel_diff
        )


class TestGridSearchCV(unittest.TestCase):
    def test_manual_grid(self):
        from sklearn.datasets import load_iris

        from lale.lib.lale import HalvingGridSearchCV
        from lale.lib.sklearn import SVC

        warnings.simplefilter("ignore")

        from lale import wrap_imported_operators

        wrap_imported_operators()
        iris = load_iris()
        parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
        svc = SVC()
        clf = HalvingGridSearchCV(estimator=svc, param_grid=parameters)
        clf.fit(iris.data, iris.target)
        clf.predict(iris.data)

    @unittest.skip("Currently flaky")
    def test_with_halving_gridsearchcv_auto_wrapped_pipe1(self):
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer

        lr = LogisticRegression()
        pca = PCA()
        trainable = pca >> lr

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from lale.lib.lale import HalvingGridSearchCV

            clf = HalvingGridSearchCV(
                estimator=trainable,
                lale_num_samples=2,
                lale_num_grids=2,
                cv=2,
                scoring=make_scorer(accuracy_score),
            )
            iris = load_iris()
            clf.fit(iris.data, iris.target)

    @unittest.skip("Currently flaky")
    def test_with_halving_gridsearchcv_auto_wrapped_pipe2(self):
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
            from lale.lib.lale import HalvingGridSearchCV

            clf = HalvingGridSearchCV(
                estimator=trainable,
                lale_num_samples=2,
                lale_num_grids=3,
                cv=2,
                scoring=make_scorer(accuracy_score),
            )
            iris = load_iris()
            clf.fit(iris.data, iris.target)


class TestKNeighborsRegressor(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_diabetes
        from sklearn.model_selection import train_test_split

        all_X, all_y = load_diabetes(return_X_y=True)
        # 15 samples, small enough so folds are likely smaller than n_neighbors
        self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(
            all_X, all_y, train_size=15, test_size=None, shuffle=True, random_state=42
        )

    def test_halving_gridsearch(self):
        planned = KNeighborsRegressor
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trained = planned.auto_configure(
                self.train_X,
                self.train_y,
                optimizer=HalvingGridSearchCV,
                cv=3,
                scoring="r2",
            )
        _ = trained.predict(self.test_X)


class TestStandardScaler(unittest.TestCase):
    def setUp(self):
        import scipy.sparse
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        # from lale.datasets.data_schemas import add_schema
        all_X, all_y = load_iris(return_X_y=True)
        denseTrainX, self.test_X, self.train_y, self.test_y = train_test_split(
            all_X, all_y, train_size=0.8, test_size=0.2, shuffle=True, random_state=42
        )
        # self.train_X = add_schema(scipy.sparse.csr_matrix(denseTrainX))
        self.train_X = scipy.sparse.csr_matrix(denseTrainX)

    def test_halving_gridsearch(self):
        planned = StandardScaler >> LogisticRegression().freeze_trainable()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trained = planned.auto_configure(
                self.train_X,
                self.train_y,
                optimizer=HalvingGridSearchCV,
                cv=3,
                scoring="r2",
            )
        _ = trained.predict(self.test_X)
