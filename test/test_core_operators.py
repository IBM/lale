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

import inspect
import io
import logging
import unittest
import warnings

import jsonschema
import sklearn.datasets

import lale.lib.lale
import lale.operators as Ops
import lale.type_checking
from lale.expressions import count, it
from lale.lib.lale import Aggregate, ConcatFeatures, Join, NoOp, Relational, Scan
from lale.lib.sklearn import (
    NMF,
    PCA,
    RFE,
    FunctionTransformer,
    KNeighborsClassifier,
    LogisticRegression,
    MissingIndicator,
    MLPClassifier,
    Nystroem,
    OneHotEncoder,
    RandomForestClassifier,
    RidgeClassifier,
    TfidfVectorizer,
)
from lale.search.lale_grid_search_cv import get_grid_search_parameter_grids
from lale.sklearn_compat import make_sklearn_compat


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

        from lale.lib.sklearn.one_hot_encoder import OneHotEncoderImpl

        if fproc._impl_class() == OneHotEncoderImpl:
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

        svm = sklearn.svm.SVR(kernel="linear")
        rfe = RFE(estimator=svm, n_features_to_select=2)
        lr = LogisticRegression()
        trainable = rfe >> lr
        data = sklearn.datasets.load_iris()
        X, y = data.data, data.target
        trained = trainable.fit(X, y)
        _ = trained.predict(X)

    def test_not_operator(self):
        with self.assertRaises(jsonschema.ValidationError):
            _ = RFE(estimator='"not an operator"', n_features_to_select=2)

    def test_attrib_sklearn(self):
        import sklearn.datasets

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


class TestMap(unittest.TestCase):
    def test_init(self):
        from lale.expressions import it, replace
        from lale.lib.lale import Map

        gender_map = {"m": "Male", "f": "Female"}
        state_map = {"NY": "New York", "CA": "California"}
        _ = Map(columns=[replace(it.gender, gender_map), replace(it.state, state_map)])

    def test_not_expression(self):
        from lale.lib.lale import Map

        with self.assertRaises(jsonschema.ValidationError):
            _ = Map(columns=[123, "hello"])

    def test_with_hyperopt(self):
        from sklearn.datasets import load_iris

        from lale.expressions import it, replace
        from lale.lib.lale import Hyperopt, Map, Relational

        X, y = load_iris(return_X_y=True)
        gender_map = {"m": "Male", "f": "Female"}
        state_map = {"NY": "New York", "CA": "California"}
        map_replace = Map(
            columns=[replace(it.gender, gender_map), replace(it.state, state_map)],
            remainder="drop",
        )
        pipeline = (
            Relational(
                operator=(Scan(table=it.main) & Scan(table=it.delay)) >> map_replace
            )
            >> LogisticRegression()
        )
        opt = Hyperopt(estimator=pipeline, cv=3, max_evals=5)
        trained = opt.fit(X, y)
        _ = trained

    def test_with_hyperopt2(self):
        import lale
        from lale.expressions import (
            count,
            it,
            max,
            mean,
            min,
            string_indexer,
            sum,
            variance,
        )
        from lale.lib.lale import Aggregate, ConcatFeatures, Join, Map, Relational, Scan
        from lale.operators import make_pipeline_graph

        lale.wrap_imported_operators()
        scan = Scan(table=it["main"])
        scan_0 = Scan(table=it["customers"])
        join = Join(
            pred=[
                (
                    it["main"]["group_customer_id"]
                    == it["customers"]["group_customer_id"]
                )
            ]
        )
        map = Map(
            columns={
                "[main](group_customer_id)[customers]|number_children|identity": it[
                    "number_children"
                ],
                "[main](group_customer_id)[customers]|name|identity": it["name"],
                "[main](group_customer_id)[customers]|income|identity": it["income"],
                "[main](group_customer_id)[customers]|address|identity": it["address"],
                "[main](group_customer_id)[customers]|age|identity": it["age"],
            },
            remainder="drop",
        )
        pipeline_4 = join >> map
        scan_1 = Scan(table=it["purchase"])
        join_0 = Join(
            pred=[(it["main"]["group_id"] == it["purchase"]["group_id"])],
            join_limit=50.0,
        )
        aggregate = Aggregate(
            columns={
                "[main](group_id)[purchase]|price|variance": variance(it["price"]),
                "[main](group_id)[purchase]|time|sum": sum(it["time"]),
                "[main](group_id)[purchase]|time|mean": mean(it["time"]),
                "[main](group_id)[purchase]|time|min": min(it["time"]),
                "[main](group_id)[purchase]|price|sum": sum(it["price"]),
                "[main](group_id)[purchase]|price|count": count(it["price"]),
                "[main](group_id)[purchase]|price|mean": mean(it["price"]),
                "[main](group_id)[purchase]|price|min": min(it["price"]),
                "[main](group_id)[purchase]|price|max": max(it["price"]),
                "[main](group_id)[purchase]|time|max": max(it["time"]),
                "[main](group_id)[purchase]|time|variance": variance(it["time"]),
            },
            group_by=it["row_id"],
        )
        pipeline_5 = join_0 >> aggregate
        map_0 = Map(
            columns={
                "[main]|group_customer_id|identity": it["group_customer_id"],
                "[main]|transaction_id|identity": it["transaction_id"],
                "[main]|group_id|identity": it["group_id"],
                "[main]|comments|identity": it["comments"],
                "[main]|id|identity": it["id"],
                "prefix_0_id": it["prefix_0_id"],
                "next_purchase": it["next_purchase"],
                "[main]|time|identity": it["time"],
            },
            remainder="drop",
        )
        scan_2 = Scan(table=it["transactions"])
        scan_3 = Scan(table=it["products"])
        join_1 = Join(
            pred=[
                (it["main"]["transaction_id"] == it["transactions"]["transaction_id"]),
                (it["transactions"]["product_id"] == it["products"]["product_id"]),
            ]
        )
        map_1 = Map(
            columns={
                "[main](transaction_id)[transactions](product_id)[products]|price|identity": it[
                    "price"
                ],
                "[main](transaction_id)[transactions](product_id)[products]|type|identity": it[
                    "type"
                ],
            },
            remainder="drop",
        )
        pipeline_6 = join_1 >> map_1
        join_2 = Join(
            pred=[
                (it["main"]["transaction_id"] == it["transactions"]["transaction_id"])
            ]
        )
        map_2 = Map(
            columns={
                "[main](transaction_id)[transactions]|description|identity": it[
                    "description"
                ],
                "[main](transaction_id)[transactions]|product_id|identity": it[
                    "product_id"
                ],
            },
            remainder="drop",
        )
        pipeline_7 = join_2 >> map_2
        map_3 = Map(
            columns=[
                string_indexer(it["[main]|comments|identity"]),
                string_indexer(
                    it["[main](transaction_id)[transactions]|description|identity"]
                ),
                string_indexer(
                    it[
                        "[main](transaction_id)[transactions](product_id)[products]|type|identity"
                    ]
                ),
                string_indexer(
                    it["[main](group_customer_id)[customers]|name|identity"]
                ),
                string_indexer(
                    it["[main](group_customer_id)[customers]|address|identity"]
                ),
            ]
        )
        pipeline_8 = ConcatFeatures() >> map_3
        relational = Relational(
            operator=make_pipeline_graph(
                steps=[
                    scan,
                    scan_0,
                    pipeline_4,
                    scan_1,
                    pipeline_5,
                    map_0,
                    scan_2,
                    scan_3,
                    pipeline_6,
                    pipeline_7,
                    pipeline_8,
                ],
                edges=[
                    (scan, pipeline_4),
                    (scan, pipeline_5),
                    (scan, map_0),
                    (scan, pipeline_6),
                    (scan, pipeline_7),
                    (scan_0, pipeline_4),
                    (pipeline_4, pipeline_8),
                    (scan_1, pipeline_5),
                    (pipeline_5, pipeline_8),
                    (map_0, pipeline_8),
                    (scan_2, pipeline_6),
                    (scan_2, pipeline_7),
                    (scan_3, pipeline_6),
                    (pipeline_6, pipeline_8),
                    (pipeline_7, pipeline_8),
                ],
            )
        )
        pipeline = relational >> (KNeighborsClassifier | LogisticRegression)
        from sklearn.datasets import load_iris

        X, y = load_iris(return_X_y=True)
        from lale.lib.lale import Hyperopt

        opt = Hyperopt(estimator=pipeline, max_evals=2)
        opt.fit(X, y)


class TestConcatFeatures(unittest.TestCase):
    def test_hyperparam_defaults(self):
        _ = ConcatFeatures()

    def test_init_fit_predict(self):
        trainable_cf = ConcatFeatures()
        A = [[11, 12, 13], [21, 22, 23], [31, 32, 33]]
        B = [[14, 15], [24, 25], [34, 35]]

        trained_cf = trainable_cf.fit(X=[A, B])
        transformed = trained_cf.transform([A, B])
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
            "n_estimators": (10, 100, 100),
            "criterion": ["entropy", "gini"],
            "max_depth": (3, 5, None),
            "min_samples_split": (2, 5, 2),
            "min_samples_leaf": (1, 5, 1),
            "max_features": (0.01, 1.0, 0.5),
        }
        self.maxDiff = None
        self.assertEqual(ranges, expected_ranges)


class TestKNeighborsClassifier(unittest.TestCase):
    def test_with_multioutput_targets(self):
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.utils import shuffle

        X, y1 = make_classification(
            n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1
        )
        y2 = shuffle(y1, random_state=1)
        y3 = shuffle(y1, random_state=2)
        Y = np.vstack((y1, y2, y3)).T
        trainable = KNeighborsClassifier()
        trained = trainable.fit(X, Y)
        _ = trained.predict(X)

    def test_predict_proba(self):
        trainable = KNeighborsClassifier()
        iris = sklearn.datasets.load_iris()
        trained = trainable.fit(iris.data, iris.target)
        # with self.assertWarns(DeprecationWarning):
        _ = trainable.predict_proba(iris.data)
        _ = trained.predict_proba(iris.data)


class TestLogisticRegression(unittest.TestCase):
    def test_hyperparam_keyword_enum(self):
        _ = LogisticRegression(
            LogisticRegression.penalty.l1, C=0.1, solver=LogisticRegression.solver.saga
        )

    def test_hyperparam_exclusive_min(self):
        with self.assertRaises(jsonschema.ValidationError):
            _ = LogisticRegression(LogisticRegression.penalty.l1, C=0.0)

    def test_hyperparam_penalty_solver_dependence(self):
        with self.assertRaises(jsonschema.ValidationError):
            _ = LogisticRegression(
                LogisticRegression.penalty.l1, LogisticRegression.solver.newton_cg
            )

    def test_hyperparam_dual_penalty_solver_dependence(self):
        with self.assertRaises(jsonschema.ValidationError):
            _ = LogisticRegression(
                LogisticRegression.penalty.l2, LogisticRegression.solver.sag, dual=True
            )

    def test_sample_weight(self):
        import numpy as np

        trainable_lr = LogisticRegression(n_jobs=1)
        iris = sklearn.datasets.load_iris()
        trained_lr = trainable_lr.fit(
            iris.data, iris.target, sample_weight=np.arange(len(iris.target))
        )
        _ = trained_lr.predict(iris.data)

    def test_predict_proba(self):
        import numpy as np

        trainable_lr = LogisticRegression(n_jobs=1)
        iris = sklearn.datasets.load_iris()
        trained_lr = trainable_lr.fit(
            iris.data, iris.target, sample_weight=np.arange(len(iris.target))
        )
        # with self.assertWarns(DeprecationWarning):
        _ = trainable_lr.predict_proba(iris.data)
        _ = trained_lr.predict_proba(iris.data)

    def test_decision_function(self):
        import numpy as np

        trainable_lr = LogisticRegression(n_jobs=1)
        iris = sklearn.datasets.load_iris()
        trained_lr = trainable_lr.fit(
            iris.data, iris.target, sample_weight=np.arange(len(iris.target))
        )
        _ = trained_lr.decision_function(iris.data)

    def test_with_sklearn_gridsearchcv(self):
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.model_selection import GridSearchCV

        lr = LogisticRegression()
        parameters = {"solver": ("liblinear", "lbfgs"), "penalty": ["l2"]}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = GridSearchCV(
                lr, parameters, cv=5, scoring=make_scorer(accuracy_score)
            )
            iris = load_iris()
            clf.fit(iris.data, iris.target)

    def test_with_randomizedsearchcv(self):
        import numpy as np
        from scipy.stats.distributions import uniform
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.model_selection import RandomizedSearchCV

        lr = LogisticRegression()
        ranges, cat_idx = lr.get_param_ranges()
        # specify parameters and distributions to sample from
        # the loguniform distribution needs to be taken care of properly
        param_dist = {"solver": ranges["solver"], "C": uniform(0.03125, np.log(32768))}
        # run randomized search
        n_iter_search = 5
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            random_search = RandomizedSearchCV(
                lr,
                param_distributions=param_dist,
                n_iter=n_iter_search,
                cv=5,
                scoring=make_scorer(accuracy_score),
            )
            iris = load_iris()
            random_search.fit(iris.data, iris.target)

    def test_grid_search_on_trained(self):
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.model_selection import GridSearchCV

        iris = load_iris()
        X, y = iris.data, iris.target
        lr = LogisticRegression()
        trained = lr.fit(X, y)
        parameters = {"solver": ("liblinear", "lbfgs"), "penalty": ["l2"]}

        _ = GridSearchCV(trained, parameters, cv=5, scoring=make_scorer(accuracy_score))

    def test_grid_search_on_trained_auto(self):
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.model_selection import GridSearchCV

        iris = load_iris()
        X, y = iris.data, iris.target
        lr = LogisticRegression()
        trained = lr.fit(X, y)
        parameters = get_grid_search_parameter_grids(lr, num_samples=2)

        _ = GridSearchCV(trained, parameters, cv=5, scoring=make_scorer(accuracy_score))

    def test_doc(self):
        from test.mock_custom_operators import MyLR

        import sklearn.datasets
        import sklearn.utils

        iris = sklearn.datasets.load_iris()
        X_all, y_all = sklearn.utils.shuffle(iris.data, iris.target, random_state=42)
        X_train, y_train = X_all[10:], y_all[10:]
        X_test, y_test = X_all[:10], y_all[:10]
        print("expected {}".format(y_test))
        import warnings

        warnings.filterwarnings("ignore", category=FutureWarning)
        trainable = MyLR(solver="lbfgs", C=0.1)
        trained = trainable.fit(X_train, y_train)
        predictions = trained.predict(X_test)
        print("actual {}".format(predictions))


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
        from sklearn.base import clone
        from sklearn.datasets import load_iris
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
        from sklearn.datasets import load_iris
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
        from sklearn.datasets import load_iris

        iris = load_iris()
        X, y = iris.data, iris.target
        trained = lr.fit(X, y)
        _ = clone(trained)

    def test_with_voting_classifier1(self):
        lr = LogisticRegression()
        knn = KNeighborsClassifier()
        from sklearn.ensemble import VotingClassifier

        vclf = VotingClassifier(estimators=[("lr", lr), ("knn", knn)])
        from sklearn.datasets import load_iris

        iris = load_iris()
        X, y = iris.data, iris.target
        vclf.fit(X, y)

    def test_with_voting_classifier2(self):
        lr = LogisticRegression()
        pca = PCA()
        trainable = pca >> lr

        from sklearn.ensemble import VotingClassifier

        vclf = VotingClassifier(estimators=[("lr", lr), ("pipe", trainable)])
        from sklearn.datasets import load_iris

        iris = load_iris()
        X, y = iris.data, iris.target
        vclf.fit(X, y)

    def test_fit_clones_impl(self):
        from sklearn.datasets import load_iris

        lr_trainable = LogisticRegression()
        iris = load_iris()
        X, y = iris.data, iris.target
        lr_trained = lr_trainable.fit(X, y)
        self.assertIsNot(lr_trainable._impl, lr_trained._impl)


class TestMLPClassifier(unittest.TestCase):
    def test_with_multioutput_targets(self):
        import numpy as np
        from sklearn.datasets import make_classification
        from sklearn.utils import shuffle

        X, y1 = make_classification(
            n_samples=10, n_features=100, n_informative=30, n_classes=3, random_state=1
        )
        y2 = shuffle(y1, random_state=1)
        y3 = shuffle(y1, random_state=2)
        Y = np.vstack((y1, y2, y3)).T
        trainable = KNeighborsClassifier()
        trained = trainable.fit(X, Y)
        _ = trained.predict(X)

    def test_predict_proba(self):
        trainable = MLPClassifier()
        iris = sklearn.datasets.load_iris()
        trained = trainable.fit(iris.data, iris.target)
        #        with self.assertWarns(DeprecationWarning):
        _ = trainable.predict_proba(iris.data)
        _ = trained.predict_proba(iris.data)


class TestOperatorChoice(unittest.TestCase):
    def test_make_choice_with_instance(self):
        from sklearn.datasets import load_iris

        from lale.operators import make_choice

        iris = load_iris()
        X, y = iris.data, iris.target
        tfm = PCA() | Nystroem() | NoOp()
        with self.assertRaises(AttributeError):
            _ = tfm.fit(X, y)
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


class TestTfidfVectorizer(unittest.TestCase):
    def test_more_hyperparam_values(self):
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

        with self.assertRaises(jsonschema.ValidationError):
            _ = TfidfVectorizer(
                max_df=2,
                min_df=2,
                max_features=1000,
                stop_words="english",
                tokenizer=my_tokenizer,
                analyzer="char",
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


class TestOperatorWithoutSchema(unittest.TestCase):
    def test_trainable_pipe_left(self):
        from sklearn.decomposition import PCA

        from lale.lib.sklearn import LogisticRegression

        iris = sklearn.datasets.load_iris()
        pipeline = PCA() >> LogisticRegression(random_state=42)
        pipeline.fit(iris.data, iris.target)

    def test_trainable_pipe_right(self):
        from sklearn.decomposition import PCA

        from lale.lib.lale import NoOp
        from lale.lib.sklearn import LogisticRegression

        iris = sklearn.datasets.load_iris()
        pipeline = NoOp() >> PCA() >> LogisticRegression(random_state=42)
        pipeline.fit(iris.data, iris.target)

    def dont_test_planned_pipe_left(self):
        from sklearn.decomposition import PCA

        from lale.lib.lale import Hyperopt, NoOp
        from lale.lib.sklearn import LogisticRegression

        iris = sklearn.datasets.load_iris()
        pipeline = NoOp() >> PCA >> LogisticRegression
        clf = Hyperopt(estimator=pipeline, max_evals=1)
        clf.fit(iris.data, iris.target)

    def dont_test_planned_pipe_right(self):
        from sklearn.decomposition import PCA

        from lale.lib.lale import Hyperopt
        from lale.lib.sklearn import LogisticRegression

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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)
        import warnings

        warnings.filterwarnings("ignore")

    def test_with_lale_classifiers(self):
        from lale.lib.sklearn import VotingClassifier

        clf = VotingClassifier(
            estimators=[("knn", KNeighborsClassifier()), ("lr", LogisticRegression())]
        )
        trained = clf.fit(self.X_train, self.y_train)
        trained.predict(self.X_test)

    def test_with_lale_pipeline(self):
        from lale.lib.sklearn import VotingClassifier

        clf = VotingClassifier(
            estimators=[
                ("knn", KNeighborsClassifier()),
                ("pca_lr", PCA() >> LogisticRegression()),
            ]
        )
        trained = clf.fit(self.X_train, self.y_train)
        trained.predict(self.X_test)

    def test_with_hyperopt(self):
        from lale.lib.lale import Hyperopt
        from lale.lib.sklearn import VotingClassifier

        clf = VotingClassifier(
            estimators=[("knn", KNeighborsClassifier()), ("lr", LogisticRegression())]
        )
        _ = clf.auto_configure(self.X_train, self.y_train, Hyperopt, max_evals=1)

    def test_with_gridsearch(self):
        from sklearn.metrics import accuracy_score, make_scorer

        from lale.lib.lale import GridSearchCV
        from lale.lib.sklearn import VotingClassifier

        clf = VotingClassifier(
            estimators=[("knn", KNeighborsClassifier()), ("rc", RidgeClassifier())],
            voting="hard",
        )
        _ = clf.auto_configure(
            self.X_train,
            self.y_train,
            GridSearchCV,
            lale_num_samples=1,
            lale_num_grids=1,
            cv=2,
            scoring=make_scorer(accuracy_score),
        )

    @unittest.skip("TODO: get this working with sklearn 0.23")
    def test_with_observed_gridsearch(self):
        from sklearn.metrics import accuracy_score, make_scorer

        from lale.lib.lale import GridSearchCV
        from lale.lib.lale.observing import LoggingObserver
        from lale.lib.sklearn import VotingClassifier

        clf = VotingClassifier(
            estimators=[("knn", KNeighborsClassifier()), ("rc", RidgeClassifier())],
            voting="hard",
        )
        _ = clf.auto_configure(
            self.X_train,
            self.y_train,
            GridSearchCV,
            lale_num_samples=1,
            lale_num_grids=1,
            cv=2,
            scoring=make_scorer(accuracy_score),
            observer=LoggingObserver,
        )


class TestBaggingClassifier(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_with_lale_classifiers(self):
        from lale.lib.sklearn import BaggingClassifier

        clf = BaggingClassifier(base_estimator=LogisticRegression())
        trained = clf.fit(self.X_train, self.y_train)
        trained.predict(self.X_test)

    def test_with_lale_pipeline(self):
        from lale.lib.sklearn import BaggingClassifier

        clf = BaggingClassifier(base_estimator=PCA() >> LogisticRegression())
        trained = clf.fit(self.X_train, self.y_train)
        trained.predict(self.X_test)

    def test_with_hyperopt(self):
        from lale.lib.lale import Hyperopt
        from lale.lib.sklearn import BaggingClassifier

        clf = BaggingClassifier(base_estimator=LogisticRegression())
        trained = clf.auto_configure(self.X_train, self.y_train, Hyperopt, max_evals=1)
        print(trained.to_json())

    def test_pipeline_with_hyperopt(self):
        from lale.lib.lale import Hyperopt
        from lale.lib.sklearn import BaggingClassifier

        clf = BaggingClassifier(base_estimator=PCA() >> LogisticRegression())
        _ = clf.auto_configure(self.X_train, self.y_train, Hyperopt, max_evals=1)

    def test_pipeline_choice_with_hyperopt(self):
        from lale.lib.lale import Hyperopt
        from lale.lib.sklearn import BaggingClassifier

        clf = BaggingClassifier(
            base_estimator=PCA() >> (LogisticRegression() | KNeighborsClassifier())
        )
        _ = clf.auto_configure(self.X_train, self.y_train, Hyperopt, max_evals=1)


class TestLazyImpl(unittest.TestCase):
    def test_lazy_impl(self):
        from lale.lib.lale import Hyperopt

        impl = Hyperopt._impl
        self.assertTrue(inspect.isclass(impl))


class TestOrdinalEncoder(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_with_hyperopt(self):
        from lale.lib.sklearn import OrdinalEncoder

        fproc = OrdinalEncoder()
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
        from sklearn.datasets import load_iris

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
        from sklearn.datasets import load_iris

        from lale.lib.lale import Hyperopt

        iris_data = load_iris()
        op = Hyperopt(
            estimator=LogisticRegression(), max_evals=1, show_progressbar=False
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            op2 = op.fit(iris_data.data[10:], iris_data.target[10:])
            _ = op2.summary


class TestFriedmanMSE(unittest.TestCase):
    # This was prompted buy a bug, keeping it as it may help with support for other sklearn versions
    def setUp(self):
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split

        X, y = make_regression(
            n_features=4, n_informative=2, random_state=0, shuffle=False
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_rfr(self):
        from lale.lib.sklearn import RandomForestRegressor

        reg = RandomForestRegressor(
            bootstrap=True,
            criterion="friedman_mse",
            max_depth=4,
            max_features=0.9832410473940374,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            min_samples_leaf=3,
            min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            n_estimators=29,
            n_jobs=4,
            oob_score=False,
            random_state=33,
            verbose=0,
            warm_start=False,
        )
        reg.fit(self.X_train, self.y_train)

    def test_etr(self):
        from lale.lib.sklearn import ExtraTreesRegressor

        reg = ExtraTreesRegressor(
            bootstrap=True,
            criterion="friedman_mse",
            max_depth=4,
            max_features=0.9832410473940374,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            min_impurity_split=None,
            min_samples_leaf=3,
            min_samples_split=2,
            min_weight_fraction_leaf=0.0,
            n_estimators=29,
            n_jobs=4,
            oob_score=False,
            random_state=33,
            verbose=0,
            warm_start=False,
        )
        reg.fit(self.X_train, self.y_train)


class TestSpuriousSideConstraintsRegression(unittest.TestCase):
    # This was prompted buy a bug, keeping it as it may help with support for other sklearn versions
    def setUp(self):
        from sklearn.datasets import make_regression
        from sklearn.model_selection import train_test_split

        X, y = make_regression(
            n_features=4, n_informative=2, random_state=0, shuffle=False
        )
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_gradient_boost_regressor(self):
        from lale.lib.sklearn import GradientBoostingRegressor

        reg = GradientBoostingRegressor(
            alpha=0.9789984970831765,
            criterion="friedman_mse",
            init=None,
            learning_rate=0.1,
            loss="ls",
        )
        reg.fit(self.X_train, self.y_train)

    def test_sgd_regressor(self):
        from lale.lib.sklearn import SGDRegressor

        reg = SGDRegressor(loss="squared_loss", epsilon=0.2)
        reg.fit(self.X_train, self.y_train)

    def test_sgd_regressor_1(self):
        from lale.lib.sklearn import SGDRegressor

        reg = SGDRegressor(learning_rate="optimal", eta0=0.2)
        reg.fit(self.X_train, self.y_train)

    def test_sgd_regressor_2(self):
        from lale.lib.sklearn import SGDRegressor

        reg = SGDRegressor(early_stopping=False, validation_fraction=0.2)
        reg.fit(self.X_train, self.y_train)

    def test_sgd_regressor_3(self):
        from sklearn.linear_model import SGDRegressor

        reg = SGDRegressor(l1_ratio=0.2, penalty="l1")
        reg.fit(self.X_train, self.y_train)


class TestSpuriousSideConstraintsClassification(unittest.TestCase):
    # This was prompted buy a bug, keeping it as it may help with support for other sklearn versions
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_sgd_classifier(self):
        from lale.lib.sklearn import SGDClassifier

        reg = SGDClassifier(loss="squared_loss", epsilon=0.2)
        reg.fit(self.X_train, self.y_train)

    def test_sgd_classifier_1(self):
        from lale.lib.sklearn import SGDClassifier

        reg = SGDClassifier(learning_rate="optimal", eta0=0.2)
        reg.fit(self.X_train, self.y_train)

    def test_sgd_classifier_2(self):
        from lale.lib.sklearn import SGDClassifier

        reg = SGDClassifier(early_stopping=False, validation_fraction=0.2)
        reg.fit(self.X_train, self.y_train)

    def test_sgd_classifier_3(self):
        from lale.lib.sklearn import SGDClassifier

        reg = SGDClassifier(l1_ratio=0.2, penalty="l1")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier(self):
        from lale.lib.sklearn import MLPClassifier

        reg = MLPClassifier(early_stopping=False, validation_fraction=0.2)
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_1(self):
        from lale.lib.sklearn import MLPClassifier

        reg = MLPClassifier(beta_1=0.8, solver="sgd")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_2b(self):
        from lale.lib.sklearn import MLPClassifier

        reg = MLPClassifier(beta_2=0.8, solver="sgd")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_2e(self):
        from lale.lib.sklearn import MLPClassifier

        reg = MLPClassifier(epsilon=0.8, solver="sgd")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_3(self):
        from lale.lib.sklearn import MLPClassifier

        reg = MLPClassifier(n_iter_no_change=100, solver="lbfgs")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_4(self):
        from lale.lib.sklearn import MLPClassifier

        reg = MLPClassifier(early_stopping=True, solver="lbfgs")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_5(self):
        from lale.lib.sklearn import MLPClassifier

        reg = MLPClassifier(nesterovs_momentum=False, solver="lbfgs")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_6(self):
        from lale.lib.sklearn import MLPClassifier

        reg = MLPClassifier(momentum=0.8, solver="lbfgs")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_7(self):
        from lale.lib.sklearn import MLPClassifier

        reg = MLPClassifier(shuffle=False, solver="lbfgs")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_8(self):
        from lale.lib.sklearn import MLPClassifier

        reg = MLPClassifier(learning_rate="invscaling", solver="lbfgs")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_9(self):
        from lale.lib.sklearn import MLPClassifier

        reg = MLPClassifier(learning_rate_init=0.002, solver="lbfgs")
        reg.fit(self.X_train, self.y_train)

    def test_mlp_classifier_10(self):
        from lale.lib.sklearn import MLPClassifier

        reg = MLPClassifier(learning_rate="invscaling", power_t=0.4, solver="lbfgs")
        reg.fit(self.X_train, self.y_train)

    def test_passive_aggressive_classifier(self):
        from lale.lib.sklearn import PassiveAggressiveClassifier

        reg = PassiveAggressiveClassifier(validation_fraction=0.4, early_stopping=False)
        reg.fit(self.X_train, self.y_train)

    def test_svc(self):
        from lale.lib.sklearn import SVC

        reg = SVC(kernel="linear", gamma=1)
        reg.fit(self.X_train, self.y_train)

    def test_simple_imputer(self):
        from lale.lib.sklearn import SimpleImputer

        reg = SimpleImputer(strategy="mean", fill_value=10)
        reg.fit(self.X_train, self.y_train)

    def test_nystroem(self):
        from lale.lib.sklearn import Nystroem

        reg = Nystroem(kernel="cosine", gamma=0.1)
        reg.fit(self.X_train, self.y_train)

    def test_nystroem_1(self):
        from lale.lib.sklearn import Nystroem

        reg = Nystroem(kernel="cosine", coef0=0.1)
        reg.fit(self.X_train, self.y_train)

    def test_nystroem_2(self):
        from lale.lib.sklearn import Nystroem

        reg = Nystroem(kernel="cosine", degree=2)
        reg.fit(self.X_train, self.y_train)

    def test_ridge_classifier(self):
        from lale.lib.sklearn import RidgeClassifier

        reg = RidgeClassifier(fit_intercept=False, normalize=True)
        reg.fit(self.X_train, self.y_train)

    def test_ridge_classifier_1(self):
        from lale.lib.sklearn import RidgeClassifier

        reg = RidgeClassifier(solver="svd", max_iter=10)
        reg.fit(self.X_train, self.y_train)


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


class TestRelationalOperator(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_fit_transform(self):
        relational = Relational(
            operator=(Scan(table=it.main) & Scan(table=it.delay))
            >> Join(
                pred=[
                    it.main.TrainId == it.delay.TrainId,
                    it.main["Arrival time"] >= it.delay.TimeStamp,
                ]
            )
            >> Aggregate(columns=[count(it.Delay)], group_by=it.MessageId)
        )
        trained_relational = relational.fit(self.X_train, self.y_train)
        _ = trained_relational.transform(self.X_test)

    def test_fit_error(self):
        relational = Relational(
            operator=(Scan(table=it.main) & Scan(table=it.delay))
            >> Join(
                pred=[
                    it.main.TrainId == it.delay.TrainId,
                    it.main["Arrival time"] >= it.delay.TimeStamp,
                ]
            )
            >> Aggregate(columns=[count(it.Delay)], group_by=it.MessageId)
        )
        with self.assertRaises(ValueError):
            _ = relational.fit([self.X_train], self.y_train)

    def test_transform_error(self):
        relational = Relational(
            operator=(Scan(table=it.main) & Scan(table=it.delay))
            >> Join(
                pred=[
                    it.main.TrainId == it.delay.TrainId,
                    it.main["Arrival time"] >= it.delay.TimeStamp,
                ]
            )
            >> Aggregate(columns=[count(it.Delay)], group_by=it.MessageId)
        )
        trained_relational = relational.fit(self.X_train, self.y_train)
        with self.assertRaises(ValueError):
            _ = trained_relational.transform([self.X_test])

    def test_fit_transform_in_pipeline(self):
        relational = Relational(
            operator=(Scan(table=it.main) & Scan(table=it.delay))
            >> Join(
                pred=[
                    it.main.TrainId == it.delay.TrainId,
                    it.main["Arrival time"] >= it.delay.TimeStamp,
                ]
            )
            >> Aggregate(columns=[count(it.Delay)], group_by=it.MessageId)
        )
        pipeline = relational >> LogisticRegression()
        trained_pipeline = pipeline.fit(self.X_train, self.y_train)
        _ = trained_pipeline.predict(self.X_test)
