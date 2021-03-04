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

import random
import unittest
import warnings

from sklearn.metrics import accuracy_score

from lale.lib.lale import Batching, NoOp
from lale.lib.sklearn import PCA, LogisticRegression, Nystroem
from lale.search.lale_grid_search_cv import get_grid_search_parameter_grids


class TestBatching(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_fit(self):
        import warnings

        import lale.lib.sklearn as lale_sklearn

        warnings.filterwarnings(action="ignore")

        pipeline = NoOp() >> Batching(
            operator=lale_sklearn.MinMaxScaler()
            >> lale_sklearn.MLPClassifier(random_state=42),
            batch_size=112,
        )
        trained = pipeline.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)
        lale_accuracy = accuracy_score(self.y_test, predictions)

        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import MinMaxScaler

        prep = MinMaxScaler()
        trained_prep = prep.partial_fit(self.X_train, self.y_train)
        X_transformed = trained_prep.transform(self.X_train)

        clf = MLPClassifier(random_state=42)
        import numpy as np

        trained_clf = clf.partial_fit(
            X_transformed, self.y_train, classes=np.unique(self.y_train)
        )
        predictions = trained_clf.predict(trained_prep.transform(self.X_test))
        sklearn_accuracy = accuracy_score(self.y_test, predictions)

        self.assertEqual(lale_accuracy, sklearn_accuracy)

    def test_fit1(self):
        import warnings

        warnings.filterwarnings(action="ignore")
        from lale.lib.sklearn import MinMaxScaler, MLPClassifier

        pipeline = Batching(
            operator=MinMaxScaler() >> MLPClassifier(random_state=42), batch_size=112
        )
        trained = pipeline.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)
        lale_accuracy = accuracy_score(self.y_test, predictions)

        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import MinMaxScaler

        prep = MinMaxScaler()
        trained_prep = prep.partial_fit(self.X_train, self.y_train)
        X_transformed = trained_prep.transform(self.X_train)

        clf = MLPClassifier(random_state=42)
        import numpy as np

        trained_clf = clf.partial_fit(
            X_transformed, self.y_train, classes=np.unique(self.y_train)
        )
        predictions = trained_clf.predict(trained_prep.transform(self.X_test))
        sklearn_accuracy = accuracy_score(self.y_test, predictions)

        self.assertEqual(lale_accuracy, sklearn_accuracy)

    def test_fit2(self):
        import warnings

        warnings.filterwarnings(action="ignore")
        from lale.lib.sklearn import MinMaxScaler

        pipeline = Batching(operator=MinMaxScaler() >> MinMaxScaler(), batch_size=112)
        trained = pipeline.fit(self.X_train, self.y_train)
        lale_transforms = trained.transform(self.X_test)

        from sklearn.preprocessing import MinMaxScaler

        prep = MinMaxScaler()
        trained_prep = prep.partial_fit(self.X_train, self.y_train)
        X_transformed = trained_prep.transform(self.X_train)

        clf = MinMaxScaler()

        trained_clf = clf.partial_fit(X_transformed, self.y_train)
        sklearn_transforms = trained_clf.transform(trained_prep.transform(self.X_test))

        for i in range(5):
            for j in range(2):
                self.assertAlmostEqual(lale_transforms[i, j], sklearn_transforms[i, j])

    def test_fit3(self):
        from lale.lib.sklearn import PCA, MinMaxScaler, MLPClassifier

        pipeline = PCA() >> Batching(
            operator=MinMaxScaler() >> MLPClassifier(random_state=42), batch_size=10
        )
        trained = pipeline.fit(self.X_train, self.y_train)
        _ = trained.predict(self.X_test)

    def test_no_partial_fit(self):
        pipeline = Batching(operator=NoOp() >> LogisticRegression())
        with self.assertRaises(AttributeError):
            _ = pipeline.fit(self.X_train, self.y_train)

    def test_fit4(self):
        import warnings

        warnings.filterwarnings(action="ignore")
        from lale.lib.sklearn import MinMaxScaler, MLPClassifier

        pipeline = Batching(
            operator=MinMaxScaler() >> MLPClassifier(random_state=42),
            batch_size=112,
            inmemory=True,
        )
        trained = pipeline.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)
        lale_accuracy = accuracy_score(self.y_test, predictions)

        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import MinMaxScaler

        prep = MinMaxScaler()
        trained_prep = prep.partial_fit(self.X_train, self.y_train)
        X_transformed = trained_prep.transform(self.X_train)

        clf = MLPClassifier(random_state=42)
        import numpy as np

        trained_clf = clf.partial_fit(
            X_transformed, self.y_train, classes=np.unique(self.y_train)
        )
        predictions = trained_clf.predict(trained_prep.transform(self.X_test))
        sklearn_accuracy = accuracy_score(self.y_test, predictions)

        self.assertEqual(lale_accuracy, sklearn_accuracy)

    # TODO: Nesting doesn't work yet
    # def test_nested_pipeline(self):
    #     from lale.lib.sklearn import MinMaxScaler, MLPClassifier
    #     pipeline = Batching(operator = MinMaxScaler() >> Batching(operator = NoOp() >> MLPClassifier(random_state=42)), batch_size = 112)
    #     trained = pipeline.fit(self.X_train, self.y_train)
    #     predictions = trained.predict(self.X_test)
    #     lale_accuracy = accuracy_score(self.y_test, predictions)


class TestPipeline(unittest.TestCase):
    def dont_test_with_gridsearchcv2_auto(self):
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.model_selection import GridSearchCV

        lr = LogisticRegression(random_state=42)
        pca = PCA(random_state=42, svd_solver="arpack")
        trainable = pca >> lr
        from sklearn.pipeline import Pipeline

        scikit_pipeline = Pipeline(
            [
                (pca.name(), PCA(random_state=42, svd_solver="arpack")),
                (lr.name(), LogisticRegression(random_state=42)),
            ]
        )
        all_parameters = get_grid_search_parameter_grids(trainable, num_samples=1)
        # otherwise the test takes too long
        parameters = random.sample(all_parameters, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = GridSearchCV(
                scikit_pipeline, parameters, cv=2, scoring=make_scorer(accuracy_score)
            )
            iris = load_iris()
            clf.fit(iris.data, iris.target)
            predicted = clf.predict(iris.data)
            accuracy_with_lale_operators = accuracy_score(iris.target, predicted)

        from sklearn.decomposition import PCA as SklearnPCA
        from sklearn.linear_model import LogisticRegression as SklearnLR
        from sklearn.pipeline import Pipeline

        scikit_pipeline = Pipeline(
            [
                (pca.name(), SklearnPCA(random_state=42, svd_solver="arpack")),
                (lr.name(), SklearnLR(random_state=42)),
            ]
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = GridSearchCV(
                scikit_pipeline, parameters, cv=2, scoring=make_scorer(accuracy_score)
            )
            iris = load_iris()
            clf.fit(iris.data, iris.target)
            predicted = clf.predict(iris.data)
            accuracy_with_scikit_operators = accuracy_score(iris.target, predicted)
        self.assertEqual(accuracy_with_lale_operators, accuracy_with_scikit_operators)

    def test_with_gridsearchcv3(self):
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.model_selection import GridSearchCV

        _ = LogisticRegression()
        from sklearn.pipeline import Pipeline

        scikit_pipeline = Pipeline(
            [("nystroem", Nystroem()), ("lr", LogisticRegression())]
        )
        parameters = {"lr__solver": ("liblinear", "lbfgs"), "lr__penalty": ["l2"]}
        clf = GridSearchCV(
            scikit_pipeline, parameters, cv=2, scoring=make_scorer(accuracy_score)
        )
        iris = load_iris()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(iris.data, iris.target)
        _ = clf.predict(iris.data)

    def test_with_gridsearchcv3_auto(self):
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
        from sklearn.model_selection import GridSearchCV

        lr = LogisticRegression()
        from sklearn.pipeline import Pipeline

        scikit_pipeline = Pipeline(
            [(Nystroem().name(), Nystroem()), (lr.name(), LogisticRegression())]
        )
        all_parameters = get_grid_search_parameter_grids(
            Nystroem() >> lr, num_samples=1
        )
        # otherwise the test takes too long
        parameters = random.sample(all_parameters, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            clf = GridSearchCV(
                scikit_pipeline, parameters, cv=2, scoring=make_scorer(accuracy_score)
            )
            iris = load_iris()
            clf.fit(iris.data, iris.target)
            _ = clf.predict(iris.data)

    def test_with_gridsearchcv3_auto_wrapped(self):
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer

        pipeline = Nystroem() >> LogisticRegression()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from lale.lib.lale import GridSearchCV

            clf = GridSearchCV(
                estimator=pipeline,
                lale_num_samples=1,
                lale_num_grids=1,
                cv=2,
                scoring=make_scorer(accuracy_score),
            )
            iris = load_iris()
            clf.fit(iris.data, iris.target)
            _ = clf.predict(iris.data)


class TestBatching2(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_batching_with_hyperopt(self):
        from lale.lib.lale import Batching, Hyperopt
        from lale.lib.sklearn import MinMaxScaler, SGDClassifier

        pipeline = Batching(operator=MinMaxScaler() >> SGDClassifier())
        trained = pipeline.auto_configure(
            self.X_train, self.y_train, optimizer=Hyperopt, max_evals=1
        )
        _ = trained.predict(self.X_test)


class TestImportFromSklearnWithCognito(unittest.TestCase):
    def test_import_from_sklearn(self):
        pipeline_str = """from lale.lib.autoai_libs import NumpyColumnSelector
from lale.lib.autoai_libs import CompressStrings
from lale.lib.autoai_libs import NumpyReplaceMissingValues
from lale.lib.autoai_libs import NumpyReplaceUnknownValues
from lale.lib.autoai_libs import boolean2float
from lale.lib.autoai_libs import CatImputer
from lale.lib.autoai_libs import CatEncoder
import numpy as np
from lale.lib.autoai_libs import float32_transform
from lale.operators import make_pipeline
from lale.lib.autoai_libs import FloatStr2Float
from lale.lib.autoai_libs import NumImputer
from lale.lib.autoai_libs import OptStandardScaler
from lale.operators import make_union
from lale.lib.autoai_libs import NumpyPermuteArray
from lale.lib.autoai_libs import TA1
import autoai_libs.utils.fc_methods
from lale.lib.autoai_libs import FS1
from xgboost import XGBRegressor

numpy_column_selector_0 = NumpyColumnSelector(columns=[1])
compress_strings = CompressStrings(compress_type='hash', dtypes_list=['int_num'], missing_values_reference_list=['', '-', '?', float('nan')], misslist_list=[[]])
numpy_replace_missing_values_0 = NumpyReplaceMissingValues(filling_values=float('nan'), missing_values=[])
numpy_replace_unknown_values = NumpyReplaceUnknownValues(filling_values=float('nan'), filling_values_list=[float('nan')], known_values_list=[[36, 45, 56, 67, 68, 75, 78, 89]], missing_values_reference_list=['', '-', '?', float('nan')])
cat_imputer = CatImputer(missing_values=float('nan'), sklearn_version_family='20', strategy='most_frequent')
cat_encoder = CatEncoder(dtype=np.float64, handle_unknown='error', sklearn_version_family='20')
pipeline_0 = make_pipeline(numpy_column_selector_0, compress_strings, numpy_replace_missing_values_0, numpy_replace_unknown_values, boolean2float(), cat_imputer, cat_encoder, float32_transform())
numpy_column_selector_1 = NumpyColumnSelector(columns=[0])
float_str2_float = FloatStr2Float(dtypes_list=['int_num'], missing_values_reference_list=[])
numpy_replace_missing_values_1 = NumpyReplaceMissingValues(filling_values=float('nan'), missing_values=[])
num_imputer = NumImputer(missing_values=float('nan'), strategy='median')
opt_standard_scaler = OptStandardScaler(num_scaler_copy=None, num_scaler_with_mean=None, num_scaler_with_std=None, use_scaler_flag=False)
pipeline_1 = make_pipeline(numpy_column_selector_1, float_str2_float, numpy_replace_missing_values_1, num_imputer, opt_standard_scaler, float32_transform())
union = make_union(pipeline_0, pipeline_1)
numpy_permute_array = NumpyPermuteArray(axis=0, permutation_indices=[1, 0])
ta1_0 = TA1(fun=np.tan, name='tan', datatypes=['float'], feat_constraints=[autoai_libs.utils.fc_methods.is_not_categorical], col_names=['age', 'weight'], col_dtypes=[np.dtype('float32'), np.dtype('float32')])
fs1_0 = FS1(cols_ids_must_keep=range(0, 2), additional_col_count_to_keep=4, ptype='regression')
ta1_1 = TA1(fun=np.square, name='square', datatypes=['numeric'], feat_constraints=[autoai_libs.utils.fc_methods.is_not_categorical], col_names=['age', 'weight', 'tan(age)'], col_dtypes=[np.dtype('float32'), np.dtype('float32'), np.dtype('float32')])
fs1_1 = FS1(cols_ids_must_keep=range(0, 2), additional_col_count_to_keep=4, ptype='regression')
ta1_2 = TA1(fun=np.sin, name='sin', datatypes=['float'], feat_constraints=[autoai_libs.utils.fc_methods.is_not_categorical], col_names=['age', 'weight', 'tan(age)', 'square(age)', 'square(tan(age))'], col_dtypes=[np.dtype('float32'), np.dtype('float32'), np.dtype('float32'), np.dtype('float32'), np.dtype('float32')])
fs1_2 = FS1(cols_ids_must_keep=range(0, 2), additional_col_count_to_keep=4, ptype='regression')
xgb_regressor = XGBRegressor(missing=float('nan'), n_jobs=4, random_state=33, silent=True, verbosity=0)
pipeline = make_pipeline(union, numpy_permute_array, ta1_0, fs1_0, ta1_1, fs1_1, ta1_2, fs1_2, xgb_regressor)
"""
        globals2 = {}
        exec(pipeline_str, globals2)
        pipeline2 = globals2["pipeline"]
        sklearn_pipeline = pipeline2.export_to_sklearn_pipeline()
        from lale import helpers

        _ = helpers.import_from_sklearn_pipeline(sklearn_pipeline)


class TestExportToSklearnForEstimator(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def create_pipeline(self):
        from sklearn.decomposition import PCA
        from sklearn.pipeline import make_pipeline

        pipeline = make_pipeline(PCA(), LogisticRegression())
        return pipeline

    def test_import_export_trained(self):
        import numpy as np
        from sklearn.pipeline import Pipeline

        from lale.helpers import import_from_sklearn_pipeline

        pipeline = self.create_pipeline()
        self.assertEquals(isinstance(pipeline, Pipeline), True)
        pipeline.fit(self.X_train, self.y_train)
        predictions_before = pipeline.predict(self.X_test)
        lale_pipeline = import_from_sklearn_pipeline(pipeline)
        predictions_after = lale_pipeline.predict(self.X_test)
        sklearn_pipeline = lale_pipeline.export_to_sklearn_pipeline()
        predictions_after_1 = sklearn_pipeline.predict(self.X_test)
        self.assertEquals(np.all(predictions_before == predictions_after), True)
        self.assertEquals(np.all(predictions_before == predictions_after_1), True)

    def test_import_export_trainable(self):
        from sklearn.exceptions import NotFittedError
        from sklearn.pipeline import Pipeline

        from lale.helpers import import_from_sklearn_pipeline

        pipeline = self.create_pipeline()
        self.assertEquals(isinstance(pipeline, Pipeline), True)
        pipeline.fit(self.X_train, self.y_train)
        lale_pipeline = import_from_sklearn_pipeline(pipeline, fitted=False)
        with self.assertRaises(ValueError):
            lale_pipeline.predict(self.X_test)
        sklearn_pipeline = lale_pipeline.export_to_sklearn_pipeline()
        with self.assertRaises(NotFittedError):
            sklearn_pipeline.predict(self.X_test)
