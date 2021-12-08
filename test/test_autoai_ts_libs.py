# Copyright 2019, 2020, 2021 IBM Corporation
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

import copy
import unittest
from multiprocessing import cpu_count
from typing import cast

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit

import lale.type_checking
from lale.datasets.uci import fetch_household_power_consumption
from lale.lib.autoai_ts_libs import (  # type: ignore # noqa; StandardRowMeanCenterMTS,; WindowTransformerMTS; DifferenceFlattenAutoEnsembler,; FlattenAutoEnsembler,; LocalizedFlattenAutoEnsembler,
    AutoaiTSPipeline,
    AutoaiWindowedWrappedRegressor,
    AutoaiWindowTransformedTargetRegressor,
    DifferenceFlattenAutoEnsembler,
    FlattenAutoEnsembler,
    LocalizedFlattenAutoEnsembler,
    MT2RForecaster,
    SmallDataWindowTargetTransformer,
    SmallDataWindowTransformer,
    StandardRowMeanCenter,
    T2RForecaster,
    WindowStandardRowMeanCenterMTS,
    WindowStandardRowMeanCenterUTS,
    cubic,
    flatten_iterative,
    linear,
)
from lale.lib.lale import Hyperopt
from lale.lib.sklearn import RandomForestRegressor, SimpleImputer


class TestAutoaiTSLibs(unittest.TestCase):
    def setUp(self):
        self.data = fetch_household_power_consumption()
        self.data = self.data.iloc[1::5000, :]
        self.X = self.data["Date"].to_numpy()
        self.y = self.data.drop(columns=["Date", "Time"]).applymap(
            lambda x: 0 if x == "?" else x
        )
        self.y = self.y.astype("float64").fillna(0).to_numpy()

    def doTestPipeline(
        self, trainable_pipeline, train_X, train_y, test_X, test_y, optimization=False
    ):
        def adjusted_smape(y_true, y_pred):
            """
            SMAPE
            """
            y_true, y_pred = np.array(y_true).ravel(), np.array(y_pred).ravel()
            if len(y_true) != len(y_pred):
                print(
                    "Size of Ground Truth and Predicted Values do not match!, returning None."
                )
                # May be raising error will interfere with daub execution if one pipeline fails
                # raise ValueError('Size of Ground Truth and Predicted Values do not match!')
                return None

            pred_diff = 2.0 * np.abs(cast(float, y_true - y_pred))
            divide = np.abs(y_true) + np.abs(y_pred)
            divide[divide < 1e-12] = 1.0
            scores = pred_diff / divide
            scores = np.array(scores, dtype=float)
            return np.nanmean(scores) * 100.0

        trained_pipeline = trainable_pipeline.fit(train_X, train_y)
        predicted = trained_pipeline.predict(test_X[:-1])
        if optimization:
            print(adjusted_smape(test_X[:-1], predicted))
        else:
            print(adjusted_smape(test_X[-1], predicted))
        with self.assertWarns(DeprecationWarning):
            trainable_pipeline.predict(train_X)
        trainable_pipeline.to_json()
        if optimization:
            hyperopt = Hyperopt(
                estimator=trainable_pipeline,
                max_evals=2,
                verbose=True,
                cv=TimeSeriesSplit(),
                scoring=make_scorer(adjusted_smape),
            )
            trained_hyperopt = hyperopt.fit(train_X, train_y)
            trained_hyperopt.predict(test_X)

    def test_pipeline_AWTTR_1(self):
        trainable = AutoaiTSPipeline(
            steps=[
                (
                    "AutoaiWindowTransformedTargetRegressor",
                    AutoaiWindowTransformedTargetRegressor(
                        regressor=SmallDataWindowTransformer()
                        >> SimpleImputer()
                        >> RandomForestRegressor()
                    ),
                )
            ]
        )
        self.doTestPipeline(trainable, self.y, self.y, self.y, self.y)

    def test_pipeline_AWTTR_2(self):
        trainable = AutoaiTSPipeline(
            steps=[
                (
                    "AutoaiWindowTransformedTargetRegressor",
                    AutoaiWindowTransformedTargetRegressor(
                        regressor=SmallDataWindowTransformer()
                        >> SimpleImputer()
                        >> RandomForestRegressor(),
                        estimator_prediction_type="rowwise",
                    ),
                )
            ]
        )
        self.doTestPipeline(
            trainable, self.y, self.y, self.y, self.y, optimization=True
        )

    def test_pipeline_SDWTT(self):
        trainable = AutoaiTSPipeline(
            steps=[
                (
                    "AutoaiWindowTransformedTargetRegressor",
                    AutoaiWindowTransformedTargetRegressor(
                        regressor=SmallDataWindowTargetTransformer(prediction_horizon=2)
                        >> SimpleImputer()
                        >> RandomForestRegressor(),
                        estimator_prediction_type="rowwise",
                    ),
                )
            ]
        )
        self.doTestPipeline(
            trainable, self.y, self.y, self.y, self.y, optimization=True
        )

    def test_pipeline_AWWR(self):
        trainable = AutoaiTSPipeline(
            steps=[
                (
                    "AutoaiWindowTransformedTargetRegressor",
                    AutoaiWindowedWrappedRegressor(
                        regressor=SmallDataWindowTransformer()
                        >> SimpleImputer()
                        >> RandomForestRegressor()
                    ),
                )
            ]
        )
        self.doTestPipeline(
            trainable, self.y, self.y, self.y, self.y, optimization=True
        )

    def test_pipeline_MT2RF(self):
        trainable = MT2RForecaster(target_columns=[0, 1, 2, 3, 4, 5, 6], trend="Linear")
        self.doTestPipeline(
            trainable, self.y, self.y, self.y, self.y, optimization=False
        )

    def test_pipeline_T2RF(self):
        trainable = T2RForecaster(trend="Linear")
        self.doTestPipeline(
            trainable,
            self.y[:, 0],
            self.y[:, 0],
            self.y[:, 0],
            self.y[:, 0],
            optimization=False,
        )


class StandardRowMeanCenterTest(unittest.TestCase):
    def setUp(self):
        infile = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/AirPassengers.csv"
        cols = ["ID", "time", "AirPassengers"]
        df = pd.read_csv(
            infile, names=cols, sep=r",", index_col="ID", engine="python", skiprows=1
        )
        trainnum = 100
        self.trainset = df.iloc[:trainnum, 1].values
        self.trainset = self.trainset.reshape(-1, 1)
        self.testset = df.iloc[trainnum:, 1].values
        self.testset = self.testset.reshape(-1, 1)

        self.lookback_window = 10
        self.prediction_horizon = 1

        # test corner case
        X, y = make_regression(
            n_features=10, n_informative=2, random_state=0, shuffle=False
        )
        self.X = X
        self.y = y

    def test_standard_row_mean_center_transformer(self):
        transformer = StandardRowMeanCenter()
        self.assertIsNotNone(transformer)
        trained = transformer.fit(self.trainset, self.trainset)
        (X_train, y_train) = trained.transform(self.trainset, self.trainset)
        self.assertTrue(X_train.shape[1] > 0)
        self.assertTrue(y_train.shape[1] > 0)

    def test_window_standard_row_mean_center_transformer_uts(self):
        transformer = WindowStandardRowMeanCenterUTS()
        self.assertIsNotNone(transformer)
        trained = transformer.fit(self.trainset, self.trainset)
        (X_train, y_train) = trained.transform(self.trainset, self.trainset)
        self.assertTrue(X_train.shape[1] > 0)
        self.assertTrue(y_train.shape[1] > 0)

    def test_window_standard_row_mean_center_transformer_mts(self):
        transformer = WindowStandardRowMeanCenterMTS()
        self.assertIsNotNone(transformer)

        trained = transformer.fit(self.X, self.X)
        (X_train, y_train) = trained.transform(self.trainset, self.trainset)
        self.assertTrue(X_train.shape[1] > 0)
        self.assertTrue(y_train.shape[1] > 0)

    def tearDown(self):
        pass


class TimeseriesWindowTransformerTest(unittest.TestCase):
    def setUp(self):
        infile = "https://vincentarelbundock.github.io/Rdatasets/csv/datasets/AirPassengers.csv"
        cols = ["ID", "time", "AirPassengers"]
        df = pd.read_csv(
            infile, names=cols, sep=r",", index_col="ID", engine="python", skiprows=1
        )
        trainnum = 100
        self.trainset = df.iloc[:trainnum, 1].values
        self.trainset = self.trainset.reshape(-1, 1)
        self.testset = df.iloc[trainnum:, 1].values
        self.testset = self.testset.reshape(-1, 1)

        self.lookback_window = 10
        self.prediction_horizon = 1

        # test corner case
        X, y = make_regression(
            n_features=10, n_informative=2, random_state=0, shuffle=False
        )
        self.X = X
        self.y = y

    def check_transformer(self, transformer, X):
        tr = transformer.fit(X)
        self.assertIsNotNone(tr)
        Xt = tr.transform(X)
        self.assertEqual(X.shape[0], Xt.shape[0])
        return Xt

    def test_small_data_window_transformer(self):
        transformer = SmallDataWindowTransformer(lookback_window=None)
        self.assertIsNotNone(transformer)
        Xt = self.check_transformer(transformer=transformer, X=self.trainset)
        self.assertTrue(Xt.shape[1] > 0)

        transformer = SmallDataWindowTransformer(
            lookback_window=self.lookback_window, cache_last_window_trainset=True
        )
        self.assertIsNotNone(transformer)
        Xt = self.check_transformer(transformer=transformer, X=self.trainset)
        self.assertTrue(Xt.shape[1] > 0)

        Xtest = transformer.transform(X=self.testset)
        self.assertTrue(Xtest.shape[1] > 0)

        transformer = SmallDataWindowTransformer(lookback_window=200)
        Xt = self.check_transformer(transformer=transformer, X=self.trainset)
        self.assertTrue(Xt.shape[1] > 0)

        transformer = SmallDataWindowTransformer(lookback_window=None)
        ftransformer = transformer.fit(X=self.X)
        self.assertIsNotNone(ftransformer)

    def test_small_data_window_target_transformer(self):
        transformer = SmallDataWindowTargetTransformer(
            prediction_horizon=self.prediction_horizon
        )
        self.assertIsNotNone(transformer)
        Yt = self.check_transformer(transformer=transformer, X=self.trainset)
        self.assertEqual(np.count_nonzero(np.isnan(Yt)), 1)

    # def test_tested(self):
    #     self.assertTrue(False, "this module was tested")

    def tearDown(self):
        pass


class TestMT2RForecaster(unittest.TestCase):
    """class for testing MT2RForecaster"""

    @classmethod
    def setUp(cls):
        x = np.arange(30)
        y = np.arange(300, 330)
        X = np.array([x, y])
        X = np.transpose(X)
        cls.target_columns = [0, 1]
        cls.X = X

    def test_fit(self):
        """method for testing the fit method of MT2RForecaster"""
        test_class = self.__class__
        model = MT2RForecaster(target_columns=test_class.target_columns)
        model.fit(test_class.X)

    def test_predict_multicols(self):
        """Tests the multivariate predict method of MT2RForecaster"""
        test_class = self.__class__
        model = MT2RForecaster(
            target_columns=test_class.target_columns, prediction_win=2
        )
        fitted_model = model.fit(test_class.X)
        ypred = fitted_model.predict()
        self.assertEqual(len(ypred), 2)
        model = MT2RForecaster(
            target_columns=test_class.target_columns, prediction_win=1
        )
        fitted_model = model.fit(test_class.X)
        ypred = fitted_model.predict()
        self.assertEqual(len(ypred), 1)
        model = MT2RForecaster(target_columns=test_class.target_columns, trend="Mean")
        fitted_model = model.fit(test_class.X)
        ypred = fitted_model.predict()
        self.assertEqual(len(ypred), 12)
        model = MT2RForecaster(target_columns=test_class.target_columns, trend="Poly")
        fitted_model = model.fit(test_class.X)
        ypred = fitted_model.predict()
        self.assertEqual(len(ypred), 12)

    def test_predict_prob(self):
        """Tests predict_prob method of MT2RForecaster"""
        test_class = self.__class__
        model = MT2RForecaster(target_columns=[0], prediction_win=2)
        fitted_model = model.fit(test_class.X)
        ypred = fitted_model.predict_proba()
        assert ypred is not None
        self.assertEqual(ypred.shape, (2, 1, 2))
        model = MT2RForecaster(target_columns=[0, 1], prediction_win=2)
        fitted_model = model.fit(test_class.X)
        ypred = fitted_model.predict_proba()
        self.assertIsNotNone(ypred)
        assert ypred is not None
        self.assertEqual(ypred.shape, (2, 2, 2))

    def test_predict_uni_cols(self):
        """Tests the univariate predict method of MT2RForecaster"""
        x = np.arange(10)
        X = x.reshape(-1, 1)
        model = MT2RForecaster(target_columns=[0], prediction_win=2)
        fitted_model = model.fit(X)
        ypred = fitted_model.predict()
        self.assertEqual(len(ypred), 2)
        model = MT2RForecaster(target_columns=[0], prediction_win=1)
        fitted_model = model.fit(X)
        ypred = fitted_model.predict()
        self.assertEqual(len(ypred), 1)


class TestPrettyPrint(unittest.TestCase):
    def test_param_grid(self):
        printed1 = """from autoai_ts_libs.srom.estimators.time_series.models.srom_estimators import (
    LocalizedFlattenAutoEnsembler,
)
from autoai_ts_libs.srom.estimators.regression.auto_ensemble_regressor import (
    EnsembleRegressor,
)
from autoai_ts_libs.srom.joint_optimizers.auto.auto_regression import (
    AutoRegression,
)
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression
import autoai_ts_libs.srom.joint_optimizers.cv.time_series_splits
import autoai_ts_libs.srom.joint_optimizers.pipeline.srom_param_grid
import sklearn.metrics
import autoai_ts_libs.srom.joint_optimizers.utils.no_op
import sklearn.preprocessing
import sklearn.multioutput
import sklearn.linear_model
import xgboost.sklearn
import lale

lale.wrap_imported_operators()
linear_regression = LinearRegression(n_jobs=1)
multi_output_regressor = MultiOutputRegressor(
    estimator=linear_regression, n_jobs=5
)
auto_regression = AutoRegression(
    cv=autoai_ts_libs.srom.joint_optimizers.cv.time_series_splits.TimeSeriesTrainTestSplit(
        n_splits=1, n_test_size=423, overlap_len=0
    ),
    execution_time_per_pipeline=3,
    level="default",
    param_grid=autoai_ts_libs.srom.joint_optimizers.pipeline.srom_param_grid.SROMParamGrid(),
    scoring=sklearn.metrics.make_scorer(
        sklearn.metrics.mean_absolute_error, greater_is_better=False
    ),
    stages=[
        [
            (
                "skipscaling",
                autoai_ts_libs.srom.joint_optimizers.utils.no_op.NoOp(),
            ),
            ("minmaxscaler", sklearn.preprocessing.MinMaxScaler()),
        ],
        [
            (
                "molinearregression",
                sklearn.multioutput.MultiOutputRegressor(
                    estimator=sklearn.linear_model.LinearRegression(n_jobs=1),
                    n_jobs=5,
                ),
            ),
            (
                "mosgdregressor",
                sklearn.multioutput.MultiOutputRegressor(
                    estimator=sklearn.linear_model.SGDRegressor(
                        random_state=0
                    ),
                    n_jobs=5,
                ),
            ),
            (
                "moxgbregressor",
                sklearn.multioutput.MultiOutputRegressor(
                    estimator=xgboost.sklearn.XGBRegressor(
                        missing=float("nan"), objective="reg:squarederror"
                    ),
                    n_jobs=5,
                ),
            ),
        ],
    ],
    total_execution_time=3,
    best_estimator_so_far=multi_output_regressor,
)
ensemble_regressor = EnsembleRegressor(
    cv=None,
    execution_platform=None,
    execution_time_per_pipeline=None,
    level=None,
    n_estimators_for_pred_interval=1,
    n_leaders_for_ensemble=1,
    num_option_per_pipeline_for_intelligent_search=None,
    num_options_per_pipeline_for_random_search=None,
    save_prefix=None,
    total_execution_time=None,
    auto_regression=auto_regression,
)
pipeline = LocalizedFlattenAutoEnsembler(
    feature_columns=[0, 1],
    target_columns=[0, 1],
    lookback_win=10,
    pred_win=5,
    dag_granularity="multioutput_flat",
    execution_time_per_pipeline=3,
    init_time_optimization=True,
    multistep_prediction_strategy="multioutput",
    multistep_prediction_win=5,
    n_estimators_for_pred_interval=1,
    n_jobs=5,
    n_leaders_for_ensemble=1,
    store_lookback_history=True,
    total_execution_time=3,
    estimator=ensemble_regressor,
)"""
        globals2 = {}
        locals2 = {}
        exec(printed1, globals2, locals2)
        pipeline2 = locals2["pipeline"]
        printed2 = pipeline2.pretty_print()
        self.maxDiff = None
        self.assertEqual(printed1, printed2)


class TestWatForeForecasters(unittest.TestCase):
    def setUp(self):
        print(
            "..................................Watfore TSLIB tests.................................."
        )

    @unittest.skip("Does not work, not sure if it is supposed to work.")
    def test_watfore_pickle_write(self):
        print(
            "................................Watfore TSLIB Training and Pickle Write.................................."
        )
        # for this make sure sys.modules['ai4ml_ts.estimators'] = watfore is removed from init otherwise package is confused
        import pickle

        from lale.lib.autoai_ts_libs import WatForeForecaster

        fr = WatForeForecaster(
            algorithm="hw", samples_per_season=0.6
        )  # , initial_training_seasons=3
        ts_val = [
            [0, 10.0],
            [1, 20.0],
            [2, 30.0],
            [3, 40.0],
            [4, 50.0],
            [5, 60.0],
            [6, 70.0],
            [7, 31.0],
            [8, 80.0],
            [9, 90.0],
        ]

        ml = fr.fit(ts_val, None)
        pr_before = fr.predict(None)
        fr2 = None
        print("Predictions before saving", pr_before)
        f_name = "./lale/datasets/autoai/watfore_pipeline.pickle"  # ./tests/watfore/watfore_pipeline.pickle
        with open(f_name, "wb") as pkl_dump:
            pickle.dump(ml, pkl_dump)
        pr2 = fr.predict(None)
        print("Predictions after pikcle dump", pr2)
        self.assertTrue(
            (np.asarray(pr2).ravel() == np.asarray(pr_before).ravel()).all()
        )
        print("Pickle Done. Now loading...")
        with open(f_name, "rb") as pkl_dump:
            fr2 = pickle.load(pkl_dump)
        if fr2 is not None:
            preds = fr2.predict(None)
            print("Predictions after loading", preds)
            self.assertTrue(len(preds) == 1)
            self.assertTrue(
                (np.asarray(preds).ravel() == np.asarray(pr_before).ravel()).all()
            )
        else:
            print("Failed to Load model(s) from location" + f_name)
            self.fail("Failed to Load model(s) from location" + f_name)
        print(
            "................................Watfor TSLIB Pickle Write Done........................."
        )

    @unittest.skip("Does not work, not sure if it is supposed to work.")
    def test_load_watfore_pipeline(self):

        print(
            "..................................Watfore TSLIB Pickle load and Predict................................."
        )
        import pickle

        # f_name = './tests/watfore/watfore_pipeline.pickle'
        f_name = "watfore_pipeline.pickle"
        print("Pickle Done. Now loading...")
        with open(f_name, "rb") as pkl_dump:
            fr2 = pickle.load(pkl_dump)

        if fr2 is not None:
            preds = fr2.predict(None)
            print("Predictions after loading", preds)
            self.assertTrue(len(preds) == 1)

            # self.assertTrue((np.asarray(preds).ravel() == np.asarray(pr_before).ravel()).all())
        else:
            print("Failed to Load model(s) from location" + f_name)
            self.fail("Failed to Load model(s) from location" + f_name)
        print(
            "................................Watfore TSLIB Read and predict Done........................."
        )


class TestImportExport(unittest.TestCase):
    def setUp(self):
        self.data = fetch_household_power_consumption()
        self.data = self.data.iloc[1::5000, :]
        self.X = self.data["Date"].to_numpy()
        self.y = self.data.drop(columns=["Date", "Time"]).applymap(
            lambda x: 0 if x == "?" else x
        )
        self.y = self.y.astype("float64").fillna(0).to_numpy()


def train_test_split(inputX, split_size):
    return inputX[:split_size], inputX[split_size:]


def get_srom_time_series_estimators(
    feature_column_indices,
    target_column_indices,
    lookback_window,
    prediction_horizon,
    time_column_index=-1,
    optimization_stetagy="Once",
    mode="Test",
    number_rows=None,
    n_jobs=cpu_count() - 1,
):
    """
    This method returns the best performing time_series estimators in SROM. The no. of estiamtors
    depend upon the mode of operation. The mode available are 'Test', 'Benchmark' and
    'benchmark_extended'.
    Parameters:
        feature_column_indices (list): feature indices.
        target_column_indices (list): target indices.
        time_column_index (int): time column index.
        lookback_window (int): Look-back window for the models returned.
        prediction_horizon (int): Look-ahead window for the models returned.
        init_time_optimization (string , optional): whether to optimize at the start of automation.
        mode (string, optional) : The available modes are test, benchmark, benchmark_extended.
    """
    srom_estimators = []
    from lale.lib.autoai_ts_libs import EnsembleRegressor

    # adding P21
    srom_estimators.append(
        MT2RForecaster(
            target_columns=target_column_indices,
            trend="Linear",
            lookback_win=lookback_window,
            prediction_win=prediction_horizon,
            n_jobs=n_jobs,
        )
    )

    ensemble_regressor = EnsembleRegressor(
        cv=None,
        execution_platform=None,
        execution_time_per_pipeline=None,
        level=None,
        n_estimators_for_pred_interval=1,
        n_leaders_for_ensemble=1,
        num_option_per_pipeline_for_intelligent_search=None,
        num_options_per_pipeline_for_random_search=None,
        save_prefix=None,
        total_execution_time=None,
    )
    # Setting commong parameters
    auto_est_params = {
        "feature_columns": feature_column_indices,
        "target_columns": target_column_indices,
        "lookback_win": lookback_window,
        "pred_win": prediction_horizon,
        "time_column": time_column_index,
        "execution_platform": "spark_node_random_search",
        "n_leaders_for_ensemble": 1,
        "n_estimators_for_pred_interval": 1,
        "max_samples_for_pred_interval": 1.0,
        "init_time_optimization": True,
        "dag_granularity": "flat",
        "total_execution_time": 3,
        "execution_time_per_pipeline": 3,
        "store_lookback_history": True,
        "n_jobs": n_jobs,
        "estimator": ensemble_regressor,
    }

    if prediction_horizon > 1:
        auto_est_params["multistep_prediction_win"] = prediction_horizon
        auto_est_params["multistep_prediction_strategy"] = "multioutput"
        auto_est_params["dag_granularity"] = "multioutput_flat"
    elif len(target_column_indices) > 1:
        auto_est_params["dag_granularity"] = "multioutput_flat"
    else:
        pass
    auto_est_params["data_transformation_scheme"] = "log"

    # adding P18
    P18_params = copy.deepcopy(auto_est_params)
    srom_estimators.append(FlattenAutoEnsembler(**P18_params))

    # adding P17, Local Model
    P17_params = copy.deepcopy(auto_est_params)
    srom_estimators.append(DifferenceFlattenAutoEnsembler(**P17_params))

    # adding P14m Local Model
    auto_est_params["data_transformation_scheme"] = None
    P14_params = copy.deepcopy(auto_est_params)
    srom_estimators.append(LocalizedFlattenAutoEnsembler(**P14_params))

    return srom_estimators


class TestSROMEnsemblers(unittest.TestCase):
    """Test various SROM Ensemblers classes"""

    @unittest.skip(
        "Does not work as the fit complains that there is no best_estimator_so_far"
    )
    def test_fit_predict_predict_sliding_window_univariate_single_step(self):
        X = np.arange(1, 441)
        X = X.reshape(-1, 1)
        SIZE = len(X)
        target_columns = [0]
        number_rows = SIZE
        prediction_horizon = 1
        lookback_window = 10
        run_mode = "test"

        srom_estimators = get_srom_time_series_estimators(
            feature_column_indices=target_columns,
            target_column_indices=target_columns,
            lookback_window=lookback_window,
            prediction_horizon=prediction_horizon,
            optimization_stetagy="Once",
            mode=run_mode,
            number_rows=number_rows,
        )
        for index, estimator in enumerate(srom_estimators[1:]):
            X_train, X_test = train_test_split(
                X, SIZE - (prediction_horizon + lookback_window)
            )
            estimator.fit(X_train)
            y_pred = estimator.predict(X_test)
            assert len(y_pred) == prediction_horizon
            assert y_pred.shape[1] == len(target_columns)
            y_pred_win = estimator.predict_sliding_window(X_test)
            assert len(y_pred_win) == lookback_window + 1
            assert y_pred_win.shape[1] == len(target_columns)

    @unittest.skip(
        "Does not work as the fit complains that there is no best_estimator_so_far"
    )
    def test_fit_predict_predict_sliding_window_univariate_multi_step(self):
        X = np.arange(1, 441)
        X = X.reshape(-1, 1)
        SIZE = len(X)
        target_columns = [0]
        number_rows = SIZE
        prediction_horizon = 8
        lookback_window = 10
        run_mode = "test"

        srom_estimators = get_srom_time_series_estimators(
            feature_column_indices=target_columns,
            target_column_indices=target_columns,
            lookback_window=lookback_window,
            prediction_horizon=prediction_horizon,
            optimization_stetagy="Once",
            mode=run_mode,
            number_rows=number_rows,
        )
        for index, estimator in enumerate(srom_estimators[1:]):
            X_train, X_test = train_test_split(
                X, SIZE - (prediction_horizon + lookback_window)
            )
            estimator.fit(X_train)
            y_pred = estimator.predict(X_test)
            assert len(y_pred) == prediction_horizon
            assert y_pred.shape[1] == len(target_columns)
            y_pred_win = estimator.predict_multi_step_sliding_window(X_test)
            assert y_pred_win.shape[1] == len(target_columns)

    @unittest.skip(
        "Does not work as the fit complains that there is no best_estimator_so_far"
    )
    def test_fit_predict_predict_sliding_window_multivariate_single_step(self):
        X = np.arange(1, 441)
        X = X.reshape(-1, 1)
        X2 = np.arange(1001, 1441)
        X2 = X2.reshape(-1, 1)
        X3 = np.arange(10001, 10441)
        X3 = X3.reshape(-1, 1)
        X = np.hstack([X, X2, X3])
        SIZE = len(X)
        target_columns = [0, 1, 2]
        number_rows = SIZE
        prediction_horizon = 1
        lookback_window = 10
        run_mode = "test"
        srom_estimators = get_srom_time_series_estimators(
            feature_column_indices=target_columns,
            target_column_indices=target_columns,
            lookback_window=lookback_window,
            prediction_horizon=prediction_horizon,
            optimization_stetagy="Once",
            mode=run_mode,
            number_rows=number_rows,
        )
        for index, estimator in enumerate(srom_estimators[1:]):
            X_train, X_test = train_test_split(
                X, SIZE - (prediction_horizon + lookback_window)
            )
            estimator.fit(X_train)
            y_pred = estimator.predict(X_test)
            assert len(y_pred) == prediction_horizon
            assert y_pred.shape[1] == len(target_columns)
            y_pred_win = estimator.predict_sliding_window(X_test)
            assert len(y_pred_win) == lookback_window + 1
            assert y_pred_win.shape[1] == len(target_columns)

    @unittest.skip(
        "Does not work as the fit complains that there is no best_estimator_so_far"
    )
    def test_fit_predict_predict_sliding_window_multivariate_multi_step(self):
        X = np.arange(1, 441)
        X = X.reshape(-1, 1)
        X2 = np.arange(1001, 1441)
        X2 = X2.reshape(-1, 1)
        X3 = np.arange(10001, 10441)
        X3 = X3.reshape(-1, 1)
        X = np.hstack([X, X2, X3])
        SIZE = len(X)
        target_columns = [0, 1, 2]
        number_rows = SIZE
        prediction_horizon = 8
        lookback_window = 10
        run_mode = "test"
        srom_estimators = get_srom_time_series_estimators(
            feature_column_indices=target_columns,
            target_column_indices=target_columns,
            lookback_window=lookback_window,
            prediction_horizon=prediction_horizon,
            optimization_stetagy="Once",
            mode=run_mode,
            number_rows=number_rows,
        )
        for index, estimator in enumerate(srom_estimators[1:]):
            X_train, X_test = train_test_split(
                X, SIZE - (prediction_horizon + lookback_window)
            )
            estimator.fit(X_train)
            y_pred = estimator.predict(X_test)
            assert len(y_pred) == prediction_horizon
            assert y_pred.shape[1] == len(target_columns)
            y_pred_win = estimator.predict_multi_step_sliding_window(X_test)
            assert y_pred_win.shape[1] == len(target_columns)


class TestInterpolatorImputers(unittest.TestCase):
    """class for testing different time-series imputers."""

    @classmethod
    def setUp(cls):
        uni_x = pd.DataFrame({"A": [12, 4, 5, None, 1]})
        multi_x = pd.DataFrame(
            {
                "A": [12, 4, 5, None, 1],
                "B": [None, 2, 54, 3, None],
                "C": [20, 16, None, 3, 8],
                "D": [14, 3, None, None, 6],
            }
        )
        cls.uni_x = uni_x
        cls.multi_x = multi_x

    def test_linear_imputer(self):
        """Test  LinearImputer"""
        test_class = self.__class__
        uni_x = test_class.uni_x
        multi_x = test_class.multi_x

        # Test univariate
        imputer = linear().convert_to_trained()
        X_tf = imputer.transform(uni_x)
        self.assertAlmostEqual(np.mean(X_tf), 5.0, 2)

        # Test multivariate
        imputer = linear().convert_to_trained()
        X_tf = imputer.transform(multi_x)
        self.assertAlmostEqual(np.mean(X_tf), 8.875, 2)

        # Test numpy
        imputer = linear().convert_to_trained()
        X_tf = imputer.transform(uni_x.values)
        self.assertAlmostEqual(np.mean(X_tf), 5.0, 2)

        # Test numpy
        imputer = linear().convert_to_trained()
        X_tf = imputer.transform(multi_x.values)
        self.assertAlmostEqual(np.mean(X_tf), 8.875, 2)

    def test_cubic_imputer(self):
        """Test  CubicImputer"""
        test_class = self.__class__
        uni_x = test_class.uni_x
        multi_x = test_class.multi_x[["A", "C"]]

        # Test univariate
        imputer = cubic().convert_to_trained()
        X_tf = imputer.transform(uni_x)
        self.assertAlmostEqual(np.mean(X_tf), 5.75, 2)

        # Test multivariate
        imputer = cubic().convert_to_trained()
        X_tf = imputer.transform(multi_x)
        self.assertAlmostEqual(np.mean(X_tf), 8.375, 2)

        # Test numpy
        imputer = cubic().convert_to_trained()
        X_tf = imputer.transform(uni_x.values)
        self.assertAlmostEqual(np.mean(X_tf), 5.75, 2)

        # Test numpy
        imputer = cubic().convert_to_trained()
        X_tf = imputer.transform(multi_x.values)
        self.assertAlmostEqual(np.mean(X_tf), 8.375, 2)


class TestFlattenImputers(unittest.TestCase):
    """class for testing different Flatten imputers"""

    @classmethod
    def setUp(cls):
        uni_x = pd.DataFrame({"A": [1, 2, 3, 4, 5, None, 7, 8, 9]})
        multi_x = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5, None, 7, 8, 9, 10],
                "B": [101, 102, None, 104, 105, 106, 107, 108, 109, 110],
                "C": [51, 52, None, 54, 55, 56, None, 58, 59, 60],
            }
        )
        imputers = [flatten_iterative]
        cls.uni_x = uni_x
        cls.multi_x = multi_x
        cls.imputers = imputers

    def test_fit_transform_flatten_imputers(self):
        """
        Test Fit and transform flatten imputers
        """
        test_class = self.__class__
        uni_x = test_class.uni_x
        multi_x = test_class.multi_x

        # Test univariate timeseries data
        est = flatten_iterative()
        try:
            for est in test_class.imputers:
                print("testing", est)
                imputer = est(order=5)
                imputer.fit(uni_x)
                interpolated = imputer.transform(uni_x)
                self.assertFalse(np.any(np.isnan(interpolated)))
        except Exception as e:
            self.fail("Failed : " + str(est.name()) + " " + str(e))

        # Test multivariate timeseries data
        try:
            for est in test_class.imputers:
                print("testing", est)
                imputer = est(order=5)
                imputer.fit(multi_x)
                interpolated = imputer.transform(multi_x)
                self.assertFalse(np.any(np.isnan(interpolated)))
        except Exception as e:
            self.fail("Failed : " + str(est.name()) + " " + str(e))

    def test_fit_transform_flatten_imputers_without_nan(self):
        """
        Test Fit and transform flatten imputers with nan
        """
        test_class = self.__class__
        uni_x = pd.DataFrame({"A": [1, 2, 3, 4, 5, 6, 7, 8, 9]})
        multi_x = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "B": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                "C": [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
            }
        )
        est = flatten_iterative()
        # Test univariate timeseries data
        try:
            for est in test_class.imputers:
                print("testing", est)
                imputer = est(order=5)
                imputer.fit(uni_x)
                interpolated = imputer.transform(uni_x)
                self.assertFalse(np.any(np.isnan(interpolated)))
        except Exception as e:
            self.fail("Failed : " + str(est.name()) + " " + str(e))

        # Test multivariate timeseries data
        try:
            for est in test_class.imputers:
                print("testing", est)
                imputer = est(order=5)
                imputer.fit(multi_x)
                interpolated = imputer.transform(multi_x)
                self.assertFalse(np.any(np.isnan(interpolated)))
        except Exception as e:
            self.fail("Failed : " + str(est.name()) + " " + str(e))

    @unittest.skip("Need to fix.")
    def test_set_params(self):
        """
        Test set_params
        """
        test_class = self.__class__
        PARAMS = {
            flatten_iterative: {
                "base_imputer__random_state": 24,
            }
        }
        est = flatten_iterative()
        try:
            for est in test_class.imputers[0:1]:
                params = PARAMS[est]
                imputer = est()
                imputer.set_params(**params)
                self.assertEqual(
                    PARAMS[est]["base_imputer__random_state"],
                    imputer.base_imputer.get_params()["random_state"],
                )
        except Exception as e:
            self.fail("Failed : " + str(est.name()) + " " + str(e))

    def test_get_params(self):
        """
        Test get_params
        """
        test_class = self.__class__
        est = flatten_iterative()
        try:
            for est in test_class.imputers[0:1]:
                imputer = est()
                self.assertIsNotNone(imputer.get_params())
        except Exception as e:
            self.fail("Failed : " + str(est.name()) + " " + str(e))

    def test_fit_transform_flatten_imputers_with_less_data(self):
        """
        Test Fit and transform flatten imputers with less data
        """
        test_class = self.__class__
        est = flatten_iterative()
        multi_x = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "B": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
                "C": [51, 52, 53, 54, 55, 56, 57, 58, 59, 60],
            }
        )
        test_x = pd.DataFrame(
            {
                "A": [7, 8, 9, 10],
                "B": [107, 108, 109, 110],
                "C": [57, 58, 59, 60],
            }
        )
        # Test multivariate timeseries data
        try:
            for est in test_class.imputers:
                print("testing", est)
                imputer = est(order=5)
                imputer.fit(multi_x)
                interpolated = imputer.transform(test_x)
                self.assertFalse(np.any(np.isnan(interpolated)))
        except Exception as e:
            self.fail("Failed : " + str(est.name()) + " " + str(e))
        test_x = pd.DataFrame(
            {
                "A": [7, None, 9, 10],
                "B": [107, None, 109, 110],
                "C": [57, 58, 59, 60],
            }
        )
        try:
            for est in test_class.imputers:
                print("testing", est)
                imputer = est(order=5)
                imputer.fit(multi_x)
                self.assertRaises(Exception, imputer, "transform", test_x)
        except Exception as e:
            self.fail("Failed : " + str(est.name()) + " " + str(e))

    def test_fit_transform_flatten_imputers_with_other_missing_value(self):
        """
        Test Fit and transform flatten imputers
        """
        test_class = self.__class__
        uni_x = pd.DataFrame({"A": [1, 2, 3, 4, 5, -999, 7, 8, 9]})
        multi_x = pd.DataFrame(
            {
                "A": [1, 2, 3, 4, 5, -999, 7, 8, 9, 10],
                "B": [101, 102, -999, 104, 105, 106, 107, 108, 109, 110],
                "C": [51, 52, -999, 54, 55, 56, -999, 58, 59, 60],
            }
        )
        est = flatten_iterative()
        # Test univariate timeseries data
        try:
            for est in test_class.imputers:
                print("testing", est)
                imputer = est(order=5, missing_val_identifier=-999)
                imputer.fit(uni_x)
                interpolated = imputer.transform(uni_x)
                self.assertFalse(np.any(np.isnan(interpolated)))
        except Exception as e:
            self.fail("Failed : " + str(est.name()) + " " + str(e))

        # Test multivariate timeseries data
        try:
            for est in test_class.imputers:
                print("testing", est)
                imputer = est(order=5, missing_val_identifier=-999)
                imputer.fit(multi_x)
                interpolated = imputer.transform(multi_x)
                self.assertFalse(np.any(np.isnan(interpolated)))
        except Exception as e:
            self.fail("Failed : " + str(est.name()) + " " + str(e))


class TestSchemas(unittest.TestCase):
    def setUp(self):
        pass


def create_function_test_schemas(obj_name):
    def test_schemas(self):
        import importlib

        module_name = ".".join(obj_name.split(".")[0:-1])
        class_name = obj_name.split(".")[-1]
        module = importlib.import_module(module_name)

        class_ = getattr(module, class_name)
        if class_name == "MT2RForecaster":
            obj = class_(target_columns=[0])
        else:
            obj = class_()

        obj._check_schemas()
        # test_schemas_are_schemas
        lale.type_checking.validate_is_schema(obj.hyperparam_schema())

    test_schemas.__name__ = "test_{0}".format(obj.split(".")[-1])
    return test_schemas


objs = [
    # "lale.lib.autoai_ts_libs.AutoaiTSPipeline",  # does not work as steps can be None or empty
    "lale.lib.autoai_ts_libs.AutoaiWindowedWrappedRegressor",
    "lale.lib.autoai_ts_libs.AutoaiWindowTransformedTargetRegressor",
    "lale.lib.autoai_ts_libs.MT2RForecaster",
    "lale.lib.autoai_ts_libs.SmallDataWindowTargetTransformer",
    "lale.lib.autoai_ts_libs.SmallDataWindowTransformer",
    "lale.lib.autoai_ts_libs.StandardRowMeanCenter",
    "lale.lib.autoai_ts_libs.T2RForecaster",
    "lale.lib.autoai_ts_libs.WindowStandardRowMeanCenterMTS",
    "lale.lib.autoai_ts_libs.cubic",
    "lale.lib.autoai_ts_libs.flatten_iterative",
    "lale.lib.autoai_ts_libs.linear",
    "lale.lib.autoai_ts_libs.fill",
    "lale.lib.autoai_ts_libs.previous",
    "lale.lib.autoai_ts_libs.next",
    "lale.lib.autoai_ts_libs.AutoRegression",
    "lale.lib.autoai_ts_libs.EnsembleRegressor",
    "lale.lib.autoai_ts_libs.WatForeForecaster",
]
for obj in objs:
    setattr(
        TestSchemas,
        "test_{0}".format(obj.split(".")[-1]),
        create_function_test_schemas(obj),
    )
