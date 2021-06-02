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

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.metrics import make_scorer
from sklearn.model_selection import TimeSeriesSplit

from lale.datasets.uci import fetch_household_power_consumption
from lale.lib.lale import Hyperopt
from lale.lib.sklearn import RandomForestRegressor, SimpleImputer

from lale.lib.autoai_ts_libs import (  # StandardRowMeanCenterMTS,; WindowTransformerMTS
    AutoaiTSPipeline,
    AutoaiWindowedWrappedRegressor,
    AutoaiWindowTransformedTargetRegressor,
    MT2RForecaster,
    SmallDataWindowTargetTransformer,
    SmallDataWindowTransformer,
    StandardRowMeanCenter,
    T2RForecaster,
    WindowStandardRowMeanCenterMTS,
    WindowStandardRowMeanCenterUTS,
)


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

            pred_diff = 2.0 * np.abs(y_true - y_pred)
            divide = np.abs(y_true) + np.abs(y_pred)
            divide[divide < 1e-12] = 1.0
            scores = pred_diff / divide
            scores = np.array(scores, dtype=np.float)
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
            pipeline=AutoaiWindowTransformedTargetRegressor(
                regressor=SmallDataWindowTransformer()
                >> SimpleImputer()
                >> RandomForestRegressor()
            )
        )
        self.doTestPipeline(trainable, self.y, self.y, self.y, self.y)

    def test_pipeline_AWTTR_2(self):
        trainable = AutoaiTSPipeline(
            pipeline=AutoaiWindowTransformedTargetRegressor(
                regressor=SmallDataWindowTransformer()
                >> SimpleImputer()
                >> RandomForestRegressor(),
                estimator_prediction_type="rowwise",
            )
        )
        self.doTestPipeline(
            trainable, self.y, self.y, self.y, self.y, optimization=True
        )

    def test_pipeline_SDWTT(self):
        trainable = AutoaiTSPipeline(
            pipeline=AutoaiWindowTransformedTargetRegressor(
                regressor=SmallDataWindowTargetTransformer(prediction_horizon=2)
                >> SimpleImputer()
                >> RandomForestRegressor(),
                estimator_prediction_type="rowwise",
            )
        )
        self.doTestPipeline(
            trainable, self.y, self.y, self.y, self.y, optimization=True
        )

    def test_pipeline_AWWR(self):
        trainable = AutoaiTSPipeline(
            pipeline=AutoaiWindowedWrappedRegressor(
                regressor=SmallDataWindowTransformer()
                >> SimpleImputer()
                >> RandomForestRegressor()
            )
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
    """ class for testing MT2RForecaster """

    @classmethod
    def setUp(cls):
        x = np.arange(30)
        y = np.arange(300, 330)
        X = np.array([x, y])
        X = np.transpose(X)
        cls.target_columns = [0, 1]
        cls.X = X

    def test_fit(self):
        """ method for testing the fit method of MT2RForecaster"""
        test_class = self.__class__
        model = MT2RForecaster(target_columns=test_class.target_columns)
        model.fit(test_class.X)

    def test_predict_multicols(self):
        """ Tests the multivariate predict method of MT2RForecaster"""
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
        """ Tests predict_prob method of MT2RForecaster"""
        test_class = self.__class__
        model = MT2RForecaster(target_columns=[0], prediction_win=2)
        fitted_model = model.fit(test_class.X)
        ypred = fitted_model.predict_proba()
        self.assertEqual(ypred.shape, (2, 1, 2))
        model = MT2RForecaster(target_columns=[0, 1], prediction_win=2)
        fitted_model = model.fit(test_class.X)
        ypred = fitted_model.predict_proba()
        self.assertEqual(ypred.shape, (2, 2, 2))

    def test_predict_uni_cols(self):
        """ Tests the univariate predict method of MT2RForecaster"""
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


class TestWatForeForecasters(unittest.TestCase):
    def setUp(self):
        print(
            "..................................Watfore TSLIB tests.................................."
        )

    def test_watfore_pickle_write(self):
        print(
            "................................Watfore TSLIB Training and Pickle Write.................................."
        )
        # for this make sure sys.modules['ai4ml_ts.estimators'] = watfore is removed from init otherwise package is confused
        try:
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
            f_name = (
                "watfore_pipeline.pickle"  # ./tests/watfore/watfore_pipeline.pickle
            )
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

        except Exception as e:
            print("Failed to Load model(s)")
            self.fail(e)

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


# class TestImportExport(unittest.TestCase):
#     def setUp(self):
#         self.data = fetch_household_power_consumption()
#         self.data = self.data.iloc[1::5000, :]
#         self.X = self.data["Date"].to_numpy()
#         self.y = self.data.drop(columns=["Date", "Time"]).applymap(
#             lambda x: 0 if x == "?" else x
#         )
#         self.y = self.y.astype("float64").fillna(0).to_numpy()
