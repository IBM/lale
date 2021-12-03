# Copyright 2020, 2021 IBM Corporation
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

import os
import traceback
import unittest
import urllib.request
import zipfile

import aif360
import jsonschema
import numpy as np
import pandas as pd
import sklearn.metrics
import sklearn.model_selection

try:
    import cvxpy  # noqa because the import is only done as a check and flake fails

    cvxpy_installed = True
except ImportError:
    cvxpy_installed = False

try:
    import numba  # noqa because the import is only done as a check and flake fails

    numba_installed = True
except ImportError:
    numba_installed = False

try:
    import tensorflow as tf

    tensorflow_installed = True
except ImportError:
    tensorflow_installed = False

import lale.helpers
import lale.lib.aif360
import lale.lib.aif360.util
from lale.datasets.data_schemas import NDArrayWithSchema
from lale.lib.aif360 import (
    LFR,
    AdversarialDebiasing,
    CalibratedEqOddsPostprocessing,
    DisparateImpactRemover,
    EqOddsPostprocessing,
    GerryFairClassifier,
    MetaFairClassifier,
    OptimPreproc,
    PrejudiceRemover,
    Redacting,
    RejectOptionClassification,
    Reweighing,
    fair_stratified_train_test_split,
)
from lale.lib.lale import ConcatFeatures, Project
from lale.lib.sklearn import (
    FunctionTransformer,
    LinearRegression,
    LogisticRegression,
    OneHotEncoder,
)


class TestAIF360Datasets(unittest.TestCase):
    downloaded_h181 = False
    downloaded_h192 = False

    @classmethod
    def setUpClass(cls) -> None:
        cls.downloaded_h181 = cls._try_download_csv("h181.csv")
        cls.downloaded_h192 = cls._try_download_csv("h192.csv")

    @classmethod
    def tearDownClass(cls) -> None:
        if cls.downloaded_h181:
            cls._cleanup_meps("h181.csv")
        if cls.downloaded_h192:
            cls._cleanup_meps("h192.csv")

    def _attempt_dataset(
        self, X, y, fairness_info, n_rows, n_columns, set_y, di_expected
    ):
        self.assertEqual(X.shape, (n_rows, n_columns))
        self.assertEqual(y.shape, (n_rows,))
        self.assertEqual(set(y), set_y)
        di_scorer = lale.lib.aif360.disparate_impact(**fairness_info)
        di_measured = di_scorer.score_data(X=X, y_pred=y)
        self.assertAlmostEqual(di_measured, di_expected, places=3)

    def test_dataset_adult_pd_cat(self):
        X, y, fairness_info = lale.lib.aif360.fetch_adult_df(preprocess=False)
        self._attempt_dataset(X, y, fairness_info, 48_842, 14, {"<=50K", ">50K"}, 0.227)

    def test_dataset_adult_pd_num(self):
        X, y, fairness_info = lale.lib.aif360.fetch_adult_df(preprocess=True)
        self._attempt_dataset(X, y, fairness_info, 48_842, 100, {0, 1}, 0.227)

    def test_dataset_bank_pd_cat(self):
        X, y, fairness_info = lale.lib.aif360.fetch_bank_df(preprocess=False)
        self._attempt_dataset(X, y, fairness_info, 45_211, 16, {1, 2}, 0.840)

    def test_dataset_bank_pd_num(self):
        X, y, fairness_info = lale.lib.aif360.fetch_bank_df(preprocess=True)
        self._attempt_dataset(X, y, fairness_info, 45_211, 51, {0, 1}, 0.840)

    def test_dataset_compas_pd_cat(self):
        X, y, fairness_info = lale.lib.aif360.fetch_compas_df(preprocess=False)
        self._attempt_dataset(X, y, fairness_info, 6_172, 51, {0, 1}, 0.747)

    def test_dataset_compas_pd_num(self):
        X, y, fairness_info = lale.lib.aif360.fetch_compas_df(preprocess=True)
        self._attempt_dataset(X, y, fairness_info, 5_278, 10, {0, 1}, 0.687)

    def test_dataset_compas_violent_pd_cat(self):
        X, y, fairness_info = lale.lib.aif360.fetch_compas_violent_df(preprocess=False)
        self._attempt_dataset(X, y, fairness_info, 4_020, 51, {0, 1}, 0.852)

    def test_dataset_compas_violent_pd_num(self):
        X, y, fairness_info = lale.lib.aif360.fetch_compas_violent_df(preprocess=True)
        self._attempt_dataset(X, y, fairness_info, 3_377, 10, {0, 1}, 0.822)

    def test_dataset_creditg_pd_cat(self):
        X, y, fairness_info = lale.lib.aif360.fetch_creditg_df(preprocess=False)
        self._attempt_dataset(X, y, fairness_info, 1_000, 20, {"bad", "good"}, 0.748)

    def test_dataset_creditg_pd_num(self):
        X, y, fairness_info = lale.lib.aif360.fetch_creditg_df(preprocess=True)
        self._attempt_dataset(X, y, fairness_info, 1_000, 58, {0, 1}, 0.748)

    def test_dataset_nursery_pd_cat(self):
        X, y, fairness_info = lale.lib.aif360.fetch_nursery_df(preprocess=False)
        self._attempt_dataset(
            X,
            y,
            fairness_info,
            12_960,
            8,
            {"not_recom", "recommend", "very_recom", "priority", "spec_prior"},
            0.461,
        )

    def test_dataset_nursery_pd_num(self):
        X, y, fairness_info = lale.lib.aif360.fetch_nursery_df(preprocess=True)
        self._attempt_dataset(X, y, fairness_info, 12_960, 25, {0, 1}, 0.461)

    def test_dataset_ricci_pd_cat(self):
        X, y, fairness_info = lale.lib.aif360.fetch_ricci_df(preprocess=False)
        self._attempt_dataset(
            X, y, fairness_info, 118, 5, {"No promotion", "Promotion"}, 0.498
        )

    def test_dataset_ricci_pd_num(self):
        X, y, fairness_info = lale.lib.aif360.fetch_ricci_df(preprocess=True)
        self._attempt_dataset(X, y, fairness_info, 118, 6, {0, 1}, 0.498)

    def test_dataset_speeddating_pd_cat(self):
        X, y, fairness_info = lale.lib.aif360.fetch_speeddating_df(preprocess=False)
        self._attempt_dataset(X, y, fairness_info, 8_378, 122, {"0", "1"}, 0.853)

    def test_dataset_speeddating_pd_num(self):
        X, y, fairness_info = lale.lib.aif360.fetch_speeddating_df(preprocess=True)
        self._attempt_dataset(X, y, fairness_info, 8_378, 70, {0, 1}, 0.853)

    def test_dataset_boston_housing_pd_cat(self):
        X, y, fairness_info = lale.lib.aif360._fetch_boston_housing_df(preprocess=False)
        # TODO: consider better way of handling "set_y" parameter for regression problems
        self._attempt_dataset(X, y, fairness_info, 506, 13, set(y), 0.814)

    def test_dataset_boston_housing_pd_num(self):
        X, y, fairness_info = lale.lib.aif360._fetch_boston_housing_df(preprocess=True)
        # TODO: consider better way of handling "set_y" parameter for regression problems
        self._attempt_dataset(X, y, fairness_info, 506, 13, set(y), 0.814)

    def test_dataset_titanic_pd_cat(self):
        X, y, fairness_info = lale.lib.aif360.fetch_titanic_df(preprocess=False)
        self._attempt_dataset(X, y, fairness_info, 1_309, 13, {"0", "1"}, 0.263)

    def test_dataset_titanic_pd_num(self):
        X, y, fairness_info = lale.lib.aif360.fetch_titanic_df(preprocess=True)
        self._attempt_dataset(X, y, fairness_info, 1_309, 37, {0, 1}, 0.263)

    def test_dataset_tae_pd_cat(self):
        X, y, fairness_info = lale.lib.aif360.fetch_tae_df(preprocess=False)
        self._attempt_dataset(X, y, fairness_info, 151, 5, {1, 2, 3}, 0.449)

    def test_dataset_tae_pd_num(self):
        X, y, fairness_info = lale.lib.aif360.fetch_tae_df(preprocess=True)
        self._attempt_dataset(X, y, fairness_info, 151, 6, {0, 1}, 0.449)

    @classmethod
    def _try_download_csv(cls, filename):
        directory = os.path.join(
            os.path.dirname(os.path.abspath(aif360.__file__)), "data", "raw", "meps"
        )
        csv_exists = os.path.exists(
            os.path.join(
                directory,
                filename,
            )
        )
        if csv_exists:
            return False
        else:
            filename_without_extension = os.path.splitext(filename)[0]
            zip_filename = f"{filename_without_extension}ssp.zip"
            urllib.request.urlretrieve(
                f"https://meps.ahrq.gov/mepsweb/data_files/pufs/{zip_filename}",
                os.path.join(directory, zip_filename),
            )
            with zipfile.ZipFile(os.path.join(directory, zip_filename), "r") as zip_ref:
                zip_ref.extractall(directory)
            ssp_filename = f"{filename_without_extension}.ssp"
            df = pd.read_sas(os.path.join(directory, ssp_filename), format="xport")
            df.to_csv(os.path.join(directory, filename), index=False)
            return True

    @classmethod
    def _cleanup_meps(cls, filename):
        directory = os.path.join(
            os.path.dirname(os.path.abspath(aif360.__file__)), "data", "raw", "meps"
        )
        filename_without_extension = os.path.splitext(filename)[0]
        zip_filename = f"{filename_without_extension}ssp.zip"
        ssp_filename = f"{filename_without_extension}.ssp"
        os.remove(os.path.join(directory, filename))
        os.remove(os.path.join(directory, zip_filename))
        os.remove(os.path.join(directory, ssp_filename))

    def test_dataset_meps_panel19_fy2015_pd_cat(self):
        X, y, fairness_info = lale.lib.aif360.fetch_meps_panel19_fy2015_df(
            preprocess=False
        )
        self._attempt_dataset(X, y, fairness_info, 16578, 1825, {0, 1}, 0.496)

    def test_dataset_meps_panel19_fy_2015_pd_num(self):
        X, y, fairness_info = lale.lib.aif360.fetch_meps_panel19_fy2015_df(
            preprocess=True
        )
        self._attempt_dataset(X, y, fairness_info, 15830, 138, {0, 1}, 0.490)

    def test_dataset_meps_panel20_fy2015_pd_cat(self):
        X, y, fairness_info = lale.lib.aif360.fetch_meps_panel20_fy2015_df(
            preprocess=False
        )
        self._attempt_dataset(X, y, fairness_info, 18849, 1825, {0, 1}, 0.493)

    def test_dataset_meps_panel20_fy2015_pd_num(self):
        X, y, fairness_info = lale.lib.aif360.fetch_meps_panel20_fy2015_df(
            preprocess=True
        )
        self._attempt_dataset(X, y, fairness_info, 17570, 138, {0, 1}, 0.488)

    def test_dataset_meps_panel21_fy2016_pd_cat(self):
        X, y, fairness_info = lale.lib.aif360.fetch_meps_panel21_fy2016_df(
            preprocess=False
        )
        self._attempt_dataset(X, y, fairness_info, 17052, 1936, {0, 1}, 0.462)

    def test_dataset_meps_panel21_fy2016_pd_num(self):
        X, y, fairness_info = lale.lib.aif360.fetch_meps_panel21_fy2016_df(
            preprocess=True
        )
        self._attempt_dataset(X, y, fairness_info, 15675, 138, {0, 1}, 0.451)


class TestAIF360Num(unittest.TestCase):
    @classmethod
    def _creditg_pd_num(cls):
        X, y, fairness_info = lale.lib.aif360.fetch_creditg_df(preprocess=True)
        cv = lale.lib.aif360.FairStratifiedKFold(**fairness_info, n_splits=3)
        splits = []
        lr = LogisticRegression()
        for train, test in cv.split(X, y):
            train_X, train_y = lale.helpers.split_with_schemas(lr, X, y, train)
            assert isinstance(train_X, pd.DataFrame), type(train_X)
            assert isinstance(train_y, pd.Series), type(train_y)
            test_X, test_y = lale.helpers.split_with_schemas(lr, X, y, test, train)
            assert isinstance(test_X, pd.DataFrame), type(test_X)
            assert isinstance(test_y, pd.Series), type(test_y)
            splits.append(
                {
                    "train_X": train_X,
                    "train_y": train_y,
                    "test_X": test_X,
                    "test_y": test_y,
                }
            )
        result = {"splits": splits, "fairness_info": fairness_info}
        return result

    @classmethod
    def _creditg_np_num(cls):
        train_X = cls.creditg_pd_num["splits"][0]["train_X"].to_numpy()
        train_y = cls.creditg_pd_num["splits"][0]["train_y"].to_numpy()
        test_X = cls.creditg_pd_num["splits"][0]["test_X"].to_numpy()
        test_y = cls.creditg_pd_num["splits"][0]["test_y"].to_numpy()
        assert isinstance(train_X, np.ndarray), type(train_X)
        assert not isinstance(train_X, NDArrayWithSchema), type(train_X)
        assert isinstance(train_y, np.ndarray), type(train_y)
        assert not isinstance(train_y, NDArrayWithSchema), type(train_y)
        assert isinstance(test_X, np.ndarray), type(test_X)
        assert not isinstance(test_X, NDArrayWithSchema), type(test_X)
        assert isinstance(test_y, np.ndarray), type(test_y)
        assert not isinstance(test_y, NDArrayWithSchema), type(test_y)
        pd_columns = cls.creditg_pd_num["splits"][0]["train_X"].columns
        fairness_info = {
            "favorable_labels": [1],
            "protected_attributes": [
                {
                    "feature": pd_columns.get_loc("sex"),
                    "reference_group": [1],
                },
                {
                    "feature": pd_columns.get_loc("age"),
                    "reference_group": [1],
                },
            ],
        }
        result = {
            "train_X": train_X,
            "train_y": train_y,
            "test_X": test_X,
            "test_y": test_y,
            "fairness_info": fairness_info,
        }
        return result

    @classmethod
    def _boston_pd_num(cls):
        # TODO: Consider investigating test failure when preprocess is set to True
        # (eo_diff is not less than 0 in this case; perhaps regression model learns differently?)
        orig_X, orig_y, fairness_info = lale.lib.aif360._fetch_boston_housing_df(
            preprocess=False
        )
        train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(
            orig_X, orig_y, test_size=0.33, random_state=42
        )
        assert isinstance(train_X, pd.DataFrame), type(train_X)
        assert isinstance(train_y, pd.Series), type(train_y)
        assert isinstance(test_X, pd.DataFrame), type(test_X)
        assert isinstance(test_y, pd.Series), type(test_y)
        result = {
            "train_X": train_X,
            "train_y": train_y,
            "test_X": test_X,
            "test_y": test_y,
            "fairness_info": fairness_info,
        }
        return result

    @classmethod
    def _boston_np_num(cls):
        train_X = cls.boston_pd_num["train_X"].to_numpy()
        train_y = cls.boston_pd_num["train_y"].to_numpy()
        test_X = cls.boston_pd_num["test_X"].to_numpy()
        test_y = cls.boston_pd_num["test_y"].to_numpy()
        assert isinstance(train_X, np.ndarray), type(train_X)
        assert not isinstance(train_X, NDArrayWithSchema), type(train_X)
        assert isinstance(train_y, np.ndarray), type(train_y)
        assert not isinstance(train_y, NDArrayWithSchema), type(train_y)
        assert isinstance(test_X, np.ndarray), type(test_X)
        assert not isinstance(test_X, NDArrayWithSchema), type(test_X)
        assert isinstance(test_y, np.ndarray), type(test_y)
        assert not isinstance(test_y, NDArrayWithSchema), type(test_y)
        pd_columns = cls.boston_pd_num["train_X"].columns
        # pulling attributes off of stored fairness_info to avoid recomputing medians
        fairness_info = {
            "favorable_labels": cls.boston_pd_num["fairness_info"]["favorable_labels"],
            "protected_attributes": [
                {
                    "feature": pd_columns.get_loc("B"),
                    "reference_group": cls.boston_pd_num["fairness_info"][
                        "protected_attributes"
                    ][0]["reference_group"],
                },
            ],
        }
        result = {
            "train_X": train_X,
            "train_y": train_y,
            "test_X": test_X,
            "test_y": test_y,
            "fairness_info": fairness_info,
        }
        return result

    @classmethod
    def setUpClass(cls):
        cls.creditg_pd_num = cls._creditg_pd_num()
        cls.creditg_np_num = cls._creditg_np_num()
        cls.boston_pd_num = cls._boston_pd_num()
        cls.boston_np_num = cls._boston_np_num()

    def test_fair_stratified_train_test_split(self):
        X = self.creditg_np_num["train_X"]
        y = self.creditg_np_num["train_y"]
        fairness_info = self.creditg_np_num["fairness_info"]
        z = range(X.shape[0])
        (
            train_X,
            test_X,
            train_y,
            test_y,
            train_z,
            test_z,
        ) = fair_stratified_train_test_split(X, y, z, **fairness_info)
        self.assertEqual(train_X.shape[0], train_y.shape[0])
        self.assertEqual(train_X.shape[0], len(train_z))
        self.assertEqual(test_X.shape[0], test_y.shape[0])
        self.assertEqual(test_X.shape[0], len(test_z))
        self.assertEqual(train_X.shape[0] + test_X.shape[0], X.shape[0])

    def _attempt_scorers(self, fairness_info, estimator, test_X, test_y):
        fi = fairness_info
        disparate_impact_scorer = lale.lib.aif360.disparate_impact(**fi)
        impact = disparate_impact_scorer(estimator, test_X, test_y)
        self.assertLess(impact, 0.9)
        if estimator.is_classifier():
            combined_scorer = lale.lib.aif360.accuracy_and_disparate_impact(**fi)
            combined = combined_scorer(estimator, test_X, test_y)
            self.assertLess(0.0, combined)
            self.assertLess(combined, 1.0)
        else:
            combined_scorer = lale.lib.aif360.r2_and_disparate_impact(**fi)
            combined = combined_scorer(estimator, test_X, test_y)
            self.assertLess(0.0, combined)
            self.assertLess(combined, 1.0)
        parity_scorer = lale.lib.aif360.statistical_parity_difference(**fi)
        parity = parity_scorer(estimator, test_X, test_y)
        self.assertLess(parity, 0.0)
        eo_diff_scorer = lale.lib.aif360.equal_opportunity_difference(**fi)
        eo_diff = eo_diff_scorer(estimator, test_X, test_y)
        self.assertLess(eo_diff, 0.0)
        ao_diff_scorer = lale.lib.aif360.average_odds_difference(**fi)
        ao_diff = ao_diff_scorer(estimator, test_X, test_y)
        self.assertLess(ao_diff, 0.1)
        theil_index_scorer = lale.lib.aif360.theil_index(**fi)
        theil_index = theil_index_scorer(estimator, test_X, test_y)
        self.assertGreater(theil_index, 0.1)
        symm_di_scorer = lale.lib.aif360.symmetric_disparate_impact(**fi)
        symm_di = symm_di_scorer(estimator, test_X, test_y)
        self.assertLess(symm_di, 0.9)

    def test_scorers_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        trainable = LogisticRegression(max_iter=1000)
        train_X = self.creditg_pd_num["splits"][0]["train_X"]
        train_y = self.creditg_pd_num["splits"][0]["train_y"]
        trained = trainable.fit(train_X, train_y)
        test_X = self.creditg_pd_num["splits"][0]["test_X"]
        test_y = self.creditg_pd_num["splits"][0]["test_y"]
        self._attempt_scorers(fairness_info, trained, test_X, test_y)

    def test_scorers_np_num(self):
        fairness_info = self.creditg_np_num["fairness_info"]
        trainable = LogisticRegression(max_iter=1000)
        train_X = self.creditg_np_num["train_X"]
        train_y = self.creditg_np_num["train_y"]
        trained = trainable.fit(train_X, train_y)
        test_X = self.creditg_np_num["test_X"]
        test_y = self.creditg_np_num["test_y"]
        self._attempt_scorers(fairness_info, trained, test_X, test_y)

    def test_scorers_regression_pd_num(self):
        fairness_info = self.boston_pd_num["fairness_info"]
        trainable = LinearRegression()
        train_X = self.boston_pd_num["train_X"]
        train_y = self.boston_pd_num["train_y"]
        trained = trainable.fit(train_X, train_y)
        test_X = self.boston_pd_num["test_X"]
        test_y = self.boston_pd_num["test_y"]
        self._attempt_scorers(fairness_info, trained, test_X, test_y)

    def test_scorers_regression_np_num(self):
        fairness_info = self.boston_np_num["fairness_info"]
        trainable = LinearRegression()
        train_X = self.boston_np_num["train_X"]
        train_y = self.boston_np_num["train_y"]
        trained = trainable.fit(train_X, train_y)
        test_X = self.boston_np_num["test_X"]
        test_y = self.boston_np_num["test_y"]
        self._attempt_scorers(fairness_info, trained, test_X, test_y)

    def test_scorers_combine_acc(self):
        dummy_fairness_info = {
            "favorable_labels": ["fav"],
            "protected_attributes": [{"feature": "prot", "reference_group": ["ref"]}],
        }
        scorer = lale.lib.aif360.accuracy_and_disparate_impact(**dummy_fairness_info)
        for acc in [0.2, 0.8, 1]:
            for di in [0.7, 0.9, 1.0]:
                score = scorer._combine(acc, di)
                self.assertLess(0.0, score)
                self.assertLessEqual(score, 1.0)
            for di in [0.0, float("inf"), float("-inf"), float("nan")]:
                score = scorer._combine(acc, di)
                self.assertEqual(score, 0.5 * acc)

    def test_scorers_combine_r2(self):
        dummy_fairness_info = {
            "favorable_labels": ["fav"],
            "protected_attributes": [{"feature": "prot", "reference_group": ["ref"]}],
        }
        scorer = lale.lib.aif360.r2_and_disparate_impact(**dummy_fairness_info)
        for r2 in [-2, 0, 0.5, 1]:
            for di in [0.7, 0.9, 1.0]:
                score = scorer._combine(r2, di)
                self.assertLess(0.0, score)
                self.assertLessEqual(score, 1.0)
            for di in [0.0, float("inf"), float("-inf"), float("nan")]:
                score = scorer._combine(r2, di)
                self.assertEqual(score, 0.5 / (2.0 - r2))

    def _attempt_remi_creditg_pd_num(
        self, fairness_info, trainable_remi, min_di, max_di
    ):
        splits = self.creditg_pd_num["splits"]
        disparate_impact_scorer = lale.lib.aif360.disparate_impact(**fairness_info)
        di_list = []
        for split in splits:
            if tensorflow_installed:  # for AdversarialDebiasing
                tf.compat.v1.reset_default_graph()
                tf.compat.v1.disable_eager_execution()
            train_X = split["train_X"]
            train_y = split["train_y"]
            trained_remi = trainable_remi.fit(train_X, train_y)
            test_X = split["test_X"]
            test_y = split["test_y"]
            di_list.append(disparate_impact_scorer(trained_remi, test_X, test_y))
        di = pd.Series(di_list)
        _, _, function_name, _ = traceback.extract_stack()[-2]
        print(f"disparate impact {di.mean():.3f} +- {di.std():.3f} {function_name}")
        if min_di > 0:
            self.assertLessEqual(min_di, di.mean())
            self.assertLessEqual(di.mean(), max_di)

    def test_disparate_impact_remover_np_num(self):
        fairness_info = self.creditg_np_num["fairness_info"]
        trainable_orig = LogisticRegression(max_iter=1000)
        trainable_remi = DisparateImpactRemover(**fairness_info) >> trainable_orig
        train_X = self.creditg_np_num["train_X"]
        train_y = self.creditg_np_num["train_y"]
        trained_orig = trainable_orig.fit(train_X, train_y)
        trained_remi = trainable_remi.fit(train_X, train_y)
        test_X = self.creditg_np_num["test_X"]
        test_y = self.creditg_np_num["test_y"]
        disparate_impact_scorer = lale.lib.aif360.disparate_impact(**fairness_info)
        impact_orig = disparate_impact_scorer(trained_orig, test_X, test_y)
        self.assertTrue(0.6 < impact_orig < 1.0, f"impact_orig {impact_orig}")
        impact_remi = disparate_impact_scorer(trained_remi, test_X, test_y)
        self.assertTrue(0.8 < impact_remi < 1.0, f"impact_remi {impact_remi}")

    def test_adversarial_debiasing_pd_num(self):
        if tensorflow_installed:
            fairness_info = self.creditg_pd_num["fairness_info"]
            tf.compat.v1.reset_default_graph()
            trainable_remi = AdversarialDebiasing(**fairness_info)
            self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.0, 1.5)

    def test_calibrated_eq_odds_postprocessing_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        estim = LogisticRegression(max_iter=1000)
        trainable_remi = CalibratedEqOddsPostprocessing(
            **fairness_info, estimator=estim
        )
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.65, 0.85)

    def test_disparate_impact_remover_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        trainable_remi = DisparateImpactRemover(**fairness_info) >> LogisticRegression(
            max_iter=1000
        )
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.78, 0.88)

    def test_eq_odds_postprocessing_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        estim = LogisticRegression(max_iter=1000)
        trainable_remi = EqOddsPostprocessing(**fairness_info, estimator=estim)
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.82, 1.02)

    def test_gerry_fair_classifier_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        trainable_remi = GerryFairClassifier(**fairness_info)
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.677, 0.678)

    def test_lfr_pd_num(self):
        if numba_installed:
            fairness_info = self.creditg_pd_num["fairness_info"]
            trainable_remi = LFR(**fairness_info) >> LogisticRegression(max_iter=1000)
            self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.95, 1.05)

    def test_meta_fair_classifier_pd_num(self):
        if aif360.__version__ >= "4.0.0":
            fairness_info = self.creditg_pd_num["fairness_info"]
            trainable_remi = MetaFairClassifier(**fairness_info)
            self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.62, 0.87)

    def test_prejudice_remover_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        trainable_remi = PrejudiceRemover(**fairness_info)
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.73, 0.83)

    def test_redacting_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        redacting = Redacting(**fairness_info)
        logistic_regression = LogisticRegression(max_iter=1000)
        trainable_remi = redacting >> logistic_regression
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.78, 0.94)

    def test_reject_option_classification_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        estim = LogisticRegression(max_iter=1000)
        trainable_remi = RejectOptionClassification(**fairness_info, estimator=estim)
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.88, 0.98)

    def test_reweighing_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        estim = LogisticRegression(max_iter=1000)
        trainable_remi = Reweighing(estimator=estim, **fairness_info)
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.82, 0.92)

    def test_sans_mitigation_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        trainable_remi = LogisticRegression(max_iter=1000)
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.5, 1.0)


class TestAIF360Cat(unittest.TestCase):
    @classmethod
    def _prep_pd_cat(cls):
        result = (
            (
                Project(columns={"type": "string"})
                >> OneHotEncoder(handle_unknown="ignore")
            )
            & Project(columns={"type": "number"})
        ) >> ConcatFeatures
        return result

    @classmethod
    def _creditg_pd_cat(cls):
        X, y, fairness_info = lale.lib.aif360.fetch_creditg_df(preprocess=False)
        cv = lale.lib.aif360.FairStratifiedKFold(**fairness_info, n_splits=3)
        splits = []
        lr = LogisticRegression()
        for train, test in cv.split(X, y):
            train_X, train_y = lale.helpers.split_with_schemas(lr, X, y, train)
            assert isinstance(train_X, pd.DataFrame), type(train_X)
            assert isinstance(train_y, pd.Series), type(train_y)
            test_X, test_y = lale.helpers.split_with_schemas(lr, X, y, test, train)
            assert isinstance(test_X, pd.DataFrame), type(test_X)
            assert isinstance(test_y, pd.Series), type(test_y)
            splits.append(
                {
                    "train_X": train_X,
                    "train_y": train_y,
                    "test_X": test_X,
                    "test_y": test_y,
                }
            )
        result = {"splits": splits, "fairness_info": fairness_info}
        return result

    @classmethod
    def _creditg_np_cat(cls):
        train_X = cls.creditg_pd_cat["splits"][0]["train_X"].to_numpy()
        train_y = cls.creditg_pd_cat["splits"][0]["train_y"].to_numpy()
        test_X = cls.creditg_pd_cat["splits"][0]["test_X"].to_numpy()
        test_y = cls.creditg_pd_cat["splits"][0]["test_y"].to_numpy()
        assert isinstance(train_X, np.ndarray), type(train_X)
        assert not isinstance(train_X, NDArrayWithSchema), type(train_X)
        assert isinstance(train_y, np.ndarray), type(train_y)
        assert not isinstance(train_y, NDArrayWithSchema), type(train_y)
        pd_columns = cls.creditg_pd_cat["splits"][0]["train_X"].columns
        pd_fav_labels = cls.creditg_pd_cat["fairness_info"]["favorable_labels"]
        pd_prot_attrs = cls.creditg_pd_cat["fairness_info"]["protected_attributes"]
        fairness_info = {
            "favorable_labels": pd_fav_labels,
            "protected_attributes": [
                {
                    "feature": pd_columns.get_loc("personal_status"),
                    "reference_group": pd_prot_attrs[0]["reference_group"],
                },
                {
                    "feature": pd_columns.get_loc("age"),
                    "reference_group": pd_prot_attrs[1]["reference_group"],
                },
            ],
        }
        result = {
            "train_X": train_X,
            "train_y": train_y,
            "test_X": test_X,
            "test_y": test_y,
            "fairness_info": fairness_info,
        }
        return result

    @classmethod
    def _creditg_pd_ternary(cls):
        X, y, _ = lale.lib.aif360.fetch_creditg_df(preprocess=False)
        fairness_info = {
            "favorable_labels": ["good"],
            "protected_attributes": [
                {
                    "feature": "personal_status",
                    "reference_group": ["male div/sep", "male mar/wid", "male single"],
                    "monitored_group": ["female div/dep/mar"],
                },
                {
                    "feature": "age",
                    "reference_group": [[26, 1000]],
                    "monitored_group": [[1, 23]],
                },
            ],
            "unfavorable_labels": ["bad"],
        }
        cv = lale.lib.aif360.FairStratifiedKFold(**fairness_info, n_splits=3)
        splits = []
        lr = LogisticRegression()
        for train, test in cv.split(X, y):
            train_X, train_y = lale.helpers.split_with_schemas(lr, X, y, train)
            assert isinstance(train_X, pd.DataFrame), type(train_X)
            assert isinstance(train_y, pd.Series), type(train_y)
            test_X, test_y = lale.helpers.split_with_schemas(lr, X, y, test, train)
            assert isinstance(test_X, pd.DataFrame), type(test_X)
            assert isinstance(test_y, pd.Series), type(test_y)
            splits.append(
                {
                    "train_X": train_X,
                    "train_y": train_y,
                    "test_X": test_X,
                    "test_y": test_y,
                }
            )
        result = {"splits": splits, "fairness_info": fairness_info}
        return result

    @classmethod
    def setUpClass(cls):
        cls.prep_pd_cat = cls._prep_pd_cat()
        cls.creditg_pd_cat = cls._creditg_pd_cat()
        cls.creditg_np_cat = cls._creditg_np_cat()
        cls.creditg_pd_ternary = cls._creditg_pd_ternary()

    def test_encoder_pd_cat(self):
        info = self.creditg_pd_cat["fairness_info"]
        orig_X = self.creditg_pd_cat["splits"][0]["train_X"]
        encoder_separate = lale.lib.aif360.ProtectedAttributesEncoder(
            protected_attributes=info["protected_attributes"]
        )
        csep_X = encoder_separate.transform(orig_X)
        encoder_and = lale.lib.aif360.ProtectedAttributesEncoder(
            protected_attributes=info["protected_attributes"], combine="and"
        )
        cand_X = encoder_and.transform(orig_X)
        for i in orig_X.index:
            orig_row = orig_X.loc[i]
            csep_row = csep_X.loc[i]
            cand_row = cand_X.loc[i]
            cand_name = list(cand_X.columns)[0]
            self.assertEqual(
                1 if orig_row["personal_status"].startswith("male") else 0,
                csep_row["personal_status"],
            )
            self.assertEqual(1 if orig_row["age"] >= 26 else 0, csep_row["age"])
            self.assertEqual(
                cand_row[cand_name],
                1 if csep_row["personal_status"] == 1 == csep_row["age"] else 0,
            )

    def test_encoder_np_cat(self):
        info = self.creditg_np_cat["fairness_info"]
        orig_X = self.creditg_np_cat["train_X"]
        encoder = lale.lib.aif360.ProtectedAttributesEncoder(
            protected_attributes=info["protected_attributes"]
        )
        conv_X = encoder.transform(orig_X)
        for i in range(orig_X.shape[0]):
            self.assertEqual(
                1 if orig_X[i, 8].startswith("male") else 0,
                conv_X.at[i, "f8"],
            )
            self.assertEqual(1 if orig_X[i, 12] >= 26 else 0, conv_X.at[i, "f12"])

    def test_encoder_pd_ternary(self):
        info = self.creditg_pd_ternary["fairness_info"]
        orig_X = self.creditg_pd_ternary["splits"][0]["train_X"]
        encoder_separate = lale.lib.aif360.ProtectedAttributesEncoder(
            protected_attributes=info["protected_attributes"]
        )
        csep_X = encoder_separate.transform(orig_X)
        encoder_and = lale.lib.aif360.ProtectedAttributesEncoder(
            protected_attributes=info["protected_attributes"], combine="and"
        )
        cand_X = encoder_and.transform(orig_X)
        for i in orig_X.index:
            orig_row = orig_X.loc[i]
            csep_row = csep_X.loc[i]
            cand_row = cand_X.loc[i]
            cand_name = list(cand_X.columns)[0]
            self.assertEqual(
                1
                if orig_row["personal_status"].startswith("male")
                else 0
                if orig_row["personal_status"].startswith("female")
                else 0.5,
                csep_row["personal_status"],
                f"personal_status {orig_row['personal_status']}",
            )
            self.assertEqual(
                1
                if 26 <= orig_row["age"] <= 1000
                else 0
                if 1 <= orig_row["age"] <= 23
                else 0.5,
                csep_row["age"],
                f"age {orig_row['age']}",
            )
            self.assertEqual(
                cand_row[cand_name],
                min(csep_row["personal_status"], csep_row["age"]),
                f"age {orig_row['age']}, personal_status {orig_row['personal_status']}",
            )

    def test_column_for_stratification(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        train_X = self.creditg_pd_cat["splits"][0]["train_X"]
        train_y = self.creditg_pd_cat["splits"][0]["train_y"]
        stratify = lale.lib.aif360.util._column_for_stratification(
            train_X, train_y, **fairness_info, unfavorable_labels=None
        )
        for i in train_X.index:
            male = train_X.loc[i]["personal_status"].startswith("male")
            old = train_X.loc[i]["age"] >= 26
            favorable = train_y.loc[i] == "good"
            strat = stratify.loc[i]
            self.assertEqual(male, strat[0] == "T")
            self.assertEqual(old, strat[1] == "T")
            self.assertEqual(favorable, strat[2] == "T")

    def test_fair_stratified_train_test_split(self):
        X, y, fairness_info = lale.lib.aif360.fetch_creditg_df(preprocess=False)
        z = range(X.shape[0])
        (
            train_X,
            test_X,
            train_y,
            test_y,
            train_z,
            test_z,
        ) = fair_stratified_train_test_split(X, y, z, **fairness_info)
        self.assertEqual(train_X.shape[0], train_y.shape[0])
        self.assertEqual(train_X.shape[0], len(train_z))
        self.assertEqual(test_X.shape[0], test_y.shape[0])
        self.assertEqual(test_X.shape[0], len(test_z))

    def _attempt_scorers(self, fairness_info, estimator, test_X, test_y):
        fi = fairness_info
        disparate_impact_scorer = lale.lib.aif360.disparate_impact(**fi)
        impact = disparate_impact_scorer(estimator, test_X, test_y)
        self.assertLess(impact, 0.9)
        if estimator.is_classifier():
            combined_scorer = lale.lib.aif360.accuracy_and_disparate_impact(**fi)
            combined = combined_scorer(estimator, test_X, test_y)
            self.assertLess(0.0, combined)
            self.assertLess(combined, 1.0)
        else:
            combined_scorer = lale.lib.aif360.r2_and_disparate_impact(**fi)
            combined = combined_scorer(estimator, test_X, test_y)
            self.assertLess(0.0, combined)
            self.assertLess(combined, 1.0)
        parity_scorer = lale.lib.aif360.statistical_parity_difference(**fi)
        parity = parity_scorer(estimator, test_X, test_y)
        self.assertLess(parity, 0.0)
        eo_diff_scorer = lale.lib.aif360.equal_opportunity_difference(**fi)
        eo_diff = eo_diff_scorer(estimator, test_X, test_y)
        self.assertLess(eo_diff, 0.0)
        ao_diff_scorer = lale.lib.aif360.average_odds_difference(**fi)
        ao_diff = ao_diff_scorer(estimator, test_X, test_y)
        self.assertLess(ao_diff, 0.1)
        theil_index_scorer = lale.lib.aif360.theil_index(**fi)
        theil_index = theil_index_scorer(estimator, test_X, test_y)
        self.assertGreater(theil_index, 0.1)
        symm_di_scorer = lale.lib.aif360.symmetric_disparate_impact(**fi)
        symm_di = symm_di_scorer(estimator, test_X, test_y)
        self.assertLess(symm_di, 0.9)

    def test_scorers_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        trainable = self.prep_pd_cat >> LogisticRegression(max_iter=1000)
        train_X = self.creditg_pd_cat["splits"][0]["train_X"]
        train_y = self.creditg_pd_cat["splits"][0]["train_y"]
        trained = trainable.fit(train_X, train_y)
        test_X = self.creditg_pd_cat["splits"][0]["test_X"]
        test_y = self.creditg_pd_cat["splits"][0]["test_y"]
        self._attempt_scorers(fairness_info, trained, test_X, test_y)

    def test_scorers_np_cat(self):
        fairness_info = self.creditg_np_cat["fairness_info"]
        train_X = self.creditg_np_cat["train_X"]
        train_y = self.creditg_np_cat["train_y"]
        cat_columns, num_columns = [], []
        for i in range(train_X.shape[1]):
            try:
                _ = train_X[:, i].astype(np.float64)
                num_columns.append(i)
            except ValueError:
                cat_columns.append(i)
        trainable = (
            (
                (Project(columns=cat_columns) >> OneHotEncoder(handle_unknown="ignore"))
                & (
                    Project(columns=num_columns)
                    >> FunctionTransformer(func=lambda x: x.astype(np.float64))
                )
            )
            >> ConcatFeatures
            >> LogisticRegression(max_iter=1000)
        )
        trained = trainable.fit(train_X, train_y)
        test_X = self.creditg_np_cat["test_X"]
        test_y = self.creditg_np_cat["test_y"]
        self._attempt_scorers(fairness_info, trained, test_X, test_y)

    def test_scorers_pd_ternary(self):
        fairness_info = self.creditg_pd_ternary["fairness_info"]
        trainable = self.prep_pd_cat >> LogisticRegression(max_iter=1000)
        train_X = self.creditg_pd_ternary["splits"][0]["train_X"]
        train_y = self.creditg_pd_ternary["splits"][0]["train_y"]
        trained = trainable.fit(train_X, train_y)
        test_X = self.creditg_pd_ternary["splits"][0]["test_X"]
        test_y = self.creditg_pd_ternary["splits"][0]["test_y"]
        self._attempt_scorers(fairness_info, trained, test_X, test_y)

    def test_scorers_warn(self):
        fairness_info = {
            "favorable_labels": ["good"],
            "protected_attributes": [{"feature": "age", "reference_group": [1]}],
        }
        trainable = self.prep_pd_cat >> LogisticRegression(max_iter=1000)
        train_X = self.creditg_pd_cat["splits"][0]["train_X"]
        train_y = self.creditg_pd_cat["splits"][0]["train_y"]
        trained = trainable.fit(train_X, train_y)
        test_X = self.creditg_pd_cat["splits"][0]["test_X"]
        test_y = self.creditg_pd_cat["splits"][0]["test_y"]
        disparate_impact_scorer = lale.lib.aif360.disparate_impact(**fairness_info)
        with self.assertLogs(lale.lib.aif360.util.logger) as log_context_manager:
            impact = disparate_impact_scorer(trained, test_X, test_y)
        self.assertRegex(log_context_manager.output[-1], "is ill-defined")
        self.assertTrue(np.isnan(impact))

    def test_scorers_warn2(self):
        fairness_info = {
            "favorable_labels": ["good"],
            "unfavorable_labels": ["good"],
            "protected_attributes": [
                {"feature": "age", "reference_group": [[26, 1000]]}
            ],
        }
        with self.assertLogs(lale.lib.aif360.util.logger) as log_context_manager:
            _ = lale.lib.aif360.disparate_impact(**fairness_info)
        self.assertRegex(
            log_context_manager.output[-1],
            "overlap between favorable labels and unfavorable labels on 'good' and 'good'",
        )

    def test_scorers_warn3(self):
        fairness_info = {
            "favorable_labels": ["good"],
            "protected_attributes": [
                {"feature": "age", "reference_group": [[1, 2, 3]]}
            ],
        }
        with self.assertRaises(jsonschema.ValidationError):
            _ = lale.lib.aif360.disparate_impact(**fairness_info)

    def test_scorers_warn4(self):
        fairness_info = {
            "favorable_labels": ["good"],
            "protected_attributes": [
                {
                    "feature": "age",
                    "reference_group": [[20, 40]],
                    "monitored_group": [30],
                }
            ],
        }
        with self.assertLogs(lale.lib.aif360.util.logger) as log_context_manager:
            _ = lale.lib.aif360.disparate_impact(**fairness_info)
        self.assertRegex(
            log_context_manager.output[-1],
            "overlap between reference group and monitored group of feature 'age'",
        )

    def test_scorers_ternary_nonexhaustive(self):
        X, y, fairness_info = lale.lib.aif360.fetch_nursery_df(preprocess=False)
        self.assertEqual(
            set(y), {"not_recom", "recommend", "very_recom", "priority", "spec_prior"}
        )
        fairness_info = {**fairness_info, "unfavorable_labels": ["not_recom"]}
        self.assertEqual(
            set(
                fairness_info["favorable_labels"] + fairness_info["unfavorable_labels"]
            ),
            {"spec_prior", "not_recom"},
        )
        di_scorer = lale.lib.aif360.disparate_impact(**fairness_info)
        di_measured = di_scorer.score_data(X=X, y_pred=y)
        self.assertAlmostEqual(di_measured, 0.461, places=3)
        sp_scorer = lale.lib.aif360.statistical_parity_difference(**fairness_info)
        sp_measured = sp_scorer.score_data(X=X, y_pred=y)
        self.assertAlmostEqual(sp_measured, -0.205, places=3)
        sdi_scorer = lale.lib.aif360.symmetric_disparate_impact(**fairness_info)
        sdi_measured = sdi_scorer.score_data(X=X, y_pred=y)
        self.assertAlmostEqual(sdi_measured, 0.461, places=3)
        adi_scorer = lale.lib.aif360.accuracy_and_disparate_impact(**fairness_info)
        adi_measured = adi_scorer.score_data(X=X, y_pred=y, y_true=y)
        self.assertAlmostEqual(adi_measured, 0.731, places=3)
        ao_scorer = lale.lib.aif360.average_odds_difference(**fairness_info)
        with self.assertRaisesRegex(ValueError, "unexpected labels"):
            _ = ao_scorer.score_data(X=X, y_pred=y)

    def _attempt_remi_creditg_pd_cat(
        self, fairness_info, trainable_remi, min_di, max_di
    ):
        splits = self.creditg_pd_cat["splits"]
        disparate_impact_scorer = lale.lib.aif360.disparate_impact(**fairness_info)
        di_list = []
        for split in splits:
            if tensorflow_installed:  # for AdversarialDebiasing
                tf.compat.v1.reset_default_graph()
            train_X = split["train_X"]
            train_y = split["train_y"]
            trained_remi = trainable_remi.fit(train_X, train_y)
            test_X = split["test_X"]
            test_y = split["test_y"]
            di_list.append(disparate_impact_scorer(trained_remi, test_X, test_y))
        di = pd.Series(di_list)
        _, _, function_name, _ = traceback.extract_stack()[-2]
        print(f"disparate impact {di.mean():.3f} +- {di.std():.3f} {function_name}")
        self.assertTrue(
            min_di <= di.mean() <= max_di,
            f"{min_di} <= {di.mean()} <= {max_di}",
        )

    def test_adversarial_debiasing_pd_cat(self):
        if tensorflow_installed:
            fairness_info = self.creditg_pd_cat["fairness_info"]
            trainable_remi = AdversarialDebiasing(
                **fairness_info, preparation=self.prep_pd_cat
            )
            self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.0, 1.5)

    def test_calibrated_eq_odds_postprocessing_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        estim = self.prep_pd_cat >> LogisticRegression(max_iter=1000)
        trainable_remi = CalibratedEqOddsPostprocessing(
            **fairness_info, estimator=estim
        )
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.65, 0.85)

    def test_disparate_impact_remover_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        trainable_remi = DisparateImpactRemover(
            **fairness_info, preparation=self.prep_pd_cat
        ) >> LogisticRegression(max_iter=1000)
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.72, 0.92)

    def test_disparate_impact_remover_pd_cat_no_redact(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        trainable_remi = DisparateImpactRemover(
            **fairness_info, redact=False, preparation=self.prep_pd_cat
        ) >> LogisticRegression(max_iter=1000)
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.65, 0.75)

    def test_eq_odds_postprocessing_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        estim = self.prep_pd_cat >> LogisticRegression(max_iter=1000)
        trainable_remi = EqOddsPostprocessing(**fairness_info, estimator=estim)
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.82, 1.02)

    def test_gerry_fair_classifier_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        trainable_remi = GerryFairClassifier(
            **fairness_info, preparation=self.prep_pd_cat
        )
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.677, 0.678)

    def test_lfr_pd_cat(self):
        if numba_installed:
            fairness_info = self.creditg_pd_cat["fairness_info"]
            trainable_remi = LFR(
                **fairness_info, preparation=self.prep_pd_cat
            ) >> LogisticRegression(max_iter=1000)
            self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.95, 1.05)

    def test_meta_fair_classifier_pd_cat(self):
        if aif360.__version__ >= "4.0.0":
            fairness_info = self.creditg_pd_cat["fairness_info"]
            trainable_remi = MetaFairClassifier(
                **fairness_info, preparation=self.prep_pd_cat
            )
            self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.62, 0.87)

    def test_optim_preproc_pd_cat(self):
        # TODO: set the optimizer options as shown in the example https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_optim_data_preproc.ipynb
        if cvxpy_installed:
            fairness_info = self.creditg_pd_cat["fairness_info"]
            _ = OptimPreproc(**fairness_info, optim_options={}) >> LogisticRegression(
                max_iter=1000
            )
            # TODO: this test does not yet call fit or predict

    def test_prejudice_remover_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        trainable_remi = PrejudiceRemover(**fairness_info, preparation=self.prep_pd_cat)
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.70, 0.80)

    def test_redacting_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        estim = self.prep_pd_cat >> LogisticRegression(max_iter=1000)
        trainable_remi = Redacting(**fairness_info) >> estim
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.78, 0.94)

    def test_reject_option_classification_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        estim = self.prep_pd_cat >> LogisticRegression(max_iter=1000)
        trainable_remi = RejectOptionClassification(**fairness_info, estimator=estim)
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.88, 0.98)

    def test_sans_mitigation_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        trainable_remi = self.prep_pd_cat >> LogisticRegression(max_iter=1000)
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.66, 0.76)

    def test_reweighing_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        estim = self.prep_pd_cat >> LogisticRegression(max_iter=1000)
        trainable_remi = Reweighing(estimator=estim, **fairness_info)
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.85, 1.00)

    def test_pd_cat_y_not_series(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        trainable_remi = DisparateImpactRemover(
            **fairness_info, preparation=self.prep_pd_cat
        ) >> LogisticRegression(max_iter=1000)
        train_X = self.creditg_pd_cat["splits"][0]["train_X"]
        train_y = self.creditg_pd_cat["splits"][0]["train_y"].to_frame()
        trained_remi = trainable_remi.fit(train_X, train_y)
        test_X = self.creditg_pd_cat["splits"][0]["test_X"]
        test_y = self.creditg_pd_cat["splits"][0]["test_y"].to_frame()
        di_scorer = lale.lib.aif360.disparate_impact(**fairness_info)
        di = di_scorer(trained_remi, test_X, test_y)
        self.assertLessEqual(0.8, di)
        self.assertLessEqual(di, 1.0)
