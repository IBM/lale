# Copyright 2020 IBM Corporation
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

import traceback
import unittest

import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import tensorflow as tf

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
    def _attempt_dataset(
        self, X, y, fairness_info, n_rows, n_columns, set_y, di_expected
    ):
        self.assertEqual(X.shape, (n_rows, n_columns))
        self.assertEqual(y.shape, (n_rows,))
        self.assertEqual(set(y), set_y)
        di_scorer = lale.lib.aif360.disparate_impact(**fairness_info)
        di_measured = di_scorer.scoring(X=X, y_pred=y)
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
        self._attempt_dataset(X, y, fairness_info, 5_278, 13, {0, 1}, 0.919)

    def test_dataset_compas_pd_num(self):
        X, y, fairness_info = lale.lib.aif360.fetch_compas_df(preprocess=True)
        self._attempt_dataset(X, y, fairness_info, 5_278, 12, {0, 1}, 0.919)

    def test_dataset_creditg_pd_cat(self):
        X, y, fairness_info = lale.lib.aif360.fetch_creditg_df(preprocess=False)
        self._attempt_dataset(X, y, fairness_info, 1_000, 20, {"bad", "good"}, 0.748)

    def test_dataset_creditg_pd_num(self):
        X, y, fairness_info = lale.lib.aif360.fetch_creditg_df(preprocess=True)
        self._attempt_dataset(X, y, fairness_info, 1_000, 58, {0, 1}, 0.748)

    def test_dataset_ricci_pd_cat(self):
        X, y, fairness_info = lale.lib.aif360.fetch_ricci_df(preprocess=False)
        self._attempt_dataset(
            X, y, fairness_info, 118, 5, {"No promotion", "Promotion"}, 0.498
        )

    def test_dataset_ricci_pd_num(self):
        X, y, fairness_info = lale.lib.aif360.fetch_ricci_df(preprocess=True)
        self._attempt_dataset(X, y, fairness_info, 118, 6, {0, 1}, 0.498)


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
    def _boston(cls):
        orig_X, orig_y = sklearn.datasets.load_boston(return_X_y=True)
        train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(
            orig_X, orig_y, test_size=0.33, random_state=42
        )
        assert isinstance(train_X, np.ndarray), type(train_X)
        assert not isinstance(train_X, NDArrayWithSchema), type(train_X)
        assert isinstance(train_y, np.ndarray), type(train_y)
        assert not isinstance(train_y, NDArrayWithSchema), type(train_y)
        black_median = np.median(train_X[:, 11])
        label_median = np.median(train_y)
        fairness_info = {
            "favorable_labels": [[-10000.0, label_median]],
            "protected_attributes": [
                # 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
                {"feature": 11, "reference_group": [[0.0, black_median]]},
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
        cls.boston = cls._boston()

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
            accuracy_scorer = sklearn.metrics.make_scorer(
                sklearn.metrics.accuracy_score
            )
            accuracy = accuracy_scorer(estimator, test_X, test_y)
            self.assertLess(combined, accuracy)
        else:
            combined_scorer = lale.lib.aif360.r2_and_disparate_impact(**fi)
            combined = combined_scorer(estimator, test_X, test_y)
            r2_scorer = sklearn.metrics.make_scorer(sklearn.metrics.r2_score)
            r2 = r2_scorer(estimator, test_X, test_y)
            self.assertLess(combined, r2)
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

    def test_scorers_regression(self):
        fairness_info = self.boston["fairness_info"]
        trainable = LinearRegression()
        train_X = self.boston["train_X"]
        train_y = self.boston["train_y"]
        trained = trainable.fit(train_X, train_y)
        test_X = self.boston["test_X"]
        test_y = self.boston["test_y"]
        self._attempt_scorers(fairness_info, trained, test_X, test_y)

    def test_scorers_combine(self):
        fairness_info = self.boston["fairness_info"]
        combined_scorer = lale.lib.aif360.r2_and_disparate_impact(**fairness_info)
        for r2 in [-2, 0, 0.5, 1]:
            for disp_impact in [0.7, 0.9, 1.0, (1 / 0.9), (1 / 0.7)]:
                score = combined_scorer._combine(r2, disp_impact)
                print(f"r2 {r2:5.2f}, di {disp_impact:.2f}, score {score:5.2f}")

    def _attempt_remi_creditg_pd_num(
        self, fairness_info, trainable_remi, min_di, max_di
    ):
        splits = self.creditg_pd_num["splits"]
        disparate_impact_scorer = lale.lib.aif360.disparate_impact(**fairness_info)
        di_list = []
        for split in splits:
            tf.reset_default_graph()  # for AdversarialDebiasing
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
        fairness_info = self.creditg_pd_num["fairness_info"]
        tf.reset_default_graph()
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
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.87, 0.97)

    def test_gerry_fair_classifier_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        trainable_remi = GerryFairClassifier(**fairness_info)
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.677, 0.678)

    def test_lfr_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        trainable_remi = LFR(**fairness_info) >> LogisticRegression(max_iter=1000)
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.95, 1.05)

    def test_meta_fair_classifier_pd_num(self):
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
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.80, 0.90)

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
    def setUpClass(cls):
        cls.prep_pd_cat = cls._prep_pd_cat()
        cls.creditg_pd_cat = cls._creditg_pd_cat()
        cls.creditg_np_cat = cls._creditg_np_cat()

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
                orig_row["personal_status"].startswith("male"),
                csep_row["personal_status"],
            )
            self.assertEqual(orig_row["age"] >= 26, csep_row["age"])
            self.assertEqual(
                cand_row[cand_name], csep_row["personal_status"] and csep_row["age"]
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
                orig_X[i, 8].startswith("male"),
                conv_X.at[i, "f8"],
            )
            self.assertEqual(orig_X[i, 12] >= 26, conv_X.at[i, "f12"])

    def test_column_for_stratification(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        train_X = self.creditg_pd_cat["splits"][0]["train_X"]
        train_y = self.creditg_pd_cat["splits"][0]["train_y"]
        stratify = lale.lib.aif360.util._column_for_stratification(
            train_X, train_y, **fairness_info
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
            accuracy_scorer = sklearn.metrics.make_scorer(
                sklearn.metrics.accuracy_score
            )
            accuracy = accuracy_scorer(estimator, test_X, test_y)
            self.assertLess(combined, accuracy)
        else:
            combined_scorer = lale.lib.aif360.r2_and_disparate_impact(**fi)
            combined = combined_scorer(estimator, test_X, test_y)
            r2_scorer = sklearn.metrics.make_scorer(sklearn.metrics.r2_score)
            r2 = r2_scorer(estimator, test_X, test_y)
            self.assertLess(combined, r2)
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
        self.assertEqual(impact, 0.0)

    def _attempt_remi_creditg_pd_cat(
        self, fairness_info, trainable_remi, min_di, max_di
    ):
        splits = self.creditg_pd_cat["splits"]
        disparate_impact_scorer = lale.lib.aif360.disparate_impact(**fairness_info)
        di_list = []
        for split in splits:
            tf.reset_default_graph()  # for AdversarialDebiasing
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
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.88, 0.98)

    def test_gerry_fair_classifier_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        trainable_remi = GerryFairClassifier(
            **fairness_info, preparation=self.prep_pd_cat
        )
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.677, 0.678)

    def test_lfr_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        trainable_remi = LFR(
            **fairness_info, preparation=self.prep_pd_cat
        ) >> LogisticRegression(max_iter=1000)
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.95, 1.05)

    def test_meta_fair_classifier_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        trainable_remi = MetaFairClassifier(
            **fairness_info, preparation=self.prep_pd_cat
        )
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.62, 0.87)

    def test_optim_preproc_pd_cat(self):
        # TODO: set the optimizer options as shown in the example https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_optim_data_preproc.ipynb
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
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.81, 0.91)

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
