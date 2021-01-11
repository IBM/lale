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

import unittest

import numpy as np
import pandas as pd
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection
import tensorflow as tf

import lale.datasets.data_schemas
import lale.datasets.openml
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


class TestAIF360(unittest.TestCase):
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
        (train_X, train_y), (test_X, test_y) = lale.datasets.openml.fetch(
            "credit-g", "classification", astype="pandas", preprocess=False
        )
        assert isinstance(train_X, pd.DataFrame), type(train_X)
        assert isinstance(train_y, pd.Series), type(train_y)
        fairness_info = {
            "favorable_labels": ["good"],
            "protected_attributes": [
                {
                    "feature": "personal_status",
                    "privileged_groups": [
                        "male div/sep",
                        "male mar/wid",
                        "male single",
                    ],
                },
                {"feature": "age", "privileged_groups": [[26, 1000]]},
            ],
        }
        all_X = pd.concat([train_X, test_X])
        all_y = pd.concat([train_y, test_y])
        train_X, test_X, train_y, test_y = fair_stratified_train_test_split(
            all_X, all_y, **fairness_info, test_size=0.33
        )
        result = {
            "train_X": train_X,
            "train_y": train_y,
            "test_X": test_X,
            "test_y": test_y,
            "fairness_info": fairness_info,
        }
        return result

    @classmethod
    def _creditg_pd_num(cls):
        (train_X, train_y), (test_X, test_y) = lale.lib.aif360.fetch_creditg_df()
        assert isinstance(train_X, pd.DataFrame), type(train_X)
        assert isinstance(train_y, pd.Series), type(train_y)
        fairness_info = {
            "favorable_labels": [1],
            "protected_attributes": [
                {"feature": "age", "privileged_groups": [1]},
                {"feature": "sex", "privileged_groups": [1]},
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
    def _creditg_np_cat(cls):
        train_X = cls.creditg_pd_cat["train_X"].to_numpy()
        train_y = cls.creditg_pd_cat["train_y"].to_numpy()
        test_X = cls.creditg_pd_cat["test_X"].to_numpy()
        test_y = cls.creditg_pd_cat["test_y"].to_numpy()
        assert isinstance(train_X, np.ndarray), type(train_X)
        assert not isinstance(train_X, NDArrayWithSchema), type(train_X)
        assert isinstance(train_y, np.ndarray), type(train_y)
        assert not isinstance(train_y, NDArrayWithSchema), type(train_y)
        fairness_info = {
            "favorable_labels": ["good"],
            "protected_attributes": [
                {
                    "feature": 8,
                    "privileged_groups": [
                        "male div/sep",
                        "male mar/wid",
                        "male single",
                    ],
                },
                {"feature": 12, "privileged_groups": [[26, 1000]]},
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
    def _creditg_np_num(cls):
        train_X = cls.creditg_pd_num["train_X"].to_numpy()
        train_y = cls.creditg_pd_num["train_y"].to_numpy()
        test_X = cls.creditg_pd_num["test_X"].to_numpy()
        test_y = cls.creditg_pd_num["test_y"].to_numpy()
        assert isinstance(train_X, np.ndarray), type(train_X)
        assert not isinstance(train_X, NDArrayWithSchema), type(train_X)
        assert isinstance(train_y, np.ndarray), type(train_y)
        assert not isinstance(train_y, NDArrayWithSchema), type(train_y)
        fairness_info = {
            "favorable_labels": [1.0],
            "protected_attributes": [
                {"feature": 57, "privileged_groups": [1.0]},
                {"feature": 55, "privileged_groups": [2.0]},
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
                {"feature": 11, "privileged_groups": [[0.0, black_median]]},
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
        cls.creditg_pd_num = cls._creditg_pd_num()
        cls.creditg_np_cat = cls._creditg_np_cat()
        cls.creditg_np_num = cls._creditg_np_num()
        cls.boston = cls._boston()

    def test_encoder_pd_cat(self):
        info = self.creditg_pd_cat["fairness_info"]
        orig_X = self.creditg_pd_cat["train_X"]
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
                orig_X[i, 8].startswith("male"), conv_X.at[i, "f8"],
            )
            self.assertEqual(orig_X[i, 12] >= 26, conv_X.at[i, "f12"])

    def test_column_for_stratification(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        train_X = self.creditg_pd_cat["train_X"]
        train_y = self.creditg_pd_cat["train_y"]
        stratify = lale.lib.aif360.util.column_for_stratification(
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
        train_X = self.creditg_pd_num["train_X"]
        train_y = self.creditg_pd_num["train_y"]
        trained = trainable.fit(train_X, train_y)
        test_X = self.creditg_pd_num["test_X"]
        test_y = self.creditg_pd_num["test_y"]
        self._attempt_scorers(fairness_info, trained, test_X, test_y)

    def test_scorers_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        trainable = self.prep_pd_cat >> LogisticRegression(max_iter=1000)
        train_X = self.creditg_pd_cat["train_X"]
        train_y = self.creditg_pd_cat["train_y"]
        trained = trainable.fit(train_X, train_y)
        test_X = self.creditg_pd_cat["test_X"]
        test_y = self.creditg_pd_cat["test_y"]
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

    def test_scorers_regression(self):
        fairness_info = self.boston["fairness_info"]
        trainable = LinearRegression()
        train_X = self.boston["train_X"]
        train_y = self.boston["train_y"]
        trained = trainable.fit(train_X, train_y)
        test_X = self.boston["test_X"]
        test_y = self.boston["test_y"]
        self._attempt_scorers(fairness_info, trained, test_X, test_y)

    def test_scorers_warn(self):
        fairness_info = {
            "favorable_labels": ["good"],
            "protected_attributes": [{"feature": "age", "privileged_groups": [1]}],
        }
        trainable = self.prep_pd_cat >> LogisticRegression(max_iter=1000)
        train_X = self.creditg_pd_cat["train_X"]
        train_y = self.creditg_pd_cat["train_y"]
        trained = trainable.fit(train_X, train_y)
        test_X = self.creditg_pd_cat["test_X"]
        test_y = self.creditg_pd_cat["test_y"]
        disparate_impact_scorer = lale.lib.aif360.disparate_impact(**fairness_info)
        with self.assertLogs(lale.lib.aif360.util.logger) as log_context_manager:
            impact = disparate_impact_scorer(trained, test_X, test_y)
        self.assertRegex(log_context_manager.output[-1], "is ill-defined")
        self.assertEqual(impact, 0.0)

    def _attempt_remi_creditg_pd_num(
        self, fairness_info, trainable_remi, min_di, max_di
    ):
        train_X = self.creditg_pd_num["train_X"]
        train_y = self.creditg_pd_num["train_y"]
        trained_remi = trainable_remi.fit(train_X, train_y)
        test_X = self.creditg_pd_num["test_X"]
        test_y = self.creditg_pd_num["test_y"]
        disparate_impact_scorer = lale.lib.aif360.disparate_impact(**fairness_info)
        impact_remi = disparate_impact_scorer(trained_remi, test_X, test_y)
        self.assertTrue(
            min_di <= impact_remi <= max_di, f"{min_di} <= {impact_remi} <= {max_di}",
        )
        return impact_remi

    def test_disparate_impact_remover_np_num(self):
        fairness_info = {
            "favorable_labels": [1.0],
            "protected_attributes": [{"feature": 57, "privileged_groups": [1.0]}],
        }
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
        self.assertTrue(0.8 < impact_orig < 1.0, f"impact_orig {impact_orig}")
        impact_remi = disparate_impact_scorer(trained_remi, test_X, test_y)
        self.assertTrue(0.9 < impact_remi < 1.0, f"impact_remi {impact_remi}")
        print(f"impact_orig {impact_orig}, impact_remi {impact_remi}")

    def test_adversarial_debiasing_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        tf.reset_default_graph()
        trainable_remi = AdversarialDebiasing(**fairness_info)
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.0, 1.1)

    def test_calibrated_eq_odds_postprocessing_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        estim = LogisticRegression(max_iter=1000)
        trainable_remi = CalibratedEqOddsPostprocessing(
            **fairness_info, estimator=estim
        )
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.7, 1.1)

    def test_disparate_impact_remover_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        trainable_remi = DisparateImpactRemover(**fairness_info) >> LogisticRegression(
            max_iter=1000
        )
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.8, 1.0)

    def test_eq_odds_postprocessing_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        estim = LogisticRegression(max_iter=1000)
        trainable_remi = EqOddsPostprocessing(**fairness_info, estimator=estim)
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.8, 1.1)

    def test_gerry_fair_classifier_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        trainable_remi = GerryFairClassifier(**fairness_info)
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.6, 1.1)

    def test_lfr_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        trainable_remi = LFR(**fairness_info) >> LogisticRegression(max_iter=1000)
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.9, 1.1)

    def test_meta_fair_classifier_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        _ = MetaFairClassifier(**fairness_info)
        # TODO: this test does not yet call fit or predict, since those hang

    def test_prejudice_remover_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        trainable_remi = PrejudiceRemover(**fairness_info)
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.6, 1.0)

    def test_redacting_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        redacting = Redacting(**fairness_info)
        logistic_regression = LogisticRegression(max_iter=1000)
        trainable_remi = redacting >> logistic_regression
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.8, 1.0)

    def test_reject_option_classification_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        estim = LogisticRegression(max_iter=1000)
        trainable_remi = RejectOptionClassification(**fairness_info, estimator=estim)
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.8, 1.1)

    def test_reweighing_pd_num(self):
        fairness_info = self.creditg_pd_num["fairness_info"]
        estim = LogisticRegression(max_iter=1000)
        trainable_remi = Reweighing(estimator=estim, **fairness_info)
        self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0.8, 1.1)

    def _attempt_remi_creditg_pd_cat(
        self, fairness_info, trainable_remi, min_di, max_di
    ):
        train_X = self.creditg_pd_cat["train_X"]
        train_y = self.creditg_pd_cat["train_y"]
        trained_remi = trainable_remi.fit(train_X, train_y)
        test_X = self.creditg_pd_cat["test_X"]
        test_y = self.creditg_pd_cat["test_y"]
        disparate_impact_scorer = lale.lib.aif360.disparate_impact(**fairness_info)
        impact_remi = disparate_impact_scorer(trained_remi, test_X, test_y)
        self.assertTrue(
            min_di <= impact_remi <= max_di, f"{min_di} <= {impact_remi} <= {max_di}",
        )
        return impact_remi

    def test_adversarial_debiasing_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        tf.reset_default_graph()
        trainable_remi = AdversarialDebiasing(
            **fairness_info, preprocessing=self.prep_pd_cat
        )
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.0, 1.1)

    def test_calibrated_eq_odds_postprocessing_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        estim = self.prep_pd_cat >> LogisticRegression(max_iter=1000)
        trainable_remi = CalibratedEqOddsPostprocessing(
            **fairness_info, estimator=estim
        )
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.7, 1.1)

    def test_disparate_impact_remover_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        trainable_remi = DisparateImpactRemover(
            **fairness_info, preprocessing=self.prep_pd_cat
        ) >> LogisticRegression(max_iter=1000)
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.8, 1.0)

    def test_eq_odds_postprocessing_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        estim = self.prep_pd_cat >> LogisticRegression(max_iter=1000)
        trainable_remi = EqOddsPostprocessing(**fairness_info, estimator=estim)
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.8, 1.2)

    def test_gerry_fair_classifier_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        trainable_remi = GerryFairClassifier(
            **fairness_info, preprocessing=self.prep_pd_cat
        )
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.0, 1.1)

    def test_lfr_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        trainable_remi = LFR(
            **fairness_info, preprocessing=self.prep_pd_cat
        ) >> LogisticRegression(max_iter=1000)
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.8, 1.0)

    def test_optim_preproc_pd_cat(self):
        # TODO: set the optimizer options as shown in the example https://github.com/Trusted-AI/AIF360/blob/master/examples/demo_optim_data_preproc.ipynb
        fairness_info = self.creditg_pd_cat["fairness_info"]
        _ = OptimPreproc(**fairness_info, optim_options={}) >> LogisticRegression(
            max_iter=1000
        )
        # TODO: this test does not yet call fit or predict

    def test_prejudice_remover_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        trainable_remi = PrejudiceRemover(
            **fairness_info, preprocessing=self.prep_pd_cat
        )
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.8, 1.0)

    def test_reject_option_classification_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        estim = self.prep_pd_cat >> LogisticRegression(max_iter=1000)
        trainable_remi = RejectOptionClassification(**fairness_info, estimator=estim)
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.7, 1.1)

    def test_reweighing_pd_cat(self):
        fairness_info = self.creditg_pd_cat["fairness_info"]
        estim = self.prep_pd_cat >> LogisticRegression(max_iter=1000)
        trainable_remi = Reweighing(estimator=estim, **fairness_info)
        self._attempt_remi_creditg_pd_cat(fairness_info, trainable_remi, 0.8, 1.2)
