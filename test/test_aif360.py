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

import lale.datasets.data_schemas
import lale.datasets.openml
import lale.lib.aif360
import lale.lib.aif360.util
from lale.datasets.data_schemas import NDArrayWithSchema
from lale.lib.lale import ConcatFeatures, Project
from lale.lib.sklearn import FunctionTransformer, LogisticRegression, OneHotEncoder


class TestAIF360(unittest.TestCase):
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
            "favorable_label": 1,
            "unfavorable_label": 0,
            "protected_attribute_names": ["sex", "age"],
            "unprivileged_groups": [{"sex": 0, "age": 0}],
            "privileged_groups": [{"sex": 1, "age": 1}],
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
    def setUpClass(cls):
        cls.creditg_pd_cat = cls._creditg_pd_cat()
        cls.creditg_pd_num = cls._creditg_pd_num()
        cls.creditg_np_cat = cls._creditg_np_cat()

    def test_converter_pd_cat(self):
        info = self.creditg_pd_cat["fairness_info"]
        orig_X = self.creditg_pd_cat["train_X"]
        orig_y = self.creditg_pd_cat["train_y"]
        converter = lale.lib.aif360.util._CategoricalFairnessConverter(**info)
        conv_X, conv_y = converter(orig_X, orig_y)
        for i in orig_X.index:
            orig_row = orig_X.loc[i]
            conv_row = conv_X.loc[i]
            self.assertEqual(
                orig_row["personal_status"].startswith("male"),
                conv_row["personal_status"],
            )
            self.assertEqual(orig_row["age"] >= 26, conv_row["age"])
            self.assertEqual(orig_y[i] == "good", conv_y[i])

    def test_converter_np_cat(self):
        info = self.creditg_np_cat["fairness_info"]
        orig_X = self.creditg_np_cat["train_X"]
        orig_y = self.creditg_np_cat["train_y"]
        converter = lale.lib.aif360.util._CategoricalFairnessConverter(**info)
        conv_X, conv_y = converter(orig_X, orig_y)
        for i in range(orig_X.shape[0]):
            self.assertEqual(
                orig_X[i, 8].startswith("male"), conv_X.at[i, "f8"],
            )
            self.assertEqual(orig_X[i, 12] >= 26, conv_X.at[i, "f12"])
            self.assertEqual(orig_y[i] == "good", conv_y[i])

    def test_disparate_impact_pd_num(self):
        info = self.creditg_pd_num["fairness_info"]
        disparate_impact_scorer = lale.lib.aif360.disparate_impact(**info)
        trainable = LogisticRegression(max_iter=1000)
        train_X = self.creditg_pd_num["train_X"]
        train_y = self.creditg_pd_num["train_y"]
        trained = trainable.fit(train_X, train_y)
        test_X = self.creditg_pd_num["test_X"]
        test_y = self.creditg_pd_num["test_y"]
        impact = disparate_impact_scorer(trained, test_X, test_y)
        self.assertLess(impact, 0.9)
        print(f"test_disparate_impact_pd_num impact {impact:.2f}")
        combined_scorer = lale.lib.aif360.accuracy_and_disparate_impact(**info)
        score = combined_scorer(trained, test_X, test_y)
        self.assertEqual(score, -99)

    def test_disparate_impact_pd_cat(self):
        info = self.creditg_pd_cat["fairness_info"]
        disparate_impact_scorer = lale.lib.aif360.disparate_impact(**info)
        trainable = (
            (
                (
                    Project(columns={"type": "string"})
                    >> OneHotEncoder(handle_unknown="ignore")
                )
                & Project(columns={"type": "number"})
            )
            >> ConcatFeatures
            >> LogisticRegression(max_iter=1000)
        )
        train_X = self.creditg_pd_cat["train_X"]
        train_y = self.creditg_pd_cat["train_y"]
        trained = trainable.fit(train_X, train_y)
        test_X = self.creditg_pd_cat["test_X"]
        test_y = self.creditg_pd_cat["test_y"]
        impact = disparate_impact_scorer(trained, test_X, test_y)
        self.assertLess(impact, 0.9)
        print(f"test_disparate_impact_pd_cat impact {impact:.2f}")
        combined_scorer = lale.lib.aif360.accuracy_and_disparate_impact(**info)
        score = combined_scorer(trained, test_X, test_y)
        self.assertEqual(score, -99)

    def test_disparate_impact_np_cat(self):
        info = self.creditg_np_cat["fairness_info"]
        disparate_impact_scorer = lale.lib.aif360.disparate_impact(**info)
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
        impact = disparate_impact_scorer(trained, test_X, test_y)
        self.assertLess(impact, 0.9)
        print(f"test_disparate_impact_np_cat impact {impact:.2f}")
        combined_scorer = lale.lib.aif360.accuracy_and_disparate_impact(**info)
        score = combined_scorer(trained, test_X, test_y)
        self.assertEqual(score, -99)
