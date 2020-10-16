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
from lale.datasets.data_schemas import pandas_to_ndarray
from lale.lib.lale import ConcatFeatures, Project
from lale.lib.sklearn import LogisticRegression, OneHotEncoder


class TestAIF360(unittest.TestCase):
    @classmethod
    def _creditg_pandas(cls):
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
    def _creditg_numeric(cls):
        (train_X, train_y), (test_X, test_y) = lale.lib.aif360.fetch_creditg_df()
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
    def _creditg_numpy(cls):
        train_X = pandas_to_ndarray(cls.creditg_pandas["train_X"])
        train_y = pandas_to_ndarray(cls.creditg_pandas["train_y"])
        test_X = pandas_to_ndarray(cls.creditg_pandas["test_X"])
        test_y = pandas_to_ndarray(cls.creditg_pandas["test_y"])
        assert isinstance(train_X, np.ndarray), type(train_X)
        assert isinstance(train_y, np.ndarray), type(train_y)
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
        cls.creditg_pandas = cls._creditg_pandas()
        cls.creditg_numeric = cls._creditg_numeric()
        cls.creditg_numpy = cls._creditg_numpy()

    def test_converter_pandas(self):
        info = self.creditg_pandas["fairness_info"]
        orig_X = self.creditg_pandas["train_X"]
        orig_y = self.creditg_pandas["train_y"]
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

    def test_converter_numpy(self):
        info = self.creditg_numpy["fairness_info"]
        orig_X = self.creditg_numpy["train_X"]
        orig_y = self.creditg_numpy["train_y"]
        converter = lale.lib.aif360.util._CategoricalFairnessConverter(**info)
        conv_X, conv_y = converter(orig_X, orig_y)
        for i in range(orig_X.shape[0]):
            conv_row = conv_X.loc[i]
            self.assertEqual(
                orig_X[i, 8].startswith("male"), conv_row["personal_status"],
            )
            self.assertEqual(orig_X[i, 12] >= 26, conv_row["age"])
            self.assertEqual(orig_y[i] == "good", conv_y[i])

    def test_disparate_impact_numeric(self):
        info = self.creditg_numeric["fairness_info"]
        disparate_impact_scorer = lale.lib.aif360.disparate_impact(**info)
        trainable = LogisticRegression(max_iter=1000)
        train_X = self.creditg_numeric["train_X"]
        train_y = self.creditg_numeric["train_y"]
        trained = trainable.fit(train_X, train_y)
        test_X = self.creditg_numeric["test_X"]
        test_y = self.creditg_numeric["test_y"]
        impact = disparate_impact_scorer(trained, test_X, test_y)
        self.assertLess(impact, 0.9)
        print(f"test_disparate_impact_numeric impact {impact:.2f}")

    def test_disparate_impact_pandas(self):
        info = self.creditg_pandas["fairness_info"]
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
        train_X = self.creditg_pandas["train_X"]
        train_y = self.creditg_pandas["train_y"]
        trained = trainable.fit(train_X, train_y)
        test_X = self.creditg_pandas["test_X"]
        test_y = self.creditg_pandas["test_y"]
        impact = disparate_impact_scorer(trained, test_X, test_y)
        self.assertLess(impact, 0.9)
        print(f"test_disparate_impact_pandas impact {impact:.2f}")
