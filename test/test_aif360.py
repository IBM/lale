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
    def _creditg_numpy(cls):
        def to_numpy_with_schema(pandas_value):
            numpy_value = pandas_value.to_numpy()
            schema = pandas_value.json_schema
            return lale.datasets.data_schemas.add_schema(numpy_value, schema)

        train_X = to_numpy_with_schema(cls.creditg_pandas["train_X"])
        train_y = to_numpy_with_schema(cls.creditg_pandas["train_y"])
        test_X = to_numpy_with_schema(cls.creditg_pandas["test_X"])
        test_y = to_numpy_with_schema(cls.creditg_pandas["test_y"])
        assert isinstance(train_X, np.ndarray), type(train_X)
        assert isinstance(train_y, np.ndarray), type(train_y)
        orig_columns = list(cls.creditg_pandas["train_X"].columns)
        fairness_info = {
            "favorable_labels": [1.0],
            "protected_attributes": [
                {
                    "feature": orig_columns.index("personal_status"),
                    "privileged_groups": [
                        "male div/sep",
                        "male mar/wid",
                        "male single",
                    ],
                },
                {
                    "feature": orig_columns.index("age"),
                    "privileged_groups": [[26, 1000]],
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
        cls.creditg_pandas = cls._creditg_pandas()
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
