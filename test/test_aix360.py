# Copyright  2020,2021 IBM Corporation
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

"""This testcase is to test the implementation of the aix360"""
import unittest

import pandas as pd
from aix360.algorithms.rbm import FeatureBinarizer

import lale
from lale.lib.aif360 import fair_stratified_train_test_split, fetch_creditg_df
from lale.lib.aix360 import logisticruleregression


class TestAIX360Datasets(unittest.TestCase):
    print(lale.__file__)

    def _attempt_dataset(self, all_x, all_y, fairness_info):
        train_x, test_x, yTrain, test_y = fair_stratified_train_test_split(
            all_x, all_y, **fairness_info, test_size=0.33, random_state=42
        )
        fb = FeatureBinarizer(negations=True, returnOrd=True)
        train_x, dfTrainStd = fb.fit_transform(train_x)
        test_x, dfTestStd = fb.transform(test_x)
        lrr = logisticruleregression(dfTrainStd=dfTrainStd, dfTestStd=dfTestStd)
        lrr.fit(train_x, yTrain)
        explain_value = lrr.explain()
        self.assertEqual(type(explain_value), pd.DataFrame)

    def test_dataset_creditg(self):
        all_X, all_y, fairness_info = fetch_creditg_df(preprocess=True)
        self._attempt_dataset(all_X, all_y, fairness_info)
