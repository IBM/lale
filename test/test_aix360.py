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

from lale.lib import aif360
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
#from lale.lib.aix360 import Diffprivlib
from lale.lib.aix360 import logisticaix360
from lale.lib.lale import ConcatFeatures, Project
from lale.lib.sklearn import (
    FunctionTransformer,
    LinearRegression,
    LogisticRegression,
    OneHotEncoder,
)

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
    def setUpClass(cls):
        cls.creditg_pd_num = cls._creditg_pd_num()
        
    
        
    
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
    
#    def test_Diffprivlib_pd_num(self):
 #           fairness_info = self.creditg_pd_num["fairness_info"]
  #          trainable_remi = Diffprivlib(**fairness_info)
   #         self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0, 1)

    def test_logisticaix360_pd_num(self):
            fairness_info = self.creditg_pd_num["fairness_info"]
            trainable_remi = logisticaix360(**fairness_info)
            self._attempt_remi_creditg_pd_num(fairness_info, trainable_remi, 0, 1)