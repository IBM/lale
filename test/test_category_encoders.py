# Copyright 2022 IBM Corporation
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

import lale.datasets
import lale.datasets.openml
from lale.lib.category_encoders import HashingEncoder, TargetEncoder
from lale.lib.rasl import ConcatFeatures, Project
from lale.lib.sklearn import LogisticRegression


class TestCategoryEncoders(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.creditg = lale.datasets.openml.fetch(
            "credit-g",
            "classification",
            preprocess=False,
            astype="pandas",
        )

    def test_hashing_encoder(self):
        (train_X, train_y), (test_X, _test_y) = self.creditg
        cat_prep = Project(columns={"type": "string"}) >> HashingEncoder()
        num_prep = Project(columns={"type": "number"})
        trainable = (
            (cat_prep & num_prep) >> ConcatFeatures >> LogisticRegression(max_iter=1000)
        )
        trained = trainable.fit(train_X, train_y)
        _ = trained.predict(test_X)

    def test_target_encoder(self):
        (train_X, train_y), (test_X, _test_y) = self.creditg
        cat_prep = Project(columns={"type": "string"}) >> TargetEncoder()
        num_prep = Project(columns={"type": "number"})
        trainable = (
            (cat_prep & num_prep) >> ConcatFeatures >> LogisticRegression(max_iter=1000)
        )
        trained = trainable.fit(train_X, train_y)
        _ = trained.predict(test_X)
