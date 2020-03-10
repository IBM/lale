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
import warnings
import random
import sys
import lale.operators as Ops
from lale.lib.lale import ConcatFeatures
from lale.lib.lale import NoOp
from lale.lib.sklearn import KNeighborsClassifier
from lale.lib.sklearn import LinearSVC
from lale.lib.sklearn import LogisticRegression
from lale.lib.sklearn import MinMaxScaler
from lale.lib.sklearn import MLPClassifier
from lale.lib.sklearn import Nystroem
from lale.lib.sklearn import OneHotEncoder
from lale.lib.sklearn import PCA
from lale.lib.sklearn import TfidfVectorizer
from lale.lib.sklearn import MultinomialNB
from lale.lib.sklearn import SimpleImputer
from lale.lib.sklearn import SVC
from lale.lib.xgboost import XGBClassifier
from lale.lib.sklearn import PassiveAggressiveClassifier
from lale.lib.sklearn import StandardScaler
from lale.lib.sklearn import FeatureAgglomeration
from typing import List

import sklearn.datasets

from lale.sklearn_compat import make_sklearn_compat
from lale.search.lale_smac import get_smac_space, lale_trainable_op_from_config
from lale.search.op2hp import hyperopt_search_space

@unittest.skip("Skipping here because travis-ci fails to allocate memory. This runs on internal travis.")
class TestResNet50(unittest.TestCase):
    def test_init_fit_predict(self):
        import torchvision.datasets as datasets
        import torchvision.transforms as transforms
        from lale.lib.pytorch import ResNet50

        transform = transforms.Compose([transforms.ToTensor()])

        data_train = datasets.FakeData(size = 50, num_classes=2 , transform = transform)#, target_transform = transform)
        clf = ResNet50(num_classes=2,num_epochs = 1)
        clf.fit(data_train)
        predicted = clf.predict(data_train)

