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
import sklearn.datasets
import sklearn.metrics

class TestSnapML(unittest.TestCase):
    def test_without_lale(self):
        import pai4sk
        X, y = sklearn.datasets.load_breast_cancer(return_X_y=True)
        clf = pai4sk.RandomForestClassifier()
        self.assertIsInstance(clf, pai4sk.RandomForestClassifier)
        fit_result = clf.fit(X, y)
        self.assertIsNone(fit_result)
        scorer = sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score)
        accuracy = scorer(clf, X, y)
