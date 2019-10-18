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
from lale.lib.lale import Batching, NoOp
from lale.lib.sklearn import MinMaxScaler
from lale.lib.sklearn import MLPClassifier, LogisticRegression
from sklearn.metrics import accuracy_score

class TestBatching(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(X, y)
           

    def test_fit(self):
        import warnings
        warnings.filterwarnings(action="ignore")
        from lale.lib.sklearn import MinMaxScaler, MLPClassifier
        pipeline = NoOp() >> Batching(operator = MinMaxScaler() >> MLPClassifier(random_state=42), batch_size = 112)
        trained = pipeline.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)
        lale_accuracy = accuracy_score(self.y_test, predictions)

        from sklearn.preprocessing import MinMaxScaler
        from sklearn.neural_network import MLPClassifier
        prep = MinMaxScaler()
        trained_prep = prep.partial_fit(self.X_train, self.y_train)
        X_transformed = trained_prep.transform(self.X_train)

        clf = MLPClassifier(random_state=42)
        import numpy as np
        trained_clf = clf.partial_fit(X_transformed, self.y_train, classes = np.unique(self.y_train))
        predictions = trained_clf.predict(trained_prep.transform(self.X_test))
        sklearn_accuracy = accuracy_score(self.y_test, predictions)

        self.assertEqual(lale_accuracy, sklearn_accuracy)

    def test_fit1(self):
        import warnings
        warnings.filterwarnings(action="ignore")
        from lale.lib.sklearn import MinMaxScaler, MLPClassifier
        pipeline = Batching(operator = MinMaxScaler() >> MLPClassifier(random_state=42), batch_size = 112)
        trained = pipeline.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)
        lale_accuracy = accuracy_score(self.y_test, predictions)

        from sklearn.preprocessing import MinMaxScaler
        from sklearn.neural_network import MLPClassifier
        prep = MinMaxScaler()
        trained_prep = prep.partial_fit(self.X_train, self.y_train)
        X_transformed = trained_prep.transform(self.X_train)

        clf = MLPClassifier(random_state=42)
        import numpy as np
        trained_clf = clf.partial_fit(X_transformed, self.y_train, classes = np.unique(self.y_train))
        predictions = trained_clf.predict(trained_prep.transform(self.X_test))
        sklearn_accuracy = accuracy_score(self.y_test, predictions)

        self.assertEqual(lale_accuracy, sklearn_accuracy)        

    def test_fit2(self):
        import warnings
        warnings.filterwarnings(action="ignore")
        from lale.lib.sklearn import MinMaxScaler, MLPClassifier
        pipeline = Batching(operator = MinMaxScaler() >> MinMaxScaler(), batch_size = 112)
        trained = pipeline.fit(self.X_train, self.y_train)
        lale_transforms = trained.transform(self.X_test)

        from sklearn.preprocessing import MinMaxScaler
        prep = MinMaxScaler()
        trained_prep = prep.partial_fit(self.X_train, self.y_train)
        X_transformed = trained_prep.transform(self.X_train)

        clf = MinMaxScaler()
        import numpy as np
        trained_clf = clf.partial_fit(X_transformed, self.y_train)
        sklearn_transforms = trained_clf.transform(trained_prep.transform(self.X_test))

        for i in range(5):
            for j in range(2):
                self.assertAlmostEqual(lale_transforms[i, j], sklearn_transforms[i, j])                

    def test_fit3(self):
        from lale.lib.sklearn import MinMaxScaler, MLPClassifier, PCA
        pipeline = PCA() >> Batching(operator = MinMaxScaler() >> MLPClassifier(random_state=42), 
                                                 batch_size = 10)        
        trained = pipeline.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)

    def test_no_partial_fit(self):
        pipeline = Batching(operator = NoOp() >> LogisticRegression())
        with self.assertRaises(AttributeError):
            trained = pipeline.fit(self.X_train, self.y_train)

    def test_fit4(self):
        import warnings
        warnings.filterwarnings(action="ignore")
        from lale.lib.sklearn import MinMaxScaler, MLPClassifier
        pipeline = Batching(operator = MinMaxScaler() >> MLPClassifier(random_state=42), batch_size = 112, inmemory=True)
        trained = pipeline.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)
        lale_accuracy = accuracy_score(self.y_test, predictions)

        from sklearn.preprocessing import MinMaxScaler
        from sklearn.neural_network import MLPClassifier
        prep = MinMaxScaler()
        trained_prep = prep.partial_fit(self.X_train, self.y_train)
        X_transformed = trained_prep.transform(self.X_train)

        clf = MLPClassifier(random_state=42)
        import numpy as np
        trained_clf = clf.partial_fit(X_transformed, self.y_train, classes = np.unique(self.y_train))
        predictions = trained_clf.predict(trained_prep.transform(self.X_test))
        sklearn_accuracy = accuracy_score(self.y_test, predictions)

        self.assertEqual(lale_accuracy, sklearn_accuracy)        

    # TODO: Nesting doesn't work yet
    # def test_nested_pipeline(self):
    #     from lale.lib.sklearn import MinMaxScaler, MLPClassifier
    #     pipeline = Batching(operator = MinMaxScaler() >> Batching(operator = NoOp() >> MLPClassifier(random_state=42)), batch_size = 112)
    #     trained = pipeline.fit(self.X_train, self.y_train)
    #     predictions = trained.predict(self.X_test)
    #     lale_accuracy = accuracy_score(self.y_test, predictions)