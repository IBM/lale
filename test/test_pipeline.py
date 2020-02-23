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
from lale.lib.lale import Batching, NoOp
from lale.lib.sklearn import MinMaxScaler
from lale.lib.sklearn import MLPClassifier, LogisticRegression
from lale.lib.sklearn import Nystroem
from lale.lib.sklearn import PCA

from sklearn.metrics import accuracy_score
from lale.search.lale_grid_search_cv import get_grid_search_parameter_grids

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

class TestPipeline(unittest.TestCase):

    def dont_test_with_gridsearchcv2_auto(self):
        from sklearn.model_selection import GridSearchCV
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
        lr = LogisticRegression(random_state = 42)
        pca = PCA(random_state = 42, svd_solver = 'arpack')
        trainable = pca >> lr
        from sklearn.pipeline import Pipeline
        scikit_pipeline = Pipeline([(pca.name(), PCA(random_state = 42, svd_solver = 'arpack')), (lr.name(), LogisticRegression(random_state = 42))])
        all_parameters = get_grid_search_parameter_grids(trainable, num_samples=1)
        # otherwise the test takes too long
        parameters = random.sample(all_parameters, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = GridSearchCV(scikit_pipeline, parameters, cv=2, scoring=make_scorer(accuracy_score))
            iris = load_iris()
            clf.fit(iris.data, iris.target)
            predicted = clf.predict(iris.data)
            accuracy_with_lale_operators = accuracy_score(iris.target, predicted)

        from sklearn.pipeline import Pipeline
        from sklearn.decomposition import PCA as SklearnPCA
        from sklearn.linear_model import LogisticRegression as SklearnLR
        scikit_pipeline = Pipeline([(pca.name(), SklearnPCA(random_state = 42, svd_solver = 'arpack')), (lr.name(), SklearnLR(random_state = 42))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = GridSearchCV(scikit_pipeline, parameters, cv=2, scoring=make_scorer(accuracy_score))
            iris = load_iris()
            clf.fit(iris.data, iris.target)
            predicted = clf.predict(iris.data)
            accuracy_with_scikit_operators = accuracy_score(iris.target, predicted)
        self.assertEqual(accuracy_with_lale_operators, accuracy_with_scikit_operators)

    def test_with_gridsearchcv3(self):
        from sklearn.model_selection import GridSearchCV
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
        lr = LogisticRegression()
        from sklearn.pipeline import Pipeline
        scikit_pipeline = Pipeline([("nystroem", Nystroem()), ("lr", LogisticRegression())])
        parameters = {'lr__solver':('liblinear', 'lbfgs'), 'lr__penalty':['l2']}
        clf = GridSearchCV(scikit_pipeline, parameters, cv=2, scoring=make_scorer(accuracy_score))
        iris = load_iris()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(iris.data, iris.target)
        predicted = clf.predict(iris.data)

    def test_with_gridsearchcv3_auto(self):
        from sklearn.model_selection import GridSearchCV
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer
        lr = LogisticRegression()
        from sklearn.pipeline import Pipeline
        scikit_pipeline = Pipeline([(Nystroem().name(), Nystroem()), (lr.name(), LogisticRegression())])
        all_parameters = get_grid_search_parameter_grids(Nystroem()>>lr, num_samples=1)
        # otherwise the test takes too long
        parameters = random.sample(all_parameters, 2)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")

            clf = GridSearchCV(scikit_pipeline, parameters, cv=2, scoring=make_scorer(accuracy_score))
            iris = load_iris()
            clf.fit(iris.data, iris.target)
            predicted = clf.predict(iris.data)

    def test_with_gridsearchcv3_auto_wrapped(self):
        from sklearn.datasets import load_iris
        from sklearn.metrics import accuracy_score, make_scorer

        pipeline = Nystroem() >> LogisticRegression()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from lale.lib.lale import GridSearchCV
            clf = GridSearchCV(
                estimator=pipeline, lale_num_samples=1, lale_num_grids=1,
                cv=2, scoring=make_scorer(accuracy_score))
            iris = load_iris()
            clf.fit(iris.data, iris.target)
            predicted = clf.predict(iris.data)

class TestBatching(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(X, y)    

    def test_batching_with_hyperopt(self):
        from lale.lib.sklearn import MinMaxScaler, SGDClassifier
        from lale.lib.lale import Hyperopt, Batching
        from sklearn.metrics import accuracy_score

        pipeline = Batching(operator=MinMaxScaler() >> SGDClassifier())
        trained = pipeline.auto_configure(self.X_train, self.y_train, optimizer=Hyperopt, max_evals=1) 
        predictions = trained.predict(self.X_test)
