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
from lale.operators import make_pipeline
from lale.operators import TrainablePipeline, TrainedPipeline
from lale.helpers import import_from_sklearn_pipeline

import pickle

from lale.lib.lale import ConcatFeatures
from lale.lib.lale import NoOp
from lale.lib.lale import Batching
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
from lale.lib.autogen import SGDClassifier
from sklearn.metrics import accuracy_score

class TestCreation(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_pipeline_create(self):
        from lale.lib.sklearn import PCA, LogisticRegression
        from lale.operators import Pipeline
        pipeline = Pipeline(([('pca1', PCA()), ('lr1', LogisticRegression())]))
        trained = pipeline.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)
        from sklearn.metrics import accuracy_score
        accuracy_score(self.y_test, predictions)
        
    def test_pipeline_clone(self):
        from sklearn.base import clone
        from lale.lib.sklearn import PCA, LogisticRegression
        from lale.operators import Pipeline
        pipeline=Pipeline(([('pca1', PCA()), ('lr1', LogisticRegression())]))
        trained=pipeline.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)
        from sklearn.metrics import accuracy_score
        orig_acc = accuracy_score(self.y_test, predictions)

        cloned_pipeline = clone(pipeline)
        trained = cloned_pipeline.fit(self.X_train, self.y_train)
        predictions = trained.predict(self.X_test)
        from sklearn.metrics import accuracy_score
        cloned_acc = accuracy_score(self.y_test, predictions)
        self.assertEqual(orig_acc, cloned_acc)

class TestImportExport(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(X, y)    

    def assert_equal_predictions(self, pipeline1, pipeline2):
        trained = pipeline1.fit(self.X_train, self.y_train)
        predictions1 = trained.predict(self.X_test)

        trained = pipeline2.fit(self.X_train, self.y_train)
        predictions2 = trained.predict(self.X_test)
        [self.assertEqual(p1, predictions2[i]) for i, p1 in enumerate(predictions1)]

    def test_import_from_sklearn_pipeline(self):
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_regression        
        from sklearn import svm
        from sklearn.pipeline import Pipeline
        anova_filter = SelectKBest(f_regression, k=3)
        clf = svm.SVC(kernel='linear')        
        sklearn_pipeline = Pipeline([('anova', anova_filter), ('svc', clf)])  
        lale_pipeline = import_from_sklearn_pipeline(sklearn_pipeline)
        for i, pipeline_step in enumerate(sklearn_pipeline.named_steps):
            sklearn_step_params = sklearn_pipeline.named_steps[pipeline_step].get_params()
            lale_sklearn_params = lale_pipeline.steps()[i]._impl._sklearn_model.get_params()
            self.assertEqual(sklearn_step_params, lale_sklearn_params)
        self.assert_equal_predictions(sklearn_pipeline, lale_pipeline)


    def test_import_from_sklearn_pipeline1(self):
        from sklearn.decomposition import PCA
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        sklearn_pipeline = make_pipeline(PCA(n_components=3), KNeighborsClassifier())
        lale_pipeline = import_from_sklearn_pipeline(sklearn_pipeline)
        for i, pipeline_step in enumerate(sklearn_pipeline.named_steps):
            sklearn_step_params = sklearn_pipeline.named_steps[pipeline_step].get_params()
            lale_sklearn_params = lale_pipeline.steps()[i]._impl._sklearn_model.get_params()
            self.assertEqual(sklearn_step_params, lale_sklearn_params)
        self.assert_equal_predictions(sklearn_pipeline, lale_pipeline)

    def test_import_from_sklearn_pipeline_feature_union(self):
        from sklearn.pipeline import FeatureUnion        
        from sklearn.decomposition import PCA
        from sklearn.kernel_approximation import Nystroem
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        union = FeatureUnion([("pca", PCA(n_components=1)), ("nys", Nystroem(n_components=2, random_state=42))])        
        sklearn_pipeline = make_pipeline(union, KNeighborsClassifier())
        lale_pipeline = import_from_sklearn_pipeline(sklearn_pipeline)
        self.assertEqual(len(lale_pipeline.edges()), 3)
        from lale.lib.sklearn.pca import PCAImpl
        from lale.lib.sklearn.nystroem import NystroemImpl
        from lale.lib.lale.concat_features import ConcatFeaturesImpl
        from lale.lib.sklearn.k_neighbors_classifier import KNeighborsClassifierImpl
        self.assertIsInstance(lale_pipeline.edges()[0][0]._impl, PCAImpl)
        self.assertIsInstance(lale_pipeline.edges()[0][1]._impl, ConcatFeaturesImpl)
        self.assertIsInstance(lale_pipeline.edges()[1][0]._impl, NystroemImpl)
        self.assertIsInstance(lale_pipeline.edges()[1][1]._impl, ConcatFeaturesImpl)
        self.assertIsInstance(lale_pipeline.edges()[2][0]._impl, ConcatFeaturesImpl)
        self.assertIsInstance(lale_pipeline.edges()[2][1]._impl, KNeighborsClassifierImpl)
        self.assert_equal_predictions(sklearn_pipeline, lale_pipeline)

    def test_import_from_sklearn_pipeline_nested_pipeline(self):
        from sklearn.pipeline import FeatureUnion, make_pipeline       
        from sklearn.decomposition import PCA
        from sklearn.kernel_approximation import Nystroem
        from sklearn.feature_selection import SelectKBest
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        union = FeatureUnion([("selectkbest_pca", make_pipeline(SelectKBest(k=3), PCA(n_components=1))), ("nys", Nystroem(n_components=2, random_state=42))])        
        sklearn_pipeline = make_pipeline(union, KNeighborsClassifier())
        lale_pipeline = import_from_sklearn_pipeline(sklearn_pipeline)
        self.assertEqual(len(lale_pipeline.edges()), 4)
        from lale.lib.sklearn.pca import PCAImpl
        from lale.lib.sklearn.nystroem import NystroemImpl
        from lale.lib.lale.concat_features import ConcatFeaturesImpl
        from lale.lib.sklearn.k_neighbors_classifier import KNeighborsClassifierImpl
        #These assertions assume topological sort
        self.assertIsInstance(lale_pipeline.edges()[0][0]._impl, SelectKBest)
        self.assertIsInstance(lale_pipeline.edges()[0][1]._impl, PCAImpl)
        self.assertIsInstance(lale_pipeline.edges()[1][0]._impl, PCAImpl)
        self.assertIsInstance(lale_pipeline.edges()[1][1]._impl, ConcatFeaturesImpl)
        self.assertIsInstance(lale_pipeline.edges()[2][0]._impl, NystroemImpl)
        self.assertIsInstance(lale_pipeline.edges()[2][1]._impl, ConcatFeaturesImpl)
        self.assertIsInstance(lale_pipeline.edges()[3][0]._impl, ConcatFeaturesImpl)
        self.assertIsInstance(lale_pipeline.edges()[3][1]._impl, KNeighborsClassifierImpl)
        self.assert_equal_predictions(sklearn_pipeline, lale_pipeline)

    def test_import_from_sklearn_pipeline_nested_pipeline1(self):
        from sklearn.pipeline import FeatureUnion, make_pipeline       
        from sklearn.decomposition import PCA
        from sklearn.kernel_approximation import Nystroem
        from sklearn.feature_selection import SelectKBest
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        union = FeatureUnion([("selectkbest_pca", make_pipeline(SelectKBest(k=3), FeatureUnion([('pca', PCA(n_components=1)), ('nested_pipeline', make_pipeline(SelectKBest(k=2), Nystroem()))]))), ("nys", Nystroem(n_components=2, random_state=42))])        
        sklearn_pipeline = make_pipeline(union, KNeighborsClassifier())
        lale_pipeline = import_from_sklearn_pipeline(sklearn_pipeline)
        print(lale_pipeline.to_json())
        self.assertEqual(len(lale_pipeline.edges()), 8)
        #These assertions assume topological sort, which may not be unique. So the assertions are brittle.
        from lale.lib.sklearn.pca import PCAImpl
        from lale.lib.sklearn.nystroem import NystroemImpl
        from lale.lib.lale.concat_features import ConcatFeaturesImpl
        from lale.lib.sklearn.k_neighbors_classifier import KNeighborsClassifierImpl
        self.assertIsInstance(lale_pipeline.edges()[0][0]._impl, SelectKBest)
        self.assertIsInstance(lale_pipeline.edges()[0][1]._impl, PCAImpl)
        self.assertIsInstance(lale_pipeline.edges()[1][0]._impl, SelectKBest)
        self.assertIsInstance(lale_pipeline.edges()[1][1]._impl, SelectKBest)
        self.assertIsInstance(lale_pipeline.edges()[2][0]._impl, SelectKBest)
        self.assertIsInstance(lale_pipeline.edges()[2][1]._impl, NystroemImpl)
        self.assertIsInstance(lale_pipeline.edges()[3][0]._impl, PCAImpl)
        self.assertIsInstance(lale_pipeline.edges()[3][1]._impl, ConcatFeaturesImpl)
        self.assertIsInstance(lale_pipeline.edges()[4][0]._impl, NystroemImpl)
        self.assertIsInstance(lale_pipeline.edges()[4][1]._impl, ConcatFeaturesImpl)
        self.assertIsInstance(lale_pipeline.edges()[5][0]._impl, ConcatFeaturesImpl)
        self.assertIsInstance(lale_pipeline.edges()[5][1]._impl, ConcatFeaturesImpl)
        self.assertIsInstance(lale_pipeline.edges()[6][0]._impl, NystroemImpl)
        self.assertIsInstance(lale_pipeline.edges()[6][1]._impl, ConcatFeaturesImpl)
        self.assertIsInstance(lale_pipeline.edges()[7][0]._impl, ConcatFeaturesImpl)
        self.assertIsInstance(lale_pipeline.edges()[7][1]._impl, KNeighborsClassifierImpl)
        self.assert_equal_predictions(sklearn_pipeline, lale_pipeline)

    def test_import_from_sklearn_pipeline_nested_pipeline2(self):
        from sklearn.pipeline import FeatureUnion, make_pipeline       
        from sklearn.decomposition import PCA
        from sklearn.kernel_approximation import Nystroem
        from sklearn.feature_selection import SelectKBest
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        union = FeatureUnion([("selectkbest_pca", make_pipeline(SelectKBest(k=3), make_pipeline(SelectKBest(k=2), PCA()))), ("nys", Nystroem(n_components=2, random_state=42))])        
        sklearn_pipeline = make_pipeline(union, KNeighborsClassifier())
        lale_pipeline = import_from_sklearn_pipeline(sklearn_pipeline)
        self.assertEqual(len(lale_pipeline.edges()), 5)
        from lale.lib.sklearn.pca import PCAImpl
        from lale.lib.sklearn.nystroem import NystroemImpl
        from lale.lib.lale.concat_features import ConcatFeaturesImpl
        from lale.lib.sklearn.k_neighbors_classifier import KNeighborsClassifierImpl
        self.assertIsInstance(lale_pipeline.edges()[0][0]._impl, SelectKBest)
        self.assertIsInstance(lale_pipeline.edges()[0][1]._impl, SelectKBest)
        self.assertIsInstance(lale_pipeline.edges()[1][0]._impl, SelectKBest)
        self.assertIsInstance(lale_pipeline.edges()[1][1]._impl, PCAImpl)
        self.assertIsInstance(lale_pipeline.edges()[2][0]._impl, PCAImpl)
        self.assertIsInstance(lale_pipeline.edges()[2][1]._impl, ConcatFeaturesImpl)
        self.assertIsInstance(lale_pipeline.edges()[3][0]._impl, NystroemImpl)
        self.assertIsInstance(lale_pipeline.edges()[3][1]._impl, ConcatFeaturesImpl)
        self.assertIsInstance(lale_pipeline.edges()[4][0]._impl, ConcatFeaturesImpl)
        self.assertIsInstance(lale_pipeline.edges()[4][1]._impl, KNeighborsClassifierImpl)

        self.assert_equal_predictions(sklearn_pipeline, lale_pipeline)

    def test_export_to_sklearn_pipeline(self):
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import KNeighborsClassifier
        from sklearn.pipeline import make_pipeline
        lale_pipeline = PCA(n_components=3) >>  KNeighborsClassifier()
        trained_lale_pipeline = lale_pipeline.fit(self.X_train, self.y_train)
        sklearn_pipeline = trained_lale_pipeline.export_to_sklearn_pipeline()
        for i, pipeline_step in enumerate(sklearn_pipeline.named_steps):
            sklearn_step_params = sklearn_pipeline.named_steps[pipeline_step].get_params()
            lale_sklearn_params = trained_lale_pipeline.steps()[i]._impl._sklearn_model.get_params()
            self.assertEqual(sklearn_step_params, lale_sklearn_params)
        self.assert_equal_predictions(sklearn_pipeline, trained_lale_pipeline)

    def test_export_to_sklearn_pipeline1(self):
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import KNeighborsClassifier
        from sklearn.feature_selection import SelectKBest
        from sklearn.pipeline import make_pipeline
        lale_pipeline = SelectKBest(k=3) >>  KNeighborsClassifier()
        trained_lale_pipeline = lale_pipeline.fit(self.X_train, self.y_train)
        sklearn_pipeline = trained_lale_pipeline.export_to_sklearn_pipeline()
        for i, pipeline_step in enumerate(sklearn_pipeline.named_steps):
            sklearn_step_params = type(sklearn_pipeline.named_steps[pipeline_step])
            lale_sklearn_params = type(trained_lale_pipeline.steps()[i]._impl._sklearn_model) if hasattr(trained_lale_pipeline.steps()[i]._impl, '_sklearn_model') else type(trained_lale_pipeline.steps()[i]._impl)                
            self.assertEqual(sklearn_step_params, lale_sklearn_params)
        self.assert_equal_predictions(sklearn_pipeline, trained_lale_pipeline)

    def test_export_to_sklearn_pipeline2(self):
        from lale.lib.lale import ConcatFeatures
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import KNeighborsClassifier
        from sklearn.feature_selection import SelectKBest
        from lale.lib.sklearn import Nystroem
        from sklearn.pipeline import FeatureUnion

        lale_pipeline = (((PCA() & SelectKBest(k=3)) >> ConcatFeatures()) & Nystroem()) >> ConcatFeatures() >> KNeighborsClassifier()
        trained_lale_pipeline = lale_pipeline.fit(self.X_train, self.y_train)
        sklearn_pipeline = trained_lale_pipeline.export_to_sklearn_pipeline()
        self.assertIsInstance(sklearn_pipeline.named_steps['featureunion'], FeatureUnion)
        from sklearn.neighbors import KNeighborsClassifier
        self.assertIsInstance(sklearn_pipeline.named_steps['kneighborsclassifier'], KNeighborsClassifier)
        self.assert_equal_predictions(sklearn_pipeline, trained_lale_pipeline)

    def test_export_to_sklearn_pipeline3(self):
        from lale.lib.lale import ConcatFeatures
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import KNeighborsClassifier, LogisticRegression, SVC 
        from sklearn.feature_selection import SelectKBest
        from lale.lib.sklearn import Nystroem
        from sklearn.pipeline import FeatureUnion

        lale_pipeline = ((PCA() >> SelectKBest(k=2)) & (Nystroem(random_state = 42) >> SelectKBest(k=3))
         & (SelectKBest(k=3))) >> ConcatFeatures() >> SelectKBest(k=2) >> LogisticRegression()
        trained_lale_pipeline = lale_pipeline.fit(self.X_train, self.y_train)
        sklearn_pipeline = trained_lale_pipeline.export_to_sklearn_pipeline()
        self.assertIsInstance(sklearn_pipeline.named_steps['featureunion'], FeatureUnion)
        self.assertIsInstance(sklearn_pipeline.named_steps['selectkbest'], SelectKBest)
        from sklearn.linear_model import LogisticRegression
        self.assertIsInstance(sklearn_pipeline.named_steps['logisticregression'], LogisticRegression)
        self.assert_equal_predictions(sklearn_pipeline, trained_lale_pipeline)

    def test_export_to_sklearn_pipeline4(self):
        from lale.lib.sklearn import LogisticRegression
        from lale.operators import make_pipeline

        lale_pipeline = make_pipeline(LogisticRegression())
        trained_lale_pipeline = lale_pipeline.fit(self.X_train, self.y_train)
        sklearn_pipeline = trained_lale_pipeline.export_to_sklearn_pipeline()
        from sklearn.linear_model import LogisticRegression
        self.assertIsInstance(sklearn_pipeline.named_steps['logisticregression'], LogisticRegression)
        self.assert_equal_predictions(sklearn_pipeline, trained_lale_pipeline)

    #Doesn't work yet.
    # def test_export_to_pickle(self):
    #     from lale.lib.sklearn import LogisticRegression
    #     from lale.operators import make_pipeline

    #     lale_pipeline = make_pipeline(LogisticRegression())
    #     trained_lale_pipeline = lale_pipeline.fit(self.X_train, self.y_train)
    #     pickle.dumps(lale_pipeline)
    #     pickle.dumps(trained_lale_pipeline)

class TestComposition(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split
        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test =  train_test_split(X, y)    
    def test_two_estimators_predict(self):
        pipeline = StandardScaler()  >> ( PCA() & Nystroem() & LogisticRegression() )>>ConcatFeatures() >> NoOp() >> LogisticRegression()
        trained = pipeline.fit(self.X_train, self.y_train)
        trained.predict(self.X_test)
    def test_two_estimators_predict1(self):
        pipeline = StandardScaler()  >> ( PCA() & Nystroem() & PassiveAggressiveClassifier() )>>ConcatFeatures() >> NoOp() >> PassiveAggressiveClassifier()
        trained = pipeline.fit(self.X_train, self.y_train)
        trained.predict(self.X_test)
    def test_two_estimators_predict_proba(self):
        pipeline = StandardScaler()  >> ( PCA() & Nystroem() & LogisticRegression() )>>ConcatFeatures() >> NoOp() >> LogisticRegression()
        pipeline.fit(self.X_train, self.y_train)
        pipeline.predict_proba(self.X_test)
    def test_two_estimators_predict_proba1(self):
        pipeline = StandardScaler()  >> ( PCA() & Nystroem() & PassiveAggressiveClassifier() )>>ConcatFeatures() >> NoOp() >> PassiveAggressiveClassifier()
        pipeline.fit(self.X_train, self.y_train)
        with self.assertRaises(ValueError):
            pipeline.predict_proba(self.X_test)
    def test_multiple_estimators_predict_predict_proba(self) :
        pipeline = (
            StandardScaler() >>
            ( LogisticRegression() & PCA() ) >> ConcatFeatures() >>
            ( NoOp() & LinearSVC() ) >> ConcatFeatures() >>
            KNeighborsClassifier()
        )
        pipeline.fit(self.X_train, self.y_train)
        tmp = pipeline.predict_proba(self.X_test)
        tmp = pipeline.predict(self.X_test)

