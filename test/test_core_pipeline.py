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
import sklearn.datasets
from lale.operators import make_pipeline
from lale.operators import TrainablePipeline, TrainedPipeline
from lale.helpers import import_from_sklearn_pipeline

import pickle

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
from lale.lib.autogen import SGDClassifier
from sklearn.metrics import accuracy_score

from lale.sklearn_compat import make_sklearn_compat
from lale.search.lale_grid_search_cv import get_grid_search_parameter_grids

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

    def test_make_pipeline(self):
        from lale.operators import make_pipeline
        tfm = PCA(n_components=10)
        clf = LogisticRegression(random_state=42)
        trainable = make_pipeline(tfm, clf)
        digits = sklearn.datasets.load_digits()
        trained = trainable.fit(digits.data, digits.target)
        predicted = trained.predict(digits.data)
    def test_compose2(self):
        from lale.operators import make_pipeline
        tfm = PCA(n_components=10)
        clf = LogisticRegression(random_state=42)
        trainable = tfm >> clf
        digits = sklearn.datasets.load_digits()
        trained = trainable.fit(digits.data, digits.target)
        predicted = trained.predict(digits.data)
    def test_compose3(self):
        from lale.operators import make_pipeline
        nys = Nystroem(n_components=15)
        pca = PCA(n_components=10)
        lr = LogisticRegression(random_state=42)
        trainable = nys >> pca >> lr
        digits = sklearn.datasets.load_digits()
        trained = trainable.fit(digits.data, digits.target)
        predicted = trained.predict(digits.data)
    def test_pca_nys_lr(self):
        from lale.operators import make_union
        nys = Nystroem(n_components=15)
        pca = PCA(n_components=10)
        lr = LogisticRegression(random_state=42)
        trainable = make_union(nys, pca) >> lr
        digits = sklearn.datasets.load_digits()
        trained = trainable.fit(digits.data, digits.target)
        predicted = trained.predict(digits.data)
    def test_compose4(self):
        from lale.operators import make_choice
        digits = sklearn.datasets.load_digits()
        ohe = OneHotEncoder(handle_unknown=OneHotEncoder.handle_unknown.ignore)
        ohe.get_params()
        no_op = NoOp()
        pca = PCA()
        nys = Nystroem()
        lr = LogisticRegression()
        knn = KNeighborsClassifier()
        step1 = ohe | no_op
        step2 = pca | nys
        step3 = lr | knn
        model_plan = step1 >> step2 >> step3
        #TODO: optimize on this plan and then fit and predict
    def test_compose5(self):
        ohe = OneHotEncoder(handle_unknown=OneHotEncoder.handle_unknown.ignore)
        digits = sklearn.datasets.load_digits()
        lr = LogisticRegression()
        lr_trained = lr.fit(digits.data, digits.target)
        lr_trained.predict(digits.data)
        pipeline1 = ohe >> lr
        pipeline1_trained = pipeline1.fit(digits.data, digits.target)
        pipeline1_trained.predict(digits.data)

    def test_compare_with_sklearn(self):
        from lale.operators import make_pipeline
        tfm = PCA()
        clf = LogisticRegression(LogisticRegression.solver.lbfgs, LogisticRegression.multi_class.auto)
        trainable = make_pipeline(tfm, clf)
        digits = sklearn.datasets.load_digits()
        trained = trainable.fit(digits.data, digits.target)
        predicted = trained.predict(digits.data)
        from sklearn.pipeline import make_pipeline as scikit_make_pipeline
        from sklearn.decomposition import PCA as SklearnPCA
        from sklearn.linear_model import LogisticRegression as SklearnLR
        sklearn_pipeline = scikit_make_pipeline(SklearnPCA(), SklearnLR(solver="lbfgs", multi_class="auto"))
        sklearn_pipeline.fit(digits.data, digits.target)
        predicted_sklearn = sklearn_pipeline.predict(digits.data)

        from sklearn.metrics import accuracy_score
        lale_score = accuracy_score(digits.target, predicted)
        scikit_score = accuracy_score(digits.target, predicted_sklearn)
        self.assertEqual(lale_score, scikit_score)

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
        self.assertEquals(lale_pipeline.edges()[0][0]._impl_class(), PCAImpl)
        self.assertEquals(lale_pipeline.edges()[0][1]._impl_class(), ConcatFeaturesImpl)
        self.assertEquals(lale_pipeline.edges()[1][0]._impl_class(), NystroemImpl)
        self.assertEquals(lale_pipeline.edges()[1][1]._impl_class(), ConcatFeaturesImpl)
        self.assertEquals(lale_pipeline.edges()[2][0]._impl_class(), ConcatFeaturesImpl)
        self.assertEquals(lale_pipeline.edges()[2][1]._impl_class(), KNeighborsClassifierImpl)
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
        from lale.lib.sklearn.select_k_best import SelectKBestImpl
        #These assertions assume topological sort
        self.assertEquals(lale_pipeline.edges()[0][0]._impl_class(), SelectKBestImpl)
        self.assertEquals(lale_pipeline.edges()[0][1]._impl_class(), PCAImpl)
        self.assertEquals(lale_pipeline.edges()[1][0]._impl_class(), PCAImpl)
        self.assertEquals(lale_pipeline.edges()[1][1]._impl_class(), ConcatFeaturesImpl)
        self.assertEquals(lale_pipeline.edges()[2][0]._impl_class(), NystroemImpl)
        self.assertEquals(lale_pipeline.edges()[2][1]._impl_class(), ConcatFeaturesImpl)
        self.assertEquals(lale_pipeline.edges()[3][0]._impl_class(), ConcatFeaturesImpl)
        self.assertEquals(lale_pipeline.edges()[3][1]._impl_class(), KNeighborsClassifierImpl)
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
        from lale.lib.sklearn.select_k_best import SelectKBestImpl
        self.assertEquals(lale_pipeline.edges()[0][0]._impl_class(), SelectKBestImpl)
        self.assertEquals(lale_pipeline.edges()[0][1]._impl_class(), PCAImpl)
        self.assertEquals(lale_pipeline.edges()[1][0]._impl_class(), SelectKBestImpl)
        self.assertEquals(lale_pipeline.edges()[1][1]._impl_class(), SelectKBestImpl)
        self.assertEquals(lale_pipeline.edges()[2][0]._impl_class(), SelectKBestImpl)
        self.assertEquals(lale_pipeline.edges()[2][1]._impl_class(), NystroemImpl)
        self.assertEquals(lale_pipeline.edges()[3][0]._impl_class(), PCAImpl)
        self.assertEquals(lale_pipeline.edges()[3][1]._impl_class(), ConcatFeaturesImpl)
        self.assertEquals(lale_pipeline.edges()[4][0]._impl_class(), NystroemImpl)
        self.assertEquals(lale_pipeline.edges()[4][1]._impl_class(), ConcatFeaturesImpl)
        self.assertEquals(lale_pipeline.edges()[5][0]._impl_class(), ConcatFeaturesImpl)
        self.assertEquals(lale_pipeline.edges()[5][1]._impl_class(), ConcatFeaturesImpl)
        self.assertEquals(lale_pipeline.edges()[6][0]._impl_class(), NystroemImpl)
        self.assertEquals(lale_pipeline.edges()[6][1]._impl_class(), ConcatFeaturesImpl)
        self.assertEquals(lale_pipeline.edges()[7][0]._impl_class(), ConcatFeaturesImpl)
        self.assertEquals(lale_pipeline.edges()[7][1]._impl_class(), KNeighborsClassifierImpl)
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
        from lale.lib.sklearn.select_k_best import SelectKBestImpl
        self.assertEquals(lale_pipeline.edges()[0][0]._impl_class(), SelectKBestImpl)
        self.assertEquals(lale_pipeline.edges()[0][1]._impl_class(), SelectKBestImpl)
        self.assertEquals(lale_pipeline.edges()[1][0]._impl_class(), SelectKBestImpl)
        self.assertEquals(lale_pipeline.edges()[1][1]._impl_class(), PCAImpl)
        self.assertEquals(lale_pipeline.edges()[2][0]._impl_class(), PCAImpl)
        self.assertEquals(lale_pipeline.edges()[2][1]._impl_class(), ConcatFeaturesImpl)
        self.assertEquals(lale_pipeline.edges()[3][0]._impl_class(), NystroemImpl)
        self.assertEquals(lale_pipeline.edges()[3][1]._impl_class(), ConcatFeaturesImpl)
        self.assertEquals(lale_pipeline.edges()[4][0]._impl_class(), ConcatFeaturesImpl)
        self.assertEquals(lale_pipeline.edges()[4][1]._impl_class(), KNeighborsClassifierImpl)

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

        lale_pipeline = (((PCA(svd_solver='randomized', random_state=42) & SelectKBest(k=3)) >> ConcatFeatures()) & Nystroem(random_state=42)) >> ConcatFeatures() >> KNeighborsClassifier()
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

    def test_export_to_pickle(self):
        from lale.lib.sklearn import LogisticRegression
        from lale.operators import make_pipeline

        lale_pipeline = make_pipeline(LogisticRegression())
        trained_lale_pipeline = lale_pipeline.fit(self.X_train, self.y_train)
        pickle.dumps(lale_pipeline)
        pickle.dumps(trained_lale_pipeline)

    def test_import_from_sklearn_pipeline2(self):
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_regression        
        from sklearn import svm
        from sklearn.pipeline import Pipeline
        anova_filter = SelectKBest(f_regression, k=3)
        clf = svm.SVC(kernel='linear')        
        sklearn_pipeline = Pipeline([('anova', anova_filter), ('svc', clf)])
        sklearn_pipeline.fit(self.X_train, self.y_train)
        lale_pipeline = import_from_sklearn_pipeline(sklearn_pipeline)
        lale_pipeline.predict(self.X_test)

    def test_import_from_sklearn_pipeline3(self):
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import f_regression        
        from sklearn import svm
        from sklearn.pipeline import Pipeline
        anova_filter = SelectKBest(f_regression, k=3)
        clf = svm.SVC(kernel='linear')        
        sklearn_pipeline = Pipeline([('anova', anova_filter), ('svc', clf)])
        lale_pipeline = import_from_sklearn_pipeline(sklearn_pipeline, fitted=False)
        with self.assertRaises(ValueError):#fitted=False returns a Trainable, so calling predict is invalid.
            lale_pipeline.predict(self.X_test)

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

    @unittest.skip('#258 - PassiveAggressiveClassifier does not have predict_proba and hence predict gets called, this is valid for iris dataset.')
    def test_two_estimators_predict1(self):
        pipeline = StandardScaler()  >> ( PCA() & Nystroem() & PassiveAggressiveClassifier() )>>ConcatFeatures() >> NoOp() >> PassiveAggressiveClassifier()
        trained = pipeline.fit(self.X_train, self.y_train)
        trained.predict(self.X_test)
    def test_two_estimators_predict_proba(self):
        pipeline = StandardScaler()  >> ( PCA() & Nystroem() & LogisticRegression() )>>ConcatFeatures() >> NoOp() >> LogisticRegression()
        pipeline.fit(self.X_train, self.y_train)
        pipeline.predict_proba(self.X_test)

    @unittest.skip('#258 - PassiveAggressiveClassifier does not have predict_proba and hence predict gets called, this is valid for iris dataset.')
    def test_two_estimators_predict_proba1(self):
        pipeline = StandardScaler()  >> ( PCA() & Nystroem() & PassiveAggressiveClassifier() )>>ConcatFeatures() >> NoOp() >> PassiveAggressiveClassifier()
        pipeline.fit(self.X_train, self.y_train)
        with self.assertRaises(ValueError):
            pipeline.predict_proba(self.X_test)

    @unittest.skip('#258 - LinearSVC does not have predict_proba and hence predict gets called, this is valid for iris dataset.')
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

    def test_two_transformers(self):
        tfm1 = PCA()
        tfm2 = Nystroem()
        trainable = tfm1 >> tfm2
        digits = sklearn.datasets.load_digits()
        trained = trainable.fit(digits.data, digits.target)
        predicted = trained.transform(digits.data)

    def test_duplicate_instances(self):
        from lale.operators import make_pipeline
        tfm = PCA()
        clf = LogisticRegression(LogisticRegression.solver.lbfgs, LogisticRegression.multi_class.auto)
        with self.assertRaises(ValueError):
            trainable = make_pipeline(tfm, tfm, clf)

    def test_increase_num_rows(self):
        from test.mock_custom_operators import IncreaseRows
        increase_rows = IncreaseRows()
        trainable = increase_rows >> NoOp()
        iris = sklearn.datasets.load_iris()
        X, y = iris.data[0:10], iris.target[0:10]

        trained = trainable.fit(X, y)
        predicted = trained.transform(X, y)

    def test_remove_last1(self):
        pipeline = StandardScaler()  >> ( PCA() & Nystroem() & PassiveAggressiveClassifier() )>>ConcatFeatures() >> NoOp() >> PassiveAggressiveClassifier()
        new_pipeline = pipeline.remove_last()
        self.assertEqual(len(new_pipeline._steps), 6)
        self.assertEqual(len(pipeline._steps), 7)

    def test_remove_last2(self):
        pipeline = StandardScaler()  >> ( PCA() & Nystroem() & PassiveAggressiveClassifier() )>>ConcatFeatures() >> NoOp() >> (PassiveAggressiveClassifier() & LogisticRegression())
        with self.assertRaises(ValueError):
            pipeline.remove_last()

    def test_remove_last3(self):
        pipeline = StandardScaler()  >> ( PCA() & Nystroem() & PassiveAggressiveClassifier() )>>ConcatFeatures() >> NoOp() >> PassiveAggressiveClassifier()
        pipeline.remove_last().freeze_trainable()

    def test_remove_last4(self):
        pipeline = StandardScaler()  >> ( PCA() & Nystroem() & PassiveAggressiveClassifier() )>>ConcatFeatures() >> NoOp() >> PassiveAggressiveClassifier()
        new_pipeline = pipeline.remove_last(inplace=True)
        self.assertEqual(len(new_pipeline._steps), 6)
        self.assertEqual(len(pipeline._steps), 6)

    def test_remove_last5(self):
        pipeline = StandardScaler()  >> ( PCA() & Nystroem() & PassiveAggressiveClassifier() )>>ConcatFeatures() >> NoOp() >> PassiveAggressiveClassifier()
        pipeline.remove_last(inplace=True).freeze_trainable()
