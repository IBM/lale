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
import jsonschema
import sys

import lale.operators as Ops
import lale.lib.lale
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
from lale.lib.sklearn import NMF
from typing import List
from lale.helpers import SubschemaError

import sklearn.datasets

from lale.sklearn_compat import make_sklearn_compat
from lale.search.lale_grid_search_cv import get_grid_search_parameter_grids
from lale.search.lale_smac import get_smac_space, lale_trainable_op_from_config

class TestToJson(unittest.TestCase):
    def test_with_operator_choice(self):
        from lale.operators import make_union, make_choice, make_pipeline
        kernel_tfm_or_not =  NoOp | Nystroem
        tfm = PCA
        clf = make_choice(LogisticRegression, KNeighborsClassifier)
        clf.to_json()
        optimizable = kernel_tfm_or_not >> tfm >> clf
        optimizable.to_json()

class TestToGraphviz(unittest.TestCase):
    def test_with_operator_choice(self):
        from lale.operators import make_union, make_choice, make_pipeline
        from lale.helpers import to_graphviz
        kernel_tfm_or_not =  NoOp | Nystroem
        tfm = PCA
        clf = make_choice(LogisticRegression, KNeighborsClassifier)
        to_graphviz(clf)
        optimizable = kernel_tfm_or_not >> tfm >> clf
        to_graphviz(optimizable)

    def test_invalid_input(self):
        from sklearn.linear_model import LogisticRegression as SklearnLR
        scikit_lr = SklearnLR()
        from lale.helpers import to_graphviz
        with self.assertRaises(ValueError):
            to_graphviz(scikit_lr)

class TestPrettyPrint(unittest.TestCase):
    def round_trip(self, string1):
        globals1 = {}
        exec(string1, globals1)
        pipeline1 = globals1['pipeline']
        from lale.pretty_print import to_string
        string2 = to_string(pipeline1)
        self.maxDiff = None
        self.assertEqual(string1, string2)
        globals2 = {}
        exec(string2, globals2)
        pipeline2 = globals2['pipeline']

    def test_reducible(self):
        string1 = \
"""from lale.lib.sklearn import MinMaxScaler
from lale.lib.lale import NoOp
from lale.lib.sklearn import PCA
from lale.lib.sklearn import Nystroem
from lale.lib.lale import ConcatFeatures
from lale.lib.sklearn import KNeighborsClassifier
from lale.lib.sklearn import LogisticRegression
pca = PCA(copy=False)
logistic_regression = LogisticRegression(solver='saga', C=0.9)
pipeline = (MinMaxScaler | NoOp) >> (pca & Nystroem) >> ConcatFeatures >> (KNeighborsClassifier | logistic_regression)"""
        self.round_trip(string1)

    def test_import_as(self):
        #code to reproduce in printing
        from lale.lib.sklearn import MinMaxScaler as Scaler
        from lale.lib.lale import NoOp
        from lale.lib.sklearn import PCA
        from lale.lib.sklearn import Nystroem
        from lale.lib.lale import ConcatFeatures as Concat
        from lale.lib.sklearn import KNeighborsClassifier as KNN
        from lale.lib.sklearn import LogisticRegression as LR
        pca = PCA(copy=False)
        lr = LR(solver='saga', C=0.9)
        pipeline = (Scaler | NoOp) >> (pca & Nystroem) >> Concat >> (KNN | lr)
        #expected string
        string1 = \
"""from lale.lib.sklearn import MinMaxScaler as Scaler
from lale.lib.lale import NoOp
from lale.lib.sklearn import PCA
from lale.lib.sklearn import Nystroem
from lale.lib.lale import ConcatFeatures as Concat
from lale.lib.sklearn import KNeighborsClassifier as KNN
from lale.lib.sklearn import LogisticRegression as LR
pca = PCA(copy=False)
lr = LR(solver='saga', C=0.9)
pipeline = (Scaler | NoOp) >> (pca & Nystroem) >> Concat >> (KNN | lr)"""
        #testing harness
        from lale.pretty_print import to_string
        string2 = to_string(pipeline)
        self.maxDiff = None
        self.assertEqual(string1, string2)
        globals2 = {}
        exec(string2, globals2)
        pipeline2 = globals2['pipeline']

    def test_irreducible(self):
        string1 = \
"""from lale.lib.sklearn import PCA
from lale.lib.sklearn import Nystroem
from lale.lib.sklearn import MinMaxScaler
from lale.lib.sklearn import LogisticRegression
from lale.lib.sklearn import KNeighborsClassifier
from lale.operators import get_pipeline_of_applicable_type
step_1 = PCA | Nystroem
pipeline = get_pipeline_of_applicable_type(
    steps=[step_1, MinMaxScaler, LogisticRegression, KNeighborsClassifier],
    edges=[(step_1,LogisticRegression), (MinMaxScaler,LogisticRegression), (MinMaxScaler,KNeighborsClassifier)])"""
        self.round_trip(string1)

class TestDatasetSchemas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sklearn.datasets import load_iris
        irisArr = load_iris()
        cls._irisArr = {'X': irisArr.data, 'y': irisArr.target}
        from lale.datasets import sklearn_to_pandas
        (train_X, train_y), (test_X, test_y) = sklearn_to_pandas.load_iris_df()
        cls._irisDf = {'X': train_X, 'y': train_y}
        (train_X, train_y), (test_X, test_y) = \
            sklearn_to_pandas.load_digits_df()
        cls._digits = {'X': train_X, 'y': train_y}
        (train_X, train_y), (test_X, test_y) = \
            sklearn_to_pandas.california_housing_df()
        cls._housing = {'X': train_X, 'y': train_y}
        from lale.datasets import openml
        (train_X, train_y), (test_X, test_y) = openml.fetch(
            'credit-g', 'classification', preprocess=False)
        cls._creditG = {'X': train_X, 'y': train_y}
        from lale.datasets import load_movie_review
        train_X, train_y = load_movie_review()
        cls._movies = {'X': train_X, 'y': train_y}
        from lale.datasets.uci.uci_datasets import fetch_drugscom
        train_X, train_y, test_X, test_y = fetch_drugscom()
        cls._drugRev = {'X': train_X, 'y': train_y}

    @classmethod
    def tearDownClass(cls):
        cls._irisArr = None
        cls._irisDf = None
        cls._digits = None
        cls._housing = None
        cls._creditG = None
        cls._movies = None
        cls._drugRev = None

    def test_datasets_with_own_schemas(self):
        from lale.datasets.data_schemas import to_schema
        from lale.helpers import validate_schema
        for name in ['irisArr', 'irisDf', 'digits', 'housing', 'creditG', 'movies', 'drugRev']:
            dataset = getattr(self, f'_{name}')
            data_X, data_y = dataset['X'], dataset['y']
            schema_X, schema_y = to_schema(data_X), to_schema(data_y)
            validate_schema(data_X, schema_X, subsample_array=False)
            validate_schema(data_y, schema_y, subsample_array=False)

    def test_ndarray_to_schema(self):
        from lale.datasets.data_schemas import to_schema
        from lale.helpers import validate_schema
        all_X, all_y = self._irisArr['X'], self._irisArr['y']
        assert not hasattr(all_X, 'json_schema')
        all_X_schema = to_schema(all_X)
        validate_schema(all_X, all_X_schema, subsample_array=False)
        assert not hasattr(all_y, 'json_schema')
        all_y_schema = to_schema(all_y)
        validate_schema(all_y, all_y_schema, subsample_array=False)
        all_X_expected = {
            'type': 'array', 'minItems': 150, 'maxItems': 150,
            'items': {
                'type': 'array', 'minItems': 4, 'maxItems': 4,
                'items': {'type': 'number'}}}
        all_y_expected = {
            'type': 'array', 'minItems': 150, 'maxItems': 150,
            'items': {'type': 'integer'}}
        self.maxDiff = None
        self.assertEqual(all_X_schema, all_X_expected)
        self.assertEqual(all_y_schema, all_y_expected)

    def test_pandas_to_schema(self):
        from lale.datasets.data_schemas import to_schema
        from lale.helpers import validate_schema
        import pandas as pd
        train_X, train_y = self._irisDf['X'], self._irisDf['y']
        assert isinstance(train_X, pd.DataFrame)
        assert not hasattr(train_X, 'json_schema')
        train_X_schema = to_schema(train_X)
        validate_schema(train_X, train_X_schema, subsample_array=False)
        assert isinstance(train_y, pd.Series)
        assert not hasattr(train_y, 'json_schema')
        train_y_schema = to_schema(train_y)
        validate_schema(train_y, train_y_schema, subsample_array=False)
        train_X_expected = {
            'type': 'array', 'minItems': 120, 'maxItems': 120,
            'items': {
                'type': 'array', 'minItems': 4, 'maxItems': 4,
                'items': [
                    {'description': 'sepal length (cm)', 'type': 'number'},
                    {'description': 'sepal width (cm)', 'type': 'number'},
                    {'description': 'petal length (cm)', 'type': 'number'},
                    {'description': 'petal width (cm)', 'type': 'number'}]}}
        train_y_expected = {
            'type': 'array', 'minItems': 120, 'maxItems': 120,
            'items': {'description': 'target', 'type': 'integer'}}
        self.maxDiff = None
        self.assertEqual(train_X_schema, train_X_expected)
        self.assertEqual(train_y_schema, train_y_expected)

    def test_arff_to_schema(self):
        from lale.datasets.data_schemas import to_schema
        from lale.helpers import validate_schema
        train_X, train_y = self._creditG['X'], self._creditG['y']
        assert hasattr(train_X, 'json_schema')
        train_X_schema = to_schema(train_X)
        validate_schema(train_X, train_X_schema, subsample_array=False)
        assert hasattr(train_y, 'json_schema')
        train_y_schema = to_schema(train_y)
        validate_schema(train_y, train_y_schema, subsample_array=False)
        train_X_expected = {
            'type': 'array', 'minItems': 670, 'maxItems': 670,
            'items': {
                'type': 'array', 'minItems': 20, 'maxItems': 20,
                'items': [
                    {'description': 'checking_status', 'enum': [
                        '<0', '0<=X<200', '>=200', 'no checking']},
                    {'description': 'duration', 'type': 'number'},
                    {'description': 'credit_history', 'enum': [
                        'no credits/all paid', 'all paid',
                        'existing paid', 'delayed previously',
                        'critical/other existing credit']},
                    {'description': 'purpose', 'enum': [
                        'new car', 'used car', 'furniture/equipment',
                        'radio/tv', 'domestic appliance', 'repairs',
                        'education', 'vacation', 'retraining', 'business',
                        'other']},
                    {'description': 'credit_amount', 'type': 'number'},
                    {'description': 'savings_status', 'enum': [
                        '<100', '100<=X<500', '500<=X<1000', '>=1000',
                        'no known savings']},
                    {'description': 'employment', 'enum': [
                        'unemployed', '<1', '1<=X<4', '4<=X<7', '>=7']},
                    {'description': 'installment_commitment', 'type': 'number'},
                    {'description': 'personal_status', 'enum': [
                        'male div/sep', 'female div/dep/mar', 'male single',
                        'male mar/wid', 'female single']},
                    {'description': 'other_parties', 'enum': [
                        'none', 'co applicant', 'guarantor']},
                    {'description': 'residence_since', 'type': 'number'},
                    {'description': 'property_magnitude', 'enum': [
                        'real estate', 'life insurance', 'car',
                        'no known property']},
                    {'description': 'age', 'type': 'number'},
                    {'description': 'other_payment_plans', 'enum': [
                        'bank', 'stores', 'none']},
                    {'description': 'housing', 'enum': [
                        'rent', 'own', 'for free']},
                    {'description': 'existing_credits', 'type': 'number'},
                    {'description': 'job', 'enum': [
                        'unemp/unskilled non res', 'unskilled resident',
                        'skilled', 'high qualif/self emp/mgmt']},
                    {'description': 'num_dependents', 'type': 'number'},
                    {'description': 'own_telephone', 'enum': ['none', 'yes']},
                    {'description': 'foreign_worker', 'enum': ['yes', 'no']}]}}
        train_y_expected = {
            'type': 'array', 'minItems': 670, 'maxItems': 670,
            'items': {'description': 'class', 'enum': [0, 1]}}
        self.maxDiff = None
        self.assertEqual(train_X_schema, train_X_expected)
        self.assertEqual(train_y_schema, train_y_expected)

    def test_keep_numbers(self):
        from lale.datasets.data_schemas import to_schema
        from lale.lib.lale import KeepNumbers
        train_X, train_y = self._creditG['X'], self._creditG['y']
        trainable = KeepNumbers()
        trained = trainable.fit(train_X)
        transformed = trained.transform(train_X)
        transformed_schema = to_schema(transformed)
        transformed_expected = {
            'type': 'array', 'minItems': 670, 'maxItems': 670,
            'items': {
                'type': 'array', 'minItems': 7, 'maxItems': 7,
                'items': [
                    {'description': 'duration', 'type': 'number'},
                    {'description': 'credit_amount', 'type': 'number'},
                    {'description': 'installment_commitment', 'type': 'number'},
                    {'description': 'residence_since', 'type': 'number'},
                    {'description': 'age', 'type': 'number'},
                    {'description': 'existing_credits', 'type': 'number'},
                    {'description': 'num_dependents', 'type': 'number'}]}}
        self.maxDiff = None
        self.assertEqual(transformed_schema, transformed_expected)

    def test_keep_non_numbers(self):
        from lale.datasets.data_schemas import to_schema
        from lale.lib.lale import KeepNonNumbers
        train_X, train_y = self._creditG['X'], self._creditG['y']
        trainable = KeepNonNumbers()
        trained = trainable.fit(train_X)
        transformed = trained.transform(train_X)
        transformed_schema = to_schema(transformed)
        transformed_expected = {
            'type': 'array', 'minItems': 670, 'maxItems': 670,
            'items': {
                'type': 'array', 'minItems': 13, 'maxItems': 13,
                'items': [
                    {'description': 'checking_status', 'enum': [
                        '<0', '0<=X<200', '>=200', 'no checking']},
                    {'description': 'credit_history', 'enum': [
                        'no credits/all paid', 'all paid',
                        'existing paid', 'delayed previously',
                        'critical/other existing credit']},
                    {'description': 'purpose', 'enum': [
                        'new car', 'used car', 'furniture/equipment',
                        'radio/tv', 'domestic appliance', 'repairs',
                        'education', 'vacation', 'retraining', 'business',
                        'other']},
                    {'description': 'savings_status', 'enum': [
                        '<100', '100<=X<500', '500<=X<1000', '>=1000',
                        'no known savings']},
                    {'description': 'employment', 'enum': [
                        'unemployed', '<1', '1<=X<4', '4<=X<7', '>=7']},
                    {'description': 'personal_status', 'enum': [
                        'male div/sep', 'female div/dep/mar', 'male single',
                        'male mar/wid', 'female single']},
                    {'description': 'other_parties', 'enum': [
                        'none', 'co applicant', 'guarantor']},
                    {'description': 'property_magnitude', 'enum': [
                        'real estate', 'life insurance', 'car',
                        'no known property']},
                    {'description': 'other_payment_plans', 'enum': [
                        'bank', 'stores', 'none']},
                    {'description': 'housing', 'enum': [
                        'rent', 'own', 'for free']},
                    {'description': 'job', 'enum': [
                        'unemp/unskilled non res', 'unskilled resident',
                        'skilled', 'high qualif/self emp/mgmt']},
                    {'description': 'own_telephone', 'enum': ['none', 'yes']},
                    {'description': 'foreign_worker', 'enum': ['yes', 'no']}]}}
        self.maxDiff = None
        self.assertEqual(transformed_schema, transformed_expected)

    def test_transform_schema_NoOp(self):
        from lale.datasets.data_schemas import to_schema
        for ds in [self._irisArr, self._irisDf, self._digits, self._housing, self._creditG, self._movies, self._drugRev]:
            s_input = to_schema(ds['X'])
            s_output = NoOp.transform_schema(s_input)
            self.assertIs(s_input, s_output)

    def test_transform_schema_Concat_irisArr(self):
        from lale.datasets.data_schemas import to_schema
        data_X, data_y = self._irisArr['X'], self._irisArr['y']
        s_in_X, s_in_y = to_schema(data_X), to_schema(data_y)
        def check(s_actual, n_expected, s_expected):
            assert s_actual['items']['minItems'] == n_expected, str(s_actual)
            assert s_actual['items']['maxItems'] == n_expected, str(s_actual)
            assert s_actual['items']['items'] == s_expected, str(s_actual)
        s_out_X = ConcatFeatures.transform_schema({'items': [s_in_X]})
        check(s_out_X, 4, {'type': 'number'})
        s_out_y = ConcatFeatures.transform_schema({'items': [s_in_y]})
        check(s_out_y, 1, {'type': 'integer'})
        s_out_XX = ConcatFeatures.transform_schema({'items': [s_in_X, s_in_X]})
        check(s_out_XX, 8, {'type': 'number'})
        s_out_yy = ConcatFeatures.transform_schema({'items': [s_in_y, s_in_y]})
        check(s_out_yy, 2, {'type': 'integer'})
        s_out_Xy = ConcatFeatures.transform_schema({'items': [s_in_X, s_in_y]})
        check(s_out_Xy, 5, {'type': 'number'})
        s_out_XXX = ConcatFeatures.transform_schema({
            'items': [s_in_X, s_in_X, s_in_X]})
        check(s_out_XXX, 12, {'type': 'number'})

    def test_transform_schema_Concat_irisDf(self):
        from lale.datasets.data_schemas import to_schema
        data_X, data_y = self._irisDf['X'], self._irisDf['y']
        s_in_X, s_in_y = to_schema(data_X), to_schema(data_y)
        def check(s_actual, n_expected, s_expected):
            assert s_actual['items']['minItems'] == n_expected, str(s_actual)
            assert s_actual['items']['maxItems'] == n_expected, str(s_actual)
            assert s_actual['items']['items'] == s_expected, str(s_actual)
        s_out_X = ConcatFeatures.transform_schema({'items': [s_in_X]})
        check(s_out_X, 4, {'type': 'number'})
        s_out_y = ConcatFeatures.transform_schema({'items': [s_in_y]})
        check(s_out_y, 1, {'description': 'target', 'type': 'integer'})
        s_out_XX = ConcatFeatures.transform_schema({'items': [s_in_X, s_in_X]})
        check(s_out_XX, 8, {'type': 'number'})
        s_out_yy = ConcatFeatures.transform_schema({'items': [s_in_y, s_in_y]})
        check(s_out_yy, 2, {'type': 'integer'})
        s_out_Xy = ConcatFeatures.transform_schema({'items': [s_in_X, s_in_y]})
        check(s_out_Xy, 5, {'type': 'number'})
        s_out_XXX = ConcatFeatures.transform_schema({
            'items': [s_in_X, s_in_X, s_in_X]})
        check(s_out_XXX, 12, {'type': 'number'})

    def test_lr_with_all_datasets(self):
        should_succeed = ['irisArr', 'irisDf', 'digits', 'housing']
        should_fail = ['creditG', 'movies', 'drugRev']
        for name in should_succeed:
            dataset = getattr(self, f'_{name}')
            LogisticRegression.validate_schema(**dataset)
        for name in should_fail:
            dataset = getattr(self, f'_{name}')
            with self.assertRaises(ValueError):
                LogisticRegression.validate_schema(**dataset)

    def test_project_with_all_datasets(self):
        import lale.lib.lale
        should_succeed = ['irisArr', 'irisDf', 'digits', 'housing', 'creditG', 'drugRev']
        should_fail = ['movies']
        for name in should_succeed:
            dataset = getattr(self, f'_{name}')
            lale.lib.lale.Project.validate_schema(**dataset)
        for name in should_fail:
            dataset = getattr(self, f'_{name}')
            with self.assertRaises(ValueError):
                lale.lib.lale.Project.validate_schema(**dataset)

    def test_nmf_with_all_datasets(self):
        should_succeed = ['digits']
        should_fail = ['irisArr', 'irisDf', 'housing', 'creditG', 'movies', 'drugRev']
        for name in should_succeed:
            dataset = getattr(self, f'_{name}')
            NMF.validate_schema(**dataset)
        for name in should_fail:
            dataset = getattr(self, f'_{name}')
            with self.assertRaises(ValueError):
                NMF.validate_schema(**dataset)

    def test_tfidf_with_all_datasets(self):
        should_succeed = ['movies']
        should_fail = ['irisArr', 'irisDf', 'digits', 'housing', 'creditG', 'drugRev']
        for name in should_succeed:
            dataset = getattr(self, f'_{name}')
            TfidfVectorizer.validate_schema(**dataset)
        for name in should_fail:
            dataset = getattr(self, f'_{name}')
            with self.assertRaises(ValueError):
                TfidfVectorizer.validate_schema(**dataset)

class TestErrorMessages(unittest.TestCase):
    def test_wrong_cont(self):
        with self.assertRaises(jsonschema.ValidationError) as cm:
            LogisticRegression(C=-1)
        summary = cm.exception.message.split('\n')[0]
        self.assertEqual(summary, "Invalid configuration for LogisticRegression(C=-1) due to invalid value C=-1.")

    def test_wrong_cat(self):
        with self.assertRaises(jsonschema.ValidationError) as cm:
            LogisticRegression(solver='adam')
        summary = cm.exception.message.split('\n')[0]
        self.assertEqual(summary, "Invalid configuration for LogisticRegression(solver='adam') due to invalid value solver=adam.")

    def test_unknown_arg(self):
        with self.assertRaises(jsonschema.ValidationError) as cm:
            LogisticRegression(activation='relu')
        summary = cm.exception.message.split('\n')[0]
        self.assertEqual(summary, "Invalid configuration for LogisticRegression(activation='relu') due to argument 'activation' was unexpected.")

    def test_constraint(self):
        with self.assertRaises(jsonschema.ValidationError) as cm:
            LogisticRegression(solver='sag', penalty='l1')
        summary = cm.exception.message.split('\n')[0]
        self.assertEqual(summary, "Invalid configuration for LogisticRegression(solver='sag', penalty='l1') due to constraint the newton-cg, sag, and lbfgs solvers support only l2 penalties.")

class TestFreeze(unittest.TestCase):
    def test_individual_op_freeze_trainable(self):
        liquid = LogisticRegression(C=0.1, solver='liblinear')
        self.assertIn('penalty', liquid.free_hyperparams())
        self.assertFalse(liquid.is_frozen_trainable())
        liquid_grid = get_grid_search_parameter_grids(liquid)
        self.assertTrue(len(liquid_grid) > 1, f'grid size {len(liquid_grid)}')
        frozen = liquid.freeze_trainable()
        self.assertEqual(len(frozen.free_hyperparams()), 0)
        self.assertTrue(frozen.is_frozen_trainable())
        frozen_grid = get_grid_search_parameter_grids(frozen)
        self.assertEqual(len(frozen_grid), 1)

    def test_pipeline_freeze_trainable(self):
        liquid = PCA() >> LogisticRegression()
        self.assertFalse(liquid.is_frozen_trainable())
        liquid_grid = get_grid_search_parameter_grids(liquid)
        self.assertTrue(len(liquid_grid) > 1, f'grid size {len(liquid_grid)}')
        frozen = liquid.freeze_trainable()
        self.assertTrue(frozen.is_frozen_trainable())
        frozen_grid = get_grid_search_parameter_grids(frozen)
        self.assertEqual(len(frozen_grid), 1)

    def test_individual_op_freeze_trained(self):
        trainable = KNeighborsClassifier(n_neighbors=1)
        X = [[0], [1], [2]]
        y_old = [0, 0, 1]
        y_new = [1, 0, 0]
        liquid_old = trainable.fit(X, y_old)
        self.assertEqual(list(liquid_old.predict(X)), y_old)
        liquid_new = liquid_old.fit(X, y_new)
        self.assertEqual(list(liquid_new.predict(X)), y_new)
        frozen_old = trainable.fit(X, y_old).freeze_trained()
        self.assertFalse(liquid_old.is_frozen_trained())
        self.assertTrue(frozen_old.is_frozen_trained())
        self.assertEqual(list(frozen_old.predict(X)), y_old)
        frozen_new = frozen_old.fit(X, y_new)
        self.assertEqual(list(frozen_new.predict(X)), y_old)

    def test_pipeline_freeze_trained(self):
        trainable = MinMaxScaler() >> LogisticRegression()
        liquid = trainable.fit([[0], [1], [2]], [0, 0, 1])
        frozen = liquid.freeze_trained()
        self.assertFalse(liquid.is_frozen_trained())
        self.assertTrue(frozen.is_frozen_trained())

class TestToAndFromJSON(unittest.TestCase):
    def test_trainable_individual_op(self):
        self.maxDiff = None
        from lale.json_operator import to_json, from_json
        from lale.lib.sklearn import LogisticRegression as LR
        operator = LR(LR.solver.sag, C=0.1)
        json_expected = {
            'class': 'lale.lib.sklearn.logistic_regression.LogisticRegressionImpl',
            'state': 'trainable',
            'operator': 'LogisticRegression', 'label': 'LR', 'id': 'lr',
            'documentation_url': 'http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html',
            'hyperparams': {'C': 0.1, 'solver': 'sag'},
            'is_frozen_trainable': False}
        json = to_json(operator)
        self.assertEqual(json, json_expected)
        operator_2 = from_json(json)
        json_2 = to_json(operator_2)
        self.assertEqual(json_2, json_expected)

    def test_operator_choice(self):
        self.maxDiff = None
        from lale.json_operator import to_json, from_json
        from lale.lib.sklearn import MinMaxScaler as Scl
        operator = PCA | Scl
        json_expected = {
            'class': 'lale.operators.OperatorChoice',
            'operator': 'OperatorChoice', 'id': 'choice',
            'state': 'planned',
            'steps': [
            {   'class': 'lale.lib.sklearn.pca.PCAImpl',
                'state': 'planned',
                'operator': 'PCA', 'label': 'PCA', 'id': 'pca',
                'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html'},
            {   'class': 'lale.lib.sklearn.min_max_scaler.MinMaxScalerImpl',
                'state': 'planned',
                'operator': 'MinMaxScaler', 'label': 'Scl', 'id': 'scl',
                'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html'}]}
        json = to_json(operator)
        self.assertEqual(json, json_expected)
        operator_2 = from_json(json)
        json_2 = to_json(operator_2)
        self.assertEqual(json_2, json_expected)

    def test_pipeline(self):
        self.maxDiff = None
        from lale.json_operator import to_json, from_json
        from lale.lib.sklearn import LogisticRegression as LR
        operator = (PCA & NoOp) >> ConcatFeatures >> LR
        json_expected = {
          'class': 'lale.operators.PlannedPipeline',
          'state': 'planned',
          'id': 'pipeline',
          'edges': [[0, 2], [1, 2], [2, 3]],
          'steps': [
          { 'class': 'lale.lib.sklearn.pca.PCAImpl',
            'state': 'planned',
            'operator': 'PCA', 'label': 'PCA', 'id': 'pca',
            'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html'},
          { 'class': 'lale.lib.lale.no_op.NoOpImpl',
            'state': 'trained',
            'operator': 'NoOp', 'label': 'NoOp', 'id': 'no_op',
            'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.no_op.html',
            'hyperparams': None,
            'coefs': None,
            'is_frozen_trainable': True, 'is_frozen_trained': True},
          { 'class': 'lale.lib.lale.concat_features.ConcatFeaturesImpl',
            'state': 'trained',
            'operator': 'ConcatFeatures', 'label': 'ConcatFeatures', 'id': 'concat_features',
            'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.concat_features.html',
            'hyperparams': None,
            'coefs': None,
            'is_frozen_trainable': True, 'is_frozen_trained': True},
          { 'class': 'lale.lib.sklearn.logistic_regression.LogisticRegressionImpl',
            'state': 'planned',
            'operator': 'LogisticRegression', 'label': 'LR', 'id': 'lr',
            'documentation_url': 'http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html'}]}
        json = to_json(operator)
        self.assertEqual(json, json_expected)
        operator_2 = from_json(json)
        json_2 = to_json(operator_2)
        self.assertEqual(json, json_2)

    def test_nested(self):
        self.maxDiff = None
        from lale.json_operator import to_json, from_json
        from lale.lib.sklearn import LogisticRegression as LR
        operator = PCA >> (LR(C=0.09) | NoOp >> LR(C=0.19))
        json_expected = {
          'class': 'lale.operators.PlannedPipeline',
          'state': 'planned',
          'id': 'pipeline_0',
          'edges': [[0, 1]],
          'steps': [
          { 'class': 'lale.lib.sklearn.pca.PCAImpl',
            'state': 'planned',
            'operator': 'PCA', 'label': 'PCA', 'id': 'pca',
            'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html'},
          { 'class': 'lale.operators.OperatorChoice',
            'state': 'planned',
            'operator': 'OperatorChoice', 'id': 'choice',
            'steps': [
            { 'class': 'lale.lib.sklearn.logistic_regression.LogisticRegressionImpl',
              'state': 'trainable',
              'operator': 'LogisticRegression', 'label': 'LR', 'id': 'lr_0',
              'documentation_url': 'http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html',
              'hyperparams': {'C': 0.09},
              'is_frozen_trainable': False},
            { 'class': 'lale.operators.TrainablePipeline',
              'state': 'trainable', 'id': 'pipeline_1',
              'edges': [[0, 1]],
              'steps': [
              { 'class': 'lale.lib.lale.no_op.NoOpImpl',
                'state': 'trained',
                'operator': 'NoOp', 'label': 'NoOp', 'id': 'no_op',
                'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.no_op.html',
                'hyperparams': None,
                'coefs': None,
                'is_frozen_trainable': True, 'is_frozen_trained': True},
              { 'class': 'lale.lib.sklearn.logistic_regression.LogisticRegressionImpl',
                'state': 'trainable',
                'operator': 'LogisticRegression', 'label': 'LR', 'id': 'lr_1',
                'documentation_url': 'http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html',
                'hyperparams': {'C': 0.19},
                'is_frozen_trainable': False}]}]}]}
        json = to_json(operator)
        self.assertEqual(json, json_expected)
        operator_2 = from_json(json)
        json_2 = to_json(operator_2)
        self.assertEqual(json, json_2)

if __name__ == '__main__':
    unittest.main()
