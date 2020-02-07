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
import jsonschema
import jsonsubschema

import lale.lib.lale
from lale.lib.lale import ConcatFeatures
from lale.lib.lale import NoOp
from lale.lib.lale import IdentityWrapper
from lale.lib.sklearn import LogisticRegression
from lale.lib.sklearn import TfidfVectorizer
from lale.lib.sklearn import NMF

import sklearn.datasets

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

    def test_input_schema_fit(self):
        self.maxDiff = None
        self.assertEqual(
            LogisticRegression.input_schema_fit(),
            LogisticRegression.get_schema('input_fit'))
        self.assertEqual(
            (NMF >> LogisticRegression).input_schema_fit(),
            NMF.get_schema('input_fit'))
        self.assertEqual(
            IdentityWrapper(op=LogisticRegression).input_schema_fit(),
            LogisticRegression.get_schema('input_fit'))
        actual = (TfidfVectorizer | NMF).input_schema_fit()
        expected = {
            'anyOf': [
            {   'type': 'object',
                'required': ['X'],
                'additionalProperties': False,
                'properties': {
                    'X': {
                        'anyOf': [
                        {   'type': 'array', 'items': {'type': 'string'}},
                        {   'type': 'array',
                            'items': {
                                'type': 'array',
                                'minItems': 1, 'maxItems': 1,
                                'items': {'type': 'string'}}}]},
                    'y': {}}},
            {   'type': 'object',
                'required': ['X'],
                'additionalProperties': False,
                'properties': {
                    'X': {
                        'type': 'array',
                        'items': {
                            'type': 'array',
                            'items': {'type': 'number', 'minimum': 0.0}}},
                    'y': {}}}]}
        self.assertEqual(actual, expected)

    def test_transform_schema_NoOp(self):
        from lale.datasets.data_schemas import to_schema
        for ds in [self._irisArr, self._irisDf, self._digits, self._housing, self._creditG, self._movies, self._drugRev]:
            s_input = to_schema(ds['X'])
            s_output = NoOp.transform_schema(s_input)
            self.assertIs(s_input, s_output)

    def test_transform_schema_pipeline(self):
        from lale.datasets.data_schemas import to_schema
        pipeline = NMF >> LogisticRegression
        input_schema = to_schema(self._digits['X'])
        transformed_schema = pipeline.transform_schema(input_schema)
        transformed_expected = {
            '$schema': 'http://json-schema.org/draft-04/schema#',
            'description': 'Predicted class label per sample.',
            'type': 'array', 'items': {'type': 'number'}}
        self.maxDiff = None
        self.assertEqual(transformed_schema, transformed_expected)

    def test_transform_schema_choice(self):
        from lale.datasets.data_schemas import to_schema
        choice = NMF | LogisticRegression
        input_schema = to_schema(self._digits['X'])
        transformed_schema = choice.transform_schema(input_schema)
        transformed_expected = {
            'anyOf': [
                {   'type': 'array',
                    'items': {
                        'type': 'array', 'items': {'type': 'number'}}},
                {   'type': 'array', 'items': {'type': 'number'}}]}
        self.maxDiff = None
        self.assertEqual(transformed_schema, transformed_expected)        

    def test_transform_schema_higher_order(self):
        from lale.datasets.data_schemas import to_schema
        inner = LogisticRegression
        outer = IdentityWrapper(op=LogisticRegression)
        input_schema = to_schema(self._digits['X'])
        transformed_inner = inner.transform_schema(input_schema)
        transformed_outer = outer.transform_schema(input_schema)
        self.maxDiff = None
        self.assertEqual(transformed_inner, transformed_outer)

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
