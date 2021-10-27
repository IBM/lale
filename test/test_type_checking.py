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
from test import EnableSchemaValidation

import jsonschema

import lale.lib.lale
from lale.lib.lale import ConcatFeatures, IdentityWrapper, NoOp
from lale.lib.sklearn import NMF, LogisticRegression, TfidfVectorizer
from lale.settings import (
    disable_data_schema_validation,
    disable_hyperparams_schema_validation,
    set_disable_data_schema_validation,
    set_disable_hyperparams_schema_validation,
)


class TestDatasetSchemas(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sklearn.datasets import load_iris

        with EnableSchemaValidation():
            irisArr = load_iris()
            cls._irisArr = {"X": irisArr.data, "y": irisArr.target}
            from lale.datasets import sklearn_to_pandas

            (train_X, train_y), (test_X, test_y) = sklearn_to_pandas.load_iris_df()
            cls._irisDf = {"X": train_X, "y": train_y}
            (train_X, train_y), (test_X, test_y) = sklearn_to_pandas.digits_df()
            cls._digits = {"X": train_X, "y": train_y}
            (
                (train_X, train_y),
                (test_X, test_y),
            ) = sklearn_to_pandas.california_housing_df()
            cls._housing = {"X": train_X, "y": train_y}
            from lale.datasets import openml

            (train_X, train_y), (test_X, test_y) = openml.fetch(
                "credit-g", "classification", preprocess=False
            )
            cls._creditG = {"X": train_X, "y": train_y}
            from lale.datasets import load_movie_review

            train_X, train_y = load_movie_review()
            cls._movies = {"X": train_X, "y": train_y}
            from lale.datasets.uci.uci_datasets import fetch_drugscom

            train_X, train_y, test_X, test_y = fetch_drugscom()
            cls._drugRev = {"X": train_X, "y": train_y}

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
        from lale.type_checking import validate_schema_directly

        for name in [
            "irisArr",
            "irisDf",
            "digits",
            "housing",
            "creditG",
            "movies",
            "drugRev",
        ]:
            dataset = getattr(self, f"_{name}")
            data_X, data_y = dataset["X"], dataset["y"]
            schema_X, schema_y = to_schema(data_X), to_schema(data_y)
            validate_schema_directly(data_X, schema_X, subsample_array=False)
            validate_schema_directly(data_y, schema_y, subsample_array=False)

    def test_ndarray_to_schema(self):
        from lale.datasets.data_schemas import to_schema
        from lale.type_checking import validate_schema_directly

        irisArr = self._irisArr
        assert irisArr is not None
        all_X, all_y = irisArr["X"], irisArr["y"]
        assert not hasattr(all_X, "json_schema")
        all_X_schema = to_schema(all_X)
        validate_schema_directly(all_X, all_X_schema, subsample_array=False)
        assert not hasattr(all_y, "json_schema")
        all_y_schema = to_schema(all_y)
        validate_schema_directly(all_y, all_y_schema, subsample_array=False)
        all_X_expected = {
            "type": "array",
            "minItems": 150,
            "maxItems": 150,
            "items": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {"type": "number"},
            },
        }
        all_y_expected = {
            "type": "array",
            "minItems": 150,
            "maxItems": 150,
            "items": {"type": "integer"},
        }
        self.maxDiff = None
        self.assertEqual(all_X_schema, all_X_expected)
        self.assertEqual(all_y_schema, all_y_expected)

    def test_pandas_to_schema(self):
        import pandas as pd

        from lale.datasets.data_schemas import to_schema
        from lale.type_checking import validate_schema_directly

        irisDf = self._irisDf
        assert irisDf is not None

        train_X, train_y = irisDf["X"], irisDf["y"]
        assert isinstance(train_X, pd.DataFrame)
        assert not hasattr(train_X, "json_schema")
        train_X_schema = to_schema(train_X)
        validate_schema_directly(train_X, train_X_schema, subsample_array=False)
        assert isinstance(train_y, pd.Series)
        assert not hasattr(train_y, "json_schema")
        train_y_schema = to_schema(train_y)
        validate_schema_directly(train_y, train_y_schema, subsample_array=False)
        train_X_expected = {
            "type": "array",
            "minItems": 120,
            "maxItems": 120,
            "items": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": [
                    {"description": "sepal length (cm)", "type": "number"},
                    {"description": "sepal width (cm)", "type": "number"},
                    {"description": "petal length (cm)", "type": "number"},
                    {"description": "petal width (cm)", "type": "number"},
                ],
            },
        }
        train_y_expected = {
            "type": "array",
            "minItems": 120,
            "maxItems": 120,
            "items": {"description": "target", "type": "integer"},
        }
        self.maxDiff = None
        self.assertEqual(train_X_schema, train_X_expected)
        self.assertEqual(train_y_schema, train_y_expected)

    def test_arff_to_schema(self):
        from lale.datasets.data_schemas import to_schema
        from lale.type_checking import validate_schema_directly

        creditG = self._creditG
        assert creditG is not None
        train_X, train_y = creditG["X"], creditG["y"]
        assert hasattr(train_X, "json_schema")
        train_X_schema = to_schema(train_X)
        validate_schema_directly(train_X, train_X_schema, subsample_array=False)
        assert hasattr(train_y, "json_schema")
        train_y_schema = to_schema(train_y)
        validate_schema_directly(train_y, train_y_schema, subsample_array=False)
        train_X_expected = {
            "type": "array",
            "minItems": 670,
            "maxItems": 670,
            "items": {
                "type": "array",
                "minItems": 20,
                "maxItems": 20,
                "items": [
                    {
                        "description": "checking_status",
                        "enum": ["<0", "0<=X<200", ">=200", "no checking"],
                    },
                    {"description": "duration", "type": "number"},
                    {
                        "description": "credit_history",
                        "enum": [
                            "no credits/all paid",
                            "all paid",
                            "existing paid",
                            "delayed previously",
                            "critical/other existing credit",
                        ],
                    },
                    {
                        "description": "purpose",
                        "enum": [
                            "new car",
                            "used car",
                            "furniture/equipment",
                            "radio/tv",
                            "domestic appliance",
                            "repairs",
                            "education",
                            "vacation",
                            "retraining",
                            "business",
                            "other",
                        ],
                    },
                    {"description": "credit_amount", "type": "number"},
                    {
                        "description": "savings_status",
                        "enum": [
                            "<100",
                            "100<=X<500",
                            "500<=X<1000",
                            ">=1000",
                            "no known savings",
                        ],
                    },
                    {
                        "description": "employment",
                        "enum": ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
                    },
                    {"description": "installment_commitment", "type": "number"},
                    {
                        "description": "personal_status",
                        "enum": [
                            "male div/sep",
                            "female div/dep/mar",
                            "male single",
                            "male mar/wid",
                            "female single",
                        ],
                    },
                    {
                        "description": "other_parties",
                        "enum": ["none", "co applicant", "guarantor"],
                    },
                    {"description": "residence_since", "type": "number"},
                    {
                        "description": "property_magnitude",
                        "enum": [
                            "real estate",
                            "life insurance",
                            "car",
                            "no known property",
                        ],
                    },
                    {"description": "age", "type": "number"},
                    {
                        "description": "other_payment_plans",
                        "enum": ["bank", "stores", "none"],
                    },
                    {"description": "housing", "enum": ["rent", "own", "for free"]},
                    {"description": "existing_credits", "type": "number"},
                    {
                        "description": "job",
                        "enum": [
                            "unemp/unskilled non res",
                            "unskilled resident",
                            "skilled",
                            "high qualif/self emp/mgmt",
                        ],
                    },
                    {"description": "num_dependents", "type": "number"},
                    {"description": "own_telephone", "enum": ["none", "yes"]},
                    {"description": "foreign_worker", "enum": ["yes", "no"]},
                ],
            },
        }
        train_y_expected = {
            "type": "array",
            "minItems": 670,
            "maxItems": 670,
            "items": {"description": "class", "enum": ["good", "bad"]},
        }
        self.maxDiff = None
        self.assertEqual(train_X_schema, train_X_expected)
        self.assertEqual(train_y_schema, train_y_expected)

    def test_keep_numbers(self):
        from lale.datasets.data_schemas import to_schema
        from lale.lib.lale import Project

        creditG = self._creditG
        assert creditG is not None
        train_X = creditG["X"]
        trainable = Project(columns={"type": "number"})
        trained = trainable.fit(train_X)
        transformed = trained.transform(train_X)
        transformed_schema = to_schema(transformed)
        transformed_expected = {
            "type": "array",
            "minItems": 670,
            "maxItems": 670,
            "items": {
                "type": "array",
                "minItems": 7,
                "maxItems": 7,
                "items": [
                    {"description": "duration", "type": "number"},
                    {"description": "credit_amount", "type": "number"},
                    {"description": "installment_commitment", "type": "number"},
                    {"description": "residence_since", "type": "number"},
                    {"description": "age", "type": "number"},
                    {"description": "existing_credits", "type": "number"},
                    {"description": "num_dependents", "type": "number"},
                ],
            },
        }
        self.maxDiff = None
        self.assertEqual(transformed_schema, transformed_expected)

    def test_keep_non_numbers(self):
        with EnableSchemaValidation():
            from lale.datasets.data_schemas import to_schema
            from lale.lib.lale import Project

            creditG = self._creditG
            assert creditG is not None
            train_X = creditG["X"]
            trainable = Project(columns={"not": {"type": "number"}})
            trained = trainable.fit(train_X)
            transformed = trained.transform(train_X)
            transformed_schema = to_schema(transformed)
            transformed_expected = {
                "type": "array",
                "minItems": 670,
                "maxItems": 670,
                "items": {
                    "type": "array",
                    "minItems": 13,
                    "maxItems": 13,
                    "items": [
                        {
                            "description": "checking_status",
                            "enum": ["<0", "0<=X<200", ">=200", "no checking"],
                        },
                        {
                            "description": "credit_history",
                            "enum": [
                                "no credits/all paid",
                                "all paid",
                                "existing paid",
                                "delayed previously",
                                "critical/other existing credit",
                            ],
                        },
                        {
                            "description": "purpose",
                            "enum": [
                                "new car",
                                "used car",
                                "furniture/equipment",
                                "radio/tv",
                                "domestic appliance",
                                "repairs",
                                "education",
                                "vacation",
                                "retraining",
                                "business",
                                "other",
                            ],
                        },
                        {
                            "description": "savings_status",
                            "enum": [
                                "<100",
                                "100<=X<500",
                                "500<=X<1000",
                                ">=1000",
                                "no known savings",
                            ],
                        },
                        {
                            "description": "employment",
                            "enum": ["unemployed", "<1", "1<=X<4", "4<=X<7", ">=7"],
                        },
                        {
                            "description": "personal_status",
                            "enum": [
                                "male div/sep",
                                "female div/dep/mar",
                                "male single",
                                "male mar/wid",
                                "female single",
                            ],
                        },
                        {
                            "description": "other_parties",
                            "enum": ["none", "co applicant", "guarantor"],
                        },
                        {
                            "description": "property_magnitude",
                            "enum": [
                                "real estate",
                                "life insurance",
                                "car",
                                "no known property",
                            ],
                        },
                        {
                            "description": "other_payment_plans",
                            "enum": ["bank", "stores", "none"],
                        },
                        {"description": "housing", "enum": ["rent", "own", "for free"]},
                        {
                            "description": "job",
                            "enum": [
                                "unemp/unskilled non res",
                                "unskilled resident",
                                "skilled",
                                "high qualif/self emp/mgmt",
                            ],
                        },
                        {"description": "own_telephone", "enum": ["none", "yes"]},
                        {"description": "foreign_worker", "enum": ["yes", "no"]},
                    ],
                },
            }
            self.maxDiff = None
            self.assertEqual(transformed_schema, transformed_expected)

    def test_input_schema_fit(self):
        self.maxDiff = None
        self.assertEqual(
            LogisticRegression.input_schema_fit(),
            LogisticRegression.get_schema("input_fit"),
        )
        self.assertEqual(
            (NMF >> LogisticRegression).input_schema_fit(), NMF.get_schema("input_fit")
        )
        self.assertEqual(
            IdentityWrapper(op=LogisticRegression).input_schema_fit(),
            LogisticRegression.get_schema("input_fit"),
        )
        actual = (TfidfVectorizer | NMF).input_schema_fit()
        expected = {
            "anyOf": [
                {
                    "type": "object",
                    "required": ["X"],
                    "additionalProperties": False,
                    "properties": {
                        "X": {
                            "anyOf": [
                                {"type": "array", "items": {"type": "string"}},
                                {
                                    "type": "array",
                                    "items": {
                                        "type": "array",
                                        "minItems": 1,
                                        "maxItems": 1,
                                        "items": {"type": "string"},
                                    },
                                },
                            ]
                        },
                        "y": {},
                    },
                },
                {
                    "type": "object",
                    "required": ["X"],
                    "additionalProperties": False,
                    "properties": {
                        "X": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "number", "minimum": 0.0},
                            },
                        },
                        "y": {},
                    },
                },
            ]
        }
        self.assertEqual(actual, expected)

    def test_transform_schema_NoOp(self):
        with EnableSchemaValidation():
            from lale.datasets.data_schemas import to_schema

            for ds in [
                self._irisArr,
                self._irisDf,
                self._digits,
                self._housing,
                self._creditG,
                self._movies,
                self._drugRev,
            ]:
                assert ds is not None
                s_input = to_schema(ds["X"])
                s_output = NoOp.transform_schema(s_input)
                self.assertIs(s_input, s_output)

    def test_transform_schema_pipeline(self):
        with EnableSchemaValidation():
            from lale.datasets.data_schemas import to_schema

            pipeline = NMF >> LogisticRegression
            digits = self._digits
            assert digits is not None
            input_schema = to_schema(digits["X"])
            transformed_schema = pipeline.transform_schema(input_schema)
            transformed_expected = {
                "description": "Probability of the sample for each class in the model.",
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
            }
            self.maxDiff = None
            self.assertEqual(transformed_schema, transformed_expected)

    def test_transform_schema_choice(self):
        with EnableSchemaValidation():
            from lale.datasets.data_schemas import to_schema

            choice = NMF | LogisticRegression
            digits = self._digits
            assert digits is not None
            input_schema = to_schema(digits["X"])
            transformed_schema = choice.transform_schema(input_schema)
            transformed_expected = {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
            }
            self.maxDiff = None
            self.assertEqual(transformed_schema, transformed_expected)

    def test_transform_schema_higher_order(self):
        with EnableSchemaValidation():
            from lale.datasets.data_schemas import to_schema

            inner = LogisticRegression
            outer = IdentityWrapper(op=LogisticRegression)
            digits = self._digits
            assert digits is not None
            input_schema = to_schema(digits["X"])
            transformed_inner = inner.transform_schema(input_schema)
            transformed_outer = outer.transform_schema(input_schema)
            self.maxDiff = None
            self.assertEqual(transformed_inner, transformed_outer)

    def test_transform_schema_Concat_irisArr(self):
        with EnableSchemaValidation():
            from lale.datasets.data_schemas import to_schema

            irisArr = self._irisArr
            assert irisArr is not None
            data_X, data_y = irisArr["X"], irisArr["y"]
            s_in_X, s_in_y = to_schema(data_X), to_schema(data_y)

            def check(s_actual, n_expected, s_expected):
                assert s_actual["items"]["minItems"] == n_expected, str(s_actual)
                assert s_actual["items"]["maxItems"] == n_expected, str(s_actual)
                assert s_actual["items"]["items"] == s_expected, str(s_actual)

            s_out_X = ConcatFeatures.transform_schema({"items": [s_in_X]})
            check(s_out_X, 4, {"type": "number"})
            s_out_y = ConcatFeatures.transform_schema({"items": [s_in_y]})
            check(s_out_y, 1, {"type": "integer"})
            s_out_XX = ConcatFeatures.transform_schema({"items": [s_in_X, s_in_X]})
            check(s_out_XX, 8, {"type": "number"})
            s_out_yy = ConcatFeatures.transform_schema({"items": [s_in_y, s_in_y]})
            check(s_out_yy, 2, {"type": "integer"})
            s_out_Xy = ConcatFeatures.transform_schema({"items": [s_in_X, s_in_y]})
            check(s_out_Xy, 5, {"type": "number"})
            s_out_XXX = ConcatFeatures.transform_schema(
                {"items": [s_in_X, s_in_X, s_in_X]}
            )
            check(s_out_XXX, 12, {"type": "number"})

    def test_transform_schema_Concat_irisDf(self):
        with EnableSchemaValidation():
            from lale.datasets.data_schemas import to_schema

            irisDf = self._irisDf
            assert irisDf is not None
            data_X, data_y = irisDf["X"], irisDf["y"]
            s_in_X, s_in_y = to_schema(data_X), to_schema(data_y)

            def check(s_actual, n_expected, s_expected):
                assert s_actual["items"]["minItems"] == n_expected, str(s_actual)
                assert s_actual["items"]["maxItems"] == n_expected, str(s_actual)
                assert s_actual["items"]["items"] == s_expected, str(s_actual)

            s_out_X = ConcatFeatures.transform_schema({"items": [s_in_X]})
            check(s_out_X, 4, {"type": "number"})
            s_out_y = ConcatFeatures.transform_schema({"items": [s_in_y]})
            check(s_out_y, 1, {"description": "target", "type": "integer"})
            s_out_XX = ConcatFeatures.transform_schema({"items": [s_in_X, s_in_X]})
            check(s_out_XX, 8, {"type": "number"})
            s_out_yy = ConcatFeatures.transform_schema({"items": [s_in_y, s_in_y]})
            check(s_out_yy, 2, {"type": "integer"})
            s_out_Xy = ConcatFeatures.transform_schema({"items": [s_in_X, s_in_y]})
            check(s_out_Xy, 5, {"type": "number"})
            s_out_XXX = ConcatFeatures.transform_schema(
                {"items": [s_in_X, s_in_X, s_in_X]}
            )
            check(s_out_XXX, 12, {"type": "number"})

    def test_lr_with_all_datasets(self):
        with EnableSchemaValidation():
            should_succeed = ["irisArr", "irisDf", "digits", "housing"]
            should_fail = ["creditG", "movies", "drugRev"]
            for name in should_succeed:
                dataset = getattr(self, f"_{name}")
                LogisticRegression.validate_schema(**dataset)
            for name in should_fail:
                dataset = getattr(self, f"_{name}")
                with self.assertRaises(ValueError):
                    LogisticRegression.validate_schema(**dataset)

    def test_project_with_all_datasets(self):
        with EnableSchemaValidation():
            should_succeed = [
                "irisArr",
                "irisDf",
                "digits",
                "housing",
                "creditG",
                "drugRev",
            ]
            should_fail = ["movies"]
            for name in should_succeed:
                dataset = getattr(self, f"_{name}")
                lale.lib.lale.Project.validate_schema(**dataset)
            for name in should_fail:
                dataset = getattr(self, f"_{name}")
                with self.assertRaises(ValueError):
                    lale.lib.lale.Project.validate_schema(**dataset)

    def test_nmf_with_all_datasets(self):
        with EnableSchemaValidation():
            should_succeed = ["digits"]
            should_fail = [
                "irisArr",
                "irisDf",
                "housing",
                "creditG",
                "movies",
                "drugRev",
            ]
            for name in should_succeed:
                dataset = getattr(self, f"_{name}")
                NMF.validate_schema(**dataset)
            for name in should_fail:
                dataset = getattr(self, f"_{name}")
                with self.assertRaises(ValueError):
                    NMF.validate_schema(**dataset)

    def test_tfidf_with_all_datasets(self):
        with EnableSchemaValidation():
            should_succeed = ["movies"]
            should_fail = [
                "irisArr",
                "irisDf",
                "digits",
                "housing",
                "creditG",
                "drugRev",
            ]
            for name in should_succeed:
                dataset = getattr(self, f"_{name}")
                TfidfVectorizer.validate_schema(**dataset)
            for name in should_fail:
                dataset = getattr(self, f"_{name}")
                with self.assertRaises(ValueError):
                    TfidfVectorizer.validate_schema(**dataset)

    def test_decision_function_binary(self):
        from lale.lib.lale import Project

        creditG = self._creditG
        assert creditG is not None
        train_X, train_y = creditG["X"], creditG["y"]
        trainable = Project(columns={"type": "number"}) >> LogisticRegression()
        trained = trainable.fit(train_X, train_y)
        _ = trained.decision_function(train_X)


class TestErrorMessages(unittest.TestCase):
    def test_wrong_cont(self):
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError) as cm:
                LogisticRegression(C=-1)
            summary = cm.exception.message.split("\n")[0]
            self.assertEqual(
                summary,
                "Invalid configuration for LogisticRegression(C=-1) due to invalid value C=-1.",
            )
            fix1 = cm.exception.message.split("\n")[2]
            self.assertRegex(fix1, "C=1.0")

    def test_fixes2(self):
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError) as cm:
                LogisticRegression(
                    penalty="l1",
                    solver="liblinear",
                    multi_class="multinomial",
                    dual=True,
                )
            summary = cm.exception.message.split("\n")[0]
            self.assertRegex(
                summary,
                "Invalid configuration for LogisticRegression(.*) due to constraint",
            )
            fix1 = cm.exception.message.split("\n")[2]
            fix2 = cm.exception.message.split("\n")[3]
            # we don't care what order they are in
            self.assertRegex(fix1 + fix2, "penalty='l2', multi_class='auto'")
            self.assertRegex(fix1 + fix2, "multi_class='auto', dual=False")

    def test_wrong_cat(self):
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError) as cm:
                LogisticRegression(solver="adam")
            summary = cm.exception.message.split("\n")[0]
            self.assertEqual(
                summary,
                "Invalid configuration for LogisticRegression(solver='adam') due to invalid value solver=adam.",
            )

    def test_unknown_arg(self):
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError) as cm:
                LogisticRegression(activation="relu")
            summary = cm.exception.message.split("\n")[0]
            self.assertEqual(
                summary,
                "Invalid configuration for LogisticRegression(activation='relu') due to argument 'activation' was unexpected.",
            )
            fix1 = cm.exception.message.split("\n")[1]
            self.assertRegex(fix1, "remove unknown key 'activation'")

    def test_constraint(self):
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError) as cm:
                LogisticRegression(solver="sag", penalty="l1")
            summary = cm.exception.message.split("\n")[0]
            self.assertEqual(
                summary,
                "Invalid configuration for LogisticRegression(solver='sag', penalty='l1') due to constraint the newton-cg, sag, and lbfgs solvers support only l2 or no penalties.",
            )
            fix1 = cm.exception.message.split("\n")[2]
            self.assertRegex(fix1, "set penalty='l2'")

    def test_unknown_arg_and_constraint2(self):
        with EnableSchemaValidation():
            with EnableSchemaValidation():
                with self.assertRaises(jsonschema.ValidationError) as cm:
                    LogisticRegression(
                        activation="relu",
                        penalty="l1",
                        solver="liblinear",
                        multi_class="multinomial",
                        dual=True,
                    )
                summary = cm.exception.message.split("\n")[0]

                self.assertRegex(
                    summary,
                    "Invalid configuration for LogisticRegression.*due to argument 'activation' was unexpected.",
                )

                fix1 = cm.exception.message.split("\n")[2]
                fix2 = cm.exception.message.split("\n")[3]
                # we don't care what order they are in
                self.assertRegex(
                    fix1 + fix2, "remove unknown key 'activation'.*set.*penalty='l2'"
                )
                self.assertRegex(
                    fix1 + fix2, "remove unknown key 'activation'.*set.*dual=False"
                )
                self.assertRegex(
                    fix1 + fix2,
                    "remove unknown key 'activation'.*set.*multi_class='auto'",
                )

    def test_unknown_arg_and_constraint(self):
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError) as cm:
                LogisticRegression(activation="relu", solver="sag", penalty="l1")
            summary = cm.exception.message.split("\n")[0]
            self.assertRegex(
                summary,
                "Invalid configuration for LogisticRegression.*due to argument 'activation' was unexpected.",
            )
            fix1 = cm.exception.message.split("\n")[2]
            self.assertRegex(fix1, "remove unknown key 'activation'.*set penalty='l2'")


class TestHyperparamConstraints(unittest.TestCase):
    def setUp(self):
        import scipy.sparse
        import sklearn.datasets

        data = sklearn.datasets.load_iris()
        X, y = data.data, data.target

        sparse_X = scipy.sparse.csr_matrix(X)
        self.sparse_X = sparse_X
        self.X = X
        self.y = y

        self.regression_X, self.regression_y = sklearn.datasets.make_regression(
            n_features=4, n_informative=2, random_state=0, shuffle=False
        )

    def test_bagging_classifier(self):
        import sklearn

        from lale.lib.sklearn import BaggingClassifier

        bad_hyperparams = {"bootstrap": False, "oob_score": True}
        trainable = sklearn.ensemble.BaggingClassifier(**bad_hyperparams)

        with self.assertRaisesRegex(
            ValueError, "Out of bag estimation only available if bootstrap=True"
        ):
            trainable.fit(self.X, self.y)

        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                BaggingClassifier(**bad_hyperparams)

    def test_bagging_classifier_2(self):
        import sklearn

        from lale.lib.sklearn import BaggingClassifier

        bad_hyperparams = {"warm_start": True, "oob_score": True}
        trainable = sklearn.ensemble.BaggingClassifier(**bad_hyperparams)
        with self.assertRaisesRegex(
            ValueError, "Out of bag estimate only available if warm_start=False"
        ):
            trainable.fit(self.X, self.y)

        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                BaggingClassifier(**bad_hyperparams)

    def test_bagging_regressor(self):
        import sklearn

        from lale.lib.sklearn import BaggingRegressor

        bad_hyperparams = {"bootstrap": False, "oob_score": True}
        trainable = sklearn.ensemble.BaggingRegressor(**bad_hyperparams)
        with self.assertRaisesRegex(
            ValueError, "Out of bag estimation only available if bootstrap=True"
        ):
            trainable.fit(self.regression_X, self.regression_y)
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                BaggingRegressor(**bad_hyperparams)

    def test_bagging_regressor_2(self):
        import sklearn

        from lale.lib.sklearn import BaggingRegressor

        bad_hyperparams = {"warm_start": True, "oob_score": True}
        trainable = sklearn.ensemble.BaggingRegressor(**bad_hyperparams)
        with self.assertRaisesRegex(
            ValueError, "Out of bag estimate only available if warm_start=False"
        ):
            trainable.fit(self.regression_X, self.regression_y)

        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                BaggingRegressor(**bad_hyperparams)

    def test_extra_trees_classifier(self):
        import sklearn

        from lale.lib.sklearn import ExtraTreesClassifier

        bad_hyperparams = {"bootstrap": False, "oob_score": True}
        trainable = sklearn.ensemble.ExtraTreesClassifier(**bad_hyperparams)
        with self.assertRaisesRegex(
            ValueError, "Out of bag estimation only available if bootstrap=True"
        ):
            trainable.fit(self.X, self.y)

        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                ExtraTreesClassifier(**bad_hyperparams)

    def test_extra_trees_regressor(self):
        import sklearn

        from lale.lib.sklearn import ExtraTreesRegressor

        bad_hyperparams = {"bootstrap": False, "oob_score": True}
        trainable = sklearn.ensemble.ExtraTreesRegressor(**bad_hyperparams)
        with self.assertRaisesRegex(
            ValueError, "Out of bag estimation only available if bootstrap=True"
        ):
            trainable.fit(self.regression_X, self.regression_y)

        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                ExtraTreesRegressor(**bad_hyperparams)

    def test_function_transformer(self):
        import sklearn

        from lale.lib.sklearn import FunctionTransformer

        bad_hyperparams = {"validate": True, "accept_sparse": False}
        bad_X = self.sparse_X
        y = self.y

        trainable = sklearn.preprocessing.FunctionTransformer(**bad_hyperparams)
        with self.assertRaisesRegex(
            TypeError, "A sparse matrix was passed, but dense data is required."
        ):
            trainable.fit(bad_X, self.y)

        trainable = FunctionTransformer(**bad_hyperparams)
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                trainable.fit(bad_X, y)

    def test_linear_svc_1(self):
        import sklearn

        from lale.lib.sklearn import LinearSVC

        bad_hyperparams = {"penalty": "l1", "loss": "hinge", "multi_class": "ovr"}
        trainable = sklearn.svm.LinearSVC(**bad_hyperparams)
        with self.assertRaisesRegex(
            ValueError,
            "The combination of penalty='l1' and loss='hinge' is not supported",
        ):
            trainable.fit(self.X, self.y)

        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                LinearSVC(**bad_hyperparams)

    def test_linear_svc_2(self):
        import sklearn

        from lale.lib.sklearn import LinearSVC

        bad_hyperparams = {
            "penalty": "l2",
            "loss": "hinge",
            "dual": False,
            "multi_class": "ovr",
        }
        trainable = sklearn.svm.LinearSVC(**bad_hyperparams)
        with self.assertRaisesRegex(
            ValueError,
            "The combination of penalty='l2' and loss='hinge' are not supported when dual=False",
        ):
            trainable.fit(self.X, self.y)

        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                LinearSVC(**bad_hyperparams)

    def test_linear_svc_3(self):
        import sklearn

        from lale.lib.sklearn import LinearSVC

        bad_hyperparams = {
            "penalty": "l1",
            "loss": "squared_hinge",
            "dual": True,
            "multi_class": "ovr",
        }
        trainable = sklearn.svm.LinearSVC(**bad_hyperparams)
        with self.assertRaisesRegex(
            ValueError,
            "The combination of penalty='l1' and loss='squared_hinge' are not supported when dual=True",
        ):
            trainable.fit(self.X, self.y)

        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                LinearSVC(**bad_hyperparams)

    def test_linear_svr(self):
        import sklearn

        from lale.lib.sklearn import LinearSVR

        bad_hyperparams = {"loss": "epsilon_insensitive", "dual": False}
        trainable = sklearn.svm.LinearSVR(**bad_hyperparams)
        with self.assertRaisesRegex(
            ValueError,
            "The combination of penalty='l2' and loss='epsilon_insensitive' are not supported when dual=False",
        ):
            trainable.fit(self.regression_X, self.regression_y)

        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                LinearSVR(**bad_hyperparams)

    def test_logistic_regression_1(self):
        import sklearn

        from lale.lib.sklearn import LogisticRegression

        bad_hyperparams = {"solver": "liblinear", "penalty": "none"}
        trainable = sklearn.linear_model.LogisticRegression(**bad_hyperparams)
        with self.assertRaisesRegex(
            ValueError, "penalty='none' is not supported for the liblinear solver"
        ):
            trainable.fit(self.X, self.y)

        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                LogisticRegression(**bad_hyperparams)

    def test_logistic_regression_2(self):
        import sklearn

        from lale.lib.sklearn import LogisticRegression

        bad_hyperparams = {
            "penalty": "elasticnet",
            "l1_ratio": None,
            "solver": "saga",
        }
        trainable = sklearn.linear_model.LogisticRegression(**bad_hyperparams)
        with self.assertRaisesRegex(ValueError, "l1_ratio must be between 0 and 1"):
            trainable.fit(self.X, self.y)

        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                LogisticRegression(**bad_hyperparams)

    def test_logistic_regression_3(self):
        import sklearn

        from lale.lib.sklearn import LogisticRegression

        bad_hyperparams = {
            "penalty": "elasticnet",
            "solver": "liblinear",
            "l1_ratio": 0.5,
        }
        trainable = sklearn.linear_model.LogisticRegression(**bad_hyperparams)
        with self.assertRaisesRegex(
            ValueError, "Only 'saga' solver supports elasticnet penalty"
        ):
            trainable.fit(self.X, self.y)
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                LogisticRegression(**bad_hyperparams)

    def test_missing_indicator(self):
        import sklearn

        from lale.lib.sklearn import MissingIndicator

        bad_X = self.sparse_X
        y = self.y

        bad_hyperparams = {"missing_values": 0}

        trainable = sklearn.impute.MissingIndicator(**bad_hyperparams)
        with self.assertRaisesRegex(
            ValueError, "Sparse input with missing_values=0 is not supported."
        ):
            trainable.fit(bad_X, self.y)

        trainable = MissingIndicator(**bad_hyperparams)
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                trainable.fit(bad_X, y)

    def test_one_hot_encoder(self):
        import sklearn

        from lale.lib.sklearn import OneHotEncoder

        bad_hyperparams = {"drop": "first", "handle_unknown": "ignore"}
        trainable = sklearn.preprocessing.OneHotEncoder(**bad_hyperparams)
        with self.assertRaisesRegex(
            ValueError,
            "`handle_unknown` must be 'error' when the drop parameter is specified",
        ):
            trainable.fit(self.X, self.y)

        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                OneHotEncoder(**bad_hyperparams)

    def test_ordinal_encoder_1(self):
        import sklearn

        from lale.lib.sklearn import OrdinalEncoder

        if sklearn.__version__ >= "0.24.1":
            bad_hyperparams = {
                "handle_unknown": "use_encoded_value",
                "unknown_value": None,
            }
            trainable = sklearn.preprocessing.OrdinalEncoder(**bad_hyperparams)
            with self.assertRaisesRegex(
                TypeError,
                "unknown_value should be an integer or np.nan when handle_unknown is 'use_encoded_value'",
            ):
                trainable.fit(self.X, self.y)

            with EnableSchemaValidation():
                with self.assertRaises(jsonschema.ValidationError):
                    OrdinalEncoder(**bad_hyperparams)

    def test_ordinal_encoder_2(self):
        import sklearn

        from lale.lib.sklearn import OrdinalEncoder

        if sklearn.__version__ >= "0.24.1":
            bad_hyperparams = {"handle_unknown": "error", "unknown_value": 1}
            trainable = sklearn.preprocessing.OrdinalEncoder(**bad_hyperparams)
            with self.assertRaisesRegex(
                TypeError,
                "unknown_value should only be set when handle_unknown is 'use_encoded_value'",
            ):
                trainable.fit(self.X, self.y)

            with EnableSchemaValidation():
                with self.assertRaises(jsonschema.ValidationError):
                    OrdinalEncoder(**bad_hyperparams)

    def test_random_forest_classifier(self):
        import sklearn

        from lale.lib.sklearn import RandomForestClassifier

        bad_hyperparams = {"bootstrap": False, "oob_score": True}
        trainable = sklearn.ensemble.RandomForestClassifier(**bad_hyperparams)
        with self.assertRaisesRegex(
            ValueError, "Out of bag estimation only available if bootstrap=True"
        ):
            trainable.fit(self.X, self.y)

        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                RandomForestClassifier(**bad_hyperparams)

    def test_random_forest_regressor(self):
        import sklearn

        from lale.lib.sklearn import RandomForestRegressor

        bad_hyperparams = {"bootstrap": False, "oob_score": True}
        trainable = sklearn.ensemble.RandomForestRegressor(**bad_hyperparams)
        with self.assertRaisesRegex(
            ValueError, "Out of bag estimation only available if bootstrap=True"
        ):
            trainable.fit(self.regression_X, self.regression_y)

        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                RandomForestRegressor(**bad_hyperparams)

    def test_ridge_1(self):
        import sklearn

        from lale.lib.sklearn import Ridge

        bad_X = self.sparse_X
        y = self.y

        bad_hyperparams = {"fit_intercept": True, "solver": "lsqr"}
        trainable = sklearn.linear_model.Ridge(**bad_hyperparams)
        with self.assertRaisesRegex(
            ValueError, "does not support fitting the intercept on sparse data."
        ):
            trainable.fit(bad_X, self.y)

        trainable = Ridge(**bad_hyperparams)
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                trainable.fit(bad_X, y)

    def test_ridge_2(self):
        import sklearn

        from lale.lib.sklearn import Ridge

        bad_X = self.sparse_X
        y = self.y

        bad_hyperparams = {"solver": "svd", "fit_intercept": False}
        trainable = sklearn.linear_model.Ridge(**bad_hyperparams)
        with self.assertRaisesRegex(
            TypeError, "SVD solver does not support sparse inputs currently"
        ):
            trainable.fit(bad_X, self.y)

        trainable = Ridge(**bad_hyperparams)
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                trainable.fit(bad_X, y)

    def test_robust_scaler(self):
        import sklearn

        from lale.lib.sklearn import RobustScaler

        bad_X = self.sparse_X
        y = self.y

        bad_hyperparams = {"with_centering": True}
        trainable = sklearn.preprocessing.RobustScaler(**bad_hyperparams)
        with self.assertRaisesRegex(
            ValueError,
            "Cannot center sparse matrices: use `with_centering=False` instead.",
        ):
            trainable.fit(bad_X, self.y)

        trainable = RobustScaler(**bad_hyperparams)
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                trainable.fit(bad_X, y)

    def test_simple_imputer(self):
        import sklearn

        from lale.lib.sklearn import SimpleImputer

        bad_X = self.sparse_X
        y = self.y

        bad_hyperparams = {"missing_values": 0}
        trainable = sklearn.impute.SimpleImputer(**bad_hyperparams)
        with self.assertRaisesRegex(
            ValueError,
            "Imputation not possible when missing_values == 0 and input is sparse.",
        ):
            trainable.fit(bad_X, self.y)

        trainable = SimpleImputer(**bad_hyperparams)
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                trainable.fit(bad_X, y)

    def test_svc(self):
        import sklearn

        from lale.lib.sklearn import SVC

        bad_X = self.sparse_X
        y = self.y

        bad_hyperparams = {"kernel": "precomputed"}
        trainable = sklearn.svm.SVC(**bad_hyperparams)
        with self.assertRaisesRegex(
            TypeError, "Sparse precomputed kernels are not supported."
        ):
            trainable.fit(bad_X, self.y)

        trainable = SVC(**bad_hyperparams)
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                trainable.fit(bad_X, y)

    def test_svr(self):
        import sklearn

        from lale.lib.sklearn import SVR

        bad_X = self.sparse_X
        y = self.y

        bad_hyperparams = {"kernel": "precomputed"}
        trainable = sklearn.svm.SVR(**bad_hyperparams)
        with self.assertRaisesRegex(
            TypeError, "Sparse precomputed kernels are not supported."
        ):
            trainable.fit(bad_X, self.y)

        trainable = SVR(**bad_hyperparams)
        with EnableSchemaValidation():
            with self.assertRaises(jsonschema.ValidationError):
                trainable.fit(bad_X, y)


class TestSchemaValidation(unittest.TestCase):
    def test_any(self):
        from lale.type_checking import is_subschema

        num_schema = {"type": "number"}
        any_schema = {"laleType": "Any"}
        jsonschema.validate(42, num_schema)
        jsonschema.validate(42, any_schema)
        self.assertTrue(is_subschema(num_schema, any_schema))
        self.assertTrue(is_subschema(any_schema, num_schema))

    def test_bool_label(self):
        import pandas as pd

        data_records = [
            {
                "IS_TENT": False,
                "GENDER": "M",
                "AGE": 20,
                "MARITAL_STATUS": "Single",
                "PROFESSION": "Sales",
            },
            {
                "IS_TENT": False,
                "GENDER": "M",
                "AGE": 20,
                "MARITAL_STATUS": "Single",
                "PROFESSION": "Sales",
            },
            {
                "IS_TENT": False,
                "GENDER": "F",
                "AGE": 37,
                "MARITAL_STATUS": "Single",
                "PROFESSION": "Other",
            },
            {
                "IS_TENT": False,
                "GENDER": "M",
                "AGE": 42,
                "MARITAL_STATUS": "Married",
                "PROFESSION": "Other",
            },
            {
                "IS_TENT": True,
                "GENDER": "F",
                "AGE": 24,
                "MARITAL_STATUS": "Married",
                "PROFESSION": "Retail",
            },
            {
                "IS_TENT": False,
                "GENDER": "F",
                "AGE": 24,
                "MARITAL_STATUS": "Married",
                "PROFESSION": "Retail",
            },
            {
                "IS_TENT": False,
                "GENDER": "M",
                "AGE": 29,
                "MARITAL_STATUS": "Single",
                "PROFESSION": "Retail",
            },
            {
                "IS_TENT": False,
                "GENDER": "M",
                "AGE": 29,
                "MARITAL_STATUS": "Single",
                "PROFESSION": "Retail",
            },
            {
                "IS_TENT": True,
                "GENDER": "M",
                "AGE": 43,
                "MARITAL_STATUS": "Married",
                "PROFESSION": "Trades",
            },
            {
                "IS_TENT": False,
                "GENDER": "M",
                "AGE": 43,
                "MARITAL_STATUS": "Married",
                "PROFESSION": "Trades",
            },
        ]
        df = pd.DataFrame.from_records(data_records)
        X = df.drop(["IS_TENT"], axis=1).values
        y = df["IS_TENT"].values
        from lale.lib.sklearn import GradientBoostingClassifier as Clf
        from lale.lib.sklearn import OneHotEncoder as Enc

        trainable = Enc() >> Clf()
        _ = trainable.fit(X, y)


class TestWithScorer(unittest.TestCase):
    def test_bare_array(self):
        import sklearn.datasets
        import sklearn.metrics
        from numpy import ndarray

        from lale.datasets.data_schemas import NDArrayWithSchema

        X, y = sklearn.datasets.load_iris(return_X_y=True)
        self.assertIsInstance(X, ndarray)
        self.assertIsInstance(y, ndarray)
        self.assertNotIsInstance(X, NDArrayWithSchema)
        self.assertNotIsInstance(y, NDArrayWithSchema)
        trainable = LogisticRegression()
        trained = trainable.fit(X, y)
        scorer = sklearn.metrics.make_scorer(sklearn.metrics.accuracy_score)
        out = scorer(trained, X, y)
        self.assertIsInstance(out, float)
        self.assertNotIsInstance(out, NDArrayWithSchema)


class TestDisablingSchemaValidation(unittest.TestCase):
    def setUp(self):
        from sklearn.datasets import load_iris
        from sklearn.model_selection import train_test_split

        data = load_iris()
        X, y = data.data, data.target
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y)

    def test_disable_schema_validation_individual_op(self):
        existing_flag = disable_data_schema_validation
        set_disable_data_schema_validation(True)
        import lale.schemas as schemas
        from lale.lib.sklearn import PCA

        pca_input = schemas.Object(
            X=schemas.AnyOf(
                [
                    schemas.Array(schemas.Array(schemas.String())),
                    schemas.Array(schemas.String()),
                ]
            )
        )

        foo = PCA.customize_schema(input_fit=pca_input)

        pca_output = schemas.Object(
            X=schemas.AnyOf(
                [
                    schemas.Array(schemas.Array(schemas.String())),
                    schemas.Array(schemas.String()),
                ]
            )
        )

        foo = foo.customize_schema(output_transform=pca_output)

        abc = foo()
        trained_pca = abc.fit(self.X_train)
        trained_pca.transform(self.X_test)
        set_disable_data_schema_validation(existing_flag)

    def test_enable_schema_validation_individual_op(self):
        with EnableSchemaValidation():
            import lale.schemas as schemas
            from lale.lib.sklearn import PCA

            pca_input = schemas.Object(
                X=schemas.AnyOf(
                    [
                        schemas.Array(schemas.Array(schemas.String())),
                        schemas.Array(schemas.String()),
                    ]
                )
            )

            foo = PCA.customize_schema(input_fit=pca_input)

            pca_output = schemas.Object(
                X=schemas.AnyOf(
                    [
                        schemas.Array(schemas.Array(schemas.String())),
                        schemas.Array(schemas.String()),
                    ]
                )
            )

            foo = foo.customize_schema(output_transform=pca_output)

            abc = foo()
            with self.assertRaises(ValueError):
                trained_pca = abc.fit(self.X_train)
                trained_pca.transform(self.X_test)

    def test_disable_schema_validation_pipeline(self):
        existing_flag = disable_data_schema_validation
        set_disable_data_schema_validation(True)
        import lale.schemas as schemas
        from lale.lib.sklearn import PCA, LogisticRegression

        lr_input = schemas.Object(
            required=["X", "y"],
            X=schemas.AnyOf(
                [
                    schemas.Array(schemas.Array(schemas.String())),
                    schemas.Array(schemas.String()),
                ]
            ),
            y=schemas.Array(schemas.String()),
        )

        foo = LogisticRegression.customize_schema(input_fit=lr_input)
        abc = foo()
        pipeline = PCA() >> abc
        trained_pipeline = pipeline.fit(self.X_train, self.y_train)
        trained_pipeline.predict(self.X_test)
        set_disable_data_schema_validation(existing_flag)

    def test_enable_schema_validation_pipeline(self):
        with EnableSchemaValidation():
            import lale.schemas as schemas
            from lale.lib.sklearn import PCA, LogisticRegression

            lr_input = schemas.Object(
                required=["X", "y"],
                X=schemas.AnyOf(
                    [
                        schemas.Array(schemas.Array(schemas.String())),
                        schemas.Array(schemas.String()),
                    ]
                ),
                y=schemas.Array(schemas.String()),
            )

            foo = LogisticRegression.customize_schema(input_fit=lr_input)
            abc = foo()
            pipeline = PCA() >> abc
            with self.assertRaises(ValueError):
                trained_pipeline = pipeline.fit(self.X_train, self.y_train)
                trained_pipeline.predict(self.X_test)

    def test_disable_enable_hyperparam_validation(self):
        from lale.lib.sklearn import PCA

        existing_flag = disable_hyperparams_schema_validation
        set_disable_hyperparams_schema_validation(True)
        PCA(n_components=True)
        set_disable_hyperparams_schema_validation(False)
        with self.assertRaises(jsonschema.ValidationError):
            PCA(n_components=True)
        set_disable_hyperparams_schema_validation(existing_flag)
