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
from test.mock_module import UnknownOp
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier as baz
from sklearn.decomposition import PCA as foo
from sklearn.linear_model import Lars as foobar
from xgboost import XGBClassifier as bar

import lale
import lale.schemas as schemas
import lale.type_checking
from lale.search.lale_grid_search_cv import get_grid_search_parameter_grids


class TestCustomSchema(unittest.TestCase):
    def setUp(self):
        import sklearn.decomposition

        import lale.lib.sklearn
        from lale.operators import make_operator

        self.sk_pca = make_operator(sklearn.decomposition.PCA, schemas={})
        self.ll_pca = lale.lib.sklearn.PCA
        self.maxDiff = None

    def test_override_schemas(self):
        with EnableSchemaValidation():
            init_schemas = self.sk_pca._schemas
            pca_schemas = self.ll_pca._schemas
            foo = self.sk_pca.customize_schema(schemas=schemas.JSON(pca_schemas))
            self.assertEqual(foo._schemas, pca_schemas)
            self.assertEqual(self.sk_pca._schemas, init_schemas)
            self.assertRaises(Exception, self.sk_pca.customize_schema, schemas={})

    def test_override_input(self):
        with EnableSchemaValidation():
            init_input_schema = self.sk_pca.get_schema("input_fit")
            pca_input = self.ll_pca.get_schema("input_fit")
            foo = self.sk_pca.customize_schema(input_fit=schemas.JSON(pca_input))
            self.assertEqual(foo.get_schema("input_fit"), pca_input)
            lale.type_checking.validate_is_schema(foo._schemas)
            self.assertEqual(self.sk_pca.get_schema("input_fit"), init_input_schema)
            self.assertRaises(Exception, self.sk_pca.customize_schema, input_fit=42)
            _ = self.sk_pca.customize_schema(input_foo=pca_input)

    def test_override_output(self):
        with EnableSchemaValidation():
            init_output_schema = self.sk_pca.get_schema("output_transform")
            pca_output = self.ll_pca.get_schema("output_transform")
            foo = self.sk_pca.customize_schema(
                output_transform=schemas.JSON(pca_output)
            )
            self.assertEqual(foo.get_schema("output_transform"), pca_output)
            lale.type_checking.validate_is_schema(foo._schemas)
            self.assertEqual(
                self.sk_pca.get_schema("output_transform"), init_output_schema
            )
            self.assertRaises(Exception, self.sk_pca.customize_schema, output=42)
            _ = self.sk_pca.customize_schema(output_foo=pca_output)

    def test_override_output2(self):
        init_output_schema = self.sk_pca.get_schema("output_transform")
        pca_output = schemas.AnyOf(
            [
                schemas.Array(schemas.Array(schemas.Float())),
                schemas.Array(schemas.Float()),
            ]
        )
        expected = {
            "anyOf": [
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"type": "number"}},
                },
                {"type": "array", "items": {"type": "number"}},
            ]
        }
        foo = self.sk_pca.customize_schema(output_transform=pca_output)
        self.assertEqual(foo.get_schema("output_transform"), expected)
        lale.type_checking.validate_is_schema(foo._schemas)
        self.assertEqual(self.sk_pca.get_schema("output_transform"), init_output_schema)

    def test_override_bool_param_sk(self):
        with EnableSchemaValidation():
            init = self.sk_pca.hyperparam_schema("whiten")
            expected = {"default": True, "type": "boolean", "description": "override"}
            foo = self.sk_pca.customize_schema(
                whiten=schemas.Bool(default=True, desc="override")
            )
            self.assertEqual(foo.hyperparam_schema("whiten"), expected)
            lale.type_checking.validate_is_schema(foo._schemas)
            self.assertEqual(self.sk_pca.hyperparam_schema("whiten"), init)
            self.assertRaises(Exception, self.sk_pca.customize_schema, whitenX=42)

    def test_override_bool_param_ll(self):
        with EnableSchemaValidation():
            init = self.ll_pca.hyperparam_schema("whiten")
            expected = {"default": True, "type": "boolean"}
            foo = self.ll_pca.customize_schema(whiten=schemas.Bool(default=True))
            self.assertEqual(foo.hyperparam_schema("whiten"), expected)
            lale.type_checking.validate_is_schema(foo._schemas)
            self.assertEqual(self.ll_pca.hyperparam_schema("whiten"), init)
            self.assertRaises(Exception, self.ll_pca.customize_schema, whitenX=42)

    def test_override_enum_param(self):
        with EnableSchemaValidation():
            init = self.ll_pca.hyperparam_schema("svd_solver")
            expected = {"default": "full", "enum": ["auto", "full"]}
            foo = self.ll_pca.customize_schema(
                svd_solver=schemas.Enum(default="full", values=["auto", "full"])
            )
            self.assertEqual(foo.hyperparam_schema("svd_solver"), expected)
            lale.type_checking.validate_is_schema(foo._schemas)
            self.assertEqual(self.ll_pca.hyperparam_schema("svd_solver"), init)

    def test_override_float_param(self):
        init = self.ll_pca.hyperparam_schema("tol")
        expected = {
            "default": 0.1,
            "type": "number",
            "minimum": -10,
            "maximum": 10,
            "exclusiveMaximum": True,
            "exclusiveMinimum": False,
        }
        foo = self.ll_pca.customize_schema(
            tol=schemas.Float(
                default=0.1,
                minimum=-10,
                maximum=10,
                exclusiveMaximum=True,
                exclusiveMinimum=False,
            )
        )
        self.assertEqual(foo.hyperparam_schema("tol"), expected)
        lale.type_checking.validate_is_schema(foo._schemas)
        self.assertEqual(self.ll_pca.hyperparam_schema("tol"), init)

    def test_override_int_param(self):
        init = self.ll_pca.hyperparam_schema("iterated_power")
        expected = {
            "default": 1,
            "type": "integer",
            "minimum": -10,
            "maximum": 10,
            "exclusiveMaximum": True,
            "exclusiveMinimum": False,
        }
        foo = self.ll_pca.customize_schema(
            iterated_power=schemas.Int(
                default=1,
                minimum=-10,
                maximum=10,
                exclusiveMaximum=True,
                exclusiveMinimum=False,
            )
        )
        self.assertEqual(foo.hyperparam_schema("iterated_power"), expected)
        lale.type_checking.validate_is_schema(foo._schemas)
        self.assertEqual(self.ll_pca.hyperparam_schema("iterated_power"), init)

    def test_override_null_param(self):
        init = self.ll_pca.hyperparam_schema("n_components")
        expected = {"enum": [None]}
        foo = self.ll_pca.customize_schema(n_components=schemas.Null())
        self.assertEqual(foo.hyperparam_schema("n_components"), expected)
        lale.type_checking.validate_is_schema(foo._schemas)
        self.assertEqual(self.ll_pca.hyperparam_schema("n_components"), init)

    def test_override_json_param(self):
        init = self.ll_pca.hyperparam_schema("tol")
        expected = {
            "description": "Tol",
            "type": "number",
            "minimum": 0.2,
            "default": 1.0,
        }
        foo = self.ll_pca.customize_schema(tol=schemas.JSON(expected))
        self.assertEqual(foo.hyperparam_schema("tol"), expected)
        lale.type_checking.validate_is_schema(foo._schemas)
        self.assertEqual(self.ll_pca.hyperparam_schema("tol"), init)

    def test_override_any_param(self):
        init = self.ll_pca.hyperparam_schema("iterated_power")
        expected = {
            "anyOf": [{"type": "integer"}, {"enum": ["auto", "full"]}],
            "default": "auto",
        }
        foo = self.ll_pca.customize_schema(
            iterated_power=schemas.AnyOf(
                [schemas.Int(), schemas.Enum(["auto", "full"])], default="auto"
            )
        )
        self.assertEqual(foo.hyperparam_schema("iterated_power"), expected)
        lale.type_checking.validate_is_schema(foo._schemas)
        self.assertEqual(self.ll_pca.hyperparam_schema("iterated_power"), init)

    def test_override_array_param(self):
        init = self.sk_pca.hyperparam_schema("copy")
        expected = {
            "type": "array",
            "minItems": 1,
            "maxItems": 20,
            "items": {"type": "integer"},
        }
        foo = self.sk_pca.customize_schema(
            copy=schemas.Array(minItems=1, maxItems=20, items=schemas.Int())
        )
        self.assertEqual(foo.hyperparam_schema("copy"), expected)
        lale.type_checking.validate_is_schema(foo._schemas)
        self.assertEqual(self.sk_pca.hyperparam_schema("copy"), init)

    def test_override_object_param(self):
        init = self.sk_pca.get_schema("input_fit")
        expected = {
            "type": "object",
            "required": ["X"],
            "additionalProperties": False,
            "properties": {"X": {"type": "array", "items": {"type": "number"}}},
        }
        foo = self.sk_pca.customize_schema(
            input_fit=schemas.Object(
                required=["X"],
                additionalProperties=False,
                X=schemas.Array(schemas.Float()),
            )
        )
        self.assertEqual(foo.get_schema("input_fit"), expected)
        lale.type_checking.validate_is_schema(foo.get_schema("input_fit"))
        self.assertEqual(self.sk_pca.get_schema("input_fit"), init)

    def test_add_constraint(self):
        init = self.sk_pca.hyperparam_schema()
        init_expected = {
            "allOf": [
                {
                    "type": "object",
                    "relevantToOptimizer": [],
                    "additionalProperties": False,
                    "properties": {
                        "n_components": {"default": None},
                        "copy": {"default": True},
                        "whiten": {"default": False},
                        "svd_solver": {"default": "auto"},
                        "tol": {"default": 0.0},
                        "iterated_power": {"default": "auto"},
                        "random_state": {"default": None},
                    },
                }
            ]
        }
        self.assertEqual(init, init_expected)
        expected = {
            "allOf": [
                init_expected["allOf"][0],
                {
                    "anyOf": [
                        {
                            "type": "object",
                            "properties": {
                                "n_components": {
                                    "not": {"enum": ["mle"]},
                                }
                            },
                        },
                        {
                            "type": "object",
                            "properties": {
                                "svd_solver": {"enum": ["full", "auto"]},
                            },
                        },
                    ]
                },
            ]
        }
        foo = self.sk_pca.customize_schema(
            constraint=schemas.AnyOf(
                [
                    schemas.Object(n_components=schemas.Not(schemas.Enum(["mle"]))),
                    schemas.Object(svd_solver=schemas.Enum(["full", "auto"])),
                ]
            )
        )
        self.assertEqual(foo.hyperparam_schema(), expected)
        lale.type_checking.validate_is_schema(foo._schemas)
        self.assertEqual(self.sk_pca.hyperparam_schema(), init)

    def test_override_relevant(self):
        init = self.ll_pca.hyperparam_schema()["allOf"][0]["relevantToOptimizer"]
        expected = ["svd_solver"]
        foo = self.ll_pca.customize_schema(relevantToOptimizer=["svd_solver"])
        self.assertEqual(
            foo.hyperparam_schema()["allOf"][0]["relevantToOptimizer"], expected
        )
        lale.type_checking.validate_is_schema(foo._schemas)
        self.assertEqual(
            self.ll_pca.hyperparam_schema()["allOf"][0]["relevantToOptimizer"], init
        )
        self.assertRaises(
            Exception, self.sk_pca.customize_schema, relevantToOptimizer={}
        )

    def test_override_tags(self):
        with EnableSchemaValidation():
            init = self.ll_pca._schemas["tags"]
            tags = {
                "pre": ["~categoricals"],
                "op": ["estimator", "classifier", "interpretable"],
                "post": ["probabilities"],
            }
            foo = self.ll_pca.customize_schema(tags=tags)
            self.assertEqual(foo._schemas["tags"], tags)
            lale.type_checking.validate_is_schema(foo._schemas)
            self.assertEqual(self.ll_pca._schemas["tags"], init)
            self.assertRaises(Exception, self.sk_pca.customize_schema, tags=42)

    def test_wrap_imported_operators(self):
        old_globals = {**globals()}
        try:
            from lale.lib.autogen import Lars
            from lale.lib.lightgbm import LGBMClassifier
            from lale.lib.sklearn import PCA
            from lale.lib.xgboost import XGBClassifier

            lale.wrap_imported_operators()
            self.assertEqual(foo._schemas, PCA._schemas)  # type: ignore
            self.assertEqual(bar._schemas, XGBClassifier._schemas)  # type: ignore
            self.assertEqual(baz._schemas, LGBMClassifier._schemas)  # type: ignore
            self.assertEqual(foobar._schemas, Lars._schemas)
        finally:
            for sym, obj in old_globals.items():
                globals()[sym] = obj


class TestConstraintMerging(unittest.TestCase):
    def test_override_float_param1(self):
        from lale.lib.sklearn import PCA
        from lale.search.schema2search_space import op_to_search_space
        from lale.search.search_space import SearchSpaceNumber, SearchSpaceObject

        pca = PCA.customize_schema(
            relevantToOptimizer=["tol"],
            tol=schemas.Float(
                minimum=0.25,
                minimumForOptimizer=0,
                maximum=0.5,
                maximumForOptimizer=1.0,
                exclusiveMaximum=True,
                exclusiveMinimum=False,
            ),
        )
        search = op_to_search_space(pca)
        assert isinstance(search, SearchSpaceObject)
        num_space = list(search.choices)[0][0]
        assert isinstance(num_space, SearchSpaceNumber)
        self.assertEqual(num_space.minimum, 0.25)
        self.assertEqual(num_space.maximum, 0.5)
        self.assertTrue(num_space.exclusiveMaximum)
        self.assertFalse(num_space.exclusiveMinimum)

    def test_override_float_param2(self):
        from lale.lib.sklearn import PCA
        from lale.search.schema2search_space import op_to_search_space
        from lale.search.search_space import SearchSpaceNumber, SearchSpaceObject

        pca = PCA.customize_schema(
            relevantToOptimizer=["tol"],
            tol=schemas.Float(
                minimum=0,
                minimumForOptimizer=0.25,
                maximum=1.5,
                maximumForOptimizer=1.0,
                exclusiveMaximumForOptimizer=False,
                exclusiveMinimumForOptimizer=True,
            ),
        )
        search = op_to_search_space(pca)
        assert isinstance(search, SearchSpaceObject)
        num_space = list(search.choices)[0][0]
        assert isinstance(num_space, SearchSpaceNumber)
        self.assertEqual(num_space.minimum, 0.25)
        self.assertEqual(num_space.maximum, 1.0)
        self.assertFalse(num_space.exclusiveMaximum)
        self.assertTrue(num_space.exclusiveMinimum)

    def test_override_int_param1(self):
        from lale.lib.sklearn import PCA
        from lale.search.schema2search_space import op_to_search_space
        from lale.search.search_space import SearchSpaceNumber, SearchSpaceObject

        pca = PCA.customize_schema(
            relevantToOptimizer=["iterated_power"],
            iterated_power=schemas.Float(
                minimum=1,
                minimumForOptimizer=0,
                maximum=5,
                maximumForOptimizer=6,
                exclusiveMaximum=True,
                exclusiveMinimum=False,
            ),
        )
        search = op_to_search_space(pca)
        assert isinstance(search, SearchSpaceObject)
        num_space = list(search.choices)[0][0]
        assert isinstance(num_space, SearchSpaceNumber)
        self.assertEqual(num_space.minimum, 1)
        self.assertEqual(num_space.maximum, 5)
        self.assertTrue(num_space.exclusiveMaximum)
        self.assertFalse(num_space.exclusiveMinimum)

    def test_override_int_param2(self):
        from lale.lib.sklearn import PCA
        from lale.search.schema2search_space import op_to_search_space
        from lale.search.search_space import SearchSpaceNumber, SearchSpaceObject

        pca = PCA.customize_schema(
            relevantToOptimizer=["iterated_power"],
            iterated_power=schemas.Float(
                minimum=0,
                minimumForOptimizer=1,
                maximum=6,
                maximumForOptimizer=5,
                exclusiveMaximumForOptimizer=False,
                exclusiveMinimumForOptimizer=True,
            ),
        )
        search = op_to_search_space(pca)
        assert isinstance(search, SearchSpaceObject)
        num_space = list(search.choices)[0][0]
        assert isinstance(num_space, SearchSpaceNumber)
        self.assertEqual(num_space.minimum, 1)
        self.assertEqual(num_space.maximum, 5)
        self.assertFalse(num_space.exclusiveMaximum)
        self.assertTrue(num_space.exclusiveMinimum)


class TestWrapUnknownOps(unittest.TestCase):
    expected_schema = {
        "allOf": [
            {
                "type": "object",
                "relevantToOptimizer": [],
                "additionalProperties": False,
                "properties": {
                    "n_neighbors": {"default": 5},
                    "algorithm": {"default": "auto"},
                },
            }
        ]
    }

    def test_wrap_from_class(self):
        from lale.operators import PlannedIndividualOp, make_operator

        self.assertFalse(isinstance(UnknownOp, PlannedIndividualOp))
        Wrapped = make_operator(UnknownOp)
        self.assertTrue(isinstance(Wrapped, PlannedIndividualOp))
        self.assertEqual(Wrapped.hyperparam_schema(), self.expected_schema)
        instance = Wrapped(n_neighbors=3)
        self.assertEqual(instance.hyperparams(), {"n_neighbors": 3})

    def test_wrapped_from_import(self):
        old_globals = {**globals()}
        try:
            from lale.operators import PlannedIndividualOp

            self.assertFalse(isinstance(UnknownOp, PlannedIndividualOp))
            lale.wrap_imported_operators()
            self.assertTrue(isinstance(UnknownOp, PlannedIndividualOp))
            uop: Any = UnknownOp
            self.assertEqual(uop.hyperparam_schema(), self.expected_schema)
            instance = uop(n_neighbors=3)
            self.assertEqual(instance.hyperparams(), {"n_neighbors": 3})
        finally:
            for sym, obj in old_globals.items():
                globals()[sym] = obj

    def test_wrap_from_instance(self):
        from sklearn.base import clone

        from lale.operators import TrainableIndividualOp, make_operator

        self.assertFalse(isinstance(UnknownOp, TrainableIndividualOp))
        instance = UnknownOp(n_neighbors=3)
        self.assertFalse(isinstance(instance, TrainableIndividualOp))
        wrapped = make_operator(instance)
        self.assertTrue(isinstance(wrapped, TrainableIndividualOp))
        assert isinstance(
            wrapped, TrainableIndividualOp
        )  # help type checkers that don't know about assertTrue
        self.assertEqual(wrapped.reduced_hyperparams(), {"n_neighbors": 3})
        cloned = clone(wrapped)
        self.assertTrue(isinstance(cloned, TrainableIndividualOp))
        self.assertEqual(cloned.reduced_hyperparams(), {"n_neighbors": 3})


class TestFreeze(unittest.TestCase):
    def test_individual_op_freeze_trainable(self):
        from lale.lib.sklearn import LogisticRegression

        liquid = LogisticRegression(C=0.1, solver="liblinear")
        self.assertIn("dual", liquid.free_hyperparams())
        self.assertFalse(liquid.is_frozen_trainable())
        liquid_grid = get_grid_search_parameter_grids(liquid)
        self.assertTrue(len(liquid_grid) > 1, f"grid size {len(liquid_grid)}")
        frozen = liquid.freeze_trainable()
        self.assertEqual(len(frozen.free_hyperparams()), 0)
        self.assertTrue(frozen.is_frozen_trainable())
        frozen_grid = get_grid_search_parameter_grids(frozen)
        self.assertEqual(len(frozen_grid), 1)

    def test_pipeline_freeze_trainable(self):
        from lale.lib.sklearn import PCA, LogisticRegression

        liquid = PCA() >> LogisticRegression()
        self.assertFalse(liquid.is_frozen_trainable())
        liquid_grid = get_grid_search_parameter_grids(liquid)
        self.assertTrue(len(liquid_grid) > 1, f"grid size {len(liquid_grid)}")
        frozen = liquid.freeze_trainable()
        self.assertTrue(frozen.is_frozen_trainable())
        frozen_grid = get_grid_search_parameter_grids(frozen)
        self.assertEqual(len(frozen_grid), 1)

    def test_individual_op_freeze_trained(self):
        from lale.lib.sklearn import KNeighborsClassifier

        with EnableSchemaValidation():
            trainable = KNeighborsClassifier(n_neighbors=1)
            X = np.array([[0.0], [1.0], [2.0]])
            y_old = np.array([0.0, 0.0, 1.0])
            y_new = np.array([1.0, 0.0, 0.0])
            liquid_old = trainable.fit(X, y_old)
            self.assertEqual(list(liquid_old.predict(X)), list(y_old))
            liquid_new = liquid_old.fit(X, y_new)
            self.assertEqual(list(liquid_new.predict(X)), list(y_new))
            frozen_old = trainable.fit(X, y_old).freeze_trained()
            self.assertFalse(liquid_old.is_frozen_trained())
            self.assertTrue(frozen_old.is_frozen_trained())
            self.assertEqual(list(frozen_old.predict(X)), list(y_old))
            frozen_new = frozen_old.fit(X, y_new)
            self.assertEqual(list(frozen_new.predict(X)), list(y_old))

    def test_pipeline_freeze_trained(self):
        from lale.lib.sklearn import LogisticRegression, MinMaxScaler

        trainable = MinMaxScaler() >> LogisticRegression()
        X = [[0.0], [1.0], [2.0]]
        y = [0.0, 0.0, 1.0]
        liquid = trainable.fit(X, y)
        frozen = liquid.freeze_trained()
        self.assertFalse(liquid.is_frozen_trained())
        self.assertTrue(frozen.is_frozen_trained())

    def test_trained_individual_op_freeze_trainable(self):
        from lale.lib.sklearn import KNeighborsClassifier
        from lale.operators import TrainedIndividualOp

        with EnableSchemaValidation():
            trainable = KNeighborsClassifier(n_neighbors=1)
            X = np.array([[0.0], [1.0], [2.0]])
            y_old = np.array([0.0, 0.0, 1.0])
            liquid = trainable.fit(X, y_old)
            self.assertIsInstance(liquid, TrainedIndividualOp)
            self.assertFalse(liquid.is_frozen_trainable())
            self.assertIn("algorithm", liquid.free_hyperparams())
            frozen = liquid.freeze_trainable()
            self.assertIsInstance(frozen, TrainedIndividualOp)
            self.assertTrue(frozen.is_frozen_trainable())
            self.assertFalse(frozen.is_frozen_trained())
            self.assertEqual(len(frozen.free_hyperparams()), 0)

    def test_trained_pipeline_freeze_trainable(self):
        from lale.lib.sklearn import LogisticRegression, MinMaxScaler
        from lale.operators import TrainedPipeline

        trainable = MinMaxScaler() >> LogisticRegression()
        X = [[0.0], [1.0], [2.0]]
        y = [0.0, 0.0, 1.0]
        liquid = trainable.fit(X, y)
        self.assertIsInstance(liquid, TrainedPipeline)
        self.assertFalse(liquid.is_frozen_trainable())
        frozen = liquid.freeze_trainable()
        self.assertFalse(liquid.is_frozen_trainable())
        self.assertTrue(frozen.is_frozen_trainable())
        self.assertIsInstance(frozen, TrainedPipeline)
