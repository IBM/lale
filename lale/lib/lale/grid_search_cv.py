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

from typing import Any, Dict

import lale.lib.sklearn
import lale.operators
import lale.search.lale_grid_search_cv
import lale.sklearn_compat

from .observing import Observing, ObservingImpl

# from abc import ABC


class GridSearchCVImpl:
    def __init__(
        self,
        estimator=None,
        cv=5,
        scoring=None,
        n_jobs=None,
        lale_num_samples=None,
        lale_num_grids=None,
        param_grid=None,
        pgo=None,
        observer=None,
    ):
        if observer is not None and isinstance(observer, type):
            # if we are given a class name, instantiate it
            observer = observer()
        if scoring is None:
            is_clf = estimator.is_classifier()
            if is_clf:
                scoring = "accuracy"
            else:
                scoring = "r2"

        self._hyperparams = {
            "estimator": estimator,
            "cv": cv,
            "scoring": scoring,
            "n_jobs": n_jobs,
            "lale_num_samples": lale_num_samples,
            "lale_num_grids": lale_num_grids,
            "pgo": pgo,
            "hp_grid": param_grid,
            "observer": observer,
        }

    def fit(self, X, y):
        if self._hyperparams["estimator"] is None:
            op = lale.lib.sklearn.LogisticRegression
        else:
            op = self._hyperparams["estimator"]

        observed_op = op
        obs = self._hyperparams["observer"]
        if obs is not None:
            observed_op = Observing(op=op, observer=obs)

        hp_grid = self._hyperparams["hp_grid"]
        data_schema = lale.helpers.fold_schema(
            X, y, self._hyperparams["cv"], op.is_classifier()
        )
        if hp_grid is None:
            hp_grid = lale.search.lale_grid_search_cv.get_parameter_grids(
                observed_op,
                num_samples=self._hyperparams["lale_num_samples"],
                num_grids=self._hyperparams["lale_num_grids"],
                pgo=self._hyperparams["pgo"],
                data_schema=data_schema,
            )
        if not hp_grid and isinstance(op, lale.operators.IndividualOp):
            hp_grid = [
                lale.search.lale_grid_search_cv.get_defaults_as_param_grid(observed_op)
            ]
        if hp_grid:
            if obs is not None:
                observed_op._impl.startObserving(
                    "optimize",
                    hp_grid=hp_grid,
                    op=op,
                    num_samples=self._hyperparams["lale_num_samples"],
                    num_grids=self._hyperparams["lale_num_grids"],
                    pgo=self._hyperparams["pgo"],
                )
            try:
                self.grid = lale.search.lale_grid_search_cv.get_lale_gridsearchcv_op(
                    lale.sklearn_compat.make_sklearn_compat(observed_op),
                    hp_grid,
                    cv=self._hyperparams["cv"],
                    scoring=self._hyperparams["scoring"],
                    n_jobs=self._hyperparams["n_jobs"],
                )
                self.grid.fit(X, y)
                be = self.grid.best_estimator_.to_lale()
            except BaseException as e:
                if obs is not None:
                    assert isinstance(observed_op._impl, ObservingImpl)
                    observed_op._impl.failObserving("optimize", e)
                raise

            if obs is not None:
                assert isinstance(be._impl, ObservingImpl)
                be_base = be._impl.getOp()
                observed_op._impl.endObserving("optimize", best=be_base)
            self._best_estimator = be
        else:
            assert isinstance(op, lale.operators.TrainableOperator)
            self._best_estimator = op
        return self

    def predict(self, X):
        return self._best_estimator.predict(X)

    def get_pipeline(self, pipeline_name=None, astype="lale"):
        if pipeline_name is not None:
            raise NotImplementedError("Cannot get pipeline by name yet.")
        result = getattr(self, "_best_estimator", None)
        if result is None or astype == "lale":
            return result
        assert astype == "sklearn", astype
        return lale.sklearn_compat.make_sklearn_compat(result)


_hyperparams_schema = {
    "allOf": [
        {
            "type": "object",
            "required": [
                "estimator",
                "cv",
                "scoring",
                "n_jobs",
                "lale_num_samples",
                "lale_num_grids",
                "pgo",
            ],
            "relevantToOptimizer": ["estimator"],
            "additionalProperties": False,
            "properties": {
                "estimator": {
                    "description": "Planned Lale individual operator or pipeline,\nby default LogisticRegression.",
                    "anyOf": [
                        {"laleType": "operator", "not": {"enum": [None]}},
                        {
                            "enum": [None],
                            "description": "lale.lib.sklearn.LogisticRegression",
                        },
                    ],
                    "default": None,
                },
                "cv": {
                    "description": "Number of folds for cross-validation.",
                    "type": "integer",
                    "minimum": 1,
                    "default": 5,
                },
                "scoring": {
                    "description": """Scorer object, or known scorer named by string.
Default of None translates to `accuracy` for classification and `r2` for regression.""",
                    "anyOf": [
                        {
                            "description": "Custom scorer object, see https://scikit-learn.org/stable/modules/model_evaluation.html",
                            "not": {"type": "string"},
                        },
                        {
                            "description": "Known scorer for classification task.",
                            "enum": [
                                "accuracy",
                                "explained_variance",
                                "max_error",
                                "roc_auc",
                                "roc_auc_ovr",
                                "roc_auc_ovo",
                                "roc_auc_ovr_weighted",
                                "roc_auc_ovo_weighted",
                                "balanced_accuracy",
                                "average_precision",
                                "neg_log_loss",
                                "neg_brier_score",
                            ],
                        },
                        {
                            "description": "Known scorer for regression task.",
                            "enum": [
                                "r2",
                                "neg_mean_squared_error",
                                "neg_mean_absolute_error",
                                "neg_root_mean_squared_error",
                                "neg_mean_squared_log_error",
                                "neg_median_absolute_error",
                            ],
                        },
                    ],
                    "default": None,
                },
                "n_jobs": {
                    "description": "Number of jobs to run in parallel.",
                    "anyOf": [
                        {
                            "description": "1 unless in joblib.parallel_backend context.",
                            "enum": [None],
                        },
                        {"description": "Use all processors.", "enum": [-1]},
                        {
                            "description": "Number of jobs to run in parallel.",
                            "type": "integer",
                            "minimum": 1,
                        },
                    ],
                    "default": None,
                },
                "lale_num_samples": {
                    "description": "How many samples to draw when discretizing a continuous hyperparameter.",
                    "anyOf": [
                        {"type": "integer", "minimum": 1},
                        {
                            "description": "lale.search.lale_grid_search_cv.DEFAULT_SAMPLES_PER_DISTRIBUTION",
                            "enum": [None],
                        },
                    ],
                    "default": None,
                },
                "lale_num_grids": {
                    "description": "How many top-level disjuncts to explore.",
                    "anyOf": [
                        {"description": "If not set, keep all grids.", "enum": [None]},
                        {
                            "description": "Fraction of grids to keep.",
                            "type": "number",
                            "minimum": 0.0,
                            "exclusiveMinimum": True,
                            "maximum": 1.0,
                            "exclusiveMaximum": True,
                        },
                        {
                            "description": "Number of grids to keep.",
                            "type": "integer",
                            "minimum": 1,
                        },
                    ],
                    "default": None,
                },
                "param_grid": {
                    "anyOf": [
                        {"enum": [None], "description": "Generated automatically."},
                        {
                            "description": "Dictionary of hyperparameter ranges in the grid."
                        },
                    ],
                    "default": None,
                },
                "pgo": {
                    "anyOf": [{"description": "lale.search.PGO"}, {"enum": [None]}],
                    "default": None,
                },
                "observer": {
                    "laleType": "Any",
                    "default": None,
                    "description": "a class or object with callbacks for observing the state of the optimization",
                },
            },
        }
    ]
}

_input_fit_schema = {
    "type": "object",
    "required": ["X", "y"],
    "properties": {"X": {}, "y": {}},
}

_input_predict_schema = {"type": "object", "required": ["X"], "properties": {"X": {}}}

_output_predict_schema: Dict[str, Any] = {}

_combined_schemas = {
    "description": """GridSearchCV_ performs an exhaustive search over a discretized space.

.. _GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.hyperopt_classifier.html",
    "import_from": "lale.lib.lale",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}

lale.docstrings.set_docstrings(GridSearchCVImpl, _combined_schemas)

GridSearchCV = lale.operators.make_operator(GridSearchCVImpl, _combined_schemas)
