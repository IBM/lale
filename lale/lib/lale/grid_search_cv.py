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

from typing import Any, Dict, Optional

import lale.docstrings
import lale.helpers
import lale.lib.sklearn
import lale.operators
import lale.search.lale_grid_search_cv
import lale.sklearn_compat
from lale.lib._common_schemas import (
    schema_estimator,
    schema_max_opt_time,
    schema_scoring_single,
    schema_simple_cv,
)

from .observing import Observing

func_timeout_installed = False
try:
    from func_timeout import FunctionTimedOut, func_timeout

    func_timeout_installed = True
except ImportError:
    pass


class _GridSearchCVImpl:
    _best_estimator: Optional[lale.operators.TrainedOperator] = None

    def __init__(
        self,
        *,
        estimator=None,
        scoring=None,
        cv=5,
        verbose=0,
        n_jobs=None,
        lale_num_samples=None,
        lale_num_grids=None,
        param_grid=None,
        pgo=None,
        observer=None,
        max_opt_time=None,
    ):
        if observer is not None and isinstance(observer, type):
            # if we are given a class name, instantiate it
            observer = observer()
        if scoring is None:
            if estimator is None:
                is_clf = True  # Since we will use LogisticRegression
            else:
                is_clf = estimator.is_classifier()
            if is_clf:
                scoring = "accuracy"
            else:
                scoring = "r2"

        self._hyperparams = {
            "estimator": estimator,
            "cv": cv,
            "verbose": verbose,
            "scoring": scoring,
            "n_jobs": n_jobs,
            "lale_num_samples": lale_num_samples,
            "lale_num_grids": lale_num_grids,
            "pgo": pgo,
            "hp_grid": param_grid,
            "observer": observer,
            "max_opt_time": max_opt_time,
        }

    def fit(self, X, y, **fit_params):
        if self._hyperparams["estimator"] is None:
            op = lale.lib.sklearn.LogisticRegression
        else:
            op = self._hyperparams["estimator"]

        observed_op = op
        obs = self._hyperparams["observer"]
        # We always create an observer.
        # Otherwise, we can have a problem with PlannedOperators
        # (that are not trainable):
        # GridSearchCV checks if a fit method is present before
        # configuring the operator, and our planned operators
        # don't have a fit method
        # Observing always has a fit method, and so solves this problem.
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
        else:
            # if hp_grid is specified manually, we need to add a level of nesting
            # since we are wrapping it in an observer
            if isinstance(hp_grid, list):
                hp_grid = lale.helpers.nest_all_HPparams("op", hp_grid)
            else:
                assert isinstance(hp_grid, dict)
                hp_grid = lale.helpers.nest_HPparams("op", hp_grid)

        if not hp_grid and isinstance(op, lale.operators.IndividualOp):
            hp_grid = [
                lale.search.lale_grid_search_cv.get_defaults_as_param_grid(observed_op)  # type: ignore
            ]
        be: lale.operators.TrainableOperator
        if hp_grid:
            if obs is not None:
                impl = observed_op._impl  # type: ignore
                impl.startObserving(
                    "optimize",
                    hp_grid=hp_grid,
                    op=op,
                    num_samples=self._hyperparams["lale_num_samples"],
                    num_grids=self._hyperparams["lale_num_grids"],
                    pgo=self._hyperparams["pgo"],
                )
            try:
                self.grid = lale.search.lale_grid_search_cv.get_lale_gridsearchcv_op(
                    observed_op,
                    hp_grid,
                    cv=self._hyperparams["cv"],
                    verbose=self._hyperparams["verbose"],
                    scoring=self._hyperparams["scoring"],
                    n_jobs=self._hyperparams["n_jobs"],
                )
                if self._hyperparams["max_opt_time"] is not None:
                    if func_timeout_installed:
                        try:
                            func_timeout(
                                self._hyperparams["max_opt_time"], self.grid.fit, (X, y)
                            )
                        except FunctionTimedOut:
                            raise BaseException("GridSearchCV timed out.")
                    else:
                        raise ValueError(
                            f"""max_opt_time is set to {self._hyperparams["max_opt_time"]} but the Python package
                            required for timeouts is not installed. Please install `func_timeout` using `pip install func_timeout`
                            or set max_opt_time to None."""
                        )
                else:
                    self.grid.fit(X, y, **fit_params)
                be = self.grid.best_estimator_
            except BaseException as e:
                if obs is not None:
                    assert isinstance(obs, Observing)  # type: ignore
                    impl = observed_op.shallow_impl  # type: ignore
                    impl.failObserving("optimize", e)
                raise

            impl = None
            if isinstance(be, lale.operators.Operator):
                impl = be._impl_instance()
            if impl is not None:
                assert isinstance(be, Observing)  # type: ignore
                be = impl.getOp()
                if obs is not None:
                    obs_impl = observed_op._impl  # type: ignore

                    obs_impl.endObserving("optimize", best=be)
        else:
            assert isinstance(op, lale.operators.TrainableOperator)
            be = op

        self._best_estimator = be.fit(X, y, **fit_params)
        return self

    def predict(self, X, **predict_params):
        assert self._best_estimator is not None
        return self._best_estimator.predict(X, **predict_params)

    def get_pipeline(self, pipeline_name=None, astype="lale"):
        if pipeline_name is not None:
            raise NotImplementedError("Cannot get pipeline by name yet.")
        result = self._best_estimator
        if result is None or astype == "lale":
            return result
        assert astype == "sklearn", astype
        # TODO: should this try and return an actual sklearn pipeline?
        return result


_hyperparams_schema = {
    "allOf": [
        {
            "type": "object",
            "required": [
                "estimator",
                "cv",
                "verbose",
                "scoring",
                "n_jobs",
                "lale_num_samples",
                "lale_num_grids",
                "pgo",
                "max_opt_time",
            ],
            "relevantToOptimizer": ["estimator"],
            "additionalProperties": False,
            "properties": {
                "estimator": schema_estimator,
                "scoring": schema_scoring_single,
                "cv": schema_simple_cv,
                "verbose": {
                    "description": "Controls the verbosity: the higher, the more messages.",
                    "type": "integer",
                    "default": 0,
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
                "max_opt_time": schema_max_opt_time,
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


GridSearchCV = lale.operators.make_operator(_GridSearchCVImpl, _combined_schemas)

lale.docstrings.set_docstrings(GridSearchCV)
