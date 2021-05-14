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

import numpy as np

import lale.docstrings
import lale.helpers
import lale.lib.sklearn
import lale.operators
import lale.search.lale_grid_search_cv

from .observing import Observing

func_timeout_installed = False
try:
    from func_timeout import FunctionTimedOut, func_timeout

    func_timeout_installed = True
except ImportError:
    pass


class _HalvingGridSearchCVImpl:
    _best_estimator: Optional[lale.operators.TrainedOperator] = None

    def __init__(
        self,
        estimator=None,
        param_grid=None,
        factor=3,
        resource="n_strings",
        max_resources="auto",
        min_resources="exhaust",
        aggressive_elimination=False,
        scoring=None,
        refit=True,
        error_score=np.nan,
        return_train_score=False,
        random_state=None,
        n_jobs=None,
        verbose=0,
        cv=5,
        lale_num_samples=None,
        lale_num_grids=None,
        pgo=None,
        observer=None,
        max_opt_time=None,
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
            "factor": factor,
            "resource": resource,
            "max_resources": max_resources,
            "min_resources": min_resources,
            "aggressive_elimination": aggressive_elimination,
            "cv": cv,
            "scoring": scoring,
            "refit": refit,
            "error_score": error_score,
            "return_train_score": return_train_score,
            "random_state": random_state,
            "n_jobs": n_jobs,
            "verbose": verbose,
            "lale_num_samples": lale_num_samples,
            "lale_num_grids": lale_num_grids,
            "pgo": pgo,
            "hp_grid": param_grid,
            "observer": observer,
            "max_opt_time": max_opt_time,
        }

    def fit(self, X, y):
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
                # explicitly require this experimental feature
                from sklearn.experimental import enable_halving_search_cv  # noqa

                import sklearn.model_selection  # isort: skip

                self.grid = sklearn.model_selection.HalvingGridSearchCV(
                    observed_op,
                    hp_grid,
                    cv=self._hyperparams["cv"],
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
                            raise BaseException("HalvingGridSearchCV timed out.")
                    else:
                        raise ValueError(
                            f"""max_opt_time is set to {self._hyperparams["max_opt_time"]} but the Python package
                            required for timeouts is not installed. Please install `func_timeout` using `pip install func_timeout`
                            or set max_opt_time to None."""
                        )
                else:
                    self.grid.fit(X, y)
                be = self.grid.best_estimator_
            except BaseException as e:
                if obs is not None:
                    assert isinstance(observed_op, Observing)  # type: ignore
                    impl = observed_op.impl  # type: ignore
                    impl.failObserving("optimize", e)
                raise

            impl = getattr(be, "impl", None)
            if impl is not None:
                assert isinstance(be, Observing)  # type: ignore
                be = impl.getOp()
                if obs is not None:
                    obs_impl = observed_op._impl  # type: ignore

                    obs_impl.endObserving("optimize", best=be)
        else:
            assert isinstance(op, lale.operators.TrainableOperator)
            be = op

        self._best_estimator = be.fit(X, y)
        return self

    def predict(self, X):
        assert self._best_estimator is not None
        return self._best_estimator.predict(X)

    def get_pipeline(self, pipeline_name=None, astype="lale"):
        if pipeline_name is not None:
            raise NotImplementedError("Cannot get pipeline by name yet.")
        result = self._best_estimator
        if result is None or astype == "lale":
            return result
        assert astype == "sklearn", astype
        return result


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
                "max_opt_time",
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
                "factor": {
                    "description": """The `halving’ parameter, which determines the proportion of candidates
that are selected for each subsequent iteration. For example, factor=3 means that only one third of the candidates are selected.""",
                    "type": "number",
                    "minimum": 1,
                    "exclusiveMinimum": True,
                    "minimumForOptimizer": 2,
                    "maximumForOptimizer": 5,
                    "default": 3,
                },
                "resource": {
                    "description": """Defines the resource that increases with each iteration.
By default, the resource is the number of samples.
It can also be set to any parameter of the base estimator that accepts positive integer values, e.g. ‘n_iterations’ or ‘n_estimators’ for a gradient boosting estimator.""",
                    "type": "string",
                    "default": "n_samples",
                },
                "max_resources": {
                    "description": "The maximum amount of resource that any candidate is allowed to use for a given iteration.",
                    "anyOf": [
                        {"enum": ["auto"]},
                        {
                            "forOptimizer": False,
                            "type": "integer",
                            "minimum": 1,
                        },
                    ],
                    "default": "auto",
                },
                "min_resources": {
                    "description": "The minimum amount of resource that any candidate is allowed to use for a given iteration",
                    "anyOf": [
                        {
                            "description": "A heuristic that sets r0 to a small value",
                            "enum": ["smallest"],
                        },
                        {
                            "description": "Sets r0 such that the last iteration uses as much resources as possible",
                            "enum": ["exhaust"],
                        },
                        {
                            "forOptimizer": False,
                            "type": "integer",
                            "minimum": 1,
                        },
                    ],
                    "default": "exhaust",
                },
                "aggressive_elimination": {
                    "description": "Enable aggresive elimination when there aren't enough resources to reduce the remaining candidates to at most factor after the last iteration",
                    "type": "boolean",
                    "default": False,
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
                "refit": {
                    "description": "Refit an estimator using the best found parameters on the whole dataset.",
                    "type": "boolean",
                    "default": True,
                },
                "error_score": {
                    "description": "Value to assign to the score if an error occurs in estimator fitting.",
                    "anyOf": [
                        {"description": "Raise the error", "enum": ["raise"]},
                        {"enum": [np.nan]},
                        {"type": "number", "forOptimizer": False},
                    ],
                    "default": np.nan,
                },
                "return_train_score": {
                    "description": "Include training scores",
                    "type": "boolean",
                    "default": False,
                },
                "random_state": {
                    "description": "Pseudo random number generator state used for subsampling the dataset when resources != 'n_samples'. Ignored otherwise.",
                    "anyOf": [
                        {
                            "description": "RandomState used by np.random",
                            "enum": [None],
                        },
                        {
                            "description": "Use the provided random state, only affecting other users of that same random state instance.",
                            "laleType": "numpy.random.RandomState",
                        },
                        {"description": "Explicit seed.", "type": "integer"},
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
                "verbose": {
                    "description": "Controls the verbosity: the higher, the more messages.",
                    "type": "integer",
                    "minimum": 0,
                    "default": 0,
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
                "max_opt_time": {
                    "description": "Maximum amout of time in seconds for the optimization.",
                    "anyOf": [
                        {"type": "number", "minimum": 0.0},
                        {"description": "No runtime bound.", "enum": [None]},
                    ],
                    "default": None,
                },
            },
        },
        {
            "description": "max_resources is set to 'auto' if and only if resource is set to 'n_samples'"
            "penalty with the liblinear solver.",
            "oneOf": [
                {"type": "object", "properties": {"resource": {"enum": ["n_samples"]}}},
                {
                    "type": "object",
                    "properties": {
                        "max_resources": {"not": {"enum": ["auto"]}},
                    },
                },
            ],
        },
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

.. _GridSearchCV: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingGridSearchCV.html""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.halving_grid_search_cv.html",
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

HalvingGridSearchCV = lale.operators.make_operator(
    _HalvingGridSearchCVImpl, _combined_schemas
)

lale.docstrings.set_docstrings(HalvingGridSearchCV)
