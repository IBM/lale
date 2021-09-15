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

import logging
import time
import traceback

import numpy as np
from sklearn.metrics import check_scoring, log_loss
from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import check_cv

import lale.docstrings
import lale.helpers
import lale.operators
import lale.sklearn_compat
from lale.helpers import cross_val_score_track_trials
from lale.lib.sklearn import LogisticRegression

from ._common_schemas import (
    schema_best_score_single,
    schema_cv,
    schema_estimator,
    schema_max_opt_time,
    schema_scoring_single,
)

try:
    # Import ConfigSpace and different types of parameters
    from smac.configspace import ConfigurationSpace

    # Import SMAC-utilities
    from smac.facade.smac_facade import SMAC as orig_SMAC
    from smac.scenario.scenario import Scenario
    from smac.tae.execute_ta_run import BudgetExhaustedException

    from lale.search.lale_smac import (
        get_smac_space,
        lale_op_smac_tae,
        lale_trainable_op_from_config,
    )

    smac_installed = True
except ImportError:
    smac_installed = False

logger = logging.getLogger(__name__)


class _SMACImpl:
    def __init__(
        self,
        *,
        estimator=None,
        scoring=None,
        best_score=0.0,
        cv=5,
        handle_cv_failure=False,
        max_evals=50,
        max_opt_time=None,
        lale_num_grids=None,
    ):
        assert smac_installed, """Your Python environment does not have smac installed. You can install it with
    pip install smac<=0.10.0
or with
    pip install 'lale[full]'"""
        self.max_evals = max_evals
        if estimator is None:
            self.estimator = LogisticRegression()
        else:
            self.estimator = estimator

        self.scoring = scoring
        if self.scoring is None:
            is_clf = self.estimator.is_classifier()
            if is_clf:
                self.scoring = "accuracy"
            else:
                self.scoring = "r2"

        self.best_score = best_score
        self.handle_cv_failure = handle_cv_failure
        self.cv = cv
        self.max_opt_time = max_opt_time
        self.lale_num_grids = lale_num_grids
        self.trials = None

    def fit(self, X_train, y_train, **fit_params):
        data_schema = lale.helpers.fold_schema(
            X_train, y_train, self.cv, self.estimator.is_classifier()
        )
        self.search_space: ConfigurationSpace = get_smac_space(
            self.estimator, lale_num_grids=self.lale_num_grids, data_schema=data_schema
        )
        # Scenario object
        scenario_options = {
            "run_obj": "quality",  # optimize quality (alternatively runtime)
            "runcount-limit": self.max_evals,  # maximum function evaluations
            "cs": self.search_space,  # configuration space
            "deterministic": "true",
            "abort_on_first_run_crash": False,
        }
        if self.max_opt_time is not None:
            scenario_options["wallclock_limit"] = self.max_opt_time
        self.scenario = Scenario(scenario_options)

        self.cv = check_cv(
            self.cv, y=y_train, classifier=self.estimator.is_classifier()
        )

        def smac_train_test(trainable, X_train, y_train):
            try:
                cv_score, logloss, execution_time = cross_val_score_track_trials(
                    trainable, X_train, y_train, cv=self.cv, scoring=self.scoring
                )
                logger.debug("Successful trial of SMAC")
            except BaseException as e:
                # If there is any error in cross validation, use the score based on a random train-test split as the evaluation criterion
                if self.handle_cv_failure:
                    (
                        X_train_part,
                        X_validation,
                        y_train_part,
                        y_validation,
                    ) = train_test_split(X_train, y_train, test_size=0.20)
                    start = time.time()
                    trained = trainable.fit(X_train_part, y_train_part, **fit_params)
                    scorer = check_scoring(trainable, scoring=self.scoring)
                    cv_score = scorer(trained, X_validation, y_validation)
                    execution_time = time.time() - start
                    y_pred_proba = trained.predict_proba(X_validation)
                    try:
                        logloss = log_loss(y_true=y_validation, y_pred=y_pred_proba)
                    except BaseException:
                        logloss = 0
                        logger.debug("Warning, log loss cannot be computed")
                else:
                    logger.debug(
                        "Error {} with pipeline:{}".format(e, trainable.to_json())
                    )
                    raise e
            return cv_score, logloss, execution_time

        def f(trainable):
            return_dict = {}
            try:
                score, logloss, execution_time = smac_train_test(
                    trainable, X_train=X_train, y_train=y_train
                )
                return_dict = {
                    "loss": self.best_score - score,
                    "time": execution_time,
                    "log_loss": logloss,
                }
            except BaseException as e:
                logger.warning(
                    f"Exception caught in SMACCV:{type(e)}, {traceback.format_exc()}, SMAC will set a cost_for_crash to MAXINT."
                )
                raise e
            return return_dict["loss"]

        try:
            smac = orig_SMAC(
                scenario=self.scenario,
                rng=np.random.RandomState(42),
                tae_runner=lale_op_smac_tae(self.estimator, f),
            )
            incumbent = smac.optimize()
            self.trials = smac.get_runhistory()
            trainable = lale_trainable_op_from_config(self.estimator, incumbent)
            # get the trainable corresponding to the best params and train it on the entire training dataset.
            trained = trainable.fit(X_train, y_train, **fit_params)
            self._best_estimator = trained
        except BudgetExhaustedException:
            logger.warning(
                "Maximum alloted optimization time exceeded. Optimization exited prematurely"
            )
        except BaseException as e:
            logger.warning("Error during optimization: {}".format(e))
            self._best_estimator = None

        return self

    def predict(self, X_eval, **predict_params):
        import warnings

        warnings.filterwarnings("ignore")
        trained = self._best_estimator
        if trained is None:
            logger.warning(
                "Could not get trained best estimator when predicting using SMACCV:{}, the error is"
            )
            return None

        try:
            predictions = trained.predict(X_eval, **predict_params)
        except ValueError as e:
            logger.warning(
                "ValueError in predicting using SMACCV:{}, the error is:{}".format(
                    trained, e
                )
            )
            predictions = None

        return predictions

    def get_trials(self):
        """Returns the trials i.e. RunHistory object.

        Returns
        -------
        smac.runhistory.runhistory.RunHistory
            RunHistory of all the trials executed during the optimization i.e. fit method of SMACCV.
        """
        return self.trials

    def get_pipeline(self, pipeline_name=None, astype="lale"):
        if pipeline_name is not None:
            raise NotImplementedError("Cannot get pipeline by name yet.")
        result = getattr(self, "_best_estimator", None)
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
                "max_evals",
                "cv",
                "handle_cv_failure",
                "max_opt_time",
                "lale_num_grids",
            ],
            "relevantToOptimizer": ["estimator"],
            "additionalProperties": False,
            "properties": {
                "estimator": schema_estimator,
                "scoring": schema_scoring_single,
                "best_score": schema_best_score_single,
                "cv": schema_cv,
                "handle_cv_failure": {
                    "description": """How to deal with cross validation failure for a trial.

If True, continue the trial by doing a 80-20 percent train-validation
split of the dataset input to fit and report the score on the
validation part. If False, terminate the trial with FAIL status.""",
                    "type": "boolean",
                    "default": False,
                },
                "max_evals": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 50,
                    "description": "Number of trials of SMAC search i.e. runcount_limit of SMAC.",
                },
                "max_opt_time": schema_max_opt_time,
                "lale_num_grids": {
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
            },
        }
    ]
}

_input_fit_schema = {
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "array", "items": {"type": ["number", "string"]}},
                    {"type": "string"},
                ]
            },
        },
        "y": {"type": "array", "items": {"type": "number"}},
    },
}

_input_predict_schema = {
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "anyOf": [
                    {"type": "array", "items": {"type": ["number", "string"]}},
                    {"type": "string"},
                ]
            },
        }
    },
}

_output_predict_schema = {"type": "array", "items": {"type": "number"}}

_combined_schemas = {
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.smac.html",
    "import_from": "lale.lib.lale",
    "description": """SMAC_, the optimizer used inside auto-weka and auto-sklearn.

.. _SMAC: https://github.com/automl/SMAC3

Examples
--------
>>> from sklearn.metrics import make_scorer, f1_score, accuracy_score
>>> lr = LogisticRegression()
>>> clf = SMAC(estimator=lr, scoring='accuracy', cv=5)
>>> from sklearn import datasets
>>> diabetes = datasets.load_diabetes()
>>> X = diabetes.data[:150]
>>> y = diabetes.target[:150]
>>> trained = clf.fit(X, y)
>>> predictions = trained.predict(X)

Other scoring metrics:

>>> clf = SMAC(estimator=lr, scoring=make_scorer(f1_score, average='macro'), cv=3, max_evals=2)
""",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}


SMAC = lale.operators.make_operator(_SMACImpl, _combined_schemas)

lale.docstrings.set_docstrings(SMAC)
