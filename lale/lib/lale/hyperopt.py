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

import copy
import logging
import multiprocessing
import sys
import time
import traceback
import warnings
from typing import Any, Dict, Optional

import hyperopt
import numpy as np
import pandas as pd
from hyperopt.exceptions import AllTrialsFailed
from sklearn.metrics import check_scoring, log_loss
from sklearn.model_selection import train_test_split
from sklearn.model_selection._split import check_cv

import lale.docstrings
import lale.helpers
import lale.operators
from lale.helpers import (
    create_instance_from_hyperopt_search_space,
    cross_val_score_track_trials,
)
from lale.lib.sklearn import LogisticRegression
from lale.search.op2hp import hyperopt_search_space
from lale.search.PGO import PGO

SEED = 42
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)


class HyperoptImpl:
    def __init__(
        self,
        estimator=None,
        max_evals=50,
        frac_evals_with_defaults=0,
        algo="tpe",
        cv=5,
        handle_cv_failure=False,
        scoring=None,
        best_score=0.0,
        max_opt_time=None,
        max_eval_time=None,
        pgo: Optional[PGO] = None,
        show_progressbar=True,
        args_to_scorer=None,
        verbose=False,
    ):
        self.max_evals = max_evals
        if estimator is None:
            self.estimator = LogisticRegression()
        else:
            self.estimator = estimator
        if frac_evals_with_defaults > 0:
            self.evals_with_defaults = int(frac_evals_with_defaults * max_evals)
        else:
            self.evals_with_defaults = 0
        self.algo = algo
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
        self._trials = hyperopt.Trials()
        self._default_trials = hyperopt.Trials()
        self.max_opt_time = max_opt_time
        self.max_eval_time = max_eval_time
        self.pgo = pgo
        self.show_progressbar = show_progressbar
        if args_to_scorer is not None:
            self.args_to_scorer = args_to_scorer
        else:
            self.args_to_scorer = {}
        self.verbose = verbose

    def _summarize_statuses(self):
        status_list = self._trials.statuses()
        status_hist = {}
        for status in status_list:
            status_hist[status] = 1 + status_hist.get(status, 0)
        if hyperopt.STATUS_FAIL in status_hist:
            print(
                f"{status_hist[hyperopt.STATUS_FAIL]} out of {len(status_list)} trials failed, call summary() for details."
            )
            if not self.verbose:
                print("Run with verbose=True to see per-trial exceptions.")

    def fit(self, X_train, y_train):
        opt_start_time = time.time()
        is_clf = self.estimator.is_classifier()
        self.cv = check_cv(self.cv, y=y_train, classifier=is_clf)
        data_schema = lale.helpers.fold_schema(X_train, y_train, self.cv, is_clf)
        self.search_space = hyperopt.hp.choice(
            "meta_model",
            [
                hyperopt_search_space(
                    self.estimator, pgo=self.pgo, data_schema=data_schema
                )
            ],
        )
        # Create a search space with default hyperparameters for all trainable parts of the pipeline.
        # This search space is used for `frac_evals_with_defaults` fraction of the total trials.
        try:
            self.search_space_with_defaults = hyperopt.hp.choice(
                "meta_model",
                [
                    hyperopt_search_space(
                        self.estimator.freeze_trainable(),
                        pgo=self.pgo,
                        data_schema=data_schema,
                    )
                ],
            )
        except Exception:
            logger.warning(
                "Exception caught during generation of default search space, setting frac_evals_with_defaults to zero."
            )
            self.evals_with_defaults = 0

        def hyperopt_train_test(params, X_train, y_train):
            warnings.filterwarnings("ignore")

            trainable = create_instance_from_hyperopt_search_space(
                self.estimator, params
            )
            try:
                cv_score, logloss, execution_time = cross_val_score_track_trials(
                    trainable,
                    X_train,
                    y_train,
                    cv=self.cv,
                    scoring=self.scoring,
                    args_to_scorer=self.args_to_scorer,
                )
                logger.debug(
                    "Successful trial of hyperopt with hyperparameters:{}".format(
                        params
                    )
                )
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
                    trained = trainable.fit(X_train_part, y_train_part)
                    scorer = check_scoring(trainable, scoring=self.scoring)
                    cv_score = scorer(
                        trained, X_validation, y_validation, **self.args_to_scorer
                    )
                    execution_time = time.time() - start
                    y_pred_proba = trained.predict_proba(X_validation)
                    try:
                        logloss = log_loss(y_true=y_validation, y_pred=y_pred_proba)
                    except BaseException:
                        logloss = 0
                        logger.debug("Warning, log loss cannot be computed")
                else:
                    logger.debug(e)
                    logger.debug(
                        "Error {} with pipeline:{}".format(e, trainable.to_json())
                    )
                    raise e
            return cv_score, logloss, execution_time

        def merge_trials(trials1, trials2):
            max_tid = max([trial["tid"] for trial in trials1.trials])

            for trial in trials2:
                tid = trial["tid"] + max_tid + 1
                hyperopt_trial = hyperopt.Trials().new_trial_docs(
                    tids=[None], specs=[None], results=[None], miscs=[None]
                )
                hyperopt_trial[0] = trial
                hyperopt_trial[0]["tid"] = tid
                hyperopt_trial[0]["misc"]["tid"] = tid
                for key in hyperopt_trial[0]["misc"]["idxs"].keys():
                    hyperopt_trial[0]["misc"]["idxs"][key] = [tid]
                trials1.insert_trial_docs(hyperopt_trial)
                trials1.refresh()
            return trials1

        def proc_train_test(params, X_train, y_train, return_dict):
            return_dict["params"] = copy.deepcopy(params)
            try:
                score, logloss, execution_time = hyperopt_train_test(
                    params, X_train=X_train, y_train=y_train
                )
                return_dict["loss"] = self.best_score - score
                return_dict["time"] = execution_time
                return_dict["log_loss"] = logloss
                return_dict["status"] = hyperopt.STATUS_OK
            except BaseException as e:
                exception_type = f"{type(e).__module__}.{type(e).__name__}"
                try:
                    trainable = create_instance_from_hyperopt_search_space(
                        self.estimator, params
                    )
                    trial_info = (
                        f'pipeline: """{trainable.pretty_print(show_imports=False)}"""'
                    )
                except BaseException:
                    trial_info = f"hyperparams: {params}"
                error_msg = f"Exception caught in Hyperopt: {exception_type}, {traceback.format_exc()}with {trial_info}"
                logger.warning(error_msg + ", setting status to FAIL")
                return_dict["status"] = hyperopt.STATUS_FAIL
                return_dict["error_msg"] = error_msg
                if self.verbose:
                    print(return_dict["error_msg"])

        def get_final_trained_estimator(params, X_train, y_train):
            warnings.filterwarnings("ignore")
            trainable = create_instance_from_hyperopt_search_space(
                self.estimator, params
            )
            trained = trainable.fit(X_train, y_train)
            return trained

        def f(params):
            current_time = time.time()
            if (self.max_opt_time is not None) and (
                (current_time - opt_start_time) > self.max_opt_time
            ):
                # if max optimization time set, and we have crossed it, exit optimization completely
                sys.exit(0)
            if self.max_eval_time:
                # Run hyperopt in a subprocess that can be interupted
                manager = multiprocessing.Manager()
                proc_dict = manager.dict()
                p = multiprocessing.Process(
                    target=proc_train_test, args=(params, X_train, y_train, proc_dict)
                )
                p.start()
                p.join(self.max_eval_time)
                if p.is_alive():
                    p.terminate()
                    p.join()
                    logger.warning(
                        f"Maximum alloted evaluation time exceeded. with hyperparams: {params}, setting status to FAIL"
                    )
                    proc_dict["status"] = hyperopt.STATUS_FAIL
                if "status" not in proc_dict:
                    logger.warning("Corrupted results, setting status to FAIL")
                    proc_dict["status"] = hyperopt.STATUS_FAIL
            else:
                proc_dict = {}
                proc_train_test(params, X_train, y_train, proc_dict)
            return proc_dict

        algo = getattr(hyperopt, self.algo)
        # Search in the search space with defaults
        if self.evals_with_defaults > 0:
            try:
                hyperopt.fmin(
                    f,
                    self.search_space_with_defaults,
                    algo=algo.suggest,
                    max_evals=self.evals_with_defaults,
                    trials=self._default_trials,
                    rstate=np.random.RandomState(SEED),
                    show_progressbar=self.show_progressbar,
                )
            except SystemExit:
                logger.warning(
                    "Maximum alloted optimization time exceeded. Optimization exited prematurely"
                )
            except AllTrialsFailed:
                self._best_estimator = None
                if hyperopt.STATUS_OK not in self._trials.statuses():
                    raise ValueError(
                        "Error from hyperopt, none of the trials succeeded."
                    )

        try:
            hyperopt.fmin(
                f,
                self.search_space,
                algo=algo.suggest,
                max_evals=self.max_evals - self.evals_with_defaults,
                trials=self._trials,
                rstate=np.random.RandomState(SEED),
                show_progressbar=self.show_progressbar,
            )
        except SystemExit:
            logger.warning(
                "Maximum alloted optimization time exceeded. Optimization exited prematurely"
            )
        except AllTrialsFailed:
            self._best_estimator = None
            if hyperopt.STATUS_OK not in self._trials.statuses():
                self._summarize_statuses()
                raise ValueError("Error from hyperopt, none of the trials succeeded.")
        self._trials = merge_trials(self._trials, self._default_trials)
        if self.show_progressbar:
            self._summarize_statuses()
        try:
            best_trial = self._trials.best_trial
            val_loss = self._trials.best_trial["result"]["loss"]
            if len(self._default_trials) > 0:
                default_val_loss = self._default_trials.best_trial["result"]["loss"]
                if default_val_loss < val_loss:
                    best_trial = self._default_trials.best_trial
            best_params = best_trial["result"]["params"]
            logger.info(
                "best score: {:.1%}\nbest hyperparams found using {} hyperopt trials: {}".format(
                    self.best_score - self._trials.average_best_error(),
                    self.max_evals,
                    best_params,
                )
            )
            trained = get_final_trained_estimator(best_params, X_train, y_train)
            self._best_estimator = trained
        except BaseException as e:
            logger.warning(
                "Unable to extract the best parameters from optimization, the error: {}".format(
                    e
                )
            )
            self._best_estimator = None

        return self

    def predict(self, X_eval):
        import warnings

        warnings.filterwarnings("ignore")
        if self._best_estimator is None:
            raise ValueError(
                "Can not predict as the best estimator is None. Either an attempt to call `predict` "
                "before calling `fit` or all the trials during `fit` failed."
            )
        trained = self._best_estimator
        try:
            predictions = trained.predict(X_eval)
        except ValueError as e:
            logger.warning(
                "ValueError in predicting using Hyperopt:{}, the error is:{}".format(
                    trained, e
                )
            )
            predictions = None

        return predictions

    def summary(self):
        """Table summarizing the trial results (ID, loss, time, log_loss, status).

Returns
-------
result : DataFrame"""

        def make_record(trial_dict):
            return {
                "name": f'p{trial_dict["tid"]}',
                "tid": trial_dict["tid"],
                "loss": trial_dict["result"].get("loss", float("nan")),
                "time": trial_dict["result"].get("time", float("nan")),
                "log_loss": trial_dict["result"].get("log_loss", float("nan")),
                "status": trial_dict["result"]["status"],
            }

        records = [make_record(td) for td in self._trials.trials]
        result = pd.DataFrame.from_records(records, index="name")
        return result

    def get_pipeline(self, pipeline_name=None, astype="lale"):
        """Retrieve one of the trials.

Parameters
----------
pipeline_name : union type, default None

    - string
        Key for table returned by summary(), return a trainable pipeline.

    - None
        When not specified, return the best trained pipeline found.

astype : 'lale' or 'sklearn', default 'lale'
    Type of resulting pipeline.

Returns
-------
result : Trained operator if best, trainable operator otherwise.
"""
        best_name = None
        if self._best_estimator is not None:
            best_name = f'p{self._trials.best_trial["tid"]}'
        if pipeline_name is None:
            pipeline_name = best_name
        if pipeline_name == best_name:
            result = getattr(self, "_best_estimator", None)
        else:
            tid = int(pipeline_name[1:])
            params = self._trials.trials[tid]["result"]["params"]
            result = create_instance_from_hyperopt_search_space(self.estimator, params)
        if result is None or astype == "lale":
            return result
        assert astype == "sklearn", astype
        return result.export_to_sklearn_pipeline()


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
                "pgo",
                "show_progressbar",
            ],
            "relevantToOptimizer": ["estimator", "max_evals", "cv"],
            "additionalProperties": False,
            "properties": {
                "estimator": {
                    "description": "Planned Lale individual operator or pipeline,\nby default LogisticRegression.",
                    "anyOf": [
                        {"laleType": "operator", "not": {"enum": [None]}},
                        {"enum": [None]},
                    ],
                    "default": None,
                },
                "algo": {
                    "description": """Algorithm for searching the space.

Use 'rand' for random search,
'tpe' for tree of parzen estimators,
'atpe' for adaptive TPE,
'anneal' for variant on random search that takes some advantage of a smooth response surface.""",
                    "enum": ["rand", "tpe", "atpe", "anneal"],
                    "default": "tpe",
                },
                "max_evals": {
                    "description": "Number of trials of Hyperopt search.",
                    "type": "integer",
                    "minimum": 1,
                    "default": 50,
                },
                "frac_evals_with_defaults": {
                    "description": """Sometimes, using default values of hyperparameters works quite well.
This value would allow a fraction of the trials to use default values. Hyperopt searches the entire search space
for (1-frac_evals_with_defaults) fraction of max_evals.""",
                    "type": "number",
                    "minimum": 0.0,
                    "default": 0,
                },
                "cv": {
                    "description": """Cross-validation as integer or as object that has a split function.

The fit method performs cross validation on the input dataset for per
trial, and uses the mean cross validation performance for optimization.
This behavior is also impacted by handle_cv_failure flag.

If integer: number of folds in sklearn.model_selection.StratifiedKFold.

If object with split function: generator yielding (train, test) splits
as arrays of indices. Can use any of the iterators from
https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators.""",
                    "anyOf": [
                        {"type": "integer"},
                        {"laleType": "Any", "forOptimizer": False},
                    ],
                    "minimum": 1,
                    "default": 5,
                },
                "handle_cv_failure": {
                    "description": """How to deal with cross validation failure for a trial.

If True, continue the trial by doing a 80-20 percent train-validation
split of the dataset input to fit and report the score on the
validation part. If False, terminate the trial with FAIL status.""",
                    "type": "boolean",
                    "default": False,
                },
                "scoring": {
                    "description": """Scorer object, or known scorer named by string.
Default of None translates to `accuracy` for classification and `r2` for regression.""",
                    "anyOf": [
                        {
                            "description": """Custom scorer object created with `make_scorer`_.

The argument to make_scorer can be one of scikit-learn's metrics_,
or it can be a user-written Python function to create a completely
custom scorer objects, following the `model_evaluation`_ example.
The metric has to return a scalar value. Note that scikit-learns's
scorer object always returns values such that higher score is
better. Since Hyperopt solves a minimization problem, we pass
(best_score - score) to Hyperopt.

.. _`make_scorer`: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer.
.. _metrics: https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
.. _`model_evaluation`: https://scikit-learn.org/stable/modules/model_evaluation.html
""",
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
                "best_score": {
                    "description": """The best score for the specified scorer.

This allows us to return a loss to hyperopt that is >=0,
where zero is the best loss.""",
                    "type": "number",
                    "default": 0.0,
                },
                "max_opt_time": {
                    "description": "Maximum amout of time in seconds for the optimization.",
                    "anyOf": [
                        {"type": "number", "minimum": 0.0},
                        {"description": "No runtime bound.", "enum": [None]},
                    ],
                    "default": None,
                },
                "max_eval_time": {
                    "description": "Maximum amout of time in seconds for each evaluation.",
                    "anyOf": [
                        {"type": "number", "minimum": 0.0},
                        {"description": "No runtime bound.", "enum": [None]},
                    ],
                    "default": None,
                },
                "pgo": {
                    "anyOf": [{"description": "lale.search.PGO"}, {"enum": [None]}],
                    "default": None,
                },
                "show_progressbar": {
                    "description": "Display progress bar during optimization.",
                    "type": "boolean",
                    "default": True,
                },
                "args_to_scorer": {
                    "anyOf": [
                        {"type": "object"},  # Python dictionary
                        {"enum": [None]},
                    ],
                    "description": """A dictionary of additional keyword arguments to pass to the scorer.
                Used for cases where the scorer has a signature such as ``scorer(estimator, X, y, **kwargs)``.
                """,
                    "default": None,
                },
                "verbose": {
                    "description": """Whether to print errors from each of the trials if any.
This is also logged using logger.warning.""",
                    "type": "boolean",
                    "default": False,
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
    "description": """Hyperopt_ is a popular open-source Bayesian optimizer.

.. _Hyperopt: https://github.com/hyperopt/hyperopt

Examples
--------
>>> from lale.lib.sklearn import LogisticRegression as LR
>>> clf = Hyperopt(estimator=LR, cv=3, max_evals=5)
>>> from sklearn import datasets
>>> diabetes = datasets.load_diabetes()
>>> X = diabetes.data[:150]
>>> y = diabetes.target[:150]
>>> trained = clf.fit(X, y)
>>> predictions = trained.predict(X)

Other scoring metrics:

>>> from sklearn.metrics import make_scorer, f1_score
>>> clf = Hyperopt(estimator=LR,
...    scoring=make_scorer(f1_score, average='macro'), cv=3, max_evals=5)
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.hyperopt.html",
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

lale.docstrings.set_docstrings(HyperoptImpl, _combined_schemas)

Hyperopt = lale.operators.make_operator(HyperoptImpl, _combined_schemas)
