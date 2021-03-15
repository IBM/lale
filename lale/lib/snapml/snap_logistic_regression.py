# Copyright 2019,2021 IBM Corporation
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
try:
    import snapml  # type: ignore

    snapml_installed = True
except ImportError:
    snapml_installed = False

import lale.datasets.data_schemas
import lale.docstrings
import lale.operators


class _SnapLogisticRegressionImpl:
    def __init__(
        self,
        max_iter=1000,
        regularizer=1.0,
        device_ids=None,
        verbose=False,
        use_gpu=False,
        class_weight=None,
        dual=True,
        n_jobs=1,
        penalty="l2",
        tol=0.001,
        generate_training_history=None,
        privacy=False,
        eta=0.3,
        batch_size=100,
        privacy_epsilon=10,
        grad_clip=1,
        fit_intercept=False,
        intercept_scaling=1.0,
        normalize=False,
        kernel=None,
        gamma=1.0,
        n_components=100,
        random_state=None,
    ):

        assert (
            snapml_installed
        ), """Your Python environment does not have snapml installed. Install using: pip install snapml"""
        self._hyperparams = {
            "max_iter": max_iter,
            "regularizer": regularizer,
            "device_ids": device_ids,
            "verbose": verbose,
            "use_gpu": use_gpu,
            "class_weight": class_weight,
            "dual": dual,
            "n_jobs": n_jobs,
            "penalty": penalty,
            "tol": tol,
            "generate_training_history": generate_training_history,
            "privacy": privacy,
            "eta": eta,
            "batch_size": batch_size,
            "privacy_epsilon": privacy_epsilon,
            "grad_clip": grad_clip,
            "fit_intercept": fit_intercept,
            "intercept_scaling": intercept_scaling,
            "normalize": normalize,
            "kernel": kernel,
            "gamma": gamma,
            "n_components": n_components,
            "random_state": random_state,
        }
        modified_hps = {**self._hyperparams}
        if modified_hps["device_ids"] is None:
            modified_hps["device_ids"] = [0]  # TODO: support list as default
        self._wrapped_model = snapml.SnapLogisticRegression(**modified_hps)

    def fit(self, X, y, **fit_params):
        X = lale.datasets.data_schemas.strip_schema(X)
        y = lale.datasets.data_schemas.strip_schema(y)
        self._wrapped_model.fit(X, y, **fit_params)
        return self

    def predict(self, X, **predict_params):
        X = lale.datasets.data_schemas.strip_schema(X)
        return self._wrapped_model.predict(X, **predict_params)

    def predict_proba(self, X, **predict_proba_params):
        X = lale.datasets.data_schemas.strip_schema(X)
        return self._wrapped_model.predict_proba(X, **predict_proba_params)


_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "relevantToOptimizer": [
                "fit_intercept",
                "regularizer",
                "max_iter",
            ],
            "additionalProperties": False,
            "properties": {
                "max_iter": {
                    "type": "integer",
                    "minimum": 1,
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 1000,
                    "default": 100,
                    "description": "Maximum number of iterations used by the solver to converge.",
                },
                "regularizer": {
                    "type": "number",
                    "minimum": 0.0,
                    "default": 1.0,
                    "exclusiveMinimum": True,
                    "minimumForOptimizer": 1.0,
                    "maximumForOptimizer": 100.0,
                    "distribution": "uniform",
                    "description": "Larger regularization values imply stronger regularization.",
                },
                "use_gpu": {
                    "type": "boolean",
                    "default": False,
                    "description": "Use GPU Acceleration.",
                },
                "device_ids": {
                    "anyOf": [
                        {"description": "Use [0].", "enum": [None]},
                        {"type": "array", "items": {"type": "integer"}},
                    ],
                    "default": None,
                    "description": "Device IDs of the GPUs which will be used when GPU acceleration is enabled.",
                },
                "class_weight": {
                    "enum": ["balanced", None],
                    "default": None,
                    "description": "If set to 'balanced' samples weights will be applied to account for class imbalance, otherwise no sample weights will be used.",
                },
                "dual": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use dual formulation (rather than primal).",
                },
                "verbose": {
                    "type": "boolean",
                    "default": False,
                    "description": "If True, it prints the training cost, one per iteration. Warning: this will increase the training time. For performance evaluation, use verbose=False.",
                },
                "n_jobs": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "The number of threads used for running the training. The value of this parameter should be a multiple of 32 if the training is performed on GPU (use_gpu=True).",
                },
                "penalty": {
                    "enum": ["l1", "l2"],
                    "default": "l2",
                    "description": "The regularization / penalty type. Possible values are 'l2' for L2 regularization (LogisticRegression) or 'l1' for L1 regularization (SparseLogisticRegression). L1 regularization is possible only for the primal optimization problem (dual=False).",
                },
                "tol": {
                    "type": "number",
                    "minimum": 0.0,
                    "default": 0.001,
                    "exclusiveMinimum": True,
                    "description": "The tolerance parameter. Training will finish when maximum change in model coefficients is less than tol.",
                },
                "generate_training_history": {
                    "enum": ["summary", "full", None],
                    "default": None,
                    "description": "Determines the level of summary statistics that are generated during training.",
                },
                "privacy": {
                    "type": "boolean",
                    "default": False,
                    "description": "Train the model using a differentially private algorithm.",
                },
                "eta": {
                    "type": "number",
                    "minimum": 0.0,
                    "default": 0.3,
                    "exclusiveMinimum": True,
                    "description": "Learning rate for the differentially private training algorithm.",
                },
                "batch_size": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 100,
                    "description": "Mini-batch size for the differentially private training algorithm.",
                },
                "privacy_epsilon": {
                    "type": "number",
                    "minimum": 0.0,
                    "default": 10.0,
                    "exclusiveMinimum": True,
                    "description": "Target privacy gaurantee. Learned model will be (privacy_epsilon, 0.01)-private.",
                },
                "grad_clip": {
                    "type": "number",
                    "minimum": 0.0,
                    "default": 1.0,
                    "description": "Gradient clipping parameter for the differentially private training algorithm.",
                },
                "fit_intercept": {
                    "type": "boolean",
                    "default": True,
                    "description": "Add bias term -- note, may affect speed of convergence, especially for sparse datasets.",
                },
                "intercept_scaling": {
                    "type": "number",
                    "minimum": 0.0,
                    "default": 1.0,
                    "exclusiveMinimum": True,
                    "description": "Scaling of bias term. The inclusion of a bias term is implemented by appending an additional feature to the dataset. This feature has a constant value, that can be set using this parameter.",
                },
                "normalize": {
                    "type": "boolean",
                    "default": True,
                    "description": "Normalize rows of dataset (recommended for fast convergence).",
                },
                "kernel": {
                    "enum": ["rbf", "linear"],
                    "default": "linear",
                    "description": "Approximate feature map of a specified kernel function.",
                },
                "gamma": {
                    "type": "number",
                    "minimum": 0.0,
                    "default": 1.0,
                    "exclusiveMinimum": True,
                    "description": "Parameter of RBF kernel: exp(-gamma * x^2).",
                },
                "n_components": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 100,
                    "description": "Dimensionality of the feature space when approximating a kernel function.",
                },
                "random_state": {
                    "description": "Seed of pseudo-random number generator.",
                    "anyOf": [
                        {
                            "description": "RandomState used by np.random",
                            "enum": [None],
                        },
                        {"description": "Explicit seed.", "type": "integer"},
                    ],
                    "default": None,
                },
            },
        },
        {
            "description": "L1 regularization is supported only for primal optimization problems.",
            "anyOf": [
                {"type": "object", "properties": {"penalty": {"enum": ["l2"]}}},
                {"type": "object", "properties": {"dual": {"enum": [False]}}},
            ],
        },
        {
            "description": "Privacy only supported for primal objective functions.",
            "anyOf": [
                {"type": "object", "properties": {"privacy": {"enum": [False]}}},
                {"type": "object", "properties": {"dual": {"enum": [False]}}},
            ],
        },
        {
            "description": "Privacy only supported for L2-regularized objective functions.",
            "anyOf": [
                {"type": "object", "properties": {"privacy": {"enum": [False]}}},
                {"type": "object", "properties": {"penalty": {"enum": ["l2"]}}},
            ],
        },
        {
            "description": "Privacy not supported with fit_intercept=True.",
            "anyOf": [
                {"type": "object", "properties": {"privacy": {"enum": [False]}}},
                {"type": "object", "properties": {"fit_intercept": {"enum": [False]}}},
            ],
        },
    ],
}

_input_fit_schema = {
    "description": "Fit the model according to the given train dataset.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "description": "The outer array is over samples aka rows.",
            "items": {
                "type": "array",
                "description": "The inner array is over features aka columns.",
                "items": {"type": "number"},
            },
        },
        "y": {
            "description": "The classes.",
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
                {"type": "array", "items": {"type": "boolean"}},
            ],
        },
    },
}

_input_predict_schema = {
    "type": "object",
    "required": ["X"],
    "properties": {
        "X": {
            "type": "array",
            "description": "The outer array is over samples aka rows.",
            "items": {
                "type": "array",
                "description": "The inner array is over features aka columns.",
                "items": {"type": "number"},
            },
        },
        "n_jobs": {
            "type": "integer",
            "minimum": 0,
            "default": 0,
            "description": "Number of threads used to run inference. By default inference runs with maximum number of available threads.",
        },
    },
}

_output_predict_schema = {
    "description": "The predicted classes.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "string"}},
        {"type": "array", "items": {"type": "boolean"}},
    ],
}

_input_predict_proba_schema = {
    "type": "object",
    "properties": {
        "X": {
            "type": "array",
            "description": "The outer array is over samples aka rows.",
            "items": {
                "type": "array",
                "description": "The inner array is over features aka columns.",
                "items": {"type": "number"},
            },
        },
        "n_jobs": {
            "type": "integer",
            "minimum": 0,
            "default": 0,
            "description": "Number of threads used to run inference. By default inference runs with maximum number of available threads.",
        },
    },
}

_output_predict_proba_schema = {
    "type": "array",
    "description": "The outer array is over samples aka rows.",
    "items": {
        "type": "array",
        "description": "The inner array contains probabilities corresponding to each class.",
        "items": {"type": "number"},
    },
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Logistic Regression`_ from `Snap ML`_.

.. _`Logistic Regression`: https://snapml.readthedocs.io/en/latest/#snapml.LogisticRegression
.. _`Snap ML`: https://www.zurich.ibm.com/snapml/
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.snapml.snap_logistic_regression.html",
    "import_from": "snapml",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "classifier"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
        "input_predict_proba": _input_predict_proba_schema,
        "output_predict_proba": _output_predict_proba_schema,
    },
}


SnapLogisticRegression = lale.operators.make_operator(
    _SnapLogisticRegressionImpl, _combined_schemas
)

lale.docstrings.set_docstrings(SnapLogisticRegression)
