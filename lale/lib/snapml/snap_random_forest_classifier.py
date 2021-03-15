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


class _SnapRandomForestClassifierImpl:
    def __init__(
        self,
        n_estimators=10,
        criterion="gini",
        max_depth=None,
        min_samples_leaf=1,
        max_features="auto",
        bootstrap=True,
        n_jobs=1,
        random_state=None,
        verbose=False,
        use_histograms=False,
        hist_nbins=256,
        use_gpu=False,
        gpu_ids=None,
    ):
        assert (
            snapml_installed
        ), """Your Python environment does not have snapml installed. Install using: pip install snapml"""
        self._hyperparams = {
            "n_estimators": n_estimators,
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "bootstrap": bootstrap,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbose": verbose,
            "use_histograms": use_histograms,
            "hist_nbins": hist_nbins,
            "use_gpu": use_gpu,
            "gpu_ids": gpu_ids,
        }
        modified_hps = {**self._hyperparams}
        if modified_hps["gpu_ids"] is None:
            modified_hps["gpu_ids"] = [0]  # TODO: support list as default
        self._wrapped_model = snapml.SnapRandomForestClassifier(**modified_hps)

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
            "relevantToOptimizer": ["n_estimators", "max_depth", "max_features"],
            "additionalProperties": False,
            "properties": {
                "n_estimators": {
                    "type": "integer",
                    "minimum": 1,
                    "minimumForOptimizer": 10,
                    "maximumForOptimizer": 100,
                    "default": 10,
                    "description": "The number of trees in the forest.",
                },
                "criterion": {
                    "enum": ["gini"],
                    "default": "gini",
                    "description": "Function to measure the quality of a split.",
                },
                "max_depth": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimum": 1,
                            "minimumForOptimizer": 3,
                            "maximumForOptimizer": 5,
                        },
                        {
                            "enum": [None],
                            "description": "Nodes are expanded until all leaves are pure or until all leaves contain less than min_samples_leaf samples.",
                        },
                    ],
                    "default": None,
                    "description": "The maximum depth of the tree.",
                },
                "min_samples_leaf": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimum": 1,
                            "forOptimizer": False,
                            "laleMaximum": "X/maxItems",  # number of rows
                            "description": "Consider min_samples_leaf as the minimum number.",
                        },
                        {
                            "type": "number",
                            "minimum": 0.0,
                            "exclusiveMinimum": True,
                            "maximum": 0.5,
                            "description": "min_samples_leaf is a fraction and ceil(min_samples_leaf * n_samples) are the minimum number of samples for each node.",
                        },
                    ],
                    "default": 1,
                    "description": "The minimum number of samples required to be at a leaf node.",
                },
                "max_features": {
                    "anyOf": [
                        {
                            "type": "integer",
                            "minimum": 1,
                            "forOptimizer": False,
                            "laleMaximum": "X/items/maxItems",  # number of columns
                            "description": "Consider max_features features at each split.",
                        },
                        {
                            "type": "number",
                            "minimum": 0.0,
                            "exclusiveMinimum": True,
                            "maximum": 1.0,
                            "minimumForOptimizer": 0.1,
                            "maximumForOptimizer": 0.9,
                            "distribution": "uniform",
                            "description": "max_features is a fraction and int(max_features * n_features) features are considered at each split.",
                        },
                        {"enum": ["auto", "sqrt", "log2", None]},
                    ],
                    "default": "auto",
                    "description": "The number of features to consider when looking for the best split.",
                },
                "bootstrap": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether bootstrap samples are used when building trees.",
                },
                "n_jobs": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "Number of CPU threads to use.",
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
                "verbose": {
                    "type": "boolean",
                    "default": False,
                    "description": "If True, it prints debugging information while training. Warning: this will increase the training time. For performance evaluation, use verbose=False.",
                },
                "use_histograms": {
                    "type": "boolean",
                    "default": False,
                    "description": "Use histogram-based splits rather than exact splits.",
                },
                "hist_nbins": {
                    "type": "integer",
                    "default": 256,
                    "description": "Number of histogram bins.",
                },
                "use_gpu": {
                    "type": "boolean",
                    "default": False,
                    "description": "Use GPU acceleration (only supported for histogram-based splits).",
                },
                "gpu_ids": {
                    "anyOf": [
                        {"description": "Use [0].", "enum": [None]},
                        {"type": "array", "items": {"type": "integer"}},
                    ],
                    "default": None,
                    "description": "Device IDs of the GPUs which will be used when GPU acceleration is enabled.",
                },
            },
        },
        {
            "description": "GPU only supported for histogram-based splits.",
            "anyOf": [
                {"type": "object", "properties": {"use_gpu": {"enum": [False]}}},
                {"type": "object", "properties": {"use_histograms": {"enum": [True]}}},
            ],
        },
    ],
}

_input_fit_schema = {
    "description": "Build a forest of trees from the training set (X, y).",
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
        "sample_weight": {
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"enum": [None], "description": "Samples are equally weighted."},
            ],
            "description": "Sample weights.",
            "default": None,
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
            "description": "Number of threads used to run inference. By default inference runs with maximum number of available threads..",
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
    "description": """`Random forest classifier`_ from `Snap ML`_. It can be used for binary classification problems.

.. _`Random forest classifier`: https://snapml.readthedocs.io/en/latest/#snapml.RandomForestClassifier
.. _`Snap ML`: https://www.zurich.ibm.com/snapml/
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.snapml.snap_random_forest_classifier.html",
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


SnapRandomForestClassifier = lale.operators.make_operator(
    _SnapRandomForestClassifierImpl, _combined_schemas
)

lale.docstrings.set_docstrings(SnapRandomForestClassifier)
