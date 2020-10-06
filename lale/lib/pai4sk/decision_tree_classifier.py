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

try:
    import pai4sk

    pai4sk_installed = True
except ImportError:
    pai4sk_installed = False
import lale.datasets.data_schemas
import lale.docstrings
import lale.operators


class DecisionTreeClassifierImpl:
    def __init__(
        self,
        criterion="gini",
        splitter="best",
        max_depth=None,
        min_samples_leaf=1,
        max_features=None,
        random_state=None,
        n_threads=1,
        use_histograms=False,
        hist_nbins=256,
        use_gpu=False,
        gpu_id=0,
        verbose=False,
    ):
        assert (
            pai4sk_installed
        ), """Your Python environment does not have pai4sk installed. For installation instructions see: https://www.zurich.ibm.com/snapml/"""
        self._hyperparams = {
            "criterion": criterion,
            "splitter": splitter,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "random_state": random_state,
            "n_threads": n_threads,
            "use_histograms": use_histograms,
            "hist_nbins": hist_nbins,
            "use_gpu": use_gpu,
            "gpu_id": gpu_id,
            "verbose": verbose,
        }
        self._wrapped_model = pai4sk.DecisionTreeClassifier(**self._hyperparams)

    def fit(self, X, y, **fit_params):
        X = lale.datasets.data_schemas.strip_schema(X)
        y = lale.datasets.data_schemas.strip_schema(y)
        self._wrapped_model.fit(X, y, **fit_params)
        return self

    def predict(self, X, **predict_params):
        X = lale.datasets.data_schemas.strip_schema(X)
        if predict_params is None:
            return self._wrapped_model.predict(X)
        else:
            return self._wrapped_model.predict(X, **predict_params)

    def predict_proba(self, X, **predict_proba_params):
        X = lale.datasets.data_schemas.strip_schema(X)
        if predict_proba_params is None:
            return self._wrapped_model.predict_proba(X)
        else:
            return self._wrapped_model.predict_proba(X, **predict_proba_params)


_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "relevantToOptimizer": [
                "criterion",
                "splitter",
                "max_depth",
                "min_samples_leaf",
                "max_features",
            ],
            "additionalProperties": False,
            "properties": {
                "criterion": {
                    "enum": ["gini"],
                    "default": "gini",
                    "description": "Function to measure the quality of a split.",
                },
                "splitter": {
                    "enum": ["best"],
                    "default": "best",
                    "description": "The strategy used to choose the split at each node.",
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
                            "minimum": 2,
                            "forOptimizer": False,
                            "description": "Consider max_features features at each split.",
                        },
                        {
                            "type": "number",
                            "minimum": 0.0,
                            "exclusiveMinimum": True,
                            "maximum": 1.0,
                            "distribution": "uniform",
                            "description": "max_features is a fraction and int(max_features * n_features) features are considered at each split.",
                        },
                        {"enum": ["auto", "sqrt", "log2", None]},
                    ],
                    "default": "auto",
                    "description": "The number of features to consider when looking for the best split.",
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
                "n_threads": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "Number of CPU threads to use.",
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
                "gpu_id": {
                    "type": "integer",
                    "default": 0,
                    "description": "Device ID of the GPU which will be used when GPU acceleration is enabled.",
                },
                "verbose": {
                    "type": "boolean",
                    "default": False,
                    "description": "If True, it prints debugging information while training. Warning: this will increase the training time. For performance evaluation, use verbose=False.",
                },
            },
        },
        {
            "description": "Only need hist_nbins when use_histograms is true.",
            "anyOf": [
                {"type": "object", "properties": {"use_histograms": {"enum": [True]}}},
                {"type": "object", "properties": {"hist_nbins": {"enum": [256]}}},
            ],
        },
        {
            "description": "GPU only supported for histogram-based splits.",
            "anyOf": [
                {"type": "object", "properties": {"use_histograms": {"enum": [True]}}},
                {"type": "object", "properties": {"use_histograms": {"enum": [False]}}},
            ],
        },
        {
            "description": "Only need gpu_id when use_gpu is true.",
            "anyOf": [
                {"type": "object", "properties": {"use_gpu": {"enum": [True]}}},
                {"type": "object", "properties": {"gpu_id": {"enum": [0]}}},
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
            "description": "The predicted classes.",
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
        "num_threads": {
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
        "num_threads": {
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
        "description": "The inner array has items corresponding to each class.",
        "items": {"type": "number"},
    },
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """`Decision tree classifier`_ from `Snap ML`_. It can be used for binary classification problems.

.. _`Decision tree classifier`: https://ibmsoe.github.io/snap-ml-doc/v1.6.0/dectreeapidoc.html
.. _`Snap ML`: https://www.zurich.ibm.com/snapml/
""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.pai4sk.random_forest_classifier.html",
    "import_from": "pai4sk",
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

lale.docstrings.set_docstrings(DecisionTreeClassifierImpl, _combined_schemas)

DecisionTreeClassifier = lale.operators.make_operator(
    DecisionTreeClassifierImpl, _combined_schemas
)
