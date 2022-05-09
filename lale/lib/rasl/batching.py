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

import numpy as np

import lale.docstrings
import lale.helpers
import lale.operators
from lale.lib.rasl import PrioResourceAware, fit_with_batches


class _BatchingImpl:
    def __init__(
        self,
        operator=None,
        batch_size=32,
        shuffle=True,
        num_workers=0,
        inmemory=False,
        num_epochs=None,
        max_resident=None,
        scoring=None,
        progress_callback=None,
        verbose=0,
    ):
        self.operator = operator
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.inmemory = inmemory
        self.num_epochs = num_epochs
        self.max_resident = max_resident
        self.scoring = scoring
        self.progress_callback = progress_callback
        self.verbose = verbose

    def fit(self, X, y=None, classes=None):
        if self.operator is None:
            raise ValueError("The pipeline object can't be None at the time of fit.")
        try:
            from torch.utils.data import DataLoader
        except ImportError:
            raise ImportError(
                """Batching uses Pytorch for data loading. It is not
            installed in the current environment, please install
            the package and try again."""
            )
        if isinstance(X, DataLoader):
            assert (
                y is None
            ), "When X is a torch.utils.data.DataLoader, y should be None"
            data_loader = X
        elif hasattr(X, "__next__") and hasattr(
            X, "__iter__"
        ):  # allow an iterable that is not a torch data loader
            assert y is None, "When X is an Iterable, y should be None"
            data_loader = X
        else:
            data_loader = lale.helpers.create_data_loader(
                X=X,
                y=y,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle,
            )
        if y is not None and classes is None:
            classes = np.unique(y)
        self.operator = fit_with_batches(
            pipeline=self.operator,
            batches=data_loader,  # type:ignore
            unique_class_labels=classes,
            max_resident=self.max_resident,
            prio=PrioResourceAware(),
            partial_transform=False,
            scoring=self.scoring,
            progress_callback=self.progress_callback,
            verbose=self.verbose,
        )
        return self

    def transform(self, X, y=None):
        try:
            from torch.utils.data import DataLoader
        except ImportError:
            raise ImportError(
                """Batching uses Pytorch for data loading. It is not
            installed in the current environment, please install
            the package and try again."""
            )
        if isinstance(X, DataLoader):
            assert (
                y is None
            ), "When X is a torch.utils.data.DataLoader, y should be None"
            data_loader = X
        elif hasattr(X, "__next__") and hasattr(
            X, "__iter__"
        ):  # allow an iterable that is not a torch data loader
            assert y is None, "When X is an Iterable, y should be None"
            data_loader = X
        else:
            data_loader = lale.helpers.create_data_loader(
                X=X,
                y=y,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                shuffle=self.shuffle,
            )

        op = self.operator
        assert op is not None
        transformed_data = op.transform_with_batches(
            data_loader, serialize=self.inmemory
        )
        return transformed_data

    def predict(self, X, y=None):
        return self.transform(X, y)


_input_fit_schema = {
    "description": "Input data schema for fit.",
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "anyOf": [
                {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"type": "number"},
                            {"type": "string"},
                            {"type": "boolean"},
                        ]
                    },
                },
                {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "number"},
                                {"type": "string"},
                                {"type": "boolean"},
                            ]
                        },
                    },
                },
                {"type": "object"},
            ],
        },
        "y": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"type": "integer"},
                            {"type": "number"},
                            {"type": "string"},
                        ]
                    },
                },
                {"enum": [None]},
            ],
        },
        "classes": {
            "anyOf": [
                {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"type": "number"},
                            {"type": "string"},
                            {"type": "boolean"},
                        ]
                    },
                },
                {"enum": [None]},
            ],
            "description": """The total number of classes in the entire training dataset.""",
        },
    },
}

_input_predict_transform_schema = {  # TODO: separate predict vs. transform
    "description": "Input data schema for predictions.",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "anyOf": [
                {
                    "type": "array",
                    "items": {
                        "anyOf": [
                            {"type": "number"},
                            {"type": "string"},
                            {"type": "boolean"},
                        ]
                    },
                },
                {
                    "type": "array",
                    "items": {
                        "type": "array",
                        "items": {
                            "anyOf": [
                                {"type": "number"},
                                {"type": "string"},
                                {"type": "boolean"},
                            ]
                        },
                    },
                },
                {},
            ],
        },
        "y": {
            "type": "array",
            "items": {"anyOf": [{"type": "integer"}, {"type": "number"}]},
        },
    },
}

_output_schema = {  # TODO: separate predict vs. transform
    "description": "Output data schema for transformed data.",
    "laleType": "Any",
}


_hyperparams_schema = {
    "description": "Hyperparameter schema.",
    "allOf": [
        {
            "description": "This first sub-object lists all constructor arguments with their "
            "types, one at a time, omitting cross-argument constraints.",
            "type": "object",
            "additionalProperties": False,
            "relevantToOptimizer": ["batch_size"],
            "properties": {
                "operator": {
                    "description": "A lale pipeline object to be used inside of batching",
                    "laleType": "operator",
                },
                "batch_size": {
                    "description": "Batch size used for transform.",
                    "type": "integer",
                    "default": 64,
                    "minimum": 1,
                    "distribution": "uniform",
                    "minimumForOptimizer": 32,
                    "maximumForOptimizer": 128,
                },
                "shuffle": {
                    "type": "boolean",
                    "default": False,
                    "description": "Shuffle dataset before batching or not.",
                },
                "num_workers": {
                    "type": "integer",
                    "default": 0,
                    "description": "Number of workers for pytorch dataloader.",
                },
                "inmemory": {
                    "type": "boolean",
                    "default": False,
                    "description": """Whether all the computations are done in memory
                    or intermediate outputs are serialized. Only applies to transform/predict.
                    For fit, use the `max_resident` argument.""",
                },
                "num_epochs": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": None,
                    "description": "Number of epochs. If the operator has `num_epochs` as a parameter, that takes precedence.",
                },
                "max_resident": {
                    "anyOf": [{"type": "integer"}, {"enum": [None]}],
                    "default": None,
                    "description": "Amount of memory to be used in bytes.",
                },
                "scoring": {
                    "anyOf": [{"laleType": "callable"}, {"enum": [None]}],
                    "default": None,
                    "description": "Batch-wise scoring metrics from `lale.lib.rasl`.",
                },
                "progress_callback": {
                    "anyOf": [{"laleType": "callable"}, {"enum": [None]}],
                    "default": None,
                    "description": "Callback function to get performance metrics per batch.",
                },
                "verbose": {
                    "type": "integer",
                    "default": 0,
                    "description": "Verbosity level, higher values mean more information.",
                },
            },
        }
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Batching trains the given pipeline using batches.
The batch_size is used across all steps of the pipeline, serializing
the intermediate outputs if specified.""",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_transform_schema,
        "output_predict": _output_schema,
        "input_transform": _input_predict_transform_schema,
        "output_transform": _output_schema,
    },
}


Batching = lale.operators.make_operator(_BatchingImpl, _combined_schemas)

lale.docstrings.set_docstrings(Batching)
