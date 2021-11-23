# Copyright 2020 IBM Corporation
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

from autoai_ts_libs.sklearn.autoai_ts_pipeline import (  # type: ignore # noqa
    AutoaiTSPipeline as model_to_be_wrapped,
)

import lale.docstrings
import lale.operators


class _AutoaiTSPipelineImpl:
    def __init__(self, steps, memory=None, verbose=False):
        self._hyperparams = {"steps": steps, "memory": memory, "verbose": verbose}
        kwargs = {"memory": memory, "verbose": verbose}
        self._wrapped_model = model_to_be_wrapped(steps=steps, **kwargs)

    def fit(self, X, y=None, **fit_params):
        self._wrapped_model.fit(X, y, **fit_params)
        return self

    def predict(self, X=None, **predict_params):
        return self._wrapped_model.predict(X, **predict_params)


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
            "type": "object",
            "additionalProperties": False,
            "required": ["steps", "memory", "verbose"],
            "relevantToOptimizer": [],
            "properties": {
                "steps": {
                    "description": "The pipeline steps.",
                    "type": "array",
                    "items": {"laleType": "Any"},
                },
                "memory": {
                    "description": """str or object with the joblib.Memory interface.
Used to cache the fitted transformers of the pipeline. By default,
no caching is performed. If a string is given, it is the path to
the caching directory. Enabling caching triggers a clone of
the transformers before fitting. Therefore, the transformer
instance given to the pipeline cannot be inspected
directly. Use the attribute ``named_steps`` or ``steps`` to
inspect estimators within the pipeline. Caching the
transformers is advantageous when fitting is time consuming.""",
                    "laleType": "Any",
                    "default": None,
                },
                "verbose": {
                    "description": "If True, the time elapsed while fitting each step will be printed as it is completed.",
                    "type": "boolean",
                    "default": False,
                },
            },
        }
    ]
}

_input_fit_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {  # Handles 1-D arrays as well
            "anyOf": [
                {"type": "array", "items": {"laleType": "Any"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"laleType": "Any"}},
                },
            ]
        },
        "y": {"laleType": "Any"},
    },
}

_input_predict_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {  # Handles 1-D arrays as well
            "anyOf": [
                {"type": "array", "items": {"laleType": "Any"}},
                {
                    "type": "array",
                    "items": {"type": "array", "items": {"laleType": "Any"}},
                },
            ]
        }
    },
}

_output_predict_schema = {
    "description": "Features; the outer array is over samples.",
    "anyOf": [
        {"type": "array", "items": {"laleType": "Any"}},
        {
            "type": "array",
            "items": {"type": "array", "items": {"laleType": "Any"}},
        },
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Operator from `autoai_ts_libs`_.
.. _`autoai_ts_libs`: https://pypi.org/project/autoai-ts-libs""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai_ts_libs.autoai_ts_pipeline.html",
    "import_from": "autoai_ts_libs.sklearn.autoai_ts_pipeline",
    "type": "object",
    "tags": {"pre": [], "op": ["classifer", "regressor", "estimator"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}

AutoaiTSPipeline = lale.operators.make_operator(
    _AutoaiTSPipelineImpl, _combined_schemas
)
lale.docstrings.set_docstrings(AutoaiTSPipeline)
