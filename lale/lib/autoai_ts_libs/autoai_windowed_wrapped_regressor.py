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

from autoai_ts_libs.sklearn.mvp_windowed_wrapped_regressor import (  # type: ignore # noqa
    AutoaiWindowedWrappedRegressor as model_to_be_wrapped,
)
from sklearn.pipeline import make_pipeline

import lale.docstrings
import lale.operators


class _AutoaiWindowedWrappedRegressorImpl:
    def __init__(self, regressor=None, n_jobs=None):
        if regressor is None:
            nested_op = None
        elif isinstance(regressor, lale.operators.TrainableIndividualOp):
            nested_op = make_pipeline(regressor.impl)
        elif isinstance(regressor, lale.operators.BasePipeline):
            nested_op = regressor.export_to_sklearn_pipeline()
        else:
            # TODO: What is the best way to handle this case?
            nested_op = None
        self._hyperparams = {"regressor": nested_op, "n_jobs": n_jobs}
        self._wrapped_model = model_to_be_wrapped(**self._hyperparams)

    def fit(self, X, y):
        self._wrapped_model.fit(X, y)
        return self

    def predict(self, X, **predict_params):
        return self._wrapped_model.predict(X, **predict_params)


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
            "type": "object",
            "additionalProperties": False,
            "required": ["regressor", "n_jobs"],
            "relevantToOptimizer": [],
            "properties": {
                "regressor": {
                    "description": """Regressor object.
For a multi-variate case, this is wrapped in sklearn.multioutput.MultiOutputRegressor.""",
                    "anyOf": [{"laleType": "operator"}, {"enum": [None]}],
                    "default": None,
                },
                "n_jobs": {
                    "description": """Number of CPU cores when parallelizing over targets in the case of a  multivariate time series.""",
                    "anyOf": [
                        {
                            "description": "1 unless in joblib.parallel_backend context.",
                            "enum": [None],
                        },
                        {"description": "Use all processors.", "enum": [-1]},
                        {
                            "description": "Number of CPU cores.",
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
        {"type": "array", "items": {"type": "array", "items": {"laleType": "Any"}}},
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Operator from `autoai_ts_libs`_.

.. _`autoai_ts_libs`: https://pypi.org/project/autoai-ts-libs""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai_ts_libs.autoai_windowed_wrapped_regressor.html",
    "import_from": "autoai_ts_libs.sklearn.mvp_windowed_wrapped_regressor",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "regressor"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_schema,
        "output_predict": _output_predict_schema,
    },
}

AutoaiWindowedWrappedRegressor = lale.operators.make_operator(
    _AutoaiWindowedWrappedRegressorImpl, _combined_schemas
)

lale.docstrings.set_docstrings(AutoaiWindowedWrappedRegressor)
