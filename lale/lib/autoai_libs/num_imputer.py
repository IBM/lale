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

import autoai_libs.transformers.exportable
import numpy as np

import lale.docstrings
import lale.operators


class _NumImputerImpl:
    def __init__(self, *args, **kwargs):
        self._wrapped_model = autoai_libs.transformers.exportable.NumImputer(
            *args, **kwargs
        )

    def fit(self, X, y=None, **fit_params):
        self._wrapped_model.fit(X, y, **fit_params)
        return self

    def transform(self, X):
        return self._wrapped_model.transform(X)


_hyperparams_schema = {
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
            "type": "object",
            "additionalProperties": False,
            "required": ["strategy", "missing_values", "activate_flag"],
            "relevantToOptimizer": ["strategy"],
            "properties": {
                "strategy": {
                    "description": "The imputation strategy.",
                    "enum": ["mean", "median", "most_frequent"],
                    "default": "mean",
                },
                "missing_values": {
                    "description": "The placeholder for the missing values. All occurrences of missing_values will be imputed.",
                    "anyOf": [
                        {"laleType": "Any"},
                        {
                            "description": "For missing values encoded as np.nan.",
                            "enum": [np.nan],
                        },
                    ],
                    "default": np.nan,
                },
                "activate_flag": {
                    "description": "If False, transform(X) outputs the input numpy array X unmodified.",
                    "type": "boolean",
                    "default": True,
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

_input_transform_schema = {
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

_output_transform_schema = {
    "description": "Features; the outer array is over samples.",
    "anyOf": [
        {"type": "array", "items": {"laleType": "Any"}},
        {"type": "array", "items": {"type": "array", "items": {"laleType": "Any"}}},
    ],
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """Operator from `autoai_libs`_. Missing value imputation for numerical features, currently internally uses the sklearn SimpleImputer_.

.. _`autoai_libs`: https://pypi.org/project/autoai-libs
.. _SimpleImputer: https://scikit-learn.org/0.20/modules/generated/sklearn.impute.SimpleImputer.html#sklearn-impute-simpleimputer""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai_libs.num_imputer.html",
    "import_from": "autoai_libs.transformers.exportable",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}

NumImputer = lale.operators.make_operator(_NumImputerImpl, _combined_schemas)

autoai_libs_version_str = getattr(autoai_libs, "__version__", None)
if autoai_libs_version_str is not None:
    import typing

    from packaging import version

    from lale.schemas import AnyOf, Array, Enum, Float, Not, Null, Object, String

    autoai_libs_version = version.parse(autoai_libs_version_str)

    if autoai_libs_version >= version.Version("1.12.18"):
        NumImputer = typing.cast(
            lale.operators.PlannedIndividualOp,
            NumImputer.customize_schema(
                set_as_available=True,
                constraint=[
                    AnyOf(
                        desc="fill_value and fill_values cannot both be specified",
                        forOptimizer=False,
                        types=[Object(fill_value=Null()), Object(fill_values=Null())],
                    ),
                    AnyOf(
                        desc="if strategy=constants, the fill_values cannot be None",
                        forOptimizer=False,
                        types=[
                            Object(strategy=Not(Enum(["constants"]))),
                            Not(Object(fill_values=Null())),
                        ],
                    ),
                ],
                fill_value=AnyOf(
                    types=[Float(), String(), Enum(values=[np.nan]), Null()],
                    desc="The placeholder for fill value used in constant strategy",
                    default=None,
                ),
                fill_values=AnyOf(
                    types=[
                        Array(
                            items=AnyOf(
                                types=[Float(), String(), Enum(values=[np.nan]), Null()]
                            )
                        ),
                        Null(),
                    ],
                    desc="The placeholder for fill values used in constants strategy",
                    default=None,
                ),
                missing_values=AnyOf(
                    types=[Float(), String(), Enum(values=[np.nan]), Null()],
                    desc="The placeholder for the missing values. All occurrences of missing_values will be imputed.",
                    default=np.nan,
                ),
                sklearn_version_family=Enum(
                    desc="The sklearn version for backward compatibiity with versions 019 and 020dev. Currently unused.",
                    values=["20", "21", "22", "23", "24", None, "1"],
                    default=None,
                ),
                strategy=AnyOf(
                    types=[
                        Enum(
                            values=["mean"],
                            desc="Replace using the mean along each column. Can only be used with numeric data.",
                        ),
                        Enum(
                            values=["median"],
                            desc="Replace using the median along each column. Can only be used with numeric data.",
                        ),
                        Enum(
                            values=["most_frequent"],
                            desc="Replace using most frequent value each column. Used with strings or numeric data.",
                        ),
                        Enum(
                            values=["constant"],
                            desc="Replace with fill_value. Can be used with strings or numeric data.",
                        ),
                        Enum(
                            values=["constants"],
                            desc="Replace missing values in columns with values in fill_values list. Can be used with list of strings or numeric data.",
                        ),
                    ],
                    desc="The imputation strategy.",
                    default="mean",
                ),
            ),
        )

lale.docstrings.set_docstrings(NumImputer)
