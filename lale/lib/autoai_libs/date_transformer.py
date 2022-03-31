# # Copyright 2022 IBM Corporation
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# # http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# import pandas as pd
# from autoai_libs.transformers.date_time.date_time_transformer import (
#     DateTransformer as model_to_be_wrapped,
# )

# import lale.docstrings
# import lale.operators


# class _DateTransformerImpl:
#     def __init__(
#         self,
#         options=None,
#         delete_source_columns=False,
#         column_headers_list=None,
#         missing_values_reference_list=None,
#         activate_flag=True,
#         float32_processing_flag=True,
#     ):
#         self._hyperparams = {
#             "options": options,
#             "delete_source_columns": delete_source_columns,
#             "column_headers_list": column_headers_list,
#             "missing_values_reference_list": missing_values_reference_list,
#             "activate_flag": activate_flag,
#             "float32_processing_flag": float32_processing_flag,
#         }
#         self._wrapped_model = model_to_be_wrapped(**self._hyperparams)

#     def fit(self, X, y=None):
#         if isinstance(X, pd.DataFrame):
#             X = X.to_numpy()
#         self._wrapped_model.fit(X, y)
#         return self

#     def transform(self, X):
#         if isinstance(X, pd.DataFrame):
#             X = X.to_numpy()
#         return self._wrapped_model.transform(X)


# _hyperparams_schema = {
#     "allOf": [
#         {
#             "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.",
#             "type": "object",
#             "additionalProperties": False,
#             "required": [
#                 "options",
#                 "delete_source_columns",
#                 "column_headers_list",
#                 "missing_values_reference_list",
#                 "activate_flag",
#                 "float32_processing_flag",
#             ],
#             "relevantToOptimizer": [],
#             "properties": {
#                 "options": {
#                     "description": """List containing the types of new feature columns to add for each detected datetime column.
# Default is None, in this case all the above options are applied""",
#                     "anyOf": [
#                         {
#                             "type": "array",
#                             "items": {
#                                 "enum": [
#                                     "all",
#                                     "Datetime",
#                                     "DateToFloatTimestamp",
#                                     "DateToTimestamp",
#                                     "Timestamp",
#                                     "FloatTimestamp",
#                                     "DayOfWeek",
#                                     "DayOfMonth",
#                                     "Hour",
#                                     "DayOfYear",
#                                     "Week",
#                                     "Month",
#                                     "Year",
#                                     "Second",
#                                     "Minute",
#                                 ]
#                             },
#                         },
#                         {"enum": [None]},
#                     ],
#                     "default": None,
#                 },
#                 "delete_source_columns": {
#                     "description": "Flag determining whether the original date columns will be deleted or not.",
#                     "type": "boolean",
#                     "default": True,
#                 },
#                 "column_headers_list": {
#                     "description": "List containing the column names of the input array",
#                     "anyOf": [
#                         {
#                             "type": "array",
#                             "items": {
#                                 "anyOf": [{"type": "string"}, {"type": "integer"}]
#                             },
#                         },
#                         {"enum": [None]},
#                     ],
#                     "default": None,
#                 },
#                 "missing_values_reference_list": {
#                     "description": "List containing missing values of the input array",
#                     "anyOf": [
#                         {"type": "array", "items": {"laleType": "Any"}},
#                         {"enum": [None]},
#                     ],
#                     "default": None,
#                 },
#                 "activate_flag": {
#                     "description": "Determines whether transformer is active or not.",
#                     "type": "boolean",
#                     "default": True,
#                 },
#                 "float32_processing_flag": {
#                     "description": "Flag that determines whether timestamps will be float32-compatible.",
#                     "type": "boolean",
#                     "default": True,
#                 },
#             },
#         }
#     ]
# }

# _input_fit_schema = {
#     "type": "object",
#     "required": ["X"],
#     "additionalProperties": False,
#     "properties": {
#         "X": {  # Handles 1-D arrays as well
#             "anyOf": [
#                 {"type": "array", "items": {"laleType": "Any"}},
#                 {
#                     "type": "array",
#                     "items": {"type": "array", "items": {"laleType": "Any"}},
#                 },
#             ]
#         },
#         "y": {"laleType": "Any"},
#     },
# }

# _input_transform_schema = {
#     "type": "object",
#     "required": ["X"],
#     "additionalProperties": False,
#     "properties": {
#         "X": {  # Handles 1-D arrays as well
#             "anyOf": [
#                 {"type": "array", "items": {"laleType": "Any"}},
#                 {
#                     "type": "array",
#                     "items": {"type": "array", "items": {"laleType": "Any"}},
#                 },
#             ]
#         }
#     },
# }

# _output_transform_schema = {
#     "description": "Features; the outer array is over samples.",
#     "anyOf": [
#         {"type": "array", "items": {"laleType": "Any"}},
#         {"type": "array", "items": {"type": "array", "items": {"laleType": "Any"}}},
#     ],
# }

# _combined_schemas = {
#     "$schema": "http://json-schema.org/draft-04/schema#",
#     "description": """Operator from `autoai_libs`_. Detects date columns on an input array and adds new feature columns for each detected date column.

# .. _`autoai_libs`: https://pypi.org/project/autoai-libs""",
#     "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai_libs.date_transformer.html",
#     "import_from": "autoai_libs.transformers.date_time",
#     "type": "object",
#     "tags": {"pre": [], "op": ["transformer"], "post": []},
#     "properties": {
#         "hyperparams": _hyperparams_schema,
#         "input_fit": _input_fit_schema,
#         "input_transform": _input_transform_schema,
#         "output_transform": _output_transform_schema,
#     },
# }


# DateTransformer = lale.operators.make_operator(_DateTransformerImpl, _combined_schemas)

# lale.docstrings.set_docstrings(DateTransformer)
