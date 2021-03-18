# Copyright 2021 IBM Corporation
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

schema_X_numbers_y_top = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        "y": {"description": "Target class labels (unused)."},
    },
}

schema_X_numbers = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
    },
}

schema_2D_numbers = {
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
