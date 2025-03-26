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

from lale.type_checking import JSON_TYPE

schema_silent: JSON_TYPE = {
    "anyOf": [
        {"type": "boolean"},
        {"enum": [None]},
    ],
    "default": None,
    "description": "Deprecated and replaced with verbosity, but adding to be backward compatible.",
}

schema_random_state: JSON_TYPE = {
    "description": "Random number seed.  (replaces seed)",
    "anyOf": [
        {
            "description": "RandomState used by np.random",
            "enum": [None],
        },
        {
            "description": "The seed used by the random number generator",
            "type": "integer",
        },
        {
            "description": "Random number generator instance.",
            "laleType": "numpy.random.RandomState",
        },
    ],
    "default": 0,
}
