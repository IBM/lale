# Copyright 2019-2023 IBM Corporation
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

import sys

__version__ = "0.9.3"

# all other code will go here.
from .operator_wrapper import (
    register_lale_wrapper_modules as register_lale_wrapper_modules,
)
from .operator_wrapper import wrap_imported_operators as wrap_imported_operators
