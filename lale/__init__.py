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

import sys

__version__ = "0.6.6"

try:
    # This variable is injected in the __builtins__ by the build
    # process. It is used to not try to import rest of the lale packages when
    # it is being installed.
    __LALE_SETUP__  # type: ignore
except NameError:
    __LALE_SETUP__ = False

if __LALE_SETUP__:  # type: ignore
    sys.stderr.write("Partial import of lale during the build process.\n")
    # We are not importing the rest of lale during the build
    # process.
else:
    # all other code will go here.
    from .operator_wrapper import wrap_imported_operators
