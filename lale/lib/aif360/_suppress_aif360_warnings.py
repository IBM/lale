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

import logging

_logger = logging.getLogger()
_old_log_level = _logger.getEffectiveLevel()
_logger.setLevel(level=logging.ERROR)

# the following triggers spurious AIF360 warning "No module named 'fairlearn'":
import aif360.algorithms.inprocessing  # isort:skip # noqa:E402,F401  # pylint:disable=wrong-import-position,wrong-import-order

# the following triggers spurious AIF360 warning "No module named 'tempeh'":
import aif360.datasets  # isort:skip # noqa:E402,F401  # pylint:disable=wrong-import-position,wrong-import-order

_logger.setLevel(_old_log_level)

dummy = "dummy"
