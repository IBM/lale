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

"""
Scikit-learn compatible wrappers for XGBoost_ along with schemas to enable hyperparameter tuning.

.. _XGBoost: https://xgboost.readthedocs.io/en/latest/

Operators:
==========
* `XGBClassifier`_
* `XGBRegressor`_

.. _`XGBClassifier`: lale.lib.xgboost.xgb_classifier.html
.. _`XGBRegressor`: lale.lib.xgboost.xgb_regressor.html
"""

from lale import register_lale_wrapper_modules

from .xgb_classifier import XGBClassifier as XGBClassifier
from .xgb_regressor import XGBRegressor as XGBRegressor

# Note: all imports should be done as
# from .xxx import XXX as XXX
# this ensures that pyright considers them to be publicly available
# and not private imports (this affects lale users that use pyright)


register_lale_wrapper_modules(__name__)
