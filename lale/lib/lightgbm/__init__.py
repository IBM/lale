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
Scikit-learn compatible wrappers for LightGBM_ along with schemas to enable hyperparameter tuning.

.. _LightGBM: https://www.microsoft.com/en-us/research/project/lightgbm/

Operators:
==========
* `LGBMClassifier`_
* `LGBMRegressor`_

.. _`LGBMClassifier`: lale.lib.lightgbm.lgbm_classifier.html
.. _`LGBMRegressor`: lale.lib.lightgbm.lgbm_regressor.html
"""

from .lgbm_classifier import LGBMClassifier
from .lgbm_regressor import LGBMRegressor
