# Copyright 2020,2021 IBM Corporation
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
Schema-enhanced versions of some of the operators from `Snap ML`_ to enable hyperparameter tuning.

.. _`Snap ML`: https://www.zurich.ibm.com/snapml/

Operators
=========

Classifiers:

* lale.lib.snapml. `BoostingMachineClassifier`_
* lale.lib.snapml. `BoostingMachineRegressor`_
* lale.lib.snapml. `DecisionTreeClassifier`_
* lale.lib.snapml. `DecisionTreeRegressor_
* lale.lib.snapml. `RandomForestClassifier`_
* lale.lib.snapml. `RandomForestRegressor`_

.. _`BoostingMachineClassifier`: lale.lib.snapml.boosting_machine_classifier.html
.. _`BoostingMachineRegressor`: lale.lib.snapml.boosting_machine_regressor.html
.. _`DecisionTreeClassifier`: lale.lib.snapml.decision_tree_classifier.html
.. _`DecisionTreeRegressor`: lale.lib.snapml.decision_tree_regressor.html
.. _`RandomForestClassifier`: lale.lib.snapml.random_forest_classifier.html
.. _`RandomForestRegressor`: lale.lib.snapml.random_forest_regressor.html
"""

from .boosting_machine_classifier import BoostingMachineClassifier
from .boosting_machine_regressor import BoostingMachineRegressor
from .decision_tree_classifier import DecisionTreeClassifier
from .decision_tree_regressor import DecisionTreeRegressor
from .random_forest_classifier import RandomForestClassifier
from .random_forest_regressor import RandomForestRegressor
