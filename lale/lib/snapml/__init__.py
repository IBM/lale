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

* lale.lib.snapml. `SnapBoostingMachineClassifier`_
* lale.lib.snapml. `SnapBoostingMachineRegressor`_
* lale.lib.snapml. `SnapDecisionTreeClassifier`_
* lale.lib.snapml. `SnapDecisionTreeRegressor_
* lale.lib.snapml. `SnapRandomForestClassifier`_
* lale.lib.snapml. `SnapRandomForestRegressor`_

.. _`SnapBoostingMachineClassifier`: lale.lib.snapml.boosting_machine_classifier.html
.. _`SnapBoostingMachineRegressor`: lale.lib.snapml.boosting_machine_regressor.html
.. _`SnapDecisionTreeClassifier`: lale.lib.snapml.decision_tree_classifier.html
.. _`SnapDecisionTreeRegressor`: lale.lib.snapml.decision_tree_regressor.html
.. _`SnapRandomForestClassifier`: lale.lib.snapml.random_forest_classifier.html
.. _`SnapRandomForestRegressor`: lale.lib.snapml.random_forest_regressor.html
"""

from .boosting_machine_classifier import SnapBoostingMachineClassifier
from .boosting_machine_regressor import SnapBoostingMachineRegressor
from .decision_tree_classifier import SnapDecisionTreeClassifier
from .decision_tree_regressor import SnapDecisionTreeRegressor
from .random_forest_classifier import SnapRandomForestClassifier
from .random_forest_regressor import SnapRandomForestRegressor
