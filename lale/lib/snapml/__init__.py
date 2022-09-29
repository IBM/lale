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
Schema-enhanced versions of the operators from `Snap ML`_ to enable hyperparameter tuning.

.. _`Snap ML`: https://www.zurich.ibm.com/snapml/

Operators
=========

Classifiers:

* lale.lib.snapml. `BatchedTreeEnsembleClassifier`_
* lale.lib.snapml. `SnapBoostingMachineClassifier`_
* lale.lib.snapml. `SnapDecisionTreeClassifier`_
* lale.lib.snapml. `SnapLogisticRegression`_
* lale.lib.snapml. `SnapRandomForestClassifier`_
* lale.lib.snapml. `SnapSVMClassifier`_

Regressors:

* lale.lib.snapml. `BatchedTreeEnsembleRegressor`_
* lale.lib.snapml. `SnapBoostingMachineRegressor`_
* lale.lib.snapml. `SnapDecisionTreeRegressor`_
* lale.lib.snapml. `SnapLinearRegression`_
* lale.lib.snapml. `SnapRandomForestRegressor`_

.. _`BatchedTreeEnsembleClassifier`: lale.lib.snapml.batched_tree_ensemble_classifier.html
.. _`BatchedTreeEnsembleRegressor`: lale.lib.snapml.batched_tree_ensemble_regressor.html
.. _`SnapBoostingMachineClassifier`: lale.lib.snapml.snap_boosting_machine_classifier.html
.. _`SnapBoostingMachineRegressor`: lale.lib.snapml.snap_boosting_machine_regressor.html
.. _`SnapDecisionTreeClassifier`: lale.lib.snapml.snap_decision_tree_classifier.html
.. _`SnapDecisionTreeRegressor`: lale.lib.snapml.snap_decision_tree_regressor.html
.. _`SnapLinearRegression`: lale.lib.snapml.snap_linear_regression.html
.. _`SnapLogisticRegression`: lale.lib.snapml.snap_logistic_regression.html
.. _`SnapRandomForestClassifier`: lale.lib.snapml.snap_random_forest_classifier.html
.. _`SnapRandomForestRegressor`: lale.lib.snapml.snap_random_forest_regressor.html
.. _`SnapSVMClassifier`: lale.lib.snapml.snap_svm_classifier.html
"""
from lale import register_lale_wrapper_modules

from .batched_tree_ensemble_classifier import (
    BatchedTreeEnsembleClassifier as BatchedTreeEnsembleClassifier,
)
from .batched_tree_ensemble_regressor import (
    BatchedTreeEnsembleRegressor as BatchedTreeEnsembleRegressor,
)
from .snap_boosting_machine_classifier import (
    SnapBoostingMachineClassifier as SnapBoostingMachineClassifier,
)
from .snap_boosting_machine_regressor import (
    SnapBoostingMachineRegressor as SnapBoostingMachineRegressor,
)
from .snap_decision_tree_classifier import (
    SnapDecisionTreeClassifier as SnapDecisionTreeClassifier,
)
from .snap_decision_tree_regressor import (
    SnapDecisionTreeRegressor as SnapDecisionTreeRegressor,
)
from .snap_linear_regression import SnapLinearRegression as SnapLinearRegression
from .snap_logistic_regression import SnapLogisticRegression as SnapLogisticRegression
from .snap_random_forest_classifier import (
    SnapRandomForestClassifier as SnapRandomForestClassifier,
)
from .snap_random_forest_regressor import (
    SnapRandomForestRegressor as SnapRandomForestRegressor,
)
from .snap_svm_classifier import SnapSVMClassifier as SnapSVMClassifier

# Note: all imports should be done as
# from .xxx import XXX as XXX
# this ensures that pyright considers them to be publicly available
# and not private imports (this affects lale users that use pyright)


register_lale_wrapper_modules(__name__)
