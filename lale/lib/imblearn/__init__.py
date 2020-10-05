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
Scikit-learn compatible wrappers for a subset of the operators from imbalanced-learn_ along with schemas to enable hyperparameter tuning.

.. _imbalanced-learn: https://imbalanced-learn.readthedocs.io/en/stable/index.html

Operators:
==========
* `CondensedNearestNeighbour`_
* `EditedNearestNeighbours`_
* `RepeatedEditedNearestNeighbours`_
* `AllKNN`_
* `InstanceHardnessThreshold`_
* `ADASYN`_
* `BorderlineSMOTE`_
* `RandomOverSampler`_
* `SMOTE`_
* `SVMSMOTE`_
* `SMOTEENN`_

.. _`CondensedNearestNeighbour`: lale.lib.imblearn.condensed_nearest_neighbour.html
.. _`EditedNearestNeighbours`: lale.lib.imblearn.edited_nearest_neighbours.html
.. _`RepeatedEditedNearestNeighbours`: lale.lib.imblearn.repeated_edited_nearest_neighbours.html
.. _`AllKNN`: lale.lib.imblearn.all_knn.html
.. _`InstanceHardnessThreshold`: lale.lib.imblearn.instance_hardness_threshold.html
.. _`ADASYN`: lale.lib.imblearn.adasyn.html
.. _`BorderlineSMOTE`: lale.lib.imblearn.borderline_smote.html
.. _`RandomOverSampler`: lale.lib.imblearn.random_over_sampler.html
.. _`SMOTE`: lale.lib.imblearn.smote.html
.. _`SVMSMOTE`: lale.lib.imblearn.svm_smote.html
.. _`SMOTEENN`: lale.lib.imblearn.smoteenn.html

"""

from .adasyn import ADASYN
from .all_knn import AllKNN
from .borderline_smote import BorderlineSMOTE
from .condensed_nearest_neighbour import CondensedNearestNeighbour
from .edited_nearest_neighbours import EditedNearestNeighbours
from .instance_hardness_threshold import InstanceHardnessThreshold
from .random_over_sampler import RandomOverSampler
from .repeated_edited_nearest_neighbours import RepeatedEditedNearestNeighbours
from .smote import SMOTE
from .smoteenn import SMOTEENN
from .svm_smote import SVMSMOTE
