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
* `SMOTEN`_
* `SMOTENC`_
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
.. _`SMOTEN`: lale.lib.imblearn.smoten.html
.. _`SMOTENC`: lale.lib.imblearn.smotenc.html
.. _`SVMSMOTE`: lale.lib.imblearn.svm_smote.html
.. _`SMOTEENN`: lale.lib.imblearn.smoteenn.html

"""

# Note: all imports should be done as
# from .xxx import XXX as XXX
# this ensures that pyright considers them to be publicly available
# and not private imports (this affects lale users that use pyright)

from .adasyn import ADASYN as ADASYN
from .all_knn import AllKNN as AllKNN
from .borderline_smote import BorderlineSMOTE as BorderlineSMOTE
from .condensed_nearest_neighbour import (
    CondensedNearestNeighbour as CondensedNearestNeighbour,
)
from .edited_nearest_neighbours import (
    EditedNearestNeighbours as EditedNearestNeighbours,
)
from .instance_hardness_threshold import (
    InstanceHardnessThreshold as InstanceHardnessThreshold,
)
from .random_over_sampler import RandomOverSampler as RandomOverSampler
from .repeated_edited_nearest_neighbours import (
    RepeatedEditedNearestNeighbours as RepeatedEditedNearestNeighbours,
)
from .smote import SMOTE as SMOTE
from .smoteenn import SMOTEENN as SMOTEENN
from .smoten import SMOTEN as SMOTEN
from .smotenc import SMOTENC as SMOTENC
from .svm_smote import SVMSMOTE as SVMSMOTE
