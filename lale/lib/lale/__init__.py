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
Lale operators with schemas.

Operators
=========

Estimators:

* lale.lib.lale. `BaselineClassifier`_
* lale.lib.lale. `BaselineRegressor`_
* lale.lib.lale. `GridSearchCV`_
* lale.lib.lale. `Hyperopt`_
* lale.lib.lale. `TopKVotingClassifier`_
* lale.lib.lale. `SMAC`_

Transformers:

* lale.lib.lale. `Batching`_
* lale.lib.lale. `ConcatFeatures`_
* lale.lib.lale. `NoOp`_
* lale.lib.lale. `Project`_
* lale.lib.lale. `SampleBasedVoting`_

Estimators and transformers:

* lale.lib.lale. `Both`_
* lale.lib.lale. `IdentityWrapper`_
* lale.lib.lale. `Observing`_

.. _`BaselineClassifier`: lale.lib.lale.baseline_classifier.html
.. _`BaselineRegressor`: lale.lib.lale.baseline_regressor.html
.. _`GridSearchCV`: lale.lib.lale.grid_search_cv.html
.. _`Hyperopt`: lale.lib.lale.hyperopt.html
.. _`TopKVotingClassifier`: lale.lib.lale.topk_voting_classifier.html
.. _`SMAC`: lale.lib.lale.smac.html
.. _`Batching`: lale.lib.lale.batching.html
.. _`ConcatFeatures`: lale.lib.lale.concat_features.html
.. _`NoOp`: lale.lib.lale.no_op.html
.. _`Project`: lale.lib.lale.project.html
.. _`SampleBasedVoting`: lale.lib.lale.sample_based_voting.html
.. _`Both`: lale.lib.lale.both.html
.. _`IdentityWrapper`: lale.lib.lale.identity_wrapper.html
.. _`Observing`: lale.lib.lale.observing.html
"""

#estimators
from .baseline_classifier import BaselineClassifier
from .baseline_regressor import BaselineRegressor
from .grid_search_cv import GridSearchCV
from .hyperopt import Hyperopt
from .topk_voting_classifier import TopKVotingClassifier
from .smac import SMAC

#transformers
from .batching import Batching
from .concat_features import ConcatFeatures
from .no_op import NoOp
from .project import Project
from .sample_based_voting import SampleBasedVoting

#estimators and transformers
from .both import Both
from .identity_wrapper import IdentityWrapper
from .observing import Observing
