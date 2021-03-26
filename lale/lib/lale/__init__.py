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

* lale.lib.lale. `AutoPipeline`_
* lale.lib.lale. `GridSearchCV`_
* lale.lib.lale. `HalvingGridSearchCV`_
* lale.lib.lale. `Hyperopt`_
* lale.lib.lale. `TopKVotingClassifier`_
* lale.lib.lale. `SMAC`_

Transformers:

* lale.lib.lale. `Batching`_
* lale.lib.lale. `ConcatFeatures`_
* lale.lib.lale. `NoOp`_
* lale.lib.lale. `Project`_
* lale.lib.lale. `SampleBasedVoting`_
* lale.lib.lale. `Aggregate`_
* lale.lib.lale. `GroupBy`_
* lale.lib.lale. `Map`_
* lale.lib.lale. `Join`_
* lale.lib.lale. `Scan`_
* lale.lib.lale. `Relational`_

Estimators and transformers:

* lale.lib.lale. `Both`_
* lale.lib.lale. `IdentityWrapper`_
* lale.lib.lale. `Observing`_

.. _`AutoPipeline`: lale.lib.lale.auto_pipeline.html
.. _`GridSearchCV`: lale.lib.lale.grid_search_cv.html
.. _`HalvingGridSearchCV`: lale.lib.lale.halving_grid_search_cv.html
.. _`Hyperopt`: lale.lib.lale.hyperopt.html
.. _`TopKVotingClassifier`: lale.lib.lale.topk_voting_classifier.html
.. _`SMAC`: lale.lib.lale.smac.html
.. _`Batching`: lale.lib.lale.batching.html
.. _`ConcatFeatures`: lale.lib.lale.concat_features.html
.. _`NoOp`: lale.lib.lale.no_op.html
.. _`Project`: lale.lib.lale.project.html
.. _`SampleBasedVoting`: lale.lib.lale.sample_based_voting.html
.. _`Aggregate`: lale.lib.lale.aggregate.html
.. _`GroupBy`: lale.lib.lale.group_by.html
.. _`Map`: lale.lib.lale.map.html
.. _`Join`: lale.lib.lale.join.html
.. _`Scan`: lale.lib.lale.scan.html
.. _`Relational`: lale.lib.lale.relational.html
.. _`Both`: lale.lib.lale.both.html
.. _`IdentityWrapper`: lale.lib.lale.identity_wrapper.html
.. _`Observing`: lale.lib.lale.observing.html

Functions:
==========

* lale.lib.lale. `categorical`_
* lale.lib.lale. `date_time`_

.. _`categorical`: lale.lib.lale.functions.html#lale.lib.lale.functions.categorical
.. _`date_time`: lale.lib.lale.functions.html#lale.lib.lale.functions.date_time
"""

from .aggregate import Aggregate

# estimators
from .auto_pipeline import AutoPipeline

# transformers
from .batching import Batching

# estimators and transformers
from .both import Both
from .concat_features import ConcatFeatures

# functions
from .functions import categorical, date_time
from .grid_search_cv import GridSearchCV
from .group_by import GroupBy
from .halving_grid_search_cv import HalvingGridSearchCV
from .hyperopt import Hyperopt
from .identity_wrapper import IdentityWrapper
from .join import Join
from .map import Map
from .no_op import NoOp
from .observing import Observing
from .project import Project
from .relational import Relational
from .sample_based_voting import SampleBasedVoting
from .scan import Scan
from .smac import SMAC
from .topk_voting_classifier import TopKVotingClassifier
