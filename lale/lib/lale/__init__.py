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
* lale.lib.lale. `OptimizeLast`_
* lale.lib.lale. `OptimizeSuffix`_
* lale.lib.lale. `SMAC`_
* lale.lib.lale. `TopKVotingClassifier`_

Transformers:

* lale.lib.rasl. `Aggregate`_
* lale.lib.lale. `Alias`_
* lale.lib.lale. `Batching`_
* lale.lib.lale. `ConcatFeatures`_
* lale.lib.lale. `Filter`_
* lale.lib.lale. `GroupBy`_
* lale.lib.lale. `Join`_
* lale.lib.rasl. `Map`_
* lale.lib.lale. `NoOp`_
* lale.lib.lale. `OrderBy`_
* lale.lib.lale. `Project`_
* lale.lib.lale. `Relational`_
* lale.lib.lale. `SampleBasedVoting`_
* lale.lib.lale. `Scan`_
* lale.lib.lale. `SplitXy`_
* lale.lib.lale. `Tee`_

Estimators and transformers:

* lale.lib.lale. `Both`_
* lale.lib.lale. `IdentityWrapper`_
* lale.lib.lale. `Observing`_

.. _`AutoPipeline`: lale.lib.lale.auto_pipeline.html
.. _`GridSearchCV`: lale.lib.lale.grid_search_cv.html
.. _`HalvingGridSearchCV`: lale.lib.lale.halving_grid_search_cv.html
.. _`Hyperopt`: lale.lib.lale.hyperopt.html
.. _`OptimizeLast`: lale.lib.lale.optimize_last.html
.. _`OptimizeSuffix`: lale.lib.lale.optimize_suffix.html

.. _`TopKVotingClassifier`: lale.lib.lale.topk_voting_classifier.html
.. _`SMAC`: lale.lib.lale.smac.html
.. _`Batching`: lale.lib.lale.batching.html
.. _`ConcatFeatures`: lale.lib.lale.concat_features.html
.. _`NoOp`: lale.lib.lale.no_op.html
.. _`Project`: lale.lib.lale.project.html
.. _`SampleBasedVoting`: lale.lib.lale.sample_based_voting.html
.. _`Aggregate`: lale.lib.rasl.aggregate.html
.. _`Filter`: lale.lib.lale.filter.html
.. _`GroupBy`: lale.lib.lale.group_by.html
.. _`Map`: lale.lib.rasl.map.html
.. _`OrderBy`: lale.lib.lale.orderby.html
.. _`Join`: lale.lib.lale.join.html
.. _`Alias`: lale.lib.lale.alias.html
.. _`Scan`: lale.lib.lale.scan.html
.. _`SplitXy`: lale.lib.lale.split_xy.html
.. _`Relational`: lale.lib.lale.relational.html
.. _`Both`: lale.lib.lale.both.html
.. _`IdentityWrapper`: lale.lib.lale.identity_wrapper.html
.. _`Observing`: lale.lib.lale.observing.html
.. _`Tee`: lale.lib.lale.tee.html

Functions:
==========

* lale.lib.lale. `categorical`_
* lale.lib.lale. `date_time`_
* SparkExplainer. `spark_explainer`_

.. _`categorical`: lale.lib.lale.functions.html#lale.lib.lale.functions.categorical
.. _`date_time`: lale.lib.lale.functions.html#lale.lib.lale.functions.date_time
.. _`spark_explainer`: lale.lib.lale.spark_explainer.html
"""

from lale.lib.rasl import Aggregate, Map

from .alias import Alias

# estimators
from .auto_pipeline import AutoPipeline

# transformers
from .batching import Batching

# estimators and transformers
from .both import Both
from .concat_features import ConcatFeatures
from .filter import Filter

# functions
from .functions import categorical, date_time
from .grid_search_cv import GridSearchCV
from .group_by import GroupBy
from .halving_grid_search_cv import HalvingGridSearchCV
from .hyperopt import Hyperopt
from .identity_wrapper import IdentityWrapper
from .join import Join
from .no_op import NoOp
from .observing import Observing
from .optimize_last import OptimizeLast
from .optimize_suffix import OptimizeSuffix
from .orderby import OrderBy
from .project import Project
from .relational import Relational
from .sample_based_voting import SampleBasedVoting
from .scan import Scan
from .smac import SMAC
from .spark_explainer import SparkExplainer
from .split_xy import SplitXy
from .tee import Tee
from .topk_voting_classifier import TopKVotingClassifier
