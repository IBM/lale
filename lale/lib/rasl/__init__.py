# Copyright 2021, 2022 IBM Corporation
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
RASL operators and functions (experimental).

Relational Algebra Operators
============================

* lale.lib.rasl. `Aggregate`_
* lale.lib.rasl. `Alias`_
* lale.lib.rasl. `Filter`_
* lale.lib.rasl. `GroupBy`_
* lale.lib.rasl. `Join`_
* lale.lib.rasl. `Map`_
* lale.lib.rasl. `OrderBy`_
* lale.lib.rasl. `Project`_
* lale.lib.rasl. `Relational`_

Transformers
============

* lale.lib.rasl. `Batching`_
* lale.lib.rasl. `ConcatFeatures`_
* lale.lib.rasl. `Convert`_
* lale.lib.rasl. `Scan`_
* lale.lib.rasl. `SortIndex`_
* lale.lib.rasl. `SplitXy`_

Scikit-learn Operators
======================

* lale.lib.rasl. `MinMaxScaler`_
* lale.lib.rasl. `OneHotEncoder`_
* lale.lib.rasl. `OrdinalEncoder`_
* lale.lib.rasl. `HashingEncoder`_
* lale.lib.rasl. `SelectKBest`_
* lale.lib.rasl. `SimpleImputer`_
* lale.lib.rasl. `StandardScaler`_

Estimators
==========

* lale.lib.rasl. `BatchedBaggingClassifier`_

Functions
=========

* lale.lib.rasl. `categorical`_
* lale.lib.rasl. `date_time`_
* lale.lib.rasl. `SparkExplainer`_

Data Loaders
============
* lale.lib.rasl. `csv_data_loader`_
* lale.lib.rasl. `mockup_data_loader`_
* lale.lib.rasl. `openml_data_loader`_

Metrics
=======
* lale.lib.rasl. `accuracy_score`_
* lale.lib.rasl. `balanced_accuracy_score`_
* lale.lib.rasl. `f1_score`_
* lale.lib.rasl. `get_scorer`_
* lale.lib.rasl. `r2_score`_

Other Facilities
================
* lale.lib.rasl. `Prio`_
* lale.lib.rasl. `PrioBatch`_
* lale.lib.rasl. `PrioResourceAware`_
* lale.lib.rasl. `PrioStep`_
* lale.lib.rasl. `cross_val_score`_
* lale.lib.rasl. `cross_validate`_
* lale.lib.rasl. `fit_with_batches`_
* lale.lib.rasl. `is_associative`_
* lale.lib.rasl. `is_incremental`_

.. _`Aggregate`: lale.lib.rasl.aggregate.html
.. _`Alias`: lale.lib.rasl.alias.html
.. _`Filter`: lale.lib.rasl.filter.html
.. _`GroupBy`: lale.lib.rasl.group_by.html
.. _`Join`: lale.lib.rasl.join.html
.. _`Map`: lale.lib.rasl.map.html
.. _`OrderBy`: lale.lib.rasl.orderby.html
.. _`Project`: lale.lib.rasl.project.html
.. _`Relational`: lale.lib.rasl.relational.html

.. _`BatchedBaggingClassifier`: lale.lib.rasl.batched_bagging_classifier.html
.. _`Batching`: lale.lib.rasl.batching.html
.. _`ConcatFeatures`: lale.lib.rasl.concat_features.html
.. _`Convert`: lale.lib.rasl.convert.html
.. _`Scan`: lale.lib.rasl.scan.html
.. _`SplitXy`: lale.lib.rasl.split_xy.html
.. _`SortIndex`: lale.lib.rasl.sort_index.html

.. _`MinMaxScaler`: lale.lib.rasl.min_max_scaler.html
.. _`OneHotEncoder`: lale.lib.rasl.one_hot_encoder.html
.. _`OrdinalEncoder`: lale.lib.rasl.ordinal_encoder.html
.. _`HashingEncoder`: lale.lib.rasl.hashing_encoder.html
.. _`SelectKBest`: lale.lib.rasl.select_k_best.html
.. _`SimpleImputer`: lale.lib.rasl.simple_imputer.html
.. _`StandardScaler`: lale.lib.rasl.standard_scaler.html

.. _`categorical`: lale.lib.rasl.functions.html#lale.lib.rasl.functions.categorical
.. _`date_time`: lale.lib.rasl.functions.html#lale.lib.rasl.functions.date_time
.. _`SparkExplainer`: lale.lib.rasl.spark_explainer.html

.. _`Prio`: lale.lib.rasl.task_graphs.html#lale.lib.rasl.task_graphs.Prio
.. _`PrioBatch`: lale.lib.rasl.task_graphs.html#lale.lib.rasl.task_graphs.PrioBatch
.. _`PrioResourceAware`: lale.lib.rasl.task_graphs.html#lale.lib.rasl.task_graphs.PrioResourceAware
.. _`PrioStep`: lale.lib.rasl.task_graphs.html#lale.lib.rasl.task_graphs.PrioStep
.. _`accuracy_score`: lale.lib.rasl.metrics.html#lale.lib.rasl.metrics.accuracy_score
.. _`balanced_accuracy_score`: lale.lib.rasl.metrics.html#lale.lib.rasl.metrics.balanced_accuracy_score
.. _`cross_val_score`: lale.lib.rasl.task_graphs.html#lale.lib.rasl.task_graphs.cross_val_score
.. _`cross_validate`: lale.lib.rasl.task_graphs.html#lale.lib.rasl.task_graphs.cross_validate
.. _`f1_score`: lale.lib.rasl.metrics.html#lale.lib.rasl.metrics.f1_score
.. _`fit_with_batches`: lale.lib.rasl.task_graphs.html#lale.lib.rasl.task_graphs.fit_with_batches
.. _`get_scorer`: lale.lib.rasl.metrics.html#lale.lib.rasl.metrics.get_scorer
.. _`is_associative`: lale.lib.rasl.task_graphs.html#lale.lib.rasl.task_graphs.is_associative
.. _`is_incremental`: lale.lib.rasl.task_graphs.html#lale.lib.rasl.task_graphs.is_incremental
.. _`csv_data_loader`: lale.lib.rasl.datasets.html#lale.lib.rasl.datasets.csv_data_loader
.. _`mockup_data_loader`: lale.lib.rasl.datasets.html#lale.lib.rasl.datasets.mockup_data_loader
.. _`openml_data_loader`: lale.lib.rasl.datasets.html#lale.lib.rasl.datasets.openml_data_loader
.. _`r2_score`: lale.lib.rasl.metrics.html#lale.lib.rasl.metrics.r2_score
"""

# Note: all imports should be done as
# from .xxx import XXX as XXX
# this ensures that pyright considers them to be publicly available
# and not private imports (this affects lale users that use pyright)

from .aggregate import Aggregate as Aggregate
from .alias import Alias as Alias
from .batched_bagging_classifier import (
    BatchedBaggingClassifier as BatchedBaggingClassifier,
)
from .batching import Batching as Batching
from .concat_features import ConcatFeatures as ConcatFeatures
from .convert import Convert as Convert
from .datasets import csv_data_loader as csv_data_loader
from .datasets import mockup_data_loader as mockup_data_loader
from .datasets import openml_data_loader as openml_data_loader
from .filter import Filter as Filter
from .functions import categorical, date_time
from .group_by import GroupBy as GroupBy
from .hashing_encoder import HashingEncoder as HashingEncoder
from .join import Join as Join
from .map import Map as Map
from .metrics import accuracy_score as accuracy_score
from .metrics import balanced_accuracy_score as balanced_accuracy_score
from .metrics import f1_score as f1_score
from .metrics import get_scorer as get_scorer
from .metrics import r2_score as r2_score
from .min_max_scaler import MinMaxScaler as MinMaxScaler
from .monoid import Monoid as Monoid
from .monoid import MonoidableOperator as MonoidableOperator
from .monoid import MonoidFactory as MonoidFactory
from .one_hot_encoder import OneHotEncoder as OneHotEncoder
from .orderby import OrderBy as OrderBy
from .ordinal_encoder import OrdinalEncoder as OrdinalEncoder
from .project import Project as Project
from .relational import Relational as Relational
from .scan import Scan as Scan
from .select_k_best import SelectKBest as SelectKBest
from .simple_imputer import SimpleImputer as SimpleImputer
from .sort_index import SortIndex as SortIndex
from .spark_explainer import SparkExplainer as SparkExplainer
from .split_xy import SplitXy as SplitXy
from .standard_scaler import StandardScaler as StandardScaler
from .task_graphs import Prio as Prio
from .task_graphs import PrioBatch as PrioBatch
from .task_graphs import PrioResourceAware as PrioResourceAware
from .task_graphs import PrioStep as PrioStep
from .task_graphs import cross_val_score as cross_val_score
from .task_graphs import cross_validate as cross_validate
from .task_graphs import fit_with_batches as fit_with_batches
from .task_graphs import is_associative as is_associative
from .task_graphs import is_incremental as is_incremental
