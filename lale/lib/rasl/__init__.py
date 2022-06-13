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

.. _`MinMaxScaler`: lale.lib.rasl.min_max_scaler.html
.. _`OneHotEncoder`: lale.lib.rasl.one_hot_encoder.html
.. _`OrdinalEncoder`: lale.lib.rasl.ordinal_encoder.html
.. _`HashingEncoder`: lale.lib.rasl.hashing_encoder.html
.. _`SelectKBest`: lale.lib.rasl.select_k_best.html
.. _`SimpleImputer`: lale.lib.rasl.simple_imputer.html
.. _`StandardScaler`: lale.lib.rasl.standard_scaler.html

.. _`BaggingMonoidClassifier`: lale.lib.rasl.batched_bagging_classifier.html

.. _`categorical`: lale.lib.rasl.functions.html#lale.lib.rasl.functions.categorical
.. _`date_time`: lale.lib.rasl.functions.html#lale.lib.rasl.functions.date_time
.. _`SparkExplainer`: lale.lib.rasl.spark_explainer.html

.. _`Prio`: lale.lib.rasl.Prio.html
.. _`PrioBatch`: lale.lib.rasl.PrioBatch.html
.. _`PrioResourceAware`: lale.lib.rasl.PrioResourceAware.html
.. _`PrioStep`: lale.lib.rasl.PrioStep.html
.. _`accuracy_score`: lale.lib.rasl.accuracy_score.html
.. _`cross_val_score`: lale.lib.rasl.cross_val_score.html
.. _`cross_validate`: lale.lib.rasl.cross_validate.html
.. _`f1_score`: lale.lib.rasl.f1_score.html
.. _`fit_with_batches`: lale.lib.rasl.fit_with_batches.html
.. _`get_scorer`: lale.lib.rasl.get_scorer.html
.. _`is_associative`: lale.lib.rasl.is_associative.html
.. _`is_incremental`: lale.lib.rasl.is_incremental.html
.. _`csv_data_loader`: lale.lib.rasl.datasets.html#lale.lib.rasl.datasets.csv_data_loader
.. _`mockup_data_loader`: lale.lib.rasl.datasets.html#lale.lib.rasl.datasets.mockup_data_loader
.. _`openml_data_loader`: lale.lib.rasl.datasets.html#lale.lib.rasl.datasets.openml_data_loader
.. _`r2_score`: lale.lib.rasl.r2_score.html
"""

from ._task_graphs import (
    Prio,
    PrioBatch,
    PrioResourceAware,
    PrioStep,
    cross_val_score,
    cross_validate,
    fit_with_batches,
    is_associative,
    is_incremental,
)
from .aggregate import Aggregate
from .alias import Alias
from .batched_bagging_classifier import BatchedBaggingClassifier
from .batching import Batching
from .concat_features import ConcatFeatures
from .convert import Convert
from .datasets import csv_data_loader, mockup_data_loader, openml_data_loader
from .filter import Filter
from .functions import categorical, date_time
from .group_by import GroupBy
from .hashing_encoder import HashingEncoder
from .join import Join
from .map import Map
from .metrics import accuracy_score, f1_score, get_scorer, r2_score
from .min_max_scaler import MinMaxScaler
from .monoid import Monoid, MonoidableOperator, MonoidFactory
from .one_hot_encoder import OneHotEncoder
from .orderby import OrderBy
from .ordinal_encoder import OrdinalEncoder
from .project import Project
from .relational import Relational
from .scan import Scan
from .select_k_best import SelectKBest
from .simple_imputer import SimpleImputer
from .spark_explainer import SparkExplainer
from .split_xy import SplitXy
from .standard_scaler import StandardScaler
