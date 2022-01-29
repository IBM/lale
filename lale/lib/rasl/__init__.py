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
* lale.lib.rasl. `Map`_

Scikit-learn Operators
======================

* lale.lib.rasl. `MinMaxScaler`_
* lale.lib.rasl. `OneHotEncoder`_
* lale.lib.rasl. `OrdinalEncoder`_
* lale.lib.rasl. `SimpleImputer`_
* lale.lib.rasl. `StandardScaler`_

Other Facilities
================

* lale.lib.rasl. `Prio`_
* lale.lib.rasl. `PrioBatch`_
* lale.lib.rasl. `PrioStep`_
* lale.lib.rasl. `cross_val_score`_
* lale.lib.rasl. `fit_with_batches`_
* lale.lib.rasl. `is_associative`_
* lale.lib.rasl. `is_incremental`_
* lale.lib.rasl. `mockup_data_loader`_

.. _`Aggregate`: lale.lib.rasl.aggregate.html
.. _`Map`: lale.lib.rasl.map.html

.. _`MinMaxScaler`: lale.lib.rasl.min_max_scaler.html
.. _`OneHotEncoder`: lale.lib.rasl.one_hot_encoder.html
.. _`OrdinalEncoder`: lale.lib.rasl.ordinal_encoder.html
.. _`SimpleImputer`: lale.lib.rasl.simple_imputer.html
.. _`StandardScaler`: lale.lib.rasl.standard_scaler.html

.. _`Prio`: lale.lib.rasl.Prio.html
.. _`PrioBatch`: lale.lib.rasl.PrioBatch.html
.. _`PrioStep`: lale.lib.rasl.PrioStep.html
.. _`cross_val_score`: lale.lib.rasl.cross_val_score.html
.. _`fit_with_batches`: lale.lib.rasl.fit_with_batches.html
.. _`is_associative`: lale.lib.rasl.is_associative.html
.. _`is_incremental`: lale.lib.rasl.is_incremental.html
.. _`mockup_data_loader`: lale.lib.rasl.mockup_data_loader.html

"""

from ._task_graphs import (
    Prio,
    PrioBatch,
    PrioStep,
    cross_val_score,
    fit_with_batches,
    is_associative,
    is_incremental,
    mockup_data_loader,
)
from .aggregate import Aggregate
from .map import Map
from .min_max_scaler import MinMaxScaler
from .one_hot_encoder import OneHotEncoder
from .ordinal_encoder import OrdinalEncoder
from .simple_imputer import SimpleImputer
from .standard_scaler import StandardScaler
