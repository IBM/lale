# Copyright 2021 IBM Corporation
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
RASL operators.

Relational Algebra Operators
============================

* lale.lib.rasl. `Aggregate`_
* lale.lib.rasl. `Map`_


Scikit-learn Operators
======================

* lale.lib.rasl. `MinMaxScaler`_
* lale.lib.rasl. `OneHotEncoder`_
* lale.lib.rasl. `OrdinalEncoder`_
* lale.lib.rasl. `StandardScaler`_


.. _`Aggregate`: lale.lib.rasl.aggregate.html
.. _`Map`: lale.lib.rasl.map.html
.. _`MinMaxScaler`: lale.lib.rasl.min_max_scaler.html
.. _`OneHotEncoder`: lale.lib.rasl.one_hot_encoder.html
.. _`OrdinalEncoder`: lale.lib.rasl.ordinal_encoder.html
.. _`StandardScaler`: lale.lib.rasl.standard_scaler.html

"""

from .aggregate import Aggregate
from .map import Map
from .min_max_scaler import MinMaxScaler
from .one_hot_encoder import OneHotEncoder
from .ordinal_encoder import OrdinalEncoder
from .standard_scaler import StandardScaler
