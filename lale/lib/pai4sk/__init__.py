# Copyright 2020 IBM Corporation
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
Schema-enhanced versions of some of the operators from `Snap ML`_ to enable hyperparameter tuning.

.. _`Snap ML`: https://www.zurich.ibm.com/snapml/

Operators
=========

Classifiers:

* lale.lib.sklearn. `RandomForestClassifier`_

.. _`RandomForestClassifier`: lale.lib.pai4sk.random_forest_classifier.html
"""

from .random_forest_classifier import RandomForestClassifier
