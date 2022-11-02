# Copyright 2022 IBM Corporation
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
Schema-enhanced versions of some of the operators from `category_encoders`_ to enable hyperparameter tuning.

.. _`category_encoders`: https://contrib.scikit-learn.org/category_encoders

Operators
=========

* lale.lib.category_encoders. `HashingEncoder`_
* lale.lib.category_encoders. `TargetEncoder`_

.. _`HashingEncoder`: lale.lib.category_encoders.hashing_encoder.html
.. _`TargetEncoder`: lale.lib.category_encoders.target_encoder.html

"""

# Note: all imports should be done as
# from .xxx import XXX as XXX
# this ensures that pyright considers them to be publicly available
# and not private imports (this affects lale users that use pyright)

from .hashing_encoder import HashingEncoder as HashingEncoder
from .target_encoder import TargetEncoder as TargetEncoder
