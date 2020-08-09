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
Scikit-learn compatible wrappers for some operators from TensorFlow_ along with schemas to enable hyperparameter tuning.

.. _TensorFlow: https://www.tensorflow.org/

Operators:
==========
* `USEPretrainedEncoder`_

.. _`USEPretrainedEncoder`: lale.lib.tensorflow.use_pretrained_encoder.html
"""

from .use_pretrained_encoder import USEPretrainedEncoder
