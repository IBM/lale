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

import lale.operators
from lale.lib.rasl.concat_features import _combined_schemas
from lale.lib.rasl.concat_features import _ConcatFeaturesImpl as _RaslConcatFeaturesImpl


class _ConcatFeaturesImpl(_RaslConcatFeaturesImpl):
    def __init__(self):
        super().__init__()


ConcatFeatures = lale.operators.make_pretrained_operator(
    _ConcatFeaturesImpl, _combined_schemas
)
