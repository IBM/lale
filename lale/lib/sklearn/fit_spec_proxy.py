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


class _FitSpecProxy:
    def __init__(self, base):
        self._base = base

    def __getattr__(self, item):
        return getattr(self._base, item)

    def get_params(self, deep=True):
        ret = {}
        ret["base"] = self._base
        return ret

    def fit(self, X, y, sample_weight=None, **fit_params):
        # the purpose of this is to have an explicit sample_weights argument,
        # since sklearn sometimes uses reflection to check whether it is there
        return self._base.fit(X, y, sample_weight=sample_weight, **fit_params)
        # not returning self, because self is not a full-fledged operator
