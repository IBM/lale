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

from abc import ABC, abstractmethod
from typing import Any, Generic, Optional, TypeVar

_InputType = TypeVar("_InputType", contravariant=True)
_OutputType = TypeVar("_OutputType", covariant=True)
_SelfType = TypeVar("_SelfType")


class Monoid(ABC):
    @abstractmethod
    def combine(self: _SelfType, other: _SelfType) -> _SelfType:
        pass


_M = TypeVar("_M", bound=Monoid)


class MonoidFactory(ABC, Generic[_InputType, _OutputType, _M]):
    @abstractmethod
    def _to_monoid(self, v: _InputType) -> _M:
        pass

    @abstractmethod
    def _from_monoid(self, v: _M) -> _OutputType:
        pass


class MonoidableOperator(MonoidFactory[Any, None, _M]):
    _monoid: Optional[_M] = None

    def partial_fit(self, X, y=None):
        lifted = self._to_monoid((X, y))
        if self._monoid is not None:  # not first fit
            lifted = self._monoid.combine(lifted)
        self._from_monoid(lifted)
        return self

    def fit(self, X, y=None):
        lifted = self._to_monoid((X, y))
        self._from_monoid(lifted)
        return self
