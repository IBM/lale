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
from typing import Any, Generic, TypeVar

_InputType = TypeVar("_InputType", contravariant=True)
_OutputType = TypeVar("_OutputType", covariant=True)
_SelfType = TypeVar("_SelfType")


class Monoid(ABC, Generic[_OutputType]):
    @abstractmethod
    def combine(self: _SelfType, other: _SelfType) -> _SelfType:
        pass

    @property
    @abstractmethod
    def result(self) -> _OutputType:
        pass


class MonoidMaker(ABC, Generic[_InputType, _OutputType]):
    @abstractmethod
    def to_monoid(self, v: _InputType) -> Monoid[_OutputType]:
        pass


class MonoidableOperator(MonoidMaker[Any, "MonoidableOperator"]):
    def partial_fit(self):
        # generic implementation here
        pass

    def fit(self):
        # generic implementation here
        pass

    @abstractmethod
    def to_monoid(self: _SelfType, v: Any) -> Monoid[_SelfType]:
        pass
