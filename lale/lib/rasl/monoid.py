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

from typing_extensions import Protocol, runtime_checkable

_InputType = TypeVar("_InputType", contravariant=True)
_OutputType = TypeVar("_OutputType", covariant=True)
_SelfType = TypeVar("_SelfType")


class Monoid(ABC):
    """
    Data that can be combined in an associative way.  See :class:MonoidFactory for ways to create/unpack
    a given monoid.
    """

    @abstractmethod
    def combine(self: _SelfType, other: _SelfType) -> _SelfType:
        """
         Combines this monoid instance with another, producing a result.
         This operation must be observationally associative, satisfying
         ``x.from_monoid(a.combine(b.combine(c))) == x.from_monoid(a.combine(b).combine(c)))``
        where `x` is the instance of :class:MonoidFactory that created these instances.
        """
        pass

    @property
    def is_absorbing(self):
        """
        A monoid value `x` is absorbing if for all `y`, `x.combine(y) == x`.
        This can help stop training early for monoids with learned coefficients.
        """
        return False


_M = TypeVar("_M", bound=Monoid)


@runtime_checkable
class MonoidFactory(Generic[_InputType, _OutputType, _M], Protocol):
    """
    This protocol determines if a class supports creating a monoid and using it
    to support associative computation.
    Due to the ``runtime_checkable`` decorator, ``isinstance(obj, MonoidFactory)`` will succeed
    if the object has the requisite methods, even if it does not have this protocol as
    a base class.
    """

    def to_monoid(self, v: _InputType) -> _M:
        """
        Create a monoid instance representing the input data
        """
        ...

    def from_monoid(self, v: _M) -> _OutputType:
        """
        Given the monoid instance, return the appropriate type of output.
        This method may also modify self based on the monoid instance.
        """
        ...


class MonoidableOperator(MonoidFactory[Any, None, _M], Protocol):
    """
    This is a useful base class for operator implementations that support associative (monoid-based) fit.
    Given the implementation supplied :class:MonoidFactory methods, this class provides
    default :method:partial_fit and :method:fit implementations.
    """

    _monoid: Optional[_M] = None

    def partial_fit(self, X, y=None):
        if self._monoid is None or not self._monoid.is_absorbing:
            lifted = self.to_monoid((X, y))
            if self._monoid is not None:  # not first fit
                lifted = self._monoid.combine(lifted)
            self.from_monoid(lifted)
        return self

    def fit(self, X, y=None):
        lifted = self.to_monoid((X, y))
        self.from_monoid(lifted)
        return self
