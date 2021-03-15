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

import abc
import os
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple, Union

import numpy

from lale.search.PGO import FrequencyDistribution
from lale.util import VisitorPathError
from lale.util.VisitorMeta import AbstractVisitorMeta

PGO_input_type = Union[FrequencyDistribution, Iterable[Tuple[Any, int]], None]


class SearchSpaceError(VisitorPathError):
    def __init__(self, sub_path: Any, message: Optional[str] = None):
        super().__init__([], message)

        self.sub_path = sub_path

    def path_string(self) -> str:
        return SearchSpace.focused_path_string(list(self.path))

    def get_message_str(self) -> str:
        msg = super().get_message_str()
        if self.sub_path is None:
            return msg
        else:
            return f"for path {self.sub_path}: {msg}"


class SearchSpace(metaclass=AbstractVisitorMeta):
    def __init__(self, default: Optional[Any] = None):
        self._default = default

    _default: Optional[Any]

    def default(self) -> Optional[Any]:
        """Return an optional default value, if None.
        if not None, the default value should be in the
        search space
        """
        return self._default

    @classmethod
    def focused_path_string(cls, path: List["SearchSpace"]) -> str:
        if path:
            return path[0].str_with_focus(path, default="")
        else:
            return ""

    def str_with_focus(
        self, path: Optional[List["SearchSpace"]] = None, default: Any = None
    ) -> Union[str, Any]:
        """Given a path list, returns a string for the focused path.
        If the path is None, returns everything, without focus.
        If the path does not start with self, returns None
        """
        if path is None:
            return self._focused_str(path=None)
        elif path and path[0] is self:
            return self._focused_str(path=path[1:])
        else:
            return default

    @abc.abstractmethod
    def _focused_str(self, path: Optional[List["SearchSpace"]] = None) -> str:
        """Given the continuation path list, returns a string for the focused path.
        If the path is None, returns everything, without focus.
        Otherwise, the path is for children
        """
        pass

    def __str__(self) -> str:
        return self.str_with_focus(path=None, default="")


class SearchSpaceEmpty(SearchSpace):
    def __init__(self):
        super(SearchSpaceEmpty, self).__init__()

    def _focused_str(self, path: Optional[List[SearchSpace]] = None) -> str:
        return "***EMPTY***"


class SearchSpacePrimitive(SearchSpace):
    def __init__(self, default: Optional[Any] = None):
        super(SearchSpacePrimitive, self).__init__(default=default)


class SearchSpaceEnum(SearchSpacePrimitive):
    pgo: Optional[FrequencyDistribution]
    vals: List[Any]

    def __init__(
        self,
        vals: Iterable[Any],
        pgo: PGO_input_type = None,
        default: Optional[Any] = None,
    ):
        super(SearchSpaceEnum, self).__init__(default=default)
        self.vals = sorted(vals, key=str)

        if pgo is None or isinstance(pgo, FrequencyDistribution):
            self.pgo = pgo
        else:
            self.pgo = FrequencyDistribution.asEnumValues(pgo, self.vals)

    def _focused_str(self, path: Optional[List[SearchSpace]] = None) -> str:
        return "<" + ",".join(map(str, self.vals)) + ">"


class SearchSpaceConstant(SearchSpaceEnum):
    def __init__(self, v, pgo: PGO_input_type = None):
        super(SearchSpaceConstant, self).__init__([v], pgo=pgo, default=v)

    def _focused_str(self, path: Optional[List[SearchSpace]] = None) -> str:
        return str(self.vals[0])


class SearchSpaceBool(SearchSpaceEnum):
    def __init__(self, pgo: PGO_input_type = None, default: Optional[Any] = None):
        super(SearchSpaceBool, self).__init__([True, False], pgo=pgo, default=default)


class SearchSpaceNumber(SearchSpacePrimitive):
    minimum: Optional[float]
    exclusiveMinumum: bool
    maximum: Optional[float]
    exclusiveMaximum: bool
    discrete: bool
    distribution: str
    pgo: Optional[FrequencyDistribution]

    def __init__(
        self,
        minimum=None,
        exclusiveMinimum: bool = False,
        maximum=None,
        exclusiveMaximum: bool = False,
        discrete: bool = False,
        distribution="uniform",
        pgo: PGO_input_type = None,
        default: Optional[Any] = None,
    ) -> None:
        super(SearchSpaceNumber, self).__init__(default=default)
        self.minimum = minimum
        self.exclusiveMinimum = exclusiveMinimum
        self.maximum = maximum
        self.exclusiveMaximum = exclusiveMaximum
        self.distribution = distribution
        self.discrete = discrete
        if pgo is None or isinstance(pgo, FrequencyDistribution):
            self.pgo = pgo
        else:
            if discrete:
                self.pgo = FrequencyDistribution.asIntegerValues(
                    pgo,
                    inclusive_min=self.getInclusiveMin(),
                    inclusive_max=self.getInclusiveMax(),
                )
            else:
                self.pgo = FrequencyDistribution.asFloatValues(
                    pgo,
                    inclusive_min=self.getInclusiveMin(),
                    inclusive_max=self.getInclusiveMax(),
                )

    def getInclusiveMax(self) -> Optional[float]:
        """Return the maximum as an inclusive maximum (exclusive maxima are adjusted accordingly)"""
        max = self.maximum
        if max is None:
            return None
        if self.exclusiveMaximum:
            if self.discrete:
                max = max - 1
            else:
                max = numpy.nextafter(max, float("-inf"))
        return max

    def getInclusiveMin(self) -> Optional[float]:
        """Return the maximum as an inclusive minimum (exclusive minima are adjusted accordingly)"""
        min = self.minimum
        if min is None:
            return None
        if self.exclusiveMinimum:
            if self.discrete:
                min = min + 1
            else:
                min = numpy.nextafter(min, float("+inf"))
        return min

    def _focused_str(self, path: Optional[List[SearchSpace]] = None) -> str:
        ret: str = ""
        if self.exclusiveMinimum or self.minimum is None:
            ret += "("
        else:

            ret += "["
        if self.discrete:
            ret += "\u2308"

        if self.minimum is None:
            ret += "\u221E"
        else:
            ret += str(self.minimum)

        if (
            not self.distribution
            or self.distribution == "uniform"
            or self.distribution == "integer"
        ):
            ret += ","
        elif self.distribution == "loguniform":
            ret += ",<log>,"
        else:
            ret += ",<" + self.distribution + ">,"

        if self.maximum is None:
            ret += "\u221E"
        else:
            ret += str(self.maximum)

        if self.discrete:
            ret += "\u2309"
        if self.exclusiveMaximum or self.maximum is None:
            ret += ")"
        else:
            ret += "]"
        return ret


class SearchSpaceArray(SearchSpace):
    def __init__(
        self,
        prefix: Optional[List[SearchSpace]],
        minimum: int = 0,
        *,
        maximum: int,
        additional: Optional[SearchSpace] = None,
        is_tuple=False,
    ) -> None:
        super(SearchSpaceArray, self).__init__()
        self.minimum = minimum
        self.maximum = maximum
        self.prefix = prefix
        self.additional = additional
        self.is_tuple = is_tuple

    def _focused_str(self, path: Optional[List[SearchSpace]] = None) -> str:
        ret: str = ""
        ret += f"Array<{self.minimum}, {self.maximum}>"
        if self.is_tuple:
            ret += "("
        else:
            ret += "["

        if self.prefix is not None:
            ret += ",".join(
                p.str_with_focus(path=path, default="") for p in self.prefix
            )
            if self.additional is not None:
                ret += ","
        if self.additional is not None:
            ret += "...,"
            ret += self.additional.str_with_focus(path=path, default="")

        if self.is_tuple:
            ret += ")"
        else:
            ret += "]"
        return ret

    def items(self, max: Optional[int] = None) -> Iterable[SearchSpace]:
        prefix_len: int
        if self.prefix is not None:
            prefix_len = len(self.prefix)
        else:
            prefix_len = 0

        num_elts = self.maximum
        if max is not None:
            num_elts = min(num_elts, max)

        for i in range(num_elts):
            if self.prefix is not None and i < prefix_len:
                yield self.prefix[i]
            else:
                if self.additional is not None:
                    yield self.additional


class SearchSpaceDict(SearchSpace):
    def __init__(self, d: Dict[str, SearchSpace]) -> None:
        super(SearchSpaceDict, self).__init__()
        self.space_dict = d

    def _focused_str(self, path: Optional[List[SearchSpace]] = None) -> str:
        ret: str = ""
        ret += "Dict{"
        dict_strs: List[str] = []
        for k, v in self.space_dict.items():
            dict_strs.append(k + "->" + v.str_with_focus(path=path, default=None))
        ret += ",".join(dict_strs) + "}"
        return ret


class SearchSpaceObject(SearchSpace):
    def __init__(self, longName: str, keys: List[str], choices: Iterable[Any]) -> None:
        super(SearchSpaceObject, self).__init__()
        self.longName = longName
        self.keys = keys
        self.choices = choices

    def _focused_str(self, path: Optional[List[SearchSpace]] = None) -> str:
        ret: str = ""
        ret += f"Object<{self.longName}>["

        choice_strs: List[str] = []
        for c in self.choices:
            opts: List[str] = []
            for k, v in zip(self.keys, c):
                vv = v.str_with_focus(path=path, default=None)
                if vv is not None:
                    opts.append(k + "->" + vv)
            if opts:
                ll = ";".join(opts)
                choice_strs.append("{" + ll + "}")
            else:
                choice_strs.append("")

        ret += ",".join(choice_strs) + "]"

        return ret


class SearchSpaceSum(SearchSpace):
    sub_spaces: List[SearchSpace]

    def __init__(self, sub_spaces: List[SearchSpace], default: Optional[Any] = None):
        super(SearchSpaceSum, self).__init__(default=default)
        self.sub_spaces = sub_spaces

    def _focused_str(self, path: Optional[List[SearchSpace]] = None) -> str:
        ret: str = "\u2211["
        ret += "|".join(
            p.str_with_focus(path=path, default="") for p in self.sub_spaces
        )
        ret += "]"
        return ret


class SearchSpaceOperator(SearchSpace):
    sub_space: SearchSpace

    def __init__(self, sub_space: SearchSpace, default: Optional[Any] = None):
        super(SearchSpaceOperator, self).__init__(default=default)
        self.sub_space = sub_space

    def _focused_str(self, path: Optional[List[SearchSpace]] = None) -> str:
        ret: str = "\u00AB"
        ret += self.sub_space.str_with_focus(path=path, default="")
        ret += "\u00BB"
        return ret


class SearchSpaceProduct(SearchSpace):
    sub_spaces: List[Tuple[str, SearchSpace]]

    def __init__(
        self, sub_spaces: List[Tuple[str, SearchSpace]], default: Optional[Any] = None
    ):
        super(SearchSpaceProduct, self).__init__(default=default)
        self.sub_spaces = sub_spaces

    def get_indexed_spaces(self) -> Iterable[Tuple[str, int, SearchSpace]]:
        indices: Dict[str, int] = {}

        def make_indexed(name: str) -> Tuple[str, int]:
            idx = 0
            if name in indices:
                idx = indices[name] + 1
                indices[name] = idx
            else:
                indices[name] = 0
            return (name, idx)

        def enhance_tuple(
            x: Tuple[str, int], space: SearchSpace
        ) -> Tuple[str, int, SearchSpace]:
            return (x[0], x[1], space)

        return [
            enhance_tuple(make_indexed(name), space) for name, space in self.sub_spaces
        ]

    def _focused_str(self, path: Optional[List[SearchSpace]] = None) -> str:
        ret: str = "\u220F{"
        vv: Optional[str]
        parts: List[str] = []
        for k, v in self.sub_spaces:
            vv = v.str_with_focus(path=path, default=None)
            if vv is not None:
                parts.append(k + "->" + vv)
        ret = ";".join(parts)

        ret += "}"
        return ret


# for debugging
_print_search_space_env_options: Optional[Set[str]] = None


def _get_print_search_space_options() -> Set[str]:
    global _print_search_space_env_options
    options: Set[str]
    if _print_search_space_env_options is None:
        debug = os.environ.get("LALE_PRINT_SEARCH_SPACE", None)
        if debug is None:
            options = set()
        else:
            options_raw = debug.split(",")
            options = set(s.strip().lower() for s in options_raw)
        _print_search_space_env_options = options
    else:
        options = _print_search_space_env_options
    return options


def should_print_search_space(*s: str):
    options: Set[str] = _get_print_search_space_options()
    for x in s:
        if x in options:
            return True
    return False
