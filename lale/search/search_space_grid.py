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

import itertools
import math
import random
import warnings
from collections import ChainMap
from typing import Any, Dict, Iterable, List, Optional, Union

from lale.helpers import (
    DUMMY_SEARCH_SPACE_GRID_PARAM_NAME,
    discriminant_name,
    make_indexed_name,
    nest_all_HPparams,
    nest_choice_all_HPparams,
    structure_type_dict,
    structure_type_list,
    structure_type_name,
    structure_type_tuple,
)
from lale.operators import PlannedOperator
from lale.search.PGO import PGO
from lale.search.schema2search_space import op_to_search_space
from lale.search.search_space import (
    SearchSpace,
    SearchSpaceArray,
    SearchSpaceConstant,
    SearchSpaceDict,
    SearchSpaceEmpty,
    SearchSpaceError,
    SearchSpaceObject,
    SearchSpaceOperator,
    SearchSpacePrimitive,
    SearchSpaceProduct,
    SearchSpaceSum,
    should_print_search_space,
)
from lale.util.Visitor import Visitor, accept

SearchSpaceGrid = Dict[str, SearchSpacePrimitive]


def search_space_grid_to_string(grid: SearchSpaceGrid) -> str:
    return "{" + ";".join(f"{k}->{str(v)}" for k, v in grid.items()) + "}"


def search_space_grids_to_string(grids: List[SearchSpaceGrid]) -> str:
    return "|".join(search_space_grid_to_string(grid) for grid in grids)


def get_search_space_grids(
    op: "PlannedOperator",
    num_grids: Optional[float] = None,
    pgo: Optional[PGO] = None,
    data_schema: Dict[str, Any] = {},
) -> List[SearchSpaceGrid]:
    """Top level function: given a lale operator, returns a list of hp grids.
    Parameters
    ----------
    op : The lale PlannedOperator
    num_grids: integer or float, optional
        if set to an integer => 1, it will determine how many parameter grids will be returned (at most)
        if set to an float between 0 and 1, it will determine what fraction should be returned
        note that setting it to 1 is treated as in integer.  To return all results, use None
    """
    all_parameters = op_to_search_space_grids(op, pgo=pgo, data_schema=data_schema)
    if should_print_search_space("true", "all", "search_space_grids", "grids"):
        name = op.name()
        if not name:
            name = "an operator"
        print(
            f"search space grids for {name}:\n{search_space_grids_to_string(all_parameters)}"
        )
    if num_grids is None:
        return all_parameters
    else:
        if num_grids <= 0:
            warnings.warn(
                f"get_search_space_grids(num_grids={num_grids}) called with a non-positive value for lale_num_grids"
            )
            return []
        if num_grids >= 1:
            samples = math.ceil(num_grids)
            if samples >= len(all_parameters):
                return all_parameters
            else:
                warnings.warn(
                    f"get_search_space_grids(num_grids={num_grids}) sampling {math.ceil(num_grids)}/{len(all_parameters)}"
                )
                return random.sample(all_parameters, math.ceil(num_grids))
        else:
            samples = round(len(all_parameters) * num_grids)
            warnings.warn(
                f"get_search_space_grids(num_grids={num_grids}) sampling {samples}/{len(all_parameters)}"
            )
            return random.sample(all_parameters, samples)


def search_space_to_grids(hp: SearchSpace) -> List[SearchSpaceGrid]:
    return SearchSpaceToGridVisitor.run(hp)


def op_to_search_space_grids(
    op: PlannedOperator, pgo: Optional[PGO] = None, data_schema: Dict[str, Any] = {}
) -> List[SearchSpaceGrid]:
    search_space = op_to_search_space(op, pgo=pgo, data_schema=data_schema)
    grids = search_space_to_grids(search_space)
    return grids


# lets handle the general case
SearchSpaceGridInternalType = Union[List[SearchSpaceGrid], SearchSpacePrimitive]


class SearchSpaceToGridVisitor(Visitor):
    @classmethod
    def run(cls, space: SearchSpace) -> List[SearchSpaceGrid]:
        visitor = cls()
        grids: SearchSpaceGridInternalType = accept(space, visitor)
        fixed_grids = cls.fixupDegenerateSearchSpaces(grids)
        return fixed_grids

    @classmethod
    def fixupDegenerateSearchSpaces(
        cls, space: SearchSpaceGridInternalType
    ) -> List[SearchSpaceGrid]:
        if isinstance(space, SearchSpacePrimitive):
            return [{DUMMY_SEARCH_SPACE_GRID_PARAM_NAME: space}]
        else:
            return space

    def __init__(self):
        super(SearchSpaceToGridVisitor, self).__init__()

    def visitSearchSpacePrimitive(
        self, space: SearchSpacePrimitive
    ) -> SearchSpacePrimitive:
        return space

    visitSearchSpaceEnum = visitSearchSpacePrimitive

    visitSearchSpaceConstant = visitSearchSpaceEnum
    visitSearchSpaceBool = visitSearchSpaceEnum

    visitSearchSpaceNumber = visitSearchSpacePrimitive

    def _searchSpaceList(
        self, space: SearchSpaceArray, *, size: int
    ) -> List[SearchSpaceGrid]:
        sub_spaces = space.items(max=size)

        param_grids: List[List[SearchSpaceGrid]] = [
            nest_all_HPparams(
                str(index), self.fixupDegenerateSearchSpaces(accept(sub, self))
            )
            for index, sub in enumerate(sub_spaces)
        ]

        param_grids_product: Iterable[Iterable[SearchSpaceGrid]] = itertools.product(
            *param_grids
        )
        chained_grids: List[SearchSpaceGrid] = [
            dict(
                ChainMap(
                    *gridline,
                )
            )
            for gridline in param_grids_product
        ]

        if space.is_tuple:
            st_val = structure_type_tuple
        else:
            st_val = structure_type_list

        discriminated_grids: List[SearchSpaceGrid] = [
            {**d, structure_type_name: SearchSpaceConstant(st_val)}
            for d in chained_grids
        ]

        return discriminated_grids

    def visitSearchSpaceArray(self, space: SearchSpaceArray) -> List[SearchSpaceGrid]:
        if space.minimum == space.maximum:
            return self._searchSpaceList(space, size=space.minimum)
        else:
            ret: List[SearchSpaceGrid] = []
            for i in range(space.minimum, space.maximum + 1):
                ret.extend(self._searchSpaceList(space, size=i))
            return ret

    def visitSearchSpaceObject(self, space: SearchSpaceObject) -> List[SearchSpaceGrid]:
        keys = space.keys
        keys_len = len(keys)
        final_choices: List[SearchSpaceGrid] = []
        for c in space.choices:
            assert keys_len == len(c)
            kvs_complex: List[List[SearchSpaceGrid]] = []
            kvs_simple: SearchSpaceGrid = {}
            for k, v in zip(keys, c):
                vspace: Union[List[SearchSpaceGrid], SearchSpacePrimitive] = accept(
                    v, self
                )
                if isinstance(vspace, SearchSpacePrimitive):
                    kvs_simple[k] = vspace
                else:
                    nested_vspace: List[SearchSpaceGrid] = nest_all_HPparams(k, vspace)
                    if nested_vspace:
                        kvs_complex.append(nested_vspace)
            nested_space_choices: Iterable[
                Iterable[SearchSpaceGrid]
            ] = itertools.product(*kvs_complex)
            nested_space_choices_lists: List[List[SearchSpaceGrid]] = list(
                map((lambda x: list(x)), nested_space_choices)
            )
            nested_space_choices_filtered: List[List[SearchSpaceGrid]] = [
                ll for ll in nested_space_choices_lists if ll
            ]
            if nested_space_choices_filtered:
                chained_grids: Iterable[SearchSpaceGrid] = [
                    dict(ChainMap(*nested_choice, kvs_simple))
                    for nested_choice in nested_space_choices_filtered
                ]
                final_choices.extend(chained_grids)
            else:
                final_choices.append(kvs_simple)
        return final_choices

    def visitSearchSpaceSum(self, op: SearchSpaceSum) -> SearchSpaceGridInternalType:
        sub_spaces: List[SearchSpace] = op.sub_spaces

        sub_grids: Iterable[SearchSpaceGridInternalType] = [
            accept(cur_space, self) for cur_space in sub_spaces
        ]

        if len(sub_spaces) == 1:
            return list(sub_grids)[0]
        else:
            fixed_grids: Iterable[List[SearchSpaceGrid]] = (
                SearchSpaceToGridVisitor.fixupDegenerateSearchSpaces(grid)
                for grid in sub_grids
            )
            final_grids: List[SearchSpaceGrid] = []
            for i, grids in enumerate(fixed_grids):
                if not grids:
                    grids = [{}]
                else:
                    # we need to add in this nesting
                    # in case a higher order operator directly contains
                    # another
                    grids = nest_choice_all_HPparams(grids)

                discriminated_grids: List[SearchSpaceGrid] = [
                    {**d, discriminant_name: SearchSpaceConstant(i)} for d in grids
                ]
                final_grids.extend(discriminated_grids)
            return final_grids

    def visitSearchSpaceProduct(
        self, op: SearchSpaceProduct
    ) -> SearchSpaceGridInternalType:

        sub_spaces = op.get_indexed_spaces()

        param_grids: List[List[SearchSpaceGrid]] = [
            nest_all_HPparams(
                make_indexed_name(name, index),
                self.fixupDegenerateSearchSpaces(accept(space, self)),
            )
            for name, index, space in sub_spaces
        ]

        param_grids_product: Iterable[Iterable[SearchSpaceGrid]] = itertools.product(
            *param_grids
        )
        chained_grids: List[SearchSpaceGrid] = [
            dict(ChainMap(*gridline)) for gridline in param_grids_product
        ]

        return chained_grids

    def visitSearchSpaceDict(self, op: SearchSpaceDict) -> SearchSpaceGridInternalType:

        sub_spaces = op.space_dict.items()

        param_grids: List[List[SearchSpaceGrid]] = [
            nest_all_HPparams(
                name,
                self.fixupDegenerateSearchSpaces(accept(space, self)),
            )
            for name, space in sub_spaces
        ]

        param_grids_product: Iterable[Iterable[SearchSpaceGrid]] = itertools.product(
            *param_grids
        )
        chained_grids: List[SearchSpaceGrid] = [
            dict(ChainMap(*gridline)) for gridline in param_grids_product
        ]

        discriminated_grids: List[SearchSpaceGrid] = [
            {**d, structure_type_name: SearchSpaceConstant(structure_type_dict)}
            for d in chained_grids
        ]

        return discriminated_grids

    def visitSearchSpaceOperator(
        self, op: SearchSpaceOperator
    ) -> SearchSpaceGridInternalType:
        return accept(op.sub_space, self)

    def visitSearchSpaceEmpty(self, op: SearchSpaceEmpty):
        raise SearchSpaceError(
            None, "Grid based backends can't compile an empty (sub-) search space"
        )
