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

from typing import TYPE_CHECKING, Optional

from lale.search.lale_hyperopt import search_space_to_hp_expr, search_space_to_hp_str
from lale.search.PGO import PGO
from lale.search.schema2search_space import op_to_search_space
from lale.search.search_space import should_print_search_space

if TYPE_CHECKING:
    from lale.operators import PlannedOperator


def hyperopt_search_space(
    op: "PlannedOperator", schema=None, pgo: Optional[PGO] = None, data_schema={}
):

    search_space = op_to_search_space(op, pgo=pgo, data_schema=data_schema)
    if search_space:
        name = op.name()

        if should_print_search_space("true", "all", "backend", "hyperopt"):
            print(
                f"hyperopt search space for {name}: {search_space_to_hp_str(search_space, name)}"
            )
        return search_space_to_hp_expr(search_space, name)
    else:
        return None
