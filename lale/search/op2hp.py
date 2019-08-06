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

from typing import Optional, Dict
import os
import logging

from lale.search.search_space import *
from lale.util.Visitor import Visitor
from lale.search import schema2search_space as opt
from lale.search.HP import search_space_to_hp_expr, search_space_to_hp_str
from lale.search.PGO import PGO

from hyperopt import hp

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lale.operators import PlannedOperator, OperatorChoice, PlannedIndividualOp, PlannedPipeline

def hyperopt_search_space(op:'PlannedOperator', 
                          schema=None,
                          pgo:Optional[PGO]=None):
    return HPOperatorVisitor.run(op, schema=schema, pgo=pgo)

class HPOperatorVisitor(Visitor):
    pgo:Optional[PGO]
    names:Dict[str,int]

    @classmethod
    def run(cls, 
            op:'PlannedOperator',
            schema=None,
            pgo:Optional[PGO]=None):
        visitor = cls(pgo=pgo)
        accepting_op:Any = op
        return accepting_op.accept(visitor, schema=schema)

    def __init__(self, pgo:Optional[PGO]=None):
        super(HPOperatorVisitor, self).__init__()
        self.pgo = pgo
        self.names = {}

    def get_unique_name(self, name:str)->str:
        if name in self.names:
            counter = self.names[name] + 1
            self.names[name] = counter
            return name + "@" + str(counter)
        else:
            self.names[name] = 0
            return name

    def visitPlannedIndividualOp(self, op:'PlannedIndividualOp', schema=None):
        if schema is None:
            schema = op.hyperparam_schema_with_hyperparams()
        module = op._impl.__module__
        if module is None or module == str.__class__.__module__:
            long_name = op.name()
        else:
            long_name = module + '.' + op.name()
        name = op.name()

        (simp, hp_s) = opt.schemaToSimplifiedAndSearchSpace(long_name, name, schema, pgo=self.pgo)
        if hp_s:
            unique_name = self.get_unique_name(name)

            if os.environ.get("LALE_PRINT_SEARCH_SPACE", "false") == "true":
                print(f"hyperopt search space for {unique_name}: {search_space_to_hp_str(hp_s, unique_name)}")
            return search_space_to_hp_expr(hp_s, unique_name)
        else:
            return None

    visitTrainableIndividualOp = visitPlannedIndividualOp
    visitTrainedIndividualOp = visitPlannedIndividualOp

    def visitPlannedPipeline(self, op:'PlannedPipeline', schema=None):
        search_spaces = [m.accept(self) for m in op.steps()]
        return search_spaces

    visitTrainablePipeline = visitPlannedPipeline
    visitTrainedPipeline = visitPlannedPipeline

    def visitOperatorChoice(self, op:'OperatorChoice', schema=None):
        unique_name:str = self.get_unique_name(op.name())
        search_spaces = hp.choice(unique_name, [{i : m.accept(self)} for i, m in enumerate(op.steps())])
        return search_spaces

