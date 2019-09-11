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

from typing import Any, Dict, Iterable, List, Optional, Tuple
import itertools
import warnings
import random
import math
from collections import ChainMap

from lale.util.Visitor import Visitor
from lale.search.search_space import SearchSpace, SearchSpaceObject, SearchSpaceEnum
from lale.search.schema2search_space import schemaToSearchSpace
from lale.search.PGO import PGO
from lale.sklearn_compat import nest_all_HPparams

# To avoid import cycle, since we only realy on lale.operators for types
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from lale.operators import PlannedOperator, OperatorChoice, PlannedIndividualOp, PlannedPipeline


SearchSpaceGrid = Dict[str,SearchSpace]

def get_search_space_grids( op:'PlannedOperator', 
                            num_grids:Optional[float]=None, 
                            pgo:Optional[PGO]=None)->List[SearchSpaceGrid]:
    """ Top level function: given a lale operator, returns a list of hp grids.
    Parameters
    ----------
    op : The lale PlannedOperator
    num_grids: integer or float, optional
        if set to an integer => 1, it will determine how many parameter grids will be returned (at most)
        if set to an float between 0 and 1, it will determine what fraction should be returned
        note that setting it to 1 is treated as in integer.  To return all results, use None
    """
    all_parameters = SearchSpaceGridVisitor.run(op, pgo=pgo)
    if num_grids is None:
        return all_parameters
    else:
        if num_grids <= 0:
            warnings.warn(f"get_search_space_grids(num_grids={num_grids}) called with a non-positive value for lale_num_grids")
            return []
        if num_grids >= 1:
            samples = math.ceil(num_grids)
            if samples >= len(all_parameters):
                return all_parameters
            else:
                warnings.warn(f"get_search_space_grids(num_grids={num_grids}) sampling {math.ceil(num_grids)}/{len(all_parameters)}")
                return random.sample(all_parameters, math.ceil(num_grids))
        else:
            samples = round(len(all_parameters)*num_grids)
            warnings.warn(f"get_search_space_grids(num_grids={num_grids}) sampling {samples}/{len(all_parameters)}")
            return random.sample(all_parameters, samples)


def SearchSpaceObjectChoiceToGrid(keys:List[str], values:Tuple)->SearchSpaceGrid:
    assert len(keys) == len(values)
    return dict(zip(keys, values))


def SearchSpaceObjectectToGrid(hp:SearchSpaceObject)->List[SearchSpaceGrid]:
    return [SearchSpaceObjectChoiceToGrid(hp.keys, c) for c in hp.choices]

def searchSpaceToGrids(hp:SearchSpace)->List[SearchSpaceGrid]:
    if isinstance(hp, SearchSpaceObject):
        return SearchSpaceObjectectToGrid(hp)
    else:
        raise ValueError("Can only convert SearchSpaceObject into a GridSearchCV schema")

def schemaToSearchSpaceGrids(longName:str, 
                             name:str, 
                             schema,
                             pgo:Optional[PGO]=None)->List[SearchSpaceGrid]:
    h = schemaToSearchSpace(longName, name, schema, pgo=pgo)
    if h is None:
        return []
    grids = searchSpaceToGrids(h)
    return grids

class SearchSpaceGridVisitor(Visitor):
    pgo:Optional[PGO]

    @classmethod
    def run(cls, op:'PlannedOperator', pgo:Optional[PGO]=None):
        visitor = cls(pgo=pgo)
        accepting_op:Any = op
        return accepting_op.accept(visitor)

    def __init__(self, pgo:Optional[PGO]=None):
        super(SearchSpaceGridVisitor, self).__init__()
        self.pgo = pgo
    
    def augment_grid(self, grid:SearchSpaceGrid, hyperparams)->SearchSpaceGrid:
        if not hyperparams:
            return grid
        ret = dict(grid)
        for (k,v) in hyperparams.items():
            if k not in ret:
                ret[k] = SearchSpaceEnum([v])
        return ret

    def visitPlannedIndividualOp(self, op:'PlannedIndividualOp')->List[SearchSpaceGrid]:
        schema = op.hyperparam_schema_with_hyperparams()
        module = op._impl.__module__
        if module is None or module == str.__class__.__module__:
            long_name = op.name()
        else:
            long_name = module + '.' + op.name()
        name = op.name()
        grids = schemaToSearchSpaceGrids(long_name, name, schema, pgo=self.pgo)
        if hasattr(op, '_hyperparams'):
            hyperparams = op._hyperparams
            if hyperparams and not grids:
                grids = [{}]
            augmented_grids = [self.augment_grid(g, hyperparams) for g in grids]
            return augmented_grids
        else:
            return grids

    visitTrainableIndividualOp = visitPlannedIndividualOp
    visitTrainedIndividualOp = visitPlannedIndividualOp
    
    def visitPlannedPipeline(self, op:'PlannedPipeline')->List[SearchSpaceGrid]:

        param_grids:List[List[SearchSpaceGrid]] = [
            nest_all_HPparams(s.name(), s.accept(self)) for s in op.steps()]

        param_grids_product:Iterable[Iterable[SearchSpaceGrid]] = itertools.product(*param_grids)
        chained_grids:List[SearchSpaceGrid] = [
            dict(ChainMap(*gridline)) for gridline in param_grids_product]

        return chained_grids
    
    visitTrainablePipeline = visitPlannedPipeline
    visitTrainedPipeline = visitPlannedPipeline

    def visitOperatorChoice(self, op:'OperatorChoice')->List[SearchSpaceGrid]:

        choice_name:str = "_lale_discriminant"
        ret:List[SearchSpaceGrid] = []
        for s in op.steps():
            # if not isinstance(s, PlannedOperator):
            #     raise ValueError("This method should really be defined on PlannedOperatorChoice")
            # else:
            grids:List[SearchSpaceGrid] = s.accept(self)
            # If there are no parameters, we still need to add a choice for the discriminant
            if not grids:
                grids = [{}]
            op_name:str = s.name()
            discriminated_grids:List[SearchSpaceGrid]=[{**d, choice_name:SearchSpaceEnum([op_name])} for d in grids]
            ret.extend(discriminated_grids)
        return ret

