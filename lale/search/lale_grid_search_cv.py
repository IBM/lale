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

from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
import sklearn.model_selection

import lale.operators as Ops
from lale.search.PGO import PGO
from lale.search.search_space import (
    SearchSpace,
    SearchSpaceArray,
    SearchSpaceEnum,
    SearchSpaceNumber,
    should_print_search_space,
)
from lale.search.search_space_grid import SearchSpaceGrid, get_search_space_grids
from lale.sklearn_compat import make_sklearn_compat

if TYPE_CHECKING:
    from lale.operators import PlannedOperator


def get_defaults_as_param_grid(op: "Ops.IndividualOp"):
    defaults = op.hyperparam_defaults()
    return {k: [v] for k, v in defaults.items()}


def get_lale_gridsearchcv_op(op, params, **kwargs):
    g = sklearn.model_selection.GridSearchCV(make_sklearn_compat(op), params, **kwargs)
    return g


#    TODO: turn it into a Lale TrainableOperator
#    name = f"GridSearchCV[{op.name()}]"
#    return Ops.TrainableIndividualOp(_name=name, _impl=g, _schemas=None)


def get_parameter_grids(
    op: "PlannedOperator",
    num_samples: Optional[int] = None,
    num_grids: Optional[float] = None,
    pgo: Optional[PGO] = None,
    data_schema: Dict[str, Any] = {},
):
    """
    Parameters
    ----------
    op : The lale PlannedOperator
    lale_num_samples : integer, optional
        If set, will limit the number of samples for each distribution
    lale_num_grids: integer or float, optional
        if set to an integer => 1, it will determine how many parameter grids will be returned (at most)
        if set to an float between 0 and 1, it will determine what fraction should be returned
        note that setting it to 1 is treated as in integer.  To return all results, use None
    """
    return get_grid_search_parameter_grids(
        op,
        num_samples=num_samples,
        num_grids=num_grids,
        pgo=pgo,
        data_schema=data_schema,
    )


def get_grid_search_parameter_grids(
    op: "PlannedOperator",
    num_samples: Optional[int] = None,
    num_grids: Optional[float] = None,
    pgo: Optional[PGO] = None,
    data_schema: Dict[str, Any] = {},
) -> List[Dict[str, List[Any]]]:
    """ Top level function: given a lale operator, returns a list of parameter grids
        suitable for passing to GridSearchCV.
        Note that you will need to wrap the lale operator for sklearn compatibility to call GridSearchCV
        directly.  The lale GridSearchCV wrapper takes care of that for you
    """
    hp_grids = get_search_space_grids(
        op, num_grids=num_grids, pgo=pgo, data_schema=data_schema
    )
    grids = SearchSpaceGridstoGSGrids(hp_grids, num_samples=num_samples)
    if should_print_search_space("true", "all", "backend", "gridsearchcv"):
        name = op.name()
        if not name:
            name = "an operator"
        print(f"GridSearchCV grids for {name}:\n{gridsearchcv_grids_to_string(grids)}")

    return grids


GSValue = Any
GSGrid = Dict[str, List[GSValue]]

DEFAULT_SAMPLES_PER_DISTRIBUTION = 2


def SearchSpaceNumberToGSValues(
    key: str, hp: SearchSpaceNumber, num_samples: Optional[int] = None
) -> List[GSValue]:
    """Returns either a list of values intended to be sampled uniformly """
    samples: int
    if num_samples is None:
        samples = DEFAULT_SAMPLES_PER_DISTRIBUTION
    else:
        samples = num_samples

    # Add preliminary support for PGO
    if hp.pgo is not None:
        ret = list(hp.pgo.samples(samples))
        return ret

    # if we are not doing PGO
    dist = "uniform"
    if hp.distribution:
        dist = hp.distribution
    if hp.maximum is None:
        raise ValueError(
            f"maximum not specified for a number with distribution {dist} for {key}"
        )
    max = hp.getInclusiveMax()
    if hp.minimum is None:
        raise ValueError(
            f"minimum not specified for a number with distribution {dist} for {key}"
        )
    min = hp.getInclusiveMin()

    dt: np.dtype
    if hp.discrete:
        dt = np.dtype(int)
    else:
        dt = np.dtype(float)

    default = hp.default()
    if default is not None:
        # always use the default as one of the samples
        # TODO: ensure that the default is valid according to the schema
        if samples <= 1:
            return [default]
        samples = samples - 1
    if dist == "uniform" or dist == "integer":
        ret = np.linspace(min, max, num=samples, dtype=dt).tolist()
    elif dist == "loguniform":
        ret = np.logspace(min, max, num=samples, dtype=dt).tolist()
    else:
        raise ValueError(f"unknown/unsupported distribution {dist} for {key}")
    if default is not None:
        ret.append(default)
    return ret


def HPValuetoGSValue(
    key: str, hp: SearchSpace, num_samples: Optional[int] = None
) -> List[GSValue]:
    if isinstance(hp, SearchSpaceEnum):
        return hp.vals
    elif isinstance(hp, SearchSpaceNumber):
        return SearchSpaceNumberToGSValues(key, hp, num_samples=num_samples)
    elif isinstance(hp, SearchSpaceArray):
        raise ValueError(
            f"Arrays are not yet supported by the GridSearchCV backend (key: {key})"
        )
    else:
        raise ValueError(
            f"Not yet supported hp description ({type(hp)}) (key: {key}) in the GridSearchCV backend"
        )


def SearchSpaceGridtoGSGrid(
    hp: SearchSpaceGrid, num_samples: Optional[int] = None
) -> GSGrid:
    return {k: HPValuetoGSValue(k, v, num_samples=num_samples) for k, v in hp.items()}


def SearchSpaceGridstoGSGrids(
    hp_grids: List[SearchSpaceGrid], num_samples: Optional[int] = None
) -> List[GSGrid]:
    return [SearchSpaceGridtoGSGrid(g, num_samples=num_samples) for g in hp_grids]


def gridsearchcv_grid_to_string(grid: GSGrid) -> str:
    return "{" + ";".join(f"{k}->{str(v)}" for k, v in grid.items()) + "}"


def gridsearchcv_grids_to_string(grids: List[GSGrid]) -> str:
    return "|".join(gridsearchcv_grid_to_string(grid) for grid in grids)
