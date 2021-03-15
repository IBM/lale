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


from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional

from ConfigSpace.conditions import EqualsCondition
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Hyperparameter,
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
)
from smac.configspace import ConfigurationSpace

from lale.search.PGO import PGO
from lale.search.search_space import (
    SearchSpace,
    SearchSpaceArray,
    SearchSpaceEnum,
    SearchSpaceNumber,
    should_print_search_space,
)
from lale.search.search_space_grid import SearchSpaceGrid, get_search_space_grids

if TYPE_CHECKING:
    import lale.operators as Ops


def lale_op_smac_tae(op: "Ops.PlannedOperator", f_min):
    # TODO: we can probably do this in a different way, but get_smac_configuration_space
    # we already have these sklearn compatibility wrappers it is easier for now to use them
    op_compat = op

    def f(cfg):
        from sklearn.base import clone

        wrapped_op = clone(op_compat)
        cfg2 = smac_fixup_params(cfg)
        trainable = wrapped_op.set_params(**cfg2)

        return f_min(trainable)

    return f


def lale_trainable_op_from_config(
    op: "Ops.PlannedOperator", cfg
) -> "Ops.TrainableOperator":
    from sklearn.base import clone

    op_compat = op

    wrapped_op = clone(op_compat)
    cfg2 = smac_fixup_params(cfg)
    trainable = wrapped_op.with_params(**cfg2)
    return trainable


def get_smac_space(
    op: "Ops.PlannedOperator",
    lale_num_grids: Optional[float] = None,
    lale_pgo: Optional[PGO] = None,
    data_schema: Dict[str, Any] = {},
) -> ConfigurationSpace:
    """Top level function: given a lale operator, returns a ConfigurationSpace for use with SMAC
    Parameters
    ----------
    op : The lale PlannedOperator
    lale_num_grids: integer or float, optional
        if set to an integer => 1, it will determine how many parameter grids will be returned (at most)
        if set to an float between 0 and 1, it will determine what fraction should be returned
        note that setting it to 1 is treated as in integer.  To return all results, use None
    """

    hp_grids = get_search_space_grids(
        op, num_grids=lale_num_grids, pgo=lale_pgo, data_schema=data_schema
    )
    cs = hp_grids_to_smac_cs(hp_grids)
    if should_print_search_space("true", "all", "backend", "smac"):
        name = op.name()
        if not name:
            name = "an operator"
        print(f"SMAC configuration for {name}:\n{str(cs)}")

    return cs


def smac_fixup_params(cfg):
    def strip_key(k: str) -> str:
        return k.rsplit("_", 1)[0]

    def transform_value(v):
        if v == "_lale_none":
            return None
        else:
            return v

    ret = {
        strip_key(k): transform_value(v)
        for (k, v) in cfg.get_dictionary().items()
        if k != "disjunct_discriminant"
    }
    return ret


# When sampling from distributions, this is the default number of samples to take.
# Users can override this by passing in num_samples to the appropriate function
SAMPLES_PER_DISTRIBUTION = 2

# We can first convert from our search space IR
# to a more limited grid structure
# This can than be converted to the format required for SMAC


def SearchSpaceNumberToSMAC(key: str, hp: SearchSpaceNumber) -> Hyperparameter:
    """Returns either a list of values intended to be sampled uniformly or a frozen scipy.stats distribution"""
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

    log: bool
    if dist == "uniform" or dist == "integer":
        log = False
    elif dist == "loguniform":
        log = True
    else:
        raise ValueError(f"unknown/unsupported distribution {dist} for {key}")

    if hp.discrete:
        return UniformIntegerHyperparameter(key, min, max, log=log)
    else:
        return UniformFloatHyperparameter(key, min, max, log=log)


class FakeNone(object):
    pass


MyFakeNone = FakeNone()


def HPValuetoSMAC(key: str, hp: SearchSpace) -> Hyperparameter:
    def val_to_str(v):
        if v is None:
            return "_lale_none"
        else:
            return v

    if isinstance(hp, SearchSpaceEnum):
        return CategoricalHyperparameter(key, list(map(val_to_str, hp.vals)))
    elif isinstance(hp, SearchSpaceNumber):
        return SearchSpaceNumberToSMAC(key, hp)
    elif isinstance(hp, SearchSpaceArray):
        raise ValueError(
            f"Arrays are not yet supported by the SMAC backend (key: {key})"
        )
    else:
        raise ValueError(
            f"Not yet supported hp description ({type(hp)}) (key: {key}) in the GridSearchCV backend"
        )


def SearchSpaceGridtoSMAC(hp: SearchSpaceGrid, disc: int) -> Iterable[Hyperparameter]:
    return (HPValuetoSMAC(f"{k}_{disc}", v) for k, v in hp.items())


disc_str = "disjunct_discriminant"


def addSearchSpaceGrid(
    hp: SearchSpaceGrid, disc: int, parent_disc: Hyperparameter, cs: ConfigurationSpace
) -> None:
    smac = SearchSpaceGridtoSMAC(hp, disc)
    for hyp in smac:
        cs.add_hyperparameter(hyp)
        cs.add_condition(EqualsCondition(child=hyp, parent=parent_disc, value=disc))


def addSearchSpaceGrids(grids: List[SearchSpaceGrid], cs: ConfigurationSpace) -> None:
    parent_disc = CategoricalHyperparameter(disc_str, range(len(grids)))
    cs.add_hyperparameter(parent_disc)
    for (i, g) in enumerate(grids):
        addSearchSpaceGrid(g, i, parent_disc, cs)


def hp_grids_to_smac_cs(grids: List[SearchSpaceGrid]) -> ConfigurationSpace:
    cs: ConfigurationSpace = ConfigurationSpace()
    addSearchSpaceGrids(grids, cs)
    return cs
