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
import warnings
from typing import Any, Dict, List, Tuple, TypeVar, Union

import lale.operators as Ops


# This method (and the to_lale() method on the returned value)
# are the only ones intended to be exported
def make_sklearn_compat(op):
    """This is a deprecated method for backward compatibility and will be removed soon"""
    warnings.warn(
        Ops._mutation_warning("sklearn_compat.make_sklearn_compat"), DeprecationWarning
    )
    return op


def sklearn_compat_clone(impl: Any) -> Any:
    """This is a deprecated method for backward compatibility and will be removed soon.
       call lale.operators.clone (or scikit-learn clone) instead"""
    warnings.warn(
        Ops._mutation_warning("sklearn_compat.make_sklearn_compat"), DeprecationWarning
    )
    if impl is None:
        return None

    from sklearn.base import clone

    cp = clone(impl, safe=False)
    return cp


def partition_sklearn_params(
    d: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    sub_parts: Dict[str, Dict[str, Any]] = {}
    main_parts: Dict[str, Any] = {}

    for k, v in d.items():
        ks = k.split("__", 1)
        if len(ks) == 1:
            assert k not in main_parts
            main_parts[k] = v
        else:
            assert len(ks) == 2
            bucket: Dict[str, Any] = {}
            group: str = ks[0]
            param: str = ks[1]
            if group in sub_parts:
                bucket = sub_parts[group]
            else:
                sub_parts[group] = bucket
            assert param not in bucket
            bucket[param] = v
    return (main_parts, sub_parts)


def partition_sklearn_choice_params(d: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    discriminant_value: int = -1
    choice_parts: Dict[str, Any] = {}

    for k, v in d.items():
        if k == discriminant_name:
            assert discriminant_value == -1
            discriminant_value = int(v)
        else:
            k_rest = unnest_choice(k)
            choice_parts[k_rest] = v
    assert discriminant_value != -1
    return (discriminant_value, choice_parts)


DUMMY_SEARCH_SPACE_GRID_PARAM_NAME: str = "$"
discriminant_name: str = "?"
choice_prefix: str = "?"
structure_type_name: str = "#"
structure_type_list: str = "list"
structure_type_tuple: str = "tuple"
structure_type_dict: str = "dict"


def get_name_and_index(name: str) -> Tuple[str, int]:
    """ given a name of the form "name@i", returns (name, i)
        if given a name of the form "name", returns (name, 0)
    """
    splits = name.split("@", 1)
    if len(splits) == 1:
        return splits[0], 0
    else:
        return splits[0], int(splits[1])


def make_degen_indexed_name(name, index):
    return f"{name}@{index}"


def make_indexed_name(name, index):
    if index == 0:
        return name
    else:
        return f"{name}@{index}"


def make_array_index_name(index, is_tuple: bool = False):
    sep = "##" if is_tuple else "#"
    return f"{sep}{str(index)}"


def is_numeric_structure(structure_type: str):

    if structure_type == "list" or structure_type == "tuple":
        return True
    elif structure_type == "dict":
        return False
    else:
        assert False, f"Unknown structure type {structure_type} found"


def set_structured_params(k, params: Dict[str, Any], hyper_parent):
    # need to handle the different encoding schemes used
    if params is None:
        return None
    if structure_type_name in params:
        # this is a structured type
        structure_type = params[structure_type_name]
        type_params, sub_params = partition_sklearn_params(params)

        hyper = None
        if isinstance(hyper_parent, dict):
            hyper = hyper_parent.get(k, None)
        elif isinstance(hyper_parent, list) and k < len(hyper_parent):
            hyper = hyper_parent[k]
        if hyper is None:
            hyper = {}
        elif isinstance(hyper, tuple):
            # to make it mutable
            hyper = list(hyper)

        del type_params[structure_type_name]
        actual_key: Union[str, int]
        for elem_key, elem_value in type_params.items():
            if elem_value is not None:
                if not isinstance(hyper, dict):
                    assert is_numeric_structure(structure_type)
                    actual_key = int(elem_key)
                    # we may need to extend the array
                    try:
                        hyper[actual_key] = elem_value
                    except IndexError:
                        assert 0 <= actual_key
                        hyper.extend((actual_key - len(hyper)) * [None])
                        hyper.append(elem_value)
                else:
                    actual_key = elem_key
                    hyper[actual_key] = elem_value

        for elem_key, elem_params in sub_params.items():
            if not isinstance(hyper, dict):
                assert is_numeric_structure(structure_type)
                actual_key = int(elem_key)
            else:
                actual_key = elem_key
            set_structured_params(actual_key, elem_params, hyper)
        if isinstance(hyper, dict) and is_numeric_structure(structure_type):
            max_key = max(map(int, hyper.keys()))
            hyper = [hyper.get(str(x), None) for x in range(max_key)]
        if structure_type == "tuple":
            hyper = tuple(hyper)
        hyper_parent[k] = hyper
    else:
        # if it is not a structured parameter
        # then it must be a nested higher order operator
        sub_op = hyper_parent[k]
        if isinstance(sub_op, list):
            if len(sub_op) == 1:
                sub_op = sub_op[0]
            else:
                (disc, chosen_params) = partition_sklearn_choice_params(params)
                assert 0 <= disc and disc < len(sub_op)
                sub_op = sub_op[disc]
                params = chosen_params
        trainable_sub_op = set_operator_params(sub_op, **params)
        hyper_parent[k] = trainable_sub_op


def set_operator_params(op: Ops.Operator, **impl_params) -> Ops.Operator:
    """May return a new operator, in which case the old one should be overwritten
    """
    if isinstance(op, Ops.PlannedIndividualOp):
        main_params, partitioned_sub_params = partition_sklearn_params(impl_params)
        hyper = op._hyperparams
        if hyper is None:
            hyper = {}
        # we set the sub params first
        for sub_key, sub_params in partitioned_sub_params.items():
            set_structured_params(sub_key, sub_params, hyper)

        # we have now updated any nested operators
        # (if this is a higher order operator)
        # and can work on the main operator
        all_params = {**hyper, **main_params}
        return op.set_op_params(**all_params)
    elif isinstance(op, Ops.BasePipeline):
        steps = op.steps()
        main_params, partitioned_sub_params = partition_sklearn_params(impl_params)
        assert not main_params, f"Unexpected non-nested arguments {main_params}"
        found_names: Dict[str, int] = {}
        step_map: Dict[Ops.Operator, Ops.Operator] = {}
        for s in steps:
            name = s.name()
            name_index = 0
            params: Dict[str, Any] = {}
            if name in found_names:
                name_index = found_names[name] + 1
                found_names[name] = name_index
                uname = make_indexed_name(name, name_index)
                if uname in partitioned_sub_params:
                    params = partitioned_sub_params[uname]
            else:
                found_names[name] = 0
                uname = make_degen_indexed_name(name, 0)
                if uname in partitioned_sub_params:
                    params = partitioned_sub_params[uname]
                    assert name not in partitioned_sub_params
                elif name in partitioned_sub_params:
                    params = partitioned_sub_params[name]
            new_s = set_operator_params(s, **params)
            if s != new_s:
                step_map[s] = new_s
        # make sure that no parameters were passed in for operations
        # that are not actually part of this pipeline
        for k in partitioned_sub_params.keys():
            n, i = get_name_and_index(k)
            assert n in found_names and i <= found_names[n]
        if step_map:
            op._subst_steps(step_map)

        pipeline_graph_class = Ops._pipeline_graph_class(op.steps())
        op.__class__ = pipeline_graph_class  # type: ignore
        assert isinstance(op, Ops.TrainableOperator)
        return op
    elif isinstance(op, Ops.OperatorChoice):
        choices = op.steps()
        choice_index: int
        choice_params: Dict[str, Any]
        if len(choices) == 1:
            choice_index = 0
            chosen_params = impl_params
        else:
            (choice_index, chosen_params) = partition_sklearn_choice_params(impl_params)

        assert 0 <= choice_index and choice_index < len(choices)
        choice: Ops.Operator = choices[choice_index]

        new_step = set_operator_params(choice, **chosen_params)
        # we remove the OperatorChoice, replacing it with the branch that was taken
        return new_step
    else:
        assert False, f"Not yet supported operation of type: {op.__class__.__name__}"


NEW_STUFF = "_new_stuff"
PLANNED_OPERATOR_NAME = "PlannedOperator"
INDIVIDUAL_OPERATOR_NAME = "IndividualOperator"


# # sklearn calls __repr__ instead of __str__
# def __repr__(self):
#     op = self.to_lale()
#     if isinstance(op, Ops.TrainableIndividualOp):
#         name = op.name()
#         hyps = ""
#         hps = op.hyperparams()
#         if hps is not None:
#             hyps = hyperparams_to_string(hps)
#         return name + "(" + hyps + ")"
#     else:
#         return super().__repr__()


# Auxiliary functions
V = TypeVar("V")


def nest_HPparam(name: str, key: str):
    if key == DUMMY_SEARCH_SPACE_GRID_PARAM_NAME:
        # we can get rid of the dummy now, since we have a name for it
        return name
    return name + "__" + key


def nest_HPparams(name: str, grid: Dict[str, V]) -> Dict[str, V]:
    return {(nest_HPparam(name, k)): v for k, v in grid.items()}


def nest_all_HPparams(name: str, grids: List[Dict[str, V]]) -> List[Dict[str, V]]:
    """ Given the name of an operator in a pipeline, this transforms every key(parameter name) in the grids
        to use the operator name as a prefix (separated by __).  This is the convention in scikit-learn pipelines.
    """
    return [nest_HPparams(name, grid) for grid in grids]


def nest_choice_HPparam(key: str):
    return choice_prefix + key


def nest_choice_HPparams(grid: Dict[str, V]) -> Dict[str, V]:
    return {(nest_choice_HPparam(k)): v for k, v in grid.items()}


def nest_choice_all_HPparams(grids: List[Dict[str, V]]) -> List[Dict[str, V]]:
    """ this transforms every key(parameter name) in the grids
        to be nested under a choice, using a ? as a prefix (separated by __).  This is the convention in scikit-learn pipelines.
    """
    return [nest_choice_HPparams(grid) for grid in grids]


def unnest_choice(k: str) -> str:
    assert k.startswith(choice_prefix)
    return k[len(choice_prefix) :]


def unnest_HPparams(k: str) -> List[str]:
    return k.split("__")


OpType = TypeVar("OpType", bound=Ops.Operator)


def clone_op(op: OpType, name: str = None) -> OpType:
    """ Clone any operator.
    """
    from sklearn.base import clone

    nop = clone(op)
    if name:
        nop._set_name(name)
    return nop
