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
from typing import Any, Dict, Union

import lale.operators as Ops

# This method (and the to_lale() method on the returned value)
# are the only ones intended to be exported
from lale.helpers import (
    get_name_and_index,
    is_numeric_structure,
    make_degen_indexed_name,
    make_indexed_name,
    partition_sklearn_choice_params,
    partition_sklearn_params,
    structure_type_name,
)


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
