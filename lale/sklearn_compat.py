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

from typing import Any, Dict, Iterable, List, Optional, Tuple, TypeVar, Union

import sklearn.base

import lale.operators as Ops
from lale.pretty_print import hyperparams_to_string
from lale.search.PGO import remove_defaults_dict
from lale.util.Visitor import Visitor, accept

# We support an argument encoding schema intended to be a
# conservative extension to sklearn's encoding schema
# sklearn uses __ to separate elements in a hierarchy
# (pipeline's have operators that have keys)

# Since we support richer computation graphs, we need to extend this encoding
# to support it. Graphs that could be represented in sklearn
# should be encoded identically

# our encoding scheme:
# * __ separates nested components (as-in sklearn)
# * ? is the discriminant (choice made) for a choice
# * ? is also a prefix for the nested parts of the chosen branch
# * x@n In a pipeline, if multiple components have identical names,
# ** everything but the first are suffixed with a number (starting with 1)
# ** indicating which one we are talking about.
# ** For example, given (x >> y >> x), we would treat this much the same as
# ** (x >> y >> x@1)
# * $ is used in the rare case that sklearn would expect the key of an object,
# ** but we allow (and have) a non-object schema.  In that case,
# ** $ is used as the key. This should only happen at the top level,
# ** since nested occurences should be removed.
# * # is a structure indicator, and the value should be one of 'list', 'tuple', or 'dict'
# * n is used to represent the nth component in an array or tuple


# This method (and the to_lale() method on the returned value)
# are the only ones intended to be exported
def make_sklearn_compat(op: Ops.Operator) -> "SKlearnCompatWrapper":
    """Top level function for providing compatibiltiy with sklearn operations
       This returns a wrapper around the provided sklearn operator graph which can be passed
       to sklearn methods such as clone and GridSearchCV
       The wrapper may modify the wrapped lale operator/pipeline as part of providing
       compatibility with these methods.
       After the sklearn operation is complete,
       SKlearnCompatWrapper.to_lale() can be called to recover the
       wrapped lale operator for future use
    """
    if isinstance(op, SKlearnCompatWrapper):
        return op
    else:
        return SKlearnCompatWrapper.make_wrapper(Ops.wrap_operator(op))


def sklearn_compat_clone(impl: Any) -> Any:
    if impl is None:
        return None

    from sklearn.base import clone

    cp = clone(impl, safe=False)
    return cp


def clone_lale(op: Ops.Operator) -> Ops.Operator:
    return op._lale_clone(sklearn_compat_clone)


class WithoutGetParams(object):
    """ This wrapper forwards everything except "get_attr" to what it is wrapping
    """

    def __init__(self, base):
        self._base = base
        assert self._base != self

    def __getattr__(self, name):
        # This is needed because in python copy skips calling the __init__ method
        if name == "_base":
            raise AttributeError
        if name == "get_params":
            raise AttributeError
        if name in ["__getstate__", "__setstate__", "__repr__"]:
            raise AttributeError
        else:
            return getattr(self._base, name)

    @classmethod
    def clone_wgp(cls, obj: "WithoutGetParams") -> "WithoutGetParams":
        while isinstance(obj, WithoutGetParams):
            obj = obj._base
        assert isinstance(obj, Ops.Operator)
        return WithoutGetParams(clone_lale(obj))

    def __str__(self):
        b = getattr(self, "_base", None)
        s: str
        if b is None:
            s = ""
        else:
            s = str(b)
        return f"WGP<{s}>"


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


def set_operator_params(op: Ops.Operator, **impl_params) -> Ops.TrainableOperator:
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
        all_params = {**main_params, **hyper}
        return op.set_params(**all_params)
    elif isinstance(op, Ops.BasePipeline):
        steps = op.steps()
        main_params, partitioned_sub_params = partition_sklearn_params(impl_params)
        assert not main_params, f"Unexpected non-nested arguments {main_params}"
        found_names: Dict[str, int] = {}
        step_map: Dict[Ops.Operator, Ops.TrainableOperator] = {}
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
            if not isinstance(op, Ops.TrainablePipeline):
                # As a result of choices made, we may now be a TrainableIndividualOp
                ret = Ops.make_pipeline_graph(op.steps(), op.edges(), ordered=True)
                if not isinstance(ret, Ops.TrainableOperator):
                    assert False
                return ret
            else:
                return op
        else:
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


class SKlearnCompatWrapper(object):
    _base: WithoutGetParams
    # This is used to trick clone into leaving us alone
    _old_params_for_clone: Optional[Dict[str, Any]]

    @classmethod
    def make_wrapper(cls, base: Ops.Operator):
        b: Any = base
        if isinstance(base, SKlearnCompatWrapper):
            return base
        elif not isinstance(base, WithoutGetParams):
            b = WithoutGetParams(base)
        return cls(__lale_wrapper_init_base=b)

    def __init__(self, **kwargs):
        if "__lale_wrapper_init_base" in kwargs:
            # if we are being called by make_wrapper
            # then we don't need to make a copy
            self._base = kwargs["__lale_wrapper_init_base"]
            self._old_params_for_clone = None

        else:
            # otherwise, we are part of a get_params/init clone
            # and we need to make a copy
            self.init_params_internal(**kwargs)
        assert self._base != self

    def init_params_internal(self, **kwargs):
        op = kwargs["__lale_wrapper_base"]
        self._base = WithoutGetParams.clone_wgp(op)
        self._old_params_for_clone = kwargs

    def get_params_internal(self, out: Dict[str, Any]):
        out["__lale_wrapper_base"] = self._base

    def set_params_internal(self, **impl_params):
        self._base = impl_params["__lale_wrapper_base"]
        assert self._base != self

    def fixup_params_internal(self, **params):
        return params

    def to_lale(self) -> Ops.Operator:
        cur: Any = self
        assert cur is not None
        assert cur._base is not None
        cur = cur._base
        while isinstance(cur, WithoutGetParams):
            cur = cur._base
        assert isinstance(cur, Ops.Operator)
        return cur

    # sklearn calls __repr__ instead of __str__
    def __repr__(self):
        op = self.to_lale()
        if isinstance(op, Ops.TrainableIndividualOp):
            name = op.name()
            hyps = ""
            hps = op.hyperparams()
            if hps is not None:
                hyps = hyperparams_to_string(hps)
            return name + "(" + hyps + ")"
        else:
            return super().__repr__()

    def __getattribute__(self, name):
        """ Try proxying unknown attributes to the underlying operator
            getattribute is used instead of getattr to ensure that the
            correct underlying error is thrown in case
            a property (such as classes_) throws an AttributeError
        """

        # This is needed because in python copy skips calling the __init__ method
        try:
            return super(SKlearnCompatWrapper, self).__getattribute__(name)
        except AttributeError as e:
            if name == "_base":
                raise AttributeError
            try:
                return getattr(self._base, name)
            except AttributeError:
                raise e

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        # TODO: We currently ignore deep
        out: Dict[str, Any] = {}
        if self._old_params_for_clone is not None:
            # lie to clone to make it happy
            params = self._old_params_for_clone
            self._old_params_for_clone = None
            return params
        else:
            self.get_params_internal(out)
        return out

    def fit(self, X, y=None, **fit_params):
        if hasattr(self._base, "fit"):
            filtered_params = remove_defaults_dict(fit_params)
            return self._base.fit(X, y=y, **filtered_params)
        else:
            pass

    def set_params(self, **impl_params):

        if "__lale_wrapper_base" in impl_params:
            self.set_params_internal(**impl_params)
        else:
            prev = self
            cur = self._base
            assert prev != cur
            assert cur is not None
            while isinstance(cur, WithoutGetParams):
                assert cur != cur._base
                prev = cur
                cur = cur._base
            if not isinstance(cur, Ops.Operator):
                assert False
            assert isinstance(cur, Ops.Operator)
            fixed_params = self.fixup_params_internal(**impl_params)
            new_s = set_operator_params(cur, **fixed_params)
            if not isinstance(new_s, Ops.TrainableOperator):
                assert False
            if new_s != cur:
                prev._base = new_s
        return self

    def hyperparam_defaults(self) -> Dict[str, Any]:
        return DefaultsVisitor.run(self.to_lale())

    def _final_individual_op(self) -> Optional[Ops.IndividualOp]:
        op: Optional[Ops.Operator] = self.to_lale()
        while op is not None and isinstance(op, Ops.BasePipeline):
            op = op._get_last()
        if op is not None and not isinstance(op, Ops.IndividualOp):
            op = None
        return op

    @property
    def _final_estimator(self):
        op: Optional[Ops.IndividualOp] = self._final_individual_op()
        model = None
        if op is not None:
            # if fit was called, we want to use trained result
            # even if the code uses the original operrator
            # since sklearn assumes that fit mutates the operator
            if hasattr(op, "_trained"):
                op = op._trained
            if hasattr(op, "_impl"):
                impl = op._impl_instance()
                if hasattr(impl, "_wrapped_model"):
                    model = impl._wrapped_model
                elif isinstance(impl, sklearn.base.BaseEstimator):
                    model = impl
        return "passthrough" if model is None else model

    @property
    def classes_(self):
        return self._final_estimator.classes_

    @property
    def n_classes_(self):
        return self._final_estimator.n_classes_

    @property
    def _estimator_type(self):
        return self._final_estimator._estimator_type

    @property
    def _get_tags(self):
        return self._final_estimator._get_tags

    @property
    def coef_(self):
        return self._final_estimator.coef_

    @property
    def feature_importances_(self):
        return self._final_estimator.feature_importances_

    def get_param_ranges(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Returns two dictionaries, ranges and cat_idx, for hyperparameters.

        The ranges dictionary has two kinds of entries. Entries for
        numeric and Boolean hyperparameters are tuples of the form
        (min, max, default). Entries for categorical hyperparameters
        are lists of their values.

        The cat_idx dictionary has (min, max, default) entries of indices
        into the corresponding list of values.

        Warning: ignores side constraints and unions."""
        op: Optional[Ops.IndividualOp] = self._final_individual_op()
        if op is None:
            raise ValueError("This pipeline does not end with an individual operator")
        else:
            return op.get_param_ranges()

    def get_param_dist(self, size=10) -> Dict[str, List[Any]]:
        """Returns a dictionary for discretized hyperparameters.

        Each entry is a list of values. For continuous hyperparameters,
        it returns up to `size` uniformly distributed values.

        Warning: ignores side constraints, unions, and distributions."""
        op: Optional[Ops.IndividualOp] = self._final_individual_op()
        if op is None:
            raise ValueError("This pipeline does not end with an individual operator")
        else:
            return op.get_param_dist(size=size)

    # sklearn compatibility
    # @property
    # def _final_estimator(self):
    #     lale_op = self.to_lale()
    #     if lale_op is _

    #     estimator = self.steps[-1][1]
    #     return 'passthrough' if estimator is None else estimator


class DefaultsVisitor(Visitor):
    @classmethod
    def run(cls, op: Ops.Operator) -> Dict[str, Any]:
        visitor = cls()
        return accept(op, visitor)

    def __init__(self):
        super(DefaultsVisitor, self).__init__()

    def visitIndividualOp(self, op: Ops.IndividualOp) -> Dict[str, Any]:
        return op.hyperparam_defaults()

    visitPlannedIndividualOp = visitIndividualOp
    visitTrainableIndividualOp = visitIndividualOp
    visitTrainedIndividualOp = visitIndividualOp

    def visitPipeline(self, op: Ops.PlannedPipeline) -> Dict[str, Any]:

        defaults_list: Iterable[Dict[str, Any]] = (
            nest_HPparams(s.name(), accept(s, self)) for s in op.steps()
        )

        defaults: Dict[str, Any] = {}
        for d in defaults_list:
            defaults.update(d)

        return defaults

    visitPlannedPipeline = visitPipeline
    visitTrainablePipeline = visitPipeline
    visitTrainedPipeline = visitPipeline

    def visitOperatorChoice(self, op: Ops.OperatorChoice) -> Dict[str, Any]:

        defaults_list: Iterable[Dict[str, Any]] = (accept(s, self) for s in op.steps())

        defaults: Dict[str, Any] = {}
        for d in defaults_list:
            defaults.update(d)

        return defaults


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

    nop = clone(make_sklearn_compat(op)).to_lale()
    if name:
        nop._set_name(name)
    return nop
