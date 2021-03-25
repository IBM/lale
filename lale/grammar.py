import random
from typing import Any, Dict, List, Optional, cast

from lale.helpers import nest_HPparams
from lale.lib.lale import NoOp
from lale.operators import (
    BasePipeline,
    IndividualOp,
    Operator,
    OperatorChoice,
    PlannedOperator,
    clone_op,
    make_choice,
    make_pipeline,
    make_pipeline_graph,
)


class NonTerminal(Operator):
    """Abstract operator for non-terminal grammar rules."""

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        out = {}
        out["name"] = self._name
        return out

    def _with_params(self, try_mutate: bool, **impl_params) -> Operator:
        """
        This method updates the parameters of the operator.  NonTerminals do not support
        in-place mutation
        """
        known_keys = set(["name"])
        if impl_params:
            new_keys = set(impl_params.keys())
            if not new_keys.issubset(known_keys):
                unknowns = {k: v for k, v in impl_params.items() if k not in known_keys}
                raise ValueError(
                    f"NonTerminal._with_params called with unknown parameters: {unknowns}"
                )
            else:
                assert "name" in impl_params
                return NonTerminal(impl_params["name"])
        else:
            return self

    def __init__(self, name):
        self._name = name

    def _has_same_impl(self):
        pass

    def is_supervised(self):
        return False

    def validate_schema(self, X, y=None):
        raise NotImplementedError()  # TODO

    def transform_schema(self, s_X):
        raise NotImplementedError()  # TODO

    def input_schema_fit(self):
        raise NotImplementedError()  # TODO

    def is_classifier(self) -> bool:
        return False  # TODO


class Grammar(Operator):
    """Base class for Lale grammars."""

    _variables: Dict[str, Operator]

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        out = {}
        out["variables"] = self._variables

        if deep:
            deep_stuff: Dict[str, Any] = {}
            for k, v in self._variables.items():
                deep_stuff.update(nest_HPparams(k, v.get_params(deep=deep)))

            out.update(deep_stuff)
        return out

    def _with_params(self, try_mutate: bool, **impl_params) -> Operator:
        """
        This method updates the parameters of the operator.
        If try_mutate is set, it will attempt to update the operator in place
        this may not always be possible
        """
        # TODO implement support
        # from this point of view, Grammar is just a higher order operator
        raise NotImplementedError("setting Grammar parameters is not yet supported")

    def __init__(self, variables: Dict[str, Operator] = {}):
        self._variables = variables

    def __getattr__(self, name):
        if name.startswith("_"):
            return self.__dict__[name]
        if name not in self._variables:
            self._variables[name] = NonTerminal(name)
        return clone_op(self._variables[name])

    def __setattr__(self, name, value):
        if name.startswith("_"):
            self.__dict__[name] = value
        else:
            self._variables[name] = value

    def _has_same_impl(self):
        pass

    def is_supervised(self):
        return False

    def validate_schema(self, X, y=None):
        raise NotImplementedError()  # TODO

    def transform_schema(self, s_X):
        raise NotImplementedError()  # TODO

    def input_schema_fit(self):
        raise NotImplementedError()  # TODO

    def is_classifier(self) -> bool:
        raise NotImplementedError()  # TODO

    def _unfold(self, op: Operator, n: int) -> Optional[Operator]:
        """Unroll all possible operators from the grammar `g` starting from    non-terminal `op` after `n` derivations.

        Parameters
        ----------
        op : Operator
            starting rule (e.g., `g.start`)
        n : int
            number of derivations

        Returns
        -------
        Optional[Operator]
        """
        if isinstance(op, BasePipeline):
            steps = op.steps()
            new_maybe_steps: List[Optional[Operator]] = [
                self._unfold(sop, n) for sop in op.steps()
            ]
            if None not in new_maybe_steps:
                new_steps: List[Operator] = cast(List[Operator], new_maybe_steps)
                step_map = {steps[i]: new_steps[i] for i in range(len(steps))}
                new_edges = [(step_map[s], step_map[d]) for s, d in op.edges()]
                return make_pipeline_graph(new_steps, new_edges, True)
            else:
                return None
        if isinstance(op, OperatorChoice):
            steps = [s for s in (self._unfold(sop, n) for sop in op.steps()) if s]
            return make_choice(*steps) if steps else None
        if isinstance(op, NonTerminal):
            return self._unfold(self._variables[op.name()], n - 1) if n > 0 else None
        if isinstance(op, IndividualOp):
            return op
        assert False, f"Unknown operator {op}"

    def unfold(self, n: int) -> PlannedOperator:
        """
        Explore the grammar `g` starting from `g.start` and generate all possible   choices after `n` derivations.

        Parameters
        ----------
        g : Grammar
            input grammar
        n : int
            number of derivations

        Returns
        -------
        PlannedOperator
        """
        assert hasattr(self, "start"), "Rule start must be defined"
        op = self._unfold(self.start, n)
        return make_pipeline(op) if op else NoOp

    def _sample(self, op: Operator, n: int) -> Optional[Operator]:
        """
        Sample the grammar `g` starting from `g.start`, that is, choose one element at random for each possible choices.

        Parameters
        ----------
        op : Operator
            starting rule (e.g., `g.start`)
        n : int
            number of derivations

        Returns
        -------
        Optional[Operator]
        """
        if isinstance(op, BasePipeline):
            steps = op.steps()
            new_maybe_steps: List[Optional[Operator]] = [
                self._sample(sop, n) for sop in op.steps()
            ]
            if None not in new_maybe_steps:
                new_steps: List[Operator] = cast(List[Operator], new_maybe_steps)
                step_map = {steps[i]: new_steps[i] for i in range(len(steps))}
                new_edges = [(step_map[s], step_map[d]) for s, d in op.edges()]
                return make_pipeline_graph(new_steps, new_edges, True)
            else:
                return None
        if isinstance(op, OperatorChoice):
            return self._sample(random.choice(op.steps()), n)
        if isinstance(op, NonTerminal):
            return self._sample(getattr(self, op.name()), n - 1) if n > 0 else None
        if isinstance(op, IndividualOp):
            return op
        assert False, f"Unknown operator {op}"

    def sample(self, n: int) -> PlannedOperator:
        """
        Sample the grammar `g` starting from `g.start`, that is, choose one element at random for each possible choices.

        Parameters
        ----------
        n : int
            number of derivations

        Returns
        -------
        PlannedOperator
        """
        assert hasattr(self, "start"), "Rule start must be defined"
        op = self._sample(self.start, n)
        return make_pipeline(op) if op else NoOp
