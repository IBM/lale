import random
from typing import Optional

from lale.lib.lale import NoOp
from lale.operators import (
    BasePipeline,
    IndividualOp,
    Operator,
    OperatorChoice,
    PlannedOperator,
    make_choice,
    make_pipeline,
    make_pipeline_graph,
)
from lale.sklearn_compat import clone_op


class NonTerminal(Operator):
    """ Abstract operator for non-terminal grammar rules.
    """

    def __init__(self, name):
        self._name = name

    def _lale_clone(self, cloner):
        return NonTerminal(self.name())

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

    def freeze_trainable(self) -> "Operator":
        raise NotImplementedError()  # TODO

    def is_frozen_trainable(self) -> bool:
        raise NotImplementedError()  # TODO

    def is_classifier(self) -> bool:
        return False  # TODO


class Grammar(Operator):
    """ Base class for Lale grammars.
    """

    def __init__(self):
        self._variables = {}

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

    def _lale_clone(self):
        pass

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

    def freeze_trainable(self) -> "Operator":
        raise NotImplementedError()  # TODO

    def is_frozen_trainable(self) -> bool:
        raise NotImplementedError()  # TODO

    def is_classifier(self) -> bool:
        raise NotImplementedError()  # TODO

    def _unfold(self, op: Operator, n: int) -> Optional[Operator]:
        """ Unroll all possible operators from the grammar `g` starting from    non-terminal `op` after `n` derivations.

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
            new_steps = [self._unfold(sop, n) for sop in op.steps()]
            step_map = {steps[i]: new_steps[i] for i in range(len(steps))}
            new_edges = [(step_map[s], step_map[d]) for s, d in op.edges()]
            if None not in new_steps:
                return make_pipeline_graph(new_steps, new_edges, True)
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
            new_steps = [self._sample(sop, n) for sop in op.steps()]
            step_map = {steps[i]: new_steps[i] for i in range(len(steps))}
            new_edges = [(step_map[s], step_map[d]) for s, d in op.edges()]
            if None not in new_steps:
                return make_pipeline_graph(new_steps, new_edges, True)
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
