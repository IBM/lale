from lale.operators import MetaModelOperator, PlannedOperator, Operator, BasePipeline, OperatorChoice, IndividualOp
from lale.operators import make_choice, make_pipeline, get_pipeline_of_applicable_type
from lale.lib.lale import NoOp
from typing import Optional

class NonTerminal(Operator):
    """ Abstract operator for non-terminal grammar rules.
    """
    def __init__(self, name):
        self._name = name
        
    def _lale_clone(self):
        pass
    
    def has_same_impl(self):
        pass
    
    def is_supervised(self):
        return False

    def name(self):
        return self._name
    
    def set_name(self, name):
        self._name = name
        
        
class Grammar(MetaModelOperator):
    """ Base class for Lale grammars.
    """
    def __init__(self):
        self._variables = {}

    def __getattr__(self, name):
        if name.startswith('_'):
            return self.__dict__[name]
        if name not in self._variables:
            self._variables[name] = NonTerminal(name)
        return self._variables[name]
        
    def __setattr__(self, name, value):
        if name.startswith('_'):
            self.__dict__[name] = value
        else:
            self._variables[name] = value
            
    def _lale_clone(self):
        pass
    
    def has_same_impl(self):
        pass
    
    def is_supervised(self):
        return False

    def name(self):
        return self._name
    
    def set_name(self, name):
        self._name = name
            
    def auto_arrange(self, planner):
        pass
    
    def arrange(self, *args, **kwargs):
        pass
            
            
def unroll(g: Grammar, op: Operator, n: int) -> Optional[Operator]:
    """ Unroll all possible operators from the grammar `g` starting from non-terminal `op` after `n` derivations.
    
    Parameters
    ----------
    g : Grammar
        input grammar
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
        new_steps = [unroll(g, sop, n) for sop in op.steps()]
        step_map = {steps[i]: new_steps[i] for i in range(len(steps))}
        new_edges = ((step_map[s], step_map[d]) for s, d in op.edges())
        if not None in new_steps:
            return get_pipeline_of_applicable_type(new_steps, new_edges, True)
        return None
    if isinstance(op, OperatorChoice):
        steps = [s for s in (unroll(g, sop, n) for sop in op.steps()) if s]
        return make_choice(*steps) if steps else None
    if isinstance(op, NonTerminal):
        return unroll(g, getattr(g, op.name()), n-1) if n > 0 else None
    if isinstance(op, IndividualOp):
        return op
    assert False, f"Unknown operator {op}"
            
def explore(g: Grammar, n: int) -> PlannedOperator:
    """
    Explore the grammar `g` starting from `g.start` and generate all possible choices after `n` derivations.
    
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
    assert hasattr(g, 'start'), "Rule start must be defined"
    op = unroll(g, g.start, n)
    return make_pipeline(op) if op else NoOp