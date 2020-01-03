from lale.operators import MetaModelOperator, PlannedOperator, Operator, BasePipeline, OperatorChoice, IndividualOp
from lale.operators import make_choice, make_pipeline
from lale.lib.lale import NoOp
from typing import Optional

class Primitive(Operator):
    """ Abstract operator for non-terminal grammar rules.
    """
    def __init__(self, name, value=None):
        self._name = name
        self.value = value
        
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
    """ Base class for Lale grammars. The for every grammar `g`, rule `g.start` must be defined.
    """
    def __init__(self):
        self._primitives = {}

    def __getattr__(self, name):
        if name.startswith('_'):
            return self.__dict__[name]
        if name not in self._primitives:
            self._primitives[name] = Primitive(name)
        return self._primitives[name]
        
    def __setattr__(self, name, value):
        if name.startswith('_'):
            self.__dict__[name] = value
        else:
            self._primitives[name] = value
            
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
            
            
def unroll(g: Grammar, op: Primitive, n: int) -> Optional[PlannedOperator]:
    """ Unroll all possible operators from the grammar `g` starting from non-terminal `op` after `n` derivations.
    
    Parameters
    ----------
    g : Grammar
        input grammar
    op : Primitive
        starting rule (e.g., `g.start`)
    n : int
        number of derivations
    
    Returns
    -------
    Optional[PlannedOperator]
    """
    if isinstance(op, BasePipeline):
        steps = [unroll(g, sop, n) for sop in op._steps]
        return make_pipeline(*steps) if not None in steps else None
    if isinstance(op, OperatorChoice):
        steps = [s for s in (unroll(g, sop, n) for sop in op._steps) if s]
        return make_choice(*steps) if steps else None
    if isinstance(op, Primitive):
        return unroll(g, getattr(g, op._name), n-1) if n > 0 else None
    if isinstance(op, IndividualOp):
        return op
    assert False, f"Unknown operator {op._name} of type {op}"
            
def explore(g: Grammar, n: int) -> PlannedOperator:
    """
    Explore the grammar `g` and generate all possible choices after `n` derivations.
    
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