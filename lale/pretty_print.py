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

import ast
import astunparse
import collections
import importlib
import inspect
import json
import pprint
import re
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import lale.helpers
import lale.json_operator
import lale.operators

JSON_TYPE = Dict[str, Any]

def hyperparams_to_string(hps: JSON_TYPE, op:'lale.operators.Operator'=None) -> str:
    if op:
        for k, v in hps.items():
            pass #TODO: use enums where possible
    def value_to_string(value):
        return pprint.pformat(value, width=10000, compact=True)
    strings = [f'{k}={value_to_string(v)}' for k, v in hps.items()]
    return ', '.join(strings)

def _get_module_name(op_label: str, op_name: str, class_name: str) -> str:
    def has_op(module_name, sym):
        module = importlib.import_module(module_name)
        if hasattr(module, sym):
            op = getattr(module, sym)
            if isinstance(op, lale.operators.IndividualOp):
                return op.class_name() == class_name
            else:
                return hasattr(op, '__init__') and hasattr(op, 'fit') and (
                    hasattr(op, 'predict') or hasattr(op, 'transform'))
        return False
    mod_name_long = class_name[:class_name.rfind('.')]
    mod_name_short = mod_name_long[:mod_name_long.rfind('.')]
    unqualified = class_name[class_name.rfind('.')+1:]
    if has_op(mod_name_short, op_label):
        return mod_name_short
    if has_op(mod_name_long, op_label):
        return mod_name_long
    if has_op(mod_name_short, op_name):
        return mod_name_short
    if has_op(mod_name_long, op_name):
        return mod_name_long
    if has_op(mod_name_short, unqualified):
        return mod_name_short
    if has_op(mod_name_long, unqualified):
        return mod_name_long
    assert False, (op_label, op_name, class_name)

def _indiv_op_jsn_to_string(jsn: JSON_TYPE, show_imports: bool) -> Tuple[str, str]:
    assert lale.json_operator.json_op_kind(jsn) == 'IndividualOp'
    label = jsn['label']
    if show_imports:
        class_name = jsn['class']
        module_name = _get_module_name(label, jsn['operator'], class_name)
        if module_name.startswith('lale.'):
            op_name = jsn['operator']
        else:
            op_name = class_name[class_name.rfind('.')+1:]
        if op_name == label:
            import_stmt = f'from {module_name} import {op_name}'
        else:
            import_stmt = f'from {module_name} import {op_name} as {label}'
    else:
        import_stmt = ''
    if 'hyperparams' in jsn and jsn['hyperparams'] is not None:
        hps = hyperparams_to_string(jsn['hyperparams'])
        op_expr = f'{label}({hps})'
    else:
        op_expr = label
    return import_stmt, op_expr

def _indiv_op_to_string(op: 'lale.operators.IndividualOp', show_imports: bool, name:str=None, module_name:str=None) -> Tuple[str, str]:
    assert isinstance(op, lale.operators.IndividualOp)
    if name is None:
        name = op.name()
    if show_imports:
        assert module_name is not None
        if module_name.startswith('lale.'):
            op_name = op.name()
        else:
            op_name = op.class_name().split('.')[-1]
        if name == op_name:
            import_stmt = f'from {module_name} import {op_name}'
        else:
            import_stmt = f'from {module_name} import {op_name} as {name}'
    else:
        import_stmt = ''
    if hasattr(op._impl, "fit") and isinstance(op, lale.operators.TrainableIndividualOp):
        hps = hyperparams_to_string(op.hyperparams(), op)
        op_expr = f'{name}({hps})'
    else:
        op_expr = name
    return import_stmt, op_expr

_Seq = collections.namedtuple('_Seq', ['src', 'dst'])
_Par = collections.namedtuple('_Par', ['s0', 's1'])
_Graph = collections.namedtuple('_Graph', ['steps', 'preds', 'succs'])

def _introduce_structure(pipeline: 'lale.operators.BasePipeline') -> Union[_Graph, 'lale.operators.Operator']:
    assert isinstance(pipeline, lale.operators.BasePipeline)
    def make_graph(pipeline: 'lale.operators.BasePipeline') -> _Graph:
        if isinstance(pipeline, lale.operators.OperatorChoice):
            return [pipeline], {pipeline:[]}, {pipeline:[]}
        steps = [*pipeline.steps()]
        preds: Dict[Any, List[Any]] = { step: [] for step in steps }
        succs: Dict[Any, List[Any]] = { step: [] for step in steps }
        for (src, dst) in pipeline.edges():
            preds[dst].append(src)
            succs[src].append(dst)
        return _Graph(steps, preds, succs)
    def find_seq(graph: _Graph) -> Optional[_Seq]:
        for src in graph.steps:
            if len(graph.succs[src]) == 1:
                dst = graph.succs[src][0]
                if len(graph.preds[dst]) == 1:
                    return _Seq(src, dst)
        return None
    def find_par(graph: _Graph) -> Optional[_Par]:
        for i0 in range(len(graph.steps)):
            for i1 in range(i0 + 1, len(graph.steps)):
                s0, s1 = graph.steps[i0], graph.steps[i1]
                preds0, preds1 = graph.preds[s0], graph.preds[s1]
                if len(preds0) == len(preds1) and set(preds0) == set(preds1):
                    succs0, succs1 = graph.succs[s0], graph.succs[s1]
                    if len(succs0)==len(succs1) and set(succs0)==set(succs1):
                        return _Par(s0, s1)
        return None
    def replace_seq(old_graph: _Graph, seq: _Seq) -> _Graph:
        result = _Graph([], {}, {})
        for step in old_graph.steps: #careful to keep topological order
            if step is seq.src:
                result.steps.append(seq)
                result.preds[seq] = old_graph.preds[seq.src]
                result.succs[seq] = old_graph.succs[seq.dst]
            elif step is not seq.dst:
                result.steps.append(step)
                def map_step(step):
                    if step in [seq.src, seq.dst]:
                        return seq
                    return step
                result.preds[step] = [
                    map_step(pred) for pred in old_graph.preds[step]]
                result.succs[step] = [
                    map_step(succ) for succ in old_graph.succs[step]]
        return result
    def replace_par(old_graph: _Graph, par: _Par) -> _Graph:
        result = _Graph([], {}, {})
        for step in old_graph.steps: #careful to keep topological order
            if step is par.s0:
                result.steps.append(par)
                result.preds[par] = old_graph.preds[step]
                result.succs[par] = old_graph.succs[step]
            elif step is not par.s1:
                result.steps.append(step)
                result.preds[step] = []
                for pred in old_graph.preds[step]:
                    if pred is par.s0:
                        result.preds[step].append(par)
                    elif pred is not par.s1:
                        result.preds[step].append(pred)
                result.succs[step] = []
                for succ in old_graph.succs[step]:
                    if succ is par.s0:
                        result.succs[step].append(par)
                    elif succ is not par.s1:
                        result.succs[step].append(succ)
        return result
    def replace_reducibles(graph: _Graph) -> Union[_Graph, 'lale.operators.Operator']:
        progress = True
        while progress:
            seq = find_seq(graph)
            if seq is not None:
                graph = replace_seq(graph, seq)
            par = find_par(graph)
            if par is not None:
                graph = replace_par(graph, par)
            progress = seq is not None or par is not None
        if len(graph.steps) == 1:
            return graph.steps[0]
        else:
            return graph
    graph = make_graph(pipeline)
    result = replace_reducibles(graph)
    return result

class _CodeGenState:
    def __init__(self):
        self.imports = []
        self.assigns = []
        self._names = {'lale','pipeline','get_pipeline_of_applicable_type'}

    def gensym(self, prefix):
        if prefix in self._names:
            suffix = 1
            while f'{prefix}_{suffix}' in self._names:
                suffix += 1
            result = f'{prefix}_{suffix}'
        else:
            result = prefix
        self._names |= {result}
        return result

def _pipeline_to_string_rec(graph: Union[_Graph, 'lale.operators.Operator'], show_imports: bool, cls2name: Dict[str,str], gen: _CodeGenState):
    if isinstance(graph, _Graph):
        steps, preds, succs = graph
        dummy = gen.gensym('step')
        step2name = {}
        for step in steps:
            if isinstance(step, lale.operators.IndividualOp):
                step2name[step] = _pipeline_to_string_rec(step, show_imports, cls2name, gen)
            else:
                name = gen.gensym('step')
                expr = _pipeline_to_string_rec(step, show_imports, cls2name, gen)
                gen.assigns.append(f'{name} = {expr}')
                step2name[step] = name
        make_pipeline = 'get_pipeline_of_applicable_type'
        gen.imports.append(f'from lale.operators import {make_pipeline}')
        gen.assigns.append(
            'pipeline = {}(\n    steps=[{}],\n    edges=[{}])'.format(
                make_pipeline,
                ', '.join([step2name[step] for step in steps]),
                ', '.join([f'({step2name[src]},{step2name[tgt]})'
                           for src in steps for tgt in succs[src]])))
        return None
    elif isinstance(graph, _Seq):
        def parens(op):
            result = _pipeline_to_string_rec(op, show_imports, cls2name, gen)
            if isinstance(op, _Par) or isinstance(op, lale.operators.OperatorChoice):
                return f'({result})'
            return result
        return f'{parens(graph.src)} >> {parens(graph.dst)}'
    elif isinstance(graph, _Par):
        def parens(op):
            result = _pipeline_to_string_rec(op, show_imports, cls2name, gen)
            if isinstance(op, _Seq) or isinstance(op, lale.operators.OperatorChoice):
                return f'({result})'
            return result
        return f'{parens(graph.s0)} & {parens(graph.s1)}'
    elif isinstance(graph, lale.operators.OperatorChoice):
        def parens(op):
            result = _pipeline_to_string_rec(op, show_imports, cls2name, gen)
            if isinstance(op, _Seq) or isinstance(op, _Par):
                return f'({result})'
            return result
        printed_steps = [parens(step) for step in graph.steps()]
        return ' | '.join(printed_steps)
    elif isinstance(graph, lale.operators.IndividualOp):
        name = gen.gensym(cls2name[graph.class_name()])
        module_name = _get_module_name(graph.name(), graph.name(), graph.class_name())
        import_stmt, op_expr = _indiv_op_to_string(graph, True, name, module_name)
        gen.imports.append(import_stmt)
        if re.fullmatch(r'.+\(.+\)', op_expr):
            new_name = gen.gensym(lale.helpers.camelCase_to_snake(name))
            gen.assigns.append(f'{new_name} = {op_expr}')
            return new_name
        else:
            return name
    else:
        assert False, f'unexpected type {type(graph)} of graph {graph}'

def _pipeline_to_string(pipeline: 'lale.operators.BasePipeline', show_imports: bool, cls2name: Dict[str,str]) -> str:
    assert isinstance(pipeline, lale.operators.BasePipeline)
    graph = _introduce_structure(pipeline)
    gen = _CodeGenState()
    expr = _pipeline_to_string_rec(graph, show_imports, cls2name, gen)
    if expr:
        gen.assigns.append(f'pipeline = {expr}')
    code = (gen.imports if show_imports else []) + gen.assigns
    result = '\n'.join(code)
    return result

def schema_to_string(schema: JSON_TYPE) -> str:
    s1 = json.dumps(schema)
    s2 = ast.parse(s1)
    s3 = astunparse.unparse(s2).strip()
    s4 = re.sub(r'}, {\n    (\s+)', r'},\n\1{   ', s3)
    s5 = re.sub(r'\[{\n    (\s+)', r'[\n\1{   ', s4)
    s6 = re.sub(r"'\$schema':[^\n{}\[\]]+\n\s+", "\1", s5)
    while True:
        s7 = re.sub(r',\n\s*([\]}])', r'\1', s6)
        if s6 == s7:
            break
        s6 = s7
    s8 = re.sub(r'{\s+}', r'{}', s7)
    return s8

def to_string(arg: Union[JSON_TYPE, 'lale.operators.Operator'], show_imports:bool=True, call_depth:int=2) -> str:
    def get_cls2name():
        frame = inspect.stack()[call_depth][0]
        result = {}
        all_items = [*frame.f_locals.items(), *frame.f_globals.items()]
        for nm, op in all_items:
            if isinstance(op, lale.operators.IndividualOp) and nm[0].isupper():
                cls = op.class_name()
                if cls not in result:
                    result[cls] = nm
        return result
    if lale.helpers.is_schema(arg):
        return schema_to_string(cast(JSON_TYPE, arg))
    elif isinstance(arg, lale.operators.IndividualOp):
        jsn = lale.json_operator.to_json(arg, call_depth=2)
        import_stmt, op_expr = _indiv_op_jsn_to_string(jsn, show_imports)
        if import_stmt == '':
            return op_expr
        else:
            return import_stmt + '\npipeline = ' + op_expr
    elif isinstance(arg, lale.operators.BasePipeline):
        return _pipeline_to_string(arg, show_imports, get_cls2name())
    else:
        raise ValueError(f'Unexpected argument type {type(arg)} for {arg}')

def ipython_display(arg: Union[JSON_TYPE, 'lale.operators.Operator'], show_imports:bool=True):
    import IPython.display
    pretty_printed = to_string(arg, show_imports, call_depth=3)
    markdown = IPython.display.Markdown(f'```python\n{pretty_printed}\n```')
    IPython.display.display(markdown)
