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
import importlib
import json
import keyword
import pprint
import re
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

import lale.helpers
import lale.json_operator
import lale.operators

JSON_TYPE = Dict[str, Any]

def hyperparams_to_string(hps: JSON_TYPE, steps:Optional[JSON_TYPE]=None) -> str:
    def value_to_string(value):
        if isinstance(value, dict) and '$ref' in value and steps is not None:
            step_uid = value['$ref'].split('/')[-1]
            return steps[step_uid]
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

class _CodeGenState:
    imports: List[str]
    assigns: List[str]
    _names: Set[str]

    def __init__(self, names: Set[str]):
        self.imports = []
        self.assigns = []
        self._names = ({'pipeline', 'get_pipeline_of_applicable_type'}
                       | {'lale', 'make_pipeline', 'make_union', 'make_choice'}
                       | set(keyword.kwlist) | names)

    def gensym(self, prefix: str) -> str:
        if prefix in self._names:
            suffix = 0
            while f'{prefix}_{suffix}' in self._names:
                suffix += 1
            result = f'{prefix}_{suffix}'
        else:
            result = prefix
        self._names |= {result}
        return result

def op_kind(op: JSON_TYPE) -> str:
    assert isinstance(op, dict)
    if 'kind' in op:
        return op['kind']
    return lale.json_operator.json_op_kind(op)

def _dict_key_at(d: dict, i: int) -> str:
    return list(d.keys())[i]

def _dict_val_at(d: dict, i: int) -> Any:
    return list(d.values())[i]

def _introduce_structure(pipeline: JSON_TYPE, gen: _CodeGenState) -> JSON_TYPE:
    assert op_kind(pipeline) == 'Pipeline'
    def make_graph(pipeline: JSON_TYPE) -> JSON_TYPE:
        steps = pipeline['steps']
        preds: Dict[str, List[str]] = { step: [] for step in steps }
        succs: Dict[str, List[str]] = { step: [] for step in steps }
        for (src, dst) in pipeline['edges']:
            preds[dst].append(src)
            succs[src].append(dst)
        return {'kind':'Graph', 'steps':steps, 'preds':preds, 'succs':succs}
    def find_seq(graph: JSON_TYPE) -> Optional[JSON_TYPE]:
        for src in graph['steps']:
            if len(graph['succs'][src]) == 1:
                dst = graph['succs'][src][0]
                if len(graph['preds'][dst]) == 1:
                    return {'kind': 'Seq',
                            'steps': {src: graph['steps'][src],
                                      dst: graph['steps'][dst]}}
        return None
    def find_par(graph: JSON_TYPE) -> Optional[JSON_TYPE]:
        step_uids = list(graph['steps'].keys())
        for i0 in range(len(step_uids)):
            for i1 in range(i0 + 1, len(step_uids)):
                s0, s1 = step_uids[i0], step_uids[i1]
                preds0, preds1 = graph['preds'][s0], graph['preds'][s1]
                if len(preds0) == len(preds1) and set(preds0) == set(preds1):
                    succs0, succs1 = graph['succs'][s0], graph['succs'][s1]
                    if len(succs0)==len(succs1) and set(succs0)==set(succs1):
                        return {'kind': 'Par',
                                'steps': {s0: graph['steps'][s0],
                                          s1: graph['steps'][s1]}}
        return None
    def replace_seq(old_graph: JSON_TYPE, seq: JSON_TYPE) -> JSON_TYPE:
        assert op_kind(old_graph) == 'Graph' and op_kind(seq) == 'Seq'
        old_steps = old_graph['steps']
        old_preds = old_graph['preds']
        old_succs = old_graph['succs']
        new_steps: Dict[str, JSON_TYPE] = {}
        new_preds: Dict[str, List[str]] = {}
        new_succs: Dict[str, List[str]] = {}
        seq_uid = gen.gensym('pipeline')
        for step_uid in old_graph['steps']: #careful to keep topological order
            if step_uid == _dict_key_at(seq['steps'], 0):
                new_steps[seq_uid] = seq
                new_preds[seq_uid] = old_preds[_dict_key_at(seq['steps'], 0)]
                new_succs[seq_uid] = old_succs[_dict_key_at(seq['steps'], 1)]
            elif step_uid != _dict_key_at(seq['steps'], 1):
                new_steps[step_uid] = old_steps[step_uid]
                def map_step(s):
                    if s in list(seq['steps'].keys()):
                        return seq_uid
                    return s
                new_preds[step_uid] = [
                    map_step(pred) for pred in old_preds[step_uid]]
                new_succs[step_uid] = [
                    map_step(succ) for succ in old_succs[step_uid]]
        return {'kind': 'Graph', 'steps': new_steps,
                'preds': new_preds, 'succs': new_succs}
    def replace_par(old_graph: JSON_TYPE, par: JSON_TYPE) -> JSON_TYPE:
        assert op_kind(old_graph) == 'Graph' and op_kind(par) == 'Par'
        old_steps = old_graph['steps']
        old_preds = old_graph['preds']
        old_succs = old_graph['succs']
        new_steps: Dict[str, JSON_TYPE] = {}
        new_preds: Dict[str, List[str]] = {}
        new_succs: Dict[str, List[str]] = {}
        par_uid = gen.gensym('pipeline')
        for step_uid in old_steps: #careful to keep topological order
            if step_uid == _dict_key_at(par['steps'], 0):
                new_steps[par_uid] = par
                new_preds[par_uid] = old_preds[step_uid]
                new_succs[par_uid] = old_succs[step_uid]
            elif step_uid != _dict_key_at(par['steps'], 1):
                new_steps[step_uid] = old_steps[step_uid]
                new_preds[step_uid] = []
                for pred in old_preds[step_uid]:
                    if pred == _dict_key_at(par['steps'], 0):
                        new_preds[step_uid].append(par_uid)
                    elif pred != _dict_key_at(par['steps'], 1):
                        new_preds[step_uid].append(pred)
                new_succs[step_uid] = []
                for succ in old_succs[step_uid]:
                    if succ == _dict_key_at(par['steps'], 0):
                        new_succs[step_uid].append(par_uid)
                    elif succ != _dict_key_at(par['steps'], 1):
                        new_succs[step_uid].append(succ)
        return {'kind': 'Graph', 'steps': new_steps,
                'preds': new_preds, 'succs': new_succs}
    def replace_reducibles(graph: JSON_TYPE) -> JSON_TYPE:
        progress = True
        while progress:
            seq = find_seq(graph)
            if seq is not None:
                graph = replace_seq(graph, seq)
            par = find_par(graph)
            if par is not None:
                graph = replace_par(graph, par)
            progress = seq is not None or par is not None
        if len(graph['steps']) == 1:
            return _dict_val_at(graph['steps'], 0)
        else:
            return graph
    graph = make_graph(pipeline)
    result = replace_reducibles(graph)
    return result

def _operator_jsn_to_string_rec(uid: str, jsn: JSON_TYPE, gen: _CodeGenState) -> str:
    if op_kind(jsn) == 'Pipeline':
        structured = _introduce_structure(jsn, gen)
        return _operator_jsn_to_string_rec(uid, structured, gen)
    elif op_kind(jsn) == 'Graph':
        steps, preds, succs = jsn['steps'], jsn['preds'], jsn['succs']
        step2name: Dict[str, str] = {}
        for step_uid, step_val in steps.items():
            expr = _operator_jsn_to_string_rec(step_uid, step_val, gen)
            if re.fullmatch('[A-Za-z][A-Za-z0-9_]*', expr):
                step2name[step_uid] = expr
            else:
                step2name[step_uid] = step_uid
                gen.assigns.append(f'{step_uid} = {expr}')
        make_pipeline = 'get_pipeline_of_applicable_type'
        gen.imports.append(f'from lale.operators import {make_pipeline}')
        result = '{}(steps=[{}], edges=[{}])'.format(
            make_pipeline,
            ', '.join([step2name[step] for step in steps]),
            ', '.join([f'({step2name[src]},{step2name[tgt]})'
                       for src in steps for tgt in succs[src]]))
        return result
    elif op_kind(jsn) == 'Seq':
        def parens(step_uid, step_val):
            result = _operator_jsn_to_string_rec(step_uid, step_val, gen)
            if op_kind(step_val) in ['Par', 'OperatorChoice']:
                return f'({result})'
            return result
        steps = jsn['steps']
        return f'{parens(_dict_key_at(steps, 0), _dict_val_at(steps, 0))} >> {parens(_dict_key_at(steps, 1), _dict_val_at(steps, 1))}'
    elif op_kind(jsn) == 'Par':
        def parens(step_uid, step_val):
            result = _operator_jsn_to_string_rec(step_uid, step_val, gen)
            if op_kind(step_val) in ['Seq', 'OperatorChoice']:
                return f'({result})'
            return result
        steps = jsn['steps']
        return f'{parens(_dict_key_at(steps, 0), _dict_val_at(steps, 0))} & {parens(_dict_key_at(steps, 1), _dict_val_at(steps, 1))}'
    elif op_kind(jsn) == 'OperatorChoice':
        def parens(step_uid, step_val):
            result = _operator_jsn_to_string_rec(step_uid, step_val, gen)
            if op_kind(step_val) in ['Seq', 'Par']:
                return f'({result})'
            return result
        printed_steps = {step_uid: parens(step_uid, step_val)
                         for step_uid, step_val in jsn['steps'].items()}
        return ' | '.join(printed_steps.values())
    elif op_kind(jsn) == 'IndividualOp':
        label = jsn['label']
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
        gen.imports.append(import_stmt)
        printed_steps = {
            step_uid: _operator_jsn_to_string_rec(step_uid, step_val, gen)
            for step_uid, step_val in jsn.get('steps', {}).items()}
        if 'hyperparams' in jsn and jsn['hyperparams'] is not None:
            hp_string = hyperparams_to_string(jsn['hyperparams'], printed_steps)
            op_expr = f'{label}({hp_string})'
        else:
            op_expr = label
        if re.fullmatch(r'.+\(.+\)', op_expr):
            gen.assigns.append(f'{uid} = {op_expr}')
            return uid
        else:
            return op_expr
    else:
        assert False, f'unexpected type {type(jsn)} of jsn {jsn}'

def _collect_names(jsn: JSON_TYPE) -> Set[str]:
    result: Set[str] = set()
    if 'steps' in jsn:
        for step_uid, step_jsn in jsn['steps'].items():
            result |= {step_uid}
            result |= _collect_names(step_jsn)
    if 'label' in jsn:
        result |= {jsn['label']}
    return result

def _operator_jsn_to_string(jsn: JSON_TYPE, show_imports: bool) -> str:
    gen = _CodeGenState(_collect_names(jsn))
    expr = _operator_jsn_to_string_rec('pipeline', jsn, gen)
    if expr != 'pipeline':
        gen.assigns.append(f'pipeline = {expr}')
    if show_imports:
        imports_set: Set[str] = set()
        imports_list: List[str] = []
        for imp in gen.imports:
            if imp not in imports_set:
                imports_set |= {imp}
                imports_list.append(imp)
        code = imports_list + gen.assigns
    else:
        code = gen.assigns
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

def to_string(arg: Union[JSON_TYPE, 'lale.operators.Operator'], show_imports:bool=True, call_depth:int=1) -> str:
    if lale.helpers.is_schema(arg):
        return schema_to_string(cast(JSON_TYPE, arg))
    elif isinstance(arg, lale.operators.Operator):
        jsn = lale.json_operator.to_json(arg, call_depth=call_depth+1)
        return _operator_jsn_to_string(jsn, show_imports)
    else:
        raise ValueError(f'Unexpected argument type {type(arg)} for {arg}')

def ipython_display(arg: Union[JSON_TYPE, 'lale.operators.Operator'], show_imports:bool=True):
    import IPython.display
    pretty_printed = to_string(arg, show_imports, call_depth=3)
    markdown = IPython.display.Markdown(f'```python\n{pretty_printed}\n```')
    IPython.display.display(markdown)
