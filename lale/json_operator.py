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

import importlib
import inspect
import jsonschema
import keyword
import lale.helpers
import lale.operators
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

SCHEMA = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'definitions': {
    'operator': {
      'anyOf': [
        {'$ref': '#/definitions/planned_individual_op'},
        {'$ref': '#/definitions/trainable_individual_op'},
        {'$ref': '#/definitions/trained_individual_op'},
        {'$ref': '#/definitions/planned_pipeline'},
        {'$ref': '#/definitions/trainable_pipeline'},
        {'$ref': '#/definitions/trained_pipeline'},
        {'$ref': '#/definitions/operator_choice'}]},
    'individual_op': {
      'type': 'object',
      'required': ['class', 'state', 'operator'],
      'properties': {
        'class': {
          'type': 'string',
          'pattern': '^([A-Za-z_][A-Za-z_0-9]*[.])*[A-Za-z_][A-Za-z_0-9]*$'},
        'state': {
          'enum': ['metamodel', 'planned', 'trainable', 'trained'] },
        'operator': {
          'type': 'string',
          'pattern': '^[A-Za-z_][A-Za-z_0-9]*$'},
        'label': {
          'type': 'string',
          'pattern': '^[A-Za-z_][A-Za-z_0-9]*$'},
        'id': {
          'type': 'string',
          'pattern': '^[a-z][a-z_0-9]*$'},
        'documentation_url': {
          'type': 'string'},
        'hyperparams': {
          'anyOf': [
            { 'enum': [None]},
            { 'type': 'object',
              'patternProperties': {'^[A-Za-z_][A-Za-z_0-9]*$': {}}}]},
        'is_frozen_trainable': {
          'type': 'boolean'},
        'is_frozen_trained': {
          'type': 'boolean'},
        'coefs': {
          'enum': [None, 'coefs_not_available']}}},
    'planned_individual_op': {
      'allOf': [
        { '$ref': '#/definitions/individual_op'},
        { 'type': 'object',
          'properties': { 'state': { 'enum': ['planned']}}}]},
    'trainable_individual_op': {
      'allOf': [
        { '$ref': '#/definitions/individual_op'},
        { 'type': 'object',
          'required': ['hyperparams', 'is_frozen_trainable'],
          'properties': { 'state': { 'enum': ['trainable']}}}]},
    'trained_individual_op': {
      'allOf': [
        { '$ref': '#/definitions/individual_op'},
        { 'type': 'object',
          'required': ['hyperparams', 'coefs', 'is_frozen_trained'],
          'properties': { 'state': { 'enum': ['trained']}}}]},
    'pipeline': {
      'type': 'object',
      'required': ['class', 'state', 'edges', 'steps'],
      'properties': {
        'class': {
          'enum': [
            'lale.operators.PlannedPipeline',
            'lale.operators.TrainablePipeline',
            'lale.operators.TrainedPipeline']},
        'state': {
          'enum': ['planned', 'trainable', 'trained']},
        'edges': {
          'type': 'array',
          'items': {
            'type': 'array',
            'minItems': 2, 'maxItems': 2,
            'items': {'type': 'integer'}}},
        'steps': {
          'type': 'array',
          'items': {'$ref': '#/definitions/operator'}}}},
    'planned_pipeline': {
      'allOf': [
        { '$ref': '#/definitions/pipeline'},
        { 'type': 'object',
          'properties': {
            'state': { 'enum': ['planned']},
            'class': { 'enum': ['lale.operators.PlannedPipeline']}}}]},
    'trainable_pipeline': {
      'allOf': [
        { '$ref': '#/definitions/pipeline'},
        { 'type': 'object',
          'properties': {
            'state': { 'enum': ['trainable']},
            'class': { 'enum': ['lale.operators.TrainablePipeline']},
            'steps': {
              'type': 'array',
              'items': {
                'type': 'object',
                'properties': {
                  'state': { 'enum': ['trainable', 'trained']}}}}}}]},
    'trained_pipeline': {
      'allOf': [
        { '$ref': '#/definitions/pipeline'},
        { 'type': 'object',
          'properties': {
            'state': { 'enum': ['trained']},
            'class': { 'enum': ['lale.operators.TrainedPipeline']},
            'steps': {
              'type': 'array',
              'items': {
                'type': 'object',
                'properties': {
                  'state': { 'enum': ['trained']}}}}}}]},
    'operator_choice': {
      'type': 'object',
      'required': ['class', 'state', 'operator', 'steps'],
      'properties': {
        'class': {
          'enum': ['lale.operators.OperatorChoice']},
        'state': {
          'enum': ['planned']},
        'operator': {
          'type': 'string'},
        'steps': {
          'type': 'array',
          'items': {'$ref': '#/definitions/operator'}}}}},
  '$ref': '#/definitions/operator'}

if __name__ == "__main__":
    lale.helpers.validate_is_schema(SCHEMA)

def json_op_kind(jsn):
    if 'steps' in jsn and 'edges' in jsn:
        return 'Pipeline'
    elif 'steps' in jsn:
        return 'OperatorChoice'
    return 'IndividualOp'

def _get_state(op) -> str:
    if isinstance(op, lale.operators.Trained):
        return 'trained'
    if isinstance(op, lale.operators.Trainable):
        return 'trainable'
    if isinstance(op, lale.operators.Planned) or isinstance(op, lale.operators.OperatorChoice):
        return 'planned'
    if isinstance(op, lale.operators.MetaModel):
        return 'metamodel'
    raise TypeError(f'Expected lale.operators.Operator, got {type(op)}.')

def _get_cls2label(call_depth: int) -> Dict[str, str]:
    frame = inspect.stack()[call_depth][0]
    cls2label: Dict[str, str] = {}
    cls2state: Dict[str, str] = {}
    all_items = [*frame.f_locals.items(), *frame.f_globals.items()]
    for label, op in all_items:
        if isinstance(op, lale.operators.IndividualOp):
            state = _get_state(op)
            cls = op.class_name()
            if cls in cls2state:
                insert = (
                    (cls2state[cls] == 'trainable' and state == 'planned') or
                    (cls2state[cls] == 'trained' and
                     state in ['trainable', 'planned']))
            else:
                insert = True
            if insert:
                cls2label[cls] = label
                cls2state[cls] = state
    return cls2label

class _GenSym:
    def __init__(self, op, cls2label):
        label2count = {}
        def populate_label2count(op):
            if isinstance(op, lale.operators.IndividualOp):
                label = cls2label.get(op.class_name(), op.name())
            elif isinstance(op, lale.operators.BasePipeline):
                for s in op.steps():
                    populate_label2count(s)
                label = 'pipeline'
            elif isinstance(op, lale.operators.OperatorChoice):
                for s in op.steps():
                    populate_label2count(s)
                label = 'choice'
            label2count[label] = label2count.get(label, 0) + 1
        populate_label2count(op)
        non_unique_labels = {l for l, c in label2count.items() if c > 1}
        snakes = {lale.helpers.camelCase_to_snake(l) for l in non_unique_labels}
        self._names = ({'lale'} | set(keyword.kwlist) |
                       non_unique_labels | snakes)

    def __call__(self, prefix):
        if prefix in self._names:
            suffix = 0
            while f'{prefix}_{suffix}' in self._names:
                suffix += 1
            result = f'{prefix}_{suffix}'
        else:
            result = prefix
        self._names |= {result}
        return result

def _to_json_rec(op, cls2label, gensym) -> Dict[str, Any]:
    result = {}
    result['class'] = op.class_name()
    result['state'] = _get_state(op)
    if isinstance(op, lale.operators.IndividualOp):
        result['operator'] = op.name()
        result['label'] = cls2label.get(op.class_name(), op.name())
        result['id'] = gensym(lale.helpers.camelCase_to_snake(result['label']))
        documentation_url = op.documentation_url()
        if documentation_url is not None:
            result['documentation_url'] = documentation_url
        if isinstance(op, lale.operators.TrainableIndividualOp):
            result['hyperparams'] = op.hyperparams()
            result['is_frozen_trainable'] = op.is_frozen_trainable()
        if isinstance(op, lale.operators.TrainedIndividualOp):
            if hasattr(op._impl, 'fit'):
                result['coefs'] = 'coefs_not_available'
            else:
                result['coefs'] = None
            result['is_frozen_trained'] = op.is_frozen_trained()
    elif isinstance(op, lale.operators.BasePipeline):
        result['id'] = gensym('pipeline')
        node2id = {s: i for (i, s) in enumerate(op.steps())}
        result['edges'] = [
            [node2id[x], node2id[y]] for (x, y) in op.edges()]
        result['steps'] = [
            _to_json_rec(s, cls2label, gensym) for s in op.steps()]
    elif isinstance(op, lale.operators.OperatorChoice):
        result['operator'] = 'OperatorChoice'
        result['id'] = gensym('choice')
        result['state'] = 'planned'
        result['steps'] = [
            _to_json_rec(s, cls2label, gensym) for s in op.steps()]
    return result

def to_json(op, call_depth=1) -> Dict[str, Any]:
    cls2label = _get_cls2label(call_depth + 1)
    gensym = _GenSym(op, cls2label)
    result = _to_json_rec(op, cls2label, gensym)
    jsonschema.validate(result, SCHEMA)
    return result

def _from_json_rec(json: Dict[str, Any]):
    kind = json_op_kind(json)
    if kind == 'Pipeline':
        steps = [_from_json_rec(s) for s in json['steps']]
        edges = [(steps[e[0]], steps[e[1]]) for e in json['edges']]
        return lale.operators.get_pipeline_of_applicable_type(steps, edges)
    elif kind == 'OperatorChoice':
        steps = [_from_json_rec(s) for s in json['steps']]
        name = json['operator']
        return lale.operators.OperatorChoice(steps, name)
    else:
        assert kind == 'IndividualOp'
        name = json['operator']
        full_class_name = json['class']
        last_period = full_class_name.rfind('.')
        module = importlib.import_module(full_class_name[:last_period])
        impl_class = getattr(module, full_class_name[last_period+1:])
        impl = impl_class()
        schemas = None #IndividualOp.__init__ should look up the schemas
        planned = lale.operators.PlannedIndividualOp(name, impl, schemas)
        if json['state'] == 'planned':
            return planned
        assert json['state'] in ['trainable', 'trained'], json["state"]
        if json['hyperparams'] is None:
            trainable = planned()
        else:
            trainable = planned(**json['hyperparams'])
        if json['is_frozen_trainable']:
            trainable = trainable.freeze_trainable()
        if json['state'] == 'trained':
            if json['coefs']=='coefs_not_available':
                logger.warning(f'Since the JSON representation of trained operator {name} lacks coefficients, from_json returns a trainable operator instead.')
            else:
                assert json['coefs'] is None, json['coefs']
                trained = lale.operators.TrainedIndividualOp(
                    name, trainable._impl, schemas)
                assert json['is_frozen_trained'] == trained.is_frozen_trained()
                return trained
        return trainable
    assert False, f'unexpected JSON {json}'

def from_json(json: Dict[str, Any]):
    jsonschema.validate(json, SCHEMA)
    result = _from_json_rec(json)
    return result
