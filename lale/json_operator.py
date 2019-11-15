import importlib
import jsonschema
import lale.helpers
import lale.operators
import lale.pretty_print
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
        'documentation_url': {
          'type': 'string'},
        'hyperparams': {
          'anyOf': [
            { 'enum': [None]},
            { 'type': 'object',
              'patternProperties': {'^[A-Za-z_][A-Za-z_0-9]*$': {}}}]},
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
          'required': ['hyperparams'],
          'properties': { 'state': { 'enum': ['trainable']}}}]},
    'trained_individual_op': {
      'allOf': [
        { '$ref': '#/definitions/individual_op'},
        { 'type': 'object',
          'required': ['hyperparams', 'coefs'],
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

def to_json(op):
    result = {}
    result['class'] = op.class_name()
    if isinstance(op, lale.operators.Trained):
        result['state'] = 'trained'
    elif isinstance(op, lale.operators.Trainable):
        result['state'] = 'trainable'
    elif isinstance(op, lale.operators.Planned):
        result['state'] = 'planned'
    elif isinstance(op, lale.operators.MetaModel):
        result['state'] = 'metamodel'
    if isinstance(op, lale.operators.IndividualOp):
        result['operator'] = op.name()
        documentation_url = op.documentation_url()
        if documentation_url is not None:
            result['documentation_url'] = documentation_url
        if isinstance(op, lale.operators.TrainableIndividualOp):
            result['hyperparams'] = op.hyperparams()
        if isinstance(op, lale.operators.TrainedIndividualOp):
            if hasattr(op._impl, 'fit'):
                result['coefs'] = 'coefs_not_available'
            else:
                result['coefs'] = None
    elif isinstance(op, lale.operators.Pipeline):
        node2id = {s: i for (i, s) in enumerate(op.steps())}
        result['edges'] = [[node2id[x], node2id[y]] for (x, y) in op.edges()]
        result['steps'] = [s.to_json() for s in op.steps()]
    elif isinstance(op, lale.operators.OperatorChoice):
        result['operator'] = op.name()
        result['state'] = 'planned'
        result['steps'] = [s.to_json() for s in op.steps()]
    jsonschema.validate(result, SCHEMA)
    return result

def from_json(json):
    jsonschema.validate(json, SCHEMA)
    if 'steps' in json and 'edges' in json:
        steps = [from_json(s) for s in json['steps']]
        edges = [(steps[e[0]], steps[e[1]]) for e in json['edges']]
        return lale.operators.get_pipeline_of_applicable_type(steps, edges)
    elif 'steps' in json:
        steps = [from_json(s) for s in json['steps']]
        name = json['operator']
        return lale.operators.OperatorChoice(steps, name)
    else:
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
        if json['state'] == 'trained':
            if json['coefs']=='coefs_not_available':
                logger.warning(f'Since the JSON representation of trained operator {name} lacks coefficients, from_json returns a trainable operator instead.')
            else:
                assert json['coefs'] is None, json['coefs']
                trained = lale.operators.TrainedIndividualOp(
                    name, trainable._impl, schemas)
                return trained
        return trainable
    assert False, f'unexpected JSON {json}'
