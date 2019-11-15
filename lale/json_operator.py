import jsonschema
import lale.helpers
import lale.operators

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
                'patternProperties': {'^[A-Za-z_][A-Za-z_0-9]*$': {}}}]}}},
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
          'required': ['hyperparams'],
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
