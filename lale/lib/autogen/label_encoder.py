
from sklearn.preprocessing.label import LabelEncoder as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class LabelEncoderImpl():

    def __init__(self):
        self._hyperparams = {
            
        }
        self._sklearn_model = SKLModel(**self._hyperparams)

    def fit(self, X, y=None):
        if (y is not None):
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X)
        return self

    def transform(self, X):
        return self._sklearn_model.transform(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for LabelEncoder    Encode labels with value between 0 and n_classes-1.',
    'allOf': [{
        'type': 'object',
        'required': [],
        'relevantToOptimizer': [],
        'additionalProperties': False,
        'properties': {
            
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit label encoder',
    'type': 'object',
    'required': ['y'],
    'properties': {
        'y': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'Target values.'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transform labels to normalized encoding.',
    'type': 'object',
    'required': ['y'],
    'properties': {
        'y': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'Target values.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transform labels to normalized encoding.',
    'type': 'array',
    'items': {
        'type': 'number'},
}
_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'type': 'object',
    'tags': {
        'pre': [],
        'op': ['transformer'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_transform': _input_transform_schema,
        'output_transform': _output_transform_schema},
}
if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
LabelEncoder = lale.operators.make_operator(LabelEncoderImpl, _combined_schemas)

