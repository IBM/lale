
from sklearn.preprocessing.data import MinMaxScaler as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class MinMaxScalerImpl():

    def __init__(self, feature_range=(0, 1), copy=True):
        self._hyperparams = {
            'feature_range': feature_range,
            'copy': copy}

    def fit(self, X, y=None):
        self._sklearn_model = SKLModel(**self._hyperparams)
        if (y is not None):
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X)
        return self

    def transform(self, X):
        return self._sklearn_model.transform(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for MinMaxScaler    Transforms features by scaling each feature to a given range.',
    'allOf': [{
        'type': 'object',
        'required': ['feature_range', 'copy'],
        'relevantToOptimizer': ['copy'],
        'additionalProperties': False,
        'properties': {
            'feature_range': {
                'XXX TODO XXX': 'tuple (min, max), default=(0, 1)',
                'description': 'Desired range of transformed data.',
                'type': 'array',
                'typeForOptimizer': 'tuple',
                'default': (0, 1)},
            'copy': {
                'type': 'boolean',
                'default': True,
                'description': 'Set to False to perform inplace row normalization and avoid a'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Compute the minimum and maximum to be used for later scaling.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'The data used to compute the per-feature minimum and maximum'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Scaling features of X according to feature_range.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'Input data that will be transformed.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Scaling features of X according to feature_range.',
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
MinMaxScaler = lale.operators.make_operator(MinMaxScalerImpl, _combined_schemas)

