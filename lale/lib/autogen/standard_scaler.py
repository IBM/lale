
from sklearn.preprocessing.data import StandardScaler as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class StandardScalerImpl():

    def __init__(self, copy=True, with_mean=True, with_std=True):
        self._hyperparams = {
            'copy': copy,
            'with_mean': with_mean,
            'with_std': with_std}
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
    'description': 'inherited docstring for StandardScaler    Standardize features by removing the mean and scaling to unit variance',
    'allOf': [{
        'type': 'object',
        'required': ['copy', 'with_mean', 'with_std'],
        'relevantToOptimizer': ['copy', 'with_mean', 'with_std'],
        'additionalProperties': False,
        'properties': {
            'copy': {
                'type': 'boolean',
                'default': True,
                'description': 'If False, try to avoid a copy and do inplace scaling instead'},
            'with_mean': {
                'type': 'boolean',
                'default': True,
                'description': 'If True, center the data before scaling'},
            'with_std': {
                'type': 'boolean',
                'default': True,
                'description': 'If True, scale the data to unit variance (or equivalently, unit standard deviation).'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Compute the mean and std to be used for later scaling.',
    'type': 'object',
    'required': ['X', 'y'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'The data used to compute the mean and standard deviation used for later scaling along the features axis.'},
        'y': {
            'laleType': 'Any',
            'XXX TODO XXX': '',
            'description': 'Ignored'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Perform standardization by centering and scaling',
    'type': 'object',
    'required': ['X', 'y'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'The data used to scale along the features axis.'},
        'y': {
            'laleType': 'Any',
            'XXX TODO XXX': '(ignored)',
            'description': ''},
        'copy': {
            'anyOf': [{
                'type': 'boolean'}, {
                'enum': [None]}],
            'default': None,
            'description': 'Copy the input X or not.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Perform standardization by centering and scaling',
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
StandardScaler = lale.operators.make_operator(StandardScalerImpl, _combined_schemas)

