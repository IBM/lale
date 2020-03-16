
from sklearn.preprocessing.data import PowerTransformer as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class PowerTransformerImpl():

    def __init__(self, method='yeo-johnson', standardize=True, copy=True):
        self._hyperparams = {
            'method': method,
            'standardize': standardize,
            'copy': copy}
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
    'description': 'inherited docstring for PowerTransformer    Apply a power transform featurewise to make data more Gaussian-like.',
    'allOf': [{
        'type': 'object',
        'required': ['method', 'standardize', 'copy'],
        'relevantToOptimizer': [],
        'additionalProperties': False,
        'properties': {
            'method': {
                'type': 'string',
                'default': 'yeo-johnson',
                'description': 'The power transform method'},
            'standardize': {
                'type': 'boolean',
                'default': True,
                'description': 'Set to True to apply zero-mean, unit-variance normalization to the transformed output.'},
            'copy': {
                'type': 'boolean',
                'default': True,
                'description': 'Set to False to perform inplace computation during transformation.'},
        }}, {
        'XXX TODO XXX': 'Parameter: method > only works with strictly positive values'}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Estimate the optimal parameter lambda for each feature.',
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
            'description': 'The data used to estimate the optimal transformation parameters.'},
        'y': {
            
        }},
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Apply the power transform to each feature using the fitted lambdas.',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'The data to be transformed using a power transformation.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'The transformed data.',
    'type': 'array',
    'items': {
        'type': 'array',
        'items': {
            'type': 'number'},
    },
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
PowerTransformer = lale.operators.make_operator(PowerTransformerImpl, _combined_schemas)

