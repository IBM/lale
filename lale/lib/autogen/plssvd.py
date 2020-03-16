
from sklearn.cross_decomposition.pls_ import PLSSVD as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class PLSSVDImpl():

    def __init__(self, n_components=2, scale=True, copy=True):
        self._hyperparams = {
            'n_components': n_components,
            'scale': scale,
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
    'description': 'inherited docstring for PLSSVD    Partial Least Square SVD',
    'allOf': [{
        'type': 'object',
        'required': ['n_components', 'scale', 'copy'],
        'relevantToOptimizer': ['n_components', 'scale', 'copy'],
        'additionalProperties': False,
        'properties': {
            'n_components': {
                'type': 'integer',
                'minimumForOptimizer': 2,
                'maximumForOptimizer': 256,
                'distribution': 'uniform',
                'default': 2,
                'description': 'Number of components to keep.'},
            'scale': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether to scale X and Y.'},
            'copy': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether to copy X and Y, or perform in-place computations.'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit model to data.',
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
            'description': 'Training vectors, where n_samples is the number of samples and n_features is the number of predictors.'},
        'Y': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'Target vectors, where n_samples is the number of samples and n_targets is the number of response variables.'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Apply the dimension reduction learned on the train data.',
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
            'description': 'Training vectors, where n_samples is the number of samples and n_features is the number of predictors.'},
        'Y': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'Target vectors, where n_samples is the number of samples and n_targets is the number of response variables.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Apply the dimension reduction learned on the train data.',
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
PLSSVD = lale.operators.make_operator(PLSSVDImpl, _combined_schemas)

