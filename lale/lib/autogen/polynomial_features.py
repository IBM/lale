
from sklearn.preprocessing.data import PolynomialFeatures as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class PolynomialFeaturesImpl():

    def __init__(self, degree=2, interaction_only=False, include_bias=True):
        self._hyperparams = {
            'degree': degree,
            'interaction_only': interaction_only,
            'include_bias': include_bias}
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
    'description': 'inherited docstring for PolynomialFeatures    Generate polynomial and interaction features.',
    'allOf': [{
        'type': 'object',
        'required': ['degree', 'interaction_only', 'include_bias'],
        'relevantToOptimizer': ['degree', 'interaction_only'],
        'additionalProperties': False,
        'properties': {
            'degree': {
                'type': 'integer',
                'minimumForOptimizer': 2,
                'maximumForOptimizer': 3,
                'distribution': 'uniform',
                'default': 2,
                'description': 'The degree of the polynomial features'},
            'interaction_only': {
                'type': 'boolean',
                'default': False,
                'description': 'If true, only interaction features are produced: features that are products of at most ``degree`` *distinct* input features (so not ``x[1] ** 2``, ``x[0] * x[2] ** 3``, etc.).'},
            'include_bias': {
                'type': 'boolean',
                'default': True,
                'description': 'If True (default), then include a bias column, the feature in which all polynomial powers are zero (i.e'},
        }}, {
        'XXX TODO XXX': 'Parameter: interaction_only > only interaction features are produced: features that are products of at most degree *distinct* input features (so not x[1] ** 2'}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Compute number of output features.',
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
            'description': 'The data.'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transform data to polynomial features',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'laleType': 'Any',
                    'XXX TODO XXX': 'item type'},
                'XXX TODO XXX': 'array-like or sparse matrix, shape [n_samples, n_features]'}, {
                'type': 'array',
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }}],
            'description': 'The data to transform, row by row'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'The matrix of features, where NP is the number of polynomial features generated from the combination of inputs.',
    'laleType': 'Any',
    'XXX TODO XXX': 'np.ndarray or CSC sparse matrix, shape [n_samples, NP]',
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
PolynomialFeatures = lale.operators.make_operator(PolynomialFeaturesImpl, _combined_schemas)

