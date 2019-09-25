
from sklearn.random_projection import GaussianRandomProjection as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class GaussianRandomProjectionImpl():

    def __init__(self, n_components='auto', eps=0.1, random_state=None):
        self._hyperparams = {
            'n_components': n_components,
            'eps': eps,
            'random_state': random_state}

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
    'description': 'inherited docstring for GaussianRandomProjection    Reduce dimensionality through Gaussian random projection',
    'allOf': [{
        'type': 'object',
        'relevantToOptimizer': ['n_components', 'eps'],
        'additionalProperties': False,
        'properties': {
            'n_components': {
                'XXX TODO XXX': "int or 'auto', optional (default = 'auto')",
                'description': 'Dimensionality of the target projection space.',
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 2,
                    'maximumForOptimizer': 256,
                    'distribution': 'uniform'}, {
                    'enum': ['auto']}],
                'default': 'auto'},
            'eps': {
                'XXX TODO XXX': 'strictly positive float, optional (default=0.1)',
                'description': 'Parameter to control the quality of the embedding according to',
                'type': 'number',
                'minimumForOptimizer': 0.001,
                'maximumForOptimizer': 0.1,
                'distribution': 'loguniform',
                'default': 0.1},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Control the pseudo random number generator used to generate the matrix'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Generate a sparse random projection matrix',
    'type': 'object',
    'properties': {
        'X': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'XXX TODO XXX': 'item type'},
                'XXX TODO XXX': 'numpy array or scipy.sparse of shape [n_samples, n_features]'}, {
                'type': 'array',
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }}],
            'description': 'Training set: only the shape is used to find optimal random'},
        'y': {
            'XXX TODO XXX': '',
            'description': 'Ignored'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Project the data by using matrix product with the random matrix',
    'type': 'object',
    'properties': {
        'X': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'XXX TODO XXX': 'item type'},
                'XXX TODO XXX': 'numpy array or scipy.sparse of shape [n_samples, n_features]'}, {
                'type': 'array',
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }}],
            'description': 'The input data to project into a smaller dimensional space.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Projected array.',
    'anyOf': [{
        'type': 'array',
        'items': {
            'XXX TODO XXX': 'item type'},
        'XXX TODO XXX': 'numpy array or scipy sparse of shape [n_samples, n_components]'}, {
        'type': 'array',
        'items': {
            'type': 'array',
            'items': {
                'type': 'number'},
        }}],
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
GaussianRandomProjection = lale.operators.make_operator(GaussianRandomProjectionImpl, _combined_schemas)

