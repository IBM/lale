
from sklearn.decomposition.truncated_svd import TruncatedSVD as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class TruncatedSVDImpl():

    def __init__(self, n_components=2, algorithm='randomized', n_iter=5, random_state=None, tol=0.0):
        self._hyperparams = {
            'n_components': n_components,
            'algorithm': algorithm,
            'n_iter': n_iter,
            'random_state': random_state,
            'tol': tol}
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
    'description': 'inherited docstring for TruncatedSVD    Dimensionality reduction using truncated SVD (aka LSA).',
    'allOf': [{
        'type': 'object',
        'required': ['n_components', 'algorithm', 'n_iter', 'random_state', 'tol'],
        'relevantToOptimizer': ['n_components', 'algorithm', 'n_iter', 'tol'],
        'additionalProperties': False,
        'properties': {
            'n_components': {
                'type': 'integer',
                'minimumForOptimizer': 2,
                'maximumForOptimizer': 256,
                'distribution': 'uniform',
                'default': 2,
                'description': 'Desired dimensionality of output data'},
            'algorithm': {
                'enum': ['arpack', 'randomized'],
                'default': 'randomized',
                'description': 'SVD solver to use'},
            'n_iter': {
                'type': 'integer',
                'minimumForOptimizer': 5,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 5,
                'description': 'Number of iterations for randomized SVD solver'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by `np.random`.'},
            'tol': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'distribution': 'uniform',
                'default': 0.0,
                'description': 'Tolerance for ARPACK'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit LSI model on training data X.',
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
            'description': 'Training data.'},
        'y': {
            
        }},
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Perform dimensionality reduction on X.',
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
            'description': 'New data.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Reduced version of X',
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
TruncatedSVD = lale.operators.make_operator(TruncatedSVDImpl, _combined_schemas)

