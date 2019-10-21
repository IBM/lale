
from sklearn.decomposition.factor_analysis import FactorAnalysis as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class FactorAnalysisImpl():

    def __init__(self, n_components=None, tol=0.01, copy=True, max_iter=1000, noise_variance_init=None, svd_method='randomized', iterated_power=3, random_state=0):
        self._hyperparams = {
            'n_components': n_components,
            'tol': tol,
            'copy': copy,
            'max_iter': max_iter,
            'noise_variance_init': noise_variance_init,
            'svd_method': svd_method,
            'iterated_power': iterated_power,
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
    'description': 'inherited docstring for FactorAnalysis    Factor Analysis (FA)',
    'allOf': [{
        'type': 'object',
        'required': ['n_components', 'tol', 'copy', 'max_iter', 'noise_variance_init', 'svd_method', 'iterated_power', 'random_state'],
        'relevantToOptimizer': ['n_components', 'tol', 'copy', 'max_iter', 'svd_method', 'iterated_power'],
        'additionalProperties': False,
        'properties': {
            'n_components': {
                'enum': ['int', None],
                'default': None,
                'description': 'Dimensionality of latent space, the number of components'},
            'tol': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'distribution': 'loguniform',
                'default': 0.01,
                'description': 'Stopping tolerance for EM algorithm.'},
            'copy': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether to make a copy of X. If ``False``, the input X gets overwritten'},
            'max_iter': {
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 1000,
                'description': 'Maximum number of iterations.'},
            'noise_variance_init': {
                'XXX TODO XXX': 'None | array, shape=(n_features,)',
                'description': 'The initial guess of the noise variance for each feature.',
                'enum': [None],
                'default': None},
            'svd_method': {
                'enum': ['lapack', 'randomized'],
                'default': 'randomized',
                'description': "Which SVD method to use. If 'lapack' use standard SVD from"},
            'iterated_power': {
                'type': 'integer',
                'minimumForOptimizer': 3,
                'maximumForOptimizer': 4,
                'distribution': 'uniform',
                'default': 3,
                'description': 'Number of iterations for the power method. 3 by default. Only used'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': 0,
                'description': 'If int, random_state is the seed used by the random number generator;'},
        }}, {
        'XXX TODO XXX': "Parameter: iterated_power > only used if svd_method equals 'randomized'"}, {
        'XXX TODO XXX': "Parameter: random_state > only used when svd_method equals 'randomized'"}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the FactorAnalysis model to X using EM',
    'type': 'object',
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
    'description': 'Apply dimensionality reduction to X using the model.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'Training data.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'The latent variables of X.',
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
FactorAnalysis = lale.operators.make_operator(FactorAnalysisImpl, _combined_schemas)

