
from sklearn.decomposition.fastica_ import FastICA as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class FastICAImpl():

    def __init__(self, n_components=None, algorithm='parallel', whiten=True, fun='logcosh', fun_args=None, max_iter=200, tol=0.0001, w_init=None, random_state=None):
        self._hyperparams = {
            'n_components': n_components,
            'algorithm': algorithm,
            'whiten': whiten,
            'fun': fun,
            'fun_args': fun_args,
            'max_iter': max_iter,
            'tol': tol,
            'w_init': w_init,
            'random_state': random_state}
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
    'description': 'inherited docstring for FastICA    FastICA: a fast algorithm for Independent Component Analysis.',
    'allOf': [{
        'type': 'object',
        'required': ['n_components', 'algorithm', 'whiten', 'fun', 'fun_args', 'max_iter', 'tol', 'w_init', 'random_state'],
        'relevantToOptimizer': ['n_components', 'algorithm', 'whiten', 'fun', 'max_iter', 'tol'],
        'additionalProperties': False,
        'properties': {
            'n_components': {
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 2,
                    'maximumForOptimizer': 256,
                    'distribution': 'uniform'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Number of components to use'},
            'algorithm': {
                'enum': ['parallel', 'deflation'],
                'default': 'parallel',
                'description': 'Apply parallel or deflational algorithm for FastICA.'},
            'whiten': {
                'type': 'boolean',
                'default': True,
                'description': 'If whiten is false, the data is already considered to be whitened, and no whitening is performed.'},
            'fun': {
                'XXX TODO XXX': "string or function, optional. Default: 'logcosh'",
                'description': 'The functional form of the G function used in the approximation to neg-entropy',
                'enum': ['exp', 'logcosh'],
                'default': 'logcosh'},
            'fun_args': {
                'XXX TODO XXX': 'dictionary, optional',
                'description': 'Arguments to send to the functional form',
                'enum': [None],
                'default': None},
            'max_iter': {
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 200,
                'description': 'Maximum number of iterations during fit.'},
            'tol': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'distribution': 'loguniform',
                'default': 0.0001,
                'description': 'Tolerance on update at each iteration.'},
            'w_init': {
                'XXX TODO XXX': 'None of an (n_components, n_components) ndarray',
                'description': 'The mixing matrix to be used to initialize the algorithm.',
                'enum': [None],
                'default': None},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by `np.random`.'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the model to X.',
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
            'description': 'Training data, where n_samples is the number of samples and n_features is the number of features.'},
        'y': {
            
        }},
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Recover the sources from X (apply the unmixing matrix).',
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
            'description': 'Data to transform, where n_samples is the number of samples and n_features is the number of features.'},
        'y': {
            'laleType': 'Any',
            'XXX TODO XXX': '(ignored)',
            'description': ''},
        'copy': {
            'laleType': 'Any',
            'XXX TODO XXX': 'bool (optional)',
            'description': 'If False, data passed to fit are overwritten'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Recover the sources from X (apply the unmixing matrix).',
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
FastICA = lale.operators.make_operator(FastICAImpl, _combined_schemas)

