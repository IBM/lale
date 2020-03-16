
from sklearn.kernel_approximation import Nystroem as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class NystroemImpl():

    def __init__(self, kernel='rbf', gamma=None, coef0=None, degree=None, kernel_params=None, n_components=100, random_state=None):
        self._hyperparams = {
            'kernel': kernel,
            'gamma': gamma,
            'coef0': coef0,
            'degree': degree,
            'kernel_params': kernel_params,
            'n_components': n_components,
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
    'description': 'inherited docstring for Nystroem    Approximate a kernel map using a subset of the training data.',
    'allOf': [{
        'type': 'object',
        'required': ['kernel', 'gamma', 'coef0', 'degree', 'kernel_params', 'n_components', 'random_state'],
        'relevantToOptimizer': ['kernel', 'n_components'],
        'additionalProperties': False,
        'properties': {
            'kernel': {
                'anyOf': [{
                    'type': 'object',
                    'forOptimizer': False}, {
                    'enum': ['linear', 'poly', 'rbf', 'sigmoid']}],
                'default': 'rbf',
                'description': 'Kernel map to be approximated'},
            'gamma': {
                'anyOf': [{
                    'type': 'number'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Gamma parameter for the RBF, laplacian, polynomial, exponential chi2 and sigmoid kernels'},
            'coef0': {
                'anyOf': [{
                    'type': 'number'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Zero coefficient for polynomial and sigmoid kernels'},
            'degree': {
                'anyOf': [{
                    'type': 'number'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Degree of the polynomial kernel'},
            'kernel_params': {
                'XXX TODO XXX': 'mapping of string to any, optional',
                'description': 'Additional parameters (keyword arguments) for kernel function passed as callable object.',
                'enum': [None],
                'default': None},
            'n_components': {
                'type': 'integer',
                'minimumForOptimizer': 2,
                'maximumForOptimizer': 256,
                'distribution': 'uniform',
                'default': 100,
                'description': 'Number of features to construct'},
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
    'description': 'Fit estimator to data.',
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
            'description': 'Training data.'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Apply feature map to X.',
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
            'description': 'Data to transform.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transformed data.',
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
Nystroem = lale.operators.make_operator(NystroemImpl, _combined_schemas)

