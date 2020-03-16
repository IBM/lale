
from sklearn.kernel_ridge import KernelRidge as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class KernelRidgeImpl():

    def __init__(self, alpha=1, kernel='linear', gamma=None, degree=3, coef0=1, kernel_params=None):
        self._hyperparams = {
            'alpha': alpha,
            'kernel': kernel,
            'gamma': gamma,
            'degree': degree,
            'coef0': coef0,
            'kernel_params': kernel_params}
        self._sklearn_model = SKLModel(**self._hyperparams)

    def fit(self, X, y=None):
        if (y is not None):
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X)
        return self

    def predict(self, X):
        return self._sklearn_model.predict(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for KernelRidge    Kernel ridge regression.',
    'allOf': [{
        'type': 'object',
        'required': ['alpha', 'kernel', 'gamma', 'degree', 'coef0', 'kernel_params'],
        'relevantToOptimizer': ['alpha', 'kernel', 'degree', 'coef0'],
        'additionalProperties': False,
        'properties': {
            'alpha': {
                'XXX TODO XXX': '{float, array-like}, shape = [n_targets]',
                'description': 'Small positive values of alpha improve the conditioning of the problem and reduce the variance of the estimates',
                'type': 'integer',
                'minimumForOptimizer': 1,
                'maximumForOptimizer': 2,
                'distribution': 'uniform',
                'default': 1},
            'kernel': {
                'anyOf': [{
                    'type': 'object',
                    'forOptimizer': False}, {
                    'enum': ['linear', 'poly', 'rbf', 'sigmoid']}],
                'default': 'linear',
                'description': 'Kernel mapping used internally'},
            'gamma': {
                'anyOf': [{
                    'type': 'number'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Gamma parameter for the RBF, laplacian, polynomial, exponential chi2 and sigmoid kernels'},
            'degree': {
                'type': 'number',
                'minimumForOptimizer': 0.0,
                'maximumForOptimizer': 1.0,
                'distribution': 'uniform',
                'default': 3,
                'description': 'Degree of the polynomial kernel'},
            'coef0': {
                'type': 'number',
                'minimumForOptimizer': 0.0,
                'maximumForOptimizer': 1.0,
                'distribution': 'uniform',
                'default': 1,
                'description': 'Zero coefficient for polynomial and sigmoid kernels'},
            'kernel_params': {
                'XXX TODO XXX': 'mapping of string to any, optional',
                'description': 'Additional parameters (keyword arguments) for kernel function passed as callable object.',
                'enum': [None],
                'default': None},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit Kernel Ridge regression model',
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
            'description': 'Training data'},
        'y': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'type': 'number'},
            }, {
                'type': 'array',
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }}],
            'description': 'Target values'},
        'sample_weight': {
            'anyOf': [{
                'type': 'number'}, {
                'type': 'array',
                'items': {
                    'type': 'number'},
            }],
            'description': 'Individual weights for each sample, ignored if None is passed.'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict using the kernel ridge model',
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
            'description': 'Samples'},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Returns predicted values.',
    'anyOf': [{
        'type': 'array',
        'items': {
            'type': 'number'},
    }, {
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
        'op': ['estimator'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_predict': _input_predict_schema,
        'output_predict': _output_predict_schema},
}
if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
KernelRidge = lale.operators.make_operator(KernelRidgeImpl, _combined_schemas)

