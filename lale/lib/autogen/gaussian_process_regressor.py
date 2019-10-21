
from sklearn.gaussian_process.gpr import GaussianProcessRegressor as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class GaussianProcessRegressorImpl():

    def __init__(self, kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, random_state=None):
        self._hyperparams = {
            'kernel': kernel,
            'alpha': alpha,
            'optimizer': optimizer,
            'n_restarts_optimizer': n_restarts_optimizer,
            'normalize_y': normalize_y,
            'copy_X_train': copy_X_train,
            'random_state': random_state}

    def fit(self, X, y=None):
        self._sklearn_model = SKLModel(**self._hyperparams)
        if (y is not None):
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X)
        return self

    def predict(self, X):
        return self._sklearn_model.predict(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for GaussianProcessRegressor    Gaussian process regression (GPR).',
    'allOf': [{
        'type': 'object',
        'required': ['kernel', 'alpha', 'optimizer', 'n_restarts_optimizer', 'normalize_y', 'copy_X_train', 'random_state'],
        'relevantToOptimizer': ['alpha', 'optimizer', 'n_restarts_optimizer', 'normalize_y'],
        'additionalProperties': False,
        'properties': {
            'kernel': {
                'XXX TODO XXX': 'kernel object',
                'description': 'The kernel specifying the covariance function of the GP. If None is',
                'enum': [None],
                'default': None},
            'alpha': {
                'anyOf': [{
                    'type': 'number',
                    'minimumForOptimizer': 1e-10,
                    'maximumForOptimizer': 1.0,
                    'distribution': 'loguniform'}, {
                    'type': 'array',
                    'items': {
                        'XXX TODO XXX': 'item type'},
                    'XXX TODO XXX': 'float or array-like, optional (default: 1e-10)',
                    'forOptimizer': False}],
                'default': 1e-10,
                'description': 'Value added to the diagonal of the kernel matrix during fitting.'},
            'optimizer': {
                'anyOf': [{
                    'type': 'object',
                    'forOptimizer': False}, {
                    'enum': ['fmin_l_bfgs_b']}],
                'default': 'fmin_l_bfgs_b',
                'description': 'Can either be one of the internally supported optimizers for optimizing'},
            'n_restarts_optimizer': {
                'type': 'integer',
                'minimumForOptimizer': 0,
                'maximumForOptimizer': 1,
                'distribution': 'uniform',
                'default': 0,
                'description': "The number of restarts of the optimizer for finding the kernel's"},
            'normalize_y': {
                'type': 'boolean',
                'default': False,
                'description': 'Whether the target values y are normalized, i.e., the mean of the'},
            'copy_X_train': {
                'type': 'boolean',
                'default': True,
                'description': 'If True, a persistent copy of the training data is stored in the'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The generator used to initialize the centers. If int, random_state is'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit Gaussian process regression model.',
    'type': 'object',
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
            'XXX TODO XXX': 'array-like, shape = (n_samples, [n_output_dims])',
            'description': 'Target values'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict using the Gaussian process regression model',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'Query points where the GP is evaluated'},
        'return_std': {
            'type': 'boolean',
            'default': False,
            'description': 'If True, the standard-deviation of the predictive distribution at'},
        'return_cov': {
            'type': 'boolean',
            'default': False,
            'description': 'If True, the covariance of the joint predictive distribution at'},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict using the Gaussian process regression model',
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
GaussianProcessRegressor = lale.operators.make_operator(GaussianProcessRegressorImpl, _combined_schemas)

