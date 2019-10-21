
from sklearn.gaussian_process.gpc import GaussianProcessClassifier as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class GaussianProcessClassifierImpl():

    def __init__(self, kernel=None, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0, max_iter_predict=100, warm_start=False, copy_X_train=True, random_state=None, multi_class='one_vs_rest', n_jobs=None):
        self._hyperparams = {
            'kernel': kernel,
            'optimizer': optimizer,
            'n_restarts_optimizer': n_restarts_optimizer,
            'max_iter_predict': max_iter_predict,
            'warm_start': warm_start,
            'copy_X_train': copy_X_train,
            'random_state': random_state,
            'multi_class': multi_class,
            'n_jobs': n_jobs}

    def fit(self, X, y=None):
        self._sklearn_model = SKLModel(**self._hyperparams)
        if (y is not None):
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X)
        return self

    def predict(self, X):
        return self._sklearn_model.predict(X)

    def predict_proba(self, X):
        return self._sklearn_model.predict_proba(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for GaussianProcessClassifier    Gaussian process classification (GPC) based on Laplace approximation.',
    'allOf': [{
        'type': 'object',
        'required': ['kernel', 'optimizer', 'n_restarts_optimizer', 'max_iter_predict', 'warm_start', 'copy_X_train', 'random_state', 'multi_class', 'n_jobs'],
        'relevantToOptimizer': ['optimizer', 'n_restarts_optimizer', 'max_iter_predict', 'multi_class'],
        'additionalProperties': False,
        'properties': {
            'kernel': {
                'XXX TODO XXX': 'kernel object',
                'description': 'The kernel specifying the covariance function of the GP. If None is',
                'enum': [None],
                'default': None},
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
            'max_iter_predict': {
                'type': 'integer',
                'minimumForOptimizer': 100,
                'maximumForOptimizer': 101,
                'distribution': 'uniform',
                'default': 100,
                'description': "The maximum number of iterations in Newton's method for approximating"},
            'warm_start': {
                'type': 'boolean',
                'default': False,
                'description': 'If warm-starts are enabled, the solution of the last Newton iteration'},
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
                'description': 'The generator used to initialize the centers.'},
            'multi_class': {
                'XXX TODO XXX': 'string, default',
                'description': 'Specifies how multi-class classification problems are handled.',
                'enum': ['auto', 'liblinear', 'one_vs_one', 'one_vs_rest'],
                'default': 'one_vs_rest'},
            'n_jobs': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The number of jobs to use for the computation.'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit Gaussian process classification model',
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
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'Target values, must be binary'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Perform classification on an array of test vectors X.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            }},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predicted target values for X, values are from ``classes_``',
    'type': 'array',
    'items': {
        'type': 'number'},
}
_input_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Return probability estimates for the test vector X.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            }},
    },
}
_output_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Returns the probability of the samples for each class in',
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
        'op': ['estimator'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_predict': _input_predict_schema,
        'output_predict': _output_predict_schema,
        'input_predict_proba': _input_predict_proba_schema,
        'output_predict_proba': _output_predict_proba_schema},
}
if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
GaussianProcessClassifier = lale.operators.make_operator(GaussianProcessClassifierImpl, _combined_schemas)

