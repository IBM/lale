
from sklearn.linear_model.bayes import BayesianRidge as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class BayesianRidgeImpl():

    def __init__(self, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False):
        self._hyperparams = {
            'n_iter': n_iter,
            'tol': tol,
            'alpha_1': alpha_1,
            'alpha_2': alpha_2,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'compute_score': compute_score,
            'fit_intercept': fit_intercept,
            'normalize': normalize,
            'copy_X': copy_X,
            'verbose': verbose}

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
    'description': 'inherited docstring for BayesianRidge    Bayesian ridge regression',
    'allOf': [{
        'type': 'object',
        'relevantToOptimizer': ['n_iter', 'tol', 'compute_score', 'fit_intercept', 'normalize', 'copy_X'],
        'additionalProperties': False,
        'properties': {
            'n_iter': {
                'type': 'integer',
                'minimumForOptimizer': 5,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 300,
                'description': 'Maximum number of iterations.  Default is 300.'},
            'tol': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'distribution': 'loguniform',
                'default': 0.001,
                'description': 'Stop the algorithm if w has converged. Default is 1.e-3.'},
            'alpha_1': {
                'type': 'number',
                'default': 1e-06,
                'description': 'Hyper-parameter : shape parameter for the Gamma distribution prior'},
            'alpha_2': {
                'type': 'number',
                'default': 1e-06,
                'description': 'Hyper-parameter : inverse scale parameter (rate parameter) for the'},
            'lambda_1': {
                'type': 'number',
                'default': 1e-06,
                'description': 'Hyper-parameter : shape parameter for the Gamma distribution prior'},
            'lambda_2': {
                'type': 'number',
                'default': 1e-06,
                'description': 'Hyper-parameter : inverse scale parameter (rate parameter) for the'},
            'compute_score': {
                'type': 'boolean',
                'default': False,
                'description': 'If True, compute the objective function at each step of the model.'},
            'fit_intercept': {
                'type': 'boolean',
                'default': True,
                'description': 'whether to calculate the intercept for this model. If set'},
            'normalize': {
                'type': 'boolean',
                'default': False,
                'description': 'This parameter is ignored when ``fit_intercept`` is set to False.'},
            'copy_X': {
                'type': 'boolean',
                'default': True,
                'description': 'If True, X will be copied; else, it may be overwritten.'},
            'verbose': {
                'type': 'boolean',
                'default': False,
                'description': 'Verbose mode when fitting the model.'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the model',
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
            'description': "Target values. Will be cast to X's dtype if necessary"},
        'sample_weight': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'Individual weights for each sample'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict using the linear model.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'Samples.'},
        'return_std': {
            'anyOf': [{
                'type': 'boolean'}, {
                'enum': [None]}],
            'default': None,
            'description': 'Whether to return the standard deviation of posterior prediction.'},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict using the linear model.',
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
BayesianRidge = lale.operators.make_operator(BayesianRidgeImpl, _combined_schemas)

