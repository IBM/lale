
from sklearn.linear_model.bayes import ARDRegression as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class ARDRegressionImpl():

    def __init__(self, n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06, compute_score=False, threshold_lambda=10000.0, fit_intercept=True, normalize=False, copy_X=True, verbose=False):
        self._hyperparams = {
            'n_iter': n_iter,
            'tol': tol,
            'alpha_1': alpha_1,
            'alpha_2': alpha_2,
            'lambda_1': lambda_1,
            'lambda_2': lambda_2,
            'compute_score': compute_score,
            'threshold_lambda': threshold_lambda,
            'fit_intercept': fit_intercept,
            'normalize': normalize,
            'copy_X': copy_X,
            'verbose': verbose}
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
    'description': 'inherited docstring for ARDRegression    Bayesian ARD regression.',
    'allOf': [{
        'type': 'object',
        'required': ['n_iter', 'tol', 'alpha_1', 'alpha_2', 'lambda_1', 'lambda_2', 'compute_score', 'threshold_lambda', 'fit_intercept', 'normalize', 'copy_X', 'verbose'],
        'relevantToOptimizer': ['n_iter', 'tol', 'compute_score', 'fit_intercept', 'normalize', 'copy_X'],
        'additionalProperties': False,
        'properties': {
            'n_iter': {
                'type': 'integer',
                'minimumForOptimizer': 5,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 300,
                'description': 'Maximum number of iterations'},
            'tol': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'distribution': 'loguniform',
                'default': 0.001,
                'description': 'Stop the algorithm if w has converged'},
            'alpha_1': {
                'type': 'number',
                'default': 1e-06,
                'description': 'Hyper-parameter : shape parameter for the Gamma distribution prior over the alpha parameter'},
            'alpha_2': {
                'type': 'number',
                'default': 1e-06,
                'description': 'Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the alpha parameter'},
            'lambda_1': {
                'type': 'number',
                'default': 1e-06,
                'description': 'Hyper-parameter : shape parameter for the Gamma distribution prior over the lambda parameter'},
            'lambda_2': {
                'type': 'number',
                'default': 1e-06,
                'description': 'Hyper-parameter : inverse scale parameter (rate parameter) for the Gamma distribution prior over the lambda parameter'},
            'compute_score': {
                'type': 'boolean',
                'default': False,
                'description': 'If True, compute the objective function at each step of the model'},
            'threshold_lambda': {
                'type': 'number',
                'default': 10000.0,
                'description': 'threshold for removing (pruning) weights with high precision from the computation'},
            'fit_intercept': {
                'type': 'boolean',
                'default': True,
                'description': 'whether to calculate the intercept for this model'},
            'normalize': {
                'type': 'boolean',
                'default': False,
                'description': 'This parameter is ignored when ``fit_intercept`` is set to False'},
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
    'description': 'Fit the ARDRegression model according to the given training data',
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
            'description': 'Training vector, where n_samples in the number of samples and n_features is the number of features.'},
        'y': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'Target values (integers)'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict using the linear model.',
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
ARDRegression = lale.operators.make_operator(ARDRegressionImpl, _combined_schemas)

