
from sklearn.linear_model.least_angle import LarsCV as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class LarsCVImpl():

    def __init__(self, fit_intercept=True, verbose=False, max_iter=500, normalize=True, precompute='auto', cv=3, max_n_alphas=1000, n_jobs=None, eps=2.220446049250313e-16, copy_X=True, positive=False):
        self._hyperparams = {
            'fit_intercept': fit_intercept,
            'verbose': verbose,
            'max_iter': max_iter,
            'normalize': normalize,
            'precompute': precompute,
            'cv': cv,
            'max_n_alphas': max_n_alphas,
            'n_jobs': n_jobs,
            'eps': eps,
            'copy_X': copy_X,
            'positive': positive}

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
    'description': 'inherited docstring for LarsCV    Cross-validated Least Angle Regression model.',
    'allOf': [{
        'type': 'object',
        'relevantToOptimizer': ['fit_intercept', 'max_iter', 'normalize', 'precompute', 'cv', 'max_n_alphas', 'eps', 'copy_X', 'positive'],
        'additionalProperties': False,
        'properties': {
            'fit_intercept': {
                'type': 'boolean',
                'default': True,
                'description': 'whether to calculate the intercept for this model. If set'},
            'verbose': {
                'anyOf': [{
                    'type': 'boolean'}, {
                    'type': 'integer'}],
                'default': False,
                'description': 'Sets the verbosity amount'},
            'max_iter': {
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 500,
                'description': 'Maximum number of iterations to perform.'},
            'normalize': {
                'type': 'boolean',
                'default': True,
                'description': 'This parameter is ignored when ``fit_intercept`` is set to False.'},
            'precompute': {
                'anyOf': [{
                    'type': 'array',
                    'items': {
                        'XXX TODO XXX': 'item type'},
                    'XXX TODO XXX': "True | False | 'auto' | array-like",
                    'forOptimizer': False}, {
                    'enum': ['auto']}],
                'default': 'auto',
                'description': 'Whether to use a precomputed Gram matrix to speed up'},
            'cv': {
                'XXX TODO XXX': 'int, cross-validation generator or an iterable, optional',
                'description': 'Determines the cross-validation splitting strategy.',
                'type': 'integer',
                'minimumForOptimizer': 3,
                'maximumForOptimizer': 4,
                'distribution': 'uniform',
                'default': 3},
            'max_n_alphas': {
                'type': 'integer',
                'minimumForOptimizer': 1000,
                'maximumForOptimizer': 1001,
                'distribution': 'uniform',
                'default': 1000,
                'description': 'The maximum number of points on the path used to compute the'},
            'n_jobs': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Number of CPUs to use during the cross validation.'},
            'eps': {
                'type': 'number',
                'minimumForOptimizer': 0.001,
                'maximumForOptimizer': 0.1,
                'distribution': 'uniform',
                'default': 2.220446049250313e-16,
                'description': 'The machine-precision regularization in the computation of the'},
            'copy_X': {
                'type': 'boolean',
                'default': True,
                'description': 'If ``True``, X will be copied; else, it may be overwritten.'},
            'positive': {
                'type': 'boolean',
                'default': False,
                'description': 'Restrict coefficients to be >= 0. Be aware that you might want to'},
        }}, {
        'description': 'precompute, XXX TODO XXX, only subsets of x'}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the model using X, y as training data.',
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
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'Target values.'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict using the linear model',
    'type': 'object',
    'properties': {
        'X': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'XXX TODO XXX': 'item type'},
                'XXX TODO XXX': 'array_like or sparse matrix, shape (n_samples, n_features)'}, {
                'type': 'array',
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }}],
            'description': 'Samples.'},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Returns predicted values.',
    'type': 'array',
    'items': {
        'type': 'number'},
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
LarsCV = lale.operators.make_operator(LarsCVImpl, _combined_schemas)

