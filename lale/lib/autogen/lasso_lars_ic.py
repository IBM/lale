
from sklearn.linear_model.least_angle import LassoLarsIC as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class LassoLarsICImpl():

    def __init__(self, criterion='aic', fit_intercept=True, verbose=False, normalize=True, precompute='auto', max_iter=500, eps=2.220446049250313e-16, copy_X=True, positive=False):
        self._hyperparams = {
            'criterion': criterion,
            'fit_intercept': fit_intercept,
            'verbose': verbose,
            'normalize': normalize,
            'precompute': precompute,
            'max_iter': max_iter,
            'eps': eps,
            'copy_X': copy_X,
            'positive': positive}
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
    'description': 'inherited docstring for LassoLarsIC    Lasso model fit with Lars using BIC or AIC for model selection',
    'allOf': [{
        'type': 'object',
        'required': ['criterion', 'fit_intercept', 'verbose', 'normalize', 'precompute', 'max_iter', 'eps', 'copy_X', 'positive'],
        'relevantToOptimizer': ['criterion', 'fit_intercept', 'normalize', 'precompute', 'max_iter', 'eps', 'copy_X', 'positive'],
        'additionalProperties': False,
        'properties': {
            'criterion': {
                'enum': ['bic', 'aic'],
                'default': 'aic',
                'description': 'The type of criterion to use.'},
            'fit_intercept': {
                'type': 'boolean',
                'default': True,
                'description': 'whether to calculate the intercept for this model'},
            'verbose': {
                'anyOf': [{
                    'type': 'boolean'}, {
                    'type': 'integer'}],
                'default': False,
                'description': 'Sets the verbosity amount'},
            'normalize': {
                'type': 'boolean',
                'default': True,
                'description': 'This parameter is ignored when ``fit_intercept`` is set to False'},
            'precompute': {
                'anyOf': [{
                    'type': 'array',
                    'items': {
                        'laleType': 'Any',
                        'XXX TODO XXX': 'item type'},
                    'XXX TODO XXX': "True | False | 'auto' | array-like",
                    'forOptimizer': False}, {
                    'enum': ['auto']}],
                'default': 'auto',
                'description': 'Whether to use a precomputed Gram matrix to speed up calculations'},
            'max_iter': {
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 500,
                'description': 'Maximum number of iterations to perform'},
            'eps': {
                'type': 'number',
                'minimumForOptimizer': 0.001,
                'maximumForOptimizer': 0.1,
                'distribution': 'uniform',
                'default': 2.220446049250313e-16,
                'description': 'The machine-precision regularization in the computation of the Cholesky diagonal factors'},
            'copy_X': {
                'type': 'boolean',
                'default': True,
                'description': 'If True, X will be copied; else, it may be overwritten.'},
            'positive': {
                'type': 'boolean',
                'default': False,
                'description': 'Restrict coefficients to be >= 0'},
        }}, {
        'XXX TODO XXX': 'Parameter: positive > only coefficients up to the smallest alpha value (alphas_[alphas_ > 0'}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the model using X, y as training data.',
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
            'description': 'training data.'},
        'y': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'target values'},
        'copy_X': {
            'type': 'boolean',
            'default': True,
            'description': 'If ``True``, X will be copied; else, it may be overwritten.'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict using the linear model',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'laleType': 'Any',
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
LassoLarsIC = lale.operators.make_operator(LassoLarsICImpl, _combined_schemas)

