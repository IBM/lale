
from sklearn.linear_model.least_angle import LassoLars as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class LassoLarsImpl():

    def __init__(self, alpha=1.0, fit_intercept=True, verbose=False, normalize=True, precompute='auto', max_iter=500, eps=2.220446049250313e-16, copy_X=True, fit_path=True, positive=False):
        self._hyperparams = {
            'alpha': alpha,
            'fit_intercept': fit_intercept,
            'verbose': verbose,
            'normalize': normalize,
            'precompute': precompute,
            'max_iter': max_iter,
            'eps': eps,
            'copy_X': copy_X,
            'fit_path': fit_path,
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
    'description': 'inherited docstring for LassoLars    Lasso model fit with Least Angle Regression a.k.a. Lars',
    'allOf': [{
        'type': 'object',
        'required': ['alpha', 'fit_intercept', 'verbose', 'normalize', 'precompute', 'max_iter', 'eps', 'copy_X', 'fit_path', 'positive'],
        'relevantToOptimizer': ['alpha', 'fit_intercept', 'normalize', 'precompute', 'max_iter', 'eps', 'copy_X', 'positive'],
        'additionalProperties': False,
        'properties': {
            'alpha': {
                'type': 'number',
                'minimumForOptimizer': 1e-10,
                'maximumForOptimizer': 1.0,
                'distribution': 'loguniform',
                'default': 1.0,
                'description': 'Constant that multiplies the penalty term. Defaults to 1.0.'},
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
            'max_iter': {
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 500,
                'description': 'Maximum number of iterations to perform.'},
            'eps': {
                'type': 'number',
                'minimumForOptimizer': 0.001,
                'maximumForOptimizer': 0.1,
                'distribution': 'loguniform',
                'default': 2.220446049250313e-16,
                'description': 'The machine-precision regularization in the computation of the'},
            'copy_X': {
                'type': 'boolean',
                'default': True,
                'description': 'If True, X will be copied; else, it may be overwritten.'},
            'fit_path': {
                'type': 'boolean',
                'default': True,
                'description': 'If ``True`` the full path is stored in the ``coef_path_`` attribute.'},
            'positive': {
                'type': 'boolean',
                'default': False,
                'description': 'Restrict coefficients to be >= 0. Be aware that you might want to'},
        }}, {
        'XXX TODO XXX': 'Parameter: positive > only coefficients up to the smallest alpha value (alphas_[alphas_ > 0'}],
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
            'description': 'Target values.'},
        'Xy': {
            'XXX TODO XXX': 'array-like, shape (n_samples,) or (n_samples, n_targets),                 optional',
            'description': 'Xy = np.dot(X.T, y) that can be precomputed. It is useful'},
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
LassoLars = lale.operators.make_operator(LassoLarsImpl, _combined_schemas)

