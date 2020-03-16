
from sklearn.linear_model.least_angle import Lars as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class LarsImpl():

    def __init__(self, fit_intercept=True, verbose=False, normalize=True, precompute='auto', n_nonzero_coefs=500, eps=2.220446049250313e-16, copy_X=True, fit_path=True, positive=False):
        self._hyperparams = {
            'fit_intercept': fit_intercept,
            'verbose': verbose,
            'normalize': normalize,
            'precompute': precompute,
            'n_nonzero_coefs': n_nonzero_coefs,
            'eps': eps,
            'copy_X': copy_X,
            'fit_path': fit_path,
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
    'description': 'inherited docstring for Lars    Least Angle Regression model a.k.a. LAR',
    'allOf': [{
        'type': 'object',
        'required': ['fit_intercept', 'verbose', 'normalize', 'precompute', 'n_nonzero_coefs', 'eps', 'copy_X', 'fit_path', 'positive'],
        'relevantToOptimizer': ['fit_intercept', 'normalize', 'precompute', 'n_nonzero_coefs', 'eps', 'copy_X', 'positive'],
        'additionalProperties': False,
        'properties': {
            'fit_intercept': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether to calculate the intercept for this model'},
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
            'n_nonzero_coefs': {
                'type': 'integer',
                'minimumForOptimizer': 500,
                'maximumForOptimizer': 501,
                'distribution': 'uniform',
                'default': 500,
                'description': 'Target number of non-zero coefficients'},
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
                'description': 'If ``True``, X will be copied; else, it may be overwritten.'},
            'fit_path': {
                'type': 'boolean',
                'default': True,
                'description': 'If True the full path is stored in the ``coef_path_`` attribute'},
            'positive': {
                'type': 'boolean',
                'default': False,
                'description': 'Restrict coefficients to be >= 0'},
        }}],
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
            'laleType': 'Any',
            'XXX TODO XXX': 'array-like, shape (n_samples,) or (n_samples, n_targets),                 optional',
            'description': 'Xy = np.dot(X.T, y) that can be precomputed'},
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
Lars = lale.operators.make_operator(LarsImpl, _combined_schemas)

