
from sklearn.linear_model.omp import OrthogonalMatchingPursuit as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class OrthogonalMatchingPursuitImpl():

    def __init__(self, n_nonzero_coefs=None, tol=None, fit_intercept=True, normalize=True, precompute='auto'):
        self._hyperparams = {
            'n_nonzero_coefs': n_nonzero_coefs,
            'tol': tol,
            'fit_intercept': fit_intercept,
            'normalize': normalize,
            'precompute': precompute}

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
    'description': 'inherited docstring for OrthogonalMatchingPursuit    Orthogonal Matching Pursuit model (OMP)',
    'allOf': [{
        'type': 'object',
        'relevantToOptimizer': ['n_nonzero_coefs', 'tol', 'fit_intercept', 'normalize', 'precompute'],
        'additionalProperties': False,
        'properties': {
            'n_nonzero_coefs': {
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 500,
                    'maximumForOptimizer': 501,
                    'distribution': 'uniform'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Desired number of non-zero entries in the solution. If None (by'},
            'tol': {
                'anyOf': [{
                    'type': 'number',
                    'minimumForOptimizer': 1e-08,
                    'maximumForOptimizer': 0.01,
                    'distribution': 'loguniform'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Maximum norm of the residual. If not None, overrides n_nonzero_coefs.'},
            'fit_intercept': {
                'type': 'boolean',
                'default': True,
                'description': 'whether to calculate the intercept for this model. If set'},
            'normalize': {
                'type': 'boolean',
                'default': True,
                'description': 'This parameter is ignored when ``fit_intercept`` is set to False.'},
            'precompute': {
                'enum': [True, False, 'auto'],
                'default': 'auto',
                'description': 'Whether to use a precomputed Gram and Xy matrix to speed up'},
        }}],
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
            'description': "Target values. Will be cast to X's dtype if necessary"},
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
OrthogonalMatchingPursuit = lale.operators.make_operator(OrthogonalMatchingPursuitImpl, _combined_schemas)

