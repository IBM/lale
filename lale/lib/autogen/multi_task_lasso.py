
from sklearn.linear_model.coordinate_descent import MultiTaskLasso as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class MultiTaskLassoImpl():

    def __init__(self, alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, random_state=None, selection='cyclic'):
        self._hyperparams = {
            'alpha': alpha,
            'fit_intercept': fit_intercept,
            'normalize': normalize,
            'copy_X': copy_X,
            'max_iter': max_iter,
            'tol': tol,
            'warm_start': warm_start,
            'random_state': random_state,
            'selection': selection}

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
    'description': 'inherited docstring for MultiTaskLasso    Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.',
    'allOf': [{
        'type': 'object',
        'relevantToOptimizer': ['alpha', 'fit_intercept', 'normalize', 'copy_X', 'max_iter', 'tol'],
        'additionalProperties': False,
        'properties': {
            'alpha': {
                'type': 'number',
                'minimumForOptimizer': 1e-10,
                'maximumForOptimizer': 1.0,
                'distribution': 'loguniform',
                'default': 1.0,
                'description': 'Constant that multiplies the L1/L2 term. Defaults to 1.0'},
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
                'description': 'If ``True``, X will be copied; else, it may be overwritten.'},
            'max_iter': {
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 1000,
                'description': 'The maximum number of iterations'},
            'tol': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'distribution': 'loguniform',
                'default': 0.0001,
                'description': 'The tolerance for the optimization: if the updates are'},
            'warm_start': {
                'type': 'boolean',
                'default': False,
                'description': 'When set to ``True``, reuse the solution of the previous call to fit as'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The seed of the pseudo random number generator that selects a random'},
            'selection': {
                'type': 'string',
                'default': 'cyclic',
                'description': "If set to 'random', a random coefficient is updated every iteration"},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit MultiTaskElasticNet model with coordinate descent',
    'type': 'object',
    'properties': {
        'X': {
            'XXX TODO XXX': 'ndarray, shape (n_samples, n_features)',
            'description': 'Data'},
        'y': {
            'XXX TODO XXX': 'ndarray, shape (n_samples, n_tasks)',
            'description': "Target. Will be cast to X's dtype if necessary"},
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
MultiTaskLasso = lale.operators.make_operator(MultiTaskLassoImpl, _combined_schemas)

