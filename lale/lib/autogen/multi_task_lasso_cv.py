
from sklearn.linear_model.coordinate_descent import MultiTaskLassoCV as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class MultiTaskLassoCVImpl():

    def __init__(self, eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=False, max_iter=1000, tol=0.0001, copy_X=True, cv=3, verbose=False, n_jobs=None, random_state=None, selection='cyclic'):
        self._hyperparams = {
            'eps': eps,
            'n_alphas': n_alphas,
            'alphas': alphas,
            'fit_intercept': fit_intercept,
            'normalize': normalize,
            'max_iter': max_iter,
            'tol': tol,
            'copy_X': copy_X,
            'cv': cv,
            'verbose': verbose,
            'n_jobs': n_jobs,
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
    'description': 'inherited docstring for MultiTaskLassoCV    Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.',
    'allOf': [{
        'type': 'object',
        'required': ['eps', 'n_alphas', 'alphas', 'fit_intercept', 'normalize', 'max_iter', 'tol', 'copy_X', 'cv', 'verbose', 'n_jobs', 'random_state', 'selection'],
        'relevantToOptimizer': ['eps', 'n_alphas', 'fit_intercept', 'normalize', 'max_iter', 'tol', 'copy_X', 'cv'],
        'additionalProperties': False,
        'properties': {
            'eps': {
                'type': 'number',
                'minimumForOptimizer': 0.001,
                'maximumForOptimizer': 0.1,
                'distribution': 'loguniform',
                'default': 0.001,
                'description': 'Length of the path. ``eps=1e-3`` means that'},
            'n_alphas': {
                'type': 'integer',
                'minimumForOptimizer': 100,
                'maximumForOptimizer': 101,
                'distribution': 'uniform',
                'default': 100,
                'description': 'Number of alphas along the regularization path'},
            'alphas': {
                'anyOf': [{
                    'type': 'array',
                    'items': {
                        'XXX TODO XXX': 'item type'},
                    'XXX TODO XXX': 'array-like, optional'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'List of alphas where to compute the models.'},
            'fit_intercept': {
                'type': 'boolean',
                'default': True,
                'description': 'whether to calculate the intercept for this model. If set'},
            'normalize': {
                'type': 'boolean',
                'default': False,
                'description': 'This parameter is ignored when ``fit_intercept`` is set to False.'},
            'max_iter': {
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 1000,
                'description': 'The maximum number of iterations.'},
            'tol': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'distribution': 'loguniform',
                'default': 0.0001,
                'description': 'The tolerance for the optimization: if the updates are'},
            'copy_X': {
                'type': 'boolean',
                'default': True,
                'description': 'If ``True``, X will be copied; else, it may be overwritten.'},
            'cv': {
                'XXX TODO XXX': 'int, cross-validation generator or an iterable, optional',
                'description': 'Determines the cross-validation splitting strategy.',
                'type': 'integer',
                'minimumForOptimizer': 3,
                'maximumForOptimizer': 4,
                'distribution': 'uniform',
                'default': 3},
            'verbose': {
                'anyOf': [{
                    'type': 'boolean'}, {
                    'type': 'integer'}],
                'default': False,
                'description': 'Amount of verbosity.'},
            'n_jobs': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Number of CPUs to use during the cross validation. Note that this is'},
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
        }}, {
        'XXX TODO XXX': 'Parameter: n_jobs > only if multiple values for l1_ratio are given'}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit linear model with coordinate descent',
    'type': 'object',
    'properties': {
        'X': {
            'XXX TODO XXX': '{array-like}, shape (n_samples, n_features)',
            'description': 'Training data. Pass directly as Fortran-contiguous data'},
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
            'description': 'Target values'},
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
MultiTaskLassoCV = lale.operators.make_operator(MultiTaskLassoCVImpl, _combined_schemas)

