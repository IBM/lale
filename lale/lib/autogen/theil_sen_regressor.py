
from sklearn.linear_model.theil_sen import TheilSenRegressor as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class TheilSenRegressorImpl():

    def __init__(self, fit_intercept=True, copy_X=True, max_subpopulation=10000, n_subsamples=None, max_iter=300, tol=0.001, random_state=None, n_jobs=None, verbose=False):
        self._hyperparams = {
            'fit_intercept': fit_intercept,
            'copy_X': copy_X,
            'max_subpopulation': max_subpopulation,
            'n_subsamples': n_subsamples,
            'max_iter': max_iter,
            'tol': tol,
            'random_state': random_state,
            'n_jobs': n_jobs,
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
    'description': 'inherited docstring for TheilSenRegressor    Theil-Sen Estimator: robust multivariate regression model.',
    'allOf': [{
        'type': 'object',
        'required': ['fit_intercept', 'copy_X', 'max_subpopulation', 'n_subsamples', 'max_iter', 'tol', 'random_state', 'n_jobs', 'verbose'],
        'relevantToOptimizer': ['fit_intercept', 'copy_X', 'max_subpopulation', 'max_iter', 'tol'],
        'additionalProperties': False,
        'properties': {
            'fit_intercept': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether to calculate the intercept for this model'},
            'copy_X': {
                'type': 'boolean',
                'default': True,
                'description': 'If True, X will be copied; else, it may be overwritten.'},
            'max_subpopulation': {
                'type': 'integer',
                'minimumForOptimizer': 10000,
                'maximumForOptimizer': 10001,
                'distribution': 'uniform',
                'default': 10000,
                'description': "Instead of computing with a set of cardinality 'n choose k', where n is the number of samples and k is the number of subsamples (at least number of features), consider only a stochastic subpopulation of a given maximal size if 'n choose k' is larger than max_subpopulation"},
            'n_subsamples': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Number of samples to calculate the parameters'},
            'max_iter': {
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 300,
                'description': 'Maximum number of iterations for the calculation of spatial median.'},
            'tol': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'distribution': 'loguniform',
                'default': 0.001,
                'description': 'Tolerance when calculating spatial median.'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'A random number generator instance to define the state of the random permutations generator'},
            'n_jobs': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Number of CPUs to use during the cross validation'},
            'verbose': {
                'type': 'boolean',
                'default': False,
                'description': 'Verbose mode when fitting the model.'},
        }}, {
        'XXX TODO XXX': "Parameter: max_subpopulation > only a stochastic subpopulation of a given maximal size if 'n choose k' is larger than max_subpopulation"}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit linear model.',
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
            'description': 'Training data'},
        'y': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'Target values'},
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
TheilSenRegressor = lale.operators.make_operator(TheilSenRegressorImpl, _combined_schemas)

