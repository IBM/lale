
from sklearn.linear_model.passive_aggressive import PassiveAggressiveRegressor as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class PassiveAggressiveRegressorImpl():

    def __init__(self, C=1.0, fit_intercept=True, max_iter=None, tol=None, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, shuffle=True, verbose=0, loss='epsilon_insensitive', epsilon=0.1, random_state=None, warm_start=False, average=False, n_iter=None):
        self._hyperparams = {
            'C': C,
            'fit_intercept': fit_intercept,
            'max_iter': max_iter,
            'tol': tol,
            'early_stopping': early_stopping,
            'validation_fraction': validation_fraction,
            'n_iter_no_change': n_iter_no_change,
            'shuffle': shuffle,
            'verbose': verbose,
            'loss': loss,
            'epsilon': epsilon,
            'random_state': random_state,
            'warm_start': warm_start,
            'average': average,
            'n_iter': n_iter}

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
    'description': 'inherited docstring for PassiveAggressiveRegressor    Passive Aggressive Regressor',
    'allOf': [{
        'type': 'object',
        'required': ['C', 'fit_intercept', 'max_iter', 'tol', 'early_stopping', 'validation_fraction', 'n_iter_no_change', 'shuffle', 'verbose', 'loss', 'epsilon', 'random_state', 'warm_start', 'average', 'n_iter'],
        'relevantToOptimizer': ['fit_intercept', 'max_iter', 'tol', 'shuffle', 'loss', 'epsilon', 'n_iter'],
        'additionalProperties': False,
        'properties': {
            'C': {
                'type': 'number',
                'default': 1.0,
                'description': 'Maximum step size (regularization). Defaults to 1.0.'},
            'fit_intercept': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether the intercept should be estimated or not. If False, the'},
            'max_iter': {
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 10,
                    'maximumForOptimizer': 1000,
                    'distribution': 'uniform'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The maximum number of passes over the training data (aka epochs).'},
            'tol': {
                'anyOf': [{
                    'type': 'number',
                    'minimumForOptimizer': 1e-08,
                    'maximumForOptimizer': 0.01,
                    'distribution': 'loguniform'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The stopping criterion. If it is not None, the iterations will stop'},
            'early_stopping': {
                'type': 'boolean',
                'default': False,
                'description': 'Whether to use early stopping to terminate training when validation.'},
            'validation_fraction': {
                'type': 'number',
                'default': 0.1,
                'description': 'The proportion of training data to set aside as validation set for'},
            'n_iter_no_change': {
                'type': 'integer',
                'default': 5,
                'description': 'Number of iterations with no improvement to wait before early stopping.'},
            'shuffle': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether or not the training data should be shuffled after each epoch.'},
            'verbose': {
                'type': 'integer',
                'default': 0,
                'description': 'The verbosity level'},
            'loss': {
                'enum': ['huber', 'squared_epsilon_insensitive', 'squared_loss', 'epsilon_insensitive'],
                'default': 'epsilon_insensitive',
                'description': 'The loss function to be used:'},
            'epsilon': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 1.35,
                'distribution': 'loguniform',
                'default': 0.1,
                'description': 'If the difference between the current prediction and the correct label'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The seed of the pseudo random number generator to use when shuffling'},
            'warm_start': {
                'type': 'boolean',
                'default': False,
                'description': 'When set to True, reuse the solution of the previous call to fit as'},
            'average': {
                'anyOf': [{
                    'type': 'boolean'}, {
                    'type': 'integer'}],
                'default': False,
                'description': 'When set to True, computes the averaged SGD weights and stores the'},
            'n_iter': {
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 5,
                    'maximumForOptimizer': 1000,
                    'distribution': 'uniform'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The number of passes over the training data (aka epochs).'},
        }}, {
        'XXX TODO XXX': 'Parameter: max_iter > only impacts the behavior in the fit method'}, {
        'description': 'validation_fraction, only used if early_stopping is true',
        'anyOf': [{
            'type': 'object',
            'properties': {
                'validation_fraction': {
                    'enum': [0.1]},
            }}, {
            'type': 'object',
            'properties': {
                'early_stopping': {
                    'enum': [True]},
            }}]}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit linear model with Passive Aggressive algorithm.',
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
            'description': 'Target values'},
        'coef_init': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'The initial coefficients to warm-start the optimization.'},
        'intercept_init': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'The initial intercept to warm-start the optimization.'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict using the linear model',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            }},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predicted target values per element in X.',
    'XXX TODO XXX': '',
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
PassiveAggressiveRegressor = lale.operators.make_operator(PassiveAggressiveRegressorImpl, _combined_schemas)

