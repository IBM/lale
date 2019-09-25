
from sklearn.neural_network.multilayer_perceptron import MLPRegressor as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class MLPRegressorImpl():

    def __init__(self, hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, n_iter_no_change=10):
        self._hyperparams = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'activation': activation,
            'solver': solver,
            'alpha': alpha,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'learning_rate_init': learning_rate_init,
            'power_t': power_t,
            'max_iter': max_iter,
            'shuffle': shuffle,
            'random_state': random_state,
            'tol': tol,
            'verbose': verbose,
            'warm_start': warm_start,
            'momentum': momentum,
            'nesterovs_momentum': nesterovs_momentum,
            'early_stopping': early_stopping,
            'validation_fraction': validation_fraction,
            'beta_1': beta_1,
            'beta_2': beta_2,
            'epsilon': epsilon,
            'n_iter_no_change': n_iter_no_change}

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
    'description': 'inherited docstring for MLPRegressor    Multi-layer Perceptron regressor.',
    'allOf': [{
        'type': 'object',
        'relevantToOptimizer': ['activation', 'solver', 'alpha', 'batch_size', 'learning_rate', 'max_iter', 'shuffle', 'tol', 'nesterovs_momentum', 'epsilon'],
        'additionalProperties': False,
        'properties': {
            'hidden_layer_sizes': {
                'XXX TODO XXX': 'tuple, length = n_layers - 2, default (100,)',
                'description': 'The ith element represents the number of neurons in the ith',
                'type': 'array',
                'typeForOptimizer': 'tuple',
                'default': (100,)},
            'activation': {
                'enum': ['identity', 'logistic', 'tanh', 'relu'],
                'default': 'relu',
                'description': 'Activation function for the hidden layer.'},
            'solver': {
                'enum': ['lbfgs', 'sgd', 'adam'],
                'default': 'adam',
                'description': 'The solver for weight optimization.'},
            'alpha': {
                'type': 'number',
                'minimumForOptimizer': 1e-10,
                'maximumForOptimizer': 1.0,
                'distribution': 'loguniform',
                'default': 0.0001,
                'description': 'L2 penalty (regularization term) parameter.'},
            'batch_size': {
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 3,
                    'maximumForOptimizer': 128,
                    'distribution': 'uniform'}, {
                    'enum': ['auto']}],
                'default': 'auto',
                'description': 'Size of minibatches for stochastic optimizers.'},
            'learning_rate': {
                'enum': ['constant', 'invscaling', 'adaptive'],
                'default': 'constant',
                'description': 'Learning rate schedule for weight updates.'},
            'learning_rate_init': {
                'type': 'number',
                'default': 0.001,
                'description': 'The initial learning rate used. It controls the step-size'},
            'power_t': {
                'type': 'number',
                'default': 0.5,
                'description': 'The exponent for inverse scaling learning rate.'},
            'max_iter': {
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 200,
                'description': 'Maximum number of iterations. The solver iterates until convergence'},
            'shuffle': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether to shuffle samples in each iteration. Only used when'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'If int, random_state is the seed used by the random number generator;'},
            'tol': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'distribution': 'loguniform',
                'default': 0.0001,
                'description': 'Tolerance for the optimization. When the loss or score is not improving'},
            'verbose': {
                'type': 'boolean',
                'default': False,
                'description': 'Whether to print progress messages to stdout.'},
            'warm_start': {
                'type': 'boolean',
                'default': False,
                'description': 'When set to True, reuse the solution of the previous'},
            'momentum': {
                'type': 'number',
                'default': 0.9,
                'description': 'Momentum for gradient descent update.  Should be between 0 and 1. Only'},
            'nesterovs_momentum': {
                'type': 'boolean',
                'default': True,
                'description': "Whether to use Nesterov's momentum. Only used when solver='sgd' and"},
            'early_stopping': {
                'type': 'boolean',
                'default': False,
                'description': 'Whether to use early stopping to terminate training when validation'},
            'validation_fraction': {
                'type': 'number',
                'default': 0.1,
                'description': 'The proportion of training data to set aside as validation set for'},
            'beta_1': {
                'type': 'number',
                'default': 0.9,
                'description': 'Exponential decay rate for estimates of first moment vector in adam,'},
            'beta_2': {
                'type': 'number',
                'default': 0.999,
                'description': 'Exponential decay rate for estimates of second moment vector in adam,'},
            'epsilon': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 1.35,
                'distribution': 'loguniform',
                'default': 1e-08,
                'description': "Value for numerical stability in adam. Only used when solver='adam'"},
            'n_iter_no_change': {
                'type': 'integer',
                'default': 10,
                'description': 'Maximum number of epochs to not meet ``tol`` improvement.'},
        }}, {
        'description': "learning_rate, only used when solver='sgd'",
        'anyOf': [{
            'type': 'object',
            'properties': {
                'learning_rate': {
                    'enum': ['constant']},
            }}, {
            'type': 'object',
            'properties': {
                'solver': {
                    'enum': ['sgd']},
            }}]}, {
        'description': "learning_rate_init, only used when solver='sgd' or 'adam'",
        'anyOf': [{
            'type': 'object',
            'properties': {
                'learning_rate_init': {
                    'enum': [0.001]},
            }}, {
            'type': 'object',
            'properties': {
                'solver': {
                    'enum': ['sgd', 'adam']},
            }}]}, {
        'description': "power_t, only used when solver='sgd'",
        'anyOf': [{
            'type': 'object',
            'properties': {
                'power_t': {
                    'enum': [0.5]},
            }}, {
            'type': 'object',
            'properties': {
                'solver': {
                    'enum': ['sgd']},
            }}]}, {
        'description': "shuffle, only used when solver='sgd' or 'adam'",
        'anyOf': [{
            'type': 'object',
            'properties': {
                'shuffle': {
                    'enum': [True]},
            }}, {
            'type': 'object',
            'properties': {
                'solver': {
                    'enum': ['sgd', 'adam']},
            }}]}, {
        'description': "momentum, only used when solver='sgd'",
        'anyOf': [{
            'type': 'object',
            'properties': {
                'momentum': {
                    'enum': [0.9]},
            }}, {
            'type': 'object',
            'properties': {
                'solver': {
                    'enum': ['sgd']},
            }}]}, {
        'description': "nesterovs_momentum, XXX TODO XXX, only used when solver='sgd' and momentum > 0"}, {
        'description': "early_stopping, only effective when solver='sgd' or 'adam'",
        'anyOf': [{
            'type': 'object',
            'properties': {
                'early_stopping': {
                    'enum': [False]},
            }}, {
            'type': 'object',
            'properties': {
                'solver': {
                    'enum': ['sgd', 'adam']},
            }}]}, {
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
            }}]}, {
        'description': "beta_1, only used when solver='adam'",
        'anyOf': [{
            'type': 'object',
            'properties': {
                'beta_1': {
                    'enum': [0.9]},
            }}, {
            'type': 'object',
            'properties': {
                'solver': {
                    'enum': ['adam']},
            }}]}, {
        'description': "beta_2, only used when solver='adam'",
        'anyOf': [{
            'type': 'object',
            'properties': {
                'beta_2': {
                    'enum': [0.999]},
            }}, {
            'type': 'object',
            'properties': {
                'solver': {
                    'enum': ['adam']},
            }}]}, {
        'description': "epsilon, only used when solver='adam'",
        'anyOf': [{
            'type': 'object',
            'properties': {
                'epsilon': {
                    'enum': [1e-08]},
            }}, {
            'type': 'object',
            'properties': {
                'solver': {
                    'enum': ['adam']},
            }}]}, {
        'description': "n_iter_no_change, only effective when solver='sgd' or 'adam' ",
        'anyOf': [{
            'type': 'object',
            'properties': {
                'n_iter_no_change': {
                    'enum': [10]},
            }}, {
            'type': 'object',
            'properties': {
                'solver': {
                    'enum': ['sgd', 'adam']},
            }}]}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the model to data matrix X and target(s) y.',
    'type': 'object',
    'properties': {
        'X': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'XXX TODO XXX': 'item type'},
                'XXX TODO XXX': 'array-like or sparse matrix, shape (n_samples, n_features)'}, {
                'type': 'array',
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }}],
            'description': 'The input data.'},
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
            'description': 'The target values (class labels in classification, real numbers in'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict using the multi-layer perceptron model.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'The input data.'},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'The predicted values.',
    'type': 'array',
    'items': {
        'type': 'array',
        'items': {
            'type': 'number'},
    },
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
MLPRegressor = lale.operators.make_operator(MLPRegressorImpl, _combined_schemas)

