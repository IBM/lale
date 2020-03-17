
from sklearn.linear_model.passive_aggressive import PassiveAggressiveClassifier as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class PassiveAggressiveClassifierImpl():

    def __init__(self, C=1.0, fit_intercept=True, max_iter=None, tol=None, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, shuffle=True, verbose=0, loss='hinge', n_jobs=None, random_state=None, warm_start=False, class_weight='balanced', average=False, n_iter=None):
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
            'n_jobs': n_jobs,
            'random_state': random_state,
            'warm_start': warm_start,
            'class_weight': class_weight,
            'average': average,
            'n_iter': n_iter}
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
    'description': 'inherited docstring for PassiveAggressiveClassifier    Passive Aggressive Classifier',
    'allOf': [{
        'type': 'object',
        'required': ['C', 'fit_intercept', 'max_iter', 'tol', 'early_stopping', 'validation_fraction', 'n_iter_no_change', 'shuffle', 'verbose', 'loss', 'n_jobs', 'random_state', 'warm_start', 'class_weight', 'average', 'n_iter'],
        'relevantToOptimizer': ['fit_intercept', 'max_iter', 'tol', 'shuffle', 'loss', 'n_iter'],
        'additionalProperties': False,
        'properties': {
            'C': {
                'type': 'number',
                'default': 1.0,
                'description': 'Maximum step size (regularization)'},
            'fit_intercept': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether the intercept should be estimated or not'},
            'max_iter': {
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 10,
                    'maximumForOptimizer': 1000,
                    'distribution': 'uniform'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The maximum number of passes over the training data (aka epochs)'},
            'tol': {
                'anyOf': [{
                    'type': 'number',
                    'minimumForOptimizer': 1e-08,
                    'maximumForOptimizer': 0.01,
                    'distribution': 'loguniform'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The stopping criterion'},
            'early_stopping': {
                'type': 'boolean',
                'default': False,
                'description': 'Whether to use early stopping to terminate training when validation'},
            'validation_fraction': {
                'type': 'number',
                'default': 0.1,
                'description': 'The proportion of training data to set aside as validation set for early stopping'},
            'n_iter_no_change': {
                'type': 'integer',
                'default': 5,
                'description': 'Number of iterations with no improvement to wait before early stopping'},
            'shuffle': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether or not the training data should be shuffled after each epoch.'},
            'verbose': {
                'type': 'integer',
                'default': 0,
                'description': 'The verbosity level'},
            'loss': {
                'enum': ['epsilon_insensitive', 'huber', 'log', 'modified_huber', 'perceptron', 'squared_epsilon_insensitive', 'squared_hinge', 'squared_loss', 'hinge'],
                'default': 'hinge',
                'description': 'The loss function to be used: hinge: equivalent to PA-I in the reference paper'},
            'n_jobs': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The number of CPUs to use to do the OVA (One Versus All, for multi-class problems) computation'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The seed of the pseudo random number generator to use when shuffling the data'},
            'warm_start': {
                'type': 'boolean',
                'default': False,
                'description': 'When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution'},
            'class_weight': {
                'XXX TODO XXX': 'dict, {class_label: weight} or "balanced" or None, optional',
                'description': 'Preset for the class_weight fit parameter',
                'enum': ['balanced'],
                'default': 'balanced'},
            'average': {
                'anyOf': [{
                    'type': 'boolean'}, {
                    'type': 'integer'}],
                'default': False,
                'description': 'When set to True, computes the averaged SGD weights and stores the result in the ``coef_`` attribute'},
            'n_iter': {
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 5,
                    'maximumForOptimizer': 1000,
                    'distribution': 'uniform'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The number of passes over the training data (aka epochs)'},
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
        'coef_init': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
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
    'description': 'Predict class labels for samples in X.',
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
    'description': 'Predicted class label per sample.',
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
PassiveAggressiveClassifier = lale.operators.make_operator(PassiveAggressiveClassifierImpl, _combined_schemas)

