
from sklearn.linear_model.stochastic_gradient import SGDClassifier as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class SGDClassifierImpl():

    def __init__(self, loss='hinge', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=None, tol=None, shuffle=True, verbose=0, epsilon=0.1, n_jobs=None, random_state=None, learning_rate='optimal', eta0=0.0, power_t=0.5, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5, class_weight='balanced', warm_start=False, average=False, n_iter=None):
        self._hyperparams = {
            'loss': loss,
            'penalty': penalty,
            'alpha': alpha,
            'l1_ratio': l1_ratio,
            'fit_intercept': fit_intercept,
            'max_iter': max_iter,
            'tol': tol,
            'shuffle': shuffle,
            'verbose': verbose,
            'epsilon': epsilon,
            'n_jobs': n_jobs,
            'random_state': random_state,
            'learning_rate': learning_rate,
            'eta0': eta0,
            'power_t': power_t,
            'early_stopping': early_stopping,
            'validation_fraction': validation_fraction,
            'n_iter_no_change': n_iter_no_change,
            'class_weight': class_weight,
            'warm_start': warm_start,
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

    def predict_proba(self, X):
        return self._sklearn_model.predict_proba(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for SGDClassifier    Linear classifiers (SVM, logistic regression, a.o.) with SGD training.',
    'allOf': [{
        'type': 'object',
        'required': ['loss', 'penalty', 'alpha', 'l1_ratio', 'fit_intercept', 'max_iter', 'tol', 'shuffle', 'verbose', 'epsilon', 'n_jobs', 'random_state', 'learning_rate', 'eta0', 'power_t', 'early_stopping', 'validation_fraction', 'n_iter_no_change', 'class_weight', 'warm_start', 'average', 'n_iter'],
        'relevantToOptimizer': ['loss', 'penalty', 'alpha', 'fit_intercept', 'max_iter', 'tol', 'shuffle', 'epsilon', 'learning_rate', 'eta0', 'n_iter'],
        'additionalProperties': False,
        'properties': {
            'loss': {
                'enum': ['epsilon_insensitive', 'huber', 'log', 'modified_huber', 'perceptron', 'squared_epsilon_insensitive', 'squared_hinge', 'squared_loss', 'hinge'],
                'default': 'hinge',
                'description': 'The loss function to be used'},
            'penalty': {
                'XXX TODO XXX': "str, 'none', 'l2', 'l1', or 'elasticnet'",
                'description': 'The penalty (aka regularization term) to be used',
                'enum': ['elasticnet', 'l1', 'l2', 'none'],
                'default': 'l2'},
            'alpha': {
                'type': 'number',
                'minimumForOptimizer': 1e-10,
                'maximumForOptimizer': 1.0,
                'distribution': 'loguniform',
                'default': 0.0001,
                'description': 'Constant that multiplies the regularization term'},
            'l1_ratio': {
                'type': 'number',
                'default': 0.15,
                'description': 'The Elastic Net mixing parameter, with 0 <= l1_ratio <= 1'},
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
            'shuffle': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether or not the training data should be shuffled after each epoch'},
            'verbose': {
                'type': 'integer',
                'default': 0,
                'description': 'The verbosity level'},
            'epsilon': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 1.35,
                'distribution': 'loguniform',
                'default': 0.1,
                'description': "Epsilon in the epsilon-insensitive loss functions; only if `loss` is 'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'"},
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
            'learning_rate': {
                'enum': ['optimal'],
                'default': 'optimal',
                'description': "The learning rate schedule:  'constant':     eta = eta0 'optimal': [default]     eta = 1.0 / (alpha * (t + t0))     where t0 is chosen by a heuristic proposed by Leon Bottou"},
            'eta0': {
                'type': 'number',
                'minimumForOptimizer': 0.01,
                'maximumForOptimizer': 1.0,
                'distribution': 'uniform',
                'default': 0.0,
                'description': "The initial learning rate for the 'constant', 'invscaling' or 'adaptive' schedules"},
            'power_t': {
                'type': 'number',
                'default': 0.5,
                'description': 'The exponent for inverse scaling learning rate [default 0.5].'},
            'early_stopping': {
                'type': 'boolean',
                'default': False,
                'description': 'Whether to use early stopping to terminate training when validation score is not improving'},
            'validation_fraction': {
                'type': 'number',
                'default': 0.1,
                'description': 'The proportion of training data to set aside as validation set for early stopping'},
            'n_iter_no_change': {
                'type': 'integer',
                'default': 5,
                'description': 'Number of iterations with no improvement to wait before early stopping'},
            'class_weight': {
                'XXX TODO XXX': 'dict, {class_label: weight} or "balanced" or None, optional',
                'description': 'Preset for the class_weight fit parameter',
                'enum': ['balanced'],
                'default': 'balanced'},
            'warm_start': {
                'type': 'boolean',
                'default': False,
                'description': 'When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution'},
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
        'description': "epsilon, only if loss is 'huber'",
        'anyOf': [{
            'type': 'object',
            'properties': {
                'epsilon': {
                    'enum': [0.1]},
            }}, {
            'type': 'object',
            'properties': {
                'loss': {
                    'enum': ['huber']},
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
            }}]}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit linear model with Stochastic Gradient Descent.',
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
        'sample_weight': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'type': 'number'},
            }, {
                'enum': [None]}],
            'default': None,
            'description': 'Weights applied to individual samples'},
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
_input_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Probability estimates.',
    'type': 'object',
    'required': ['X'],
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
_output_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Returns the probability of the sample for each class in the model, where classes are ordered as they are in `self.classes_`.',
    'laleType': 'Any',
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
        'output_predict': _output_predict_schema,
        'input_predict_proba': _input_predict_proba_schema,
        'output_predict_proba': _output_predict_proba_schema},
}
if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
SGDClassifier = lale.operators.make_operator(SGDClassifierImpl, _combined_schemas)

