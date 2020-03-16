
from sklearn.svm.classes import LinearSVR as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class LinearSVRImpl():

    def __init__(self, epsilon=0.0, tol=0.0001, C=1.0, loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1.0, dual=True, verbose=0, random_state=None, max_iter=1000):
        self._hyperparams = {
            'epsilon': epsilon,
            'tol': tol,
            'C': C,
            'loss': loss,
            'fit_intercept': fit_intercept,
            'intercept_scaling': intercept_scaling,
            'dual': dual,
            'verbose': verbose,
            'random_state': random_state,
            'max_iter': max_iter}
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
    'description': 'inherited docstring for LinearSVR    Linear Support Vector Regression.',
    'allOf': [{
        'type': 'object',
        'required': ['epsilon', 'tol', 'C', 'loss', 'fit_intercept', 'intercept_scaling', 'dual', 'verbose', 'random_state', 'max_iter'],
        'relevantToOptimizer': ['epsilon', 'tol', 'loss', 'fit_intercept', 'dual', 'max_iter'],
        'additionalProperties': False,
        'properties': {
            'epsilon': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 1.35,
                'distribution': 'loguniform',
                'default': 0.0,
                'description': 'Epsilon parameter in the epsilon-insensitive loss function'},
            'tol': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'distribution': 'loguniform',
                'default': 0.0001,
                'description': 'Tolerance for stopping criteria.'},
            'C': {
                'type': 'number',
                'default': 1.0,
                'description': 'Penalty parameter C of the error term'},
            'loss': {
                'enum': ['hinge', 'l2', 'squared_epsilon_insensitive', 'squared_hinge', 'epsilon_insensitive'],
                'default': 'epsilon_insensitive',
                'description': 'Specifies the loss function'},
            'fit_intercept': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether to calculate the intercept for this model'},
            'intercept_scaling': {
                'type': 'number',
                'default': 1.0,
                'description': 'When self.fit_intercept is True, instance vector x becomes [x, self.intercept_scaling], i.e'},
            'dual': {
                'type': 'boolean',
                'default': True,
                'description': 'Select the algorithm to either solve the dual or primal optimization problem'},
            'verbose': {
                'type': 'integer',
                'default': 0,
                'description': 'Enable verbose output'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The seed of the pseudo random number generator to use when shuffling the data'},
            'max_iter': {
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 1000,
                'description': 'The maximum number of iterations to be run.'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the model according to the given training data.',
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
            'description': 'Training vector, where n_samples in the number of samples and n_features is the number of features.'},
        'y': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'Target vector relative to X'},
        'sample_weight': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'type': 'number'},
            }, {
                'enum': [None]}],
            'default': None,
            'description': 'Array of weights that are assigned to individual samples'},
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
LinearSVR = lale.operators.make_operator(LinearSVRImpl, _combined_schemas)

