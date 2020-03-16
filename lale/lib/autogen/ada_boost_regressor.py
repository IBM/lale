
from sklearn.ensemble.weight_boosting import AdaBoostRegressor as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class AdaBoostRegressorImpl():

    def __init__(self, base_estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None):
        self._hyperparams = {
            'base_estimator': base_estimator,
            'n_estimators': n_estimators,
            'learning_rate': learning_rate,
            'loss': loss,
            'random_state': random_state}
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
    'description': 'inherited docstring for AdaBoostRegressor    An AdaBoost regressor.',
    'allOf': [{
        'type': 'object',
        'required': ['base_estimator', 'n_estimators', 'learning_rate', 'loss', 'random_state'],
        'relevantToOptimizer': ['n_estimators', 'loss'],
        'additionalProperties': False,
        'properties': {
            'base_estimator': {
                'anyOf': [{
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The base estimator from which the boosted ensemble is built'},
            'n_estimators': {
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 100,
                'distribution': 'uniform',
                'default': 50,
                'description': 'The maximum number of estimators at which boosting is terminated'},
            'learning_rate': {
                'type': 'number',
                'default': 1.0,
                'description': 'Learning rate shrinks the contribution of each regressor by ``learning_rate``'},
            'loss': {
                'enum': ['linear', 'square', 'exponential'],
                'default': 'linear',
                'description': 'The loss function to use when updating the weights after each boosting iteration.'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by `np.random`.'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Build a boosted regressor from the training set (X, y).',
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
            'description': 'The training input samples'},
        'y': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'The target values (real numbers).'},
        'sample_weight': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'type': 'number'},
            }, {
                'enum': [None]}],
            'default': None,
            'description': 'Sample weights'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict regression value for X.',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'The training input samples'},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'The predicted regression values.',
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
AdaBoostRegressor = lale.operators.make_operator(AdaBoostRegressorImpl, _combined_schemas)

