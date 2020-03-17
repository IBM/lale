
from sklearn.linear_model.ransac import RANSACRegressor as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class RANSACRegressorImpl():

    def __init__(self, base_estimator=None, min_samples=None, residual_threshold=None, is_data_valid=None, is_model_valid=None, max_trials=100, max_skips='inf', stop_n_inliers='inf', stop_score='inf', stop_probability=0.99, loss='absolute_loss', random_state=None):
        self._hyperparams = {
            'base_estimator': base_estimator,
            'min_samples': min_samples,
            'residual_threshold': residual_threshold,
            'is_data_valid': is_data_valid,
            'is_model_valid': is_model_valid,
            'max_trials': max_trials,
            'max_skips': max_skips,
            'stop_n_inliers': stop_n_inliers,
            'stop_score': stop_score,
            'stop_probability': stop_probability,
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
    'description': 'inherited docstring for RANSACRegressor    RANSAC (RANdom SAmple Consensus) algorithm.',
    'allOf': [{
        'type': 'object',
        'required': ['base_estimator', 'min_samples', 'residual_threshold', 'is_data_valid', 'is_model_valid', 'max_trials', 'max_skips', 'stop_n_inliers', 'stop_score', 'stop_probability', 'loss', 'random_state'],
        'relevantToOptimizer': ['min_samples', 'max_trials', 'max_skips', 'stop_n_inliers', 'loss'],
        'additionalProperties': False,
        'properties': {
            'base_estimator': {
                'anyOf': [{
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Base estimator object which implements the following methods:   * `fit(X, y)`: Fit model to given training data and target values'},
            'min_samples': {
                'XXX TODO XXX': 'int (>= 1) or float ([0, 1]), optional',
                'description': 'Minimum number of samples chosen randomly from original data',
                'anyOf': [{
                    'type': 'number',
                    'minimumForOptimizer': 0.0,
                    'maximumForOptimizer': 1.0,
                    'distribution': 'uniform'}, {
                    'enum': [None]}],
                'default': None},
            'residual_threshold': {
                'anyOf': [{
                    'type': 'number'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Maximum residual for a data sample to be classified as an inlier'},
            'is_data_valid': {
                'anyOf': [{
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'This function is called with the randomly selected data before the model is fitted to it: `is_data_valid(X, y)`'},
            'is_model_valid': {
                'anyOf': [{
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'This function is called with the estimated model and the randomly selected data: `is_model_valid(model, X, y)`'},
            'max_trials': {
                'type': 'integer',
                'minimumForOptimizer': 100,
                'maximumForOptimizer': 101,
                'distribution': 'uniform',
                'default': 100,
                'description': 'Maximum number of iterations for random sample selection.'},
            'max_skips': {
                'anyOf': [{
                    'type': 'integer',
                    'forOptimizer': False}, {
                    'type': 'number',
                    'minimumForOptimizer': 0.0,
                    'maximumForOptimizer': 1.0,
                    'distribution': 'uniform'}],
                'default': inf,
                'description': 'Maximum number of iterations that can be skipped due to finding zero inliers or invalid data defined by ``is_data_valid`` or invalid models defined by ``is_model_valid``'},
            'stop_n_inliers': {
                'anyOf': [{
                    'type': 'integer',
                    'forOptimizer': False}, {
                    'type': 'number',
                    'minimumForOptimizer': 0.0,
                    'maximumForOptimizer': 1.0,
                    'distribution': 'uniform'}],
                'default': inf,
                'description': 'Stop iteration if at least this number of inliers are found.'},
            'stop_score': {
                'type': 'number',
                'default': inf,
                'description': 'Stop iteration if score is greater equal than this threshold.'},
            'stop_probability': {
                'XXX TODO XXX': 'float in range [0, 1], optional',
                'description': 'RANSAC iteration stops if at least one outlier-free set of the training data is sampled in RANSAC',
                'type': 'number',
                'default': 0.99},
            'loss': {
                'anyOf': [{
                    'type': 'object',
                    'forOptimizer': False}, {
                    'enum': ['absolute_loss', 'squared_loss']}],
                'default': 'absolute_loss',
                'description': 'String inputs, "absolute_loss" and "squared_loss" are supported which find the absolute loss and squared loss per sample respectively'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The generator used to initialize the centers'},
        }}, {
        'XXX TODO XXX': 'Parameter: base_estimator > only supports regression estimators'}, {
        'XXX TODO XXX': 'Parameter: is_model_valid > only be used if the estimated model is needed for making the rejection decision'}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit estimator using RANSAC algorithm.',
    'type': 'object',
    'required': ['X', 'y'],
    'properties': {
        'X': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'laleType': 'Any',
                    'XXX TODO XXX': 'item type'},
                'XXX TODO XXX': 'array-like or sparse matrix, shape [n_samples, n_features]'}, {
                'type': 'array',
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }}],
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
            'description': 'Target values.'},
        'sample_weight': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'Individual weights for each sample raises error if sample_weight is passed and base_estimator fit method does not support it.'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict using the estimated model.',
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
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Returns predicted values.',
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
RANSACRegressor = lale.operators.make_operator(RANSACRegressorImpl, _combined_schemas)

