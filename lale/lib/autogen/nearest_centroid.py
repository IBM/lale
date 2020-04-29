
from sklearn.neighbors.nearest_centroid import NearestCentroid as Op
import lale.helpers
import lale.operators
import lale.docstrings
from numpy import nan, inf

class NearestCentroidImpl():

    def __init__(self, metric='euclidean', shrink_threshold=None):
        self._hyperparams = {
            'metric': metric,
            'shrink_threshold': shrink_threshold}
        self._wrapped_model = Op(**self._hyperparams)

    def fit(self, X, y=None):
        if (y is not None):
            self._wrapped_model.fit(X, y)
        else:
            self._wrapped_model.fit(X)
        return self

    def predict(self, X):
        return self._wrapped_model.predict(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for NearestCentroid    Nearest centroid classifier.',
    'allOf': [{
        'type': 'object',
        'required': ['metric', 'shrink_threshold'],
        'relevantToOptimizer': ['metric'],
        'additionalProperties': False,
        'properties': {
            'metric': {
                'anyOf': [{
                    'type': 'object',
                    'forOptimizer': False}, {
                    'enum': ['euclidean', 'manhattan', 'minkowski']}],
                'default': 'euclidean',
                'description': 'The metric to use when calculating distance between instances in a feature array'},
            'shrink_threshold': {
                'anyOf': [{
                    'type': 'number'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Threshold for shrinking centroids to remove features.'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the NearestCentroid model according to the given training data.',
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
            'description': 'Training vector, where n_samples is the number of samples and n_features is the number of features'},
        'y': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'Target values (integers)'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Perform classification on an array of test vectors X.',
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
    'description': 'Perform classification on an array of test vectors X.',
    'type': 'array',
    'items': {
        'type': 'number'},
}
_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'documentation_url': 'https://scikit-learn.org/0.20/modules/generated/sklearn.neighbors.NearestCentroid#sklearn-neighbors-nearestcentroid',
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
lale.docstrings.set_docstrings(NearestCentroidImpl, _combined_schemas)
NearestCentroid = lale.operators.make_operator(NearestCentroidImpl, _combined_schemas)

