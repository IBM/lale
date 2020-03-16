
from sklearn.cluster.birch import Birch as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class BirchImpl():

    def __init__(self, threshold=0.5, branching_factor=50, n_clusters=3, compute_labels=True, copy=True):
        self._hyperparams = {
            'threshold': threshold,
            'branching_factor': branching_factor,
            'n_clusters': n_clusters,
            'compute_labels': compute_labels,
            'copy': copy}
        self._sklearn_model = SKLModel(**self._hyperparams)

    def fit(self, X, y=None):
        if (y is not None):
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X)
        return self

    def transform(self, X):
        return self._sklearn_model.transform(X)

    def predict(self, X):
        return self._sklearn_model.predict(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for Birch    Implements the Birch clustering algorithm.',
    'allOf': [{
        'type': 'object',
        'required': ['threshold', 'branching_factor', 'n_clusters', 'compute_labels', 'copy'],
        'relevantToOptimizer': ['branching_factor', 'n_clusters', 'compute_labels', 'copy'],
        'additionalProperties': False,
        'properties': {
            'threshold': {
                'type': 'number',
                'default': 0.5,
                'description': 'The radius of the subcluster obtained by merging a new sample and the closest subcluster should be lesser than the threshold'},
            'branching_factor': {
                'type': 'integer',
                'minimumForOptimizer': 50,
                'maximumForOptimizer': 51,
                'distribution': 'uniform',
                'default': 50,
                'description': 'Maximum number of CF subclusters in each node'},
            'n_clusters': {
                'XXX TODO XXX': 'int, instance of sklearn.cluster model, default 3',
                'description': 'Number of clusters after the final clustering step, which treats the subclusters from the leaves as new samples',
                'type': 'integer',
                'minimumForOptimizer': 2,
                'maximumForOptimizer': 8,
                'distribution': 'uniform',
                'default': 3},
            'compute_labels': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether or not to compute labels for each fit.'},
            'copy': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether or not to make a copy of the given data'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Build a CF Tree for the input data.',
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
            'description': 'Input data.'},
        'y': {
            
        }},
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transform X into subcluster centroids dimension.',
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
            'description': 'Input data.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transformed data.',
    'type': 'array',
    'items': {
        'type': 'array',
        'items': {
            'type': 'number'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict data using the ``centroids_`` of subclusters.',
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
            'description': 'Input data.'},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Labelled data.',
    'laleType': 'Any',
    'XXX TODO XXX': 'ndarray, shape(n_samples)',
}
_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'type': 'object',
    'tags': {
        'pre': [],
        'op': ['transformer'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_transform': _input_transform_schema,
        'output_transform': _output_transform_schema,
        'input_predict': _input_predict_schema,
        'output_predict': _output_predict_schema},
}
if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
Birch = lale.operators.make_operator(BirchImpl, _combined_schemas)

