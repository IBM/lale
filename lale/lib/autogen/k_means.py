
from sklearn.cluster.k_means_ import KMeans as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class KMeansImpl():

    def __init__(self, n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=None, algorithm='auto'):
        self._hyperparams = {
            'n_clusters': n_clusters,
            'init': init,
            'n_init': n_init,
            'max_iter': max_iter,
            'tol': tol,
            'precompute_distances': precompute_distances,
            'verbose': verbose,
            'random_state': random_state,
            'copy_x': copy_x,
            'n_jobs': n_jobs,
            'algorithm': algorithm}
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
    'description': 'inherited docstring for KMeans    K-Means clustering',
    'allOf': [{
        'type': 'object',
        'required': ['n_clusters', 'init', 'n_init', 'max_iter', 'tol', 'precompute_distances', 'verbose', 'random_state', 'copy_x', 'n_jobs', 'algorithm'],
        'relevantToOptimizer': ['n_clusters', 'init', 'n_init', 'max_iter', 'tol', 'precompute_distances', 'copy_x', 'algorithm'],
        'additionalProperties': False,
        'properties': {
            'n_clusters': {
                'type': 'integer',
                'minimumForOptimizer': 2,
                'maximumForOptimizer': 8,
                'distribution': 'uniform',
                'default': 8,
                'description': 'The number of clusters to form as well as the number of centroids to generate.'},
            'init': {
                'enum': ['k-means++', 'random', 'ndarray'],
                'default': 'k-means++',
                'description': "Method for initialization, defaults to 'k-means++':  'k-means++' : selects initial cluster centers for k-mean clustering in a smart way to speed up convergence"},
            'n_init': {
                'type': 'integer',
                'minimumForOptimizer': 3,
                'maximumForOptimizer': 10,
                'distribution': 'uniform',
                'default': 10,
                'description': 'Number of time the k-means algorithm will be run with different centroid seeds'},
            'max_iter': {
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 300,
                'description': 'Maximum number of iterations of the k-means algorithm for a single run.'},
            'tol': {
                'type': 'number',
                'minimumForOptimizer': 1e-08,
                'maximumForOptimizer': 0.01,
                'distribution': 'loguniform',
                'default': 0.0001,
                'description': 'Relative tolerance with regards to inertia to declare convergence'},
            'precompute_distances': {
                'enum': ['auto', True, False],
                'default': 'auto',
                'description': 'Precompute distances (faster but takes more memory)'},
            'verbose': {
                'type': 'integer',
                'default': 0,
                'description': 'Verbosity mode.'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Determines random number generation for centroid initialization'},
            'copy_x': {
                'type': 'boolean',
                'default': True,
                'description': 'When pre-computing distances it is more numerically accurate to center the data first'},
            'n_jobs': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The number of jobs to use for the computation'},
            'algorithm': {
                'XXX TODO XXX': '"auto", "full" or "elkan", default="auto"',
                'description': 'K-means algorithm to use',
                'enum': ['auto', 'elkan', 'full'],
                'default': 'auto'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Compute k-means clustering.',
    'type': 'object',
    'required': ['X', 'y'],
    'properties': {
        'X': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'laleType': 'Any',
                    'XXX TODO XXX': 'item type'},
                'XXX TODO XXX': 'array-like or sparse matrix, shape=(n_samples, n_features)'}, {
                'type': 'array',
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }}],
            'description': 'Training instances to cluster'},
        'y': {
            'description': 'not used, present here for API consistency by convention.'},
        'sample_weight': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'type': 'number'},
            }, {
                'enum': [None]}],
            'default': None,
            'description': 'The weights for each observation in X'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transform X to a cluster-distance space.',
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
            'description': 'New data to transform.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'X transformed in the new space.',
    'type': 'array',
    'items': {
        'type': 'array',
        'items': {
            'type': 'number'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict the closest cluster each sample in X belongs to.',
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
            'description': 'New data to predict.'},
        'sample_weight': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'type': 'number'},
            }, {
                'enum': [None]}],
            'default': None,
            'description': 'The weights for each observation in X'},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Index of the cluster each sample belongs to.',
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
KMeans = lale.operators.make_operator(KMeansImpl, _combined_schemas)

