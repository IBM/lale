
from sklearn.manifold.isomap import Isomap as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class IsomapImpl():

    def __init__(self, n_neighbors=5, n_components=2, eigen_solver='auto', tol=0, max_iter=None, path_method='auto', neighbors_algorithm='auto', n_jobs=None):
        self._hyperparams = {
            'n_neighbors': n_neighbors,
            'n_components': n_components,
            'eigen_solver': eigen_solver,
            'tol': tol,
            'max_iter': max_iter,
            'path_method': path_method,
            'neighbors_algorithm': neighbors_algorithm,
            'n_jobs': n_jobs}
        self._sklearn_model = SKLModel(**self._hyperparams)

    def fit(self, X, y=None):
        if (y is not None):
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X)
        return self

    def transform(self, X):
        return self._sklearn_model.transform(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for Isomap    Isomap Embedding',
    'allOf': [{
        'type': 'object',
        'required': ['n_neighbors', 'n_components', 'eigen_solver', 'tol', 'max_iter', 'path_method', 'neighbors_algorithm', 'n_jobs'],
        'relevantToOptimizer': ['n_neighbors', 'n_components', 'eigen_solver', 'tol', 'path_method', 'neighbors_algorithm'],
        'additionalProperties': False,
        'properties': {
            'n_neighbors': {
                'type': 'integer',
                'minimumForOptimizer': 5,
                'maximumForOptimizer': 20,
                'distribution': 'uniform',
                'default': 5,
                'description': 'number of neighbors to consider for each point.'},
            'n_components': {
                'type': 'integer',
                'minimumForOptimizer': 2,
                'maximumForOptimizer': 256,
                'distribution': 'uniform',
                'default': 2,
                'description': 'number of coordinates for the manifold'},
            'eigen_solver': {
                'enum': ['auto', 'arpack', 'dense'],
                'default': 'auto',
                'description': "'auto' : Attempt to choose the most efficient solver for the given problem"},
            'tol': {
                'type': 'number',
                'minimumForOptimizer': 0.0,
                'maximumForOptimizer': 1.0,
                'distribution': 'uniform',
                'default': 0,
                'description': 'Convergence tolerance passed to arpack or lobpcg'},
            'max_iter': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Maximum number of iterations for the arpack solver'},
            'path_method': {
                'enum': ['D', 'FW', 'auto'],
                'default': 'auto',
                'description': 'Method to use in finding shortest path'},
            'neighbors_algorithm': {
                'enum': ['auto', 'ball_tree', 'brute', 'kd_tree'],
                'default': 'auto',
                'description': 'Algorithm to use for nearest neighbors search, passed to neighbors.NearestNeighbors instance.'},
            'n_jobs': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The number of parallel jobs to run'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Compute the embedding vectors for data X',
    'type': 'object',
    'required': ['X', 'y'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'laleType': 'Any',
                'XXX TODO XXX': 'item type'},
            'XXX TODO XXX': '{array-like, sparse matrix, BallTree, KDTree, NearestNeighbors}',
            'description': 'Sample data, shape = (n_samples, n_features), in the form of a numpy array, precomputed tree, or NearestNeighbors object.'},
        'y': {
            
        }},
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transform X.',
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
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transform X.',
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
        'op': ['transformer'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_transform': _input_transform_schema,
        'output_transform': _output_transform_schema},
}
if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
Isomap = lale.operators.make_operator(IsomapImpl, _combined_schemas)

