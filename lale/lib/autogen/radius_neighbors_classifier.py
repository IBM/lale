
from sklearn.neighbors.classification import RadiusNeighborsClassifier as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class RadiusNeighborsClassifierImpl():

    def __init__(self, radius=1.0, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', outlier_label=None, metric_params=None, n_jobs=None):
        self._hyperparams = {
            'radius': radius,
            'weights': weights,
            'algorithm': algorithm,
            'leaf_size': leaf_size,
            'p': p,
            'metric': metric,
            'outlier_label': outlier_label,
            'metric_params': metric_params,
            'n_jobs': n_jobs}
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
    'description': 'inherited docstring for RadiusNeighborsClassifier    Classifier implementing a vote among neighbors within a given radius',
    'allOf': [{
        'type': 'object',
        'required': ['radius', 'weights', 'algorithm', 'leaf_size', 'p', 'metric', 'outlier_label', 'metric_params', 'n_jobs'],
        'relevantToOptimizer': ['weights', 'algorithm', 'leaf_size', 'p', 'metric'],
        'additionalProperties': False,
        'properties': {
            'radius': {
                'type': 'number',
                'default': 1.0,
                'description': 'Range of parameter space to use by default for :meth:`radius_neighbors` queries.'},
            'weights': {
                'anyOf': [{
                    'type': 'object',
                    'forOptimizer': False}, {
                    'enum': ['distance', 'uniform']}],
                'default': 'uniform',
                'description': 'weight function used in prediction'},
            'algorithm': {
                'enum': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'default': 'auto',
                'description': "Algorithm used to compute the nearest neighbors:  - 'ball_tree' will use :class:`BallTree` - 'kd_tree' will use :class:`KDTree` - 'brute' will use a brute-force search"},
            'leaf_size': {
                'type': 'integer',
                'minimumForOptimizer': 30,
                'maximumForOptimizer': 31,
                'distribution': 'uniform',
                'default': 30,
                'description': 'Leaf size passed to BallTree or KDTree'},
            'p': {
                'type': 'integer',
                'minimumForOptimizer': 2,
                'maximumForOptimizer': 3,
                'distribution': 'uniform',
                'default': 2,
                'description': 'Power parameter for the Minkowski metric'},
            'metric': {
                'anyOf': [{
                    'type': 'object',
                    'forOptimizer': False}, {
                    'enum': ['euclidean', 'manhattan', 'minkowski', 'precomputed']}],
                'default': 'minkowski',
                'description': 'the distance metric to use for the tree'},
            'outlier_label': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Label, which is given for outlier samples (samples with no neighbors on given radius)'},
            'metric_params': {
                'anyOf': [{
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Additional keyword arguments for the metric function.'},
            'n_jobs': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The number of parallel jobs to run for neighbors search'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the model using X as training data and y as target values',
    'type': 'object',
    'required': ['X', 'y'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'laleType': 'Any',
                'XXX TODO XXX': 'item type'},
            'XXX TODO XXX': '{array-like, sparse matrix, BallTree, KDTree}',
            'description': 'Training data'},
        'y': {
            'type': 'array',
            'items': {
                'laleType': 'Any',
                'XXX TODO XXX': 'item type'},
            'XXX TODO XXX': '{array-like, sparse matrix}',
            'description': 'Target values of shape = [n_samples] or [n_samples, n_outputs]'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict the class labels for the provided data',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'laleType': 'Any',
            'XXX TODO XXX': "array-like, shape (n_query, n_features),                 or (n_query, n_indexed) if metric == 'precomputed'",
            'description': 'Test samples.'},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Class labels for each data sample.',
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
RadiusNeighborsClassifier = lale.operators.make_operator(RadiusNeighborsClassifierImpl, _combined_schemas)

