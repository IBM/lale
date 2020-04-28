
from sklearn.cluster.hierarchical import FeatureAgglomeration as Op
import lale.helpers
import lale.operators
import lale.docstrings
from numpy import nan, inf

class FeatureAgglomerationImpl():

    def __init__(self, n_clusters=2, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', pooling_func=None):
        self._hyperparams = {
            'n_clusters': n_clusters,
            'affinity': affinity,
            'memory': memory,
            'connectivity': connectivity,
            'compute_full_tree': compute_full_tree,
            'linkage': linkage,
            'pooling_func': pooling_func}
        self._wrapped_model = Op(**self._hyperparams)

    def fit(self, X, y=None):
        if (y is not None):
            self._wrapped_model.fit(X, y)
        else:
            self._wrapped_model.fit(X)
        return self

    def transform(self, X):
        return self._wrapped_model.transform(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for FeatureAgglomeration    Agglomerate features.',
    'allOf': [{
        'type': 'object',
        'required': ['n_clusters', 'affinity', 'memory', 'connectivity', 'compute_full_tree', 'linkage', 'pooling_func'],
        'relevantToOptimizer': ['n_clusters', 'affinity', 'compute_full_tree', 'linkage'],
        'additionalProperties': False,
        'properties': {
            'n_clusters': {
                'type': 'integer',
                'minimumForOptimizer': 2,
                'maximumForOptimizer': 8,
                'distribution': 'uniform',
                'default': 2,
                'description': 'The number of clusters to find.'},
            'affinity': {
                'anyOf': [{
                    'type': 'object',
                    'forOptimizer': False}, {
                    'enum': ['euclidean']}],
                'default': 'euclidean',
                'description': 'Metric used to compute the linkage'},
            'memory': {
                'XXX TODO XXX': 'None, str or object with the joblib.Memory interface, optional',
                'description': 'Used to cache the output of the computation of the tree',
                'enum': [None],
                'default': None},
            'connectivity': {
                'anyOf': [{
                    'type': 'array',
                    'items': {
                        'laleType': 'Any',
                        'XXX TODO XXX': 'item type'},
                    'XXX TODO XXX': 'array-like or callable, optional'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Connectivity matrix'},
            'compute_full_tree': {
                'XXX TODO XXX': 'bool or \'auto\', optional, default "auto"',
                'description': 'Stop early the construction of the tree at n_clusters',
                'enum': ['auto'],
                'default': 'auto'},
            'linkage': {
                'enum': ['ward', 'complete', 'average', 'single'],
                'default': 'ward',
                'description': 'Which linkage criterion to use'},
            'pooling_func': {
                'laleType': 'Any',
                'XXX TODO XXX': 'callable, default np.mean',
                'description': 'This combines the values of agglomerated features into a single value, and should accept an array of shape [M, N] and the keyword argument `axis=1`, and reduce it to an array of size [M].'},
        }}, {
        'XXX TODO XXX': 'Parameter: affinity > only "euclidean" is accepted'}, {
        'XXX TODO XXX': 'Parameter: compute_full_tree > only when specifying a connectivity matrix'}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the hierarchical clustering on the data',
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
            'description': 'The data'},
        'y': {
            
        }},
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transform a new matrix using the built clustering',
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }}, {
                'type': 'array',
                'items': {
                    'type': 'number'},
            }],
            'description': 'A M by N array of M observations in N dimensions or a length M array of M one-dimensional observations.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'The pooled values for each feature cluster.',
    'anyOf': [{
        'type': 'array',
        'items': {
            'type': 'array',
            'items': {
                'type': 'number'},
        }}, {
        'type': 'array',
        'items': {
            'type': 'number'},
    }],
}
_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'documentation_url': 'https://scikit-learn.org/0.20/modules/generated/sklearn.cluster.FeatureAgglomeration#sklearn-cluster-featureagglomeration',
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
lale.docstrings.set_docstrings(FeatureAgglomerationImpl, _combined_schemas)
FeatureAgglomeration = lale.operators.make_operator(FeatureAgglomerationImpl, _combined_schemas)

