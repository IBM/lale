
from sklearn.decomposition.sparse_pca import MiniBatchSparsePCA as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class MiniBatchSparsePCAImpl():

    def __init__(self, n_components=None, alpha=1, ridge_alpha=0.01, n_iter=100, callback=None, batch_size=3, verbose=False, shuffle=True, n_jobs=None, method='lars', random_state=None, normalize_components=False):
        self._hyperparams = {
            'n_components': n_components,
            'alpha': alpha,
            'ridge_alpha': ridge_alpha,
            'n_iter': n_iter,
            'callback': callback,
            'batch_size': batch_size,
            'verbose': verbose,
            'shuffle': shuffle,
            'n_jobs': n_jobs,
            'method': method,
            'random_state': random_state,
            'normalize_components': normalize_components}
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
    'description': 'inherited docstring for MiniBatchSparsePCA    Mini-batch Sparse Principal Components Analysis',
    'allOf': [{
        'type': 'object',
        'required': ['n_components', 'alpha', 'ridge_alpha', 'n_iter', 'callback', 'batch_size', 'verbose', 'shuffle', 'n_jobs', 'method', 'random_state', 'normalize_components'],
        'relevantToOptimizer': ['n_components', 'alpha', 'n_iter', 'batch_size', 'shuffle', 'method'],
        'additionalProperties': False,
        'properties': {
            'n_components': {
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 2,
                    'maximumForOptimizer': 256,
                    'distribution': 'uniform'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'number of sparse atoms to extract'},
            'alpha': {
                'type': 'integer',
                'minimumForOptimizer': 1,
                'maximumForOptimizer': 2,
                'distribution': 'uniform',
                'default': 1,
                'description': 'Sparsity controlling parameter'},
            'ridge_alpha': {
                'type': 'number',
                'default': 0.01,
                'description': 'Amount of ridge shrinkage to apply in order to improve conditioning when calling the transform method.'},
            'n_iter': {
                'type': 'integer',
                'minimumForOptimizer': 5,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 100,
                'description': 'number of iterations to perform for each mini batch'},
            'callback': {
                'anyOf': [{
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'callable that gets invoked every five iterations'},
            'batch_size': {
                'type': 'integer',
                'minimumForOptimizer': 3,
                'maximumForOptimizer': 128,
                'distribution': 'uniform',
                'default': 3,
                'description': 'the number of features to take in each mini batch'},
            'verbose': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'boolean'}],
                'default': False,
                'description': 'Controls the verbosity; the higher, the more messages'},
            'shuffle': {
                'type': 'boolean',
                'default': True,
                'description': 'whether to shuffle the data before splitting it in batches'},
            'n_jobs': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Number of parallel jobs to run'},
            'method': {
                'enum': ['lars', 'cd'],
                'default': 'lars',
                'description': 'lars: uses the least angle regression method to solve the lasso problem (linear_model.lars_path) cd: uses the coordinate descent method to compute the Lasso solution (linear_model.Lasso)'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by `np.random`.'},
            'normalize_components': {
                'type': 'boolean',
                'default': False,
                'description': '- if False, use a version of Sparse PCA without components   normalization and without data centering'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the model from data in X.',
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
            
        }},
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Least Squares projection of the data onto the sparse components.',
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
            'description': 'Test data to be transformed, must have the same number of features as the data used to train the model.'},
        'ridge_alpha': {
            'type': 'number',
            'default': 0.01,
            'description': 'Amount of ridge shrinkage to apply in order to improve conditioning'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transformed data.',
    'laleType': 'Any',
    'XXX TODO XXX': '',
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
MiniBatchSparsePCA = lale.operators.make_operator(MiniBatchSparsePCAImpl, _combined_schemas)

