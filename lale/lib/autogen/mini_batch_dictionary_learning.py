
from sklearn.decomposition.dict_learning import MiniBatchDictionaryLearning as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class MiniBatchDictionaryLearningImpl():

    def __init__(self, n_components=None, alpha=1, n_iter=1000, fit_algorithm='lars', n_jobs=None, batch_size=3, shuffle=True, dict_init=None, transform_algorithm='omp', transform_n_nonzero_coefs=None, transform_alpha=None, verbose=False, split_sign=False, random_state=None, positive_code=False, positive_dict=False):
        self._hyperparams = {
            'n_components': n_components,
            'alpha': alpha,
            'n_iter': n_iter,
            'fit_algorithm': fit_algorithm,
            'n_jobs': n_jobs,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'dict_init': dict_init,
            'transform_algorithm': transform_algorithm,
            'transform_n_nonzero_coefs': transform_n_nonzero_coefs,
            'transform_alpha': transform_alpha,
            'verbose': verbose,
            'split_sign': split_sign,
            'random_state': random_state,
            'positive_code': positive_code,
            'positive_dict': positive_dict}
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
    'description': 'inherited docstring for MiniBatchDictionaryLearning    Mini-batch dictionary learning',
    'allOf': [{
        'type': 'object',
        'required': ['n_components', 'alpha', 'n_iter', 'fit_algorithm', 'n_jobs', 'batch_size', 'shuffle', 'dict_init', 'transform_algorithm', 'transform_n_nonzero_coefs', 'transform_alpha', 'verbose', 'split_sign', 'random_state', 'positive_code', 'positive_dict'],
        'relevantToOptimizer': ['n_components', 'alpha', 'n_iter', 'fit_algorithm', 'batch_size', 'shuffle', 'transform_algorithm'],
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
                'description': 'number of dictionary elements to extract'},
            'alpha': {
                'type': 'number',
                'minimumForOptimizer': 1e-10,
                'maximumForOptimizer': 1.0,
                'distribution': 'loguniform',
                'default': 1,
                'description': 'sparsity controlling parameter'},
            'n_iter': {
                'type': 'integer',
                'minimumForOptimizer': 5,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 1000,
                'description': 'total number of iterations to perform'},
            'fit_algorithm': {
                'enum': ['lars', 'cd'],
                'default': 'lars',
                'description': 'lars: uses the least angle regression method to solve the lasso problem (linear_model.lars_path) cd: uses the coordinate descent method to compute the Lasso solution (linear_model.Lasso)'},
            'n_jobs': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Number of parallel jobs to run'},
            'batch_size': {
                'type': 'integer',
                'minimumForOptimizer': 3,
                'maximumForOptimizer': 128,
                'distribution': 'uniform',
                'default': 3,
                'description': 'number of samples in each mini-batch'},
            'shuffle': {
                'type': 'boolean',
                'default': True,
                'description': 'whether to shuffle the samples before forming batches'},
            'dict_init': {
                'anyOf': [{
                    'type': 'array',
                    'items': {
                        'type': 'array',
                        'items': {
                            'type': 'number'},
                    }}, {
                    'enum': [None]}],
                'default': None,
                'description': 'initial value of the dictionary for warm restart scenarios'},
            'transform_algorithm': {
                'enum': ['lasso_lars', 'lasso_cd', 'lars', 'omp', 'threshold'],
                'default': 'omp',
                'description': 'Algorithm used to transform the data'},
            'transform_n_nonzero_coefs': {
                'XXX TODO XXX': 'int, ``0.1 * n_features`` by default',
                'description': 'Number of nonzero coefficients to target in each column of the solution',
                'enum': [None],
                'default': None},
            'transform_alpha': {
                'anyOf': [{
                    'type': 'number'}, {
                    'enum': [None]}],
                'default': None,
                'description': "If `algorithm='lasso_lars'` or `algorithm='lasso_cd'`, `alpha` is the penalty applied to the L1 norm"},
            'verbose': {
                'type': 'boolean',
                'default': False,
                'description': 'To control the verbosity of the procedure.'},
            'split_sign': {
                'type': 'boolean',
                'default': False,
                'description': 'Whether to split the sparse feature vector into the concatenation of its negative part and its positive part'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by `np.random`.'},
            'positive_code': {
                'type': 'boolean',
                'default': False,
                'description': 'Whether to enforce positivity when finding the code'},
            'positive_dict': {
                'type': 'boolean',
                'default': False,
                'description': 'Whether to enforce positivity when finding the dictionary'},
        }}, {
        'XXX TODO XXX': "Parameter: transform_n_nonzero_coefs > only used by algorithm='lars' and algorithm='omp' and is overridden by alpha in the omp case"}],
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
    'description': 'Encode the data as a sparse combination of the dictionary atoms.',
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
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transformed data',
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
MiniBatchDictionaryLearning = lale.operators.make_operator(MiniBatchDictionaryLearningImpl, _combined_schemas)

