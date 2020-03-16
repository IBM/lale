
from sklearn.decomposition.online_lda import LatentDirichletAllocation as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class LatentDirichletAllocationImpl():

    def __init__(self, n_components=10, doc_topic_prior=None, topic_word_prior=None, learning_method='batch', learning_decay=0.7, learning_offset=10.0, max_iter=10, batch_size=128, evaluate_every=(- 1), total_samples=1000000.0, perp_tol=0.1, mean_change_tol=0.001, max_doc_update_iter=100, n_jobs=None, verbose=0, random_state=None, n_topics=None):
        self._hyperparams = {
            'n_components': n_components,
            'doc_topic_prior': doc_topic_prior,
            'topic_word_prior': topic_word_prior,
            'learning_method': learning_method,
            'learning_decay': learning_decay,
            'learning_offset': learning_offset,
            'max_iter': max_iter,
            'batch_size': batch_size,
            'evaluate_every': evaluate_every,
            'total_samples': total_samples,
            'perp_tol': perp_tol,
            'mean_change_tol': mean_change_tol,
            'max_doc_update_iter': max_doc_update_iter,
            'n_jobs': n_jobs,
            'verbose': verbose,
            'random_state': random_state,
            'n_topics': n_topics}
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
    'description': 'inherited docstring for LatentDirichletAllocation    Latent Dirichlet Allocation with online variational Bayes algorithm',
    'allOf': [{
        'type': 'object',
        'required': ['n_components', 'doc_topic_prior', 'topic_word_prior', 'learning_method', 'learning_decay', 'learning_offset', 'max_iter', 'batch_size', 'evaluate_every', 'total_samples', 'perp_tol', 'mean_change_tol', 'max_doc_update_iter', 'n_jobs', 'verbose', 'random_state', 'n_topics'],
        'relevantToOptimizer': ['n_components', 'learning_method', 'max_iter', 'batch_size', 'evaluate_every', 'total_samples', 'max_doc_update_iter'],
        'additionalProperties': False,
        'properties': {
            'n_components': {
                'type': 'integer',
                'minimumForOptimizer': 2,
                'maximumForOptimizer': 256,
                'distribution': 'uniform',
                'default': 10,
                'description': 'Number of topics.'},
            'doc_topic_prior': {
                'anyOf': [{
                    'type': 'number'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Prior of document topic distribution `theta`'},
            'topic_word_prior': {
                'anyOf': [{
                    'type': 'number'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Prior of topic word distribution `beta`'},
            'learning_method': {
                'enum': ['batch', 'online'],
                'default': 'batch',
                'description': 'Method used to update `_component`'},
            'learning_decay': {
                'type': 'number',
                'default': 0.7,
                'description': 'It is a parameter that control learning rate in the online learning method'},
            'learning_offset': {
                'type': 'number',
                'default': 10.0,
                'description': 'A (positive) parameter that downweights early iterations in online learning'},
            'max_iter': {
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 10,
                'description': 'The maximum number of iterations.'},
            'batch_size': {
                'type': 'integer',
                'minimumForOptimizer': 3,
                'maximumForOptimizer': 128,
                'distribution': 'uniform',
                'default': 128,
                'description': 'Number of documents to use in each EM iteration'},
            'evaluate_every': {
                'type': 'integer',
                'minimumForOptimizer': (- 1),
                'maximumForOptimizer': 0,
                'distribution': 'uniform',
                'default': (- 1),
                'description': 'How often to evaluate perplexity'},
            'total_samples': {
                'anyOf': [{
                    'type': 'integer',
                    'forOptimizer': False}, {
                    'type': 'number',
                    'minimumForOptimizer': 0.0,
                    'maximumForOptimizer': 1.0,
                    'distribution': 'uniform'}],
                'default': 1000000.0,
                'description': 'Total number of documents'},
            'perp_tol': {
                'type': 'number',
                'default': 0.1,
                'description': 'Perplexity tolerance in batch learning'},
            'mean_change_tol': {
                'type': 'number',
                'default': 0.001,
                'description': 'Stopping tolerance for updating document topic distribution in E-step.'},
            'max_doc_update_iter': {
                'type': 'integer',
                'minimumForOptimizer': 100,
                'maximumForOptimizer': 101,
                'distribution': 'uniform',
                'default': 100,
                'description': 'Max number of iterations for updating document topic distribution in the E-step.'},
            'n_jobs': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The number of jobs to use in the E-step'},
            'verbose': {
                'type': 'integer',
                'default': 0,
                'description': 'Verbosity level.'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by `np.random`.'},
            'n_topics': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'This parameter has been renamed to n_components and will be removed in version 0.21'},
        }}, {
        'description': 'learning_method, only used in fit method',
        'anyOf': [{
            'type': 'object',
            'properties': {
                'learning_method': {
                    'enum': ['batch']},
            }}, {
            'type': 'object',
            'properties': {
                'method': {
                    'enum': ['fit']},
            }}]}, {
        'description': 'batch_size, only used in online learning',
        'anyOf': [{
            'type': 'object',
            'properties': {
                'batch_size': {
                    'enum': [128]},
            }}, {
            'type': 'object',
            'properties': {
                'learning': {
                    'enum': ['online']},
            }}]}, {
        'description': 'evaluate_every, only used in fit method',
        'anyOf': [{
            'type': 'object',
            'properties': {
                'evaluate_every': {
                    'enum': [(- 1)]},
            }}, {
            'type': 'object',
            'properties': {
                'method': {
                    'enum': ['fit']},
            }}]}, {
        'description': 'total_samples, only used in the partial_fit method',
        'anyOf': [{
            'type': 'object',
            'properties': {
                'total_samples': {
                    'enum': [1000000.0]},
            }}, {
            'type': 'object',
            'properties': {
                'method': {
                    'enum': ['partial_fit']},
            }}]}, {
        'XXX TODO XXX': 'Parameter: perp_tol > only used when evaluate_every is greater than 0'}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Learn model for the data X with variational Bayes method.',
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
            'description': 'Document word matrix.'},
        'y': {
            
        }},
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transform data X according to the fitted model.',
    'type': 'object',
    'required': ['X'],
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
            'description': 'Document word matrix.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Document topic distribution for X.',
    'laleType': 'Any',
    'XXX TODO XXX': 'shape=(n_samples, n_components)',
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
LatentDirichletAllocation = lale.operators.make_operator(LatentDirichletAllocationImpl, _combined_schemas)

