
from sklearn.neural_network.rbm import BernoulliRBM as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class BernoulliRBMImpl():

    def __init__(self, n_components=256, learning_rate=0.1, batch_size=10, n_iter=10, verbose=0, random_state=33):
        self._hyperparams = {
            'n_components': n_components,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'n_iter': n_iter,
            'verbose': verbose,
            'random_state': random_state}

    def fit(self, X, y=None):
        self._sklearn_model = SKLModel(**self._hyperparams)
        if (y is not None):
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X)
        return self

    def transform(self, X):
        return self._sklearn_model.transform(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for BernoulliRBM    Bernoulli Restricted Boltzmann Machine (RBM).',
    'allOf': [{
        'type': 'object',
        'required': ['n_components', 'learning_rate', 'batch_size', 'n_iter', 'verbose', 'random_state'],
        'relevantToOptimizer': ['n_components', 'batch_size', 'n_iter'],
        'additionalProperties': False,
        'properties': {
            'n_components': {
                'type': 'integer',
                'minimumForOptimizer': 2,
                'maximumForOptimizer': 256,
                'distribution': 'uniform',
                'default': 256,
                'description': 'Number of binary hidden units.'},
            'learning_rate': {
                'type': 'number',
                'default': 0.1,
                'description': 'The learning rate for weight updates. It is *highly* recommended'},
            'batch_size': {
                'type': 'integer',
                'minimumForOptimizer': 3,
                'maximumForOptimizer': 128,
                'distribution': 'uniform',
                'default': 10,
                'description': 'Number of examples per minibatch.'},
            'n_iter': {
                'type': 'integer',
                'minimumForOptimizer': 5,
                'maximumForOptimizer': 1000,
                'distribution': 'uniform',
                'default': 10,
                'description': 'Number of iterations/sweeps over the training dataset to perform'},
            'verbose': {
                'type': 'integer',
                'default': 0,
                'description': 'The verbosity level. The default, zero, means silent mode.'},
            'random_state': {
                'XXX TODO XXX': 'integer or RandomState, optional',
                'description': 'A random number generator instance to define the state of the',
                'type': 'integer',
                'default': 33},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the model to the data X.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'Training data.'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Compute the hidden layer activation probabilities, P(h=1|v=X).',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'The data to be transformed.'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Latent representations of the data.',
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
BernoulliRBM = lale.operators.make_operator(BernoulliRBMImpl, _combined_schemas)

