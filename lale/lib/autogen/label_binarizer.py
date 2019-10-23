
from sklearn.preprocessing.label import LabelBinarizer as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class LabelBinarizerImpl():

    def __init__(self, neg_label=0, pos_label=1, sparse_output=False):
        self._hyperparams = {
            'neg_label': neg_label,
            'pos_label': pos_label,
            'sparse_output': sparse_output}

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
    'description': 'inherited docstring for LabelBinarizer    Binarize labels in a one-vs-all fashion',
    'allOf': [{
        'type': 'object',
        'required': ['neg_label', 'pos_label', 'sparse_output'],
        'relevantToOptimizer': ['neg_label', 'pos_label', 'sparse_output'],
        'additionalProperties': False,
        'properties': {
            'neg_label': {
                'type': 'integer',
                'minimumForOptimizer': 0,
                'maximumForOptimizer': 1,
                'distribution': 'uniform',
                'default': 0,
                'description': 'Value with which negative labels must be encoded.'},
            'pos_label': {
                'type': 'integer',
                'minimumForOptimizer': 1,
                'maximumForOptimizer': 2,
                'distribution': 'uniform',
                'default': 1,
                'description': 'Value with which positive labels must be encoded.'},
            'sparse_output': {
                'type': 'boolean',
                'default': False,
                'description': 'True if the returned array from transform is desired to be in sparse'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit label binarizer',
    'type': 'object',
    'properties': {
        'y': {
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
            'description': 'Target values. The 2-d matrix should only contain 0 and 1,'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transform multi-class labels to binary labels',
    'type': 'object',
    'properties': {
        'y': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'XXX TODO XXX': 'item type'},
                'XXX TODO XXX': 'array or sparse matrix of shape [n_samples,] or             [n_samples, n_classes]'}, {
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
            'description': 'Target values. The 2-d matrix should only contain 0 and 1,'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Shape will be [n_samples, 1] for binary problems.',
    'XXX TODO XXX': 'numpy array or CSR matrix of shape [n_samples, n_classes]',
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
LabelBinarizer = lale.operators.make_operator(LabelBinarizerImpl, _combined_schemas)

