
from sklearn.preprocessing.label import MultiLabelBinarizer as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class MultiLabelBinarizerImpl():

    def __init__(self, classes=None, sparse_output=False):
        self._hyperparams = {
            'classes': classes,
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
    'description': 'inherited docstring for MultiLabelBinarizer    Transform between iterable of iterables and a multilabel format',
    'allOf': [{
        'type': 'object',
        'relevantToOptimizer': ['sparse_output'],
        'additionalProperties': False,
        'properties': {
            'classes': {
                'XXX TODO XXX': 'array-like of shape [n_classes] (optional)',
                'description': 'Indicates an ordering for the class labels.',
                'enum': [None],
                'default': None},
            'sparse_output': {
                'type': 'boolean',
                'default': False,
                'description': 'Set to true if output binary array is desired in CSR sparse format'},
        }}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the label sets binarizer, storing `classes_`',
    'type': 'object',
    'properties': {
        'y': {
            'XXX TODO XXX': 'iterable of iterables',
            'description': 'A set of labels (any orderable and hashable object) for each'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Transform the given label sets',
    'type': 'object',
    'properties': {
        'y': {
            'XXX TODO XXX': 'iterable of iterables',
            'description': 'A set of labels (any orderable and hashable object) for each'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'A matrix such that `y_indicator[i, j] = 1` iff `classes_[j]` is in',
    'XXX TODO XXX': 'array or CSR matrix, shape (n_samples, n_classes)',
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
MultiLabelBinarizer = lale.operators.make_operator(MultiLabelBinarizerImpl, _combined_schemas)

