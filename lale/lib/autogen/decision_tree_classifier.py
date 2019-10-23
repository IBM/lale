
from sklearn.tree.tree import DecisionTreeClassifier as SKLModel
import lale.helpers
import lale.operators
from numpy import nan, inf

class DecisionTreeClassifierImpl():

    def __init__(self, criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, class_weight='balanced', presort=False):
        self._hyperparams = {
            'criterion': criterion,
            'splitter': splitter,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'min_weight_fraction_leaf': min_weight_fraction_leaf,
            'max_features': max_features,
            'random_state': random_state,
            'max_leaf_nodes': max_leaf_nodes,
            'min_impurity_decrease': min_impurity_decrease,
            'min_impurity_split': min_impurity_split,
            'class_weight': class_weight,
            'presort': presort}

    def fit(self, X, y=None):
        self._sklearn_model = SKLModel(**self._hyperparams)
        if (y is not None):
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X)
        return self

    def predict(self, X):
        return self._sklearn_model.predict(X)

    def predict_proba(self, X):
        return self._sklearn_model.predict_proba(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for DecisionTreeClassifier    A decision tree classifier.',
    'allOf': [{
        'type': 'object',
        'required': ['criterion', 'splitter', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features', 'random_state', 'max_leaf_nodes', 'min_impurity_decrease', 'min_impurity_split', 'class_weight', 'presort'],
        'relevantToOptimizer': ['criterion', 'splitter', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'],
        'additionalProperties': False,
        'properties': {
            'criterion': {
                'enum': ['entropy', 'gini'],
                'default': 'gini',
                'description': 'The function to measure the quality of a split. Supported criteria are'},
            'splitter': {
                'enum': ['random', 'best'],
                'default': 'best',
                'description': 'The strategy used to choose the split at each node. Supported'},
            'max_depth': {
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 3,
                    'maximumForOptimizer': 5,
                    'distribution': 'uniform'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The maximum depth of the tree. If None, then nodes are expanded until'},
            'min_samples_split': {
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 2,
                    'maximumForOptimizer': 5,
                    'distribution': 'uniform'}, {
                    'type': 'number',
                    'forOptimizer': False}],
                'default': 2,
                'description': 'The minimum number of samples required to split an internal node:'},
            'min_samples_leaf': {
                'anyOf': [{
                    'type': 'integer',
                    'minimumForOptimizer': 1,
                    'maximumForOptimizer': 5,
                    'distribution': 'uniform'}, {
                    'type': 'number',
                    'forOptimizer': False}],
                'default': 1,
                'description': 'The minimum number of samples required to be at a leaf node.'},
            'min_weight_fraction_leaf': {
                'type': 'number',
                'default': 0.0,
                'description': 'The minimum weighted fraction of the sum total of weights (of all'},
            'max_features': {
                'anyOf': [{
                    'type': 'integer',
                    'forOptimizer': False}, {
                    'type': 'number',
                    'minimumForOptimizer': 0.01,
                    'maximumForOptimizer': 1.0,
                    'distribution': 'uniform'}, {
                    'type': 'string',
                    'forOptimizer': False}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The number of features to consider when looking for the best split:'},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'If int, random_state is the seed used by the random number generator;'},
            'max_leaf_nodes': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Grow a tree with ``max_leaf_nodes`` in best-first fashion.'},
            'min_impurity_decrease': {
                'type': 'number',
                'default': 0.0,
                'description': 'A node will be split if this split induces a decrease of the impurity'},
            'min_impurity_split': {
                'anyOf': [{
                    'type': 'number'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Threshold for early stopping in tree growth. A node will split'},
            'class_weight': {
                'XXX TODO XXX': 'dict, list of dicts, "balanced" or None, default=None',
                'description': 'Weights associated with classes in the form ``{class_label: weight}``.',
                'enum': ['balanced'],
                'default': 'balanced'},
            'presort': {
                'type': 'boolean',
                'default': False,
                'description': 'Whether to presort the data to speed up the finding of best splits in'},
        }}, {
        'XXX TODO XXX': 'Parameter: min_samples_leaf > only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches'}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Build a decision tree classifier from the training set (X, y).',
    'type': 'object',
    'properties': {
        'X': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'XXX TODO XXX': 'item type'},
                'XXX TODO XXX': 'array-like or sparse matrix, shape = [n_samples, n_features]'}, {
                'type': 'array',
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }}],
            'description': 'The training input samples. Internally, it will be converted to'},
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
            'description': 'The target values (class labels) as integers or strings.'},
        'sample_weight': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'type': 'number'},
            }, {
                'enum': [None]}],
            'description': 'Sample weights. If None, then samples are equally weighted. Splits'},
        'check_input': {
            'type': 'boolean',
            'default': True,
            'description': 'Allow to bypass several input checking.'},
        'X_idx_sorted': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }}, {
                'enum': [None]}],
            'default': None,
            'description': 'The indexes of the sorted training input samples. If many tree'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict class or regression value for X.',
    'type': 'object',
    'properties': {
        'X': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'XXX TODO XXX': 'item type'},
                'XXX TODO XXX': 'array-like or sparse matrix of shape = [n_samples, n_features]'}, {
                'type': 'array',
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }}],
            'description': 'The input samples. Internally, it will be converted to'},
        'check_input': {
            'type': 'boolean',
            'default': True,
            'description': 'Allow to bypass several input checking.'},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'The predicted classes, or the predict values.',
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
_input_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict class probabilities of the input samples X.',
    'type': 'object',
    'properties': {
        'X': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'XXX TODO XXX': 'item type'},
                'XXX TODO XXX': 'array-like or sparse matrix of shape = [n_samples, n_features]'}, {
                'type': 'array',
                'items': {
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }}],
            'description': 'The input samples. Internally, it will be converted to'},
        'check_input': {
            'type': 'boolean',
            'description': 'Run check_array on X.'},
    },
}
_output_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'such arrays if n_outputs > 1.',
    'XXX TODO XXX': 'array of shape = [n_samples, n_classes], or a list of n_outputs',
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
        'output_predict': _output_predict_schema,
        'input_predict_proba': _input_predict_proba_schema,
        'output_predict_proba': _output_predict_proba_schema},
}
if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
DecisionTreeClassifier = lale.operators.make_operator(DecisionTreeClassifierImpl, _combined_schemas)

