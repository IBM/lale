
from sklearn.ensemble.gradient_boosting import GradientBoostingClassifier as Op
import lale.helpers
import lale.operators
import lale.docstrings
from numpy import nan, inf

class GradientBoostingClassifierImpl():

    def __init__(self, loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, presort='auto', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001):
        self._hyperparams = {
            'loss': loss,
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'criterion': criterion,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'min_weight_fraction_leaf': min_weight_fraction_leaf,
            'max_depth': max_depth,
            'min_impurity_decrease': min_impurity_decrease,
            'min_impurity_split': min_impurity_split,
            'init': init,
            'random_state': random_state,
            'max_features': max_features,
            'verbose': verbose,
            'max_leaf_nodes': max_leaf_nodes,
            'warm_start': warm_start,
            'presort': presort,
            'validation_fraction': validation_fraction,
            'n_iter_no_change': n_iter_no_change,
            'tol': tol}
        self._wrapped_model = Op(**self._hyperparams)

    def fit(self, X, y=None):
        if (y is not None):
            self._wrapped_model.fit(X, y)
        else:
            self._wrapped_model.fit(X)
        return self

    def predict(self, X):
        return self._wrapped_model.predict(X)

    def predict_proba(self, X):
        return self._wrapped_model.predict_proba(X)

    def decision_function(self, X):
        return self._wrapped_model.decision_function(X)
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'inherited docstring for GradientBoostingClassifier    Gradient Boosting for classification.',
    'allOf': [{
        'type': 'object',
        'required': ['loss', 'learning_rate', 'n_estimators', 'subsample', 'criterion', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_depth', 'min_impurity_decrease', 'min_impurity_split', 'init', 'random_state', 'max_features', 'verbose', 'max_leaf_nodes', 'warm_start', 'presort', 'validation_fraction', 'n_iter_no_change', 'tol'],
        'relevantToOptimizer': ['loss', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'max_depth', 'max_features', 'presort'],
        'additionalProperties': False,
        'properties': {
            'loss': {
                'enum': ['deviance', 'exponential'],
                'default': 'deviance',
                'description': 'loss function to be optimized'},
            'learning_rate': {
                'type': 'number',
                'default': 0.1,
                'description': 'learning rate shrinks the contribution of each tree by `learning_rate`'},
            'n_estimators': {
                'type': 'integer',
                'minimumForOptimizer': 10,
                'maximumForOptimizer': 100,
                'distribution': 'uniform',
                'default': 100,
                'description': 'The number of boosting stages to perform'},
            'subsample': {
                'type': 'number',
                'default': 1.0,
                'description': 'The fraction of samples to be used for fitting the individual base learners'},
            'criterion': {
                'type': 'string',
                'default': 'friedman_mse',
                'description': 'The function to measure the quality of a split'},
            'min_samples_split': {
                'anyOf': [{
                    'type': 'integer',
                    'forOptimizer': False}, {
                    'type': 'number',
                    'minimumForOptimizer': 0.01,
                    'maximumForOptimizer': 0.5,
                    'distribution': 'uniform'}],
                'default': 2,
                'description': 'The minimum number of samples required to split an internal node:  - If int, then consider `min_samples_split` as the minimum number'},
            'min_samples_leaf': {
                'anyOf': [{
                    'type': 'integer',
                    'forOptimizer': False}, {
                    'type': 'number',
                    'minimumForOptimizer': 0.01,
                    'maximumForOptimizer': 0.5,
                    'distribution': 'uniform'}],
                'default': 1,
                'description': 'The minimum number of samples required to be at a leaf node'},
            'min_weight_fraction_leaf': {
                'type': 'number',
                'default': 0.0,
                'description': 'The minimum weighted fraction of the sum total of weights (of all the input samples) required to be at a leaf node'},
            'max_depth': {
                'type': 'integer',
                'minimumForOptimizer': 3,
                'maximumForOptimizer': 5,
                'distribution': 'uniform',
                'default': 3,
                'description': 'maximum depth of the individual regression estimators'},
            'min_impurity_decrease': {
                'type': 'number',
                'default': 0.0,
                'description': 'A node will be split if this split induces a decrease of the impurity greater than or equal to this value'},
            'min_impurity_split': {
                'anyOf': [{
                    'type': 'number'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Threshold for early stopping in tree growth'},
            'init': {
                'XXX TODO XXX': 'estimator, optional',
                'description': 'An estimator object that is used to compute the initial predictions',
                'enum': [None],
                'default': None},
            'random_state': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'type': 'object'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by `np.random`.'},
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
                'description': 'The number of features to consider when looking for the best split:  - If int, then consider `max_features` features at each split'},
            'verbose': {
                'type': 'integer',
                'default': 0,
                'description': 'Enable verbose output'},
            'max_leaf_nodes': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'Grow trees with ``max_leaf_nodes`` in best-first fashion'},
            'warm_start': {
                'type': 'boolean',
                'default': False,
                'description': 'When set to ``True``, reuse the solution of the previous call to fit and add more estimators to the ensemble, otherwise, just erase the previous solution'},
            'presort': {
                'XXX TODO XXX': "bool or 'auto', optional (default='auto')",
                'description': 'Whether to presort the data to speed up the finding of best splits in fitting',
                'enum': ['auto'],
                'default': 'auto'},
            'validation_fraction': {
                'type': 'number',
                'default': 0.1,
                'description': 'The proportion of training data to set aside as validation set for early stopping'},
            'n_iter_no_change': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': '``n_iter_no_change`` is used to decide if early stopping will be used to terminate training when validation score is not improving'},
            'tol': {
                'type': 'number',
                'default': 0.0001,
                'description': 'Tolerance for the early stopping'},
        }}, {
        'XXX TODO XXX': 'Parameter: min_samples_leaf > only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches'}, {
        'XXX TODO XXX': 'Parameter: validation_fraction > only used if n_iter_no_change is set to an integer'}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the gradient boosting model.',
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
            'description': 'The input samples'},
        'y': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'Target values (strings or integers in classification, real numbers in regression) For classification, labels must correspond to classes.'},
        'sample_weight': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'type': 'number'},
            }, {
                'enum': [None]}],
            'description': 'Sample weights'},
        'monitor': {
            'anyOf': [{
                'type': 'object'}, {
                'enum': [None]}],
            'default': None,
            'description': 'The monitor is called after each iteration with the current iteration, a reference to the estimator and the local variables of ``_fit_stages`` as keyword arguments ``callable(i, self, locals())``'},
    },
}
_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict class for X.',
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
            'description': 'The input samples'},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'The predicted values.',
    'type': 'array',
    'items': {
        'type': 'number'},
}
_input_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict class probabilities for X.',
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
            'description': 'The input samples'},
    },
}
_output_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'The class probabilities of the input samples',
    'type': 'array',
    'items': {
        'type': 'array',
        'items': {
            'type': 'number'},
    },
}
_input_decision_function_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Compute the decision function of ``X``.',
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
            'description': 'The input samples'},
    },
}
_output_decision_function_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'The decision function of the input samples',
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
    'documentation_url': 'https://scikit-learn.org/0.20/modules/generated/sklearn.ensemble.GradientBoostingClassifier#sklearn-ensemble-gradientboostingclassifier',
    'type': 'object',
    'tags': {
        'pre': [],
        'op': ['estimator', 'classifier'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_predict': _input_predict_schema,
        'output_predict': _output_predict_schema,
        'input_predict_proba': _input_predict_proba_schema,
        'output_predict_proba': _output_predict_proba_schema,
        'input_decision_function': _input_decision_function_schema,
        'output_decision_function': _output_decision_function_schema},
}
lale.docstrings.set_docstrings(GradientBoostingClassifierImpl, _combined_schemas)
GradientBoostingClassifier = lale.operators.make_operator(GradientBoostingClassifierImpl, _combined_schemas)

