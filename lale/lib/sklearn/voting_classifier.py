# Copyright 2019 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sklearn.ensemble import VotingClassifier as SKLModel
import lale.helpers
import lale.operators

class VotingClassifierImpl():
    def __init__(self, estimators=None, voting='hard', weights=None, n_jobs=None, flatten_transform=True):
        self._hyperparams = {
            'estimators': estimators,
            'voting': voting,
            'weights': weights,
            'n_jobs': n_jobs,
            'flatten_transform': flatten_transform}
        self._sklearn_model = SKLModel(**self._hyperparams)

    def fit(self, X, y=None):
        if (y is not None):
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X)
        return self
    
    def transform(self, X):
        return self._sklearn_model.transform(X)

    def predict(self, X):
        return self._sklearn_model.predict(X)
    
    def predict_proba(self, X):
        return self._sklearn_model.predict_proba(X)

_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Soft Voting/Majority Rule classifier for unfitted estimators.',
    'allOf': [{
        'type': 'object',
        'required': ['estimators', 'voting', 'weights', 'n_jobs', 'flatten_transform'],
        'relevantToOptimizer': ['voting'],
        'additionalProperties': False,
        'properties': {
            'estimators': {
                'type': 'array',
                'items': {
                    'type': 'array',
                    'laleType': 'tuple',
                    'items': [
                        {'type':'string'},
                        {'laleType': 'operator'}
                    ]
                },
                'description': 'list of (string, estimator) tuples. Invoking the ``fit`` method on the ``VotingClassifier`` will fit clones'},
            'voting': {
                'enum': ['hard', 'soft'],
                'default': 'hard',
                'description': "If 'hard', uses predicted class labels for majority rule voting."},
            'weights': {
                'anyOf': [{
                    'type': 'array',
                    'items': {
                        'type': 'number'},
                }, {
                    'enum': [None]}],
                'default': None,
                'description': 'Sequence of weights (`float` or `int`) to weight the occurrences of'},
            'n_jobs': {
                'anyOf': [{
                    'type': 'integer'}, {
                    'enum': [None]}],
                'default': None,
                'description': 'The number of jobs to run in parallel for ``fit``.'},
            'flatten_transform': {
                'type': 'boolean',
                'default': True,
                'description': "Affects shape of transform output only when voting='soft'"},
        }}, {
        'description': "Parameter: flatten_transform > only when voting='soft' if voting='soft' and flatten_transform=true",
        'anyOf': [{
            'type': 'object',
            'properties': {
                'voting': {
                    'enum': ['soft']},
            }}, {
            'type': 'object',
            'properties': {
                'flatten_transform': {
                    'enum': [True]},
            }}]}],
}
_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Fit the estimators.',
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
            'description': 'Training vectors, where n_samples is the number of samples and'},
        'y': {
            'type': 'array',
            'items': {
                'type': 'number'},
            'description': 'Target values.'},
        'sample_weight': {
            'anyOf': [{
                'type': 'array',
                'items': {
                    'type': 'number'},
            }, {
                'enum': [None]}],
            'description': 'Sample weights. If None, then samples are equally weighted.'},
    },
}
_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Return class labels or probabilities for X for each estimator.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'Training vectors, where n_samples is the number of samples and'},
    },
}
_output_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': "If `voting='soft'` and `flatten_transform=True`:",
    'type': 'array',
    'items': {
        'type': 'array',
        'items': { 'anyOf':[{
            'type': 'number'},{
            'type': 'array',
            'items': {'type':'number'}
            }]}}}

_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predict class labels for X.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'The input samples.'},
    },
}
_output_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Predicted class labels.',
    'type': 'array',
    'items': {
        'type': 'number'},
}
_input_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Compute probabilities of possible outcomes for samples in X.',
    'type': 'object',
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'},
            },
            'description': 'The input samples.'},
    },
}
_output_predict_proba_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Weighted average probability for each class per sample.',
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
    'documentation_url': 'https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html',
    'type': 'object',
    'tags': {
        'pre': [],
        'op': ['transformer'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_predict': _input_predict_schema,
        'output_predict': _output_predict_schema,
        'input_predict_proba': _input_predict_proba_schema,
        'output_predict_proba': _output_predict_proba_schema,
        'input_transform': _input_transform_schema,
        'output_transform': _output_transform_schema}}

if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)
VotingClassifier = lale.operators.make_operator(VotingClassifierImpl, _combined_schemas)
