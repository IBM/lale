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

import sklearn.cluster
import lale.docstrings
import lale.operators
import numpy as np

class FeatureAgglomerationImpl():
    def __init__(self, **hyperparams):
        self._hyperparams = hyperparams
        self._wrapped_model = sklearn.cluster.FeatureAgglomeration(**self._hyperparams)

    def fit(self, X, y=None):
        self._wrapped_model.fit(X, y)
        return self

    def transform(self, X):
        return self._wrapped_model.transform(X)

_hyperparams_schema = {
    'description': 'Agglomerate features.',
    'allOf': [{
        'type': 'object',
        'required': ['memory', 'compute_full_tree', 'pooling_func'],
        'relevantToOptimizer': ['n_clusters', 'affinity', 'compute_full_tree', 'linkage'],
        'additionalProperties': False,
        'properties': {
            'n_clusters': {
                'anyOf': [
                {   'type': 'integer',
                    'minimumForOptimizer': 2,
                    'maximumForOptimizer': 8},
                {   'enum': [None]}],
                'default': 2,
                'description': 'The number of clusters to find.'},
            'affinity': {
                'anyOf': [
                {   'enum': ['euclidean', 'l1', 'l2', 'manhattan', 'cosine',
                    'precomputed']},
                {   'forOptimizer': False,
                    'laleType': 'callable' }],
                'default': 'euclidean',
                'description': 'Metric used to compute the linkage. Can be "euclidean", "l1", "l2",'},
            'memory': {
                'anyOf': [{
                    'type': 'string'}, {
                    'forOptimizer':False,
                    'type': 'object' }, { #object with the joblib.Memory interface
                    'enum':[None]}],  
                'default': None,             
                'description': 'Used to cache the output of the computation of the tree.'},
            'connectivity': {
                'anyOf': [
                {   'type': 'array',
                    'items': {
                        'type': 'array', 'items': { 'type': 'number' }}},
                {   'laleType': 'callable',
                    'forOptimizer': False,
                    'description': 'A callable that transforms the data into a connectivity matrix, such as derived from kneighbors_graph.'},
                {'enum': [None]}],
                'default': None,
                'description': 'Connectivity matrix. Defines for each feature the neighboring features following a given structure of the data.'},
            'compute_full_tree': {
                'anyOf':[{
                    'type': 'boolean'
                },{
                    'enum':['auto']
                }],
                'default': 'auto',
                'description': 'Stop early the construction of the tree at n_clusters. This is'},
            'linkage': {
                'enum': ['ward', 'complete', 'average', 'single'],
                'default': 'ward',
                'description': 'Which linkage criterion to use. The linkage criterion determines which'},
            'pooling_func': {
                'description': 'This combines the values of agglomerated features into a single',
                'laleType': 'callable',
                'default': np.mean},
        }},
    {   'description': 'affinity, if linkage is "ward", only "euclidean" is accepted',
          'anyOf': [
            { 'type': 'object',
              'properties': {'affinity': {'enum': ['euclidean']}}},
            { 'type': 'object',
              'properties': {
                'linkage':{'not': {'enum': ['ward']}}}}]},
    {   'description': 'n_clusters must be None if distance_threshold is not None.',
        'anyOf': [
        { 'type': 'object',
            'properties': {'n_clusters': {'enum': [None]}}},
        { 'type': 'object',
            'properties': {
            'distance_threshold': {'enum': [None]}}}]},
    {   'description': 'compute_full_tree must be True if distance_threshold is not None.',
        'anyOf': [
        { 'type': 'object',
            'properties': {'compute_full_tree': {'enum': [True]}}},
        { 'type': 'object',
            'properties': {
            'distance_threshold': {'enum': [None]}}}]}]}

_input_fit_schema = {
    'description': 'Fit the hierarchical clustering on the data',
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
            'description': 'The data'},
        'y': {'description': 'Ignored'},
}}

_input_transform_schema = {
    'description': 'Transform a new matrix using the built clustering',
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
            'description': 'A M by N array of M observations in N dimensions or a length'},
    },
}
_output_transform_schema = {
    'description': 'The pooled values for each feature cluster.',
    'type': 'array',
    'items': {
        'type': 'array',
        'items': {
            'type': 'number'},
    },
}
_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': """`Feature agglomeration`_ transformer from scikit-learn.

.. _`Feature agglomeration`: https://scikit-learn.org/0.20/modules/generated/sklearn.cluster.FeatureAgglomeration.html#sklearn-cluster-featureagglomeration
""",
    'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.sklearn.feature_agglomeration.html',
    'import_from': 'sklearn.cluster',
    'type': 'object',
    'tags': {
        'pre': [],
        'op': ['transformer'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_transform': _input_transform_schema,
        'output_transform': _output_transform_schema}}

FeatureAgglomeration : lale.operators.IndividualOp
FeatureAgglomeration = lale.operators.make_operator(FeatureAgglomerationImpl, _combined_schemas)

if sklearn.__version__ >= '0.21':
    # old: https://scikit-learn.org/0.20/modules/generated/sklearn.cluster.FeatureAgglomeration.html
    # new: https://scikit-learn.org/0.23/modules/generated/sklearn.cluster.FeatureAgglomeration.html
    from lale.schemas import AnyOf, Float, Null
    FeatureAgglomeration = FeatureAgglomeration.customize_schema(
        distance_threshold=AnyOf(
            types=[Float(), Null()],
            desc='The linkage distance threshold above which, clusters will not be merged.',
            default=None))

lale.docstrings.set_docstrings(FeatureAgglomerationImpl, FeatureAgglomeration._schemas)
