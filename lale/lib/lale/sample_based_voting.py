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

import lale.helpers
import lale.operators
import pandas as pd
import numpy as np

class SampleBasedVotingImpl():
    def __init__(self, hyperparams=None):
        self._hyperparams = hyperparams
        self.end_index_list = None

    # def fit(self, X, y = None):
    #     result = NoOpImpl(self._hyperparams)
    #     if isinstance(X, pd.DataFrame):
    #         result._feature_names = list(X.columns)
    #     else:
    #         #This assumes X is a 2d array which is consistent with its schema.
    #         result._feature_names = ['x%d' % i for i in range(X.shape[1])] 
    #     return result

    def set_meta_data(self, meta_data_dict):
        if 'end_index_list' in meta_data_dict.keys():
            self.end_index_list = meta_data_dict['end_index_list']

    def transform(self, X, end_index_list = None):
        """Treat the input as labels and use the end_index_list to produce
        labels using voting. Note that here, X contains the label and no y is accepted.
        
        Parameters
        ----------
        X : [type]
            X is actually the labels from the previous component in a pipeline.
        end_index_list : [type], optional
            For each output label to be produced, end_index_list is supposed to contain 
            the index of the last element corresponding to the original input.
        
        Returns
        -------
        [type]
            [description]
        """
        if end_index_list is None:
            end_index_list = self.end_index_list # in case the end_index_list was set as meta_data

        if end_index_list is None:
            return X
        else:
            voted_labels = []
            prev_index = 0
            if not isinstance(X, np.ndarray):
                if isinstance(X, list):
                    X = np.array(X)
                elif isinstance(X, pd.dataframe):
                    X = X.as_matrix()
            for index in end_index_list:
                labels = X[prev_index:index]
                (values,counts) = np.unique(labels,return_counts=True)
                ind=np.argmax(counts) #If two labels are in majority, this will pick the first one.
                voted_labels.append(ind)
            return np.array(voted_labels)
    

_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Hyperparameter schema for the NoOp, which is a place-holder for no operation.',
    'allOf': [
    {   'description': 'This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters',
        'type': 'object',
        'additionalProperties': False,
        'relevantToOptimizer': [],
        'properties': {}}]}

_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Input data schema for training NoOp.',
    'type': 'object',
    'required': ['X'],
    'additionalProperties': False,
    'properties': {
        'X': {
            'description': 'Features; the outer array is over samples.',
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'type': 'number'}}}}}

_input_transform_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Input data schema for transformations using NoOp.',
    'type': 'object',
    'required': ['X'],
    'additionalProperties': False,
    'properties': {
        'X': {
            'description': 'Features; the outer array is over samples.',
            'type': 'array'}}}

_output_transform_schema = {
  'type': 'array',
  'items': {'laleType': 'Any'}}

_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.sample_based_voting.html',
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

if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)

SampleBasedVoting = lale.operators.make_operator(SampleBasedVotingImpl, _combined_schemas)
