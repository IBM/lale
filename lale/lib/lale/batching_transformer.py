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

from lale.operators import make_operator

import numpy as np
import pandas as pd
import logging
import lale.helpers as helpers
logging.basicConfig(level=logging.INFO)

class BatchingTransformerImpl():
    def __init__(self, pipeline = None, batch_size = 32, shuffle = True, num_workers = 0):
        self.pipeline = pipeline
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers

    def fit(self, X, y = None):
        data_loader = helpers.create_data_loader(X = X, y = y, batch_size = self.batch_size)
        classes = np.unique(y)
        self.pipeline = self.pipeline.fit_with_batches(data_loader, y = classes, serialize = True)
        return self

    def transform(self, X, y = None):
        data_loader = helpers.create_data_loader(X = X, y = y, batch_size = self.batch_size)
        transformed_data = self.pipeline.transform_with_batches(data_loader, serialize = True)
        return transformed_data

_input_schema_predict = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'description': 'Input data schema for predictions.',
  'type': 'object',
  'required': ['X'],
  'additionalProperties': False,
  'properties': {
    'X': {
      'description': 'Features; the outer array is over samples.',
      'anyOf': [
        { 'type': 'array',
          'items': {'type': 'string'}},
        { 'type': 'array',
          'items': {
            'type': 'array', 'minItems': 1, 'maxItems': 1,
            'items': {'type': 'string'}}}]},
        'y': {
            'type': 'array',
            'items': {'anyOf':[{'type': 'integer'}, {'type':'string'}]}
        }
  }
}

_output_schema = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'description': 'Output data schema for transformed data.'}
  

_hyperparams_schema = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'description': 'Hyperparameter schema.',
  'allOf': [
    { 'description':
        'This first sub-object lists all constructor arguments with their '
        'types, one at a time, omitting cross-argument constraints.',
      'type': 'object',
      'additionalProperties': False,
      'relevantToOptimizer': ['batch_size'],
      'properties': {
        'batch_size':{
          'description': 'Batch size used for transform.',
          'type': 'integer',
          'default': 64,
          'minimum': 1,
          'distribution': 'uniform',
          'minimumForOptimizer': 32,
          'maximumForOptimizer': 128},
        'shuffle':{

        }  
          }}]}

_combined_schemas = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'description': 'Combined schema for expected data and hyperparameters for a transformer for'
                ' a text data transformer based on pre-trained BERT model '
                '(https://github.com/huggingface/pytorch-pretrained-BERT).',
  'type': 'object',
  'tags': {
    'pre': [],
    'op': ['transformer'],
    'post': []},
  'properties': {
    'input_predict': _input_schema_predict,
    'output': _output_schema,
    'hyperparams': _hyperparams_schema } }

if __name__ == "__main__":
    lale.helpers.validate_is_schema(_combined_schemas)

BatchingTransformer = make_operator(BatchingTransformerImpl, _combined_schemas)
