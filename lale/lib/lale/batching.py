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

class BatchingImpl():
  """BatchingTransformer trains the given pipeline using batches.
  The batch_size is used across all steps of the pipeline, serializing
  the intermediate outputs if specified.

  Parameters
  ----------
  operator : lale.operators.Pipeline
      A Lale pipeline object that needs to be trained/used for transform or predictions,
      by default None
  batch_size : int, optional
      Batch size to be used for all steps in the pipeline, by default 32
  shuffle : bool, optional
      Shuffle dataset before batching or not, by default True
  num_workers : int, optional
      Number of workers for pytorch dataloader, by default 0
  serialize_intermediate : bool, optional
      Serialize the intermediate transform outputs or not, by default True

  Examples
  --------
  >>> from lale.lib.sklearn import MinMaxScaler, MLPClassifier
  >>> pipeline = NoOp() >> Batching(
    operator = MinMaxScaler() >> MLPClassifier(random_state=42), batch_size = 112)
  >>> trained = pipeline.fit(X_train, y_train)
  >>> predictions = trained.predict(X_test)

  """
  def __init__(self, operator = None, batch_size = 32, shuffle = True, num_workers = 0, serialize_intermediate=True):    
    self.operator = operator
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.num_workers = num_workers
    self.serialize_intermediate = serialize_intermediate

  def fit(self, X, y = None):
    if self.operator is None:
      raise ValueError("The pipeline object can't be None at the time of fit.")
    data_loader = helpers.create_data_loader(X = X, y = y, batch_size = self.batch_size)
    classes = np.unique(y)
    self.operator = self.operator.fit_with_batches(data_loader, y = classes, serialize = self.serialize_intermediate)
    return self

  def transform(self, X, y = None):
    data_loader = helpers.create_data_loader(X = X, y = y, batch_size = self.batch_size)
    transformed_data = self.operator.transform_with_batches(data_loader, serialize = self.serialize_intermediate)
    return transformed_data

  def predict(self, X, y = None):
    return self.transform(X, y)

_input_schema_fit = {
  '$schema': 'http://json-schema.org/draft-04/schema#',
  'description': 'Input data schema for fit.',
  'type': 'object',
  'required': ['X'],
  'additionalProperties': False,
  'properties': {
    'X': {
      'description': 'Features; the outer array is over samples.',
      'anyOf': [
        { 'type': 'array',
          'items': {'type': 'number'}},
        { 'type': 'array',
          'items': {
            'type': 'array', 'minItems': 1,
            'items': {'type': 'number'}}}]},
        'y': {
            'type': 'array',
            'items': {'anyOf':[{'type': 'integer'}, {'type':'number'}]}
        }
  }
}

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
          'items': {'type': 'number'}},
        { 'type': 'array',
          'items': {
            'type': 'array', 'minItems': 1,
            'items': {'type': 'number'}}}]},
        'y': {
            'type': 'array',
            'items': {'anyOf':[{'type': 'integer'}, {'type':'number'}]}
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
        'operator':{
          'description':'A lale operator object to be used inside of batching',
        },
        'batch_size':{
          'description': 'Batch size used for transform.',
          'type': 'integer',
          'default': 64,
          'minimum': 1,
          'distribution': 'uniform',
          'minimumForOptimizer': 32,
          'maximumForOptimizer': 128},
        'shuffle':{
          'type':'boolean',
          'default': False,
          'description': 'Shuffle dataset before batching or not.'
          },
        'num_workers':{
          'type':'integer',
          'default':0,
          'description': 'Number of workers for pytorch dataloader.'
          },
        'serialize_intermediate':{
          'type':'boolean',
          'default': True,
          'description': 'Serialize the intermediate transform outputs or not.'
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
    'input_fit': _input_schema_fit,
    'input_predict': _input_schema_predict,
    'output': _output_schema,
    'hyperparams': _hyperparams_schema } }

if __name__ == "__main__":
    helpers.validate_is_schema(_combined_schemas)

Batching = make_operator(BatchingImpl, _combined_schemas)
