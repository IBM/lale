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

import aif360.algorithms.preprocessing
import aif360.datasets
import lale.helpers
import lale.operators
import numpy as np
import pandas as pd

class DisparateImpactRemoverImpl:
    """Scikit-learn compatibility wrapper for the DisparateImpactRemover
    from AIF360."""

    def __init__(self, repair_level=1.0, sensitive_attribute=None):
        self._hyperparams = {
            'repair_level': repair_level,
            'sensitive_attribute': sensitive_attribute}

    def fit(self, X, y=None):
        repair_level = self._hyperparams['repair_level']
        sensitive_attribute = self._hyperparams['sensitive_attribute']
        dimpr = aif360.algorithms.preprocessing.DisparateImpactRemover(
            repair_level=repair_level, sensitive_attribute=sensitive_attribute)
        features = X.to_numpy().tolist()
        index = X.columns.to_list().index(sensitive_attribute)
        #since DisparateImpactRemover does not have separate fit and transform
        self._repairer = dimpr.Repairer(features, index, repair_level, False)
        return self

    def transform(self, X):
        features = X.to_numpy().tolist()
        repaired = self._repairer.repair(features)
        result = pd.DataFrame(features, columns=X.columns)
        return result

_input_schema_fit = {
  'description': 'Input data schema for training.',
  'type': 'object',
  'required': ['X'],
  'additionalProperties': False,
  'properties': {
    'X': {
      'description': 'Features; the outer array is over samples.',
      'type': 'array',
      'items': {'type': 'array', 'items': {'type': 'number'}}},
    'y': {
      'description': 'Target class labels; the array is over samples.'}}}

_input_schema_predict = {
  'description': 'Input data schema for transform.',
  'type': 'object',
  'required': ['X'],
  'additionalProperties': False,
  'properties': {
    'X': {
      'description': 'Features; the outer array is over samples.',
      'type': 'array',
      'items': {'type': 'array', 'items': {'type': 'number'}}}}}

_output_schema = {
    'description': 'Output data schema for predictions (reweighed features).',
    'type': 'array',
    'items': {
        'type': 'array',
        'items': {'type': 'number'}}}

_hyperparams_schema = {
  'description': 'Hyperparameter schema.',
  'allOf': [
    { 'type': 'object',
      'additionalProperties': False,
      'required': ['repair_level', 'sensitive_attribute'],
      'relevantToOptimizer': ['repair_level'],
      'properties': {
        'repair_level': {
          'description': 'Repair amount from 0 = none to 1 = full.',
          'type': 'number',
          'minimum': 0,
          'maximum': 1,
          'default': 1 },
        'sensitive_attribute': {
          'description': 'Column name of protected attribute.',
          'type': 'string' }}}]}

_combined_schemas = {
  'description': 'Combined schema for expected data and hyperparameters.',
  'documentation_url': 'https://aif360.readthedocs.io/en/latest/modules/preprocessing.html#disparate-impact-remover',
  'type': 'object',
  'tags': {
    'pre': ['~categoricals'],
    'op': ['transformer'],
    'post': []},
  'properties': {
    'input_fit': _input_schema_fit,
    'input_predict': _input_schema_predict,
    'output': _output_schema,
    'hyperparams': _hyperparams_schema } }

if __name__ == "__main__":
    lale.helpers.validate_is_schema(_combined_schemas)

DisparateImpactRemover = lale.operators.make_operator(DisparateImpactRemoverImpl, _combined_schemas)
