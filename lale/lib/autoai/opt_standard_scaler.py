# Copyright 2020 IBM Corporation
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

import lale.docstrings
import lale.helpers
import lale.operators
import autoai_libs.transformers.exportable

class OptStandardScalerImpl():
    def __init__(self, use_scaler_flag, num_scaler_copy, num_scaler_with_mean, num_scaler_with_std):
        self._hyperparams = {
            'use_scaler_flag': use_scaler_flag,
            'num_scaler_copy': num_scaler_copy,
            'num_scaler_with_mean': num_scaler_with_mean,
            'num_scaler_with_std': num_scaler_with_std}
        self._autoai_tfm = sklearn.decomposition.PCA(**self._hyperparams)

    def fit(self, X, y=None):
        return self._autoai_tfm.fit(X, y)

    def transform(self, X):
        return self._autoai_tfm.transform(X)

_hyperparams_schema = {
    'allOf': [{
        'description': 'This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.',
        'type': 'object',
        'additionalProperties': False,
        'required': ['use_scaler_flag', 'num_scaler_copy', 'num_scaler_with_mean', 'num_scaler_with_std'],
        'relevantToOptimizer': ['use_scaler_flag', 'num_scaler_with_mean', 'num_scaler_with_std'],
        'properties': {
            'use_scaler_flag': {
                'type': 'boolean',
                'default': True,
                'description': 'If False, return the input array unchanged.'},
            'copy': {
                'type': 'boolean',
                'default': True,
                'description': 'If False, try to avoid a copy and do inplace scaling instead.'},
            'num_scaler_with_mean': {
                'type': 'boolean',
                'default': True,
                'description': 'If True, center the data before scaling.'},
            'num_scaler_with_std': {
                'type': 'boolean',
                'default': True,
                'description': 'If True, scale the data to unit variance (or equivalently, unit standard deviation).'},
}}]}

_input_fit_schema = {
    'type': 'object',
    'required': ['X'],
    'additionalProperties': False,
    'properties': {
        'X': {'type': 'array',
              'items': {'type': 'array', 'items': {'type': 'number'}}},
        'y': {}}}

_input_transform_schema = {
    'type': 'object',
    'required': ['X'],
    'additionalProperties': False,
    'properties': {
        'X': {'type': 'array',
              'items': {'type': 'array', 'items': {'type': 'number'}}}}}

_output_transform_schema = {
    'description': 'Features; the outer array is over samples.',
    'type': 'array',
    'items': {'type': 'array', 'items': {'type': 'number'}}}

_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': """Operator from `autoai_libs`_. Acts like an optional StandardScaler_.

.. _`autoai_libs`: https://pypi.org/project/autoai-libs
.. _StandardScaler: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html""",
    'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.autoai.numpy_column_selector.html',
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

lale.docstrings.set_docstrings(OptStandardScalerImpl, _combined_schemas)

OptStandardScaler = lale.operators.make_operator(OptStandardScalerImpl, _combined_schemas)
