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
import numpy as np
import pandas as pd
import scipy.sparse
import jsonsubschema
import torch
import lale.pretty_print

class ConcatFeaturesImpl():
    """Transformer to concatenate input datasets. 

    This transformer concatenates the input datasets column-wise.

    Examples
    --------
    >>> A = [ [11, 12, 13],
              [21, 22, 23],
              [31, 32, 33] ]
    >>> B = [ [14, 15],
              [24, 25],
              [34, 35] ]
    >>> trainable_cf = ConcatFeatures()
    >>> trained_cf = trainable_cf.fit(X = [A, B])
    >>> trained_cf.transform([A, B])
        [ [11, 12, 13, 14, 15],
          [21, 22, 23, 24, 25],
          [31, 32, 33, 34, 35] ]
    """

    def __init__(self):
        pass

    def transform(self, X):
        """Transform the list of datasets to one single dataset by concatenating column-wise.
        
        Parameters
        ----------
        X : list
            List of datasets to be concatenated.
        
        Returns
        -------
        [type]
            [description]
        """
        np_datasets = []
        #Preprocess the datasets to convert them to 2-d numpy arrays
        for dataset in X:
            if isinstance(dataset, pd.DataFrame) or isinstance(dataset, pd.Series):
                np_dataset = dataset.values
            elif isinstance(dataset, scipy.sparse.csr_matrix):
                np_dataset = dataset.toarray()
            elif isinstance(dataset, torch.Tensor):
                np_dataset = dataset.detach().cpu().numpy()
            else:
                np_dataset = dataset
            if hasattr(np_dataset, 'shape'):
                if len(np_dataset.shape) == 1: #To handle numpy column vectors
                    np_dataset = np.reshape(np_dataset, (np_dataset.shape[0], 1))
            np_datasets.append(np_dataset)
            
        result = np.concatenate(np_datasets, axis=1)
        return result

    def transform_schema(self, s_X):
        min_cols, max_cols, elem_schema = 0, 0, None
        def join_schemas(s_a, s_b):
            if s_a is None:
                return s_b
            s_a = lale.helpers.dict_without(s_a, 'description')
            s_b = lale.helpers.dict_without(s_b, 'description')
            if jsonsubschema.isSubschema(s_a, s_b):
                return s_b
            if jsonsubschema.isSubschema(s_b, s_a):
                return s_a
            return jsonsubschema.joinSchemas(s_a, s_b)
        def add_ranges(min_a, max_a, min_b, max_b):
            min_ab = min_a + min_b
            if max_a == 'unbounded' or max_b == 'unbounded':
                max_ab = 'unbounded'
            else:
                max_ab = max_a + max_b
            return min_ab, max_ab
        for s_dataset in s_X['items']:
            assert 'items' in s_dataset, lale.pretty_print.to_string(s_dataset)
            s_rows = s_dataset['items']
            if 'type' in s_rows and 'array' == s_rows['type']:
                s_cols = s_rows['items']
                if isinstance(s_cols, dict):
                    min_c = s_rows['minItems'] if 'minItems' in s_rows else 1
                    max_c = s_rows['maxItems'] if 'maxItems' in s_rows else 'unbounded'
                    elem_schema = join_schemas(elem_schema, s_cols)
                else:
                    min_c, max_c = len(s_cols), len(s_cols)
                    for s_col in s_cols:
                        elem_schema = join_schemas(elem_schema, s_col)
                min_cols, max_cols = add_ranges(min_cols,max_cols,min_c,max_c)
            else:
                elem_schema = join_schemas(elem_schema, s_rows)
                min_cols, max_cols = add_ranges(min_cols, max_cols, 1, 1)
        s_result = {
            '$schema': 'http://json-schema.org/draft-04/schema#',
            'type': 'array',
            'items': {
                'type': 'array',
                'minItems': min_cols,
                'items': elem_schema}}
        if max_cols != 'unbounded':
            s_result['items']['maxItems'] = max_cols
        lale.helpers.validate_is_schema(s_result)
        return s_result
    
_hyperparams_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Hyperparameter schema for the ConcatFeatures operator.\n',
    'allOf': [{
        'description': 'This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.',
        'type': 'object',
        'additionalProperties': False,
        'relevantToOptimizer': [],
        'properties': {}}]}

_input_fit_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Input data schema for training the ConcatFeatures operator. As this operator does not actually require training, this is the same as the input schema for making predictions.',
    'type': 'object',
    'required': ['X'],
    'additionalProperties': True,
    'properties': {
        'X': {
            'description': 'Outermost array dimension is over datasets.',
            'type': 'array',
            'items': {
                'description': 'Middle array dimension is over samples (aka rows).',
                'type': 'array',
                'items': {
                    'description': 'Innermost array dimension is over features (aka columns).',
                    'anyOf': [{
                        'type': 'array',
                        'items': {
                            'type': 'number'},
                    }, {
                        'type': 'number'}]}}}}}

_input_predict_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Input data schema for making predictions using the ConcatFeatures operator.',
    'type': 'object',
    'required': ['X'],
    'additionalProperties': False,
    'properties': {
        'X': {
            'description': 'Outermost array dimension is over datasets.',
            'type': 'array',
            'items': {
                'description': 'Middle array dimension is over samples (aka rows).',
                'type': 'array',
                'items': {
                    'description': 'Innermost array dimension is over features (aka columns).',
                    'anyOf': [{
                        'type': 'array',
                        'items': {
                            'type': 'number'},
                    }, {
                        'type': 'number'}]}}}}}

_output_schema = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Output data schema for transformed data using the ConcatFeatures operator.',
    'type': 'array',
    'items': {
        'type': 'array',
        'items': {
            'type': 'number'}}}

_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': 'Combined schema for expected data and hyperparameters.',
    'documentation_url': 'https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.concat_features.html',
    'type': 'object',
    'tags': {
        'pre': [],
        'op': ['transformer'],
        'post': []},
    'properties': {
        'hyperparams': _hyperparams_schema,
        'input_fit': _input_fit_schema,
        'input_predict': _input_predict_schema,
        'output': _output_schema }}

if (__name__ == '__main__'):
    lale.helpers.validate_is_schema(_combined_schemas)

ConcatFeatures = lale.operators.make_operator(ConcatFeaturesImpl, _combined_schemas)
