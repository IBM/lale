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

import sklearn.preprocessing
import lale.helpers
import lale.operators

class FunctionTransformerImpl():
    def __init__(self, func=None, inverse_func=None, validate=None, accept_sparse=False, pass_y='deprecated', check_inverse=True, kw_args=None, inv_kw_args=None):
        self._hyperparams = {
            'func': func,
            'inverse_func': inverse_func,
            'validate': validate,
            'accept_sparse': accept_sparse,
            'pass_y': pass_y,
            'check_inverse': check_inverse,
            'kw_args': kw_args,
            'inv_kw_args': inv_kw_args}
        self._sklearn_model = sklearn.preprocessing.FunctionTransformer(
            **self._hyperparams)

    def fit(self, X, y=None):
        if y is not None:
            self._sklearn_model.fit(X, y)
        else:
            self._sklearn_model.fit(X)
        return self

    def transform(self, X):
        return self._sklearn_model.transform(X)

_hyperparams_schema = {
    'allOf': [
    {   'description': 'This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters.\n',
        'type': 'object',
        'additionalProperties': False,
        'required': ['func', 'inverse_func', 'validate', 'accept_sparse', 'pass_y', 'check_inverse', 'kw_args', 'inv_kw_args'],
        'relevantToOptimizer': [],
        'properties': {
            'func': {
                'anyOf': [
                {   'laleType': 'Any'},
                {   'enum': [None]}],
                'default': None,
                'description': 'The callable to use for the transformation.'},
            'inverse_func': {
                'anyOf': [
                {   'laleType': 'Any'},
                {   'enum': [None]}],
                'default': None,
                'description': 'The callable to use for the inverse transformation.'},
            'validate': {
                'default': False,
                'description': 'Indicate that the input X array should be checked before calling ``func``.'},
            'accept_sparse': {
                'type': 'boolean',
                'default': False,
                'description': 'Indicate that func accepts a sparse matrix as input.'},
            'pass_y': {
                'anyOf': [{
                    'type': 'boolean'}, {
                    'enum': ['deprecated']}],
                'default': 'deprecated',
                'description': 'Indicate that transform should forward the y argument to the inner callable.'},
            'check_inverse': {
                'type': 'boolean',
                'default': True,
                'description': 'Whether to check that or ``func`` followed by ``inverse_func`` leads to the original inputs.'},
            'kw_args': {
                'anyOf': [
                {   'type': 'object'},
                {   'enum': [None]}],
                'default': None,
                'description': 'Dictionary of additional keyword arguments to pass to func.'},
            'inv_kw_args': {
                'anyOf': [
                {   'type': 'object'},
                {   'enum': [None]}],
                'default': None,
                'description': 'Dictionary of additional keyword arguments to pass to inverse_func.'}}},
    {   'description': 'If validate is False, then accept_sparse has no effect.',
        'anyOf': [
        {   'type': 'object',
            'properties': {'validate': {'not': {'enum': [False]}}}},
        {   'type': 'object',
            'properties': {'accept_sparse': {'enum': [False]}}}]}]}

_input_fit_schema = {
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'anyOf':[{'type': 'number'}, {'type':'string'}]}}},
        'y': {'laleType': 'Any'}}}

_input_transform_schema = {
    'type': 'object',
    'required': ['X'],
    'properties': {
        'X': {
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {
                    'anyOf':[{'type': 'number'}, {'type':'string'}]}}}}}

_output_transform_schema = {
    'type': 'array',
    'items': {'laleType': 'Any'}}

_combined_schemas = {
    '$schema': 'http://json-schema.org/draft-04/schema#',
    'description': """FunctionTransformer_ constructs a transformer from an arbitrary callable.

.. _FunctionTransformer: https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.FunctionTransformer.html
""",
    'documentation_url': 'https://scikit-learn.org/0.20/modules/generated/sklearn.preprocessing.FunctionTransformer.html',
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

lale.docstrings.set_docstrings(FunctionTransformerImpl, _combined_schemas)

FunctionTransformer = lale.operators.make_operator(FunctionTransformerImpl, _combined_schemas)
