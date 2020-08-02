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

import aif360.algorithms.postprocessing
import aif360.datasets
import aif360.metrics
import lale.datasets.data_schemas
import lale.type_checking
import pandas as pd

def dataset_to_pandas(dataset, return_only='Xy'):
    """
    Return pandas representation of the AIF360 dataset.

    Parameters
    ----------
    dataset : aif360.datasets.BinaryLabelDataset

      AIF360 dataset to convert to a pandas representation.

    return_only : 'Xy', 'X', or 'y'

      Which part of features X or labels y to convert and return.

    Returns
    -------
    result : tuple

      - item 0: pandas Dataframe or None, features X

      - item 1: pandas Series or None, labels y
    """
    if 'X' in return_only:
        X = pd.DataFrame(dataset.features, columns=dataset.feature_names)
        result_X = lale.datasets.data_schemas.add_schema(X)
        assert isinstance(result_X, pd.DataFrame), type(result_X)
    else:
        result_X = None
    if 'y' in return_only:
        y = pd.Series(dataset.labels.ravel(), name=dataset.label_names[0])
        result_y = lale.datasets.data_schemas.add_schema(y)
        assert isinstance(result_y, pd.Series), type(result_y)
    else:
        result_y = None
    return result_X, result_y

_dataset_fairness_properties: lale.type_checking.JSON_TYPE = {
    'favorable_label': {
        'description': 'Label value which is considered favorable (i.e. "positive").',
        'type': 'number'},
    'unfavorable_label': {
        'description': 'Label value which is considered unfavorable (i.e. "negative").',
        'type': 'number'},
    'protected_attribute_names': {
        'description': 'Subset of feature names for which fairness is desired.',
        'type': 'array',
        'items': {'type': 'string'}},
    'unprivileged_groups': {
        'description': 'Representation for unprivileged group.',
        'type': 'array',
        'items': {
            'description': 'Map from feature names to group-indicating values.',
            'type': 'object',
            'additionalProperties': {
                'type': 'number'}}},
    'privileged_groups': {
        'description': 'Representation for privileged group.',
        'type': 'array',
        'items': {
            'description': 'Map from feature names to group-indicating values.',
            'type': 'object',
            'additionalProperties': {
                'type': 'number'}}}}

_dataset_fairness_schema = {
    'type': 'object',
    'properties': _dataset_fairness_properties}

def dataset_fairness_info(dataset):
    """
    Inspect the AIF360 dataset and return its fairness metadata as JSON.

    Parameters
    ----------
    dataset : aif360.datasets.BinaryLabelDataset

    Returns
    -------
    result : dict

      JSON data structure with fairness information.

      - favorable_label : number

          Label value which is considered favorable (i.e. "positive").

      - unfavorable_label : number

          Label value which is considered unfavorable (i.e. "negative").

      - protected_attribute_names : array **of** items : string

          Subset of feature names for which fairness is desired.

      - unprivileged_groups : array

          Representation for unprivileged group.

          - items : dict

              Map from feature names to group-indicating values.

      - privileged_groups : array

          Representation for privileged group.

          - items : dict

              Map from feature names to group-indicating values.
    """
    def attributes_to_groups(names, value_arrays):
        result = [{}]
        for i in range(len(names)):
            next_result = []
            for d in result:
                for next_v in value_arrays[i]:
                    next_d = {**d, names[i]: next_v}
                    next_result.append(next_d)
            result = next_result
        return result
    unprivileged_groups = attributes_to_groups(
        dataset.protected_attribute_names,
        dataset.unprivileged_protected_attributes)
    privileged_groups = attributes_to_groups(
        dataset.protected_attribute_names,
        dataset.privileged_protected_attributes)
    result = {
        'favorable_label': dataset.favorable_label,
        'unfavorable_label': dataset.unfavorable_label,
        'protected_attribute_names': dataset.protected_attribute_names,
        'unprivileged_groups': unprivileged_groups,
        'privileged_groups': privileged_groups}
    lale.type_checking.validate_schema(result, _dataset_fairness_schema)
    return result

class _PandasToDatasetConverter:
    def __init__(self, favorable_label, unfavorable_label, protected_attribute_names):
        lale.type_checking.validate_schema(favorable_label,
            _dataset_fairness_properties['favorable_label'])
        self.favorable_label = favorable_label
        lale.type_checking.validate_schema(unfavorable_label,
            _dataset_fairness_properties['unfavorable_label'])
        self.unfavorable_label = unfavorable_label
        lale.type_checking.validate_schema(protected_attribute_names,
            _dataset_fairness_properties['protected_attribute_names'])
        self.protected_attribute_names = protected_attribute_names

    def __call__(self, X, y):
        assert isinstance(X, pd.DataFrame), type(X)
        assert isinstance(y, pd.Series), type(y)
        assert X.shape[0] == y.shape[0], f'X.shape {X.shape}, y.shape {y.shape}'
        df = pd.concat([X, y], axis=1)
        assert not df.isna().any().any(), f'df\n{df}\nX\n{X}\ny\n{y}'
        label_names = [y.name]
        result = aif360.datasets.BinaryLabelDataset(
            favorable_label=self.favorable_label,
            unfavorable_label=self.unfavorable_label,
            protected_attribute_names=self.protected_attribute_names,
            df=df,
            label_names=label_names)
        return result

def _ensure_series(data, index, dtype, name):
    if isinstance(data, pd.Series):
        return data
    result = pd.Series(data=data, index=index, dtype=dtype, name=name)
    return result

class _BinaryLabelScorer:
    def __init__(self, metric, favorable_label, unfavorable_label, protected_attribute_names, unprivileged_groups, privileged_groups):
        assert hasattr(aif360.metrics.BinaryLabelDatasetMetric, metric)
        self.metric = metric
        self.fairness_info = {
            'favorable_label': favorable_label,
            'unfavorable_label': unfavorable_label,
            'protected_attribute_names': protected_attribute_names,
            'unprivileged_groups': unprivileged_groups,
            'privileged_groups': privileged_groups}
        lale.type_checking.validate_schema(
            self.fairness_info, _dataset_fairness_schema)
        self.pandas_to_dataset = _PandasToDatasetConverter(
            favorable_label, unfavorable_label, protected_attribute_names)

    def __call__(self, estimator, X, y):
        predicted = estimator.predict(X)
        y_pred = _ensure_series(predicted, X.index, y.dtype, y.name)
        dataset_pred = self.pandas_to_dataset(X, y_pred)
        fairness_metrics = aif360.metrics.BinaryLabelDatasetMetric(
            dataset_pred,
            self.fairness_info['unprivileged_groups'],
            self.fairness_info['privileged_groups'])
        method = getattr(fairness_metrics, self.metric)
        result = method()
        return result

def disparate_impact(favorable_label, unfavorable_label, protected_attribute_names, unprivileged_groups, privileged_groups):
    """
    Make a scikit-learn compatible scorer given the fairness info.

    Parameters
    ----------
    favorable_label : number

      Label value which is considered favorable (i.e. "positive").

    unfavorable_label : number

      Label value which is considered unfavorable (i.e. "negative").

    protected_attribute_names : array **of** items : string

      Subset of feature names for which fairness is desired.

    unprivileged_groups : array

      Representation for unprivileged group.

      - items : dict

          Map from feature names to group-indicating values.

    privileged_groups : array

      Representation for privileged group.

      - items : dict

          Map from feature names to group-indicating values.

    Returns
    -------
    result : callable

      Scorer that takes three arguments (estimator, X, y) and returns score.
    """
    return _BinaryLabelScorer('disparate_impact', favorable_label, unfavorable_label, protected_attribute_names, unprivileged_groups, privileged_groups)

def statistical_parity_difference(favorable_label, unfavorable_label, protected_attribute_names, unprivileged_groups, privileged_groups):
    """
    Make a scikit-learn compatible scorer given the fairness info.

    Parameters
    ----------
    favorable_label : number

      Label value which is considered favorable (i.e. "positive").

    unfavorable_label : number

      Label value which is considered unfavorable (i.e. "negative").

    protected_attribute_names : array **of** items : string

      Subset of feature names for which fairness is desired.

    unprivileged_groups : array

      Representation for unprivileged group.

      - items : dict

          Map from feature names to group-indicating values.

    privileged_groups : array

      Representation for privileged group.

      - items : dict

          Map from feature names to group-indicating values.

    Returns
    -------
    result : callable

      Scorer that takes three arguments (estimator, X, y) and returns score.
    """
    return _BinaryLabelScorer('statistical_parity_difference', favorable_label, unfavorable_label, protected_attribute_names, unprivileged_groups, privileged_groups)

_postprocessing_base_hyperparams = {
    'estimator': {
        'description': 'Nested supervised learning operator for which to mitigate fairness.',
        'laleType': 'operator'},
    'favorable_label': _dataset_fairness_properties['favorable_label'],
    'unfavorable_label': _dataset_fairness_properties['unfavorable_label'],
    'protected_attribute_names': _dataset_fairness_properties['protected_attribute_names']}

class _BasePostprocessingImpl:
    def __init__(self, mitigator, estimator, favorable_label, unfavorable_label, protected_attribute_names):
        self.mitigator = mitigator
        self.estimator = estimator
        self.pandas_to_dataset = _PandasToDatasetConverter(
            favorable_label, unfavorable_label, protected_attribute_names)
        self.y_dtype = None
        self.y_name = None

    def fit(self, X, y):
        self.y_dtype = y.dtype
        self.y_name = y.name
        y_true = y
        self.estimator = self.estimator.fit(X, y_true)
        predicted = self.estimator.predict(X)
        y_pred = _ensure_series(predicted, X.index, self.y_dtype, self.y_name)
        dataset_true = self.pandas_to_dataset(X, y_true)
        dataset_pred = self.pandas_to_dataset(X, y_pred)
        self.mitigator = self.mitigator.fit(dataset_true, dataset_pred)
        return self

    def predict(self, X):
        predicted = self.estimator.predict(X)
        y_pred = _ensure_series(predicted, X.index, self.y_dtype, self.y_name)
        dataset_pred = self.pandas_to_dataset(X, y_pred)
        dataset_out = self.mitigator.predict(dataset_pred)
        _, y_out = dataset_to_pandas(dataset_out, return_only='y')
        return y_out

_numeric_supervised_input_fit_schema = {
    'type': 'object',
    'required': ['X', 'y'],
    'additionalProperties': False,
    'properties': {
        'X': {
            'description': 'Features; the outer array is over samples.',
            'type': 'array',
            'items': {
                'type': 'array',
                'items': {'type': 'number'}}},
        'y': {
            'description': 'Target class labels; the array is over samples.',
            'type': 'array',
            'items': {'type': 'number'}}}}

_numeric_input_predict_schema = {
    'type': 'object',
    'required': ['X'],
    'additionalProperties': False,
    'properties': {
        'X': {
            'description': 'Features; the outer array is over samples.',
            'type': 'array',
            'items': {'type': 'array', 'items': {'type': 'number'}}}}}

_numeric_output_predict_schema = {
    'description': 'Predicted class label per sample.',
    'type': 'array', 'items': {'type': 'number'}}
