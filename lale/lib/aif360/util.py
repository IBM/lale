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
import pandas as pd
import lale.helpers

def dataset_to_pandas(dataset):
    X = pd.DataFrame(dataset.features, columns=dataset.feature_names)
    y = pd.Series(dataset.labels.ravel(), name=dataset.label_names[0])
    result_X = lale.datasets.data_schemas.add_schema(X)
    result_y = lale.datasets.data_schemas.add_schema(y)
    assert isinstance(result_X, pd.DataFrame), type(result_X)
    assert isinstance(result_y, pd.Series), type(result_y)
    return result_X, result_y

class PandasToDatasetConverter:
    def __init__(self, favorable_label, unfavorable_label, protected_attribute_names):
        self.favorable_label = favorable_label
        self.unfavorable_label = unfavorable_label
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

def _make_series(data, index, dtype, name):
    if isinstance(data, pd.Series):
        return data
    result = pd.Series(data=data, index=index, dtype=dtype, name=name)
    return result

def dataset_fairness_info(dataset):
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
    return result
    
class BinaryLabelScorer:
    def __init__(self, metric, favorable_label, unfavorable_label, protected_attribute_names, unprivileged_groups, privileged_groups):
        assert hasattr(aif360.metrics.BinaryLabelDatasetMetric, metric)
        self.metric = metric
        self.unprivileged_groups = unprivileged_groups
        self.privileged_groups = privileged_groups
        self.pandas_to_dataset = PandasToDatasetConverter(
            favorable_label, unfavorable_label, protected_attribute_names)

    def __call__(self, estimator, X, y):
        predicted = estimator.predict(X)
        y_pred = _make_series(predicted, X.index, y.dtype, y.name)
        dataset_pred = self.pandas_to_dataset(X, y_pred)
        fairness_metrics = aif360.metrics.BinaryLabelDatasetMetric(
            dataset_pred, self.unprivileged_groups, self.privileged_groups)
        method = getattr(fairness_metrics, self.metric)
        result = method()
        return result

_postprocessing_base_hyperparams = {
    'estimator': {
        'description': 'Supervised learning sub-pipeline for which to mitigate fairness.',
        'laleType': 'estimator'},
    'favorable_label': {
        'description': 'Label value which is considered favorable (i.e. "positive").',
        'type': 'number'},
    'unfavorable_label': {
        'description': 'Label value which is considered unfavorable (i.e. "negative").',
        'type': 'number'},
    'protected_attribute_names': {
        'description': 'Subset of feature names for which fairness is desired.',
        'type': 'array',
        'items': {'type': 'string'}}}

class _BasePostprocessingImpl:
    def __init__(self, mitigator, estimator, favorable_label, unfavorable_label, protected_attribute_names):
        self.mitigator = mitigator
        self.estimator = estimator
        self.pandas_to_dataset = PandasToDatasetConverter(
            favorable_label, unfavorable_label, protected_attribute_names)
        self.y_dtype = None
        self.y_name = None

    def fit(self, X, y):
        self.y_dtype = y.dtype
        self.y_name = y.name
        y_true = y
        self.estimator = self.estimator.fit(X, y_true)
        predicted = self.estimator.predict(X)
        y_pred = _make_series(predicted, X.index, self.y_dtype, self.y_name)
        dataset_true = self.pandas_to_dataset(X, y_true)
        dataset_pred = self.pandas_to_dataset(X, y_pred)
        self.mitigator = self.mitigator.fit(dataset_true, dataset_pred)
        return self

    def predict(self, X):
        predicted = self.estimator.predict(X)
        y_pred = _make_series(predicted, X.index, self.y_dtype, self.y_name)
        dataset_pred = self.pandas_to_dataset(X, y_pred)
        dataset_out = self.mitigator.predict(dataset_pred)
        _, y_out = dataset_to_pandas(dataset_out)
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
