# Copyright 2019-2022 IBM Corporation
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

import logging
from typing import Optional, Tuple

import aif360.algorithms.postprocessing
import aif360.datasets
import numpy as np
import pandas as pd

import lale.datasets.data_schemas
import lale.datasets.openml
import lale.lib.lale
from lale.operators import TrainablePipeline
from lale.type_checking import JSON_TYPE, validate_schema_directly

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def dataset_to_pandas(
    dataset, return_only="Xy"
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
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
    if "X" in return_only:
        X = pd.DataFrame(dataset.features, columns=dataset.feature_names)
        result_X = lale.datasets.data_schemas.add_schema(X)
        assert isinstance(result_X, pd.DataFrame), type(result_X)
    else:
        result_X = None
    if "y" in return_only:
        y = pd.Series(dataset.labels.ravel(), name=dataset.label_names[0])
        result_y = lale.datasets.data_schemas.add_schema(y)
        assert isinstance(result_y, pd.Series), type(result_y)
    else:
        result_y = None
    return result_X, result_y


_categorical_fairness_properties: JSON_TYPE = {
    "favorable_labels": {
        "description": 'Label values which are considered favorable (i.e. "positive").',
        "type": "array",
        "minItems": 1,
        "items": {
            "anyOf": [
                {"description": "Literal value.", "type": "string"},
                {"description": "Numerical value.", "type": "number"},
                {
                    "description": "Numeric range [a,b] from a to b inclusive.",
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "number"},
                },
            ]
        },
    },
    "protected_attributes": {
        "description": "Features for which fairness is desired.",
        "type": "array",
        "minItems": 1,
        "items": {
            "type": "object",
            "required": ["feature", "reference_group"],
            "properties": {
                "feature": {
                    "description": "Column name or column index.",
                    "anyOf": [{"type": "string"}, {"type": "integer"}],
                },
                "reference_group": {
                    "description": "Values or ranges that indicate being a member of the privileged group.",
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "anyOf": [
                            {"description": "Literal value.", "type": "string"},
                            {"description": "Numerical value.", "type": "number"},
                            {
                                "description": "Numeric range [a,b] from a to b inclusive.",
                                "type": "array",
                                "minItems": 2,
                                "maxItems": 2,
                                "items": {"type": "number"},
                            },
                        ]
                    },
                },
                "monitored_group": {
                    "description": "Values or ranges that indicate being a member of the unprivileged group.",
                    "anyOf": [
                        {
                            "description": "If `monitored_group` is not explicitly specified, consider any values not captured by `reference_group` as monitored.",
                            "enum": [None],
                        },
                        {
                            "type": "array",
                            "minItems": 1,
                            "items": {
                                "anyOf": [
                                    {"description": "Literal value.", "type": "string"},
                                    {
                                        "description": "Numerical value.",
                                        "type": "number",
                                    },
                                    {
                                        "description": "Numeric range [a,b] from a to b inclusive.",
                                        "type": "array",
                                        "minItems": 2,
                                        "maxItems": 2,
                                        "items": {"type": "number"},
                                    },
                                ]
                            },
                        },
                    ],
                    "default": None,
                },
            },
        },
    },
    "unfavorable_labels": {
        "description": 'Label values which are considered unfavorable (i.e. "negative").',
        "anyOf": [
            {
                "description": "If `unfavorable_labels` is not explicitly specified, consider any labels not captured by `favorable_labels` as unfavorable.",
                "enum": [None],
            },
            {
                "type": "array",
                "minItems": 1,
                "items": {
                    "anyOf": [
                        {"description": "Literal value.", "type": "string"},
                        {"description": "Numerical value.", "type": "number"},
                        {
                            "description": "Numeric range [a,b] from a to b inclusive.",
                            "type": "array",
                            "minItems": 2,
                            "maxItems": 2,
                            "items": {"type": "number"},
                        },
                    ],
                },
            },
        ],
        "default": None,
    },
}

FAIRNESS_INFO_SCHEMA = {
    "type": "object",
    "properties": _categorical_fairness_properties,
}


def _validate_fairness_info(
    favorable_labels, protected_attributes, unfavorable_labels, check_schema
):
    if check_schema:
        validate_schema_directly(
            {
                "favorable_labels": favorable_labels,
                "protected_attributes": protected_attributes,
                "unfavorable_labels": unfavorable_labels,
            },
            FAIRNESS_INFO_SCHEMA,
        )

    def _check_ranges(base_name, name, groups):
        for group in groups:
            if isinstance(group, list):
                if group[0] > group[1]:
                    if base_name is None:
                        logger.warning(f"range {group} in {name} has min>max")
                    else:
                        logger.warning(
                            f"range {group} in {name} of feature '{base_name}' has min>max"
                        )

    def _check_overlaps(base_name, name1, groups1, name2, groups2):
        for g1 in groups1:
            for g2 in groups2:
                overlap = False
                if isinstance(g1, list):
                    if isinstance(g2, list):
                        overlap = g1[0] <= g2[0] <= g1[1] or g1[0] <= g2[1] <= g1[1]
                    else:
                        overlap = g1[0] <= g2 <= g1[1]
                else:
                    if isinstance(g2, list):
                        overlap = g2[0] <= g1 <= g2[1]
                    else:
                        overlap = g1 == g2
                if overlap:
                    s1 = f"'{g1}'" if isinstance(g1, str) else str(g1)
                    s2 = f"'{g2}'" if isinstance(g2, str) else str(g2)
                    if base_name is None:
                        logger.warning(
                            f"overlap between {name1} and {name2} on {s1} and {s2}"
                        )
                    else:
                        logger.warning(
                            f"overlap between {name1} and {name2} of feature '{base_name}' on {s1} and {s2}"
                        )

    _check_ranges(None, "favorable labels", favorable_labels)
    if unfavorable_labels is not None:
        _check_ranges(None, "unfavorable labels", unfavorable_labels)
        _check_overlaps(
            None,
            "favorable labels",
            favorable_labels,
            "unfavorable labels",
            unfavorable_labels,
        )
    for attr in protected_attributes:
        base_name = attr["feature"]
        reference = attr["reference_group"]
        _check_ranges(base_name, "reference group", reference)
        monitored = attr.get("monitored_group", None)
        if monitored is not None:
            _check_ranges(base_name, "monitored group", monitored)
            _check_overlaps(
                base_name, "reference group", reference, "monitored group", monitored
            )


class _PandasToDatasetConverter:
    def __init__(self, favorable_label, unfavorable_label, protected_attribute_names):
        self.favorable_label = favorable_label
        self.unfavorable_label = unfavorable_label
        self.protected_attribute_names = protected_attribute_names

    def convert(self, X, y, probas=None):
        assert isinstance(X, pd.DataFrame), type(X)
        assert isinstance(y, pd.Series), type(y)
        assert X.shape[0] == y.shape[0], f"X.shape {X.shape}, y.shape {y.shape}"
        assert not X.isna().any().any(), f"X\n{X}\n"
        assert not y.isna().any().any(), f"y\n{X}\n"
        y_reindexed = pd.Series(data=y.values, index=X.index, name=y.name)
        df = pd.concat([X, y_reindexed], axis=1)
        assert df.shape[0] == X.shape[0], f"df.shape {df.shape}, X.shape {X.shape}"
        assert not df.isna().any().any(), f"df\n{df}\nX\n{X}\ny\n{y}"
        label_names = [y.name]
        result = aif360.datasets.BinaryLabelDataset(
            favorable_label=self.favorable_label,
            unfavorable_label=self.unfavorable_label,
            protected_attribute_names=self.protected_attribute_names,
            df=df,
            label_names=label_names,
        )
        if probas is not None:
            pos_ind = 1  # TODO: is this always the case?
            result.scores = probas[:, pos_ind].reshape(-1, 1)
        return result


def _ensure_str(str_or_int):
    return f"f{str_or_int}" if isinstance(str_or_int, int) else str_or_int


def _ndarray_to_series(data, name, index=None, dtype=None) -> pd.Series:
    if isinstance(data, pd.Series):
        return data
    result = pd.Series(data=data, index=index, dtype=dtype, name=_ensure_str(name))
    schema = getattr(data, "json_schema", None)
    if schema is not None:
        result = lale.datasets.data_schemas.add_schema(result, schema)
    return result


def _ndarray_to_dataframe(array) -> pd.DataFrame:
    assert len(array.shape) == 2
    column_names = None
    schema = getattr(array, "json_schema", None)
    if schema is not None:
        column_schemas = schema.get("items", {}).get("items", None)
        if isinstance(column_schemas, list):
            column_names = [s.get("description", None) for s in column_schemas]
    if column_names is None or None in column_names:
        column_names = [_ensure_str(i) for i in range(array.shape[1])]
    result = pd.DataFrame(array, columns=column_names)
    if schema is not None:
        result = lale.datasets.data_schemas.add_schema(result, schema)
    return result


class _BaseInEstimatorImpl:
    def __init__(
        self,
        *,
        favorable_labels,
        protected_attributes,
        unfavorable_labels,
        redact,
        preparation,
        mitigator,
    ):
        _validate_fairness_info(
            favorable_labels, protected_attributes, unfavorable_labels, False
        )
        self.favorable_labels = favorable_labels
        self.protected_attributes = protected_attributes
        self.unfavorable_labels = unfavorable_labels
        self.redact = redact
        if preparation is None:
            preparation = lale.lib.lale.NoOp
        self.preparation = preparation
        self.mitigator = mitigator

    def _prep_and_encode(self, X, y=None):
        prepared_X = self.redact_and_prep.transform(X, y)
        encoded_X, encoded_y = self.prot_attr_enc.transform(X, y)
        combined_attribute_names = list(prepared_X.columns) + [
            name for name in encoded_X.columns if name not in prepared_X.columns
        ]
        combined_columns = [
            encoded_X[name] if name in encoded_X else prepared_X[name]
            for name in combined_attribute_names
        ]
        combined_X = pd.concat(combined_columns, axis=1)
        result = self.pandas_to_dataset.convert(combined_X, encoded_y)
        return result

    def _decode(self, y):
        assert isinstance(y, pd.Series)
        assert len(self.favorable_labels) == 1 and len(self.not_favorable_labels) == 1
        favorable, not_favorable = (
            self.favorable_labels[0],
            self.not_favorable_labels[0],
        )
        result = y.map(lambda label: favorable if label == 1 else not_favorable)
        return result

    def fit(self, X, y):
        from lale.lib.aif360 import ProtectedAttributesEncoder, Redacting

        fairness_info = {
            "favorable_labels": self.favorable_labels,
            "protected_attributes": self.protected_attributes,
            "unfavorable_labels": self.unfavorable_labels,
        }
        redacting = Redacting(**fairness_info) if self.redact else lale.lib.lale.NoOp
        trainable_redact_and_prep = redacting >> self.preparation
        assert isinstance(trainable_redact_and_prep, TrainablePipeline)
        self.redact_and_prep = trainable_redact_and_prep.fit(X, y)
        self.prot_attr_enc = ProtectedAttributesEncoder(
            **fairness_info,
            remainder="drop",
            return_X_y=True,
        )
        prot_attr_names = [pa["feature"] for pa in self.protected_attributes]
        self.pandas_to_dataset = _PandasToDatasetConverter(
            favorable_label=1,
            unfavorable_label=0,
            protected_attribute_names=prot_attr_names,
        )
        encoded_data = self._prep_and_encode(X, y)
        self.mitigator.fit(encoded_data)
        self.classes_ = set(list(y))
        self.not_favorable_labels = list(
            self.classes_ - set(list(self.favorable_labels))
        )
        self.classes_ = np.array(list(self.classes_))
        return self

    def predict(self, X, **predict_params):
        encoded_data = self._prep_and_encode(X)
        result_data = self.mitigator.predict(encoded_data, **predict_params)
        _, result_y = dataset_to_pandas(result_data, return_only="y")
        decoded_y = self._decode(result_y)
        return decoded_y

    def predict_proba(self, X):
        # Note, will break for GerryFairClassifier
        encoded_data = self._prep_and_encode(X)
        result_data = self.mitigator.predict(encoded_data)
        favorable_probs = result_data.scores
        all_probs = np.hstack([1 - favorable_probs, favorable_probs])
        return all_probs


class _BasePostEstimatorImpl:
    def __init__(
        self,
        *,
        favorable_labels,
        protected_attributes,
        unfavorable_labels,
        estimator,
        redact,
        mitigator,
    ):
        _validate_fairness_info(
            favorable_labels, protected_attributes, unfavorable_labels, True
        )
        self.favorable_labels = favorable_labels
        self.protected_attributes = protected_attributes
        self.unfavorable_labels = unfavorable_labels
        self.estimator = estimator
        self.redact = redact
        self.mitigator = mitigator

    def _decode(self, y):
        assert isinstance(y, pd.Series)
        assert len(self.favorable_labels) == 1 and len(self.not_favorable_labels) == 1
        favorable, not_favorable = (
            self.favorable_labels[0],
            self.not_favorable_labels[0],
        )
        result = y.map(lambda label: favorable if label == 1 else not_favorable)
        return result

    def fit(self, X, y):
        from lale.lib.aif360 import ProtectedAttributesEncoder, Redacting

        fairness_info = {
            "favorable_labels": self.favorable_labels,
            "protected_attributes": self.protected_attributes,
            "unfavorable_labels": self.unfavorable_labels,
        }
        redacting = Redacting(**fairness_info) if self.redact else lale.lib.lale.NoOp
        trainable_redact_and_estim = redacting >> self.estimator
        assert isinstance(trainable_redact_and_estim, TrainablePipeline)
        self.redact_and_estim = trainable_redact_and_estim.fit(X, y)
        self.prot_attr_enc = ProtectedAttributesEncoder(
            **fairness_info,
            remainder="drop",
            return_X_y=True,
        )
        prot_attr_names = [pa["feature"] for pa in self.protected_attributes]
        self.pandas_to_dataset = _PandasToDatasetConverter(
            favorable_label=1,
            unfavorable_label=0,
            protected_attribute_names=prot_attr_names,
        )
        encoded_X, encoded_y = self.prot_attr_enc.transform(X, y)
        self.y_dtype = encoded_y.dtype
        self.y_name = encoded_y.name
        predicted_y = self.redact_and_estim.predict(X)
        predicted_y = _ndarray_to_series(predicted_y, self.y_name, X.index)
        _, predicted_y = self.prot_attr_enc.transform(X, predicted_y)
        predicted_probas = self.redact_and_estim.predict_proba(X)
        dataset_true = self.pandas_to_dataset.convert(encoded_X, encoded_y)
        dataset_pred = self.pandas_to_dataset.convert(
            encoded_X, predicted_y, predicted_probas
        )
        self.mitigator = self.mitigator.fit(dataset_true, dataset_pred)
        self.classes_ = set(list(y))
        self.not_favorable_labels = list(
            self.classes_ - set(list(self.favorable_labels))
        )
        self.classes_ = np.array(list(self.classes_))
        return self

    def predict(self, X):
        predicted_y = self.redact_and_estim.predict(X)
        predicted_probas = self.redact_and_estim.predict_proba(X)
        predicted_y = _ndarray_to_series(predicted_y, self.y_name, X.index)
        encoded_X, predicted_y = self.prot_attr_enc.transform(X, predicted_y)
        dataset_pred = self.pandas_to_dataset.convert(
            encoded_X, predicted_y, predicted_probas
        )
        dataset_out = self.mitigator.predict(dataset_pred)
        _, result_y = dataset_to_pandas(dataset_out, return_only="y")
        decoded_y = self._decode(result_y)
        return decoded_y

    def predict_proba(self, X):
        predicted_y = self.redact_and_estim.predict(X)
        predicted_probas = self.redact_and_estim.predict_proba(X)
        predicted_y = _ndarray_to_series(predicted_y, self.y_name, X.index)
        encoded_X, predicted_y = self.prot_attr_enc.transform(X, predicted_y)
        dataset_pred = self.pandas_to_dataset.convert(
            encoded_X, predicted_y, predicted_probas
        )
        dataset_out = self.mitigator.predict(dataset_pred)
        favorable_probs = dataset_out.scores
        all_probs = np.hstack([1 - favorable_probs, favorable_probs])
        return all_probs


_categorical_supervised_input_fit_schema = {
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        },
        "y": {
            "description": "Target class labels; the array is over samples.",
            "anyOf": [
                {"type": "array", "items": {"type": "number"}},
                {"type": "array", "items": {"type": "string"}},
            ],
        },
    },
}

_categorical_unsupervised_input_fit_schema = {
    "description": "Input data schema for training.",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        },
        "y": {"description": "Target values; the array is over samples."},
    },
}

_categorical_input_predict_schema = {
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        }
    },
}

_categorical_output_predict_schema = {
    "description": "Predicted class label per sample.",
    "anyOf": [
        {"type": "array", "items": {"type": "number"}},
        {"type": "array", "items": {"type": "string"}},
    ],
}

_categorical_input_predict_proba_schema = {
    "type": "object",
    "additionalProperties": False,
    "required": ["X"],
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        }
    },
}

_categorical_output_predict_proba_schema = {
    "description": "The class probabilities of the input samples",
    "anyOf": [
        {"type": "array", "items": {"laleType": "Any"}},
        {"type": "array", "items": {"type": "array", "items": {"laleType": "Any"}}},
    ],
}

_categorical_input_transform_schema = {
    "description": "Input data schema for transform.",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {
        "X": {
            "description": "Features; the outer array is over samples.",
            "type": "array",
            "items": {
                "type": "array",
                "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
            },
        }
    },
}

_categorical_output_transform_schema = {
    "description": "Output data schema for reweighted features.",
    "type": "array",
    "items": {
        "type": "array",
        "items": {"anyOf": [{"type": "number"}, {"type": "string"}]},
    },
}

_numeric_output_transform_schema = {
    "description": "Output data schema for reweighted features.",
    "type": "array",
    "items": {"type": "array", "items": {"type": "number"}},
}
