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

import collections
import os
import sys
import warnings

import numpy as np
from scipy.signal import resample
from sklearn import preprocessing

import lale.docstrings
import lale.operators

sys.path.append(os.getcwd())


seizure_type_data = collections.namedtuple(
    "seizure_type_data", ["seizure_type", "data"]
)

# Some of the classes and modules have been taken from https://github.com/MichaelHills/seizure-detection


class Slice:
    """
    Job: Take a slice of the data on the last axis.
    Note: Slice(x, y) works like a normal python slice, that is x to (y-1) will be taken.
    """

    def __init__(self, start, stop):
        self.start = start
        self.stop = stop + 1

    def get_name(self):
        return "slice%d-%d" % (self.start, self.stop)

    def apply(self, data):
        s = [
            slice(None),
        ] * data.ndim
        s[-1] = slice(self.start, self.stop)
        return data[s]


class Magnitude:
    """
    Job: Take magnitudes of Complex data
    """

    def get_name(self):
        return "mag"

    def apply(self, data):
        return np.absolute(data)


class Log10:
    """
    Apply Log10
    """

    def get_name(self):
        return "log10"

    def apply(self, data):
        # 10.0 * log10(re * re + im * im)
        indices = np.where(data <= 0)
        data[indices] = np.max(data)
        data[indices] = np.min(data) * 0.1
        return np.log10(data)


class Pipeline(object):
    """
    A Pipeline is an object representing the data transformations to make
    on the input data, finally outputting extracted features.
    pipeline: List of transforms to apply one by one to the input data
    """

    def __init__(self, pipeline):
        self.transforms = pipeline
        names = [t.get_name() for t in self.transforms]
        self.name = "empty" if len(names) == 0 else "_".join(names)

    def get_name(self):
        return self.name

    def apply(self, data):
        for transform in self.transforms:
            data = transform.apply(data)
        return data


class FFT:
    """
    Apply Fast Fourier Transform to the last axis.
    """

    def get_name(self):
        return "fft"

    def apply(self, data):
        axis = data.ndim - 1
        return np.fft.rfft(data, axis=axis)


class Resample:
    """
    Resample time-series data.
    """

    def __init__(self, sample_rate):
        self.f = sample_rate

    def get_name(self):
        return "resample%d" % self.f

    def apply(self, data):
        axis = data.ndim - 1
        if data.shape[-1] > self.f:
            return resample(data, self.f, axis=axis)
        return data


class StandardizeLast:
    """
    Scale across the last axis.
    """

    def get_name(self):
        return "standardize-last"

    def apply(self, data):
        return preprocessing.scale(data, axis=data.ndim - 1)


class StandardizeFirst:
    """
    Scale across the first axis.
    """

    def get_name(self):
        return "standardize-first"

    def apply(self, data):
        return preprocessing.scale(data, axis=0)


class CorrelationMatrix:
    """
    Calculate correlation coefficients matrix across all EEG channels.
    """

    def get_name(self):
        return "correlation-matrix"

    def apply(self, data):
        return np.corrcoef(data)


class Eigenvalues:
    """
    Take eigenvalues of a matrix, and sort them by magnitude in order to
    make them useful as features (as they have no inherent order).
    """

    def get_name(self):
        return "eigenvalues"

    def apply(self, data):
        w, v = np.linalg.eig(data)
        w = np.absolute(w)
        w.sort()
        return w


# Take the upper right triangle of a matrix
def upper_right_triangle(matrix):
    accum = []
    for i in range(matrix.shape[0]):
        for j in range(i + 1, matrix.shape[1]):
            accum.append(matrix[i, j])

    return np.array(accum)


class FreqCorrelation:
    """
    Correlation in the frequency domain. First take FFT with (start, end) slice options,
    then calculate correlation co-efficients on the FFT output, followed by calculating
    eigenvalues on the correlation co-efficients matrix.
    The output features are (fft, upper_right_diagonal(correlation_coefficients), eigenvalues)
    Features can be selected/omitted using the constructor arguments.
    """

    def __init__(
        self, start, end, scale_option, with_fft=False, with_corr=True, with_eigen=True
    ):
        self.start = start
        self.end = end
        self.scale_option = scale_option
        self.with_fft = with_fft
        self.with_corr = with_corr
        self.with_eigen = with_eigen
        assert scale_option in ("first_axis", "last_axis", "none")
        assert with_corr or with_eigen

    def get_name(self):
        selections = []
        if not self.with_corr:
            selections.append("nocorr")
        if not self.with_eigen:
            selections.append("noeig")
        if len(selections) > 0:
            selection_str = "-" + "-".join(selections)
        else:
            selection_str = ""
        return "freq-correlation-%d-%d-%s-%s%s" % (
            self.start,
            self.end,
            "withfft" if self.with_fft else "nofft",
            self.scale_option,
            selection_str,
        )

    def apply(self, data):
        data1 = FFT().apply(data)
        data1 = Slice(self.start, self.end).apply(data1)
        data1 = Magnitude().apply(data1)
        data1 = Log10().apply(data1)

        data2 = data1
        if self.scale_option == "first_axis":
            data2 = StandardizeFirst().apply(data2)
        elif self.scale_option == "last_axis":
            data2 = StandardizeLast().apply(data2)

        data2 = CorrelationMatrix().apply(data2)

        out = []

        if self.with_corr:
            ur = upper_right_triangle(data2)
            out.append(ur)
        if self.with_eigen:
            w = Eigenvalues().apply(data2)
            out.append(w)
        if self.with_fft:
            data1 = data1.ravel()
            out.append(data1)
        for d in out:
            assert d.ndim == 1

        return np.concatenate(out, axis=0)


class TimeCorrelation:
    """
    Correlation in the time domain. First downsample the data, then calculate correlation co-efficients
    followed by calculating eigenvalues on the correlation co-efficients matrix.
    The output features are (upper_right_diagonal(correlation_coefficients), eigenvalues)
    Features can be selected/omitted using the constructor arguments.
    """

    def __init__(self, max_hz, scale_option, with_corr=True, with_eigen=True):
        self.max_hz = max_hz
        self.scale_option = scale_option
        self.with_corr = with_corr
        self.with_eigen = with_eigen
        assert scale_option in ("first_axis", "last_axis", "none")
        assert with_corr or with_eigen

    def get_name(self):
        selections = []
        if not self.with_corr:
            selections.append("nocorr")
        if not self.with_eigen:
            selections.append("noeig")
        if len(selections) > 0:
            selection_str = "-" + "-".join(selections)
        else:
            selection_str = ""
        return "time-correlation-r%d-%s%s" % (
            self.max_hz,
            self.scale_option,
            selection_str,
        )

    def apply(self, data):
        # so that correlation matrix calculation doesn't crash
        for ch in data:
            if np.alltrue(ch == 0.0):
                ch[-1] += 0.00001

        data1 = data
        if data1.shape[1] > self.max_hz:
            data1 = Resample(self.max_hz).apply(data1)

        if self.scale_option == "first_axis":
            data1 = StandardizeFirst().apply(data1)
        elif self.scale_option == "last_axis":
            data1 = StandardizeLast().apply(data1)

        data1 = CorrelationMatrix().apply(data1)

        out = []
        if self.with_corr:
            ur = upper_right_triangle(data1)
            out.append(ur)
        if self.with_eigen:
            w = Eigenvalues().apply(data1)
            out.append(w)

        for d in out:
            assert d.ndim == 1

        return np.concatenate(out, axis=0)


class FFTWithTimeFreqCorrelation:
    """
    Combines FFT with time and frequency correlation, taking both correlation coefficients and eigenvalues.
    """

    def __init__(self, start, end, max_hz, scale_option):
        self.start = start
        self.end = end
        self.max_hz = max_hz
        self.scale_option = scale_option
        assert scale_option in ("first_axis", "last_axis", "none")

    def get_name(self):
        return "fft-with-time-freq-corr-%d-%d-r%d-%s" % (
            self.start,
            self.end,
            self.max_hz,
            self.scale_option,
        )

    def apply(self, data):
        data1 = TimeCorrelation(self.max_hz, self.scale_option).apply(data)
        data2 = FreqCorrelation(
            self.start, self.end, self.scale_option, with_fft=True
        ).apply(data)
        assert data1.ndim == data2.ndim

        return np.concatenate((data1, data2), axis=data1.ndim - 1)


class _TimeFreqEigenVectorsImpl:
    def __init__(
        self,
        window_length=1,
        window_step=0.5,
        fft_min_freq=1,
        fft_max_freq=24,
        sampling_frequency=250,
    ):
        self.window_length = window_length
        self.window_step = window_step
        self.fft_min_freq = fft_min_freq
        self.fft_max_freq = fft_max_freq
        self.sampling_frequency = sampling_frequency

    def transform(self, X, y=None):
        warnings.filterwarnings("ignore")
        pipeline = Pipeline(
            [
                FFTWithTimeFreqCorrelation(
                    self.fft_min_freq,
                    self.fft_max_freq,
                    self.sampling_frequency,
                    "first_axis",
                )
            ]
        )
        X_transformed = []
        y_transformed = np.empty((0))
        self.end_index_list = (
            []
        )  # This is the list of end indices for samples generated per seizure

        # The transformation map is just a list of indices corresponding to the last sample generated by each time-series.
        for i in range(len(X)):
            seizure_data = X[i]
            start, step = 0, int(np.floor(self.window_step * self.sampling_frequency))
            stop = start + int(np.floor(self.window_length * self.sampling_frequency))
            fft_data = []
            while stop < seizure_data.shape[1]:
                signal_window = seizure_data[:, start:stop]
                fft_window = pipeline.apply(signal_window)
                fft_data.append(fft_window)
                start, stop = start + step, stop + step
            X_transformed.extend(fft_data)
            if y is not None:
                seizure_label = y[i]
                labels_for_all_seizure_samples = np.full(len(fft_data), seizure_label)
                y_transformed = np.hstack(
                    (y_transformed, labels_for_all_seizure_samples)
                )
            previous_element = self.end_index_list[i - 1] if (i - 1) >= 0 else 0
            self.end_index_list.append(previous_element + len(fft_data))

        X_transformed = np.array(X_transformed)
        if y is None:
            y_transformed = None

        return X_transformed, y_transformed

    def get_transform_meta_output(self):
        if self.end_index_list is not None:
            return {"end_index_list": self.end_index_list}
        else:
            raise ValueError(
                "Must call transform before trying to access its meta output."
            )


_hyperparams_schema = {
    "description": "TODO",
    "allOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "window_length",
                "window_step",
                "fft_min_freq",
                "fft_max_freq",
                "sampling_frequency",
            ],
            "relevantToOptimizer": ["window_length", "window_step", "fft_max_freq"],
            "properties": {
                "window_length": {
                    "type": "number",
                    "default": 1,
                    "description": "TODO",
                    "minimumForOptimizer": 0.25,
                    "maximumForOptimizer": 2,
                    "distribution": "uniform",
                },
                "window_step": {
                    "type": "number",
                    "default": 0.5,
                    "description": "TODO",
                    "minimumForOptimizer": 0.25,
                    "maximumForOptimizer": 1,  # TODO: This is $data->window_length once $data is implemented
                    "distribution": "uniform",
                },
                "fft_min_freq": {
                    "type": "integer",
                    "default": 1,
                    "description": "TODO",
                },
                "fft_max_freq": {
                    "type": "integer",
                    "default": 24,
                    "description": "TODO",
                    "minimumForOptimizer": 2,
                    "maximumForOptimizer": 30,  # TODO: This is $data->sampling_frequency/2 once $data is implemented
                    "distribution": "uniform",
                },
                "sampling_frequency": {
                    "type": "integer",
                    "default": 250,
                    "description": "TODO",
                },
            },
        }
        # TODO: Any constraints on hyper-parameter combinations?
    ],
}

_input_transform_schema = {
    "description": "Input format for data passed to the transform method.",
    "type": "object",
    "required": ["X", "y"],
    "properties": {
        "X": {
            "type": "array",
            "items": {
                "type": "array",
                "items": {"type": "array", "items": {"type": "number"}},
            },
            "description": "The input data to complete.",
        },
        "y": {
            "type": "array",
            "items": {"anyOf": [{"type": "integer"}, {"type": "string"}]},
        },
    },
}
_output_transform_schema = {
    "description": "The input data to complete.",
    "type": "array",  # This is actually a tuple of X and y
    "items": {"type": "array"},
}
_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": "Combined schema for expected data and hyperparameters.",
    "type": "object",
    "tags": {"pre": [], "op": ["transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_transform": _input_transform_schema,
        "output_transform": _output_transform_schema,
    },
}


TimeFreqEigenVectors = lale.operators.make_operator(
    _TimeFreqEigenVectorsImpl, _combined_schemas
)

lale.docstrings.set_docstrings(TimeFreqEigenVectors)
