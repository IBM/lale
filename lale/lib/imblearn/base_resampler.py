# Copyright 2019-2023 IBM Corporation
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

import warnings


class _BaseResamplerImpl:
    def __init__(self, operator=None, resampler=None):
        self.operator = operator
        self.resampler = resampler

    def fit(self, X, y=None):
        resampler = self.resampler
        assert resampler is not None
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            X, y = resampler.fit_resample(X, y)

        op = self.operator
        assert op is not None
        self.trained_operator = op.fit(X, y)
        if hasattr(self.trained_operator, "classes_"):
            self.classes_ = self.trained_operator.classes_
        return self

    def transform(self, X, y=None):
        return self.trained_operator.transform(X, y)

    def predict(self, X, **predict_params):
        return self.trained_operator.predict(X, **predict_params)

    def predict_proba(self, X):
        return self.trained_operator.predict_proba(X)

    def decision_function(self, X):
        return self.trained_operator.decision_function(X)
