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

import lale.lib.sklearn


def wrap_pipeline_segments(orig_pipeline):
    """Wrap segments of the pipeline to mark them for pretty_print() and visualize().

If the pipeline does not look like it came from AutoAI, just return it
unchanged. Otherwise, find the NumpyPermuteArray operator. Everything
before that operator is preprocessing. Everything after
NumpyPermuteArray but before the final estimator is feature
engineering."""
    if len(orig_pipeline.steps()) <= 2:
        return orig_pipeline
    estimator = orig_pipeline._get_last()
    prep = orig_pipeline.remove_last()
    cognito = None
    PREP_END = "lale.lib.autoai_libs.numpy_permute_array.NumpyPermuteArrayImpl"
    while True:
        last = prep._get_last()
        if last is None or not last.class_name().startswith("lale.lib.autoai_libs."):
            return orig_pipeline
        if last.class_name() == PREP_END:
            break
        prep = prep.remove_last()
        if cognito is None:
            cognito = last
        else:
            cognito = last >> cognito
    prep_wrapped = lale.lib.sklearn.Pipeline(steps=[("preprocessing_pipeline", prep)])
    if cognito is None:
        result = prep_wrapped >> estimator
    else:
        cognito_wrapped = lale.lib.sklearn.Pipeline(
            steps=[("feature_engineering_pipeline", cognito)]
        )
        result = prep_wrapped >> cognito_wrapped >> estimator
    return result
