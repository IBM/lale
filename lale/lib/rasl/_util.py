# Copyright 2023 IBM Corporation
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

import pandas as pd

from lale.helpers import _is_pandas_df, _is_spark_df


# From https://github.com/scikit-learn-contrib/category_encoders/blob/master/category_encoders/utils.py
def _is_category(dtype):
    return pd.api.types.is_categorical_dtype(dtype)


# Based on https://github.com/scikit-learn-contrib/category_encoders/blob/master/category_encoders/utils.py
def get_obj_cols(df):
    """
    Returns names of 'object' columns in the DataFrame.
    """
    obj_cols = []
    if _is_pandas_df(df):
        for idx, dt in enumerate(df.dtypes):
            if dt == "object" or _is_category(dt):
                obj_cols.append(df.columns.values[idx])
    elif _is_spark_df(df):
        for idx, (col, dt) in enumerate(df.dtypes):
            if dt == "string":
                obj_cols.append(col)
    else:
        assert False
    return obj_cols
