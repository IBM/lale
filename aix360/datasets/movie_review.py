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

import os
import tarfile
import urllib.request

import numpy as np


def load_movie_review():
    """Loads the sentiment classification from a movie reviews dataset.
    Read the readme from data/movie_review for more details.
    """
    download_base_url = "https://www.cs.cornell.edu/people/pabo/movie%2Dreview%2Ddata/rt-polaritydata.tar.gz"
    download_data_dir = os.path.join(
        os.path.dirname(__file__), "data", "movie_review", "download_data"
    )
    data_file_path = os.path.join(download_data_dir, "rt-polaritydata.tar.gz")
    if not os.path.exists(download_data_dir):
        os.makedirs(download_data_dir)
        print("created directory {}".format(download_data_dir))
        urllib.request.urlretrieve(download_base_url, data_file_path)

    X = []
    y = []
    with tarfile.open(data_file_path) as data_file:
        data_file.extractall(path=download_data_dir)

    with open(
        os.path.join(download_data_dir, "rt-polaritydata", "rt-polarity.neg"), "rb"
    ) as neg_data_file:
        for line in neg_data_file.readlines():
            X.append(str(line))
            y.append(-1)
    with open(
        os.path.join(download_data_dir, "rt-polaritydata", "rt-polarity.pos"), "rb"
    ) as pos_data_file:
        for line in pos_data_file.readlines():
            X.append(str(line))
            y.append(1)

    X = np.asarray(X, dtype=np.string_)
    y = np.asarray(y)

    from sklearn.utils import shuffle

    X, y = shuffle(X, y)

    return X, y
