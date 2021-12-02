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

import logging
import os

# import sys
from datetime import datetime

from setuptools import find_packages, setup

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

try:
    import builtins

    # This trick is borrowed from scikit-learn
    # This is a bit (!) hackish: we are setting a global variable so that the
    # main lale __init__ can detect if it is being loaded by the setup
    # routine, to avoid attempting to import components before installation.
    builtins.__LALE_SETUP__ = True  # type: ignore
except ImportError:
    pass

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

on_rtd = os.environ.get("READTHEDOCS") == "True"
if on_rtd:
    install_requires = []
else:
    install_requires = [
        "numpy",
        "black==19.10b0",
        "graphviz",
        "hyperopt<=0.2.5",
        "jsonschema",
        "jsonsubschema",
        "scikit-learn",
        "scipy",
        "pandas",
        "decorator",
        "astunparse",
    ]

import lale  # noqa: E402

if "TRAVIS" in os.environ:
    now = datetime.now().strftime("%y%m%d%H%M")
    VERSION = f"{lale.__version__}-{now}"
else:
    VERSION = lale.__version__

extras_require = {
    "full": [
        "xgboost<=1.3.3",
        "lightgbm",
        "snapml>=1.7.0rc3<=1.8.0",
        "liac-arff>=2.4.0",
        "tensorflow==2.4.0",
        "smac<=0.10.0",
        "numba==0.49.0",
        "aif360>=0.4.0",
        "torch>=1.0",
        "BlackBoxAuditing",
        "imbalanced-learn",
        "cvxpy>=1.0",
        "fairlearn",
        "h5py",
    ],
    "dev": ["pre-commit"],
    "test": [
        "autoai-libs>=1.12.6",
        "joblib",
        "jupyter",
        "numpydoc",
        "sphinx==2.4.4",
        "m2r",
        "sphinx_rtd_theme",
        "sphinxcontrib.apidoc",
        "pytest",
        "pyspark",
        "func_timeout",
    ],
    "fairness": [
        "liac-arff>=2.4.0",
        "aif360<0.4.0",
        "BlackBoxAuditing",
    ],
}

# # Inserting tensorflow 1.x only for Python 3.7
# if sys.version_info.minor == 7:
#     extras_require["full"].append("tensorflow>=1.13.1,<2")

setup(
    name="lale",
    version=VERSION,
    author="Guillaume Baudart, Martin Hirzel, Kiran Kate, Parikshit Ram, Avraham Shinnar",
    description="Library for Semi-Automated Data Science",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IBM/lale",
    python_requires=">=3.6",
    packages=find_packages(),
    license="",
    install_requires=install_requires,
    extras_require=extras_require,
)
