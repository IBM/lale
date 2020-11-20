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
        "black",
        "graphviz",
        "hyperopt",
        "jsonschema",
        "jsonsubschema",
        "scikit-learn>=0.20.3",
        "scipy",
        "pandas",
        "decorator",
        "h5py",
        "astunparse",
    ]

import lale  # noqa: E402

if "TRAVIS" in os.environ:
    now = datetime.now().strftime("%y%m%d%H%M")
    VERSION = f"{lale.__version__}-{now}"
else:
    VERSION = lale.__version__

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
    extras_require={
        "full": [
            "xgboost",
            "lightgbm",
            "liac-arff>=2.4.0",
            "pytorch-pretrained-bert>=0.6.1",
            "torchvision>=0.2.2",
            "tensorflow-datasets>=1.0.1",
            "tensorflow>=1.13.1",
            "tensorflow_hub",
            "spacy",
            "smac<=0.10.0",
            "numba==0.49.0",
            "aif360",
            "torch>=1.0",
            "BlackBoxAuditing",
            "imbalanced-learn",
        ],
        "dev": ["pre-commit"],
        "test": [
            "autoai-libs",
            "joblib",
            "jupyter",
            "numpydoc",
            "sphinx",
            "m2r",
            "sphinx_rtd_theme",
            "sphinxcontrib.apidoc",
            "pytest-cov",
            "codecov",
        ],
    },
)
