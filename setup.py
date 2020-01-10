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

from setuptools import setup, find_packages
import os
import sys

with open("README.md", "r") as fh:
    long_description = fh.read()

on_rtd = os.environ.get('READTHEDOCS') == 'True'
if on_rtd:
    install_requires = []
else:
    install_requires=[
        'astunparse',
        'graphviz',
        'hyperopt==0.1.2',
        'jsonschema==2.6.0',
        'jsonsubschema',
        'liac-arff>=2.4.0',
        'numpy',
        'PyYAML',
        'scikit-learn==0.20.3',
        'scipy',
        'pandas',
        'xgboost',
        'lightgbm',
        'decorator',
        'torch>=1.0',
        'h5py']

setup(
    name='lale',
    version='0.3.2',
    author="Guillaume Baudart, Martin Hirzel, Kiran Kate, Parikshit Ram, Avraham Shinnar",
    description="Library for Semi-Automated Data Science",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IBM/lale",
    python_requires='>=3.6',
    packages=find_packages(),
    license='',
    install_requires = install_requires,
    extras_require={
        'full': [
            'pytorch-pretrained-bert>=0.6.1',
            'pillow<=6.2.1', #TODO: remove this when torchvision issue solved
            'torchvision>=0.2.2',
            'tensorflow-datasets>=1.0.1',
            'tensorflow>=1.13.1',
            'tensorflow_hub',
            'spacy',
            'smac<=0.10.0',
            'aif360',
            'numba',
            'BlackBoxAuditing'],
        'test':[
            'jupyter',
            'mypy<=0.740',
            'flake8',
            'numpydoc',
            'sphinx',
            'm2r',
            'sphinx_rtd_theme',
            'sphinxcontrib.apidoc'
        ]}
)
