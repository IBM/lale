# Lale

[![Build Status](https://travis-ci.com/IBM/lale.svg?branch=master)](https://travis-ci.com/IBM/lale)
[![Documentation Status](https://readthedocs.org/projects/lale/badge/?version=latest)](https://lale.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![PyPI version shields.io](https://img.shields.io/pypi/v/lale)](https://pypi.python.org/pypi/lale/)
[![codecov](https://codecov.io/gh/IBM/lale/branch/master/graph/badge.svg)](https://codecov.io/gh/IBM/lale)
<br />
<img src="https://github.com/IBM/lale/raw/master/docs/img/lale_logo.jpg" alt="logo" width="55px"/>

Lale is a Python library for semi-automated data science.
Lale makes it easy to automatically select algorithms and tune
hyperparameters of pipelines that are compatible with
[scikit-learn](https://scikit-learn.org), in a type-safe fashion.  If
you are a data scientist who wants to experiment with automated
machine learning, this library is for you!
Lale adds value beyond scikit-learn along three dimensions:
automation, correctness checks, and interoperability.
For *automation*, Lale provides a consistent high-level interface to
existing pipeline search tools including Hyperopt, GridSearchCV, and SMAC.
For *correctness checks*, Lale uses JSON Schema to catch mistakes when
there is a mismatch between hyperparameters and their type, or between
data and operators.
And for *interoperability*, Lale has a growing library of transformers
and estimators from popular libraries such as scikit-learn, XGBoost,
PyTorch etc.
Lale can be installed just like any other Python package and can be
edited with off-the-shelf Python tools such as Jupyter notebooks.

* [Introductory guide](https://nbviewer.jupyter.org/github/IBM/lale/blob/master/examples/docs_guide_for_sklearn_users.ipynb) for scikit-learn users
* [Installation instructions](https://github.com/IBM/lale/blob/master/docs/installation.rst)
* Technical overview [slides](https://github.com/IBM/lale/blob/master/talks/2019-1105-lale.pdf), [notebook](https://nbviewer.jupyter.org/github/IBM/lale/blob/master/examples/talk_2019-1105-lale.ipynb), and [video](https://www.youtube.com/watch?v=R51ZDJ64X18&list=PLGVZCDnMOq0pwoOqsaA87cAoNM4MWr51M&index=35&t=0s)
* IBM's [AutoAI SDK](http://wml-api-pyclient-v4.mybluemix.net/#autoai-beta-ibm-cloud-only) uses Lale, see demo [notebook](https://dataplatform.cloud.ibm.com/exchange/public/entry/view/a2d87b957b60c846267137bfae130dca)
* Guide for wrapping [new operators](https://nbviewer.jupyter.org/github/IBM/lale/blob/master/examples/docs_new_operators.ipynb)
* Guide for [contributing](https://github.com/IBM/lale/blob/master/CONTRIBUTING.md) to Lale
* [FAQ](https://github.com/IBM/lale/blob/master/docs/faq.rst)
* Python [API documentation](https://lale.readthedocs.io/en/latest/)
* arXiv [paper](https://arxiv.org/pdf/1906.03957.pdf)

The name Lale, pronounced *laleh*, comes from the Persian word for
tulip. Similarly to popular machine-learning libraries such as
scikit-learn, Lale is also just a Python library, not a new stand-alone
programming language. It does not require users to install new tools
nor learn new syntax.

The following paper has a technical deep-dive:
```
@Article{arxiv19-lale,
  author = "Hirzel, Martin and Kate, Kiran and Shinnar, Avraham and Roy, Subhrajit and Ram, Parikshit",
  title = "Type-Driven Automated Learning with {Lale}",
  journal = "CoRR",
  volume = "abs/1906.03957",
  year = 2019,
  month = may,
  url = "https://arxiv.org/abs/1906.03957" }
```

Lale is distributed under the terms of the Apache 2.0 License, see
[LICENSE.txt](https://github.com/IBM/lale/blob/master/LICENSE.txt).
It is currently in an **Alpha release**, without warranties of any
kind.
