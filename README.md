# Lale

[![Build Status](https://travis-ci.com/IBM/lale.svg?branch=master)](https://travis-ci.com/IBM/lale)
[![Documentation Status](https://readthedocs.org/projects/lale/badge/?version=latest)](https://lale.readthedocs.io/en/latest/?badge=latest)
<br />
<img src="docs/img/lale_logo.jpg" alt="logo" width="55px"/>

Lale is a Python library for semi-automated data science.
Lale is compatible with [scikit-learn](https://scikit-learn.org) while
adding value for automation, correctness checks, and interoperability.
For *automation*, Lale provides a consistent high-level interface to
existing pipeline search tools including GridSearchCV, SMAC, and
Hyperopt.
For *correctness checks*, Lale uses JSON Schema to catch mistakes when
there is a mismatch between hyperparameters and their type, or between
data and operators.
And for *interoperability*, Lale has a growing library of transformers
and estimators from popular libraries such as scikit-learn, XGBoost,
PyTorch etc.
Lale can be installed just like any other Python package and can be
edited with off-the-shelf Python tools such as Jupyter notebooks.

Lale is distributed under the terms of the Apache 2.0 License, see
[LICENSE.txt](LICENSE.txt). It is currently in an **Alpha release**,
without warranties of any kind.

* Technical overview [slides](talks/2019-0529-lale.pdf)
* [Installation instructions](docs/installation.rst)
* Python [API documentation](https://lale.readthedocs.io/en/latest/)
* Guide for wrapping [new operators](docs/new_operators.md)
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

Contributors are expected to submit a "Developer's Certificate of
Origin", which can be found in [DCO1.1.txt](DCO1.1.txt).
