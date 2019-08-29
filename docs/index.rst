.. LALE documentation master file, created by
   sphinx-quickstart on Sat Mar 23 07:09:13 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/IBM/lale

Welcome to LALE's documentation!
================================

.. image:: img/lale_logo.jpg
  :width: 50
  :alt: logo

Lale (pronounced *laleh*) is a Python library for semi-automated data science.
Lale is compatible with `scikit-learn <https://scikit-learn.org>`_ while
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
`LICENSE.txt <https://github.com/IBM/lale/blob/master/LICENSE.txt>`_. It is currently in an **Alpha release**,
without warranties of any kind.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation


The following paper has a technical deep-dive::


        @Article{arxiv19-lale,
        author = "Hirzel, Martin and Kate, Kiran and Shinnar, Avraham and Roy, Subhrajit and Ram, Parikshit",
        title = "Type-Driven Automated Learning with {Lale}",
        journal = "CoRR",
        volume = "abs/1906.03957",
        year = 2019,
        month = may,
        url = "https://arxiv.org/abs/1906.03957" }




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
