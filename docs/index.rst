.. LALE documentation master file, created by
   sphinx-quickstart on Sat Mar 23 07:09:13 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/lale/lale

Welcome to LALE's documentation!
================================

.. image:: img/lale_logo.jpg
  :width: 50
  :alt: logo

Lale is a Python library for data science with an emphasis on automation, usability, and interoperability.
Lale can be installed just like any other Python package and can be
edited with off-the-shelf Python tools such as Jupyter notebooks.
Lale is compatible with Scikit-learn while adding value for
automation, correctness checks, and interoperability.
Lale performs type-checking to catch mistakes when there is a mismatch
between hyperparameters and their type, or between data and operators.
Lale provides a consistent high-level interface to a growing set of AI
automation tools including GridSearchCV, SMAC, and Hyperopt.
Lale has a growing library of transformers and estimators from popular
libraries such as Scikit-learn, XGBoost, PyTorch etc.

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   installation


The name Lale, pronounced *laleh*, comes from the Persian word for
tulip. Similarly to popular machine-learning libraries such as
Scikit-learn, Lale is also just a Python library, not a new stand-alone
programming language. It does not require users to install new tools
nor learn new syntax.

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
