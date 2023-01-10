.. LALE documentation master file, created by
   sphinx-quickstart on Sat Mar 23 07:09:13 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

:github_url: https://github.com/IBM/lale

Welcome to LALE's API documentation!
====================================

.. image:: img/lale_logo.jpg
  :width: 50
  :alt: logo

These pages show API documentation for Lale, which is auto-generated
from docstrings, many of which are themselves auto-generated from
schemas.  For an introductory overview, installation instructions,
examples, talks, code, a paper, etc., please check the Lale github
page: https://github.com/IBM/lale .

Core Modules
------------

* `lale.operators`_ Classes for operators including pipelines.
* `lale.datasets`_ Schema-augmented datasets.
* `lale.grammar`_ Pipeline topology search syntax.
* `lale.json_operator`_ Includes a from_json() function.
* `lale.schemas`_ Python API for writing JSON schemas.

.. _`lale.operators`: modules/lale.operators.html
.. _`lale.datasets`: modules/lale.datasets.html
.. _`lale.grammar`: modules/lale.grammar.html
.. _`lale.json_operator`: modules/lale.json_operator.html
.. _`lale.schemas`: modules/lale.schemas.html

Operator libraries
------------------

* `lale.lib.sklearn`_ based on `scikit-learn`_
* `lale.lib.lale`_ containing Lale-specific operators
* `lale.lib.aif360`_ based on `AI Fairness 360`_
* `lale.lib.autoai_libs`_ based on `autoai-libs`_
* `lale.lib.autogen`_ auto-generated based on `scikit-learn`_
* `lale.lib.category_encoders`_ based on `category_encoders`_
* `lale.lib.imblearn`_ based on `imbalanced-learn`_
* `lale.lib.lightgbm`_ based on `LightGBM`_
* `lale.lib.pytorch`_ based on `PyTorch`_
* `lale.lib.rasl`_ relational algebra in scikit-learn
* `lale.lib.snapml`_ based on `Snap ML`_
* `lale.lib.spacy`_ based on `spaCy`_
* `lale.lib.tensorflow`_ based on `TensorFlow`_
* `lale.lib.xgboost`_ based on `XGBoost`_

.. _`lale.lib.sklearn`: modules/lale.lib.sklearn.html#module-lale.lib.sklearn
.. _`scikit-learn`: https://scikit-learn.org/
.. _`lale.lib.lale`: modules/lale.lib.lale.html#module-lale.lib.lale
.. _`lale.lib.aif360`: modules/lale.lib.aif360.html#module-lale.lib.aif360
.. _`AI Fairness 360`: https://github.com/IBM/AIF360
.. _`lale.lib.autoai_libs`: modules/lale.lib.autoai_libs.html#module-lale.lib.autoai_libs
.. _`autoai-libs`: https://pypi.org/project/autoai-libs/
.. _`autoai-ts-libs`: https://pypi.org/project/autoai-ts-libs/
.. _`lale.lib.autogen`: modules/lale.lib.autogen.html#module-lale.lib.autogen
.. _`lale.lib.category_encoders`: modules/lale.lib.category_encoders.html#module-lale.lib.category_encoders
.. _`category_encoders`: https://contrib.scikit-learn.org/category_encoders/
.. _`lale.lib.imblearn`: modules/lale.lib.imblearn.html#module-lale.lib.imblearn
.. _`imbalanced-learn`: https://imbalanced-learn.readthedocs.io/en/stable/index.html
.. _`lale.lib.lightgbm`: modules/lale.lib.lightgbm.html#module-lale.lib.lightgbm
.. _`LightGBM`: https://lightgbm.readthedocs.io/en/latest/Python-API.html
.. _`lale.lib.snapml`: modules/lale.lib.snapml.html#module-lale.lib.snapml
.. _`Snap ML`: https://www.zurich.ibm.com/snapml/
.. _`lale.lib.rasl`: modules/lale.lib.rasl.html#module-lale.lib.rasl
.. _`lale.lib.pytorch`: modules/lale.lib.pytorch.html#module-lale.lib.pytorch
.. _`PyTorch`: https://pytorch.org/
.. _`lale.lib.spacy`: modules/lale.lib.spacy.html#module-lale.lib.spacy
.. _`spaCy`: https://spacy.io/
.. _`lale.lib.tensorflow`: modules/lale.lib.tensorflow.html#module-lale.lib.tensorflow
.. _`TensorFlow`: https://www.tensorflow.org/
.. _`lale.lib.xgboost`: modules/lale.lib.xgboost.html#module-lale.lib.xgboost
.. _`XGBoost`: https://xgboost.readthedocs.io/en/latest/python/python_api.html

Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`

.. toctree::
   :hidden:
   :glob:

   README*
   faq
   installation
   modules/modules
   papers
