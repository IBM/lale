# Lale

[![Tests](https://github.com/IBM/lale/workflows/Tests/badge.svg?branch=master)](https://github.com/IBM/lale/actions?query=workflow%3ATests+branch%3Amaster)
[![Documentation Status](https://readthedocs.org/projects/lale/badge/?version=latest)](https://lale.readthedocs.io/en/latest/?badge=latest)
[![PyPI version shields.io](https://img.shields.io/pypi/v/lale?color=success)](https://pypi.python.org/pypi/lale/)
[![Imports: isort](https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336)](https://pycqa.github.io/isort/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![linting: pylint](https://img.shields.io/badge/linting-pylint-yellowgreen)](https://github.com/PyCQA/pylint)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![CII Best Practices](https://bestpractices.coreinfrastructure.org/projects/5863/badge)](https://bestpractices.coreinfrastructure.org/projects/5863)

<br />
<img src="https://github.com/IBM/lale/raw/master/docs/img/lale_logo.jpg" alt="logo" width="55px"/>


Lale est une bibliothèque Python pour la science des données semi-automatique. 
Lale utilise des techniques de typage pour faciliter la sélection automatique d'algorithmes et le réglage des hyperparamètres pour des pipelines compatibles avec scikit-learn. 
Si vous souhaitez essayer l'apprentissage automatique, cette bibliothèque est faite pour vous ! 
Lale complète scikit-learn selon trois axes: automatisation, correction, et interopérabilité. 
Pour l'*automatisation*, Lale fournit une interface cohérente de haut niveau pour les outils d'optimisation existants, notamment Hyperopt, GridSearchCV et SMAC. 
Pour la *correction*, Lale utilise les schémas JSON pour détecter les erreurs d'incompatibilité entre les hyperparamètres et leur type, ou entre les données et les opérateurs. 
Enfin, pour l'*interopérabilité*, Lale dispose d'une bibliothèque grandissante de transformateurs et d'estimateurs issus de projets populaires comme scikit-learn, XGBoost, PyTorch, etc.
Lale peut être installé comme n'importe quelle autre bibliothèque python et peut être utilisé avec des outils Python standards comme les notebooks Jupyter.

* [Introduction](https://github.com/IBM/lale/blob/master/examples/docs_guide_for_sklearn_users.ipynb) pour les utilisateurs de scikit-learn
* Instructions pour l'[installation](https://github.com/IBM/lale/blob/master/docs/installation.rst)
* Aperçu technique [slides](https://github.com/IBM/lale/blob/master/talks/2019-1105-lale.pdf), [notebook](https://github.com/IBM/lale/blob/master/examples/talk_2019-1105-lale.ipynb), et [video](https://www.youtube.com/watch?v=R51ZDJ64X18&list=PLGVZCDnMOq0pwoOqsaA87cAoNM4MWr51M&index=35&t=0s)
* IBM [AutoAI SDK](http://wml-api-pyclient-v4.mybluemix.net/#autoai-beta-ibm-cloud-only) utilise Lale, voir le [demo notebook](https://dataplatform.cloud.ibm.com/exchange/public/entry/view/a2d87b957b60c846267137bfae130dca)
* Guide pour ajouter de [nouveaux operateurs](https://github.com/IBM/lale/blob/master/examples/docs_new_operators.ipynb)
* Guide pour [contribuer](https://github.com/IBM/lale/blob/master/CONTRIBUTING.md) au projet Lale
* [FAQ](https://github.com/IBM/lale/blob/master/docs/faq.rst)
* [Articles](https://github.com/IBM/lale/blob/master/docs/papers.rst)
* [Documentation de l'API](https://lale.readthedocs.io/en/latest/)

Le nom Lale, prononcé *laleh*, vient de tulipe en persan. 
Comme les autres bibliothèques populaires d'apprentissage automatique comme scikit-learn, Lale est une simple bibliothèque python, pas un nouveau langage de programmation. 
Il n'est pas necessaire d'installer de nouveaux outils ni d'apprendre une nouvelle syntaxe.

Lale est distribué sous les termes de la licence Apache 2.0, voir
[LICENSE.txt](https://github.com/IBM/lale/blob/master/LICENSE.txt).
Il est actuellement en **version Alpha**, sans aucune garantie.
