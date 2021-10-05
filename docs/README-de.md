# Lale

[![Build Status](https://travis-ci.com/IBM/lale.svg?branch=master)](https://travis-ci.com/IBM/lale)
[![Documentation Status](https://readthedocs.org/projects/lale/badge/?version=latest)](https://lale.readthedocs.io/en/latest/?badge=latest)
[![PyPI version shields.io](https://img.shields.io/pypi/v/lale?color=success)](https://pypi.python.org/pypi/lale/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<br />
<img src="https://github.com/IBM/lale/raw/master/docs/img/lale_logo.jpg" alt="logo" width="55px"/>

Lale ist eine Python-Bibliothek für halbautomatische
Datenwissenschaft.  Lale macht es einfach, für Pipelines, die mit
[scikit-learn](https://scikit-learn.org) kompatibel sind, auf eine
typsichere Weise automatisch Algorithmen auszuwählen und
Hyperparameter zu konfigurieren.  Wenn Sie Datenwissenschaftler sind,
die gerne mit automatischem maschinellem Lernen experimentieren
möchten, dann ist dies die richtige Bibliothek für Sie!  Lale bietet
einen Mehrwert, der über scikit-learn in drei Dimensionen hinausgeht:
Automatisierung, Korrektheitsprüfungen, und Interoperabilität.  Für
*Automatisierung* bietet Lale eine konsistente
High-Level-Schnittstelle zu vorhandenen Pipelinesuchwerkzeugen,
einschließlich Hyperopt, GridSearchCV, und SMAC.  Für
Korrektheitsprüfungen nutzt Lale JSON Schema, um Fehler zu findern,
wenn Hyperparameter nicht zu ihrem Typ passen oder wenn Daten nicht zu
Operatoren passen.  Und für Interoperabilität hat Lale eine wachsende
Bibliothek von Transformers und Estimators aus beliebten Bibliotheken
wie scikit-learn, XGBoost, PyTorch, usw.  Lale kann wie jedes andere
Python-Paket installiert werden und kann mit üblichen
Python-Werkzeugen, so wie Jupyter Notebooks, editiert werden.

* [Einführungshandbuch](https://nbviewer.jupyter.org/github/IBM/lale/blob/master/examples/docs_guide_for_sklearn_users.ipynb) für scikit-learn Nutzer
* [Installationsanleitung](https://github.com/IBM/lale/blob/master/docs/installation.rst)
* Technische Übersicht [Slides](https://github.com/IBM/lale/blob/master/talks/2019-1105-lale.pdf), [Notebook](https://nbviewer.jupyter.org/github/IBM/lale/blob/master/examples/talk_2019-1105-lale.ipynb), und [Video](https://www.youtube.com/watch?v=R51ZDJ64X18&list=PLGVZCDnMOq0pwoOqsaA87cAoNM4MWr51M&index=35&t=0s)
* IBM's [AutoAI SDK](http://wml-api-pyclient-v4.mybluemix.net/#autoai-beta-ibm-cloud-only) benutzt Lale, siehe [Demonstrationsnotebook](https://dataplatform.cloud.ibm.com/exchange/public/entry/view/a2d87b957b60c846267137bfae130dca)
* Anleitung zur Einbeziehung [neuer Operatoren](https://nbviewer.jupyter.org/github/IBM/lale/blob/master/examples/docs_new_operators.ipynb)
* Leitfaden für [Beiträge](https://github.com/IBM/lale/blob/master/CONTRIBUTING.md) zu Lale
* [Veröffentlichungen](https://github.com/IBM/lale/blob/master/docs/papers.rst)
* [Häufig gestellte Fragen](https://github.com/IBM/lale/blob/master/docs/faq.rst)
* Python [API Dokumentation](https://lale.readthedocs.io/en/latest/)

Der Name Lale, ausgesprochen *laleh*, kommt vom persischen Wort für
Tulpe.  Ähnlich wie gängige Bibliotheken für maschinelles Lernen wie
scikit-learn ist Lale auch nur eine Python-Bibliothek, keine neue
eigenständige Programmiersprache. Benutzer müssen weder neue Tools
installieren noch neue Syntax lernen.

Lale wird unter den Bedingungen der Apache 2.0-Lizenz bereitgestellt, siehe
[LICENSE.txt](https://github.com/IBM/lale/blob/master/LICENSE.txt).
Es befindet sich derzeit in einer **Alpha-Version** ohne jegliche
Gewährleistung.
