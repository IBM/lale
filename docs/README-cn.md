# Lale

[![Build Status](https://travis-ci.com/IBM/lale.svg?branch=master)](https://travis-ci.com/IBM/lale)
[![Documentation Status](https://readthedocs.org/projects/lale/badge/?version=latest)](https://lale.readthedocs.io/en/latest/?badge=latest)
[![PyPI version shields.io](https://img.shields.io/pypi/v/lale?color=success)](https://pypi.python.org/pypi/lale/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
<br />
<img src="https://github.com/IBM/lale/raw/master/docs/img/lale_logo.jpg" alt="logo" width="55px"/>

Lale是一个半自动化数据科学的Python库. 通过Lale，您可以简单便捷地用类型安全的方式自动选择与[scikit-learn](https://scikit-learn.org)兼容的管道的算法和调整其超参数。如果您是想尝试自动化机器学习的数据科学家，那么此库适合您！除了scikit-learn之外，Lale还在以下三个方面有重要作用：自动化，正确性检查和互操作性。对于*自动化*，Lale为现有管道搜索工具（包括Hyperopt，GridSearchCV和SMAC）提供了一致的高级界面。对于*正确性检查*，当超参数与其类型之间或数据与运算符之间不匹配时，Lale使用JSON Schema来找到错误。对于*互操作性*，Lale有正在不断增加的转换器和估计器的库，其中包括来自于例如scikit-learn，XGBoost，PyTorch等较热门的转换器和估计器。Lale可以像其他任何Python软件包一样安装，并可以使用现成的Python工具进行编辑，例如Jupyter笔记本。

* [scikit-learn用户入门指南](https://nbviewer.jupyter.org/github/IBM/lale/blob/master/examples/docs_guide_for_sklearn_users.ipynb) 
* [安装说明](https://github.com/IBM/lale/blob/master/docs/installation.rst)
* 技术概述[幻灯片](https://github.com/IBM/lale/blob/master/talks/2019-1105-lale.pdf), [笔记本](https://nbviewer.jupyter.org/github/IBM/lale/blob/master/examples/talk_2019-1105-lale.ipynb), and [视频](https://www.youtube.com/watch?v=R51ZDJ64X18&list=PLGVZCDnMOq0pwoOqsaA87cAoNM4MWr51M&index=35&t=0s)
* IBM的[AutoAI SDK](http://wml-api-pyclient-v4.mybluemix.net/#autoai-beta-ibm-cloud-only)使用Lale, [请参阅演示 笔记本](https://dataplatform.cloud.ibm.com/exchange/public/entry/view/a2d87b957b60c846267137bfae130dca)
* 包装指南[新运算符](https://nbviewer.jupyter.org/github/IBM/lale/blob/master/examples/docs_new_operators.ipynb)
* [贡献]指南(https://github.com/IBM/lale/blob/master/CONTRIBUTING.md)
* [常问问题](https://github.com/IBM/lale/blob/master/docs/faq.rst)
* [Papers](https://github.com/IBM/lale/blob/master/docs/papers.rst)
* Python [API文档](https://lale.readthedocs.io/en/latest/)

Lale这个名字的发音为*laleh*，来自波斯语，意为郁金香。和热门的机器学习库如scikit-learn一样，Lale也是一个Python库而不是一个新的独立编程语言。用户不需要安装新工具也不需要学习新的语法。

Lale根据Apache 2.0许可的条款分发，请参阅[LICENSE.txt](https://github.com/IBM/lale/blob/master/LICENSE.txt).
目前处于**Alpha release**, 没有任何保证。 