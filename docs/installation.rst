Installation
============

Lale is easy to install. Assuming you already have a Python 3.6+
environment, all you need is the following:

.. code:: Bash

    pip install git+https://git@github.com/IBM/lale.git

This will install the **Lale Core** setup target, which includes many
operators, pipelines, and search space generation targeting hyperopt
and scikit-learn's GridSearchCV.  It has a smaller set of dependencies
than the **Lale Full** setup target, which also includes search space
generation for SMAC and some deep-learning operators. You can install
it as follows:

.. code:: Bash

    pip install git+https://git@github.com/IBM/lale.git#egg=lale[full]

Now you should be ready to start using Lale.

Installing from Source
----------------------

As an alternative to installing Lale directly from the online github
repository, you can also first clone the repository and then install
Lale from your local clone. For the **Lale Core** setup target:

.. code:: Bash

    git clone https://github.com/IBM/lale.git
    cd lale
    pip install .

For the **Lale Full** and **Lale Test** setup targets:

.. code:: Bash

    git clone https://github.com/IBM/lale.git
    cd lale
    pip install .[full,test]

Now, you are ready to run some tests. For a quick check, do the
following in the ``lale`` directory:

.. code:: Bash

    export PYTHONPATH=`pwd`
    python -m unittest test.test.TestLogisticRegression

This should say something like::

    Ran 20 tests in 105.201s
    OK

To run the full test suite, do the following in the ``lale``
directory:

.. code:: Bash

    make run_tests
