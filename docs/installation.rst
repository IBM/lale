Installation
============

Installing from PyPI
----------------------

Lale is easy to install. Assuming you already have a Python 3.6+
environment, all you need is the following:

.. code:: Bash

    pip install lale

This will install the **Lale Core** setup target, which includes many
operators, pipelines, and search space generation targeting hyperopt
and scikit-learn's GridSearchCV.  It has a smaller set of dependencies
than the **Lale Full** setup target, which also includes search space
generation for SMAC, support for loading OpenML datasets in ARFF
format, and some deep-learning operators. You can install
it as follows:

.. code:: Bash

    pip install lale[full]

Now you should be ready to start using Lale, for instance, in a
Jupyter notebook.

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

    pip install .[full,test]

Now, you are ready to run some tests. For a quick check, do the
following in the ``lale`` directory:

.. code:: Bash

    export PYTHONPATH=`pwd`
    python -m unittest test.test.TestLogisticRegression

The output should look like::

    Ran 20 tests in 105.201s
    OK

To run the full test suite, do the following in the ``lale``
directory:

.. code:: Bash

    make run_tests

Setting up the Environment
--------------------------

For the full functionality of Lale, you will need a Python 3.6+
environment, as well as g++, graphviz, make, and swig. You can use
Lale on Linux, Windows 10, or Mac OS X. Depending on your operating
system, you can skip ahead to the appropriate section below.

On Windows 10
~~~~~~~~~~~~~

First, you should enable the Windows Subsystem for Linux (WSL):

- Start/search -> Settings -> Update&Security -> For developers -> activate "Developer Mode"

- Start/search -> Turn Windows features on or off -> WSL -> restart

- Internet Explorer -> https://aka.ms/wslstore -> Ubuntu -> get

- Start/search -> Ubuntu

At this point, you can continue with the instructions in section
`On Ubuntu Linux`_.

On Ubuntu Linux
~~~~~~~~~~~~~~~

Start by making sure your Ubuntu installation is up-to-date and check
the version. In a command shell, type:

.. code:: Bash

    sudo apt update
    sudo apt upgrade
    lsb_release -a

This should output something like "Description: Ubuntu 16.04.4 LTS".

Also, make sure you have g++, make, graphviz, and swig
installed. Otherwise, you can install them:

.. code:: Bash

    sudo apt install g++
    sudo apt install graphviz
    sudo apt install make
    sudo apt install swig

Next, set up a Python virtual environment with Python 3.6.

.. code:: Bash

    sudo add-apt-repository ppa:jonathonf/python-3.6
    sudo apt update
    sudo apt install python3.6
    sudo apt install python3.6-dev
    sudo apt install virtualenv
    virtualenv -p /usr/bin/python3.6 ~/python3.6venv
    source ~/python3.6venv/bin/activate

At this point, you can continue with the Lale `Installation`_
instructions at the top of this file.

On Mac OS X
~~~~~~~~~~~

Assuming you already have a Python 3.6+ virtual environment, you will
need to install swig using brew before you can install Lale.

If you encounter any issues in installing SMAC:

MacOS 10.14

.. code:: Bash

    open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg

Then

.. code:: Bash

    export CPATH=/Library/Developer/CommandLineTools/usr/include/c++/v1

MacOS 10.15 Catalina:

.. code:: Bash

    CFLAGS=-stdlib=libc++  pip install smac
