# Lale Installation Instructions

We need your help: if you find these instructions incomplete or wrong,
please let us know, so we can improve them! Either reach out to us
directly, or post on the `#lale-users` Slack channel in the IBM
Research org. With your help, we can make the Lale installation
a delightful experience.

### 1. On Windows 10

First, you should enable the Windows Subsystem for Linux (WSL):

- Start/search -> Settings -> Update&Security -> For developers -> activate "Developer Mode"
- Start/search -> Turn Windows features on or off -> WSL -> restart
- Internet Explorer -> https://aka.ms/wslstore -> Ubuntu -> get
- Start/search -> Ubuntu

At this point, you can continue with the instructions in section
"1.2. On Ubuntu Linux".

### 2. On Ubuntu Linux

Start by making sure your Ubuntu installation is up-to-date and check
the version. In a command shell, type:

```bash
sudo apt update
sudo apt upgrade
lsb_release -a
```

This should output something like "Description: Ubuntu 16.04.4 LTS".

Also, make sure you have g++, make, graphviz, and swig installed. Otherwise, you can install them:
```bash
sudo apt install g++
sudo apt install graphviz
sudo apt install make
sudo apt install swig
```

Next, set up a Python virtual environment with Python 3.6.

```bash
sudo add-apt-repository ppa:jonathonf/python-3.6
sudo apt update
sudo apt install python3.6
sudo apt install python3.6-dev
sudo apt install virtualenv
virtualenv -p /usr/bin/python3.6 ~/python3.6venv
source ~/python3.6venv/bin/activate
```

At this point, you can continue with the instructions in section
"3. In a Python 3.6 Virtual Environment".

### 3. In a Python 3.6 Virtual Environment

These instructions here should resemble the steps in the top-level
[.travis.yml](../master/.travis.yml) file. When in doubt, that file is
more likely to be up-to-date than these instructions here. First, you
need to clone the github repository. In a command shell, type:

```bash
git clone git@github.ibm.com:aimodels/lale.git
cd lale
```

Next, you will need to install some Python packages in your local
environment. Eliding `full` will speed up the installation at the
expense of only enabling a subset of the functionality.

```bash
pip install numpy
pip install .[full,test]
export PYTHONPATH=`pwd`
```

Now, you are ready to run some tests. For a quick check, start with:

```bash
python -m unittest test.test.TestLogisticRegression
```

This should say something like

```
Ran 20 tests in 105.201s
OK
```

To run the full test suite, use:

```
make run_tests
```

#### Mac OSX issues following a virtual environment install.
Install swig using brew before you install LALE.

For issues in installing SMAC
```
open /Library/Developer/CommandLineTools/Packages/macOS_SDK_headers_for_macOS_10.14.pkg
```
Then
```
export CPATH=/Library/Developer/CommandLineTools/usr/include/c++/v1
```

### 4. In a Jupyter Notebook

If you have followed the preceding instructions to set up a Python 3.6
virtual environment, then at this point, you should be able to write
code in a notebook. In a command shell, type:

```
make launch_notebook
```

This should open a tab in your web browser with a file browser. Go to
`examples -> demo_help_and_errors.ipynb` for some simple API calls.
Or create your own new notebook to start using Lale.
