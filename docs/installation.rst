Installation
============

Lale supports Python 3.6+.  

Install Lale Core
-------------------

    Lale's core functionality such as core Lale operators, pipelines, lifecycle stages etc. is packaged into Lale Core and 
    has a smaller set of dependencies than Lale full.

    Install as::

        pip install git+https://git@github.com/IBM/lale.git

    or install from source::

        git clone https://github.com/IBM/lale.git
        cd lale
        pip install .


Install Lale Full
-------------------

    The full set of functionality also includes Lale wrappers to some deep learning models, 
    support for SMAC etc. 

    Install as::

        pip install git+https://git@github.com/IBM/lale.git#egg=lale[full]

    or install from source::

        git clone https://github.com/IBM/lale.git
        cd lale
        pip install .[full]

