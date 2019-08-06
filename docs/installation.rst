Installation
============

Lale supports Python 3.6+.  

Install Lale Core
-------------------

    Lale's core functionality such as core Lale operators, pipelines, lifecycle stages etc. is packaged into Lale Core and 
    has a smaller set of dependencies than Lale full.

    Install as::

        pip install git+ssh://git@github.ibm.com/Lale/lale

    or install from source::

        git clone git@github.ibm.com:Lale/lale.git
        cd lale
        pip install .


Install Lale Full
-------------------

    The full set of functionality also includes Lale wrappers to some deep learning models, 
    support for SMAC etc. 

    Install as::

        pip install numpy && pip install git+ssh://git@github.ibm.com/Lale/lale.git#egg=lale[full]

    or install from source::

        git clone git@github.ibm.com:Lale/lale.git
        cd lale
        pip install .[full]

