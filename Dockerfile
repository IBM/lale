ARG python_version=3.7.3
FROM python:${python_version}-stretch

RUN echo 'deb http://lib.stat.cmu.edu/R/CRAN/bin/linux/debian stretch-cran35/' >> /etc/apt/sources.list

RUN apt-get update \
&& apt-get install --assume-yes graphviz build-essential swig \
&& apt-get install --assume-yes dirmngr apt-transport-https ca-certificates software-properties-common gnupg2 

RUN pip install numpy
COPY setup.py README.md /lale/
WORKDIR /lale
# First install the dependencies
RUN pip install .[full,test]

COPY . /lale

RUN pip install .[full,test]

ENV PYTHONPATH /lale

