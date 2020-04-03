all: run_tests

run_tests:
	export PYTHONPATH=`pwd`; \
	python -m unittest discover -v

check:
	python -m compileall .
	flake8 . --count --select=E9,F63,F72,F82 --show-source --statistics
	mypy -p lale

PYTHON_VERSION ?= 3.6

docker_build: docker_build$(PYTHON_VERSION)
docker_run: docker_run$(PYTHON_VERSION)
docker: docker$(PYTHON_VERSION)

docker_build%: 
	docker build --build-arg=python_version=$* -t lale$* .

docker_run%:
	docker run -it lale$* bash

docker%: docker_build% docker_run%
# I am not sure why this is needed, but Make does not work without it
	- true

launch_notebook:
	export PYTHONPATH=`pwd`; \
	jupyter notebook

clean:
	-rm -rf lale/__pycache__
	-rm -rf lale/*.pyc
	-rm -rf lale/tests/__pycache__
	-rm -rf test/__pycache__
	-rm -rf test/*.pyc
	-rm -rf examples/.ipynb_checkpoints
	-rm -rf lale/search/__pycache__
	-rm -rf lale/lib/lale/__pycache__
.PHONY: clean docker% docker_build% docker_run%
