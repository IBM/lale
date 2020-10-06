# Copyright 2019 IBM Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
from functools import wraps

import lale.docstrings
import lale.operators

logger = logging.getLogger(__name__)
# since we want the LoggingObserver

logger.setLevel(logging.INFO)


def observe(f):
    @wraps(f)
    def wrapper(self, *args, **kwds):
        name = f.__name__
        self.startObserving(name, *args, **kwds)
        try:
            ret = f(self, *args, **kwds)
            self.endObserving(name, ret)
        except BaseException as e:
            self.failObserving(name, e)
            raise
        return ret

    return wrapper


start_prefix = "start_"
end_prefix = "end_"
fail_prefix = "fail_"


class ObservingImpl:
    def __init__(self, op=None, observer=None):
        if observer is not None and isinstance(observer, type):
            # if we are given a class name, instantiate it
            observer = observer()
        self._hyperparams = {"op": op, "observer": observer}

    def getOp(self):
        return self._hyperparams["op"]

    def getObserver(self):
        return self._hyperparams["observer"]

    def _observe(self, methodName, *args, **kwargs):
        o = self.getObserver()
        if o is not None:
            m = getattr(o, methodName, None)
            if m is not None:
                m(self.getOp(), *args, **kwargs)

    def startObserving(self, methodName, *args, **kwargs):
        self._observe(f"{start_prefix}{methodName}", *args, **kwargs)

    def endObserving(self, methodName, *args, **kwargs):
        self._observe(f"{end_prefix}{methodName}", *args, **kwargs)

    def failObserving(self, methodName, e: BaseException):
        self._observe(f"{fail_prefix}{methodName}", e)

    @observe
    def transform(self, X, y=None):
        ret = self.getOp().transform(X, y=y)
        self.endObserving("transform", ret)
        return ret

    @observe
    def transform_schema(self, s_X):
        return self.getOp().transform_schema(s_X)

    @observe
    def input_schema_fit(self):
        return self.getOp().input_schema_fit()

    @observe
    def predict(self, X):
        return self.getOp().predict(X)

    @observe
    def predict_proba(self, X):
        return self.getOp().predict_proba(self, X)

    @observe
    def fit(self, X, y=None):
        return self.getOp().fit(X, y=y)


_hyperparams_schema = {
    "description": "Hyperparameter schema for the identity Higher Order Operator, which wraps another operator and runs it as usual",
    "allOf": [
        {
            "description": "This first object lists all constructor arguments with their types, but omits constraints for conditional hyperparameters",
            "type": "object",
            "additionalProperties": False,
            "relevantToOptimizer": ["op"],
            "properties": {
                "op": {"laleType": "operator"},
                "observer": {"laleType": "Any"},
            },
        }
    ],
}

# TODO: can we surface the base op input/output schema?
_input_fit_schema = {
    "description": "Input data schema for training identity.",
    "type": "object",
    "required": ["X"],
    "additionalProperties": False,
    "properties": {"X": {}},
}

_input_predict_transform_schema = {  # TODO: separate predict vs. predict_proba vs. transform
    "description": "Input data schema for transformations using identity.",
    "type": "object",
    "required": ["X", "y"],
    "additionalProperties": False,
    "properties": {"X": {}, "y": {}},
}

_output_schema = {  # TODO: separate predict vs. predict_proba vs. transform
    "description": "Output data schema for transformations using identity.",
    "laleType": "Any",
}

_combined_schemas = {
    "$schema": "http://json-schema.org/draft-04/schema#",
    "description": """This should functionally be identical to the identity wrapper, except that it calls methods on the observer (if they exist) before and after calls to the underlying wrapper. This is similar to aspect-oriented programming.""",
    "documentation_url": "https://lale.readthedocs.io/en/latest/modules/lale.lib.lale.identity.html",
    "import_from": "lale.lib.lale",
    "type": "object",
    "tags": {"pre": [], "op": ["estimator", "transformer"], "post": []},
    "properties": {
        "hyperparams": _hyperparams_schema,
        "input_fit": _input_fit_schema,
        "input_predict": _input_predict_transform_schema,
        "output_predict": _output_schema,
        "input_predict_proba": _input_predict_transform_schema,
        "output_predict_proba": _output_schema,
        "input_transform": _input_predict_transform_schema,
        "output_transform": _output_schema,
    },
}

lale.docstrings.set_docstrings(ObservingImpl, _combined_schemas)

Observing = lale.operators.make_operator(ObservingImpl, _combined_schemas)


class LoggingObserver:
    """An observer that logs everything.
        This is also useful for debugging, since you can set breakpoints here
    """

    _indent: int

    def __init__(self):
        self._indent = 0

    def __getattr__(self, prop: str):
        if prop.startswith("_"):
            raise AttributeError
        elif prop.startswith(start_prefix):
            suffix = prop[len(start_prefix) :]

            def startfun(*args, **kwargs):
                if logger.isEnabledFor(logging.INFO):
                    s: str = "  " * self._indent
                    s += f"[observing({suffix})->] "
                    s += ",".join(map(str, args))
                    if len(args) > 0 and len(kwargs) > 0:
                        s += ", "
                    for k, v in kwargs.items():
                        s += f"{k}->{v}"
                    logger.info(s)
                self._indent += 1

            return startfun
        elif prop.startswith(end_prefix):
            suffix = prop[len(end_prefix) :]

            def endfun(*args, **kwargs):
                assert self._indent > 0
                self._indent -= 1
                if logger.isEnabledFor(logging.INFO):
                    s: str = "  " * self._indent
                    s += f"[<-observed({suffix})] "
                    s += ",".join(map(str, args))
                    for k, v in kwargs.items():
                        s += f"{k}->{v}"
                    logger.info(s)

            return endfun
        elif prop.startswith(fail_prefix):
            suffix = prop[len(fail_prefix) :]

            def failfun(*args, **kwargs):
                assert self._indent > 0
                self._indent -= 1
                if logger.isEnabledFor(logging.INFO):
                    s: str = "  " * self._indent
                    s += f"[!error!<-observed({suffix})] "
                    s += ",".join(map(str, args))
                    for k, v in kwargs.items():
                        s += f"{k}->{v}"
                    logger.info(s)

            return failfun
        else:
            logger.debug(f"trying to observe {prop}, which is not a start or stop")
