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

"""Classes for Lale operators including individual operators, pipelines, and operator choice.

This module declares several functions for constructing individual
operators, pipelines, and operator choices.

- Functions `make_pipeline`_ and `Pipeline`_ compose linear sequential
  pipelines, where each step has an edge to the next step. Instead of
  these functions you can also use the `>>` combinator.

- Functions `make_union_no_concat`_ and `make_union`_ compose
  pipelines that operate over the same data without edges between
  their steps. Instead of these functions you can also use the `&`
  combinator.

- Function `make_choice` creates an operator choice. Instead of this
  function you can also use the `|` combinator.

- Function `make_pipeline_graph`_ creates a pipeline from
  steps and edges, thus supporting any arbitrary acyclic directed
  graph topology.

- Function `make_operator`_ creates an individual Lale operator from a
  schema and an implementation class or object. This is called for each
  of the operators in module lale.lib when it is being imported.

- Functions `get_available_operators`_, `get_available_estimators`_,
  and `get_available_transformers`_ return lists of individual
  operators previously registered by `make_operator`.

.. _make_operator: lale.operators.html#lale.operators.make_operator
.. _get_available_operators: lale.operators.html#lale.operators.get_available_operators
.. _get_available_estimators: lale.operators.html#lale.operators.get_available_estimators
.. _get_available_transformers: lale.operators.html#lale.operators.get_available_transformers
.. _make_pipeline_graph: lale.operators.html#lale.operators.make_pipeline_graph
.. _make_pipeline: lale.operators.html#lale.operators.make_pipeline
.. _Pipeline: Lale.Operators.Html#Lale.Operators.Pipeline
.. _make_union_no_concat: lale.operators.html#lale.operators.make_union_no_concat
.. _make_union: lale.operators.html#lale.operators.make_union
.. _make_choice: lale.operators.html#lale.operators.make_choice

The root of the hierarchy is the abstract class Operator_, all other
Lale operators inherit from this class, either directly or indirectly.

- The abstract classes Operator_, PlannedOperator_,
  TrainableOperator_, and TrainedOperator_ correspond to lifecycle
  states.

- The concrete classes IndividualOp_, PlannedIndividualOp_,
  TrainableIndividualOp_, and TrainedIndividualOp_ inherit from the
  corresponding abstract operator classes and encapsulate
  implementations of individual operators from machine-learning
  libraries such as scikit-learn.

- The concrete classes BasePipeline_, PlannedPipeline_,
  TrainablePipeline_, and TrainedPipeline_ inherit from the
  corresponding abstract operator classes and represent directed
  acyclic graphs of operators. The steps of a pipeline can be any
  operators, including individual operators, other pipelines, or
  operator choices, whose lifecycle state is at least that of the
  pipeline.

- The concrete class OperatorChoice_ represents a planned operator
  that offers a choice for automated algorithm selection. The steps of
  a choice can be any planned operators, including individual
  operators, pipelines, or other operator choices.

The following picture illustrates the core operator class hierarchy.

.. image:: ../../docs/img/operator_classes.png
  :alt: operators class hierarchy

.. _BasePipeline: lale.operators.html#lale.operators.BasePipeline
.. _IndividualOp: lale.operators.html#lale.operators.IndividualOp
.. _Operator: lale.operators.html#lale.operators.Operator
.. _OperatorChoice: lale.operators.html#lale.operators.OperatorChoice
.. _PlannedIndividualOp: lale.operators.html#lale.operators.PlannedIndividualOp
.. _PlannedOperator: lale.operators.html#lale.operators.PlannedOperator
.. _PlannedPipeline: lale.operators.html#lale.operators.PlannedPipeline
.. _TrainableIndividualOp: lale.operators.html#lale.operators.TrainableIndividualOp
.. _TrainableOperator: lale.operators.html#lale.operators.TrainableOperator
.. _TrainablePipeline: lale.operators.html#lale.operators.TrainablePipeline
.. _TrainedIndividualOp: lale.operators.html#lale.operators.TrainedIndividualOp
.. _TrainedOperator: lale.operators.html#lale.operators.TrainedOperator
.. _TrainedPipeline: lale.operators.html#lale.operators.TrainedPipeline

"""

import copy
import enum as enumeration
import inspect
import logging
import os
import shutil
import time
import warnings
from abc import abstractmethod
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import jsonschema
import pandas as pd
import sklearn.base
from sklearn.pipeline import if_delegate_has_method

import lale.datasets.data_schemas
import lale.helpers
import lale.json_operator
import lale.pretty_print
import lale.type_checking
from lale import schema2enums as enum_gen
from lale.json_operator import JSON_TYPE
from lale.schemas import Schema
from lale.search.PGO import remove_defaults_dict
from lale.util.VisitorMeta import AbstractVisitorMeta

logger = logging.getLogger(__name__)

_LALE_SKL_PIPELINE = "lale.lib.sklearn.pipeline.PipelineImpl"

_combinators_docstrings = """
    Methods
    -------

    step_1 >> step_2 -> PlannedPipeline
        Pipe combinator, create two-step pipeline with edge from step_1 to step_2.

        If step_1 is a pipeline, create edges from all of its sinks.
        If step_2 is a pipeline, create edges to all of its sources.

        Parameters
        ^^^^^^^^^^
        step_1 : Operator
            The origin of the edge(s).
        step_2 : Operator
            The destination of the edge(s).

        Returns
        ^^^^^^^
        BasePipeline
            Pipeline with edge from step_1 to step_2.

    step_1 & step_2 -> PlannedPipeline
        And combinator, create two-step pipeline without an edge between step_1 and step_2.

        Parameters
        ^^^^^^^^^^
        step_1 : Operator
            The first step.
        step_2 : Operator
            The second step.

        Returns
        ^^^^^^^
        BasePipeline
            Pipeline without any additional edges beyond those already inside of step_1 or step_2.

    step_1 | step_2 -> OperatorChoice
        Or combinator, create operator choice between step_1 and step_2.

        Parameters
        ^^^^^^^^^^
        step_1 : Operator
            The first step.
        step_2 : Operator
            The second step.

        Returns
        ^^^^^^^
        OperatorChoice
            Algorithmic coice between step_1 or step_2."""


class Operator(metaclass=AbstractVisitorMeta):
    """Abstract base class for all Lale operators.

    Pipelines and individual operators extend this."""

    _name: str

    def __and__(self, other: "Operator") -> "PlannedPipeline":
        return make_union_no_concat(self, other)

    def __rand__(self, other: "Operator") -> "PlannedPipeline":
        return make_union_no_concat(other, self)

    def __rshift__(self, other: "Operator") -> "PlannedPipeline":
        return make_pipeline(self, other)

    def __rrshift__(self, other: "Operator") -> "PlannedPipeline":
        return make_pipeline(other, self)

    def __or__(self, other: "Operator") -> "OperatorChoice":
        return make_choice(self, other)

    def __ror__(self, other: "Operator") -> "OperatorChoice":
        return make_choice(other, self)

    def name(self) -> str:
        """Get the name of this operator instance."""
        return self._name

    def _set_name(self, name: str):
        """Set the name of this operator instance."""
        self._name = name

    def class_name(self) -> str:
        """Fully qualified Python class name of this operator."""
        cls = self.__class__
        return cls.__module__ + "." + cls.__name__

    @abstractmethod
    def validate_schema(self, X, y=None):
        """Validate that X and y are valid with respect to the input schema of this operator.

        Parameters
        ----------
        X :
            Features.
        y :
            Target class labels or None for unsupervised operators.

        Raises
        ------
        ValueError
            If X or y are invalid as inputs."""
        pass

    @abstractmethod
    def transform_schema(self, s_X) -> JSON_TYPE:
        """Return the output schema given the input schema.

        Parameters
        ----------
        s_X :
            Input dataset or schema.

        Returns
        -------
        JSON schema
            Schema of the output data given the input data schema."""
        pass

    @abstractmethod
    def input_schema_fit(self) -> JSON_TYPE:
        """Input schema for the fit method."""
        pass

    def to_json(self) -> JSON_TYPE:
        """Returns the JSON representation of the operator.

        Returns
        -------
        JSON document
            JSON representation that describes this operator and is valid with respect to lale.json_operator.SCHEMA.
        """
        return lale.json_operator.to_json(self, call_depth=2)

    def visualize(self, ipython_display: bool = True):
        """Visualize the operator using graphviz (use in a notebook).

        Parameters
        ----------
        ipython_display : bool, default True
            If True, proactively ask Jupyter to render the graph.
            Otherwise, the graph will only be rendered when visualize()
            was called in the last statement in a notebook cell.

        Returns
        -------
        Digraph
            Digraph object from the graphviz package.
        """
        return lale.helpers.to_graphviz(self, ipython_display, call_depth=2)

    def pretty_print(
        self,
        show_imports: bool = True,
        combinators: bool = True,
        astype: str = "lale",
        ipython_display: Union[bool, str] = False,
    ):
        """Returns the Python source code representation of the operator.

        Parameters
        ----------
        show_imports : bool, default True

            Whether to include import statements in the pretty-printed code.

        combinators : bool, default True

            If True, pretty-print with combinators (`>>`, `|`, `&`). Otherwise, pretty-print with functions (`make_pipeline`, `make_choice`, `make_union`) instead. Always False when astype is 'sklearn'.

        astype : union type, default 'lale'

            - 'lale'

              Use `lale.operators.make_pipeline` and `lale.operators.make_union` when pretty-printing wth functions.

            - 'sklearn'

              Set combinators to False and use `sklearn.pipeline.make_pipeline` and `sklearn.pipeline.make_union` for pretty-printed functions.

        ipython_display : union type, default False

            - False

              Return the pretty-printed code as a plain old Python string.

            - True:

              Pretty-print in notebook cell output with syntax highlighting.

            - 'input'

              Create a new notebook cell with pretty-printed code as input.

        Returns
        -------
        str or None
            If called with ipython_display=False, return pretty-printed Python source code as a Python string.
        """
        result = lale.pretty_print.to_string(
            self, show_imports, combinators, astype, call_depth=2
        )
        if ipython_display is False:
            return result
        elif ipython_display == "input":
            import IPython.core

            ipython = IPython.core.getipython.get_ipython()
            comment = "# generated by pretty_print(ipython_display='input') from previous cell\n"
            ipython.set_next_input(comment + result, replace=False)
        else:
            assert ipython_display in [True, "output"]
            import IPython.display

            markdown = IPython.display.Markdown(f"```python\n{result}\n```")
            return IPython.display.display(markdown)

    @abstractmethod
    def _has_same_impl(self, other: "Operator") -> bool:
        """Checks if the type of the operator implementations are compatible
        """
        pass

    @abstractmethod
    def _lale_clone(self, cloner: Callable[[Any], Any]):
        """ Method for cloning a lale operator, currently intended for internal use
        """
        pass

    @abstractmethod
    def is_supervised(self) -> bool:
        """Checks if this operator needs labeled data for learning.

        Returns
        -------
        bool
            True if the fit method requires a y argument.
        """
        pass

    @abstractmethod
    def is_classifier(self) -> bool:
        """Checks if this operator is a clasifier.

        Returns
        -------
        bool
            True if the classifier tag is set.
        """
        pass

    @abstractmethod
    def freeze_trainable(self) -> "Operator":
        """Return a copy of the trainable parts of this operator that is the same except
        that all hyperparameters are bound and none are free to be tuned.
        If there is an operator choice, it is kept as is.
        """
        pass

    @abstractmethod
    def is_frozen_trainable(self) -> bool:
        """Return true if all hyperparameters are bound, in other words,
        search spaces contain no free hyperparameters to be tuned.
        """
        pass


Operator.__doc__ = cast(str, Operator.__doc__) + "\n" + _combinators_docstrings


class PlannedOperator(Operator):
    """Abstract class for Lale operators in the planned lifecycle state."""

    def auto_configure(
        self, X, y=None, optimizer=None, cv=None, scoring=None, **kwargs
    ) -> "TrainableOperator":
        """
        Perform combined algorithm selection and hyperparameter tuning on this planned operator.

        Parameters
        ----------
        X:
            Features that conform to the X property of input_schema_fit.
        y: optional
            Labels that conform to the y property of input_schema_fit.
            Default is None.
        optimizer:
            lale.lib.lale.HyperoptCV or lale.lib.lale.GridSearchCV
            default is None.
        cv:
            cross-validation option that is valid for the optimizer.
            Default is None, which will use the optimizer's default value.
        scoring:
            scoring option that is valid for the optimizer.
            Default is None, which will use the optimizer's default value.
        kwargs:
            Other keyword arguments to be passed to the optimizer.

        Returns
        -------
        TrainableOperator
            Best operator discovered by the optimizer.
        """
        if optimizer is None:
            raise ValueError("Please provide a valid optimizer for auto_configure.")
        if kwargs is None:
            kwargs = {}
        if cv is not None:
            kwargs["cv"] = cv
        if scoring is not None:
            kwargs["scoring"] = scoring
        optimizer_obj = optimizer(estimator=self, **kwargs)
        trained = optimizer_obj.fit(X, y)
        return trained.get_pipeline()


PlannedOperator.__doc__ = (
    cast(str, PlannedOperator.__doc__) + "\n" + _combinators_docstrings
)


class TrainableOperator(PlannedOperator):
    """Abstract class for Lale operators in the trainable lifecycle state."""

    @abstractmethod
    def fit(self, X, y=None, **fit_params) -> "TrainedOperator":
        """Train the learnable coefficients of this operator, if any.

        Return a trained version of this operator.  If this operator
        has free learnable coefficients, bind them to values that fit
        the data according to the operator's algorithm.  Do nothing if
        the operator implementation lacks a `fit` method or if the
        operator has been marked as `is_frozen_trained`.

        Parameters
        ----------
        X:
            Features that conform to the X property of input_schema_fit.
        y: optional
            Labels that conform to the y property of input_schema_fit.
            Default is None.
        fit_params: Dictionary, optional
            A dictionary of keyword parameters to be used during training.

        Returns
        -------
        TrainedOperator
            A new copy of this operators that is the same except that its
            learnable coefficients are bound to their trained values.

        """
        pass

    @abstractmethod
    def is_transformer(self) -> bool:
        """ Checks if the operator is a transformer
        """
        pass


TrainableOperator.__doc__ = (
    cast(str, TrainableOperator.__doc__) + "\n" + _combinators_docstrings
)


class TrainedOperator(TrainableOperator):
    """Abstract class for Lale operators in the trained lifecycle state."""

    @abstractmethod
    def transform(self, X, y=None):
        """Transform the data.

        Parameters
        ----------
        X :
            Features; see input_transform schema of the operator.

        Returns
        -------
        result :
            Transformed features; see output_transform schema of the operator.
        """
        pass

    @abstractmethod
    def _predict(self, X):
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions.

        Parameters
        ----------
        X :
            Features; see input_predict schema of the operator.

        Returns
        -------
        result :
            Predictions; see output_predict schema of the operator.
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """Probability estimates for all classes.

        Parameters
        ----------
        X :
            Features; see input_predict_proba schema of the operator.

        Returns
        -------
        result :
            Probabilities; see output_predict_proba schema of the operator.
        """
        pass

    @abstractmethod
    def decision_function(self, X):
        """Confidence scores for all classes.

        Parameters
        ----------
        X :
            Features; see input_decision_function schema of the operator.

        Returns
        -------
        result :
            Confidences; see output_decision_function schema of the operator.
        """
        pass

    @abstractmethod
    def is_frozen_trained(self) -> bool:
        """Return true if all learnable coefficients are bound, in other
           words, there are no free parameters to be learned by fit.
        """
        pass

    @abstractmethod
    def freeze_trained(self) -> "TrainedOperator":
        """Return a copy of this trainable operator that is the same except
           that all learnable coefficients are bound and thus fit is a no-op.
        """
        pass


TrainedOperator.__doc__ = (
    cast(str, TrainedOperator.__doc__) + "\n" + _combinators_docstrings
)

_schema_derived_attributes = ["_enum_attributes", "_hyperparam_defaults"]


class _DictionaryObjectForEnum:
    _d: Dict[str, enumeration.Enum]

    def __init__(self, d: Dict[str, enumeration.Enum]):
        self._d = d

    def __contains__(self, key: str) -> bool:
        return key in self._d

    def __getattr__(self, key: str) -> enumeration.Enum:
        if key in self._d:
            return self._d[key]
        else:
            raise AttributeError("No enumeration found for hyper-parameter: " + key)

    def __getitem__(self, key: str) -> enumeration.Enum:
        if key in self._d:
            return self._d[key]
        else:
            raise KeyError("No enumeration found for hyper-parameter: " + key)


class IndividualOp(Operator):
    """
    This is a concrete class that can instantiate a new individual
    operator and provide access to its metadata.
    The enum property can be used to access enumerations for hyper-parameters,
    auto-generated from the operator's schema.
    For example, `LinearRegression.enum.solver.saga`
    As a short-hand, if the hyper-parameter name does not conflict with
    any fields of this class, the auto-generated enums can also be accessed
    directly.
    For example, `LinearRegression.solver.saga`"""

    _name: str
    _impl: Any

    def __init__(self, name: str, impl, schemas) -> None:
        """Create a new IndividualOp.

        Parameters
        ----------
        name : String
            Name of the operator.
        impl :
            An instance of operator implementation class. This is a class that
            contains fit, predict/transform methods implementing an underlying
            algorithm.
        schemas : dict
            This is a dictionary of json schemas for the operator.
        """
        self._impl = impl
        self._name = name
        self._enum_attributes = None
        if schemas:
            self._schemas = schemas
        else:
            self._schemas = lale.type_checking.get_default_schema(impl)

    def _check_schemas(self):
        lale.type_checking.validate_is_schema(self._schemas)
        from lale.pretty_print import json_to_string

        assert (
            self.has_tag("transformer") == self.is_transformer()
        ), f"{self.class_name()}: {json_to_string(self._schemas)}"
        assert self.has_tag("estimator") == hasattr(
            self._impl, "predict"
        ), f"{self.class_name()}: {json_to_string(self._schemas)}"
        if self.has_tag("classifier") or self.has_tag("regressor"):
            assert self.has_tag(
                "estimator"
            ), f"{self.class_name()}: {json_to_string(self._schemas)}"

        # Add enums from the hyperparameter schema to the object as fields
        # so that their usage looks like LogisticRegression.penalty.l1

    #        enum_gen.addSchemaEnumsAsFields(self, self.hyperparam_schema())

    _enum_attributes: Optional[_DictionaryObjectForEnum]

    @property
    def enum(self) -> _DictionaryObjectForEnum:
        ea = getattr(self, "_enum_attributes", None)
        if ea is None:
            nea = enum_gen.schemaToPythonEnums(self.hyperparam_schema())
            doe = _DictionaryObjectForEnum(nea)
            self._enum_attributes = doe
            return doe
        else:
            return ea

    def _invalidate_enum_attributes(self) -> None:
        for k in _schema_derived_attributes:
            try:
                delattr(self, k)
            except AttributeError:
                pass

    def __getattr__(self, name: str):
        if name in _schema_derived_attributes or name in ["__setstate__", "_schemas"]:
            raise AttributeError

        if name in [
            "get_pipeline",
            "summary",
            "transform",
            "predict",
            "predict_proba",
            "decision_function",
        ]:
            if isinstance(self, TrainedIndividualOp):
                raise AttributeError(
                    f"The underlying operator impl does not define {name}"
                )
            elif isinstance(self, TrainableIndividualOp):
                raise AttributeError(
                    f"The underlying operator impl does not define {name}.  Also, calling {name} on a TrainableOperator is deprecated.  Perhaps you meant to train this operator first?  Note that in lale, the result of fit is a new TrainedOperator that should be used with {name}."
                )
            else:
                raise AttributeError(
                    f"Calling {name} on a TrainableOperator is deprecated.  Perhaps you meant to train this operator first?  Note that in lale, the result of fit is a new TrainedOperator that should be used with {name}."
                )

        if name == "_estimator_type":
            if self.is_classifier():
                return "classifier"  # satisfy sklearn.base.is_classifier(op)

        ea = self.enum
        if name in ea:
            return ea[name]
        else:
            raise AttributeError

    def __getstate__(self):
        state = self.__dict__.copy()
        # Remove entries that can't be pickled
        for k in _schema_derived_attributes:
            state.pop(k, None)
        return state

    def get_schema(self, schema_kind: str) -> Dict[str, Any]:
        """Return a schema of the operator.

        Parameters
        ----------
        schema_kind : string, 'hyperparams' or 'input_fit' or 'input_transform'  or 'input_predict' or 'input_predict_proba' or 'input_decision_function' or 'output_transform' or 'output_predict' or 'output_predict_proba' or 'output_decision_function'
                Type of the schema to be returned.

        Returns
        -------
        dict
            The python object containing the json schema of the operator.
            For all the schemas currently present, this would be a dictionary.
        """
        props = self._schemas["properties"]
        assert (
            schema_kind in props
        ), f"missing schema {schema_kind} for operator {self.name()} with class {self.class_name()}"
        result = props[schema_kind]
        return result

    def documentation_url(self):
        if "documentation_url" in self._schemas:
            return self._schemas["documentation_url"]
        return None

    def get_tags(self) -> Dict[str, List[str]]:
        """Return the tags of an operator.

        Returns
        -------
        list
            A list of tags describing the operator.
        """
        return self._schemas.get("tags", {})

    def has_tag(self, tag: str) -> bool:
        """Check the presence of a tag for an operator.

        Parameters
        ----------
        tag : string

        Returns
        -------
        boolean
            Flag indicating the presence or absence of the given tag
            in this operator's schemas.
        """
        tags = [t for ll in self.get_tags().values() for t in ll]
        return tag in tags

    def input_schema_fit(self) -> JSON_TYPE:
        """Input schema for the fit method."""
        return self.get_schema("input_fit")

    def input_schema_transform(self) -> JSON_TYPE:
        """Input schema for the transform method."""
        return self.get_schema("input_transform")

    def input_schema_predict(self) -> JSON_TYPE:
        """Input schema for the predict method."""
        return self.get_schema("input_predict")

    def input_schema_predict_proba(self) -> JSON_TYPE:
        """Input schema for the predict_proba method."""
        return self.get_schema("input_predict_proba")

    def input_schema_decision_function(self) -> JSON_TYPE:
        """Input schema for the decision_function method."""
        return self.get_schema("input_decision_function")

    def output_schema_transform(self) -> JSON_TYPE:
        """Oputput schema for the transform method."""
        return self.get_schema("output_transform")

    def output_schema_predict(self) -> JSON_TYPE:
        """Output schema for the predict method."""
        return self.get_schema("output_predict")

    def output_schema_predict_proba(self) -> JSON_TYPE:
        """Output schema for the predict_proba method."""
        return self.get_schema("output_predict_proba")

    def output_schema_decision_function(self) -> JSON_TYPE:
        """Output schema for the decision_function method."""
        return self.get_schema("output_decision_function")

    def hyperparam_schema(self, name: Optional[str] = None) -> JSON_TYPE:
        """Returns the hyperparameter schema for the operator.

        Parameters
        ----------
        name : string, optional
            Name of the hyperparameter.

        Returns
        -------
        dict
            Full hyperparameter schema for this operator or part of the schema
            corresponding to the hyperparameter given by parameter `name`.
        """
        hp_schema = self.get_schema("hyperparams")
        if name is None:
            return hp_schema
        else:
            params = next(iter(hp_schema.get("allOf", [])))
            return params.get("properties", {}).get(name)

    def hyperparam_defaults(self):
        """Returns the default values of hyperparameters for the operator.

        Returns
        -------
        dict
            A dictionary with names of the hyperparamers as keys and
            their default values as values.
        """
        if not hasattr(self, "_hyperparam_defaults"):
            schema = self.hyperparam_schema()
            props = next(iter(schema.get("allOf", [])), {}).get("properties", {})
            defaults = {k: props[k].get("default") for k in props.keys()}
            self._hyperparam_defaults = defaults
        return self._hyperparam_defaults

    def get_param_ranges(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Returns two dictionaries, ranges and cat_idx, for hyperparameters.

        The ranges dictionary has two kinds of entries. Entries for
        numeric and Boolean hyperparameters are tuples of the form
        (min, max, default). Entries for categorical hyperparameters
        are lists of their values.

        The cat_idx dictionary has (min, max, default) entries of indices
        into the corresponding list of values.

        Warning: ignores side constraints and unions."""

        hyperparam_obj = next(iter(self.hyperparam_schema().get("allOf", [])))
        original = hyperparam_obj.get("properties")

        def is_relevant(hp, s):
            if "relevantToOptimizer" in hyperparam_obj:
                return hp in hyperparam_obj["relevantToOptimizer"]
            return True

        relevant = {hp: s for hp, s in original.items() if is_relevant(hp, s)}

        def pick_one_type(schema):
            if "anyOf" in schema:

                def by_type(typ):
                    for s in schema["anyOf"]:
                        if "type" in s and s["type"] == typ:
                            if ("forOptimizer" not in s) or s["forOptimizer"]:
                                return s
                    return None

                for typ in ["number", "integer", "string"]:
                    s = by_type(typ)
                    if s:
                        return s
                if s is None:
                    for s in schema["anyOf"]:
                        if "enum" in s:
                            if ("forOptimizer" not in s) or s["forOptimizer"]:
                                return s
                return schema["anyOf"][0]
            return schema

        unityped = {hp: pick_one_type(relevant[hp]) for hp in relevant}

        def add_default(schema):
            if "type" in schema:
                minimum, maximum = 0.0, 1.0
                if "minimumForOptimizer" in schema:
                    minimum = schema["minimumForOptimizer"]
                elif "minimum" in schema:
                    minimum = schema["minimum"]
                if "maximumForOptimizer" in schema:
                    maximum = schema["maximumForOptimizer"]
                elif "maximum" in schema:
                    maximum = schema["maximum"]
                result = {**schema}
                if schema["type"] in ["number", "integer"]:
                    if "default" not in schema:
                        schema["default"] = None
                    if "minimumForOptimizer" not in schema:
                        result["minimumForOptimizer"] = minimum
                    if "maximumForOptimizer" not in schema:
                        result["maximumForOptimizer"] = maximum
                return result
            elif "enum" in schema:
                if "default" in schema:
                    return schema
                return {"default": schema["enum"][0], **schema}
            return schema

        defaulted = {hp: add_default(unityped[hp]) for hp in unityped}

        def get_range(hp, schema):
            if "enum" in schema:
                default = schema["default"]
                non_default = [v for v in schema["enum"] if v != default]
                return [*non_default, default]
            elif schema["type"] == "boolean":
                return (False, True, schema["default"])
            else:

                def get(schema, key):
                    return schema[key] if key in schema else None

                keys = ["minimumForOptimizer", "maximumForOptimizer", "default"]
                return tuple([get(schema, key) for key in keys])

        def get_cat_idx(schema):
            if "enum" not in schema:
                return None
            return (0, len(schema["enum"]) - 1, len(schema["enum"]) - 1)

        autoai_ranges = {hp: get_range(hp, s) for hp, s in defaulted.items()}
        if "min_samples_split" in autoai_ranges and "min_samples_leaf" in autoai_ranges:
            if self._impl.__name__ not in (
                "GradientBoostingRegressorImpl",
                "GradientBoostingClassifierImpl",
                "ExtraTreesClassifierImpl",
            ):
                autoai_ranges["min_samples_leaf"] = (1, 5, 1)
                autoai_ranges["min_samples_split"] = (2, 5, 2)
        autoai_cat_idx = {
            hp: get_cat_idx(s) for hp, s in defaulted.items() if "enum" in s
        }
        return autoai_ranges, autoai_cat_idx

    def get_param_dist(self, size=10) -> Dict[str, List[Any]]:
        """Returns a dictionary for discretized hyperparameters.

        Each entry is a list of values. For continuous hyperparameters,
        it returns up to `size` uniformly distributed values.

        Warning: ignores side constraints, unions, and distributions."""
        autoai_ranges, autoai_cat_idx = self.get_param_ranges()

        def one_dist(key: str) -> List[Any]:
            one_range = autoai_ranges[key]
            if isinstance(one_range, tuple):
                minimum, maximum, default = one_range
                if minimum is None:
                    dist = [default]
                elif isinstance(minimum, bool):
                    if minimum == maximum:
                        dist = [minimum]
                    else:
                        dist = [minimum, maximum]
                elif isinstance(minimum, int) and isinstance(maximum, int):
                    step = float(maximum - minimum) / (size - 1)
                    fdist = [minimum + i * step for i in range(size)]
                    dist = list(set([round(f) for f in fdist]))
                    dist.sort()
                elif isinstance(minimum, (int, float)):
                    # just in case the minimum or maximum is exclusive
                    epsilon = (maximum - minimum) / (100 * size)
                    minimum += epsilon
                    maximum -= epsilon
                    step = (maximum - minimum) / (size - 1)
                    dist = [minimum + i * step for i in range(size)]
                else:
                    assert False, f"key {key}, one_range {one_range}"
            else:
                dist = [*one_range]
            return dist

        autoai_dists = {k: one_dist(k) for k in autoai_ranges.keys()}
        return autoai_dists

    def _enum_to_strings(self, arg: "enumeration.Enum") -> Tuple[str, Any]:
        """[summary]

        Parameters
        ----------
        arg : [type]
            [description]

        Raises
        ------
        ValueError
            [description]

        Returns
        -------
        [type]
            [description]
        """

        if not isinstance(arg, enumeration.Enum):
            raise ValueError("Missing keyword on argument {}.".format(arg))
        return arg.__class__.__name__, arg.value

    def _impl_class(self):
        if inspect.isclass(self._impl):
            return self._impl
        return self._impl.__class__

    def _impl_instance(self):
        if inspect.isclass(self._impl):
            class_ = self._impl
            try:
                instance = class_()  # always with default values of hyperparams
            except TypeError as e:
                logger.debug(
                    f"Constructor for {class_.__module__}.{class_.__name__} "
                    f"threw exception {e}"
                )
                instance = class_.__new__(class_)
            self._impl = instance
        return self._impl

    def class_name(self) -> str:
        module = None
        if self._impl is not None:
            module = self._impl.__module__
        if module is None or module == str.__class__.__module__:
            class_name = self.name()
        else:
            class_name = module + "." + self._impl_class().__name__
        return class_name

    def __str__(self) -> str:
        return self.name()

    def _has_same_impl(self, other: Operator) -> bool:
        """Checks if the type of the operator implementations are compatible
        """
        if not isinstance(other, IndividualOp):
            return False
        return self._impl_class() == other._impl_class()

    def _lale_clone(self, cloner: Callable[[Any], Any]):
        impl = self._impl
        if impl is not None and not inspect.isclass(impl):
            impl = cloner(impl)
        cp = self.__class__(self._name, impl, self._schemas)
        return cp

    def customize_schema(
        self,
        schemas: Optional[Schema] = None,
        relevantToOptimizer: Optional[List[str]] = None,
        constraint: Optional[Schema] = None,
        tags: Optional[Dict] = None,
        **kwargs: Optional[Schema],
    ) -> "IndividualOp":
        """Return a new operator with a customized schema

        Parameters
        ----------
        schemas : Schema
            A dictionary of json schemas for the operator. Override the entire schema and ignore other arguments
        input : Schema
            (or `input_*`) override the input schema for method `*`.
            `input_*` must be an existing method (already defined in the schema for lale operators, existing method for external operators)
        output : Schema
            (or `output_*`) override the output schema for method `*`.
            `output_*` must be an existing method (already defined in the schema for lale operators, existing method for external operators)
        constraint : Schema
            Add a constraint in JSON schema format.
        relevantToOptimizer : String list
            update the set parameters that will be optimized.
        param : Schema
            Override the schema of the hyperparameter.
            `param` must be an existing parameter (already defined in the schema for lale operators, __init__ parameter for external operators)
        tags : Dict
            Override the tags of the operator.

        Returns
        -------
        IndividualOp
            Copy of the operator with a customized schema
        """
        op = copy.deepcopy(self)
        methods = ["fit", "transform", "predict", "predict_proba", "decision_function"]

        if schemas is not None:
            schemas.schema["$schema"] = "http://json-schema.org/draft-04/schema#"
            lale.type_checking.validate_is_schema(schemas.schema)
            op._schemas = schemas.schema
        else:
            if relevantToOptimizer is not None:
                assert isinstance(relevantToOptimizer, list)
                op._schemas["properties"]["hyperparams"]["allOf"][0][
                    "relevantToOptimizer"
                ] = relevantToOptimizer
            if constraint is not None:
                op._schemas["properties"]["hyperparams"]["allOf"].append(
                    constraint.schema
                )
            if tags is not None:
                assert isinstance(tags, dict)
                op._schemas["tags"] = tags

            for arg in kwargs:
                value = kwargs[arg]
                if arg in [p + n for p in ["input_", "output_"] for n in methods]:
                    # multiple input types (e.g., fit, predict)
                    assert value is not None
                    lale.type_checking.validate_method(op, arg)
                    lale.type_checking.validate_is_schema(value.schema)
                    op._schemas["properties"][arg] = value.schema
                elif value is None:
                    scm = op._schemas["properties"]["hyperparams"]["allOf"][0]
                    scm["required"] = [k for k in scm["required"] if k != arg]
                    scm["relevantToOptimizer"] = [
                        k for k in scm["relevantToOptimizer"] if k != arg
                    ]
                    scm["properties"] = {
                        k: scm["properties"][k] for k in scm["properties"] if k != arg
                    }
                else:
                    op._schemas["properties"]["hyperparams"]["allOf"][0]["properties"][
                        arg
                    ] = value.schema
        # since the schema has changed, we need to invalidate any
        # cached enum attributes
        self._invalidate_enum_attributes()
        return op

    def _validate_hyperparams(self, hp_explicit, hp_all, hp_schema):
        try:
            lale.type_checking.validate_schema(hp_all, hp_schema)
        except jsonschema.ValidationError as e_orig:
            e = e_orig if e_orig.parent is None else e_orig.parent
            lale.type_checking.validate_is_schema(e.schema)
            schema = lale.pretty_print.to_string(e.schema)
            if [*e.schema_path][:3] == ["allOf", 0, "properties"]:
                arg = e.schema_path[3]
                reason = f"invalid value {arg}={e.instance}"
                schema_path = f"argument {arg}"
            elif [*e.schema_path][:3] == ["allOf", 0, "additionalProperties"]:
                pref, suff = "Additional properties are not allowed (", ")"
                assert e.message.startswith(pref) and e.message.endswith(suff)
                reason = "argument " + e.message[len(pref) : -len(suff)]
                schema_path = "arguments and their defaults"
                schema = self.hyperparam_defaults()
            elif e.schema_path[0] == "allOf" and int(e.schema_path[1]) != 0:
                assert e.schema_path[2] == "anyOf"
                descr = e.schema["description"]
                if descr.endswith("."):
                    descr = descr[:-1]
                reason = f"constraint {descr[0].lower()}{descr[1:]}"
                schema_path = f"constraint {e.schema_path[1]}"
            else:
                reason = e.message
                schema_path = e.schema_path
            msg = (
                f"Invalid configuration for {self.name()}("
                + f"{lale.pretty_print.hyperparams_to_string(hp_explicit)}) "
                + f"due to {reason}.\n"
                + f"Schema of {schema_path}: {schema}\n"
                + f"Value: {e.instance}"
            )
            raise jsonschema.ValidationError(msg)

    def _validate_hyperparam_data_constraints(self, X, y=None):
        hp_schema = self.hyperparam_schema()
        if not hasattr(self, "__has_data_constraints"):
            has_dc = lale.type_checking.has_data_constraints(hp_schema)
            self.__has_data_constraints = has_dc
        if self.__has_data_constraints:
            hp_explicit = self._hyperparams
            hp_all = self._get_params_all()
            data_schema = lale.helpers.fold_schema(X, y)
            hp_schema_2 = lale.type_checking.replace_data_constraints(
                hp_schema, data_schema
            )
            self._validate_hyperparams(hp_explicit, hp_all, hp_schema_2)

    def validate_schema(self, X, y=None):
        if hasattr(self._impl, "fit"):
            X = self._validate_input_schema("X", X, "fit")
        method = "transform" if self.is_transformer() else "predict"
        self._validate_input_schema("X", X, method)
        if self.is_supervised(default_if_missing=False):
            if y is None:
                raise ValueError(f"{self.name()}.fit() y cannot be None")
            else:
                if hasattr(self._impl, "fit"):
                    y = self._validate_input_schema("y", y, "fit")
                self._validate_input_schema("y", y, method)

    def _validate_input_schema(self, arg_name, arg, method):
        if not lale.helpers.is_empty_dict(arg):
            if method == "fit" or method == "partial_fit":
                schema = self.input_schema_fit()
            elif method == "transform":
                schema = self.input_schema_transform()
            elif method == "predict":
                schema = self.input_schema_predict()
            elif method == "predict_proba":
                schema = self.input_schema_predict_proba()
            elif method == "decision_function":
                schema = self.input_schema_decision_function()
            if "properties" in schema and arg_name in schema["properties"]:
                arg = lale.datasets.data_schemas.add_schema(arg)
                try:
                    sup = schema["properties"][arg_name]
                    lale.type_checking.validate_schema_or_subschema(arg, sup)
                except lale.type_checking.SubschemaError as e:
                    sub = lale.pretty_print.json_to_string(e.sub)
                    sup = lale.pretty_print.json_to_string(e.sup)
                    raise ValueError(
                        f"{self.name()}.{method}() invalid {arg_name}, the schema of the actual data is not a subschema of the expected schema of the argument.\nactual_schema = {sub}\nexpected_schema = {sup}"
                    )
                except Exception as e:
                    exception_type = f"{type(e).__module__}.{type(e).__name__}"
                    raise ValueError(
                        f"{self.name()}.{method}() invalid {arg_name}: {exception_type}: {e}"
                    ) from None
        return arg

    def _validate_output_schema(self, result, method):
        if method == "transform":
            schema = self.output_schema_transform()
        elif method == "predict":
            schema = self.output_schema_predict()
        elif method == "predict_proba":
            schema = self.output_schema_predict_proba()
        elif method == "decision_function":
            schema = self.output_schema_decision_function()
        result = lale.datasets.data_schemas.add_schema(result)
        try:
            lale.type_checking.validate_schema_or_subschema(result, schema)
        except Exception as e:
            print(f"{self.name()}.{method}() invalid result: {e}")
            raise ValueError(f"{self.name()}.{method}() invalid result: {e}") from e
        return result

    def transform_schema(self, s_X) -> JSON_TYPE:
        if self.is_transformer():
            return self.output_schema_transform()
        elif hasattr(self._impl, "predict_proba"):
            return self.output_schema_predict_proba()
        elif hasattr(self._impl, "decision_function"):
            return self.output_schema_decision_function()
        else:
            return self.output_schema_predict()

    def is_supervised(self, default_if_missing=True) -> bool:
        if hasattr(self._impl, "fit"):
            schema_fit = self.input_schema_fit()
            return lale.type_checking.is_subschema(schema_fit, _is_supervised_schema)
        return default_if_missing

    def is_classifier(self) -> bool:
        return self.has_tag("classifier")

    def is_transformer(self) -> bool:
        """ Checks if the operator is a transformer
        """
        return hasattr(self._impl, "transform")


_is_supervised_schema = {"type": "object", "required": ["y"]}


class PlannedIndividualOp(IndividualOp, PlannedOperator):
    """
    This is a concrete class that returns a trainable individual
    operator through its __call__ method. A configure method can use
    an optimizer and return the best hyperparameter combination.
    """

    _hyperparams: Optional[Dict[str, Any]]

    def __init__(self, _name: str, _impl, _schemas) -> None:
        super(PlannedIndividualOp, self).__init__(_name, _impl, _schemas)
        self._hyperparams = None

    def _should_configure_trained(self, impl):
        # TODO: may also want to do this for other higher-order operators
        if self.class_name() != _LALE_SKL_PIPELINE:
            return False
        return isinstance(impl._pipeline, TrainedPipeline)

    def _configure(self, *args, **kwargs) -> "TrainableIndividualOp":
        class_ = self._impl_class()
        hyperparams = {}
        for arg in args:
            k, v = self._enum_to_strings(arg)
            hyperparams[k] = v
        for k, v in _fixup_hyperparams_dict(kwargs).items():

            if k in hyperparams:
                raise ValueError("Duplicate argument {}.".format(k))
            v = lale.helpers.val_wrapper.unwrap(v)
            if isinstance(v, enumeration.Enum):
                k2, v2 = self._enum_to_strings(v)
                if k != k2:
                    raise ValueError(
                        "Invalid keyword {} for argument {}.".format(k2, v2)
                    )
            else:
                v2 = v
            hyperparams[k] = v2
        # using params_all instead of hyperparams to ensure the construction is consistent with schema
        trainable_to_get_params = TrainableIndividualOp(
            _name=self.name(), _impl=None, _schemas=self._schemas
        )
        trainable_to_get_params._hyperparams = hyperparams
        params_all = trainable_to_get_params._get_params_all()
        self._validate_hyperparams(hyperparams, params_all, self.hyperparam_schema())
        if len(params_all) == 0:
            impl = class_()
        else:
            impl = class_(**params_all)

        if self._should_configure_trained(impl):
            result: TrainableIndividualOp = TrainedIndividualOp(
                _name=self.name(), _impl=impl, _schemas=self._schemas
            )
        else:
            result = TrainableIndividualOp(
                _name=self.name(), _impl=impl, _schemas=self._schemas
            )
        result._hyperparams = hyperparams
        return result

    def __call__(self, *args, **kwargs) -> "TrainableIndividualOp":
        return self._configure(*args, **kwargs)

    def _hyperparam_schema_with_hyperparams(self, data_schema={}):
        def fix_hyperparams(schema):
            hyperparams = None
            try:
                hyperparams = self._hyperparams
            except AttributeError:
                pass
            if not hyperparams:
                return schema
            props = {k: {"enum": [v]} for k, v in hyperparams.items()}
            obj = {"type": "object", "properties": props}
            obj["relevantToOptimizer"] = list(hyperparams.keys())
            obj["required"] = list(hyperparams.keys())
            top = {"allOf": [schema, obj]}
            return top

        s_1 = self.hyperparam_schema()
        s_2 = fix_hyperparams(s_1)
        s_3 = lale.type_checking.replace_data_constraints(s_2, data_schema)
        return s_3

    # This should *only* ever be called by the sklearn_compat wrapper
    def set_params(self, **impl_params):
        return self._configure(**impl_params)

    def freeze_trainable(self) -> "TrainableIndividualOp":
        return self._configure().freeze_trainable()

    def free_hyperparams(self):
        hyperparam_schema = self.hyperparam_schema()
        if (
            "allOf" in hyperparam_schema
            and "relevantToOptimizer" in hyperparam_schema["allOf"][0]
        ):
            to_bind = hyperparam_schema["allOf"][0]["relevantToOptimizer"]
        else:
            to_bind = []
        if self._hyperparams:
            bound = self._hyperparams.keys()
        else:
            bound = []
        return set(to_bind) - set(bound)

    def is_frozen_trainable(self) -> bool:
        free = self.free_hyperparams()
        return len(free) == 0


def _mutation_warning(method_name: str) -> str:
    msg = str(
        "The `{}` method is deprecated on a trainable "
        "operator, because the learned coefficients could be "
        "accidentally overwritten by retraining. Call `{}` "
        "on the trained operator returned by `fit` instead."
    )
    return msg.format(method_name, method_name)


class TrainableIndividualOp(PlannedIndividualOp, TrainableOperator):
    def __init__(self, _name, _impl, _schemas):
        super(TrainableIndividualOp, self).__init__(_name, _impl, _schemas)

    def _clone_impl(self):
        impl_instance = self._impl_instance()
        if hasattr(impl_instance, "get_params"):
            result = sklearn.base.clone(impl_instance)
        else:
            try:
                result = copy.deepcopy(impl_instance)
            except Exception:
                impl_class = self._impl_class()
                params_all = self._get_params_all()
                result = impl_class(**params_all)
        return result

    def _trained_hyperparams(self, trained_impl):
        # TODO: may also want to do this for other higher-order operators
        if self.class_name() != _LALE_SKL_PIPELINE:
            return self._hyperparams
        names_list = [name for name, op in self._hyperparams["steps"]]
        steps_list = trained_impl._pipeline.steps()
        trained_steps = list(zip(names_list, steps_list))
        result = {**self._hyperparams, "steps": trained_steps}
        return result

    def fit(self, X, y=None, **fit_params) -> "TrainedIndividualOp":
        logger.info("%s enter fit %s", time.asctime(), self.name())
        X = self._validate_input_schema("X", X, "fit")
        y = self._validate_input_schema("y", y, "fit")
        self._validate_hyperparam_data_constraints(X, y)
        filtered_fit_params = _fixup_hyperparams_dict(fit_params)
        trainable_impl = self._clone_impl()
        if filtered_fit_params is None:
            trained_impl = trainable_impl.fit(X, y)
        else:
            trained_impl = trainable_impl.fit(X, y, **filtered_fit_params)
        # if the trainable fit method returns None, assume that
        # the trainableshould be used as the trained impl as well
        if trained_impl is None:
            trained_impl = trainable_impl
        result = TrainedIndividualOp(self.name(), trained_impl, self._schemas)
        result._hyperparams = self._trained_hyperparams(trained_impl)
        self._trained = result
        logger.info("%s exit  fit %s", time.asctime(), self.name())
        return result

    def partial_fit(self, X, y=None, **fit_params) -> TrainedOperator:
        if not hasattr(self._impl, "partial_fit"):
            raise AttributeError(f"{self.name()} has no partial_fit implemented.")
        X = self._validate_input_schema("X", X, "partial_fit")
        y = self._validate_input_schema("y", y, "partial_fit")
        self._validate_hyperparam_data_constraints(X, y)
        filtered_fit_params = _fixup_hyperparams_dict(fit_params)
        trainable_impl = self._clone_impl()
        if filtered_fit_params is None:
            trained_impl = trainable_impl.partial_fit(X, y)
        else:
            trained_impl = trainable_impl.partial_fit(X, y, **filtered_fit_params)
        if trained_impl is None:
            trained_impl = trainable_impl
        result = TrainedIndividualOp(self.name(), trained_impl, self._schemas)
        result._hyperparams = self._hyperparams
        self._trained = result
        return result

    def freeze_trained(self) -> "TrainedIndividualOp":
        """
        .. deprecated:: 0.0.0
           The `freeze_trained` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `freeze_trained`
           on the trained operator returned by `fit` instead.
        """
        warnings.warn(_mutation_warning("freeze_trained"), DeprecationWarning)
        try:
            return self._trained.freeze_trained()
        except AttributeError:
            raise ValueError("Must call `fit` before `freeze_trained`.")

    def is_frozen_trained(self) -> bool:
        return False

    @if_delegate_has_method(delegate="_impl")
    def get_pipeline(
        self, pipeline_name=None, astype="lale"
    ) -> Optional[TrainableOperator]:
        """
        .. deprecated:: 0.0.0
           The `get_pipeline` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `get_pipeline`
           on the trained operator returned by `fit` instead.
        """
        warnings.warn(_mutation_warning("get_pipeline"), DeprecationWarning)
        try:
            return self._trained.get_pipeline(pipeline_name, astype)
        except AttributeError:
            raise ValueError("Must call `fit` before `get_pipeline`.")

    @if_delegate_has_method(delegate="_impl")
    def summary(self) -> pd.DataFrame:
        """
        .. deprecated:: 0.0.0
           The `summary` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `summary`
           on the trained operator returned by `fit` instead.
        """
        warnings.warn(_mutation_warning("summary"), DeprecationWarning)
        try:
            return self._trained.summary()
        except AttributeError:
            raise ValueError("Must call `fit` before `summary`.")

    @if_delegate_has_method(delegate="_impl")
    def transform(self, X, y=None):
        """
        .. deprecated:: 0.0.0
           The `transform` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `transform`
           on the trained operator returned by `fit` instead.
        """
        warnings.warn(_mutation_warning("transform"), DeprecationWarning)
        try:
            return self._trained.transform(X, y)
        except AttributeError:
            raise ValueError("Must call `fit` before `transform`.")

    @if_delegate_has_method(delegate="_impl")
    def predict(self, X):
        """
        .. deprecated:: 0.0.0
           The `predict` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `predict`
           on the trained operator returned by `fit` instead.
        """
        warnings.warn(_mutation_warning("predict"), DeprecationWarning)
        try:
            return self._trained.predict(X)
        except AttributeError:
            raise ValueError("Must call `fit` before `predict`.")

    @if_delegate_has_method(delegate="_impl")
    def predict_proba(self, X):
        """
        .. deprecated:: 0.0.0
           The `predict_proba` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `predict_proba`
           on the trained operator returned by `fit` instead.
        """
        warnings.warn(_mutation_warning("predict_proba"), DeprecationWarning)
        try:
            return self._trained.predict_proba(X)
        except AttributeError:
            raise ValueError("Must call `fit` before `predict_proba`.")

    @if_delegate_has_method(delegate="_impl")
    def decision_function(self, X):
        """
        .. deprecated:: 0.0.0
           The `decision_function` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `decision_function`
           on the trained operator returned by `fit` instead.
        """
        warnings.warn(_mutation_warning("decision_function"), DeprecationWarning)
        try:
            return self._trained.decision_function(X)
        except AttributeError:
            raise ValueError("Must call `fit` before `decision_function`.")

    def free_hyperparams(self):
        hyperparam_schema = self.hyperparam_schema()
        if (
            "allOf" in hyperparam_schema
            and "relevantToOptimizer" in hyperparam_schema["allOf"][0]
        ):
            to_bind = hyperparam_schema["allOf"][0]["relevantToOptimizer"]
        else:
            to_bind = []
        if self._hyperparams:
            bound = self._hyperparams.keys()
        else:
            bound = []
        return set(to_bind) - set(bound)

    def is_frozen_trainable(self) -> bool:
        free = self.free_hyperparams()
        return len(free) == 0

    def _freeze_trainable_bindings(self):
        old_bindings = self._hyperparams if self._hyperparams else {}
        free = self.free_hyperparams()
        defaults = self.hyperparam_defaults()
        new_bindings = {name: defaults[name] for name in free}
        bindings = {**old_bindings, **new_bindings}
        return bindings

    def freeze_trainable(self) -> "TrainableIndividualOp":
        bindings = self._freeze_trainable_bindings()
        result = self._configure(**bindings)
        assert result.is_frozen_trainable(), str(result.free_hyperparams())
        return result

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Get parameters for this operator.

        This method follows scikit-learn's convention that all operators
        have a constructor which takes a list of keyword arguments.
        This is not required for operator impls which do not desire
        scikit-compatibility.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this operator state wrapper and
            its impl object
        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        out["_name"] = self._name
        out["_schemas"] = self._schemas
        impl = self._impl_instance()
        out["_impl"] = impl
        if deep and hasattr(impl, "get_params"):
            deep_items = impl.get_params().items()
            out.update((self._name + "__" + k, val) for k, val in deep_items)
        return out

    def hyperparams(self):
        if self._hyperparams is None:
            return None
        actuals = self._hyperparams
        defaults = self.hyperparam_defaults()
        actuals_minus_defaults = {
            k: actuals[k]
            for k in actuals
            if k not in defaults or actuals[k] != defaults[k]
        }
        if not hasattr(self, "_hyperparam_positionals"):
            sig = inspect.signature(self._impl_class().__init__)
            positionals = {
                name: defaults[name]
                for name, param in sig.parameters.items()
                if name != "self"
                and param.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
                and param.default == inspect.Parameter.empty
            }
            self._hyperparam_positionals = positionals
        result = {**self._hyperparam_positionals, **actuals_minus_defaults}
        return result

    def _get_params_all(self):
        output = {}
        if self._hyperparams is not None:
            output.update(self._hyperparams)
        defaults = self.hyperparam_defaults()
        for k in defaults.keys():
            if k not in output:
                output[k] = defaults[k]
        return output

    # This should *only* ever be called by the sklearn_compat wrapper
    def set_params(self, **impl_params):
        # TODO: This mutates the operator, should we mark it deprecated?
        filtered_impl_params = _fixup_hyperparams_dict(impl_params)
        self._impl = lale.helpers.create_individual_op_using_reflection(
            self.class_name(), self._name, filtered_impl_params
        )
        self._hyperparams = filtered_impl_params
        return self

    def transform_schema(self, s_X):
        if hasattr(self._impl, "transform_schema"):
            try:
                return self._impl_instance().transform_schema(s_X)
            except BaseException as e:
                raise ValueError(
                    f"unexpected error in {self.name()}.transform_schema({lale.pretty_print.to_string(s_X)}"
                ) from e
        else:
            return super(TrainableIndividualOp, self).transform_schema(s_X)

    def input_schema_fit(self) -> JSON_TYPE:
        if hasattr(self._impl, "input_schema_fit"):
            return self._impl_instance().input_schema_fit()
        else:
            return super(TrainableIndividualOp, self).input_schema_fit()

    def _lale_clone(self, cloner: Callable[[Any], Any]):
        impl = self._impl
        if impl is not None and not inspect.isclass(impl):
            impl = cloner(impl)
        cp = make_operator(impl, schemas=self._schemas, name=self._name)
        if isinstance(cp, PlannedIndividualOp):
            cp._hyperparams = self._hyperparams
        return cp


class TrainedIndividualOp(TrainableIndividualOp, TrainedOperator):
    _frozen_trained: bool

    def __init__(self, _name, _impl, _schemas):
        super(TrainedIndividualOp, self).__init__(_name, _impl, _schemas)
        self._frozen_trained = not hasattr(self._impl, "fit")

    def __call__(self, *args, **kwargs) -> "TrainedIndividualOp":
        filtered_kwargs_params = _fixup_hyperparams_dict(kwargs)

        trainable = self._configure(*args, **filtered_kwargs_params)
        instance = TrainedIndividualOp(
            trainable._name, trainable._impl, trainable._schemas
        )
        instance._hyperparams = trainable._hyperparams
        return instance

    def fit(self, X, y=None, **fit_params) -> "TrainedIndividualOp":
        if hasattr(self._impl, "fit") and not self.is_frozen_trained():
            filtered_fit_params = _fixup_hyperparams_dict(fit_params)
            return super(TrainedIndividualOp, self).fit(X, y, **filtered_fit_params)
        else:
            return self

    @if_delegate_has_method(delegate="_impl")
    def transform(self, X, y=None):
        """Transform the data.

        Parameters
        ----------
        X :
            Features; see input_transform schema of the operator.

        Returns
        -------
        result :
            Transformed features; see output_transform schema of the operator.
        """
        logger.info("%s enter transform %s", time.asctime(), self.name())
        X = self._validate_input_schema("X", X, "transform")
        if "y" in [
            required_property.lower()
            for required_property in self.input_schema_transform().get("required", [])
        ]:
            y = self._validate_input_schema("y", y, "transform")
            raw_result = self._impl_instance().transform(X, y)
        else:
            raw_result = self._impl_instance().transform(X)
        result = self._validate_output_schema(raw_result, "transform")
        logger.info("%s exit  transform %s", time.asctime(), self.name())
        return result

    def _predict(self, X):
        X = self._validate_input_schema("X", X, "predict")
        raw_result = self._impl_instance().predict(X)
        result = self._validate_output_schema(raw_result, "predict")
        return result

    @if_delegate_has_method(delegate="_impl")
    def predict(self, X):
        """Make predictions.

        Parameters
        ----------
        X :
            Features; see input_predict schema of the operator.

        Returns
        -------
        result :
            Predictions; see output_predict schema of the operator.
        """
        logger.info("%s enter predict %s", time.asctime(), self.name())
        result = self._predict(X)
        logger.info("%s exit  predict %s", time.asctime(), self.name())
        if isinstance(result, lale.datasets.data_schemas.NDArrayWithSchema):
            return lale.datasets.data_schemas.strip_schema(
                result
            )  # otherwise scorers return zero-dim array
        return result

    @if_delegate_has_method(delegate="_impl")
    def predict_proba(self, X):
        """Probability estimates for all classes.

        Parameters
        ----------
        X :
            Features; see input_predict_proba schema of the operator.

        Returns
        -------
        result :
            Probabilities; see output_predict_proba schema of the operator.
        """
        logger.info("%s enter predict_proba %s", time.asctime(), self.name())
        X = self._validate_input_schema("X", X, "predict_proba")
        raw_result = self._impl_instance().predict_proba(X)
        result = self._validate_output_schema(raw_result, "predict_proba")
        logger.info("%s exit  predict_proba %s", time.asctime(), self.name())
        return result

    @if_delegate_has_method(delegate="_impl")
    def decision_function(self, X):
        """Confidence scores for all classes.

        Parameters
        ----------
        X :
            Features; see input_decision_function schema of the operator.

        Returns
        -------
        result :
            Confidences; see output_decision_function schema of the operator.
        """
        logger.info("%s enter decision_function %s", time.asctime(), self.name())
        X = self._validate_input_schema("X", X, "decision_function")
        raw_result = self._impl_instance().decision_function(X)
        result = self._validate_output_schema(raw_result, "decision_function")
        logger.info("%s exit  decision_function %s", time.asctime(), self.name())
        return result

    def freeze_trainable(self) -> "TrainedIndividualOp":
        result = copy.deepcopy(self)
        result._hyperparams = self._freeze_trainable_bindings()
        assert result.is_frozen_trainable(), str(result.free_hyperparams())
        assert isinstance(result, TrainedIndividualOp)
        return result

    def is_frozen_trained(self) -> bool:
        return self._frozen_trained

    def freeze_trained(self) -> "TrainedIndividualOp":
        if self.is_frozen_trained():
            return self
        result = copy.deepcopy(self)
        result._frozen_trained = True
        assert result.is_frozen_trained()
        return result

    @if_delegate_has_method(delegate="_impl")
    def get_pipeline(
        self, pipeline_name=None, astype="lale"
    ) -> Optional[TrainableOperator]:
        result = self._impl_instance().get_pipeline(pipeline_name, astype)
        return result

    @if_delegate_has_method(delegate="_impl")
    def summary(self) -> pd.DataFrame:
        return self._impl_instance().summary()

    def _lale_clone(self, cloner: Callable[[Any], Any]):
        """ This is really used for sklearn clone compatibility.
            Which mandates that clone returns something that has not been fit.
            So we enforce that here as well.
        """
        impl = self._impl
        if impl is not None and not inspect.isclass(impl):
            impl = cloner(impl)
        cp = make_operator(impl, schemas=self._schemas, name=self._name)
        if isinstance(cp, PlannedIndividualOp):
            cp._hyperparams = self._hyperparams
        return cp


_all_available_operators: List[PlannedOperator] = []


def wrap_operator(impl) -> Operator:
    if isinstance(impl, Operator):
        return impl
    else:
        return make_operator(impl)


def make_operator(impl, schemas=None, name=None) -> PlannedIndividualOp:
    if name is None:
        name = lale.helpers.assignee_name()
        if name is None:
            if inspect.isclass(impl) and impl.__name__.endswith("Impl"):
                name = impl.__name__[: -len("Impl")]
    if inspect.isclass(impl):
        if hasattr(impl, "fit"):
            operatorObj = PlannedIndividualOp(name, impl, schemas)
        else:
            operatorObj = TrainedIndividualOp(name, impl, schemas)
    else:
        if hasattr(impl, "fit"):
            operatorObj = TrainableIndividualOp(name, impl, schemas)
        else:
            operatorObj = TrainedIndividualOp(name, impl, schemas)
        if hasattr(impl, "get_params"):
            operatorObj._hyperparams = {**impl.get_params()}
    operatorObj._check_schemas()
    _all_available_operators.append(operatorObj)
    return operatorObj


def get_available_operators(
    tag: str, more_tags: AbstractSet[str] = None
) -> List[PlannedOperator]:
    singleton = set([tag])
    tags = singleton if (more_tags is None) else singleton.union(more_tags)

    def filter(op):
        tags_dict = op.get_tags()
        if tags_dict is None:
            return False
        tags_set = {tag for prefix in tags_dict for tag in tags_dict[prefix]}
        return tags.issubset(tags_set)

    return [op for op in _all_available_operators if filter(op)]


def get_available_estimators(tags: AbstractSet[str] = None) -> List[PlannedOperator]:
    return get_available_operators("estimator", tags)


def get_available_transformers(tags: AbstractSet[str] = None) -> List[PlannedOperator]:
    return get_available_operators("transformer", tags)


OpType = TypeVar("OpType", bound=Operator)


class BasePipeline(Operator, Generic[OpType]):
    """
    This is a concrete class that can instantiate a new pipeline operator and provide access to its meta data.
    """

    _steps: List[OpType]
    _preds: Dict[OpType, List[OpType]]
    _name: str
    _estimator_type: Optional[str]

    def _lale_clone(self, cloner: Callable[[Any], Any]):
        steps = self._steps
        new_steps: List[OpType] = [s._lale_clone(cloner) for s in steps]
        if self._preds is None:
            return self.__class__(new_steps, None, True)

        step_map: Dict[OpType, OpType] = {
            steps[i]: new_steps[i] for i in range(len(steps))
        }
        new_edges = ((step_map[s], step_map[d]) for s, d in self.edges())
        return self.__class__(new_steps, new_edges, True)

    def __init__(
        self,
        steps: List[OpType],
        edges: Optional[Iterable[Tuple[OpType, OpType]]],
        ordered: bool = False,
    ) -> None:
        self._name = "pipeline_" + str(id(self))
        self._preds = {}
        for step in steps:
            assert isinstance(step, Operator)
        if edges is None:
            # Which means there is a linear pipeline #TODO:Test extensively with clone and get_params
            # This constructor is mostly called due to cloning. Make sure the objects are kept the same.
            self.__constructor_for_cloning(steps)
        else:
            self._steps = []

            for step in steps:
                if step in self._steps:
                    raise ValueError(
                        "Same instance of {} already exists in the pipeline. "
                        "This is not allowed.".format(step.name())
                    )
                if isinstance(step, BasePipeline):
                    # Flatten out the steps and edges
                    self._steps.extend(step.steps())
                    # from step's edges, find out all the source and sink nodes
                    source_nodes = [
                        dst
                        for dst in step.steps()
                        if (step._preds[dst] is None or step._preds[dst] == [])
                    ]
                    sink_nodes = step._find_sink_nodes()
                    # Now replace the edges to and from the inner pipeline to to and from source and sink nodes respectively
                    new_edges = step.edges()
                    # list comprehension at the cost of iterating edges thrice
                    new_edges.extend(
                        [
                            (node, edge[1])
                            for edge in edges
                            if edge[0] == step
                            for node in sink_nodes
                        ]
                    )
                    new_edges.extend(
                        [
                            (edge[0], node)
                            for edge in edges
                            if edge[1] == step
                            for node in source_nodes
                        ]
                    )
                    new_edges.extend(
                        [
                            edge
                            for edge in edges
                            if (edge[1] != step and edge[0] != step)
                        ]
                    )
                    edges = new_edges
                else:
                    self._steps.append(step)
            self._preds = {step: [] for step in self._steps}
            for (src, dst) in edges:
                self._preds[dst].append(src)
            if not ordered:
                self.__sort_topologically()
            assert self.__is_in_topological_order()
        if self.is_classifier():
            self._estimator_type = (
                "classifier"  # satisfy sklearn.base.is_classifier(op)
            )

    def __constructor_for_cloning(self, steps: List[OpType]):
        edges: List[Tuple[OpType, OpType]] = []
        prev_op: Optional[OpType] = None
        # This is due to scikit base's clone method that needs the same list object
        self._steps = steps
        prev_leaves: List[OpType]
        curr_roots: List[OpType]

        for curr_op in self._steps:
            if isinstance(prev_op, BasePipeline):
                prev_leaves = prev_op._find_sink_nodes()
            else:
                prev_leaves = [] if prev_op is None else [prev_op]
            if isinstance(curr_op, BasePipeline):
                curr_roots = curr_op._find_source_nodes()
                self._steps.extend(curr_op.steps())
                edges.extend(curr_op.edges())
            else:
                curr_roots = [curr_op]
            edges.extend([(src, tgt) for src in prev_leaves for tgt in curr_roots])
            prev_op = curr_op

        seen_steps: List[OpType] = []
        for step in self._steps:
            if step in seen_steps:
                raise ValueError(
                    "Same instance of {} already exists in the pipeline. "
                    "This is not allowed.".format(step.name())
                )
            seen_steps.append(step)
        self._preds = {step: [] for step in self._steps}
        for (src, dst) in edges:
            self._preds[dst].append(src)
        # Since this case is only allowed for linear pipelines, it is always
        # expected to be in topological order
        assert self.__is_in_topological_order()

    def edges(self) -> List[Tuple[OpType, OpType]]:
        return [(src, dst) for dst in self._steps for src in self._preds[dst]]

    def __is_in_topological_order(self) -> bool:
        seen: Dict[OpType, bool] = {}
        for operator in self._steps:
            for pred in self._preds[operator]:
                if pred not in seen:
                    return False
            seen[operator] = True
        return True

    def steps(self) -> List[OpType]:
        return self._steps

    def _subst_steps(self, m: Dict[OpType, OpType]) -> None:
        if dict:
            # for i, s in enumerate(self._steps):
            #     self._steps[i] = m.get(s,s)
            self._steps = [m.get(s, s) for s in self._steps]
            self._preds = {
                m.get(k, k): [m.get(s, s) for s in v] for k, v in self._preds.items()
            }

    def __sort_topologically(self) -> None:
        class state(enumeration.Enum):
            TODO = (enumeration.auto(),)
            DOING = (enumeration.auto(),)
            DONE = enumeration.auto()

        states: Dict[OpType, state] = {op: state.TODO for op in self._steps}
        result: List[OpType] = []

        def dfs(operator: OpType) -> None:
            if states[operator] is state.DONE:
                return
            if states[operator] is state.DOING:
                raise ValueError("Cycle detected.")
            states[operator] = state.DOING
            for pred in self._preds[operator]:
                dfs(pred)
            states[operator] = state.DONE
            result.append(operator)

        for operator in self._steps:
            if states[operator] is state.TODO:
                dfs(operator)
        self._steps = result

    def _has_same_impl(self, other: Operator) -> bool:
        """Checks if the type of the operator imnplementations are compatible
        """
        if not isinstance(other, BasePipeline):
            return False
        my_steps = self.steps()
        other_steps = other.steps()
        if len(my_steps) != len(other_steps):
            return False

        for (m, o) in zip(my_steps, other_steps):
            if not m._has_same_impl(o):
                return False
        return True

    def _find_sink_nodes(self) -> List[OpType]:
        is_sink = {s: True for s in self.steps()}
        for src, _ in self.edges():
            is_sink[src] = False
        result = [s for s in self.steps() if is_sink[s]]
        return result

    def _find_source_nodes(self) -> List[OpType]:
        is_source = {s: True for s in self.steps()}
        for _, dst in self.edges():
            is_source[dst] = False
        result = [s for s in self.steps() if is_source[s]]
        return result

    def _validate_or_transform_schema(self, X, y=None, validate=True):
        def combine_schemas(schemas):
            n_datasets = len(schemas)
            if n_datasets == 1:
                result = schemas[0]
            else:
                result = {
                    "type": "array",
                    "minItems": n_datasets,
                    "maxItems": n_datasets,
                    "items": [lale.datasets.data_schemas.to_schema(i) for i in schemas],
                }
            return result

        outputs = {}
        for operator in self._steps:
            preds = self._preds[operator]
            if len(preds) == 0:
                inputs = X
            else:
                inputs = combine_schemas([outputs[pred] for pred in preds])
            if validate:
                operator.validate_schema(X=inputs, y=y)
            output = operator.transform_schema(inputs)
            outputs[operator] = output
        if not validate:
            sinks = self._find_sink_nodes()
            pipeline_outputs = [outputs[sink] for sink in sinks]
            return combine_schemas(pipeline_outputs)

    def validate_schema(self, X, y=None):
        self._validate_or_transform_schema(X, y, validate=True)

    def transform_schema(self, s_X):
        return self._validate_or_transform_schema(s_X, validate=False)

    def input_schema_fit(self) -> JSON_TYPE:
        sources = self._find_source_nodes()
        pipeline_inputs = [source.input_schema_fit() for source in sources]
        result = lale.type_checking.join_schemas(*pipeline_inputs)
        return result

    def is_supervised(self) -> bool:
        s = self.steps()
        if len(s) == 0:
            return False
        return self.steps()[-1].is_supervised()

    def remove_last(self, inplace=False):
        sink_nodes = self._find_sink_nodes()
        if len(sink_nodes) > 1:
            raise ValueError(
                "This pipeline has more than 1 sink nodes, can not remove last step meaningfully."
            )
        elif not inplace:
            modified_pipeline = copy.deepcopy(self)
            old_clf = modified_pipeline._steps[-1]
            modified_pipeline._steps.remove(old_clf)
            del modified_pipeline._preds[old_clf]
            return modified_pipeline
        else:
            old_clf = self._steps[-1]
            self._steps.remove(old_clf)
            del self._preds[old_clf]
            return self

    def _get_last(self) -> Optional[OpType]:
        sink_nodes = self._find_sink_nodes()
        if len(sink_nodes) > 1:
            return None
        else:
            old_clf = self._steps[-1]
            return old_clf

    def export_to_sklearn_pipeline(self):
        from sklearn.pipeline import FeatureUnion, make_pipeline

        from lale.lib.lale.concat_features import ConcatFeaturesImpl
        from lale.lib.lale.no_op import NoOpImpl
        from lale.lib.lale.relational import RelationalImpl

        def convert_nested_objects(node):
            for element in dir(node):  # Looking at only 1 level for now.
                try:
                    value = getattr(node, element)
                    if isinstance(value, Operator) and hasattr(
                        value._impl_instance(), "_wrapped_model"
                    ):
                        # node is a higher order operator
                        setattr(node, element, value._impl_instance()._wrapped_model)

                    stripped = lale.datasets.data_schemas.strip_schema(value)
                    if value is stripped:
                        continue
                    setattr(node, element, stripped)
                except BaseException:
                    # This is an optional processing, so if there is any exception, continue.
                    # For example, some scikit-learn classes will fail at getattr because they have
                    # that property defined conditionally.
                    pass

        def create_pipeline_from_sink_node(sink_node):
            # Ensure that the pipeline is either linear or has a "union followed by concat" construct
            # Translate the "union followed by concat" constructs to "featureUnion"
            # Inspect the node and convert any data with schema objects to original data types
            if isinstance(sink_node, OperatorChoice):
                raise ValueError(
                    "A pipeline that has an OperatorChoice can not be converted to "
                    " a scikit-learn pipeline:{}".format(self.to_json())
                )
            if isinstance(sink_node._impl, RelationalImpl):
                return None
            convert_nested_objects(sink_node._impl)
            if sink_node._impl_class() == ConcatFeaturesImpl:
                list_of_transformers = []
                for pred in self._preds[sink_node]:
                    pred_transformer = create_pipeline_from_sink_node(pred)
                    list_of_transformers.append(
                        (
                            pred.name() + "_" + str(id(pred)),
                            make_pipeline(*pred_transformer)
                            if isinstance(pred_transformer, list)
                            else pred_transformer,
                        )
                    )
                return FeatureUnion(list_of_transformers)
            else:
                preds = self._preds[sink_node]
                if preds is not None and len(preds) > 1:
                    raise ValueError(
                        "A pipeline graph that has operators other than ConcatFeatures with "
                        "multiple incoming edges is not a valid scikit-learn pipeline:{}".format(
                            self.to_json()
                        )
                    )
                else:
                    if hasattr(sink_node._impl_instance(), "_wrapped_model"):
                        sklearn_op = sink_node._impl_instance()._wrapped_model
                        convert_nested_objects(
                            sklearn_op
                        )  # This case needs one more level of conversion
                    else:
                        sklearn_op = sink_node._impl_instance()
                    sklearn_op = copy.deepcopy(sklearn_op)
                    if preds is None or len(preds) == 0:
                        return sklearn_op
                    else:
                        output_pipeline_steps = []
                        previous_sklearn_op = create_pipeline_from_sink_node(preds[0])
                        if previous_sklearn_op is not None and not isinstance(
                            previous_sklearn_op, NoOpImpl
                        ):
                            if isinstance(previous_sklearn_op, list):
                                output_pipeline_steps = previous_sklearn_op
                            else:
                                output_pipeline_steps.append(previous_sklearn_op)
                        if not isinstance(
                            sklearn_op, NoOpImpl
                        ):  # Append the current op only if not NoOp
                            output_pipeline_steps.append(sklearn_op)
                        return output_pipeline_steps

        sklearn_steps_list = []
        # Finding the sink node so that we can do a backward traversal
        sink_nodes = self._find_sink_nodes()
        # For a trained pipeline that is scikit compatible, there should be only one sink node
        if len(sink_nodes) != 1:
            raise ValueError(
                "A pipeline graph that ends with more than one estimator is not a"
                " valid scikit-learn pipeline:{}".format(self.to_json())
            )
        else:
            sklearn_steps_list = create_pipeline_from_sink_node(sink_nodes[0])
            # not checking for isinstance(sklearn_steps_list, NoOpImpl) here as there is no valid sklearn pipeline with just one NoOp.
        try:
            sklearn_pipeline = (
                make_pipeline(*sklearn_steps_list)
                if isinstance(sklearn_steps_list, list)
                else make_pipeline(sklearn_steps_list)
            )
        except TypeError:
            raise TypeError(
                "Error creating a scikit-learn pipeline, most likely because the steps are not scikit compatible."
            )
        return sklearn_pipeline

    def is_classifier(self) -> bool:
        sink_nodes = self._find_sink_nodes()
        for op in sink_nodes:
            if not op.is_classifier():
                return False
        return True


PlannedOpType = TypeVar("PlannedOpType", bound=PlannedOperator)


class PlannedPipeline(BasePipeline[PlannedOpType], PlannedOperator):
    def __init__(
        self,
        steps: List[PlannedOpType],
        edges: Optional[Iterable[Tuple[PlannedOpType, PlannedOpType]]],
        ordered: bool = False,
    ) -> None:
        super(PlannedPipeline, self).__init__(steps, edges, ordered=ordered)

    def freeze_trainable(self) -> "PlannedPipeline":
        frozen_steps = []
        frozen_map = {}
        for liquid in self._steps:
            frozen = liquid.freeze_trainable()
            frozen_map[liquid] = frozen
            frozen_steps.append(frozen)
        frozen_edges = [(frozen_map[x], frozen_map[y]) for x, y in self.edges()]
        result = make_pipeline_graph(frozen_steps, frozen_edges, ordered=True)
        assert result.is_frozen_trainable()
        return result

    def is_frozen_trainable(self) -> bool:
        for step in self.steps():
            if not step.is_frozen_trainable():
                return False
        return True


TrainableOpType = TypeVar("TrainableOpType", bound=TrainableIndividualOp)


class TrainablePipeline(PlannedPipeline[TrainableOpType], TrainableOperator):
    def __init__(
        self,
        steps: List[TrainableOpType],
        edges: Optional[Iterable[Tuple[TrainableOpType, TrainableOpType]]],
        ordered: bool = False,
    ) -> None:
        super(TrainablePipeline, self).__init__(steps, edges, ordered=ordered)

    def fit(self, X, y=None, **fit_params) -> "TrainedPipeline":
        X = lale.datasets.data_schemas.add_schema(X)
        y = lale.datasets.data_schemas.add_schema(y)
        self.validate_schema(X, y)
        trained_steps: List[TrainedOperator] = []
        outputs: Dict[Operator, Any] = {}
        meta_outputs: Dict[Operator, Any] = {}
        edges: List[Tuple[TrainableOpType, TrainableOpType]] = self.edges()
        trained_map: Dict[TrainableOpType, TrainedOperator] = {}

        sink_nodes = self._find_sink_nodes()
        for operator in self._steps:
            preds = self._preds[operator]
            if len(preds) == 0:
                inputs = [X]
                meta_data_inputs: Dict[Operator, Any] = {}
            else:
                inputs = [outputs[pred] for pred in preds]
                # we create meta_data_inputs as a dictionary with metadata from all previous steps
                # Note that if multiple previous steps generate the same key, it will retain only one of those.

                meta_data_inputs = {
                    key: meta_outputs[pred][key]
                    for pred in preds
                    if meta_outputs[pred] is not None
                    for key in meta_outputs[pred]
                }
            trainable = operator
            if len(inputs) == 1:
                inputs = inputs[0]
            if hasattr(operator._impl, "set_meta_data"):
                operator._impl_instance().set_meta_data(meta_data_inputs)
            meta_output: Dict[Operator, Any] = {}
            trained: TrainedOperator
            if isinstance(
                inputs, tuple
            ):  # This is the case for transformers which return X and y, such as resamplers.
                inputs, y = inputs
            if trainable.is_supervised():
                trained = trainable.fit(X=inputs, y=y)
            else:
                trained = trainable.fit(X=inputs)
            trained_map[operator] = trained
            trained_steps.append(trained)
            if (
                trainable not in sink_nodes
            ):  # There is no need to transform/predict on the last node during fit
                if trained.is_transformer():
                    output = trained.transform(X=inputs, y=y)
                    if hasattr(trained._impl, "get_transform_meta_output"):
                        meta_output = (
                            trained._impl_instance().get_transform_meta_output()
                        )
                else:
                    if trainable in sink_nodes:
                        output = trained._predict(
                            X=inputs
                        )  # We don't support y for predict yet as there is no compelling case
                    else:
                        # This is ok because trainable pipelines steps
                        # must only be individual operators
                        if hasattr(trained._impl, "predict_proba"):  # type: ignore
                            output = trained.predict_proba(X=inputs)
                        elif hasattr(trained._impl, "decision_function"):  # type: ignore
                            output = trained.decision_function(X=inputs)
                        else:
                            output = trained._predict(X=inputs)
                    if hasattr(trained._impl, "get_predict_meta_output"):
                        meta_output = trained._impl_instance().get_predict_meta_output()
                outputs[operator] = output
                meta_output_so_far = {
                    key: meta_outputs[pred][key]
                    for pred in preds
                    if meta_outputs[pred] is not None
                    for key in meta_outputs[pred]
                }
                meta_output_so_far.update(
                    meta_output
                )  # So newest gets preference in case of collisions
                meta_outputs[operator] = meta_output_so_far

        trained_edges = [(trained_map[x], trained_map[y]) for (x, y) in edges]

        trained_steps2: Any = trained_steps
        result: TrainedPipeline = TrainedPipeline(
            trained_steps2, trained_edges, ordered=True
        )
        self._trained = result
        return result

    def transform(self, X, y=None):
        """
        .. deprecated:: 0.0.0
           The `transform` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `transform`
           on the trained operator returned by `fit` instead.
        """
        warnings.warn(_mutation_warning("transform"), DeprecationWarning)
        try:
            return self._trained.transform(X, y=None)
        except AttributeError:
            raise ValueError("Must call `fit` before `transform`.")

    def predict(self, X):
        """
        .. deprecated:: 0.0.0
           The `predict` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `predict`
           on the trained operator returned by `fit` instead.
        """
        warnings.warn(_mutation_warning("predict"), DeprecationWarning)
        try:
            return self._trained.predict(X)
        except AttributeError:
            raise ValueError("Must call `fit` before `predict`.")

    def predict_proba(self, X):
        """
        .. deprecated:: 0.0.0
           The `predict_proba` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `predict_proba`
           on the trained operator returned by `fit` instead.
        """
        warnings.warn(_mutation_warning("predict_proba"), DeprecationWarning)
        try:
            return self._trained.predict_proba(X)
        except AttributeError:
            raise ValueError("Must call `fit` before `predict_proba`.")

    def decision_function(self, X):
        """
        .. deprecated:: 0.0.0
           The `decision_function` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `decision_function`
           on the trained operator returned by `fit` instead.
        """
        warnings.warn(_mutation_warning("decision_function"), DeprecationWarning)
        try:
            return self._trained.decision_function(X)
        except AttributeError:
            raise ValueError("Must call `fit` before `decision_function`.")

    def freeze_trainable(self) -> "TrainablePipeline":
        frozen_steps = []
        frozen_map = {}
        for liquid in self._steps:
            frozen = liquid.freeze_trainable()
            frozen_map[liquid] = frozen
            frozen_steps.append(frozen)
        frozen_edges = [(frozen_map[x], frozen_map[y]) for x, y in self.edges()]
        result = cast(
            TrainablePipeline,
            make_pipeline_graph(frozen_steps, frozen_edges, ordered=True),
        )
        assert result.is_frozen_trainable()
        return result

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if not deep:
            out["steps"] = self._steps
            out["edges"] = None
            out["ordered"] = True
        else:
            pass  # TODO
        return out

    def fit_with_batches(self, X, y=None, serialize=True, num_epochs_batching=None):
        """[summary]

        Parameters
        ----------
        X :
            [description]
        y : [type], optional
            For a supervised pipeline, this is an array with the unique class labels
            in the entire dataset, by default None
        Returns
        -------
        [type]
            [description]
        """
        trained_steps: List[TrainedOperator] = []
        outputs: Dict[Operator, Any] = {}
        edges: List[Tuple[TrainableOpType, TrainableOpType]] = self.edges()
        trained_map: Dict[TrainableOpType, TrainedOperator] = {}

        if serialize:
            serialization_out_dir = os.path.join(
                os.path.dirname(__file__), "temp_serialized"
            )
            if not os.path.exists(serialization_out_dir):
                os.mkdir(serialization_out_dir)

        sink_nodes = self._find_sink_nodes()
        operator_idx = 0
        for operator in self._steps:
            preds = self._preds[operator]
            if len(preds) == 0:
                inputs = [X]
            else:
                inputs = [
                    outputs[pred][0]
                    if isinstance(outputs[pred], tuple)
                    else outputs[pred]
                    for pred in preds
                ]
            trainable = operator
            if len(inputs) == 1:
                inputs = inputs[0]
            trained: TrainedOperator
            if hasattr(trainable._impl, "partial_fit"):
                try:
                    num_epochs = trainable._impl_instance().num_epochs
                except AttributeError:
                    if num_epochs_batching is None:
                        warnings.warn(
                            "Operator {} does not have num_epochs and none given to Batching operator, using 1 as a default".format(
                                trainable.name()
                            )
                        )
                        num_epochs = 1
                    else:
                        num_epochs = num_epochs_batching
            else:
                raise AttributeError(
                    "All operators to be trained with batching need to implement partial_fit. {} doesn't.".format(
                        operator.name()
                    )
                )
            inputs_for_transform = inputs
            for epoch in range(num_epochs):
                for _, batch_data in enumerate(
                    inputs
                ):  # batching_transformer will output only one obj
                    if isinstance(batch_data, tuple):
                        batch_X, batch_y = batch_data
                    elif isinstance(batch_data, list):
                        batch_X = batch_data[0]
                        batch_y = batch_data[1]
                    else:
                        batch_X = batch_data
                        batch_y = None
                    if trainable.is_supervised():
                        try:
                            trained = trainable.partial_fit(batch_X, batch_y, classes=y)
                        except TypeError:
                            trained = trainable.partial_fit(batch_X, batch_y)
                    else:
                        trained = trainable.partial_fit(batch_X)
            trained = TrainedIndividualOp(
                trained.name(), trained._impl, trained._schemas
            )
            trained_map[operator] = trained
            trained_steps.append(trained)

            output = None
            for batch_idx, batch_data in enumerate(
                inputs_for_transform
            ):  # batching_transformer will output only one obj
                if isinstance(batch_data, tuple):
                    batch_X, batch_y = batch_data
                elif isinstance(batch_data, list):
                    batch_X = batch_data[0]
                    batch_y = batch_data[1]
                else:
                    batch_X = batch_data
                    batch_y = None
                if trained.is_transformer():
                    batch_output = trained.transform(batch_X, batch_y)
                else:
                    if trainable in sink_nodes:
                        batch_output = trained._predict(
                            X=batch_X
                        )  # We don't support y for predict yet as there is no compelling case
                    else:
                        # This is ok because trainable pipelines steps
                        # must only be individual operators
                        if hasattr(trained._impl, "predict_proba"):  # type: ignore
                            batch_output = trained.predict_proba(X=batch_X)
                        elif hasattr(trained._impl, "decision_function"):  # type: ignore
                            batch_output = trained.decision_function(X=batch_X)
                        else:
                            batch_output = trained._predict(X=batch_X)
                if isinstance(batch_output, tuple):
                    batch_out_X, batch_out_y = batch_output
                else:
                    batch_out_X = batch_output
                    batch_out_y = None
                if serialize:
                    output = lale.helpers.write_batch_output_to_file(
                        output,
                        os.path.join(
                            serialization_out_dir,
                            "fit_with_batches" + str(operator_idx) + ".hdf5",
                        ),
                        len(inputs_for_transform.dataset),
                        batch_idx,
                        batch_X,
                        batch_y,
                        batch_out_X,
                        batch_out_y,
                    )
                else:
                    if batch_out_y is None:
                        output = lale.helpers.append_batch(
                            output, (batch_output, batch_y)
                        )
                    else:
                        output = lale.helpers.append_batch(output, batch_output)
            if serialize:
                output.close()
                output = lale.helpers.create_data_loader(
                    os.path.join(
                        serialization_out_dir,
                        "fit_with_batches" + str(operator_idx) + ".hdf5",
                    ),
                    batch_size=inputs_for_transform.batch_size,
                )
            else:
                if isinstance(output, tuple):
                    output = lale.helpers.create_data_loader(
                        X=output[0],
                        y=output[1],
                        batch_size=inputs_for_transform.batch_size,
                    )
                else:
                    output = lale.helpers.create_data_loader(
                        X=output, y=None, batch_size=inputs_for_transform.batch_size
                    )
            outputs[operator] = output
            operator_idx += 1

        if serialize:
            shutil.rmtree(serialization_out_dir)
        trained_edges = [(trained_map[x], trained_map[y]) for (x, y) in edges]

        trained_steps2: Any = trained_steps
        result: TrainedPipeline = TrainedPipeline(
            trained_steps2, trained_edges, ordered=True
        )
        self._trained = result
        return result

    def is_transformer(self) -> bool:
        """ Checks if the operator is a transformer
        """
        sink_nodes = self._find_sink_nodes()
        all_transformers = [
            True if hasattr(operator._impl, "transform") else False
            for operator in sink_nodes
        ]
        return all(all_transformers)


TrainedOpType = TypeVar("TrainedOpType", bound=TrainedIndividualOp)


class TrainedPipeline(TrainablePipeline[TrainedOpType], TrainedOperator):
    def __init__(
        self,
        steps: List[TrainedOpType],
        edges: List[Tuple[TrainedOpType, TrainedOpType]],
        ordered: bool = False,
    ) -> None:
        super(TrainedPipeline, self).__init__(steps, edges, ordered=ordered)

    def _predict(self, X, y=None):
        return self._predict_based_on_type("predict", "_predict", X, y)

    def predict(self, X):
        result = self._predict(X)
        if isinstance(result, lale.datasets.data_schemas.NDArrayWithSchema):
            return lale.datasets.data_schemas.strip_schema(
                result
            )  # otherwise scorers return zero-dim array
        return result

    def transform(self, X, y=None):
        # TODO: What does a transform on a pipeline mean, if the last step is not a transformer
        # can it be just the output of predict of the last step?
        # If this implementation changes, check to make sure that the implementation of
        # self.is_transformer is kept in sync with the new assumptions.
        return self._predict_based_on_type("transform", "transform", X, y)

    def _predict_based_on_type(self, impl_method_name, operator_method_name, X, y=None):
        outputs = {}
        meta_outputs = {}
        sink_nodes = self._find_sink_nodes()
        for operator in self._steps:
            preds = self._preds[operator]
            if len(preds) == 0:
                inputs = [X]
                meta_data_inputs = {}
            else:
                inputs = [
                    outputs[pred][0]
                    if isinstance(outputs[pred], tuple)
                    else outputs[pred]
                    for pred in preds
                ]
                # we create meta_data_inputs as a dictionary with metadata from all previous steps
                # Note that if multiple previous steps generate the same key, it will retain only one of those.

                meta_data_inputs = {
                    key: meta_outputs[pred][key]
                    for pred in preds
                    if meta_outputs[pred] is not None
                    for key in meta_outputs[pred]
                }
            if len(inputs) == 1:
                inputs = inputs[0]
            if hasattr(operator._impl, "set_meta_data"):
                operator._impl_instance().set_meta_data(meta_data_inputs)
            meta_output = {}
            if operator in sink_nodes:
                if hasattr(
                    operator._impl, impl_method_name
                ):  # Since this is pipeline's predict, we should invoke predict from sink nodes
                    method_to_call_on_operator = getattr(operator, operator_method_name)
                    output = method_to_call_on_operator(X=inputs)
                else:
                    raise AttributeError(
                        "The sink node of the pipeline does not support",
                        operator_method_name,
                    )
            elif operator.is_transformer():
                output = operator.transform(X=inputs, y=y)
                if hasattr(operator._impl, "get_transform_meta_output"):
                    meta_output = operator._impl_instance().get_transform_meta_output()
            elif hasattr(
                operator._impl, "predict_proba"
            ):  # For estimator as a transformer, use predict_proba if available
                output = operator.predict_proba(X=inputs)
            elif hasattr(
                operator._impl, "decision_function"
            ):  # For estimator as a transformer, use decision_function if available
                output = operator.decision_function(X=inputs)
            else:
                output = operator._predict(X=inputs)
                if hasattr(operator._impl, "get_predict_meta_output"):
                    meta_output = operator._impl_instance().get_predict_meta_output()
            outputs[operator] = output
            meta_output_so_far = {
                key: meta_outputs[pred][key]
                for pred in preds
                if meta_outputs[pred] is not None
                for key in meta_outputs[pred]
            }
            meta_output_so_far.update(
                meta_output
            )  # So newest gets preference in case of collisions
            meta_outputs[operator] = meta_output_so_far
        result = outputs[self._steps[-1]]
        return result

    def predict_proba(self, X):
        """Probability estimates for all classes.

        Parameters
        ----------
        X :
            Features; see input_predict_proba schema of the operator.

        Returns
        -------
        result :
            Probabilities; see output_predict_proba schema of the operator.
        """
        return self._predict_based_on_type("predict_proba", "predict_proba", X)

    def decision_function(self, X):
        """Confidence scores for all classes.

        Parameters
        ----------
        X :
            Features; see input_decision_function schema of the operator.

        Returns
        -------
        result :
            Confidences; see output_decision_function schema of the operator.
        """
        return self._predict_based_on_type("decision_function", "decision_function", X)

    def transform_with_batches(self, X, y=None, serialize=True):
        """[summary]

        Parameters
        ----------
        X : [type]
            [description]
        y : [type], optional
            by default None
        Returns
        -------
        [type]
            [description]
        """
        outputs = {}

        if serialize:
            serialization_out_dir = os.path.join(
                os.path.dirname(__file__), "temp_serialized"
            )
            if not os.path.exists(serialization_out_dir):
                os.mkdir(serialization_out_dir)

        sink_nodes = self._find_sink_nodes()
        operator_idx = 0
        for operator in self._steps:
            preds = self._preds[operator]
            if len(preds) == 0:
                inputs = [X]
            else:
                inputs = [
                    outputs[pred][0]
                    if isinstance(outputs[pred], tuple)
                    else outputs[pred]
                    for pred in preds
                ]
            if len(inputs) == 1:
                inputs = inputs[0]
            trained = operator
            output = None
            for batch_idx, batch_data in enumerate(
                inputs
            ):  # batching_transformer will output only one obj
                if isinstance(batch_data, Tuple):
                    batch_X, batch_y = batch_data
                else:
                    batch_X = batch_data
                    batch_y = None
                if trained.is_transformer():
                    batch_output = trained.transform(batch_X, batch_y)
                else:
                    if trained in sink_nodes:
                        batch_output = trained._predict(
                            X=batch_X
                        )  # We don't support y for predict yet as there is no compelling case
                    else:
                        # This is ok because trainable pipelines steps
                        # must only be individual operators
                        if hasattr(trained._impl, "predict_proba"):  # type: ignore
                            batch_output = trained.predict_proba(X=batch_X)
                        elif hasattr(trained._impl, "decision_function"):  # type: ignore
                            batch_output = trained.decision_function(X=batch_X)
                        else:
                            batch_output = trained._predict(X=batch_X)
                if isinstance(batch_output, tuple):
                    batch_out_X, batch_out_y = batch_output
                else:
                    batch_out_X = batch_output
                    batch_out_y = None
                if serialize:
                    output = lale.helpers.write_batch_output_to_file(
                        output,
                        os.path.join(
                            serialization_out_dir,
                            "fit_with_batches" + str(operator_idx) + ".hdf5",
                        ),
                        len(inputs.dataset),
                        batch_idx,
                        batch_X,
                        batch_y,
                        batch_out_X,
                        batch_out_y,
                    )
                else:
                    if batch_out_y is not None:
                        output = lale.helpers.append_batch(
                            output, (batch_output, batch_out_y)
                        )
                    else:
                        output = lale.helpers.append_batch(output, batch_output)
            if serialize:
                output.close()
                output = lale.helpers.create_data_loader(
                    os.path.join(
                        serialization_out_dir,
                        "fit_with_batches" + str(operator_idx) + ".hdf5",
                    ),
                    batch_size=inputs.batch_size,
                )
            else:
                if isinstance(output, tuple):
                    output = lale.helpers.create_data_loader(
                        X=output[0], y=output[1], batch_size=inputs.batch_size
                    )
                else:
                    output = lale.helpers.create_data_loader(
                        X=output, y=None, batch_size=inputs.batch_size
                    )
            outputs[operator] = output
            operator_idx += 1

        return_data = outputs[self._steps[-1]].dataset.get_data()
        if serialize:
            shutil.rmtree(serialization_out_dir)

        return return_data

    def freeze_trainable(self) -> "TrainedPipeline":
        result = super(TrainedPipeline, self).freeze_trainable()
        return cast(TrainedPipeline, result)

    def is_frozen_trained(self) -> bool:
        for step in self.steps():
            if not step.is_frozen_trained():
                return False
        return True

    def freeze_trained(self) -> "TrainedPipeline":
        frozen_steps = []
        frozen_map = {}
        for liquid in self._steps:
            frozen = liquid.freeze_trained()
            frozen_map[liquid] = frozen
            frozen_steps.append(frozen)
        frozen_edges = [(frozen_map[x], frozen_map[y]) for x, y in self.edges()]
        result = TrainedPipeline(frozen_steps, frozen_edges, ordered=True)
        assert result.is_frozen_trained()
        return result

    def _lale_clone(self, cloner: Callable[[Any], Any]):
        """ This is really used for sklearn clone compatibility.
            Which mandates that clone returns something that has not been fit.
            So we enforce that here as well.
        """
        op = super()._lale_clone(cloner)
        return TrainedPipeline(op._steps, op._preds, True)


OperatorChoiceType = TypeVar("OperatorChoiceType", bound=Operator)


class OperatorChoice(PlannedOperator, Generic[OperatorChoiceType]):
    _steps: List[OperatorChoiceType]

    def __init__(self, steps, name: Optional[str]) -> None:
        # name is not optional as we assume that only make_choice calls this constructor and
        # it will always pass a name, and in case it is passed as None, we assign name as below:
        if name is None or name == "":
            name = lale.helpers.assignee_name(level=2)
        if name is None or name == "":
            name = "OperatorChoice"

        self._name = name
        self._steps = steps.copy()
        if self.is_classifier():
            self._estimator_type = (
                "classifier"  # satisfy sklearn.base.is_classifier(op)
            )

    def steps(self) -> List[OperatorChoiceType]:
        return self._steps

    def _lale_clone(self, cloner: Callable[[Any], Any]):
        steps = self._steps
        new_steps: List[OperatorChoiceType] = [s._lale_clone(cloner) for s in steps]
        return self.__class__(new_steps, self._name)

    def _has_same_impl(self, other: Operator) -> bool:
        """Checks if the type of the operator imnplementations are compatible
        """
        if not isinstance(other, OperatorChoice):
            return False
        my_steps = self.steps()
        other_steps = other.steps()
        if len(my_steps) != len(other_steps):
            return False

        for (m, o) in zip(my_steps, other_steps):
            if not m._has_same_impl(o):
                return False
        return True

    def is_supervised(self) -> bool:
        s = self.steps()
        if len(s) == 0:
            return False
        return self.steps()[-1].is_supervised()

    def validate_schema(self, X, y=None):
        for step in self.steps():
            step.validate_schema(X, y)

    def transform_schema(self, s_X):
        transformed_schemas = [st.transform_schema(s_X) for st in self.steps()]
        result = lale.type_checking.join_schemas(*transformed_schemas)
        return result

    def input_schema_fit(self) -> JSON_TYPE:
        pipeline_inputs = [s.input_schema_fit() for s in self.steps()]
        result = lale.type_checking.join_schemas(*pipeline_inputs)
        return result

    def freeze_trainable(self) -> "OperatorChoice":
        frozen_steps = []
        frozen_map = {}
        for liquid in self._steps:
            frozen = liquid.freeze_trainable()
            frozen_map[liquid] = frozen
            frozen_steps.append(frozen)
        result: "OperatorChoice" = OperatorChoice(frozen_steps, self._name)
        assert result.is_frozen_trainable()
        return result

    def is_frozen_trainable(self) -> bool:
        for step in self.steps():
            if not step.is_frozen_trainable():
                return False
        return True

    def is_classifier(self) -> bool:
        for op in self.steps():
            if not op.is_classifier():
                return False
        return True


class _PipelineFactory:
    def __init__(self):
        pass

    def __call__(self, steps: List[Any]):
        warnings.warn(
            "lale.operators.Pipeline is deprecated, use sklearn.pipeline.Pipeline or lale.lib.sklearn.Pipeline instead",
            DeprecationWarning,
        )
        for i in range(len(steps)):
            op = steps[i]
            if isinstance(op, tuple):
                assert isinstance(op[1], Operator)
                op[1]._set_name(op[0])
                steps[i] = op[1]
        return make_pipeline(*steps)


Pipeline = _PipelineFactory()


def make_pipeline_graph(steps, edges, ordered=False) -> PlannedPipeline:
    """
    Based on the state of the steps, it is important to decide an appropriate type for
    a new Pipeline. This method will decide the type, create a new Pipeline of that type and return it.
    #TODO: If multiple independently trained components are composed together in a pipeline,
    should it be of type TrainedPipeline?
    Currently, it will be TrainablePipeline, i.e. it will be forced to train it again.
    """

    isTrainable: bool = True
    isTrained: bool = True
    for operator in steps:
        if not isinstance(operator, TrainedOperator):
            isTrained = False  # Even if a single step is not trained, the pipeline can't be used for predict/transform
            # without training it first
        if isinstance(operator, OperatorChoice) or not isinstance(
            operator, TrainableOperator
        ):
            isTrainable = False
    if isTrained:
        return TrainedPipeline(steps, edges, ordered=ordered)
    elif isTrainable:
        return TrainablePipeline(steps, edges, ordered=ordered)
    else:
        return PlannedPipeline(steps, edges, ordered=ordered)


def make_pipeline(*orig_steps: Union[Operator, Any]) -> PlannedPipeline:
    steps, edges = [], []
    prev_op = None
    for curr_op in orig_steps:
        if isinstance(prev_op, BasePipeline):
            prev_leaves: Any = prev_op._find_sink_nodes()
        else:
            prev_leaves = [] if prev_op is None else [prev_op]
        if isinstance(curr_op, BasePipeline):
            curr_roots = curr_op._find_source_nodes()
            steps.extend(curr_op.steps())
            edges.extend(curr_op.edges())
        else:
            if not isinstance(curr_op, Operator):
                curr_op = make_operator(curr_op, name=curr_op.__class__.__name__)
            curr_roots = [curr_op]
            steps.append(curr_op)
        edges.extend([(src, tgt) for src in prev_leaves for tgt in curr_roots])
        prev_op = curr_op
    return make_pipeline_graph(steps, edges, ordered=True)


def make_union_no_concat(*orig_steps: Union[Operator, Any]) -> PlannedPipeline:
    steps, edges = [], []
    for curr_op in orig_steps:
        if isinstance(curr_op, BasePipeline):
            steps.extend(curr_op._steps)
            edges.extend(curr_op.edges())
        else:
            if not isinstance(curr_op, Operator):
                curr_op = make_operator(curr_op, name=curr_op.__class__.__name__)
            steps.append(curr_op)
    return make_pipeline_graph(steps, edges, ordered=True)


def make_union(*orig_steps: Union[Operator, Any]) -> PlannedPipeline:
    from lale.lib.lale import ConcatFeatures

    return make_union_no_concat(*orig_steps) >> ConcatFeatures()


def make_choice(
    *orig_steps: Union[Operator, Any], name: Optional[str] = None
) -> OperatorChoice:
    if name is None:
        name = ""
    name_: str = name  # to make mypy happy
    steps: List[Operator] = []
    for operator in orig_steps:
        if isinstance(operator, OperatorChoice):
            steps.extend(operator.steps())
        else:
            if not isinstance(operator, Operator):
                operator = make_operator(operator, name=operator.__class__.__name__)
            steps.append(operator)
        name_ = name_ + " | " + operator.name()
    return OperatorChoice(steps, name_[3:])


def _fixup_hyperparams_dict(d):
    d1 = remove_defaults_dict(d)
    d2 = {k: lale.helpers.val_wrapper.unwrap(v) for k, v in d1.items()}
    return d2
