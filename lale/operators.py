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

import lale.helpers
import lale.type_checking
from abc import ABC, abstractmethod
import importlib
import enum
import os
import itertools
from lale import schema2enums as enum_gen
import numpy as np
import pandas as pd
import lale.datasets.data_schemas

from typing import AbstractSet, Any, Callable, Dict, Generic, Iterable, Iterator, List, Tuple, TypeVar, Optional, Union, cast
import warnings
import copy
from lale.util.VisitorMeta import AbstractVisitorMeta
from lale.search.PGO import remove_defaults_dict
import inspect
from lale.schemas import Schema 
import jsonschema
import lale.pretty_print
import logging
import h5py
import shutil
import lale.json_operator
from sklearn.pipeline import if_delegate_has_method
import sklearn.base

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)
	
class Operator(metaclass=AbstractVisitorMeta):
    """Abstract base class for a LALE operator.

    Pipelines and individual operators extend this.
    
    """

    def __and__(self, other:'Operator')->'Operator':
        """Overloaded `and` operator. 
        
        Creates a union of the two operators.
        TODO:Explain more.

        Parameters
        ----------
        other : Operator
        
        Returns
        -------
        Pipeline
            Returns a pipeline that consists of union of self and other.
        """
        return make_union_no_concat(self, other)

    def __rand__(self, other:'Operator')->'Operator':
        return make_union_no_concat(other, self)

    def __rshift__(self, other:'Operator')->'Operator':
        """Overloaded `>>` operator. 
        
        Creates a pipeline of the two operators, current operator followed
        by the operator passed as parameter.

        Parameters
        ----------
        other : Operator
        
        Returns
        -------
        Pipeline
            Returns a pipeline that contains `self` followed by `other`.
        """
        return make_pipeline(self, other)

    def __rrshift__(self, other:'Operator')->'Operator':
        return make_pipeline(other, self)

    def __or__(self, other:'Operator')->'Operator':
        """Overloaded `or` operator. 
        
        Creates an OperatorChoice consisting of the two operators, current operator followed
        by the operator passed as parameter.

        Parameters
        ----------
        other : Operator
        
        Returns
        -------
        OperatorChoice
            Returns an OperatorChoice object that contains `self` and `other`.
        """
        return make_choice(self, other)

    def __ror__(self, other:'Operator')->'Operator':
        return make_choice(other, self)

    @abstractmethod
    def name(self)->str:
        """Returns the name of the operator.        
        """
        pass

    def class_name(self)->str:
        cls = self.__class__
        return cls.__module__ + '.' + cls.__name__

    @abstractmethod
    def set_name(self, name:str):
        """Sets the name of the operator.        
        """
        pass

    @abstractmethod
    def validate_schema(self, X, y=None):
        pass

    @abstractmethod
    def transform_schema(self, s_X):
        pass

    @abstractmethod
    def input_schema_fit(self):
        pass

    def to_json(self):
        """Returns the json representation of the operator.
        """
        return lale.json_operator.to_json(self, call_depth=2)

    def visualize(self, ipython_display:bool=True):
        """Visualize the operator using graphviz (use in a notebook).
        """
        return lale.helpers.to_graphviz(self, ipython_display, call_depth=2)

    def pretty_print(self, show_imports:bool=True, combinators:bool=True, ipython_display:bool=False):
        """Returns the Python source code representation of the operator.
        """
        result = lale.pretty_print.to_string(self, show_imports, combinators, call_depth=2)
        if ipython_display:
            import IPython.display
            markdown = IPython.display.Markdown(f'```python\n{result}\n```')
            return IPython.display.display(markdown)
        else:
            return result

    @abstractmethod
    def has_same_impl(self, other:'Operator')->bool:
        """Checks if the type of the operator implementations are compatible
        """
        pass

    @abstractmethod
    def _lale_clone(self, cloner:Callable[[Any],Any]):
        """ Method for cloning a lale operator, currently intended for internal use
        """
        pass

    @abstractmethod
    def is_supervised(self)->bool:
        """Checks if this operator needs labeled data for learning (the `y' argument for fit)
        """
        pass

class PlannedOperator(Operator):

    @abstractmethod
    def configure(self, *args, **kwargs)->'TrainableOperator':
        """Bind hyperparameters (for individual operators) and operator choices (for pipelines).

        Usually, this will be invoked via `__call__`, to conform to
        the look-and-feel of scikit-learn's `__init__` convention.
        Not all hyperparameters need to be explicitly specified. If
        some of the hyperparameters are not specified, then a
        subsequent `fit` invocation will bind them to their default
        values.  On the other hand, if some hyperparameters are still
        free and the operator is not marked as `is_frozen_trainable`,
        then a subsequent `auto_configure` search will include those
        free hyperparameters in its search space, to be tuned and then
        bound to the best found values.

        Parameters
        ----------
        args:
            The non-keyword arguments must be valid enumeration constants
            for hyperparameters of this operator that accepts categoricals,
            according to the `hyperparam_schema` of this operator.
        kwargs:
            Keyword arguments must be valid according to the
            `hyperparam_schema` of this operator, and can be categorical
            or continuous.

        Returns
        -------
        TrainableOperator
            A new copy of this operator that is the same except that some
            of its hyperparameters are bound to the specified values.

        """
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs)->'TrainableOperator':
        """Abstract method to make Planned objects callable.

        Operators in a Planned states are made callable by overriding 
        the __call__ method (https://docs.python.org/3/reference/datamodel.html#special-method-names).
        It is supposed to return an operator in a Trainable state.

        Parameters
        ----------
        args, kwargs: 
            The arguments are used to configure an operator in a Planned
            state to bind hyper-parameter values such that it becomes Trainable.

        Returns
        -------
        TrainableOperator
            A new copy of this operator that is the same except that some
            of its hyperparameters are bound to the specified values.
        """
        pass

    def auto_configure(self, X, y = None, optimizer = None, cv = None, scoring = None, **kwargs)->'TrainableOperator':
        """
        Perform CASH (Combined algorithm selection and hyper-parameter tuning) on the planned
        operator. The output is a trainable/trained operator chosen by the optimizer.
        
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
            default is None, which will use the optimizer's default value.
        scoring:
            scoring option that is valid for the optimizer.
            default is None, which will use the optimizer's default value.
        kwargs:
            other keyword arguments to be passed to the optimizer.
        
        Returns
        -------
        Trainable that is chosen after performing CASH.
        """
        if optimizer is None:
            raise ValueError("Please provide a valid optimizer for auto_configure.")
        if kwargs is None:
            kwargs = {}
        if cv is not None:
            kwargs['cv'] = cv
        if scoring is not None:
            kwargs['scoring'] = scoring
        optimizer_obj = optimizer(estimator=self, **kwargs)
        trained = optimizer_obj.fit(X, y)
        return trained.get_pipeline()

class TrainableOperator(PlannedOperator):

    @abstractmethod
    def fit(self, X, y=None, **fit_params)->'TrainedOperator':
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
    def is_frozen_trainable(self)->bool:
        """Return true if all hyperparameters are bound, in other words,
           search spaces contain no free hyperparameters to be tuned.
        """
        pass

    @abstractmethod
    def freeze_trainable(self)->'TrainableOperator':
        """Return a copy of this trainable operator that is the same except
           that all hyperparameters are bound and none are free to be tuned.
        """
        pass

class TrainedOperator(TrainableOperator):

    @abstractmethod
    def predict(self, X):
        """Abstract predict method to be overriden by trained operators as applicable.
        
        Parameters
        ----------
        X : 
            The type of X is as per input_predict schema of the operator.
        """
        pass

    @abstractmethod
    def transform(self, X, y = None):
        """Abstract transform method to be overriden by trained operators as applicable.
        
        Parameters
        ----------
        X : 
            The type of X is as per input_predict schema of the operator.
        """
        pass

    @abstractmethod
    def predict_proba(self, X):
        """Abstract predict method to be overriden by trained operators as applicable.

        Parameters
        ----------
        X :
            The type of X is as per input_predict schema of the operator.
        """
        pass

    @abstractmethod
    def is_frozen_trained(self)->bool:
        """Return true if all learnable coefficients are bound, in other
           words, there are no free parameters to be learned by fit.
        """
        pass

    @abstractmethod
    def freeze_trained(self)->'TrainedOperator':
        """Return a copy of this trainable operator that is the same except
           that all learnable coefficients are bound and thus fit is a no-op.
        """
        pass
    
    @abstractmethod
    def is_transformer(self)->bool:
        """ Checks if the operator is a transformer
        """
        pass

_schema_derived_attributes = ['_enum_attributes', '_hyperparam_defaults']

class IndividualOp(Operator):
    """
    This is a concrete class that can instantiate a new individual
    operator and provide access to its metadata.
    """

    _name:str
    _impl:Any

    def __init__(self, name:str, impl, schemas) -> None:
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

        # Add enums from the hyperparameter schema to the object as fields
        # so that their usage looks like LogisticRegression.penalty.l1
#        enum_gen.addSchemaEnumsAsFields(self, self.hyperparam_schema())

    _enum_attributes:Optional[Dict[str, enum.Enum]]

    @property
    def enum_attributes(self)->Dict[str, enum.Enum]:
        ea = getattr(self, '_enum_attributes', None)
        if ea is None:
            nea = enum_gen.schemaToPythonEnums(self.hyperparam_schema())
            self._enum_attributes = nea
            return nea
        else:
            return ea


    def _invalidate_enum_attributes(self)->None:
        for k in _schema_derived_attributes:
            try:
                delattr(self, k)
            except AttributeError:
                pass

    def __getattr__(self, name:str):
        if name in _schema_derived_attributes or name in [
            '__setstate__', '_schemas']:
            raise AttributeError
        ea = self.enum_attributes
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

    def get_schema(self, schema_kind:str)->Dict[str, Any]:
        """Return a schema of the operator.
        
        Parameters
        ----------
        schema_kind : string, 'hyperparams' or 'input_fit' or 'input_predict' or 'input_predict_proba' or 'input_transform' 'output_predict' or 'output_predict_proba' or 'output_transform'
                Type of the schema to be returned.    
                    
        Returns
        -------
        dict
            The python object containing the json schema of the operator. 
            For all the schemas currently present, this would be a dictionary.
        """
        props = self._schemas['properties']
        assert schema_kind in props, f'missing schema {schema_kind} for operator {self.name()} with class {self.class_name()}'
        result = props[schema_kind]
        return result

    def documentation_url(self):
        if 'documentation_url' in self._schemas:
            return self._schemas['documentation_url']
        return None

    def get_tags(self)->Dict[str, List[str]]:
        """Return the tags of an operator.
        
        Returns
        -------
        list
            A list of tags describing the operator.
        """
        return self._schemas.get('tags', {})

    def has_tag(self, tag:str)->bool:
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
        tags = [t for l in self.get_tags().values() for t in l]
        return tag in tags

    def input_schema_fit(self):
        """Returns the schema for fit method's input.
        
        Returns
        -------
        dict
            Logical schema describing input required by this 
            operator's fit method.
        """
        return self.get_schema('input_fit')

    def input_schema_predict(self):
        """Returns the schema for predict method's input.
        
        Returns
        -------
        dict
            Logical schema describing input required by this 
            operator's predict method.
        """
        return self.get_schema('input_predict')

    def input_schema_predict_proba(self):
        """Returns the schema for predict proba method's input.

        Returns
        -------
        dict
            Logical schema describing input required by this
            operator's predict proba method.
        """
        return self.get_schema('input_predict_proba')

    def input_schema_transform(self):
        """Returns the schema for transform method's input.
        
        Returns
        -------
        dict
            Logical schema describing input required by this 
            operator's transform method.
        """
        return self.get_schema('input_transform')

    def output_schema_predict(self):
        """Returns the schema for predict method's output.

        Returns
        -------
        dict
            Logical schema describing output of this
            operator's predict proba method.
        """
        return self.get_schema('output_predict')

    def output_schema_predict_proba(self):
        """Returns the schema for predict_proba method's output.

        Returns
        -------
        dict
            Logical schema describing output of this
            operator's predict_proba method.
        """
        return self.get_schema('output_predict_proba')

    def output_schema_transform(self):
        """Returns the schema for transform method's output.

        Returns
        -------
        dict
            Logical schema describing output of this
            operator's transform method.
        """
        return self.get_schema('output_transform')

    def hyperparam_schema(self, name:Optional[str]=None):
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
        hp_schema = self.get_schema('hyperparams')
        if name is None:
            return hp_schema
        else:
            params = next(iter(hp_schema.get('allOf',[])))
            return params.get('properties', {}).get(name)

    def hyperparam_defaults(self):
        """Returns the default values of hyperparameters for the operator.
        
        Returns
        -------
        dict
            A dictionary with names of the hyperparamers as keys and 
            their default values as values.
        """
        if not hasattr(self, '_hyperparam_defaults'):
            schema = self.hyperparam_schema()
            props = next(iter(schema.get('allOf',[])), {}).get('properties', {})
            defaults = { k: props[k].get('default') for k in props.keys() }
            self._hyperparam_defaults = defaults
        return self._hyperparam_defaults

    def get_param_ranges(self)->Tuple[Dict[str,Any], Dict[str,Any]]:
        """Returns two dictionaries, ranges and cat_idx, for hyperparameters.

        The ranges dictionary has two kinds of entries. Entries for
        numeric and Boolean hyperparameters are tuples of the form
        (min, max, default). Entries for categorical hyperparameters
        are lists of their values.

        The cat_idx dictionary has (min, max, default) entries of indices
        into the corresponding list of values.
        """

        hyperparam_obj = next(iter(self.hyperparam_schema().get('allOf',[])))
        original = hyperparam_obj.get('properties')
        def is_relevant(hp, s):
            if 'relevantToOptimizer' in hyperparam_obj:
                return hp in hyperparam_obj['relevantToOptimizer']
            return True
        relevant = {hp: s for hp, s in original.items() if is_relevant(hp, s)}
        def pick_one_type(schema):
            if 'anyOf' in schema:
                def by_type(typ):
                    for s in schema['anyOf']:
                        if 'type' in s and s['type'] == typ:
                            if ('forOptimizer' not in s) or s['forOptimizer']:
                                return s
                    return None
                for typ in ['number', 'integer', 'string']:
                    s = by_type(typ)
                    if s:
                        return s
                return schema['anyOf'][0]
            return schema
        unityped = {hp: pick_one_type(relevant[hp]) for hp in relevant}
        def add_default(schema):
            if 'type' in schema:
                minimum, maximum = 0.0, 1.0
                if 'minimumForOptimizer' in schema:
                    minimum = schema['minimumForOptimizer']
                elif 'minimum' in schema:
                    minimum = schema['minimum']
                if 'maximumForOptimizer' in schema:
                    maximum = schema['maximumForOptimizer']
                elif 'maximum' in schema:
                    maximum = schema['maximum']
                result = {**schema}
                if schema['type'] in ['number', 'integer']:
                    if 'default' not in schema:
                        schema['default'] = None
                    if 'minimumForOptimizer' not in schema:
                        result['minimumForOptimizer'] = minimum
                    if 'maximumForOptimizer' not in schema:
                        result['maximumForOptimizer'] = maximum
                return result
            elif 'enum' in schema:
                if 'default' in schema:
                    return schema
                return {'default': schema['enum'][0], **schema}
            return schema
        defaulted = {hp: add_default(unityped[hp]) for hp in unityped}
        def get_range(hp, schema):
            if 'enum' in schema:
                default = schema['default']
                non_default = [v for v in schema['enum'] if v != default]
                return [*non_default, default]
            elif schema['type'] == 'boolean':
                return (False, True, schema['default'])
            else:
                def get(schema, key):
                    return schema[key] if key in schema else None
                keys = ['minimumForOptimizer', 'maximumForOptimizer', 'default']
                return tuple([get(schema, key) for key in keys])
        def get_cat_idx(schema):
            if 'enum' not in schema:
                return None
            return (0, len(schema['enum'])-1, len(schema['enum'])-1)
        autoai_ranges = {hp: get_range(hp, s) for hp, s in defaulted.items()}
        autoai_cat_idx = {hp: get_cat_idx(s)
                       for hp, s in defaulted.items() if 'enum' in s}
        return autoai_ranges, autoai_cat_idx

    def _enum_to_strings(self, arg:enum.Enum)->Tuple[str, Any]:
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

        if not isinstance(arg, enum.Enum):
            raise ValueError('Missing keyword on argument {}.'.format(arg))
        return arg.__class__.__name__, arg.value

    def name(self)->str:
        """[summary]
        
        Returns
        -------
        [type]
            [description]
        """

        return self._name

    def set_name(self, name):
        self._name = name

    def _impl_class(self):
        if inspect.isclass(self._impl):
            return self._impl
        return self._impl.__class__

    def _impl_instance(self):
        if inspect.isclass(self._impl):
            class_ = self._impl
            try:
                instance = class_() #always with default values of hyperparams
            except TypeError as e:
                logger.debug(
                    f'Constructor for {class_.__module__}.{class_.__name__} '
                    f'threw exception {e}')
                instance = class_.__new__(class_)
            self._impl = instance
        return self._impl

    def class_name(self)->str:
        module = self._impl.__module__
        if module is None or module == str.__class__.__module__:
            class_name = self.name()
        else:
            class_name = module + '.' + self._impl_class().__name__
        return class_name

    def __str__(self)->str:
        return self.name()

    def has_same_impl(self, other:Operator)->bool:
        """Checks if the type of the operator implementations are compatible
        """
        if not isinstance(other, IndividualOp):
            return False
        return self._impl_class() == other._impl_class()
        
    def _lale_clone(self, cloner:Callable[[Any],Any]):
        impl = self._impl
        if impl is not None and not inspect.isclass(impl):
            impl = cloner(impl)
        cp = self.__class__(self._name, impl, self._schemas)
        return cp

    def customize_schema(self, **kwargs: Schema) -> 'IndividualOp':
        """Return a new operator with a customized schema
        
        Parameters
        ----------
        schema : Schema
            A dictionary of json schemas for the operator. Override the entire schema and ignore other arguments
        input : Schema
            (or `input_*`) override the input schema for method `*`.
            `input_*` must be an existing method (already defined in the schema for lale operators, exising method for external operators)
        output : Schema
            (or `output_*`) override the output schema for method `*`.
            `output_*` must be an existing method (already defined in the schema for lale operators, exising method for external operators)
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
        for arg in kwargs:
            value = kwargs[arg]
            if arg == 'schemas':
                value.schema['$schema'] = 'http://json-schema.org/draft-04/schema#'
                lale.helpers.validate_is_schema(value.schema)
                op._schemas = value.schema
                break
            elif arg.startswith('input') or arg.startswith('output'):
            # multiple input types (e.g., fit, predict)
                lale.type_checking.validate_method(op, arg)
                lale.helpers.validate_is_schema(value.schema)
                op._schemas['properties'][arg] = value.schema
            elif arg == 'constraint':
                op._schemas['properties']['hyperparams']['allOf'].append(value.schema)
            elif arg == 'relevantToOptimizer':
                assert isinstance(value, list)
                op._schemas['properties']['hyperparams']['allOf'][0]['relevantToOptimizer'] = value
            elif arg in lale.type_checking.get_hyperparam_names(op):
                op._schemas['properties']['hyperparams']['allOf'][0]['properties'][arg] = value.schema
            elif arg == 'tags':
                assert isinstance(value, dict)
                op._schemas['tags'] = value
            else:
                assert False, "Unknown method or parameter."
        # since the schema has changed, we need to invalidate any
        # cached enum attributes
        self._invalidate_enum_attributes()
        return op

    def validate_schema(self, X, y=None):
        if hasattr(self._impl, 'fit'):
            X = self._validate_input_schema('X', X, 'fit')
        method = 'transform' if self.is_transformer() else 'predict'
        self._validate_input_schema('X', X, method)
        if self.is_supervised(default_if_missing=False):
            if y is None:
                raise ValueError(f'{self.name()}.fit() y cannot be None')
            else:
                if hasattr(self._impl, 'fit'):
                    y = self._validate_input_schema('y', y, 'fit')
                self._validate_input_schema('y', y, method)

    def _validate_input_schema(self, arg_name, arg, method):
        if not lale.helpers.is_empty_dict(arg):
            if method == 'fit' or method == 'partial_fit':
                schema = self.input_schema_fit()
            elif method == 'predict':
                schema = self.input_schema_predict()
            elif method == 'predict_proba':
                schema = self.input_schema_predict_proba()
            elif method == 'transform':
                schema = self.input_schema_transform()
            if 'properties' in schema and arg_name in schema['properties']:
                arg = lale.datasets.data_schemas.add_schema(arg)
                try:
                    sup = schema['properties'][arg_name]
                    lale.type_checking.validate_schema_or_subschema(arg, sup)
                except Exception as e:
                    raise ValueError(f'{self.name()}.{method}() invalid {arg_name}: {e}') from e
        return arg

    def _validate_output_schema(self, result, method):
        if method == 'predict':
            schema = self.output_schema_predict()
        elif method == 'predict_proba':
            schema = self.output_schema_predict_proba()
        elif method == 'transform':
            schema = self.output_schema_transform()
        result = lale.datasets.data_schemas.add_schema(result)
        try:
            lale.type_checking.validate_schema_or_subschema(result, schema)
        except Exception as e:
            print(f'{self.name()}.{method}() invalid result: {e}')
            raise ValueError(f'{self.name()}.{method}() invalid result: {e}') from e
        return result

    def transform_schema(self, s_X):
        if self.is_transformer():
            return self.output_schema_transform()
        if hasattr(self._impl, 'predict_proba'):
            return self.output_schema_predict_proba()
        return self.output_schema_predict()

    def is_supervised(self, default_if_missing=True)->bool:
        if hasattr(self._impl, 'fit'):
            schema_fit = self.input_schema_fit()
            return lale.type_checking.is_subschema(schema_fit, _is_supervised_schema)
        return default_if_missing

_is_supervised_schema = {
    'type': 'object',
    'required': ['y']
}

class PlannedIndividualOp(IndividualOp, PlannedOperator):
    """
    This is a concrete class that returns a trainable individual
    operator through its __call__ method. A configure method can use
    an optimizer and return the best hyperparameter combination.
    """
    _hyperparams:Optional[Dict[str,Any]]

    def __init__(self, _name:str, _impl, _schemas) -> None:
        super(PlannedIndividualOp, self).__init__(_name, _impl, _schemas)
        self._hyperparams = None

    def _configure(self, *args, **kwargs)->'TrainableIndividualOp':
        class_ = self._impl_class()
        hyperparams = { }
        for arg in args:
            k, v = self._enum_to_strings(arg)
            hyperparams[k] = v
        for k, v in fixup_hyperparams_dict(kwargs).items():
            
            if k in hyperparams:
                raise ValueError('Duplicate argument {}.'.format(k))
            v = lale.helpers.val_wrapper.unwrap(v)
            if isinstance(v, enum.Enum):
                k2, v2 = self._enum_to_strings(v)
                if k != k2:
                    raise ValueError(
                        'Invalid keyword {} for argument {}.'.format(k2, v2))
            else:
                v2 = v
            hyperparams[k] = v2
        #using params_all instead of hyperparams to ensure the construction is consistent with schema
        trainable_to_get_params = TrainableIndividualOp(_name=self.name(), _impl=None, _schemas=self._schemas)
        trainable_to_get_params._hyperparams = hyperparams
        params_all = trainable_to_get_params.get_params_all()
        try:
            lale.type_checking.validate_schema(params_all, self.hyperparam_schema())
        except jsonschema.ValidationError as e_orig:
            e = e_orig if e_orig.parent is None else e_orig.parent
            lale.helpers.validate_is_schema(e.schema)
            schema = lale.pretty_print.to_string(e.schema)
            if [*e.schema_path][:3] == ['allOf', 0, 'properties']:
                arg = e.schema_path[3]
                reason = f'invalid value {arg}={e.instance}'
                schema_path = f'argument {arg}'
            elif [*e.schema_path][:3] == ['allOf', 0, 'additionalProperties']:
                pref, suff = 'Additional properties are not allowed (', ')'
                assert e.message.startswith(pref) and e.message.endswith(suff)
                reason = 'argument ' + e.message[len(pref):-len(suff)]
                schema_path = 'arguments and their defaults'
                schema = self.hyperparam_defaults()
            elif e.schema_path[0] == 'allOf' and int(e.schema_path[1]) != 0:
                assert e.schema_path[2] == 'anyOf'
                descr = e.schema['description']
                if descr.endswith('.'):
                    descr = descr[:-1]
                reason = f'constraint {descr[0].lower()}{descr[1:]}'
                schema_path = f'constraint {e.schema_path[1]}'
            else:
                reason = e.message
                schema_path = e.schema_path
            msg = f'Invalid configuration for {self.name()}(' \
                + f'{lale.pretty_print.hyperparams_to_string(hyperparams)}) ' \
                + f'due to {reason}.\n' \
                + f'Schema of {schema_path}: {schema}\n' \
                + f'Value: {e.instance}'
            raise jsonschema.ValidationError(msg) from e

        if len(params_all) == 0:
            impl = class_()
        else:
            impl = class_(**params_all)

        result = TrainableIndividualOp(_name=self.name(), _impl=impl, _schemas=self._schemas)
        result._hyperparams = hyperparams
        return result

    def __call__(self, *args, **kwargs)->'TrainableIndividualOp':
        return self._configure(*args, **kwargs)

    def configure(self, *args, **kwargs)->'TrainableIndividualOp':
        return self.__call__(*args, **kwargs)

    def is_transformer(self)->bool:
        """ Checks if the operator is a transformer
        """
        return hasattr(self._impl, 'transform')

    def hyperparam_schema_with_hyperparams(self):
        schema = self.hyperparam_schema()
        params = None
        try:
            params = self._hyperparams
        except AttributeError:
            pass
        if not params:
            return schema
        props = {k : {'enum' : [v]} for k, v in params.items()}
        obj = {'type':'object', 'properties':props}
        obj['relevantToOptimizer'] = list(params.keys())
        top = {'allOf':[schema, obj]}
        return top
    # This should *only* ever be called by the sklearn_compat wrapper
    def set_params(self, **impl_params):
        return self._configure(**impl_params)

def _mutation_warning(method_name:str)->str:
    msg = str("The `{}` method is deprecated on a trainable "
              "operator, because the learned coefficients could be "
              "accidentally overwritten by retraining. Call `{}` "
              "on the trained operator returned by `fit` instead.")
    return msg.format(method_name, method_name)

class TrainableIndividualOp(PlannedIndividualOp, TrainableOperator):
    def __init__(self, _name, _impl, _schemas):
        super(TrainableIndividualOp, self).__init__(_name, _impl, _schemas)

    def fit(self, X, y = None, **fit_params)->'TrainedIndividualOp':
        X = self._validate_input_schema('X', X, 'fit')
        y = self._validate_input_schema('y', y, 'fit')
        filtered_fit_params = fixup_hyperparams_dict(fit_params)
        try:
            trainable_impl = sklearn.base.clone(self._impl_instance())
        except BaseException: #as clone can raise TypeError or RuntimeError
            trainable_impl = self._impl_instance()
        if filtered_fit_params is None:
            trained_impl = trainable_impl.fit(X, y)
        else:
            trained_impl = trainable_impl.fit(X, y, **filtered_fit_params)
        result = TrainedIndividualOp(self.name(), trained_impl, self._schemas)
        result._hyperparams = self._hyperparams
        self.__trained = result
        return result

    def partial_fit(self, X, y = None, **fit_params)->TrainedOperator:
        if not hasattr(self._impl, "partial_fit"):
            raise AttributeError(f'{self.name()} has no partial_fit implemented.')
        X = self._validate_input_schema('X', X, 'partial_fit')
        y = self._validate_input_schema('y', y, 'partial_fit')
        filtered_fit_params = fixup_hyperparams_dict(fit_params)
        if filtered_fit_params is None:
            trained_impl = self._impl_instance().partial_fit(X, y)
        else:
            trained_impl = self._impl_instance().partial_fit(
                X, y, **filtered_fit_params)
        result = TrainedIndividualOp(self.name(), trained_impl, self._schemas)
        result._hyperparams = self._hyperparams
        self.__trained = result
        return result

    @if_delegate_has_method(delegate='_impl')
    def predict(self, X):
        """
        .. deprecated:: 0.0.0
           The `predict` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `predict`
           on the trained operator returned by `fit` instead.
        """
        warnings.warn(_mutation_warning('predict'), DeprecationWarning)
        try:
            return self.__trained.predict(X)
        except AttributeError:
            raise ValueError('Must call `fit` before `predict`.')

    @if_delegate_has_method(delegate='_impl')
    def transform(self, X, y = None):
        """
        .. deprecated:: 0.0.0
           The `transform` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `transform`
           on the trained operator returned by `fit` instead.
        """
        warnings.warn(_mutation_warning('transform'), DeprecationWarning)
        try:
            return self.__trained.transform(X, y)
        except AttributeError:
            raise ValueError('Must call `fit` before `transform`.')

    @if_delegate_has_method(delegate='_impl')
    def predict_proba(self, X):
        """
        .. deprecated:: 0.0.0
           The `predict_proba` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `predict_proba`
           on the trained operator returned by `fit` instead.
        """
        warnings.warn(_mutation_warning('predict_proba'), DeprecationWarning)
        try:
            return self.__trained.predict_proba(X)
        except AttributeError:
            raise ValueError('Must call `fit` before `predict_proba`.')

    def free_hyperparams(self):
        hyperparam_schema = self.hyperparam_schema()
        if 'allOf' in hyperparam_schema and \
           'relevantToOptimizer' in hyperparam_schema['allOf'][0]:
            to_bind = hyperparam_schema['allOf'][0]['relevantToOptimizer']
        else:
            to_bind = []
        if self._hyperparams:
            bound = self._hyperparams.keys()
        else:
            bound = []
        return set(to_bind) - set(bound)

    def is_frozen_trainable(self)->bool:
        free = self.free_hyperparams()
        return len(free) == 0

    def _freeze_trainable_bindings(self):
        old_bindings = self._hyperparams if self._hyperparams else {}
        free = self.free_hyperparams()
        defaults = self.hyperparam_defaults()
        new_bindings = {name: defaults[name] for name in free}
        bindings = {**old_bindings, **new_bindings}
        return bindings

    def freeze_trainable(self)->'TrainableIndividualOp':
        bindings = self._freeze_trainable_bindings()
        result = self._configure(**bindings)
        assert result.is_frozen_trainable(), str(result.free_hyperparams())
        return result

    def get_params(self, deep:bool=True)->Dict[str,Any]:
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
        out['_name'] = self._name
        out['_schemas'] = self._schemas
        impl = self._impl_instance()
        out['_impl'] = impl
        if deep and hasattr(impl, 'get_params'):
            deep_items = impl.get_params().items()
            out.update((self._name + '__' + k, val) for k, val in deep_items)
        return out

    def get_params_set_by_user(self):
        #also to make sure that self._hyperparams is a dictionary?
        if self._hyperparams is None:
            return None
        return {**self._hyperparams}

    def hyperparams(self):
        if self._hyperparams is None:
            return None
        actuals, defaults = self._hyperparams, self.hyperparam_defaults()
        return {k: actuals[k] for k in actuals if
                k not in defaults or actuals[k] != defaults[k]}

    def get_params_all(self):
        output = {}
        result = self.get_params_set_by_user()
        if result is not None:
            output.update(result)
        defaults = self.hyperparam_defaults()
        for k in defaults.keys():
            if k not in output:
                output[k] = defaults[k]
        return output

    # This should *only* ever be called by the sklearn_compat wrapper
    def set_params(self, **impl_params):
        #TODO: This mutates the operator, should we mark it deprecated?
        filtered_impl_params = fixup_hyperparams_dict(impl_params)
        self._impl = lale.helpers.create_individual_op_using_reflection(
            self.class_name(), self._name, filtered_impl_params)
        self._hyperparams = filtered_impl_params
        return self

    def transform_schema(self, s_X):
        if hasattr(self._impl, 'transform_schema'):
            try:
                return self._impl_instance().transform_schema(s_X)
            except BaseException as e:
                raise ValueError(f'unexpected error in {self.name()}.transform_schema({lale.pretty_print.to_string(s_X)}') from e                
        else:
            return super(TrainableIndividualOp, self).transform_schema(s_X)

    def input_schema_fit(self):
        if hasattr(self._impl, 'input_schema_fit'):
            return self._impl_instance().input_schema_fit()
        else:
            return super(TrainableIndividualOp, self).input_schema_fit()

    def _lale_clone(self, cloner:Callable[[Any],Any]):
        impl = self._impl
        if impl is not None and not inspect.isclass(impl):
            impl = cloner(impl)
        cp = make_operator(impl, schemas=self._schemas, name=self._name)
        if isinstance(cp, PlannedIndividualOp):
            cp._hyperparams = self._hyperparams
        return cp

class TrainedIndividualOp(TrainableIndividualOp, TrainedOperator):
    _frozen_trained:bool

    def __init__(self, _name, _impl, _schemas):
        super(TrainedIndividualOp, self).__init__(_name, _impl, _schemas)
        self._frozen_trained = not hasattr(self._impl, 'fit')

    def __call__(self, *args, **kwargs)->'TrainedIndividualOp':
        filtered_kwargs_params = fixup_hyperparams_dict(kwargs)

        trainable = self._configure(*args, **filtered_kwargs_params)
        instance = TrainedIndividualOp(trainable._name, trainable._impl, trainable._schemas)
        instance._hyperparams = trainable._hyperparams
        return instance

    def fit(self, X, y = None, **fit_params)->'TrainedIndividualOp':
        if hasattr(self._impl, "fit") and not self.is_frozen_trained():
            filtered_fit_params = fixup_hyperparams_dict(fit_params)
            return super(TrainedIndividualOp, self).fit(X, y, **filtered_fit_params)
        else:
            return self 

    @if_delegate_has_method(delegate='_impl')
    def predict(self, X):
        X = self._validate_input_schema('X', X, 'predict')
        result = self._impl_instance().predict(X)
        self._validate_output_schema(result, 'predict')
        return result

    @if_delegate_has_method(delegate='_impl')
    def transform(self, X, y = None):
        X = self._validate_input_schema('X', X, 'transform')
        if ('y' in [required_property.lower() for required_property 
                    in self.input_schema_transform().get('required',[])]):
            y = self._validate_input_schema('y', y, 'transform')
            result = self._impl_instance().transform(X, y)
        else:
            result = self._impl_instance().transform(X)
        result = self._validate_output_schema(result, 'transform')
        return result

    @if_delegate_has_method(delegate='_impl')
    def predict_proba(self, X):
        X = self._validate_input_schema('X', X, 'predict_proba')
        if hasattr(self._impl, 'predict_proba'):
            result = self._impl_instance().predict_proba(X)
        else:
            raise ValueError("The operator {} does not support predict_proba".format(self.name()))
        result = self._validate_output_schema(result, 'predict_proba')
        return result

    def freeze_trainable(self)->'TrainedIndividualOp':
        result = copy.deepcopy(self)
        result._hyperparams = self._freeze_trainable_bindings()
        assert result.is_frozen_trainable(), str(result.free_hyperparams())
        assert isinstance(result, TrainedIndividualOp)
        return result

    def is_frozen_trained(self)->bool:
        return self._frozen_trained

    def freeze_trained(self)->'TrainedIndividualOp':
        if self.is_frozen_trained():
            return self
        result = copy.deepcopy(self)
        result._frozen_trained = True
        assert result.is_frozen_trained()
        return result

    @if_delegate_has_method(delegate='_impl')
    def get_pipeline(self, pipeline_name=None, astype='lale')->Optional[TrainableOperator]:
        result = self._impl_instance().get_pipeline(pipeline_name, astype)
        return result

    @if_delegate_has_method(delegate='_impl')
    def summary(self)->pd.DataFrame:
        return self._impl_instance().summary()

    def _lale_clone(self, cloner:Callable[[Any],Any]):
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

all_available_operators: List[PlannedOperator] = []

def make_operator(impl, schemas = None, name = None) -> PlannedIndividualOp:
    if name is None:
        name = lale.helpers.assignee_name()
    if inspect.isclass(impl):
        if hasattr(impl, 'fit'):
            operatorObj = PlannedIndividualOp(name, impl, schemas)
        else:
            operatorObj = TrainedIndividualOp(name, impl, schemas)
    else:
        if hasattr(impl, 'fit'):
            operatorObj = TrainableIndividualOp(name, impl, schemas)
        else:
            operatorObj = TrainedIndividualOp(name, impl, schemas)
        if hasattr(impl, 'get_params'):
            operatorObj._hyperparams = {**impl.get_params()}
    all_available_operators.append(operatorObj)
    return operatorObj

def get_available_operators(tag: str, more_tags: AbstractSet[str] = None) -> List[PlannedOperator]:
    singleton = set([tag])
    tags = singleton if (more_tags is None) else singleton.union(more_tags)
    def filter(op):
        tags_dict = op.get_tags()
        if tags_dict is None:
            return False
        tags_set = {tag for prefix in tags_dict for tag in tags_dict[prefix]}
        return tags.issubset(tags_set)
    return [op for op in all_available_operators if filter(op)]

def get_available_estimators(tags: AbstractSet[str] = None) -> List[PlannedOperator]:
    return get_available_operators('estimator', tags)

def get_available_transformers(tags: AbstractSet[str] = None) -> List[PlannedOperator]:
    return get_available_operators('transformer', tags)

OpType = TypeVar('OpType', bound=Operator)
class BasePipeline(Operator, Generic[OpType]):
    """
    This is a concrete class that can instantiate a new pipeline operator and provide access to its meta data.
    """
    _steps:List[OpType]
    _preds:Dict[OpType, List[OpType]]
    _name:str

    def _lale_clone(self, cloner:Callable[[Any], Any]):
        steps = self._steps
        new_steps:List[OpType] = [s._lale_clone(cloner) for s in steps]
        if self._preds is None:
            return self.__class__(new_steps, None, True)

        step_map:Dict[OpType, OpType] = {steps[i]:new_steps[i] for i in range(len(steps))}
        new_edges = ((step_map[s],step_map[d]) for s,d in self.edges())
        return self.__class__(new_steps, new_edges, True)
        
    def __init__(self, 
                steps:List[OpType], 
                edges:Optional[Iterable[Tuple[OpType, OpType]]], 
                ordered:bool=False) -> None:
        self._name = "pipeline_" + str(id(self))  
        self._preds = {}              
        for step in steps:
            assert isinstance(step, Operator)
        if edges is None: 
            #Which means there is a linear pipeline #TODO:Test extensively with clone and get_params
            #This constructor is mostly called due to cloning. Make sure the objects are kept the same.
            self.constructor_for_cloning(steps)
        else:
            self._steps = []
            
            for step in steps:
                if step in self._steps:
                    raise ValueError('Same instance of {} already exists in the pipeline. '\
                    'This is not allowed.'.format(step.name()))
                if isinstance(step, BasePipeline):
                    #Flatten out the steps and edges
                    self._steps.extend(step.steps())
                    #from step's edges, find out all the source and sink nodes
                    source_nodes = [dst for dst in step.steps() if (step._preds[dst] is None or step._preds[dst] == [])]
                    sink_nodes = step.find_sink_nodes()                  
                    #Now replace the edges to and from the inner pipeline to to and from source and sink nodes respectively
                    new_edges = step.edges()
                    #list comprehension at the cost of iterating edges thrice
                    new_edges.extend([(node, edge[1]) for edge in edges if edge[0] == step for node in sink_nodes])
                    new_edges.extend([(edge[0], node) for edge in edges if edge[1] == step for node in source_nodes])
                    new_edges.extend([edge for edge in edges if (edge[1] != step and edge[0] != step)])
                    edges = new_edges
                else:
                    self._steps.append(step)
            self._preds = { step: [] for step in self._steps}
            for (src, dst) in edges:
                self._preds[dst].append(src)
            if not ordered:
                self.sort_topologically()
            assert self.is_in_topological_order()

    def constructor_for_cloning(self, steps:List[OpType]):
        edges:List[Tuple[OpType, OpType]] = []
        prev_op:Optional[OpType] = None
        #This is due to scikit base's clone method that needs the same list object
        self._steps = steps
        prev_leaves:List[OpType]
        curr_roots:List[OpType]

        for curr_op in self._steps:
            if isinstance(prev_op, BasePipeline):
                prev_leaves = prev_op.get_leaves()
            else:
                prev_leaves = [] if prev_op is None else [prev_op]
            if isinstance(curr_op, BasePipeline):
                curr_roots = curr_op.get_roots()
                self._steps.extend(curr_op.steps())
                edges.extend(curr_op.edges())
            else:
                curr_roots = [curr_op]
            edges.extend([(src, tgt) for src in prev_leaves for tgt in curr_roots])
            prev_op = curr_op

        seen_steps:List[OpType] = []
        for step in self._steps:
            if step in seen_steps:
                raise ValueError('Same instance of {} already exists in the pipeline. '\
                'This is not allowed.'.format(step.name()))
            seen_steps.append(step)
        self._preds = { step: [] for step in self._steps }
        for (src, dst) in edges:
            self._preds[dst].append(src)
        #Since this case is only allowed for linear pipelines, it is always
        #expected to be in topological order
        assert self.is_in_topological_order()

    def edges(self)->List[Tuple[OpType, OpType]]:
        return [(src, dst) for dst in self._steps for src in self._preds[dst]]

    def is_in_topological_order(self)->bool:
        seen:Dict[OpType, bool] = { }
        for operator in self._steps:
            for pred in self._preds[operator]:
                if pred not in seen:
                    return False
            seen[operator] = True
        return True

    def steps(self)->List[OpType]:
        return self._steps

    def subst_steps(self, m:Dict[OpType,OpType])->None:
        if dict:
            # for i, s in enumerate(self._steps):
            #     self._steps[i] = m.get(s,s)
            self._steps = [m.get(s, s) for s in self._steps]
            self._preds = {m.get(k,k):[m.get(s,s) for s in v] for k,v in self._preds.items()}

    def get_leaves(self)->List[OpType]:
        num_succs:Dict[OpType, int] = { operator: 0 for operator in self._steps }
        for src, _ in self.edges():
            num_succs[src] += 1
        return [op for op in self._steps if num_succs[op] == 0]

    def get_roots(self)->List[OpType]:
        num_preds:Dict[OpType, int] = { op: 0 for op in self._steps }
        for _, tgt in self.edges():
            num_preds[tgt] += 1
        return [op for op in self._steps if num_preds[op] == 0]

    def sort_topologically(self)->None:
        class state(enum.Enum):
            TODO=enum.auto(), 
            DOING=enum.auto(),
            DONE=enum.auto()
        
        states:Dict[OpType, state] = { op: state.TODO for op in self._steps }
        result:List[OpType] = [ ]

        def dfs(operator:OpType)->None:
            if states[operator] is state.DONE:
                return
            if states[operator] is state.DOING:
                raise ValueError('Cycle detected.')
            states[operator] = state.DOING
            for pred in self._preds[operator]:
                dfs(pred)
            states[operator] = state.DONE
            result.append(operator)
        
        for operator in self._steps:
            if states[operator] is state.TODO:
                dfs(operator)
        self._steps = result

    def name(self)->str:            
        return self._name

    def set_name(self, name):
        self._name = name

    def has_same_impl(self, other:Operator)->bool:
        """Checks if the type of the operator imnplementations are compatible
        """
        if not isinstance(other, BasePipeline):
            return False
        my_steps = self.steps()
        other_steps = other.steps()
        if len(my_steps) != len(other_steps):
            return False

        for (m,o) in zip(my_steps, other_steps):
            if not m.has_same_impl(o):
                return False
        return True

    def find_sink_nodes(self) -> List[OpType]:
        is_sink = {s: True for s in self.steps()}
        for src, _ in self.edges():
            is_sink[src] = False
        result = [s for s in self.steps() if is_sink[s]]
        return result

    def find_source_nodes(self) -> List[OpType]:
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
                    'type': 'array',
                    'minItems': n_datasets, 'maxItems': n_datasets,
                    'items': [lale.datasets.data_schemas.to_schema(i)
                              for i in schemas]}
            return result
        outputs = { }
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
            sinks = self.find_sink_nodes()
            pipeline_outputs = [outputs[sink] for sink in sinks]
            return combine_schemas(pipeline_outputs)

    def validate_schema(self, X, y=None):
        self._validate_or_transform_schema(X, y, validate=True)

    def transform_schema(self, s_X):
        return self._validate_or_transform_schema(s_X, validate=False)

    def input_schema_fit(self):
        sources = self.find_source_nodes()
        pipeline_inputs = [source.input_schema_fit() for source in sources]
        result = lale.type_checking.join_schemas(*pipeline_inputs)
        return result

    def is_supervised(self)->bool:
        s = self.steps()
        if len(s) == 0:
            return False
        return self.steps()[-1].is_supervised()

    def remove_last(self, inplace=False):
        sink_nodes = self.find_sink_nodes()
        if len(sink_nodes) > 1:
            raise ValueError("This pipeline has more than 1 sink nodes, can not remove last step meaningfully.")
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

    def get_last(self)->Optional[OpType]:
        sink_nodes = self.find_sink_nodes()
        if len(sink_nodes) > 1:
            return None
        else:
            old_clf = self._steps[-1]
            return old_clf

    def export_to_sklearn_pipeline(self):
        from sklearn.pipeline import make_pipeline
        from sklearn.base import clone
        from lale.lib.lale.concat_features import ConcatFeaturesImpl
        from sklearn.pipeline import FeatureUnion

        def convert_data_with_schemas_to_data(node):
            for element in dir(node):#Looking at only 1 level for now.
                try:
                    value = getattr(node,element)
                    if isinstance(value, lale.datasets.data_schemas.NDArrayWithSchema):
                        modified_value = np.array(value)
                    elif isinstance(value, lale.datasets.data_schemas.DataFrameWithSchema):
                        modified_value = pd.DataFrame(value)
                    elif isinstance(value, lale.datasets.data_schemas.SeriesWithSchema):
                        modified_value = pd.Series(value)
                    else:
                        continue
                    setattr(node, element, modified_value)
                except BaseException:
                    #This is an optional processing, so if there is any exception, continue.
                    #For example, some scikit-learn classes will fail at getattr because they have
                    #that property defined conditionally. 
                    pass

        def create_pipeline_from_sink_node(sink_node):
            #Ensure that the pipeline is either linear or has a "union followed by concat" construct
            #Translate the "union followed by concat" constructs to "featureUnion"
            #Inspect the node and convert any data with schema objects to original data types
            convert_data_with_schemas_to_data(sink_node._impl)
            if sink_node._impl_class() == ConcatFeaturesImpl:
                list_of_transformers = []
                for pred in self._preds[sink_node]:
                    pred_transformer = create_pipeline_from_sink_node(pred)
                    list_of_transformers.append((pred.name()+"_"+str(id(pred)), make_pipeline(*pred_transformer) if isinstance(pred_transformer, list) else pred_transformer))
                return FeatureUnion(list_of_transformers)
            else:
                preds = self._preds[sink_node]                    
                if preds is not None and len(preds) > 1:
                    raise ValueError("A pipeline graph that has operators other than ConcatFeatures with "
                    "multiple incoming edges is not a valid scikit-learn pipeline:{}".format(self.to_json()))
                else:
                    if hasattr(sink_node._impl_instance(), '_sklearn_model'):
                        sklearn_op = sink_node._impl_instance()._sklearn_model
                        convert_data_with_schemas_to_data(sklearn_op)#This case needs one more level of conversion
                    else:
                        sklearn_op = sink_node._impl_instance()
                    sklearn_op = copy.deepcopy(sklearn_op)
                    if preds is None or len(preds) == 0:
                        return sklearn_op
                    else:
                        previous_sklearn_op = create_pipeline_from_sink_node(preds[0])
                        if isinstance(previous_sklearn_op, list):
                            previous_sklearn_op.append(sklearn_op)
                            return previous_sklearn_op
                        else:
                            return [previous_sklearn_op, sklearn_op]

        sklearn_steps_list = []
        #Finding the sink node so that we can do a backward traversal
        sink_nodes = self.find_sink_nodes()
        #For a trained pipeline that is scikit compatible, there should be only one sink node
        if len(sink_nodes) != 1:
            raise ValueError("A pipeline graph that ends with more than one estimator is not a"
            " valid scikit-learn pipeline:{}".format(self.to_json()))
        else:
            sklearn_steps_list = create_pipeline_from_sink_node(sink_nodes[0])
        try:
            sklearn_pipeline = make_pipeline(*sklearn_steps_list) \
                    if isinstance(sklearn_steps_list, list) else make_pipeline(sklearn_steps_list)
        except TypeError:
            raise TypeError("Error creating a scikit-learn pipeline, most likely because the steps are not scikit compatible.")
        return sklearn_pipeline

PlannedOpType = TypeVar('PlannedOpType', bound=PlannedOperator)

class PlannedPipeline(BasePipeline[PlannedOpType], PlannedOperator):
    def __init__(self, 
                 steps:List[PlannedOpType],
                 edges:Optional[Iterable[Tuple[PlannedOpType, PlannedOpType]]], 
                 ordered:bool=False) -> None:
        super(PlannedPipeline, self).__init__(steps, edges, ordered=ordered)

    def configure(self, *args, **kwargs)->'TrainablePipeline':
        """
        Make sure the args i.e. operators form a trainable pipeline and
        return it. It takes only one argument which is a list of steps
        like make_pipeline so, need to check that it is consistent
        with the steps and edges already present
        """
        steps:List[TrainableIndividualOp] = args[0]
        if len(steps) != len(self._steps):
            raise ValueError("Please make sure that you pass a list of trainable individual operators with the same length as the operator instances in the current pipeline")

        num_steps:int = len(steps)
        edges:List[Tuple[PlannedOpType, PlannedOpType]] = self.edges()
        
        op_map:Dict[PlannedOpType, TrainableIndividualOp] = {}
        #for (i, orig_op, op) in enumerate(zip(self.steps(), steps))
        for i in range(num_steps):
            orig_op = self.steps()[i]
            op = steps[i]
            if not isinstance(op, TrainableIndividualOp):
                raise ValueError(f"Please make sure that you pass a list of trainable individual operators ({i}th element is incompatible)")
            if not orig_op.has_same_impl(op):
                raise ValueError(f"Please make sure that you pass a list of trainable individual operators with the same type as operator instances in the current pipeline ({i}th element is incompatible)")

            op_map[orig_op] = op

        trainable_edges:List[Tuple[TrainableIndividualOp, TrainableIndividualOp]]
        try:
            trainable_edges = [(op_map[x], op_map[y]) for (x, y) in edges]
        except KeyError as e:
            raise ValueError("An edge was found with an endpoint that is not a step (" + str(e) + ")")
        
        return TrainablePipeline(steps, trainable_edges, ordered=self.is_in_topological_order())

    def __call__(self, *args, **kwargs):
        self.configure(args, kwargs)
    

TrainableOpType = TypeVar('TrainableOpType', bound=TrainableIndividualOp)

class TrainablePipeline(PlannedPipeline[TrainableOpType], TrainableOperator):

    def __init__(self, 
                 steps:List[TrainableOpType],
                 edges:Optional[Iterable[Tuple[TrainableOpType, TrainableOpType]]], 
                 ordered:bool=False) -> None:
        super(TrainablePipeline, self).__init__(steps, edges, ordered=ordered)

    def fit(self, X, y=None, **fit_params)->'TrainedPipeline':
        X = lale.datasets.data_schemas.add_schema(X)
        y = lale.datasets.data_schemas.add_schema(y)
        self.validate_schema(X, y)
        trained_steps:List[TrainedOperator] = [ ]
        outputs:Dict[Operator, Any] = { }
        meta_outputs:Dict[Operator, Any] = {}
        edges:List[Tuple[TrainableOpType, TrainableOpType]] = self.edges()
        trained_map:Dict[TrainableOpType, TrainedOperator] = {}

        sink_nodes = self.find_sink_nodes()
        for operator in self._steps:
            preds = self._preds[operator]
            if len(preds) == 0:
                inputs = [X]
                meta_data_inputs:Dict[Operator, Any] = {}
            else:
                inputs = [outputs[pred][0] if isinstance(outputs[pred], tuple) else outputs[pred] for pred in preds]
                #we create meta_data_inputs as a dictionary with metadata from all previous steps
                #Note that if multiple previous steps generate the same key, it will retain only one of those.
                
                meta_data_inputs = {key: meta_outputs[pred][key] for pred in preds 
                        if meta_outputs[pred] is not None for key in meta_outputs[pred]}
            trainable = operator
            if len(inputs) == 1:
                inputs = inputs[0]
            if hasattr(operator._impl, "set_meta_data"):
                operator._impl_instance().set_meta_data(meta_data_inputs)
            meta_output:Dict[Operator, Any] = {}
            trained:TrainedOperator
            if trainable.is_supervised():
                trained = trainable.fit(X = inputs, y = y)
            else:
                trained = trainable.fit(X = inputs)
            trained_map[operator] = trained
            trained_steps.append(trained)
            if trainable not in sink_nodes:#There is no need to transform/predict on the last node during fit
                if trained.is_transformer():
                    output = trained.transform(X = inputs, y = y)
                    if hasattr(operator._impl, "get_transform_meta_output"):
                        meta_output = operator._impl_instance().get_transform_meta_output()
                else:
                    if trainable in sink_nodes:
                        output = trained.predict(X = inputs) #We don't support y for predict yet as there is no compelling case
                    else:
                        # This is ok because trainable pipelines steps
                        # must only be individual operators
                        if hasattr(trained._impl, 'predict_proba'): # type: ignore
                            output = trained.predict_proba(X = inputs)
                        else:
                            output = trained.predict(X = inputs)
                    if hasattr(operator._impl, "get_predict_meta_output"):
                        meta_output = operator._impl_instance().get_predict_meta_output()
                outputs[operator] = output
                meta_output.update({key:meta_outputs[pred][key] for pred in preds 
                        if meta_outputs[pred] is not None for key in meta_outputs[pred]})
                meta_outputs[operator] = meta_output

        trained_edges = [(trained_map[x], trained_map[y]) for (x, y) in edges]

        trained_steps2:Any = trained_steps
        result:TrainedPipeline = TrainedPipeline(trained_steps2, trained_edges, ordered=True)
        self.__trained = result
        return result

    def predict(self, X):
        """
        .. deprecated:: 0.0.0
           The `predict` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `predict`
           on the trained operator returned by `fit` instead.
        """
        warnings.warn(_mutation_warning('predict'), DeprecationWarning)
        try:
            return self.__trained.predict(X)
        except AttributeError:
            raise ValueError('Must call `fit` before `predict`.')

    def transform(self, X, y = None):
        """
        .. deprecated:: 0.0.0
           The `transform` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `transform`
           on the trained operator returned by `fit` instead.
        """
        warnings.warn(_mutation_warning('transform'), DeprecationWarning)
        try:
            return self.__trained.transform(X, y = None)
        except AttributeError:
            raise ValueError('Must call `fit` before `transform`.')

    def predict_proba(self, X):
        """
        .. deprecated:: 0.0.0
           The `predict_proba` method is deprecated on a trainable
           operator, because the learned coefficients could be
           accidentally overwritten by retraining. Call `predict_proba`
           on the trained operator returned by `fit` instead.
        """
        warnings.warn(_mutation_warning('predict_proba'), DeprecationWarning)
        try:
            return self.__trained.predict_proba(X)
        except AttributeError:
            raise ValueError('Must call `fit` before `predict_proba`.')

    def is_frozen_trainable(self)->bool:
        for step in self.steps():
            if not step.is_frozen_trainable():
                return False
        return True

    def freeze_trainable(self)->'TrainablePipeline':
        frozen_steps = []
        frozen_map = {}
        for liquid in self._steps:
            frozen = liquid.freeze_trainable()
            frozen_map[liquid] = frozen
            frozen_steps.append(frozen)
        frozen_edges = [(frozen_map[x], frozen_map[y]) for x, y in self.edges()]
        result = cast(TrainablePipeline, get_pipeline_of_applicable_type(
            frozen_steps, frozen_edges, ordered=True))
        assert result.is_frozen_trainable()
        return result

    def get_params(self, deep:bool = True)->Dict[str,Any]:
        out:Dict[str,Any] = {}
        if not deep:
            out['steps'] = self._steps
            out['edges'] = None
            out['ordered'] = True
        else:
            pass #TODO
        return out

    @classmethod
    def import_from_sklearn_pipeline(cls, sklearn_pipeline):
        #For all pipeline steps, identify equivalent lale wrappers if present,
        #if not, call make operator on sklearn classes and create a lale pipeline.

        def get_equivalent_lale_op(sklearn_obj):
            module_name = "lale.lib.sklearn"
            from sklearn.base import clone

            class_name = sklearn_obj.__class__.__name__
            module = importlib.import_module(module_name)
            try:
                class_ = getattr(module, class_name)
            except AttributeError:
                class_ = make_operator(sklearn_obj.__class__, name=class_name)
            class_ = class_(**sklearn_obj.get_params())
            class_._impl_instance()._sklearn_model =  clone(sklearn_obj)
            return class_         

        from sklearn.pipeline import FeatureUnion, Pipeline
        from sklearn.base import BaseEstimator
        if isinstance(sklearn_pipeline, Pipeline):
            nested_pipeline_steps = sklearn_pipeline.named_steps.values()
            nested_pipeline_lale_objects = [cls.import_from_sklearn_pipeline(nested_pipeline_step) for nested_pipeline_step in nested_pipeline_steps]
            lale_op_obj = make_pipeline(*nested_pipeline_lale_objects)
        elif isinstance(sklearn_pipeline, FeatureUnion):
            transformer_list = sklearn_pipeline.transformer_list
            concat_predecessors = [cls.import_from_sklearn_pipeline(transformer[1]) for transformer in transformer_list]
            lale_op_obj = make_union(*concat_predecessors)
        else:
            lale_op_obj = get_equivalent_lale_op(sklearn_pipeline)
        return lale_op_obj
    
    def fit_with_batches(self, X, y=None, serialize=True):
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
        trained_steps:List[TrainedOperator] = [ ]
        outputs:Dict[Operator, Any] = { }
        edges:List[Tuple[TrainableOpType, TrainableOpType]] = self.edges()
        trained_map:Dict[TrainableOpType, TrainedOperator] = {}

        if serialize:
            serialization_out_dir = os.path.join(os.path.dirname(__file__), 'temp_serialized')
            if not os.path.exists(serialization_out_dir):
                os.mkdir(serialization_out_dir)

        sink_nodes = self.find_sink_nodes()
        operator_idx = 0
        for operator in self._steps:
            preds = self._preds[operator]
            if len(preds) == 0:
                inputs = [X]
            else:
                inputs = [outputs[pred][0] if isinstance(outputs[pred], tuple) else outputs[pred] for pred in preds]
            trainable = operator
            if len(inputs) == 1:
                inputs = inputs[0]
            trained:TrainedOperator
            if hasattr(trainable._impl, "partial_fit"):
                try:
                    num_epochs = trainable._impl_instance().num_epochs
                except AttributeError:
                    warnings.warn("Operator {} does not have num_epochs, using 1 as a default".format(trainable.name()))
                    num_epochs = 1
            else:
                raise AttributeError("All operators to be trained with batching need to implement partial_fit. {} doesn't.".format(operator.name()))
            inputs_for_transform = inputs
            for epoch in range(num_epochs):
                training_loss = 0
                nb_tr_examples, nb_tr_steps = 0, 0  
                for _, batch_data in enumerate(inputs):#batching_transformer will output only one obj
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
                            loss = trainable.partial_fit(batch_X, batch_y, classes = y)
                        except TypeError:
                            loss = trainable.partial_fit(batch_X, batch_y)
                    else:
                        loss = trainable.partial_fit(batch_X)
                    if type(loss) == float:#This is very brittle, but it is a practice to report loss per epoch or 
                        #per few epochs in DL, how to achieve it better?
                        training_loss +=loss
                        nb_tr_examples += len(batch_data)
                        nb_tr_steps += 1
                        print("Train loss of {} at epoch {}: {}".format(trainable.name(), epoch, (training_loss/nb_tr_steps)))
            try:
                trained = copy.deepcopy(trainable)
            except BaseException: #Deepcopy fails for models implemented in tensorflow/keras.
                trained = trainable
            trained = TrainedIndividualOp(trained.name(), trained._impl, trained._schemas)                
            trained_map[operator] = trained
            trained_steps.append(trained)

            output = None
            for batch_idx, batch_data in enumerate(inputs_for_transform):#batching_transformer will output only one obj
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
                        batch_output = trained.predict(X = batch_X) #We don't support y for predict yet as there is no compelling case
                    else:
                        # This is ok because trainable pipelines steps
                        # must only be individual operators
                        if hasattr(trained._impl, 'predict_proba'): # type: ignore
                            batch_output = trained.predict_proba(X = batch_X)
                        else:
                            batch_output = trained.predict(X = batch_X)
                if isinstance(batch_output, tuple):
                    batch_out_X, batch_out_y = batch_output
                else:
                    batch_out_X = batch_output
                    batch_out_y = None
                if serialize:
                    output = lale.helpers.write_batch_output_to_file(output, os.path.join(serialization_out_dir, 'fit_with_batches'+str(operator_idx)+'.hdf5'), 
                                len(inputs_for_transform.dataset), batch_idx, batch_X, batch_y, batch_out_X, batch_out_y)                            
                else:
                    if batch_out_y is None:
                        output = lale.helpers.append_batch(output, (batch_output, batch_y)) 
                    else:
                        output = lale.helpers.append_batch(output, batch_output) 
            if serialize:
                output.close()
                output = lale.helpers.create_data_loader(os.path.join(serialization_out_dir, 'fit_with_batches'+str(operator_idx)+'.hdf5'), batch_size=inputs_for_transform.batch_size)
            else: 
                if isinstance(output, tuple):
                    output = lale.helpers.create_data_loader(X = output[0], y=output[1], batch_size=inputs_for_transform.batch_size)
                else:
                    output = lale.helpers.create_data_loader(X = output, y = None, batch_size=inputs_for_transform.batch_size)
            outputs[operator] = output
            operator_idx += 1

        if serialize:
            shutil.rmtree(serialization_out_dir)
        trained_edges = [(trained_map[x], trained_map[y]) for (x, y) in edges]

        trained_steps2:Any = trained_steps
        result:TrainedPipeline = TrainedPipeline(trained_steps2, trained_edges, ordered=True)
        self.__trained = result
        return result

TrainedOpType = TypeVar('TrainedOpType', bound=TrainedIndividualOp)

class TrainedPipeline(TrainablePipeline[TrainedOpType], TrainedOperator):

    def __init__(self, 
                 steps:List[TrainedOpType],
                 edges:List[Tuple[TrainedOpType, TrainedOpType]], 
                 ordered:bool=False) -> None:
        super(TrainedPipeline, self).__init__(steps, edges, ordered=ordered)


    def predict(self, X, y = None):
        outputs = { }
        meta_outputs = {}
        sink_nodes = self.find_sink_nodes()
        for operator in self._steps:
            preds = self._preds[operator]
            if len(preds) == 0:
                inputs = [X]
                meta_data_inputs = {}
            else:
                inputs = [outputs[pred][0] if isinstance(outputs[pred], tuple) else outputs[pred] for pred in preds]
                #we create meta_data_inputs as a dictionary with metadata from all previous steps
                #Note that if multiple previous steps generate the same key, it will retain only one of those.
                
                meta_data_inputs = {key: meta_outputs[pred][key] for pred in preds 
                        if meta_outputs[pred] is not None for key in meta_outputs[pred]}
            if len(inputs) == 1:
                inputs = inputs[0]
            if hasattr(operator._impl, "set_meta_data"):
                operator._impl_instance().set_meta_data(meta_data_inputs)
            meta_output = {}
            if operator in sink_nodes and hasattr(operator._impl, 'predict'):#Since this is pipeline's predict, we should invoke predict from sink nodes
                output = operator.predict(X = inputs)
            elif operator.is_transformer():
                output = operator.transform(X = inputs, y = y)
                if hasattr(operator._impl, "get_transform_meta_output"):
                    meta_output = operator._impl_instance().get_transform_meta_output()
            elif hasattr(operator._impl, 'predict_proba'):#For estimator as a transformer, use predict_proba if available
                output = operator.predict_proba(X = inputs)
            else:
                output = operator.predict(X = inputs)
                if hasattr(operator._impl, "get_predict_meta_output"):
                    meta_output = operator._impl_instance().get_predict_meta_output()
            outputs[operator] = output
            meta_output.update({key:meta_outputs[pred][key] for pred in preds 
                    if meta_outputs[pred] is not None for key in meta_outputs[pred]})
            meta_outputs[operator] = meta_output
        return outputs[self._steps[-1]]

    def transform(self, X, y = None):
        #TODO: What does a transform on a pipeline mean, if the last step is not a transformer
        #can it be just the output of predict of the last step?
        # If this implementation changes, check to make sure that the implementation of 
        # self.is_transformer is kept in sync with the new assumptions.
        return self.predict(X, y)

    def predict_proba(self, X):
        outputs = { }
        sink_nodes = self.find_sink_nodes()
        for operator in self._steps:
            preds = self._preds[operator]
            if len(preds) == 0:
                inputs = [X]
            else:
                inputs = [outputs[pred][0] if isinstance(outputs[pred], tuple) else outputs[pred] for pred in preds]
            if len(inputs) == 1:
                inputs = inputs[0]
            if operator.is_transformer():
                output = operator.transform(X = inputs)
            else:
                if operator in sink_nodes:
                    if hasattr(operator._impl, 'predict_proba'):
                        output = operator.predict_proba(X = inputs)
                    else:
                        raise ValueError("The sink node of the pipeline {} does not support a predict_proba method.".format(operator.name()))
                else:#this behavior may be different later if we add user input.
                    if hasattr(operator._impl, 'predict_proba'):
                        output = operator.predict_proba(X = inputs)
                    else:
                        output = operator.predict(X = inputs)
            outputs[operator] = output
        return outputs[self._steps[-1]]

    def transform_with_batches(self, X, y=None, serialize = True):
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
        outputs = { }

        if serialize:
            serialization_out_dir = os.path.join(os.path.dirname(__file__), 'temp_serialized')
            if not os.path.exists(serialization_out_dir):
                os.mkdir(serialization_out_dir)

        sink_nodes = self.find_sink_nodes()
        operator_idx = 0
        for operator in self._steps:
            preds = self._preds[operator]
            if len(preds) == 0:
                inputs = [X]
            else:
                inputs = [outputs[pred][0] if isinstance(outputs[pred], tuple) else outputs[pred] for pred in preds]
            if len(inputs) == 1:
                inputs = inputs[0]
            trained = operator
            output = None
            for batch_idx, batch_data in enumerate(inputs):#batching_transformer will output only one obj
                if isinstance(batch_data, Tuple):
                    batch_X, batch_y = batch_data
                else:
                    batch_X = batch_data
                    batch_y = None
                if trained.is_transformer():
                    batch_output = trained.transform(batch_X, batch_y)
                else:
                    if trained in sink_nodes:
                        batch_output = trained.predict(X = batch_X) #We don't support y for predict yet as there is no compelling case
                    else:
                        # This is ok because trainable pipelines steps
                        # must only be individual operators
                        if hasattr(trained._impl, 'predict_proba'): # type: ignore
                            batch_output = trained.predict_proba(X = batch_X)
                        else:
                            batch_output = trained.predict(X = batch_X)
                if isinstance(batch_output, tuple):
                    batch_out_X, batch_out_y = batch_output
                else:
                    batch_out_X = batch_output
                    batch_out_y = None
                if serialize:
                    output = lale.helpers.write_batch_output_to_file(output, os.path.join(serialization_out_dir, 'fit_with_batches'+str(operator_idx)+'.hdf5'), 
                        len(inputs.dataset), batch_idx, batch_X, batch_y, batch_out_X, batch_out_y)
                else:
                    if batch_out_y is not None:
                        output = lale.helpers.append_batch(output, (batch_output, batch_out_y)) 
                    else:
                        output = lale.helpers.append_batch(output, batch_output)
            if serialize:
                output.close()
                output = lale.helpers.create_data_loader(os.path.join(serialization_out_dir, 'fit_with_batches'+str(operator_idx)+'.hdf5'), batch_size=inputs.batch_size)
            else: 
                if isinstance(output, tuple):
                    output = lale.helpers.create_data_loader(X = output[0], y=output[1], batch_size=inputs.batch_size)
                else:
                    output = lale.helpers.create_data_loader(X = output, y = None, batch_size=inputs.batch_size)            
            outputs[operator] = output
            operator_idx += 1

        return_data = outputs[self._steps[-1]].dataset.get_data()
        if serialize: 
            shutil.rmtree(serialization_out_dir)
            
        return return_data

    def freeze_trainable(self)->'TrainedPipeline':
        result = super(TrainedPipeline, self).freeze_trainable()
        return cast(TrainedPipeline, result)

    def is_frozen_trained(self)->bool:
        for step in self.steps():
            if not step.is_frozen_trained():
                return False
        return True

    def freeze_trained(self)->'TrainedPipeline':
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

    def _lale_clone(self, cloner:Callable[[Any],Any]):
        """ This is really used for sklearn clone compatibility.
            Which mandates that clone returns something that has not been fit.
            So we enforce that here as well.
        """
        op = super()._lale_clone(cloner)
        return TrainedPipeline(op._steps, op._preds, True)

    def is_transformer(self)->bool:
        """ Checks if the operator is a transformer
        """
        # Currently, all TrainedPipelines implement transform
        return True

OperatorChoiceType = TypeVar('OperatorChoiceType', bound=Operator)
class OperatorChoice(PlannedOperator, Generic[OperatorChoiceType]):
    _name:str
    _steps:List[OperatorChoiceType]

    def __init__(self, steps, name:Optional[str]) -> None:
        #name is not optional as we assume that only make_choice calls this constructor and
        #it will always pass a name, and in case it is passed as None, we assign name as below:
        if name is None or name == '':
            name = lale.helpers.assignee_name(level=2)
        if name is None or name == '':
            name = 'OperatorChoice'

        self._name = name
        self._steps = steps.copy()

    def steps(self)->List[OperatorChoiceType]:
        return self._steps

    def name(self)->str:
        return self._name

    def set_name(self, name):
        self._name = name

    def _lale_clone(self, cloner:Callable[[Any], Any]):
        steps = self._steps
        new_steps:List[OperatorChoiceType] = [s._lale_clone(cloner) for s in steps]
        return self.__class__(new_steps, self._name)
       
    def configure(self, *args, **kwargs):
        return self.__call__(args, kwargs)

    def __call__(self, *args, **kwargs):
        operator = args[0]
        #TODO: schemas = None below is not correct. What are the right schemas to use?
        return TrainableIndividualOp(name=operator.name(), impl=operator, schemas = None)

    def has_same_impl(self, other:Operator)->bool:
        """Checks if the type of the operator imnplementations are compatible
        """
        if not isinstance(other, OperatorChoice):
            return False
        my_steps = self.steps()
        other_steps = other.steps()
        if len(my_steps) != len(other_steps):
            return False

        for (m,o) in zip(my_steps, other_steps):
            if not m.has_same_impl(o):
                return False
        return True

    def is_supervised(self)->bool:
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

    def input_schema_fit(self):
        pipeline_inputs = [s.input_schema_fit() for s in self.steps()]
        result = lale.type_checking.join_schemas(*pipeline_inputs)
        return result

class PipelineFactory():
    def __init__(self):
        pass

    def __call__(self, steps:List[Any]):
        for i in range(len(steps)):
            op = steps[i]
            if isinstance(op, tuple):
                assert isinstance(op[1], Operator)
                op[1].set_name(op[0])
                steps[i] = op[1]
        return make_pipeline(*steps)

Pipeline = PipelineFactory()

def get_pipeline_of_applicable_type(steps, edges, ordered=False)->PlannedPipeline:
    """
    Based on the state of the steps, it is important to decide an appropriate type for
    a new Pipeline. This method will decide the type, create a new Pipeline of that type and return it.
    #TODO: If multiple independently trained components are composed together in a pipeline,
    should it be of type TrainedPipeline?
    Currently, it will be TrainablePipeline, i.e. it will be forced to train it again.
    """

    isTrainable:bool = True
    isTrained:bool = True
    for operator in steps:
        if not isinstance(operator, TrainedOperator):
            isTrained = False #Even if a single step is not trained, the pipeline can't be used for predict/transform 
            # without training it first
        if isinstance(operator, OperatorChoice) or not isinstance(operator, TrainableOperator):
            isTrainable = False
    if isTrained:
        return TrainedPipeline(steps, edges, ordered=ordered)
    elif isTrainable:
        return TrainablePipeline(steps, edges, ordered=ordered)
    else:
        return PlannedPipeline(steps, edges, ordered=ordered)

def make_pipeline(*orig_steps:Union[Operator,Any])->PlannedPipeline:
    steps, edges = [], []
    prev_op = None
    for curr_op in orig_steps:
        if isinstance(prev_op, BasePipeline):
            prev_leaves: Any = prev_op.get_leaves()
        else:
            prev_leaves = [] if prev_op is None else [prev_op]
        if isinstance(curr_op, BasePipeline):
            curr_roots = curr_op.get_roots()
            steps.extend(curr_op.steps())
            edges.extend(curr_op.edges())
        else:
            if not isinstance(curr_op, Operator):
                curr_op = make_operator(curr_op, name = curr_op.__class__.__name__)
            curr_roots = [curr_op]
            steps.append(curr_op)
        edges.extend([(src, tgt) for src in prev_leaves for tgt in curr_roots])
        prev_op = curr_op
    return get_pipeline_of_applicable_type(steps, edges, ordered=True)

def make_union_no_concat(*orig_steps:Union[Operator,Any])->Operator:
    steps, edges = [], []
    for curr_op in orig_steps:
        if isinstance(curr_op, BasePipeline):
            steps.extend(curr_op._steps)
            edges.extend(curr_op.edges())
        else:
            if not isinstance(curr_op, Operator):
                curr_op = make_operator(curr_op, name = curr_op.__class__.__name__)
            steps.append(curr_op)
    return get_pipeline_of_applicable_type(steps, edges, ordered=True)

def make_union(*orig_steps:Union[Operator,Any])->Operator:
    from lale.lib.lale import ConcatFeatures
    return make_union_no_concat(*orig_steps) >> ConcatFeatures()

def make_choice(*orig_steps:Union[Operator,Any], name:Optional[str]=None)->OperatorChoice:
    if name is None:
        name = ""
    name_:str = name # to make mypy happy
    steps:List[Operator] = [ ]
    for operator in orig_steps:
        if isinstance(operator, OperatorChoice):
            steps.extend(operator.steps())
        else:
            if not isinstance(operator, Operator):
                operator = make_operator(operator, name = operator.__class__.__name__)
            steps.append(operator)
        name_ = name_ + " | " + operator.name()
    return OperatorChoice(steps, name_[3:])

def fixup_hyperparams_dict(d):
    d1 = remove_defaults_dict(d)
    d2 = {k:lale.helpers.val_wrapper.unwrap(v) for k,v in d1.items()}
    return d2

