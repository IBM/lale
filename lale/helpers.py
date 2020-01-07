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

import ast
import jsonschema
import jsonsubschema
import numpy as np
import pandas as pd
import os
import re
import sys
import time
import traceback
import warnings
import yaml
import scipy.sparse
import importlib
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics.scorer import check_scoring
from sklearn.utils.metaestimators import _safe_split
from lale.util.numpy_to_torch_dataset import NumpyTorchDataset
from lale.util.hdf5_to_torch_dataset import HDF5TorchDataset
from torch.utils.data import DataLoader, TensorDataset
import copy
import logging
import inspect
import pkgutil
import torch
import h5py
from typing import Any, Dict, List, Optional, Union
import lale.datasets.data_schemas
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class assert_raises:
    def __init__(self, expected_exc_type):
        self.__expected_exc_type = expected_exc_type

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        assert exc_type is self.__expected_exc_type
        print_yaml('error', str(exc_value), file=sys.stderr)
        return True

class assert_warns:
    def __init__(self, expected_exc_type):
        self.__expected_exc_type = expected_exc_type

    def __enter__(self):
        warnings.filterwarnings("error", category=self.__expected_exc_type)

    def __exit__(self, exc_type, exc_value, traceback):
        warnings.resetwarnings()
        assert exc_type is self.__expected_exc_type
        print_yaml('error', str(exc_value), file=sys.stderr)
        return True

def assignee_name(level=1) -> Optional[str]:
    tb = traceback.extract_stack()
    file_name, line_number, function_name, text = tb[-(level+2)]
    tree = ast.parse(text, file_name)
    assert isinstance(tree, ast.Module)
    if len(tree.body) == 1 and isinstance(tree.body[0], ast.Assign):
        lhs = tree.body[0].targets
        if len(lhs) == 1 and isinstance(lhs[0], ast.Name):
            return lhs[0].id
    return None

def data_to_json(data, subsample_array:bool=True) -> Union[list, dict]:
    if type(data) is tuple:
        # convert to list
        return [data_to_json(elem, subsample_array) for elem in data]
    if type(data) is list:
        return [data_to_json(elem, subsample_array) for elem in data]
    elif type(data) is dict:
        return {key: data_to_json(data[key], subsample_array) for key in data}
    elif isinstance(data, np.ndarray):
        return ndarray_to_json(data, subsample_array)
    elif type(data) is scipy.sparse.csr_matrix:
        return ndarray_to_json(data.toarray(), subsample_array)
    elif isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
        np_array = data.values
        return ndarray_to_json(np_array, subsample_array)
    elif isinstance(data, torch.Tensor):
        np_array = data.detach().numpy()
        return ndarray_to_json(np_array, subsample_array)
    else:
        return data

def dict_without(orig_dict: Dict[str, Any], key: str) -> Dict[str, Any]:
    return {k: orig_dict[k] for k in orig_dict if k != key}

def ndarray_to_json(arr, subsample_array:bool=True) -> Union[list, dict]:
    #sample 10 rows and no limit on columns
    if subsample_array:
        num_subsamples = [10, np.iinfo(np.int).max, np.iinfo(np.int).max]
    else:
        num_subsamples = [np.iinfo(np.int).max, np.iinfo(np.int).max, np.iinfo(np.int).max]
    def subarray_to_json(indices):
        if len(indices) == len(arr.shape):
            if isinstance(arr[indices], bool) or \
               isinstance(arr[indices], int) or \
               isinstance(arr[indices], float) or \
               isinstance(arr[indices], str):
                return arr[indices]
            elif np.issubdtype(arr.dtype, np.bool_):
                return bool(arr[indices])
            elif np.issubdtype(arr.dtype, np.integer):
                return int(arr[indices])
            elif np.issubdtype(arr.dtype, np.number):
                return float(arr[indices])
            elif arr.dtype.kind in ['U', 'S', 'O']:
                return str(arr[indices])
            else:
                raise ValueError(f'Unexpected dtype {arr.dtype}, '
                                 f'kind {arr.dtype.kind}, '
                                 f'type {type(arr[indices])}.')
        else:
            assert len(indices) < len(arr.shape)
            return [subarray_to_json(indices + (i,))
                    for i in range(min(num_subsamples[len(indices)], arr.shape[len(indices)]))]
    return subarray_to_json(())

def print_yaml(what: str, doc: Union[list, dict], file=sys.stdout):
    print(yaml.dump({what: doc}).strip(), file=file)

def validate_schema(value, schema: Dict[str, Any], subsample_array:bool=True):
    json_value = data_to_json(value, subsample_array)
    jsonschema.validate(json_value, schema)

def validate_schema_or_subschema(part, super_schema):
    if is_schema(part):
        sub_schema = part
    else:
        try:
            sub_schema = lale.datasets.data_schemas.to_schema(part)
        except ValueError as e:
            sub_schema = None
    if sub_schema is None:
        validate_schema(part, super_schema)
    else:
        validate_subschema(sub_schema, super_schema)

JSON_META_SCHEMA_URL = 'http://json-schema.org/draft-04/schema#'

def json_meta_schema() -> Dict[str, Any]:
    return jsonschema.Draft4Validator.META_SCHEMA

def validate_is_schema(value: Dict[str, Any]):
    if '$schema' in value:
        assert value['$schema'] == JSON_META_SCHEMA_URL
    jsonschema.validate(value, json_meta_schema())

def is_schema(value) -> bool:
    if isinstance(value, dict):
        try:
            jsonschema.validate(value, json_meta_schema())
        except:
            return False
        return True
    return False

class SubschemaError(Exception):
    def __init__(self, sub, sup, sub_name='sub', sup_name='super'):
        self.sub = sub
        self.sup = sup
        self.sub_name = sub_name
        self.sup_name = sup_name

    def __str__(self):
        summary = f'Expected {self.sub_name} to be a subschema of {self.sup_name}.'
        import lale.pretty_print
        sub = lale.pretty_print.schema_to_string(self.sub)
        sup = lale.pretty_print.schema_to_string(self.sup)
        details = f'\n{self.sub_name} = {sub}\n{self.sup_name} = {sup}'
        return summary + details

def validate_subschema(sub, sup, sub_name='sub', sup_name='super'):
    if not jsonsubschema.isSubschema(sub, sup):
        raise SubschemaError(sub, sup, sub_name, sup_name)

def split_with_schemas(estimator, all_X, all_y, indices, train_indices=None):
    subset_X, subset_y = _safe_split(
        estimator, all_X, all_y, indices, train_indices)
    if hasattr(all_X, 'json_schema'):
        n_rows = subset_X.shape[0]
        schema = {
            'type': 'array', 'minItems': n_rows, 'maxItems': n_rows,
            'items': all_X.json_schema['items']}
        lale.datasets.data_schemas.add_schema(subset_X, schema)
    if hasattr(all_y, 'json_schema'):
        n_rows = subset_y.shape[0]
        schema = {
            'type': 'array', 'minItems': n_rows, 'maxItems': n_rows,
            'items': all_y.json_schema['items']}
        lale.datasets.data_schemas.add_schema(subset_y, schema)
    return subset_X, subset_y

def cross_val_score_track_trials(estimator, X, y=None, scoring=accuracy_score, cv=5):
    """
    Use the given estimator to perform fit and predict for splits defined by 'cv' and compute the given score on 
    each of the splits.

    Parameters
    ----------

    estimator: A valid sklearn_wrapper estimator
    X, y: Valid data and target values that work with the estimator
    scoring: string or a scorer object created using 
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.make_scorer.html#sklearn.metrics.make_scorer.
        A string from sklearn.metrics.SCORERS.keys() can be used or a scorer created from one of 
        sklearn.metrics (https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics).
        A completely custom scorer object can be created from a python function following the example at 
        https://scikit-learn.org/stable/modules/model_evaluation.html
        The metric has to return a scalar value,
    cv: an integer or an object that has a split function as a generator yielding (train, test) splits as arrays of indices.
        Integer value is used as number of folds in sklearn.model_selection.StratifiedKFold, default is 5.
        Note that any of the iterators from https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators can be used here.

    Returns
    -------
        cv_results: a list of scores corresponding to each cross validation fold
    """
    if isinstance(cv, int):
        cv = StratifiedKFold(cv)

    scorer = check_scoring(estimator, scoring=scoring)
    cv_results:List[float] = []
    log_loss_results = []
    time_results = []
    for train, test in cv.split(X, y):
        X_train, y_train = split_with_schemas(estimator, X, y, train)
        X_test, y_test = split_with_schemas(estimator, X, y, test, train)
        start = time.time()
        trained = estimator.fit(X_train, y_train)
        score_value  = scorer(trained, X_test, y_test)
        execution_time = time.time() - start
        # not all estimators have predict probability
        try:
            y_pred_proba = trained.predict_proba(X_test)
            logloss = log_loss(y_true=y_test, y_pred=y_pred_proba)
            log_loss_results.append(logloss)
        except BaseException:
            logger.debug("Warning, log loss cannot be computed")
        cv_results.append(score_value)
        time_results.append(execution_time)
    return np.array(cv_results).mean(), np.array(log_loss_results).mean(), np.array(execution_time).mean()


def cross_val_score(estimator, X, y=None, scoring=accuracy_score, cv=5):
    """
    Use the given estimator to perform fit and predict for splits defined by 'cv' and compute the given score on
    each of the splits.
    :param estimator: A valid sklearn_wrapper estimator
    :param X, y: Valid data and target values that work with the estimator
    :param scoring: a scorer object from sklearn.metrics (https://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics)
             Default value is accuracy_score.
    :param cv: an integer or an object that has a split function as a generator yielding (train, test) splits as arrays of indices.
        Integer value is used as number of folds in sklearn.model_selection.StratifiedKFold, default is 5.
        Note that any of the iterators from https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators can be used here.
    :return: cv_results: a list of scores corresponding to each cross validation fold
    """
    if isinstance(cv, int):
        cv = StratifiedKFold(cv)

    cv_results = []
    for train, test in cv.split(X, y):
        X_train, y_train = split_with_schemas(estimator, X, y, train)
        X_test, y_test = split_with_schemas(estimator, X, y, test, train)
        trained_estimator = estimator.fit(X_train, y_train)
        predicted_values = trained_estimator.predict(X_test)
        cv_results.append(scoring(y_test, predicted_values))

    return cv_results

def create_operator_using_reflection(class_name, operator_name, param_dict):
    instance = None
    if class_name is not None:
        class_name_parts = class_name.split(".")
        assert(len(class_name_parts)) >1, "The class name needs to be fully qualified, i.e. module name + class name"
        module_name = ".".join(class_name_parts[0:-1])
        class_name = class_name_parts[-1]

        module = importlib.import_module(module_name)

        class_ = getattr(module, class_name)
        
        if param_dict is None:
            instance = class_.create(name=operator_name)
        else:
            instance = class_.create(name=operator_name, kwargs = param_dict)
    return instance

def create_individual_op_using_reflection(class_name, operator_name, param_dict):
    instance = None
    if class_name is not None:
        class_name_parts = class_name.split(".")
        assert(len(class_name_parts)) >1, "The class name needs to be fully qualified, i.e. module name + class name"
        module_name = ".".join(class_name_parts[0:-1])
        class_name = class_name_parts[-1]

        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)
        
        if param_dict is None:
            instance = class_()
        else:
            instance = class_(**param_dict)
    return instance

def to_graphviz(lale_operator, **dot_graph_attr):
    import lale.visualize
    if not isinstance(lale_operator, lale.operators.Operator):
        raise ValueError("The input to to_graphviz needs to be a valid LALE operator.")
    jsn = lale.json_operator.to_json(lale_operator, call_depth=2)
    dot = lale.visualize.json_to_graphviz(jsn, dot_graph_attr)
    return dot

def println_pos(message, out_file=sys.stdout):
    tb = traceback.extract_stack()[-2]
    match = re.search(r'<ipython-input-([0-9]+)-', tb[0])
    if match:
        pos = 'notebook cell [{}] line {}'.format(match[1], tb[1])
    else:
        pos = '{}:{}'.format(tb[0], tb[1])
    strtime = time.strftime('%Y-%m-%d_%H-%M-%S')
    to_log = '{}: {} {}'.format(pos, strtime, message)
    print(to_log, file=out_file)
    if match:
        os.system('echo {}'.format(to_log))

def create_instance_from_hyperopt_search_space(lale_object, hyperparams):
    '''
    Hyperparams is a n-tuple of dictionaries of hyper-parameters, each
    dictionary corresponds to an operator in the pipeline
    '''
    #lale_object can either be an individual operator, a pipeline or an operatorchoice
    #Validate that the number of elements in the n-tuple is the same
    #as the number of steps in the current pipeline

    from lale.operators import IndividualOp
    from lale.operators import BasePipeline
    from lale.operators import TrainablePipeline
    from lale.operators import OperatorChoice
    if isinstance(lale_object, IndividualOp):
        return lale_object(**dict_without(hyperparams, 'name'))
    elif isinstance(lale_object, BasePipeline):
        if len(hyperparams) != len(lale_object.steps()):
            raise ValueError('The number of steps in the hyper-parameter space does not match the number of steps in the pipeline.')
        op_instances = []
        edges = lale_object.edges()
        #op_map:Dict[PlannedOpType, TrainableOperator] = {}
        op_map = {}
        for op_index in range(len(hyperparams)):
            state_params = hyperparams[op_index]
            #TODO: Should ideally check if the class_name is the same as the class name of the op from self.operators() at op_index
            op_instance = create_instance_from_hyperopt_search_space(lale_object.steps()[op_index], state_params)
            op_instances.append(op_instance)
            orig_op = lale_object._steps[op_index]
            op_map[orig_op] = op_instance

        #trainable_edges:List[Tuple[TrainableOperator, TrainableOperator]]
        try:
            trainable_edges = [(op_map[x], op_map[y]) for (x, y) in edges]
        except KeyError as e:
            raise ValueError("An edge was found with an endpoint that is not a step (" + str(e) + ")")

        return TrainablePipeline(op_instances, trainable_edges, ordered=True)
    elif isinstance(lale_object, OperatorChoice):
        #Hyperopt search space for an OperatorChoice is generated as a dictionary with a single element
        #corresponding to the choice made, the only key is the index of the step and the value is 
        #the params corresponding to that step.
        step_index, hyperparams = list(hyperparams.items())[0]
        step_object = lale_object.steps()[step_index]
        return create_instance_from_hyperopt_search_space(step_object, hyperparams)
        
def import_from_sklearn_pipeline(sklearn_pipeline):
    #For all pipeline steps, identify equivalent lale wrappers if present,
    #if not, call make operator on sklearn classes and create a lale pipeline.

    def get_equivalent_lale_op(sklearn_obj):
        module_name = "lale.lib.sklearn"
        from sklearn.base import clone
        from lale.operators import make_operator

        class_name = sklearn_obj.__class__.__name__
        module = importlib.import_module(module_name)
        try:
            class_ = getattr(module, class_name)
        except AttributeError:
            class_ = make_operator(sklearn_obj, name=class_name)
        class_ = class_(**sklearn_obj.get_params())
        class_._impl._sklearn_model =  copy.deepcopy(sklearn_obj)
        return class_         

    from sklearn.pipeline import FeatureUnion, Pipeline
    from sklearn.base import BaseEstimator
    from lale.operators import make_pipeline, make_union
    
    if isinstance(sklearn_pipeline, Pipeline):
        nested_pipeline_steps = sklearn_pipeline.named_steps.values()
        nested_pipeline_lale_objects = [import_from_sklearn_pipeline(nested_pipeline_step) for nested_pipeline_step in nested_pipeline_steps]
        lale_op_obj = make_pipeline(*nested_pipeline_lale_objects)
    elif isinstance(sklearn_pipeline, FeatureUnion):
        transformer_list = sklearn_pipeline.transformer_list
        concat_predecessors = [import_from_sklearn_pipeline(transformer[1]) for transformer in transformer_list]
        lale_op_obj = make_union(*concat_predecessors)
    else:
        lale_op_obj = get_equivalent_lale_op(sklearn_pipeline)
    return lale_op_obj
        
def get_hyperparam_names(op):
    import inspect
    if op._impl.__module__.startswith('lale'):
        hp_schema = op.hyperparam_schema()
        params = next(iter(hp_schema.get('allOf', []))).get('properties', {})
        return list(params.keys())
    else:
        return inspect.getargspec(op._impl.__class__.__init__).args
                
def validate_method(op, m):
    if op._impl.__module__.startswith('lale'):
        assert (m in op._schemas['properties'].keys())
    else:
        a = ''
        if m.startswith('input_'):
            a = m[len('input_'):]
        elif m.startswith('output_'):
            a = m[len('output_'):]
        if a:
            assert (hasattr(op._impl, a))

def camelCase_to_snake(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

def get_lib_schema(impl):
    if impl.__module__.startswith('lale.lib'):
        m = importlib.import_module(impl.__module__)
        return m._combined_schemas
    module_name = impl.__module__.split('.')[0]
    class_name = camelCase_to_snake(impl.__class__.__name__)
    try:
        lib_name = '.'.join(['lale.lib', module_name, class_name])
        m = importlib.import_module(lib_name)
        return m._combined_schemas
    except (ModuleNotFoundError, AttributeError):
        try:
            lib_name = '.'.join(['lale.lib.autogen', class_name])
            m = importlib.import_module(lib_name)
            return m._combined_schemas
        except:
            return None

def signature_to_schema(sig):
    sig_schema = {'type': 'object', 'properties': {}}
    for name, param in sig.parameters.items():
        ignored_kinds = [inspect.Parameter.VAR_POSITIONAL,
                         inspect.Parameter.VAR_KEYWORD]
        if param.kind not in ignored_kinds:
            param_schema = {}
            if param.default != inspect.Parameter.empty:
                param_schema['default'] = param.default
            sig_schema['properties'][name] = param_schema
    return sig_schema

def get_default_schema(impl):
    if hasattr(impl, '__init__'):
        sig = inspect.signature(impl.__init__)
        arg_schemas = signature_to_schema(sig)
    else:
        arg_schemas = {'type': 'object', 'properties': {}}
    return {
        '$schema': 'http://json-schema.org/draft-04/schema#',
        'description':
        'Combined schema for expected data and hyperparameters.',
        'type': 'object',
        'properties': {
            'input_fit': {},
            'input_predict': {},
            'output': {},
            'hyperparams': {
                'allOf': [arg_schemas]}}}
    
def wrap_imported_operators():
    import lale.lib
    from lale.operators import Operator, make_operator
    calling_frame = inspect.stack()[1][0]
    symtab = calling_frame.f_globals
    lib_modules = [p.name for p in pkgutil.iter_modules(lale.lib.__path__)]
    for name, impl in symtab.items():
        if inspect.isclass(impl) and not issubclass(impl, Operator) and not isinstance(impl, Operator):
            module = impl.__module__.split('.')[0]
            looks_like_op = hasattr(impl, 'fit') and (
                hasattr(impl, 'predict') or hasattr(impl, 'transform'))
            if module in lib_modules or looks_like_op:
                logger.info(f'Lale:Wrapped operator:{name}')
                symtab[name] = make_operator(impl=impl, name=name)

class val_wrapper():
    """This is used to wrap values that cause problems for hyper-optimizer backends
    lale will unwrap these when given them as the value of a hyper-parameter"""

    def __init__(self, base):
        self._base = base

    def unwrap_self(self):
        return self._base

    @classmethod
    def unwrap(cls, obj):
        if isinstance(obj, cls):
            return cls.unwrap(obj.unwrap_self())
        else:
            return obj

def append_batch(data, batch_data):
    if data is None:
        return batch_data
    elif isinstance(data, np.ndarray):
        if isinstance(batch_data, np.ndarray):
            if len(data.shape) == 1 and len(batch_data.shape) == 1:
                return np.concatenate([data, batch_data])
            else:
                return np.vstack((data, batch_data))
    elif isinstance(data, tuple):
        X, y = data
        if isinstance(batch_data, tuple):
            batch_X, batch_y = batch_data
            X = append_batch(X, batch_X)
            y = append_batch(y, batch_y)
            return X, y
    elif isinstance(data, torch.Tensor):
        if isinstance(batch_data, torch.Tensor):
            return torch.cat((data, batch_data))
    elif isinstance(data, h5py.File):
        if isinstance(batch_data, tuple):
            batch_X, batch_y = batch_data
            
    #TODO:Handle dataframes

def create_data_loader(X, y = None, batch_size = 1):
    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
    elif isinstance(X, np.ndarray):
        #unfortunately, NumpyTorchDataset won't accept a subclass of np.ndarray
        if isinstance(X, lale.datasets.data_schemas.NDArrayWithSchema):
            X = X.view(np.ndarray)
        if isinstance(y, lale.datasets.data_schemas.NDArrayWithSchema):
            y = y.view(np.ndarray)
        dataset = NumpyTorchDataset(X, y)
    elif isinstance(X, str):#Assume that this is path to hdf5 file
        dataset = HDF5TorchDataset(X)
    elif isinstance(X, dict): #Assumed that it is data indexed by batch number
        #dataset = BatchDataDictDataset(X)
        return X.values()
    elif isinstance(X, torch.Tensor) and y is not None:
        dataset = TensorDataset(X, y)
    elif isinstance(X, torch.Tensor):
        dataset = TensorDataset(X)
    else:
        raise TypeError("Can not create a data loader for a dataset with type {}".format(type(X)))
    return DataLoader(dataset, batch_size=batch_size)

def write_batch_output_to_file(file_obj, file_path, total_len, batch_idx, batch_X, batch_y, batch_out_X, batch_out_y):
    if file_obj is None and file_path is None:
        raise ValueError("Only one of the file object or file path can be None.")
    if file_obj is None:
        file_obj = h5py.File(file_path, 'w')
        #estimate the size of the dataset based on the first batch output size
        transform_ratio = int(len(batch_out_X)/len(batch_X))
        if len(batch_out_X.shape) == 1:
            h5_data_shape = (transform_ratio*total_len, )
        if len(batch_out_X.shape) == 2:
            h5_data_shape = (transform_ratio*total_len, batch_out_X.shape[1])
        elif len(batch_out_X.shape) == 3:
            h5_data_shape = (transform_ratio*total_len, batch_out_X.shape[1], batch_out_X.shape[2])
        dataset = file_obj.create_dataset(name='X', shape=h5_data_shape, chunks=True, compression="gzip")
        if batch_out_y is None and batch_y is not None:
            batch_out_y = batch_y
        if batch_out_y is not None:
            if len(batch_out_y.shape) == 1:
                h5_labels_shape = (transform_ratio*total_len, )
            elif len(batch_out_y.shape) == 2:
                h5_labels_shape = (transform_ratio*total_len, batch_out_y.shape[1])
            dataset = file_obj.create_dataset(name='y', shape=h5_labels_shape, chunks=True, compression="gzip")
    dataset = file_obj['X']
    dataset[batch_idx*len(batch_out_X):(batch_idx+1)*len(batch_out_X)] = batch_out_X
    if batch_out_y is not None or batch_y is not None:
        labels = file_obj['y']
        if batch_out_y is not None:
            labels[batch_idx*len(batch_out_y):(batch_idx+1)*len(batch_out_y)] = batch_out_y
        else:
            labels[batch_idx*len(batch_y):(batch_idx+1)*len(batch_y)] = batch_y
    return file_obj

def best_estimator(obj):
    if hasattr(obj, '_impl'): #TrainedIndividualOp
        if hasattr(obj._impl, 'best_estimator'): #HyperoptRegressorImpl
            return obj._impl.best_estimator
        return None
    return obj

def is_empty_dict(val) -> bool:
    return isinstance(val, dict) and len(val) == 0
