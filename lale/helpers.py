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
import copy
import importlib
import logging
import os
import re
import sys
import time
import traceback
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import h5py
import numpy as np
import pandas as pd
import scipy.sparse
import sklearn.pipeline
from sklearn.metrics import accuracy_score, check_scoring, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.metaestimators import _safe_split

import lale.datasets.data_schemas

try:
    import torch

    torch_installed = True
except ImportError:
    torch_installed = False

logger = logging.getLogger(__name__)

LALE_NESTED_SPACE_KEY = "__lale_nested_space"


def make_nested_hyperopt_space(sub_space):
    return {LALE_NESTED_SPACE_KEY: sub_space}


def assignee_name(level=1) -> Optional[str]:
    tb = traceback.extract_stack()
    file_name, line_number, function_name, text = tb[-(level + 2)]
    try:
        tree = ast.parse(text, file_name)
    except SyntaxError:
        return None
    assert tree is not None and isinstance(tree, ast.Module)
    if len(tree.body) == 1:
        stmt = tree.body[0]
        if isinstance(stmt, ast.Assign):
            lhs = stmt.targets
            if len(lhs) == 1:
                res = lhs[0]
                if isinstance(res, ast.Name):
                    return res.id
    return None


def arg_name(pos=0, level=1) -> Optional[str]:
    tb = traceback.extract_stack()
    file_name, line_number, function_name, text = tb[-(level + 2)]
    try:
        tree = ast.parse(text, file_name)
    except SyntaxError:
        return None
    assert tree is not None and isinstance(tree, ast.Module)
    if len(tree.body) == 1:
        stmt = tree.body[0]
        if isinstance(stmt, ast.Expr):
            expr = stmt.value
            if isinstance(expr, ast.Call):
                args = expr.args
                if pos < len(args):
                    res = args[pos]
                    if isinstance(res, ast.Name):
                        return res.id
    return None


def data_to_json(data, subsample_array: bool = True) -> Union[list, dict, int, float]:
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
    elif torch_installed and isinstance(data, torch.Tensor):
        np_array = data.detach().numpy()
        return ndarray_to_json(np_array, subsample_array)
    elif (
        isinstance(data, np.int64)
        or isinstance(data, np.int32)
        or isinstance(data, np.int16)
    ):
        return int(data)
    elif isinstance(data, np.float64) or isinstance(data, np.float32):
        return float(data)
    else:
        return data


def is_empty_dict(val) -> bool:
    return isinstance(val, dict) and len(val) == 0


def dict_without(orig_dict: Dict[str, Any], key: str) -> Dict[str, Any]:
    return {k: orig_dict[k] for k in orig_dict if k != key}


def json_lookup(ptr, jsn, default=None):
    steps = ptr.split("/")
    sub_jsn = jsn
    for s in steps:
        if s not in sub_jsn:
            return default
        sub_jsn = sub_jsn[s]
    return sub_jsn


def ndarray_to_json(arr: np.ndarray, subsample_array: bool = True) -> Union[list, dict]:
    # sample 10 rows and no limit on columns
    num_subsamples: List[int]
    if subsample_array:
        num_subsamples = [10, np.iinfo(np.int).max, np.iinfo(np.int).max]
    else:
        num_subsamples = [
            np.iinfo(np.int).max,
            np.iinfo(np.int).max,
            np.iinfo(np.int).max,
        ]

    def subarray_to_json(indices: Tuple[int, ...]) -> Any:
        if len(indices) == len(arr.shape):
            if (
                isinstance(arr[indices], bool)
                or isinstance(arr[indices], int)
                or isinstance(arr[indices], float)
                or isinstance(arr[indices], str)
            ):
                return arr[indices]
            elif np.issubdtype(arr.dtype, np.bool_):
                return bool(arr[indices])
            elif np.issubdtype(arr.dtype, np.integer):
                return int(arr[indices])
            elif np.issubdtype(arr.dtype, np.number):
                return float(arr[indices])
            elif arr.dtype.kind in ["U", "S", "O"]:
                return str(arr[indices])
            else:
                raise ValueError(
                    f"Unexpected dtype {arr.dtype}, "
                    f"kind {arr.dtype.kind}, "
                    f"type {type(arr[indices])}."
                )
        else:
            assert len(indices) < len(arr.shape)
            return [
                subarray_to_json(indices + (i,))
                for i in range(
                    min(num_subsamples[len(indices)], arr.shape[len(indices)])
                )
            ]

    return subarray_to_json(())


def split_with_schemas(estimator, all_X, all_y, indices, train_indices=None):
    subset_X, subset_y = _safe_split(estimator, all_X, all_y, indices, train_indices)
    if hasattr(all_X, "json_schema"):
        n_rows = subset_X.shape[0]
        schema = {
            "type": "array",
            "minItems": n_rows,
            "maxItems": n_rows,
            "items": all_X.json_schema["items"],
        }
        lale.datasets.data_schemas.add_schema(subset_X, schema)
    if hasattr(all_y, "json_schema"):
        n_rows = subset_y.shape[0]
        schema = {
            "type": "array",
            "minItems": n_rows,
            "maxItems": n_rows,
            "items": all_y.json_schema["items"],
        }
        lale.datasets.data_schemas.add_schema(subset_y, schema)
    return subset_X, subset_y


def fold_schema(X, y, cv=1, is_classifier=True):
    def fold_schema_aux(data, n_rows):
        orig_schema = lale.datasets.data_schemas.to_schema(data)
        aux_result = {**orig_schema, "minItems": n_rows, "maxItems": n_rows}
        return aux_result

    n_splits = cv if isinstance(cv, int) else cv.get_n_splits()
    n_samples = X.shape[0] if hasattr(X, "shape") else len(X)
    if n_splits == 1:
        n_rows_fold = n_samples
    elif is_classifier:
        n_classes = len(set(y))
        n_rows_unstratified = (n_samples // n_splits) * (n_splits - 1)
        # in stratified case, fold sizes can differ by up to n_classes
        n_rows_fold = max(1, n_rows_unstratified - n_classes)
    else:
        n_rows_fold = (n_samples // n_splits) * (n_splits - 1)
    schema_X = fold_schema_aux(X, n_rows_fold)
    schema_y = fold_schema_aux(y, n_rows_fold)
    result = {"properties": {"X": schema_X, "y": schema_y}}
    return result


def cross_val_score_track_trials(
    estimator, X, y=None, scoring=accuracy_score, cv=5, args_to_scorer=None
):
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
    args_to_scorer: A dictionary of additional keyword arguments to pass to the scorer.
                Used for cases where the scorer has a signature such as ``scorer(estimator, X, y, **kwargs)``.
    Returns
    -------
        cv_results: a list of scores corresponding to each cross validation fold
    """
    if isinstance(cv, int):
        cv = StratifiedKFold(cv)

    if args_to_scorer is None:
        args_to_scorer = {}
    scorer = check_scoring(estimator, scoring=scoring)
    cv_results: List[float] = []
    log_loss_results = []
    time_results = []
    for train, test in cv.split(X, y):
        X_train, y_train = split_with_schemas(estimator, X, y, train)
        X_test, y_test = split_with_schemas(estimator, X, y, test, train)
        start = time.time()
        # Not calling sklearn.base.clone() here, because:
        #  (1) For Lale pipelines, clone() calls the pipeline constructor
        #      with edges=None, so the resulting topology is incorrect.
        #  (2) For Lale individual operators, the fit() method already
        #      clones the impl object, so cloning again is redundant.
        trained = estimator.fit(X_train, y_train)
        score_value = scorer(trained, X_test, y_test, **args_to_scorer)
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
    result = (
        np.array(cv_results).mean(),
        np.array(log_loss_results).mean(),
        np.array(time_results).mean(),
    )
    return result


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


def create_individual_op_using_reflection(class_name, operator_name, param_dict):
    instance = None
    if class_name is not None:
        class_name_parts = class_name.split(".")
        assert (
            len(class_name_parts)
        ) > 1, (
            "The class name needs to be fully qualified, i.e. module name + class name"
        )
        module_name = ".".join(class_name_parts[0:-1])
        class_name = class_name_parts[-1]

        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)

        if param_dict is None:
            instance = class_()
        else:
            instance = class_(**param_dict)
    return instance


if TYPE_CHECKING:
    import lale.operators


def to_graphviz(
    lale_operator: "lale.operators.Operator",
    ipython_display: bool = True,
    call_depth: int = 1,
    **dot_graph_attr,
):
    import lale.json_operator
    import lale.operators
    import lale.visualize

    if not isinstance(lale_operator, lale.operators.Operator):
        raise TypeError("The input to to_graphviz needs to be a valid LALE operator.")
    jsn = lale.json_operator.to_json(lale_operator, call_depth=call_depth + 1)
    dot = lale.visualize.json_to_graphviz(jsn, ipython_display, dot_graph_attr)
    return dot


def println_pos(message, out_file=sys.stdout):
    tb = traceback.extract_stack()[-2]
    match = re.search(r"<ipython-input-([0-9]+)-", tb[0])
    if match:
        pos = "notebook cell [{}] line {}".format(match[1], tb[1])
    else:
        pos = "{}:{}".format(tb[0], tb[1])
    strtime = time.strftime("%Y-%m-%d_%H-%M-%S")
    to_log = "{}: {} {}".format(pos, strtime, message)
    print(to_log, file=out_file)
    if match:
        os.system("echo {}".format(to_log))


def instantiate_from_hyperopt_search_space(obj_hyperparams, new_hyperparams):
    if isinstance(new_hyperparams, dict) and LALE_NESTED_SPACE_KEY in new_hyperparams:
        sub_params = new_hyperparams[LALE_NESTED_SPACE_KEY]

        sub_op = obj_hyperparams
        if isinstance(sub_op, list):
            if len(sub_op) == 1:
                sub_op = sub_op[0]
            else:
                step_index, step_params = list(sub_params)[0]
                if step_index < len(sub_op):
                    sub_op = sub_op[step_index]
                    sub_params = step_params

        return create_instance_from_hyperopt_search_space(sub_op, sub_params)

    elif isinstance(new_hyperparams, (list, tuple)):
        assert isinstance(obj_hyperparams, (list, tuple))
        params_len = len(new_hyperparams)
        assert params_len == len(obj_hyperparams)
        res: Optional[List[Any]] = None

        for i in range(params_len):
            nhi = new_hyperparams[i]
            ohi = obj_hyperparams[i]
            updated_params = instantiate_from_hyperopt_search_space(ohi, nhi)
            if updated_params is not None:
                if res is None:
                    res = list(new_hyperparams)
                res[i] = updated_params
        if res is not None:
            if isinstance(obj_hyperparams, tuple):
                return tuple(res)
            else:
                return res
        # workaround for what seems to be a hyperopt bug
        # where hyperopt returns a tuple even though the
        # hyperopt search space specifies a list
        is_obj_tuple = isinstance(obj_hyperparams, tuple)
        is_new_tuple = isinstance(new_hyperparams, tuple)
        if is_obj_tuple != is_new_tuple:
            if is_obj_tuple:
                return tuple(new_hyperparams)
            else:
                return list(new_hyperparams)
        return None

    elif isinstance(new_hyperparams, dict):
        assert isinstance(obj_hyperparams, dict)

        for k, sub_params in new_hyperparams.items():
            if k in obj_hyperparams:
                sub_op = obj_hyperparams[k]
                updated_params = instantiate_from_hyperopt_search_space(
                    sub_op, sub_params
                )
                if updated_params is not None:
                    new_hyperparams[k] = updated_params
        return None
    else:
        return None


def create_instance_from_hyperopt_search_space(lale_object, hyperparams):
    """
    Hyperparams is a n-tuple of dictionaries of hyper-parameters, each
    dictionary corresponds to an operator in the pipeline
    """
    # lale_object can either be an individual operator, a pipeline or an operatorchoice
    # Validate that the number of elements in the n-tuple is the same
    # as the number of steps in the current pipeline

    from lale.operators import (
        BasePipeline,
        OperatorChoice,
        PlannedIndividualOp,
        TrainablePipeline,
    )

    if isinstance(lale_object, PlannedIndividualOp):
        new_hyperparams: Dict[str, Any] = dict_without(hyperparams, "name")
        hps = lale_object.hyperparams()
        if hps is not None:
            obj_hyperparams = dict(hps)
        else:
            obj_hyperparams = {}

        for k, sub_params in new_hyperparams.items():
            if k in obj_hyperparams:
                sub_op = obj_hyperparams[k]
                updated_params = instantiate_from_hyperopt_search_space(
                    sub_op, sub_params
                )
                if updated_params is not None:
                    new_hyperparams[k] = updated_params

        all_hyperparams = {**obj_hyperparams, **new_hyperparams}
        return lale_object(**all_hyperparams)
    elif isinstance(lale_object, BasePipeline):
        steps = lale_object.steps()
        if len(hyperparams) != len(steps):
            raise ValueError(
                "The number of steps in the hyper-parameter space does not match the number of steps in the pipeline."
            )
        op_instances = []
        edges = lale_object.edges()
        # op_map:Dict[PlannedOpType, TrainableOperator] = {}
        op_map = {}
        for op_index, sub_params in enumerate(hyperparams):
            sub_op = steps[op_index]
            op_instance = create_instance_from_hyperopt_search_space(sub_op, sub_params)
            assert (
                isinstance(sub_op, OperatorChoice)
                or sub_op.class_name() == op_instance.class_name()
            ), f"sub_op {sub_op.class_name()}, op_instance {op_instance.class_name()}"
            op_instances.append(op_instance)
            op_map[sub_op] = op_instance

        # trainable_edges:List[Tuple[TrainableOperator, TrainableOperator]]
        try:
            trainable_edges = [(op_map[x], op_map[y]) for (x, y) in edges]
        except KeyError as e:
            raise ValueError(
                "An edge was found with an endpoint that is not a step (" + str(e) + ")"
            )

        return TrainablePipeline(op_instances, trainable_edges, ordered=True)
    elif isinstance(lale_object, OperatorChoice):
        # Hyperopt search space for an OperatorChoice is generated as a dictionary with a single element
        # corresponding to the choice made, the only key is the index of the step and the value is
        # the params corresponding to that step.
        step_index: int
        choices = lale_object.steps()

        if len(choices) == 1:
            step_index = 0
        else:
            step_index_str, hyperparams = list(hyperparams.items())[0]
            step_index = int(step_index_str)
        step_object = choices[step_index]
        return create_instance_from_hyperopt_search_space(step_object, hyperparams)


def import_from_sklearn_pipeline(sklearn_pipeline, fitted=True):
    # For all pipeline steps, identify equivalent lale wrappers if present,
    # if not, call make operator on sklearn classes and create a lale pipeline.
    def get_equivalent_lale_op(sklearn_obj, fitted):
        import lale.operators
        import lale.type_checking

        if isinstance(sklearn_obj, lale.operators.TrainableIndividualOp) and fitted:
            if hasattr(sklearn_obj, "_trained"):
                return sklearn_obj._trained
            elif not hasattr(
                sklearn_obj._impl_instance(), "fit"
            ):  # Operators such as NoOp do not have a fit, so return them as is.
                return sklearn_obj
            else:
                raise ValueError(
                    """The input pipeline has an operator that is not trained and fitted is set to True,
                    please pass fitted=False if you want a trainable pipeline as output."""
                )
        elif isinstance(sklearn_obj, lale.operators.Operator):
            return sklearn_obj

        # Validate that the sklearn_obj is a valid sklearn-compatible object
        if sklearn_obj is None or not hasattr(sklearn_obj, "get_params"):
            raise ValueError(
                "The input pipeline has a step that is not scikit-learn compatible."
            )

        orig_hyperparams = sklearn_obj.get_params()
        higher_order = False
        for hp_name, hp_val in orig_hyperparams.items():
            higher_order = higher_order or hasattr(hp_val, "get_params")
        if higher_order:
            hyperparams = {}
            for hp_name, hp_val in orig_hyperparams.items():
                if hasattr(hp_val, "get_params"):
                    nested_op = get_equivalent_lale_op(hp_val, fitted)
                    hyperparams[hp_name] = nested_op
                else:
                    hyperparams[hp_name] = hp_val
        else:
            hyperparams = orig_hyperparams

        module_names = [
            "lale.lib.sklearn",
            "lale.lib.autoai_libs",
            "lale.lib.xgboost",
            "lale.lib.lightgbm",
            "lale.lib.snapml",
        ]

        lale_wrapper_found = False
        class_name = sklearn_obj.__class__.__name__
        for module_name in module_names:
            module = importlib.import_module(module_name)
            try:
                class_ = getattr(module, class_name)
                lale_wrapper_found = True
                break
            except AttributeError:
                continue
        else:
            class_ = lale.operators.make_operator(sklearn_obj, name=class_name)

        if (
            not fitted
        ):  # If fitted is False, we do not want to return a Trained operator.
            lale_op = class_
        else:
            lale_op = lale.operators.TrainedIndividualOp(
                class_._name, class_._impl, class_._schemas, None, _lale_trained=True
            )

        class_ = lale_op(**hyperparams)

        if lale_wrapper_found and hasattr(class_._impl_instance(), "_wrapped_model"):
            wrapped_model = copy.deepcopy(sklearn_obj)
            class_._impl_instance()._wrapped_model = wrapped_model
        else:  # If there is no lale wrapper, there is no _wrapped_model
            class_._impl = copy.deepcopy(sklearn_obj)
            class_._impl_class_ = class_._impl.__class__
        return class_

    if isinstance(sklearn_pipeline, sklearn.pipeline.Pipeline):
        nested_pipeline_steps = sklearn_pipeline.named_steps.values()
        nested_pipeline_lale_objects = [
            import_from_sklearn_pipeline(nested_pipeline_step, fitted=fitted)
            for nested_pipeline_step in nested_pipeline_steps
        ]
        lale_op_obj = lale.operators.make_pipeline(*nested_pipeline_lale_objects)
    elif isinstance(sklearn_pipeline, sklearn.pipeline.FeatureUnion):
        transformer_list = sklearn_pipeline.transformer_list
        concat_predecessors = [
            import_from_sklearn_pipeline(transformer[1], fitted=fitted)
            for transformer in transformer_list
        ]
        lale_op_obj = lale.operators.make_union(*concat_predecessors)
    else:
        lale_op_obj = get_equivalent_lale_op(sklearn_pipeline, fitted=fitted)
    return lale_op_obj


class val_wrapper:
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
    elif torch_installed and isinstance(data, torch.Tensor):
        if isinstance(batch_data, torch.Tensor):
            return torch.cat((data, batch_data))
    elif isinstance(data, h5py.File):
        if isinstance(batch_data, tuple):
            batch_X, batch_y = batch_data

    # TODO:Handle dataframes


def create_data_loader(X, y=None, batch_size=1):
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    from lale.util.batch_data_dictionary_dataset import BatchDataDict
    from lale.util.hdf5_to_torch_dataset import HDF5TorchDataset
    from lale.util.numpy_to_torch_dataset import NumpyTorchDataset

    if isinstance(X, pd.DataFrame):
        X = X.to_numpy()
        if isinstance(y, pd.Series):
            y = y.to_numpy()
        dataset = NumpyTorchDataset(X, y)
    elif isinstance(X, scipy.sparse.csr.csr_matrix):
        # unfortunately, NumpyTorchDataset won't accept a subclass of np.ndarray
        X = X.toarray()
        if isinstance(y, lale.datasets.data_schemas.NDArrayWithSchema):
            y = y.view(np.ndarray)
        dataset = NumpyTorchDataset(X, y)
    elif isinstance(X, np.ndarray):
        # unfortunately, NumpyTorchDataset won't accept a subclass of np.ndarray
        if isinstance(X, lale.datasets.data_schemas.NDArrayWithSchema):
            X = X.view(np.ndarray)
        if isinstance(y, lale.datasets.data_schemas.NDArrayWithSchema):
            y = y.view(np.ndarray)
        dataset = NumpyTorchDataset(X, y)
    elif isinstance(X, str):  # Assume that this is path to hdf5 file
        dataset = HDF5TorchDataset(X)
    elif isinstance(X, BatchDataDict):
        dataset = X

        def my_collate_fn(batch):
            return batch[
                0
            ]  # because BatchDataDict's get_item returns a batch, so no collate is required.

        return DataLoader(dataset, batch_size=1, collate_fn=my_collate_fn)
    elif isinstance(X, dict):  # Assumed that it is data indexed by batch number
        return [X]
    elif isinstance(X, torch.Tensor) and y is not None:
        if isinstance(y, np.ndarray):
            y = torch.from_numpy(y)
        dataset = TensorDataset(X, y)
    elif isinstance(X, torch.Tensor):
        dataset = TensorDataset(X)
    else:
        raise TypeError(
            "Can not create a data loader for a dataset with type {}".format(type(X))
        )
    return DataLoader(dataset, batch_size=batch_size)


def write_batch_output_to_file(
    file_obj,
    file_path,
    total_len,
    batch_idx,
    batch_X,
    batch_y,
    batch_out_X,
    batch_out_y,
):
    if file_obj is None and file_path is None:
        raise ValueError("Only one of the file object or file path can be None.")
    if file_obj is None:
        file_obj = h5py.File(file_path, "w")
        # estimate the size of the dataset based on the first batch output size
        transform_ratio = int(len(batch_out_X) / len(batch_X))
        if len(batch_out_X.shape) == 1:
            h5_data_shape = (transform_ratio * total_len,)
        elif len(batch_out_X.shape) == 2:
            h5_data_shape = (transform_ratio * total_len, batch_out_X.shape[1])
        elif len(batch_out_X.shape) == 3:
            h5_data_shape = (
                transform_ratio * total_len,
                batch_out_X.shape[1],
                batch_out_X.shape[2],
            )
        else:
            raise ValueError(
                "batch_out_X is expected to be a 1-d, 2-d or 3-d array. Any other data types are not handled."
            )
        dataset = file_obj.create_dataset(
            name="X", shape=h5_data_shape, chunks=True, compression="gzip"
        )
        if batch_out_y is None and batch_y is not None:
            batch_out_y = batch_y
        if batch_out_y is not None:
            if len(batch_out_y.shape) == 1:
                h5_labels_shape = (transform_ratio * total_len,)
            elif len(batch_out_y.shape) == 2:
                h5_labels_shape = (transform_ratio * total_len, batch_out_y.shape[1])
            else:
                raise ValueError(
                    "batch_out_y is expected to be a 1-d or 2-d array. Any other data types are not handled."
                )
            dataset = file_obj.create_dataset(
                name="y", shape=h5_labels_shape, chunks=True, compression="gzip"
            )
    dataset = file_obj["X"]
    dataset[
        batch_idx * len(batch_out_X) : (batch_idx + 1) * len(batch_out_X)
    ] = batch_out_X
    if batch_out_y is not None or batch_y is not None:
        labels = file_obj["y"]
        if batch_out_y is not None:
            labels[
                batch_idx * len(batch_out_y) : (batch_idx + 1) * len(batch_out_y)
            ] = batch_out_y
        else:
            labels[batch_idx * len(batch_y) : (batch_idx + 1) * len(batch_y)] = batch_y
    return file_obj


def add_missing_values(orig_X, missing_rate=0.1, seed=None):
    # see scikit-learn.org/stable/auto_examples/impute/plot_missing_values.html
    n_samples, n_features = orig_X.shape
    n_missing_samples = int(n_samples * missing_rate)
    if seed is None:
        rng = np.random.RandomState()
    else:
        rng = np.random.RandomState(seed)
    missing_samples = np.zeros(n_samples, dtype=np.bool)
    missing_samples[:n_missing_samples] = True
    rng.shuffle(missing_samples)
    missing_features = rng.randint(0, n_features, n_missing_samples)
    missing_X = orig_X.copy()
    if isinstance(missing_X, np.ndarray):
        missing_X[missing_samples, missing_features] = np.nan
    else:
        assert isinstance(missing_X, pd.DataFrame)
        i_missing_sample = 0
        for i_sample in range(n_samples):
            if missing_samples[i_sample]:
                i_feature = missing_features[i_missing_sample]
                i_missing_sample += 1
                missing_X.iloc[i_sample, i_feature] = np.nan
    return missing_X


# helpers for manipulating (extended) sklearn style paths.
# documentation of the path format is part of the operators module docstring


def partition_sklearn_params(
    d: Dict[str, Any]
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    sub_parts: Dict[str, Dict[str, Any]] = {}
    main_parts: Dict[str, Any] = {}

    for k, v in d.items():
        ks = k.split("__", 1)
        if len(ks) == 1:
            assert k not in main_parts
            main_parts[k] = v
        else:
            assert len(ks) == 2
            bucket: Dict[str, Any] = {}
            group: str = ks[0]
            param: str = ks[1]
            if group in sub_parts:
                bucket = sub_parts[group]
            else:
                sub_parts[group] = bucket
            assert param not in bucket
            bucket[param] = v
    return (main_parts, sub_parts)


def partition_sklearn_choice_params(d: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    discriminant_value: int = -1
    choice_parts: Dict[str, Any] = {}

    for k, v in d.items():
        if k == discriminant_name:
            assert discriminant_value == -1
            discriminant_value = int(v)
        else:
            k_rest = unnest_choice(k)
            choice_parts[k_rest] = v
    assert discriminant_value != -1
    return (discriminant_value, choice_parts)


DUMMY_SEARCH_SPACE_GRID_PARAM_NAME: str = "$"
discriminant_name: str = "?"
choice_prefix: str = "?"
structure_type_name: str = "#"
structure_type_list: str = "list"
structure_type_tuple: str = "tuple"
structure_type_dict: str = "dict"


def get_name_and_index(name: str) -> Tuple[str, int]:
    """given a name of the form "name@i", returns (name, i)
    if given a name of the form "name", returns (name, 0)
    """
    splits = name.split("@", 1)
    if len(splits) == 1:
        return splits[0], 0
    else:
        return splits[0], int(splits[1])


def make_degen_indexed_name(name, index):
    return f"{name}@{index}"


def make_indexed_name(name, index):
    if index == 0:
        return name
    else:
        return f"{name}@{index}"


def make_array_index_name(index, is_tuple: bool = False):
    sep = "##" if is_tuple else "#"
    return f"{sep}{str(index)}"


def is_numeric_structure(structure_type: str):

    if structure_type == "list" or structure_type == "tuple":
        return True
    elif structure_type == "dict":
        return False
    else:
        assert False, f"Unknown structure type {structure_type} found"


V = TypeVar("V")


def nest_HPparam(name: str, key: str):
    if key == DUMMY_SEARCH_SPACE_GRID_PARAM_NAME:
        # we can get rid of the dummy now, since we have a name for it
        return name
    return name + "__" + key


def nest_HPparams(name: str, grid: Mapping[str, V]) -> Dict[str, V]:
    return {(nest_HPparam(name, k)): v for k, v in grid.items()}


def nest_all_HPparams(
    name: str, grids: Iterable[Mapping[str, V]]
) -> List[Dict[str, V]]:
    """Given the name of an operator in a pipeline, this transforms every key(parameter name) in the grids
    to use the operator name as a prefix (separated by __).  This is the convention in scikit-learn pipelines.
    """
    return [nest_HPparams(name, grid) for grid in grids]


def nest_choice_HPparam(key: str):
    return choice_prefix + key


def nest_choice_HPparams(grid: Mapping[str, V]) -> Dict[str, V]:
    return {(nest_choice_HPparam(k)): v for k, v in grid.items()}


def nest_choice_all_HPparams(grids: Iterable[Mapping[str, V]]) -> List[Dict[str, V]]:
    """this transforms every key(parameter name) in the grids
    to be nested under a choice, using a ? as a prefix (separated by __).  This is the convention in scikit-learn pipelines.
    """
    return [nest_choice_HPparams(grid) for grid in grids]


def unnest_choice(k: str) -> str:
    assert k.startswith(choice_prefix)
    return k[len(choice_prefix) :]


def unnest_HPparams(k: str) -> List[str]:
    return k.split("__")


def are_hyperparameters_equal(hyperparam1, hyperparam2):
    if isinstance(
        hyperparam1, np.ndarray
    ):  # hyperparam2 is from schema default, so it may not always be an array
        return np.all(hyperparam1 == hyperparam2)
    else:
        return hyperparam1 == hyperparam2
