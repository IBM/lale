# Copyright 2021, 2022 IBM Corporation
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

import enum
import functools
from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)

import graphviz
import pandas as pd
import sklearn.model_selection
import sklearn.tree

import lale.helpers
import lale.json_operator
from lale.operators import (
    TrainableIndividualOp,
    TrainablePipeline,
    TrainedIndividualOp,
    TrainedPipeline,
)

_TaskStatus = enum.Enum("_TaskStatus", "FRESH READY WAITING DONE")

_Operation = enum.Enum(
    "_Operation", "SCAN TRANSFORM PREDICT FIT PARTIAL_FIT LIFT COMBINE"
)

_DUMMY_INPUT_STEP = -1


def is_incremental(op: TrainableIndividualOp) -> bool:
    return op.has_method("partial_fit")


def is_associative(op: TrainableIndividualOp) -> bool:
    return op.has_method("_lift") and op.has_method("_combine")


def _batch_id(fold: str, idx: int) -> str:
    return fold + str(idx)


def _get_fold(batch_id: str) -> str:
    return batch_id[0]


def _get_idx(batch_id: str) -> int:
    return int(batch_id[1:])


_MemoKey = Tuple[Type["_Task"], int, Tuple[str, ...], Optional[str]]


class _Task:
    preds: List["_Task"]
    succs: List["_Task"]
    seq_id: Optional[int]

    def __init__(
        self, step_id: int, batch_ids: Tuple[str, ...], held_out: Optional[str]
    ):
        self.step_id = step_id
        self.batch_ids = batch_ids
        self.held_out = held_out
        self.status = _TaskStatus.FRESH
        self.preds = []
        self.succs = []
        self.seq_id = None  # for verbose visualization

    @abstractmethod
    def get_operation(
        self, pipeline: TrainablePipeline[TrainableIndividualOp]
    ) -> _Operation:
        pass

    def memo_key(self) -> _MemoKey:
        return type(self), self.step_id, self.batch_ids, self.held_out


class _TrainTask(_Task):
    lifted: Optional[Tuple[Any, ...]]
    trained: Optional[TrainedIndividualOp]

    def __init__(self, step_id: int, batch_ids: Tuple[str, ...], held_out: str):
        super(_TrainTask, self).__init__(step_id, batch_ids, held_out)
        self.lifted = None
        self.trained = None

    def get_operation(
        self, pipeline: TrainablePipeline[TrainableIndividualOp]
    ) -> _Operation:
        any_pred_train = any(isinstance(p, _TrainTask) for p in self.preds)
        any_succ_train = any(isinstance(s, _TrainTask) for s in self.succs)
        if not any_pred_train and not any_succ_train:
            return _Operation.FIT
        step = pipeline.steps_list()[self.step_id]
        if is_associative(step):
            return _Operation.LIFT if len(self.batch_ids) == 1 else _Operation.COMBINE
        return _Operation.PARTIAL_FIT

    def get_trained(
        self, pipeline: TrainablePipeline[TrainableIndividualOp]
    ) -> TrainedIndividualOp:
        if self.trained is None:
            assert self.lifted is not None
            trainable = pipeline.steps_list()[self.step_id]
            self.trained = trainable.convert_to_trained()
            hyperparams = trainable.impl._hyperparams
            self.trained._impl = trainable._impl_class()(**hyperparams)
            self.trained._impl._set_fit_attributes(self.lifted)
        return self.trained


_Batch = Tuple[pd.DataFrame, pd.Series]


class _ApplyTask(_Task):
    batch: Optional[_Batch]

    def __init__(self, step_id: int, batch_ids: Tuple[str, ...], held_out: str):
        super(_ApplyTask, self).__init__(step_id, batch_ids, held_out)
        self.batch = None

    def get_operation(self, pipeline: TrainablePipeline) -> _Operation:
        if self.step_id == _DUMMY_INPUT_STEP:
            return _Operation.SCAN
        step = pipeline.steps_list()[self.step_id]
        return _Operation.TRANSFORM if step.is_transformer() else _Operation.PREDICT


class Prio(ABC):
    arity: int

    def bottom(self) -> Any:  # tuple of "inf" means all others are more important
        return self.arity * (float("inf"),)

    @abstractmethod
    def task_priority(self, task: _Task) -> Any:  # prefer to do first if lower
        pass


class PrioStep(Prio):
    arity = 5

    def task_priority(self, task: _Task) -> Any:
        result = (
            task.status.value,
            task.step_id,
            len(task.batch_ids),
            task.batch_ids,
            0 if isinstance(task, _TrainTask) else 1,
        )
        assert len(result) == self.arity
        return result


class PrioBatch(Prio):
    arity = 5

    def task_priority(self, task: _Task) -> Any:
        result = (
            task.status.value,
            len(task.batch_ids),
            task.batch_ids,
            task.step_id,
            0 if isinstance(task, _TrainTask) else 1,
        )
        assert len(result) == self.arity
        return result


def _visualize_tasks(
    tasks: Dict[_MemoKey, _Task],
    pipeline: TrainablePipeline[TrainableIndividualOp],
    prio: Prio,
    call_depth: int,
) -> None:
    steps = pipeline.steps_list()
    cls2label = lale.json_operator._get_cls2label(call_depth + 1)

    def step_id_to_string(step_id: int) -> str:
        if step_id == _DUMMY_INPUT_STEP:
            return "INP"
        cls = steps[step_id].class_name()
        return cls2label[cls] if cls in cls2label else steps[step_id].name()

    def task_to_string(task: _Task) -> str:
        seq_id_s = "" if task.seq_id is None else f"{task.seq_id} "
        operation_s = task.get_operation(pipeline).name.lower()
        step_s = step_id_to_string(task.step_id)
        batches_s = ",".join(task.batch_ids)
        held_out_s = "" if task.held_out is None else f"#~{task.held_out}"
        return f"{seq_id_s}{operation_s}\n{step_s}({batches_s}){held_out_s}"

    dot = graphviz.Digraph()
    dot.attr("graph", rankdir="LR", nodesep="0.1")
    dot.attr("node", fontsize="11", margin="0.03,0.03", shape="box", height="0.1")
    next_task = min(tasks.values(), key=lambda t: prio.task_priority(t))
    for task in tasks.values():
        if task.status is _TaskStatus.FRESH:
            color = "white"
        elif task.status is _TaskStatus.READY:
            color = "lightgreen" if task is next_task else "yellow"
        elif task.status is _TaskStatus.WAITING:
            color = "coral"
        else:
            assert task.status is _TaskStatus.DONE
            color = "lightgray"
        dot.node(task_to_string(task), style="filled", fillcolor=color)
    for task in tasks.values():
        for succ in task.succs:
            dot.edge(task_to_string(task), task_to_string(succ))

    import IPython.display

    IPython.display.display(dot)


class _TaskGraph:
    step_ids: Dict[TrainableIndividualOp, int]
    step_id_preds: Dict[int, List[int]]
    fresh_tasks: List[_Task]
    all_tasks: Dict[_MemoKey, _Task]

    def __init__(self, pipeline: TrainablePipeline[TrainableIndividualOp]):
        self.pipeline = pipeline
        self.step_ids = {step: i for i, step in enumerate(pipeline.steps_list())}
        self.step_id_preds = {
            self.step_ids[s]: (
                [_DUMMY_INPUT_STEP]
                if len(pipeline._preds[s]) == 0
                else [self.step_ids[p] for p in pipeline._preds[s]]
            )
            for s in pipeline.steps_list()
        }
        self.fresh_tasks = []
        self.all_tasks = {}

    def find_or_create(
        self,
        task_class: Type["_Task"],
        step_id: int,
        batch_ids: Tuple[str, ...],
        held_out: Optional[str],
    ) -> _Task:
        memo_key = task_class, step_id, batch_ids, held_out
        if memo_key not in self.all_tasks:
            task = task_class(step_id, batch_ids, held_out)
            self.all_tasks[memo_key] = task
            self.fresh_tasks.append(task)
        return self.all_tasks[memo_key]


def _create_tasks_batching(
    pipeline: TrainablePipeline[TrainableIndividualOp],
    all_batch_ids: Tuple[str, ...],
    incremental: bool,
) -> Dict[_MemoKey, _Task]:
    tg = _TaskGraph(pipeline)
    for step in pipeline._find_sink_nodes():
        tg.find_or_create(_TrainTask, tg.step_ids[step], all_batch_ids, None)
    while len(tg.fresh_tasks) > 0:
        task = tg.fresh_tasks.pop()
        if isinstance(task, _TrainTask):
            if len(task.batch_ids) == 1:
                for pred_step_id in tg.step_id_preds[task.step_id]:
                    task.preds.append(
                        tg.find_or_create(
                            _ApplyTask, pred_step_id, task.batch_ids, None
                        )
                    )
            else:
                step = pipeline.steps_list()[task.step_id]
                if is_associative(step):
                    for batch_id in task.batch_ids:
                        task.preds.append(
                            tg.find_or_create(
                                _TrainTask, task.step_id, (batch_id,), None
                            )
                        )
                elif is_incremental(step):
                    task.preds.append(
                        tg.find_or_create(
                            _TrainTask, task.step_id, task.batch_ids[:-1], None
                        )
                    )
                    for pred_step_id in tg.step_id_preds[task.step_id]:
                        task.preds.append(
                            tg.find_or_create(
                                _ApplyTask, pred_step_id, task.batch_ids[-1:], None
                            )
                        )
                else:
                    assert False, "non-incremental operator " + step.name()
        if isinstance(task, _ApplyTask) and task.step_id != _DUMMY_INPUT_STEP:
            if incremental:
                fit_upto = _get_idx(task.batch_ids[-1]) + 1
            else:
                fit_upto = len(all_batch_ids)
            fold = _get_fold(task.batch_ids[-1])
            task.preds.append(
                tg.find_or_create(
                    _TrainTask,
                    task.step_id,
                    tuple(_batch_id(fold, idx) for idx in range(fit_upto)),
                    None,
                )
            )
            for pred_step_id in tg.step_id_preds[task.step_id]:
                task.preds.append(
                    tg.find_or_create(_ApplyTask, pred_step_id, task.batch_ids, None)
                )
        for pred_task in task.preds:
            pred_task.succs.append(task)
    return tg.all_tasks


def _create_tasks_cross_val(
    pipeline: TrainablePipeline[TrainableIndividualOp],
    folds: List[str],
    n_batches_per_fold: int,
    same_fold: bool,
) -> Dict[_MemoKey, _Task]:
    tg = _TaskGraph(pipeline)
    held_out: Optional[str]
    for step in pipeline._find_sink_nodes():
        for held_out in folds:
            for idx in range(n_batches_per_fold):
                tg.find_or_create(
                    _ApplyTask, tg.step_ids[step], (_batch_id(held_out, idx),), held_out
                )
    while len(tg.fresh_tasks) > 0:
        task = tg.fresh_tasks.pop()
        if isinstance(task, _TrainTask):
            if len(task.batch_ids) == 1:
                for pred_step_id in tg.step_id_preds[task.step_id]:
                    if pred_step_id == _DUMMY_INPUT_STEP:
                        held_out = None
                    elif same_fold:
                        held_out = task.held_out
                    else:
                        held_out = _get_fold(task.batch_ids[0])
                    task.preds.append(
                        tg.find_or_create(
                            _ApplyTask, pred_step_id, task.batch_ids, held_out
                        )
                    )
            else:
                if tg.step_id_preds[task.step_id] == [_DUMMY_INPUT_STEP]:
                    held_out = None
                else:
                    if task.held_out is None:
                        hofs = set(folds) - set(_get_fold(b) for b in task.batch_ids)
                        assert len(hofs) == 1, hofs
                        held_out = next(iter(hofs))
                    else:
                        held_out = task.held_out
                if is_associative(pipeline.steps_list()[task.step_id]):
                    if not same_fold:
                        held_out = None
                    for batch_id in task.batch_ids:
                        task.preds.append(
                            tg.find_or_create(
                                _TrainTask, task.step_id, (batch_id,), held_out
                            )
                        )
                elif is_incremental(pipeline.steps_list()[task.step_id]):
                    task.preds.append(
                        tg.find_or_create(
                            _TrainTask, task.step_id, task.batch_ids[:-1], held_out
                        )
                    )
                    for pred_step_id in tg.step_id_preds[task.step_id]:
                        if pred_step_id != _DUMMY_INPUT_STEP and not same_fold:
                            held_out = _get_fold(task.batch_ids[0])
                        task.preds.append(
                            tg.find_or_create(
                                _ApplyTask, pred_step_id, task.batch_ids[-1:], held_out
                            )
                        )
        if isinstance(task, _ApplyTask) and task.step_id != _DUMMY_INPUT_STEP:
            task.preds.append(
                tg.find_or_create(
                    _TrainTask,
                    task.step_id,
                    tuple(
                        _batch_id(fold, idx)
                        for fold in folds
                        if fold != task.held_out
                        for idx in range(n_batches_per_fold)
                    ),
                    None,
                )
            )
            for pred_step_id in tg.step_id_preds[task.step_id]:
                task.preds.append(
                    tg.find_or_create(
                        _ApplyTask,
                        pred_step_id,
                        task.batch_ids,
                        None if pred_step_id == _DUMMY_INPUT_STEP else task.held_out,
                    )
                )
        for pred_task in task.preds:
            pred_task.succs.append(task)
    return tg.all_tasks


def _run_tasks(
    tasks: Dict[_MemoKey, _Task],
    pipeline: TrainablePipeline[TrainableIndividualOp],
    batches: Sequence[_Batch],
    unique_class_labels: List[Union[str, int, float]],
    all_batch_ids: Tuple[str, ...],
    prio: Prio,
    verbose: int,
    call_depth: int = 1,
) -> None:
    for task in tasks.values():
        assert task.status is _TaskStatus.FRESH
        if task.step_id == _DUMMY_INPUT_STEP:
            task.status = _TaskStatus.READY
        else:
            task.status = _TaskStatus.WAITING
    batch_id2idx = {batch_id: idx for idx, batch_id in enumerate(all_batch_ids)}
    assert len(batches) == len(all_batch_ids) == len(batch_id2idx)
    ready_keys = {k for k, t in tasks.items() if t.status is _TaskStatus.READY}

    def find_task(task_class: Type["_Task"], task_list: List[_Task]) -> _Task:
        return next(t for t in task_list if isinstance(t, task_class))

    def mark_done(task: _Task) -> None:
        if task.status is _TaskStatus.DONE:
            return
        if task.status is _TaskStatus.READY:
            ready_keys.remove(task.memo_key())
        task.status = _TaskStatus.DONE
        for succ in task.succs:
            if succ.status is _TaskStatus.WAITING:
                if all(p.status is _TaskStatus.DONE for p in succ.preds):
                    succ.status = _TaskStatus.READY
                    ready_keys.add(succ.memo_key())
        for pred in task.preds:
            if all(s.status is _TaskStatus.DONE for s in pred.succs):
                mark_done(pred)

    seq_id = 0
    while len(ready_keys) > 0:
        if verbose >= 3:
            _visualize_tasks(tasks, pipeline, prio, call_depth + 1)
        task = tasks[min(ready_keys, key=lambda k: prio.task_priority(tasks[k]))]
        operation = task.get_operation(pipeline)
        if operation is _Operation.SCAN:
            assert isinstance(task, _ApplyTask)
            assert len(task.batch_ids) == 1 and len(task.preds) == 0
            task.batch = batches[batch_id2idx[task.batch_ids[0]]]
        elif operation in [_Operation.TRANSFORM, _Operation.PREDICT]:
            assert isinstance(task, _ApplyTask)
            assert len(task.batch_ids) == 1 and len(task.preds) == 2
            train_pred = cast(_TrainTask, find_task(_TrainTask, task.preds))
            trained = train_pred.get_trained(pipeline)
            apply_pred = cast(_ApplyTask, find_task(_ApplyTask, task.preds))
            assert apply_pred.batch is not None
            input_X, input_y = apply_pred.batch
            if operation is _Operation.TRANSFORM:
                task.batch = trained.transform(input_X), input_y
            else:
                task.batch = input_X, pd.Series(trained.predict(input_X))
        elif operation is _Operation.FIT:
            assert isinstance(task, _TrainTask)
            assert len(task.batch_ids) == len(task.preds) == 1
            assert isinstance(task.preds[0], _ApplyTask)
            assert task.preds[0].batch is not None
            trainable = pipeline.steps_list()[task.step_id]
            input_X, input_y = task.preds[0].batch
            task.trained = trainable.fit(input_X, input_y)
        elif operation is _Operation.PARTIAL_FIT:
            assert isinstance(task, _TrainTask)
            assert len(task.preds) in [1, 2]
            if len(task.preds) == 1:
                trainee = pipeline.steps_list()[task.step_id]
            else:
                train_pred = cast(_TrainTask, find_task(_TrainTask, task.preds))
                trainee = train_pred.get_trained(pipeline)
            apply_pred = cast(_ApplyTask, find_task(_ApplyTask, task.preds))
            assert apply_pred.batch is not None
            input_X, input_y = apply_pred.batch
            if trainee.is_supervised():
                task.trained = trainee.partial_fit(
                    input_X, input_y, classes=unique_class_labels
                )
            else:
                task.trained = trainee.partial_fit(input_X, input_y)
        elif operation is _Operation.LIFT:
            assert isinstance(task, _TrainTask)
            assert len(task.batch_ids) == len(task.preds) == 1
            assert isinstance(task.preds[0], _ApplyTask)
            assert task.preds[0].batch is not None
            trainable = pipeline.steps_list()[task.step_id]
            input_X, input_y = task.preds[0].batch
            hyperparams = trainable.impl._hyperparams
            task.lifted = trainable.impl._lift(input_X, hyperparams)
        elif operation is _Operation.COMBINE:
            assert isinstance(task, _TrainTask)
            assert len(task.batch_ids) > 1
            assert len(task.preds) == len(task.batch_ids)
            assert all(isinstance(p, _TrainTask) for p in task.preds)
            trainable = pipeline.steps_list()[task.step_id]
            lifteds = (cast(_TrainTask, p).lifted for p in task.preds)
            task.lifted = functools.reduce(trainable.impl._combine, lifteds)
        else:
            assert False, operation
        task.seq_id = seq_id
        seq_id += 1
        mark_done(task)
    if verbose >= 2:
        _visualize_tasks(tasks, pipeline, prio, call_depth + 1)


def mockup_data_loader(
    X: pd.DataFrame, y: pd.Series, n_splits: int
) -> Sequence[_Batch]:
    if n_splits == 1:
        return [(X, y)]
    cv = sklearn.model_selection.StratifiedKFold(n_splits)
    estimator = sklearn.tree.DecisionTreeClassifier()
    result = [
        lale.helpers.split_with_schemas(estimator, X, y, test, train)
        for train, test in cv.split(X, y)
    ]
    return result


def fit_with_batches(
    pipeline: TrainablePipeline[TrainableIndividualOp],
    batches: Sequence[_Batch],
    unique_class_labels: List[Union[str, int, float]],
    prio: Prio,
    incremental: bool,
    verbose: int,
) -> TrainedPipeline[TrainedIndividualOp]:
    all_batch_ids = tuple(_batch_id("d", idx) for idx in range(len(batches)))
    tasks = _create_tasks_batching(pipeline, all_batch_ids, incremental)
    if verbose >= 3:
        _visualize_tasks(tasks, pipeline, prio, call_depth=2)
    _run_tasks(
        tasks,
        pipeline,
        batches,
        unique_class_labels,
        all_batch_ids,
        prio,
        verbose,
        call_depth=2,
    )

    def get_trained_step(step_id: int) -> TrainedIndividualOp:
        task = cast(_TrainTask, tasks[(_TrainTask, step_id, all_batch_ids, None)])
        return task.get_trained(pipeline)

    step_map = {
        old_step: get_trained_step(step_id)
        for step_id, old_step in enumerate(pipeline.steps_list())
    }
    trained_edges = [(step_map[x], step_map[y]) for x, y in pipeline.edges()]
    result = TrainedPipeline(
        list(step_map.values()), trained_edges, ordered=True, _lale_trained=True
    )
    return result


def cross_val_score(
    pipeline: TrainablePipeline[TrainableIndividualOp],
    batches: Sequence[_Batch],
    unique_class_labels: List[Union[str, int, float]],
    prio: Prio,
    scoring: Callable[[pd.Series, pd.Series], float],
    n_folds: int,
    same_fold: bool,
    verbose: int,
) -> List[float]:
    folds = [chr(ord("d") + i) for i in range(n_folds)]
    n_batches_per_fold = int(len(batches) / n_folds)
    assert n_folds * n_batches_per_fold == len(batches)
    all_batch_ids = tuple(
        _batch_id(fold, idx) for fold in folds for idx in range(n_batches_per_fold)
    )
    tasks = _create_tasks_cross_val(pipeline, folds, n_batches_per_fold, same_fold)
    if verbose >= 3:
        _visualize_tasks(tasks, pipeline, prio, call_depth=2)
    _run_tasks(
        tasks,
        pipeline,
        batches,
        unique_class_labels,
        all_batch_ids,
        prio,
        verbose,
        call_depth=2,
    )
    last_step_id = len(pipeline.steps_list()) - 1

    def predictions(fold: str) -> pd.Series:
        return pd.concat(
            tasks[(_ApplyTask, last_step_id, (_batch_id(fold, idx),), fold)].batch[1]  # type: ignore
            for idx in range(n_batches_per_fold)
        )

    batch_id2idx = {batch_id: idx for idx, batch_id in enumerate(all_batch_ids)}

    def labels(fold: str) -> pd.Series:
        return pd.concat(
            batches[batch_id2idx[_batch_id(fold, idx)]][1]
            for idx in range(n_batches_per_fold)
        )

    result = [scoring(labels(fold), predictions(fold)) for fold in folds]
    return result
