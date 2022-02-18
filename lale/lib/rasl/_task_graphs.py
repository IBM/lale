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
import sys
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union, cast

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

from ._metrics import MetricMonoid, MetricMonoidFactory
from ._monoid import Monoid, MonoidFactory

_TaskStatus = enum.Enum("_TaskStatus", "FRESH READY WAITING DONE")

_Operation = enum.Enum(
    "_Operation", "SCAN TRANSFORM PREDICT FIT PARTIAL_FIT TO_MONOID COMBINE"
)

_DUMMY_INPUT_STEP = -1

_DUMMY_SCORE_STEP = sys.maxsize


def is_pretrained(op: TrainableIndividualOp) -> bool:
    return isinstance(op, TrainedIndividualOp) and (
        op.is_frozen_trained() or not hasattr(op.impl, "fit")
    )


def is_incremental(op: TrainableIndividualOp) -> bool:
    return op.has_method("partial_fit") or is_pretrained(op)


def is_associative(op: TrainableIndividualOp) -> bool:
    return (
        op.has_method("_lift")
        and op.has_method("_combine")
        or is_pretrained(op)
        or isinstance(op.impl, MonoidFactory)
    )


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
        self.deletable_output = True

    @abstractmethod
    def get_operation(
        self, pipeline: TrainablePipeline[TrainableIndividualOp]
    ) -> _Operation:
        pass

    def memo_key(self) -> _MemoKey:
        return type(self), self.step_id, self.batch_ids, self.held_out


class _TrainTask(_Task):
    monoid: Optional[Union[Tuple[Any, ...], Monoid]]
    trained: Optional[TrainedIndividualOp]

    def __init__(self, step_id: int, batch_ids: Tuple[str, ...], held_out: str):
        super().__init__(step_id, batch_ids, held_out)
        self.monoid = None
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
            if len(self.batch_ids) == 1:
                return _Operation.TO_MONOID
            return _Operation.COMBINE
        return _Operation.PARTIAL_FIT

    def get_trained(
        self, pipeline: TrainablePipeline[TrainableIndividualOp]
    ) -> TrainedIndividualOp:
        if self.trained is None:
            assert self.monoid is not None
            trainable = pipeline.steps_list()[self.step_id]
            self.trained = trainable.convert_to_trained()
            hyperparams = trainable.impl._hyperparams
            self.trained._impl = trainable._impl_class()(**hyperparams)
            if trainable.has_method("_set_fit_attributes"):
                self.trained._impl._set_fit_attributes(self.monoid)
            elif trainable.has_method("_from_monoid"):
                self.trained._impl._from_monoid(self.monoid)
            else:
                assert False, self.trained
        return self.trained


_Batch = Tuple[pd.DataFrame, pd.Series]


class _ApplyTask(_Task):
    batch: Optional[_Batch]

    def __init__(self, step_id: int, batch_ids: Tuple[str, ...], held_out: str):
        super().__init__(step_id, batch_ids, held_out)
        self.batch = None

    def get_operation(self, pipeline: TrainablePipeline) -> _Operation:
        if self.step_id == _DUMMY_INPUT_STEP:
            return _Operation.SCAN
        step = pipeline.steps_list()[self.step_id]
        return _Operation.TRANSFORM if step.is_transformer() else _Operation.PREDICT


class _MetricTask(_Task):
    score: Optional[MetricMonoid]

    def __init__(self, step_id: int, batch_ids: Tuple[str, ...], held_out: str):
        super().__init__(step_id, batch_ids, held_out)
        self.score = None

    def get_operation(self, pipeline: TrainablePipeline) -> _Operation:
        if len(self.batch_ids) == 1:
            return _Operation.TO_MONOID
        return _Operation.COMBINE


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


def _step_id_to_string(
    step_id: int,
    pipeline: TrainablePipeline,
    cls2label: Dict[str, str] = {},
) -> str:
    if step_id == _DUMMY_INPUT_STEP:
        return "INP"
    if step_id == _DUMMY_SCORE_STEP:
        return "SCR"
    step = pipeline.steps_list()[step_id]
    cls = step.class_name()
    return cls2label[cls] if cls in cls2label else step.name()


def _task_to_string(
    task: _Task,
    pipeline: TrainablePipeline,
    cls2label: Dict[str, str] = {},
    sep: str = "\n",
) -> str:
    seq_id_s = "" if task.seq_id is None else f"{task.seq_id} "
    operation_s = task.get_operation(pipeline).name.lower()
    step_s = _step_id_to_string(task.step_id, pipeline, cls2label)
    batches_s = ",".join(task.batch_ids)
    held_out_s = "" if task.held_out is None else f"#~{task.held_out}"
    return f"{seq_id_s}{operation_s}{sep}{step_s}({batches_s}){held_out_s}"


def _visualize_tasks(
    tasks: Dict[_MemoKey, _Task],
    pipeline: TrainablePipeline[TrainableIndividualOp],
    prio: Prio,
    call_depth: int,
) -> None:
    cls2label = lale.json_operator._get_cls2label(call_depth + 1)
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
        # https://www.graphviz.org/doc/info/shapes.html
        if isinstance(task, _TrainTask):
            style = "filled,rounded"
        elif isinstance(task, _ApplyTask):
            style = "filled"
        elif isinstance(task, _MetricTask):
            style = "filled,diagonals"
        else:
            assert False, type(task)
        task_s = _task_to_string(task, pipeline, cls2label)
        dot.node(task_s, style=style, fillcolor=color)
    for task in tasks.values():
        task_s = _task_to_string(task, pipeline, cls2label)
        for succ in task.succs:
            succ_s = _task_to_string(succ, pipeline, cls2label)
            dot.edge(task_s, succ_s)

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
    for step_id in range(len(pipeline.steps_list())):
        task = tg.find_or_create(_TrainTask, step_id, all_batch_ids, None)
        task.deletable_output = False
    while len(tg.fresh_tasks) > 0:
        task = tg.fresh_tasks.pop()
        if isinstance(task, _TrainTask):
            step = pipeline.steps_list()[task.step_id]
            if is_pretrained(step):
                pass
            elif len(task.batch_ids) == 1:
                for pred_step_id in tg.step_id_preds[task.step_id]:
                    task.preds.append(
                        tg.find_or_create(
                            _ApplyTask, pred_step_id, task.batch_ids, None
                        )
                    )
            else:
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
                    for pred_step_id in tg.step_id_preds[task.step_id]:
                        for batch_id in task.batch_ids:
                            task.preds.append(
                                tg.find_or_create(
                                    _ApplyTask, pred_step_id, (batch_id,), None
                                )
                            )
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
    for held_out in folds:
        task = tg.find_or_create(
            _MetricTask,
            _DUMMY_SCORE_STEP,
            tuple(_batch_id(held_out, idx) for idx in range(n_batches_per_fold)),
            held_out,
        )
        task.deletable_output = False
    while len(tg.fresh_tasks) > 0:
        task = tg.fresh_tasks.pop()
        if isinstance(task, _TrainTask):
            step = pipeline.steps_list()[task.step_id]
            if is_pretrained(step):
                pass
            elif len(task.batch_ids) == 1:
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
                if is_associative(step):
                    if not same_fold:
                        held_out = None
                    for batch_id in task.batch_ids:
                        task.preds.append(
                            tg.find_or_create(
                                _TrainTask, task.step_id, (batch_id,), held_out
                            )
                        )
                elif is_incremental(step):
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
                else:
                    for pred_step_id in tg.step_id_preds[task.step_id]:
                        if pred_step_id != _DUMMY_INPUT_STEP and not same_fold:
                            held_out = _get_fold(task.batch_ids[0])
                        for batch_id in task.batch_ids:
                            task.preds.append(
                                tg.find_or_create(
                                    _ApplyTask, pred_step_id, (batch_id,), held_out
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
        if isinstance(task, _MetricTask):
            if len(task.batch_ids) == 1:
                task.preds.append(
                    tg.find_or_create(
                        _ApplyTask, _DUMMY_INPUT_STEP, task.batch_ids, None
                    )
                )
                sink = pipeline.get_last()
                assert sink is not None
                task.preds.append(
                    tg.find_or_create(
                        _ApplyTask, tg.step_ids[sink], task.batch_ids, task.held_out
                    )
                )
            else:
                for batch_id in task.batch_ids:
                    task.preds.append(
                        tg.find_or_create(
                            _MetricTask, task.step_id, (batch_id,), task.held_out
                        )
                    )
        for pred_task in task.preds:
            pred_task.succs.append(task)
    return tg.all_tasks


def _run_tasks(
    tasks: Dict[_MemoKey, _Task],
    pipeline: TrainablePipeline[TrainableIndividualOp],
    batches: Iterable[_Batch],
    scoring: Optional[MetricMonoidFactory],
    unique_class_labels: List[Union[str, int, float]],
    all_batch_ids: Tuple[str, ...],
    prio: Prio,
    verbose: int,
    call_depth: int = 1,
) -> None:
    for task in tasks.values():
        assert task.status is _TaskStatus.FRESH
        if len(task.preds) == 0:
            task.status = _TaskStatus.READY
        else:
            task.status = _TaskStatus.WAITING
    ready_keys = {k for k, t in tasks.items() if t.status is _TaskStatus.READY}

    def find_task(
        task_class: Type["_Task"], task_list: List[_Task]
    ) -> Union[_Task, List[_Task]]:
        task_list = [t for t in task_list if isinstance(t, task_class)]
        if len(task_list) == 1:
            return task_list[0]
        else:
            return task_list

    def try_to_delete_output(task: _Task) -> None:
        if task.deletable_output:
            if all(s.status is _TaskStatus.DONE for s in task.succs):
                if isinstance(task, _ApplyTask):
                    task.batch = None
                elif isinstance(task, _TrainTask):
                    task.monoid = None
                    task.trained = None
                elif isinstance(task, _MetricTask):
                    task.score = None
                else:
                    assert False, type(task)

    def mark_done(task: _Task) -> None:
        try_to_delete_output(task)
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
    batches_iterator = iter(batches)
    while len(ready_keys) > 0:
        if verbose >= 3:
            _visualize_tasks(tasks, pipeline, prio, call_depth + 1)
        task = tasks[min(ready_keys, key=lambda k: prio.task_priority(tasks[k]))]
        operation = task.get_operation(pipeline)
        if operation is _Operation.SCAN:
            assert isinstance(task, _ApplyTask)
            assert len(task.batch_ids) == 1 and len(task.preds) == 0
            task.batch = next(batches_iterator)
        elif operation in [_Operation.TRANSFORM, _Operation.PREDICT]:
            assert isinstance(task, _ApplyTask)
            assert len(task.batch_ids) == 1
            train_pred = cast(_TrainTask, find_task(_TrainTask, task.preds))
            trained = train_pred.get_trained(pipeline)
            apply_preds = find_task(_ApplyTask, task.preds)
            if isinstance(apply_preds, _Task):
                apply_pred = cast(_ApplyTask, apply_preds)
                assert apply_pred.batch is not None
                input_X, input_y = apply_pred.batch
            else:  # a list of tasks
                apply_preds = cast(List[_ApplyTask], apply_preds)  # type: ignore
                assert not any(apply_pred.batch is None for apply_pred in apply_preds)  # type: ignore
                input_X = [pred.batch[0] for pred in apply_preds]  # type: ignore
                # The assumption is that input_y is not changed by the preds, so we can
                # use it from any one of them.
                input_y = apply_preds[0].batch[1]  # type: ignore
            if operation is _Operation.TRANSFORM:
                task.batch = trained.transform(input_X), input_y
            else:
                y_pred = trained.predict(input_X)
                if not isinstance(y_pred, pd.Series):
                    y_pred = pd.Series(y_pred, input_y.index, input_y.dtype, "y_pred")
                task.batch = input_X, y_pred
        elif operation is _Operation.FIT:
            assert isinstance(task, _TrainTask)
            assert all(isinstance(p, _ApplyTask) for p in task.preds)
            assert not any(cast(_ApplyTask, p).batch is None for p in task.preds)
            trainable = pipeline.steps_list()[task.step_id]
            if is_pretrained(trainable):
                assert len(task.preds) == 0
                task.trained = cast(TrainedIndividualOp, trainable)
            else:
                if len(task.preds) == 1:
                    input_X, input_y = task.preds[0].batch  # type: ignore
                else:
                    assert not is_incremental(trainable)
                    input_X = pd.concat([p.batch[0] for p in task.preds])  # type: ignore
                    input_y = pd.concat([p.batch[1] for p in task.preds])  # type: ignore
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
        elif operation is _Operation.TO_MONOID:
            assert len(task.batch_ids) == 1
            assert all(isinstance(p, _ApplyTask) for p in task.preds)
            assert all(cast(_ApplyTask, p).batch is not None for p in task.preds)
            if isinstance(task, _TrainTask):
                assert len(task.preds) == 1
                trainable = pipeline.steps_list()[task.step_id]
                input_X, input_y = task.preds[0].batch  # type: ignore
                if trainable.has_method("_lift"):
                    hyperparams = trainable.impl._hyperparams
                    task.monoid = trainable.impl._lift(input_X, hyperparams)
                elif trainable.has_method("_to_monoid"):
                    task.monoid = trainable.impl._to_monoid((input_X, input_y))
                else:
                    assert False, operation
            elif isinstance(task, _MetricTask):
                assert len(task.preds) == 2
                assert task.preds[0].step_id == _DUMMY_INPUT_STEP
                assert scoring is not None
                _, y_true = task.preds[0].batch  # type: ignore
                _, y_pred = task.preds[1].batch  # type: ignore
                task.score = scoring._to_monoid((y_true, y_pred))
            else:
                assert False, type(task)
        elif operation is _Operation.COMBINE:
            assert len(task.batch_ids) > 1
            assert len(task.preds) == len(task.batch_ids)
            if isinstance(task, _TrainTask):
                assert all(isinstance(p, _TrainTask) for p in task.preds)
                trainable = pipeline.steps_list()[task.step_id]
                monoids = (cast(_TrainTask, p).monoid for p in task.preds)
                if trainable.has_method("_combine"):
                    task.monoid = functools.reduce(trainable.impl._combine, monoids)
                elif trainable.has_method("_monoid"):

                    def _combine(x, y):
                        assert isinstance(x, Monoid)
                        return x.combine(y)

                    task.monoid = functools.reduce(_combine, monoids)
                else:
                    assert False, operation
            elif isinstance(task, _MetricTask):
                scores = (cast(_MetricTask, p).score for p in task.preds)
                task.score = functools.reduce(lambda x, y: x.combine(y), scores)  # type: ignore
            else:
                assert False, type(task)
        else:
            assert False, operation
        task.seq_id = seq_id
        seq_id += 1
        mark_done(task)
    if verbose >= 2:
        _visualize_tasks(tasks, pipeline, prio, call_depth + 1)


def mockup_data_loader(
    X: pd.DataFrame, y: pd.Series, n_splits: int
) -> Iterable[_Batch]:
    if n_splits == 1:
        return [(X, y)]
    cv = sklearn.model_selection.KFold(n_splits)
    estimator = sklearn.tree.DecisionTreeClassifier()
    result = (
        lale.helpers.split_with_schemas(estimator, X, y, test, train)
        for train, test in cv.split(X, y)
    )  # generator expression returns object with __iter__() method
    return result


def _clear_tasks_dict(tasks: Dict[_MemoKey, _Task]):
    for task in tasks.values():  # preds form a cycle with succs
        task.preds.clear()
        task.succs.clear()
    tasks.clear()


def fit_with_batches(
    pipeline: TrainablePipeline[TrainableIndividualOp],
    batches: Iterable[_Batch],
    n_batches: int,
    unique_class_labels: List[Union[str, int, float]],
    prio: Prio,
    incremental: bool,
    verbose: int,
) -> TrainedPipeline[TrainedIndividualOp]:
    all_batch_ids = tuple(_batch_id("d", idx) for idx in range(n_batches))
    tasks = _create_tasks_batching(pipeline, all_batch_ids, incremental)
    if verbose >= 3:
        _visualize_tasks(tasks, pipeline, prio, call_depth=2)
    _run_tasks(
        tasks,
        pipeline,
        batches,
        None,
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
    _clear_tasks_dict(tasks)
    return result


def cross_val_score(
    pipeline: TrainablePipeline[TrainableIndividualOp],
    batches: Iterable[_Batch],
    n_batches: int,
    n_folds: int,
    n_batches_per_fold: int,
    scoring: MetricMonoidFactory,
    unique_class_labels: List[Union[str, int, float]],
    prio: Prio,
    same_fold: bool,
    verbose: int,
) -> List[float]:
    assert n_batches == n_folds * n_batches_per_fold
    folds = [chr(ord("d") + i) for i in range(n_folds)]
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
        scoring,
        unique_class_labels,
        all_batch_ids,
        prio,
        verbose,
        call_depth=2,
    )

    def get_score(held_out: str) -> float:
        batches = tuple(_batch_id(held_out, idx) for idx in range(n_batches_per_fold))
        task = tasks[(_MetricTask, _DUMMY_SCORE_STEP, batches, held_out)]
        assert isinstance(task, _MetricTask) and task.score is not None
        return scoring._from_monoid(task.score)

    result = [get_score(held_out) for held_out in folds]
    _clear_tasks_dict(tasks)
    return result
