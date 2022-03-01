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
import pathlib
import sys
import tempfile
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterable, List, Optional, Tuple, Type, Union, cast

import graphviz
import pandas as pd
import sklearn.model_selection
import sklearn.tree

import lale.helpers
import lale.json_operator
import lale.pretty_print
from lale.operators import (
    TrainableIndividualOp,
    TrainablePipeline,
    TrainedIndividualOp,
    TrainedPipeline,
)

from .metrics import MetricMonoid, MetricMonoidFactory
from .monoid import Monoid, MonoidFactory

_BatchStatus = enum.Enum("BatchStatus", "RESIDENT SPILLED")

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
    return is_pretrained(op) or isinstance(op.impl, MonoidFactory)


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

    def __init__(
        self, step_id: int, batch_ids: Tuple[str, ...], held_out: Optional[str]
    ):
        self.step_id = step_id
        self.batch_ids = batch_ids
        self.held_out = held_out
        self.status = _TaskStatus.FRESH
        self.preds = []
        self.succs = []
        self.deletable_output = True

    @abstractmethod
    def get_operation(
        self, pipeline: TrainablePipeline[TrainableIndividualOp]
    ) -> _Operation:
        pass

    def memo_key(self) -> _MemoKey:
        return type(self), self.step_id, self.batch_ids, self.held_out


class _TrainTask(_Task):
    monoid: Optional[Monoid]
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
            elif trainable.has_method("from_monoid"):
                self.trained._impl.from_monoid(self.monoid)
            else:
                assert False, self.trained
        return self.trained


_RawBatch = Tuple[pd.DataFrame, pd.Series]


class _Batch:
    X: Union[pd.DataFrame, pathlib.Path]
    y: Union[pd.Series, pathlib.Path]

    def __init__(self, X: pd.DataFrame, y: pd.Series, task: Optional["_ApplyTask"]):
        self.X = X
        self.y = y
        self.task = task
        self.size = X.size + y.size

    def spill(self, spill_dir: pathlib.Path) -> None:
        assert self.status == _BatchStatus.RESIDENT and self.task is not None
        assert len(self.task.batch_ids) == 1, self.task.batch_ids
        batch_id = self.task.batch_ids[0]
        suffix = f"{self.task.step_id}_{batch_id}_{self.task.held_out}.pkl"
        name_X = spill_dir / ("X_" + suffix)
        name_y = spill_dir / ("y_" + suffix)
        cast(pd.DataFrame, self.X).to_pickle(name_X)
        cast(pd.Series, self.y).to_pickle(name_y)
        self.X, self.y = name_X, name_y

    def load_spilled(self) -> None:
        assert isinstance(self.X, pathlib.Path) and isinstance(self.y, pathlib.Path)
        data_X, data_y = pd.read_pickle(self.X), pd.read_pickle(self.y)
        self.X, self.y = data_X, data_y

    def delete_if_spilled(self) -> None:
        if isinstance(self.X, pathlib.Path) and isinstance(self.y, pathlib.Path):
            pathlib.Path(self.X).unlink()
            pathlib.Path(self.y).unlink()

    @property
    def Xy(self) -> _RawBatch:
        assert isinstance(self.X, pd.DataFrame) and isinstance(self.y, pd.Series)
        return self.X, self.y

    @property
    def status(self) -> _BatchStatus:
        if isinstance(self.X, pd.DataFrame) and isinstance(self.y, pd.Series):
            return _BatchStatus.RESIDENT
        if isinstance(self.X, pathlib.Path) and isinstance(self.y, pathlib.Path):
            return _BatchStatus.SPILLED
        assert False, (type(self.X), type(self.y))


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

    def batch_priority(self, batch: _Batch) -> Any:  # prefer to keep resident if lower
        assert batch.task is not None
        return min(
            (
                self.task_priority(s)
                for s in batch.task.succs
                if s.status in [_TaskStatus.READY, _TaskStatus.WAITING]
            ),
            default=self.bottom(),
        )

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


class PrioResourceAware(Prio):
    arity = 5

    def task_priority(self, task: _Task) -> Any:
        non_res = sum(
            1
            for p in task.preds
            if isinstance(p, _ApplyTask) and p.batch is not None
            if p.batch.status != _BatchStatus.RESIDENT
        )
        result = (
            task.status.value,
            non_res,
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
    trace_id: int = None,
) -> str:
    trace_id_s = "" if trace_id is None else f"{trace_id} "
    operation_s = task.get_operation(pipeline).name.lower()
    step_s = _step_id_to_string(task.step_id, pipeline, cls2label)
    batches_s = ",".join(task.batch_ids)
    held_out_s = "" if task.held_out is None else f"#~{task.held_out}"
    return f"{trace_id_s}{operation_s}{sep}{step_s}({batches_s}){held_out_s}"


class _RunStats:
    _values: Dict[str, float]

    def __init__(self):
        object.__setattr__(
            self,
            "_values",
            {
                "spill_count": 0,
                "load_count": 0,
                "train_count": 0,
                "apply_count": 0,
                "metric_count": 0,
                "train_time": 0,
                "apply_time": 0,
                "metric_time": 0,
                "critical_count": 0,
                "critical_time": 0,
            },
        )

    def __getattr__(self, name: str) -> float:
        if name in self._values:
            return self._values[name]
        raise AttributeError(f"'{name}' not in {self._values.keys()}")

    def __setattr__(self, name: str, value: float) -> None:
        if name in self._values:
            self._values[name] = value
        else:
            raise AttributeError(f"'{name}' not in {self._values.keys()}")

    def __repr__(self) -> str:
        return lale.pretty_print.json_to_string(self._values)


class _TraceRecord:
    def __init__(self, task, time):
        self.task = task
        self.time = time
        if isinstance(task, _ApplyTask):
            assert task.batch is not None
            self.space = task.batch.size
        else:
            self.space = 0  # TODO: size for train tasks and metrics tasks


def _visualize_tasks(
    tasks: Dict[_MemoKey, _Task],
    pipeline: TrainablePipeline[TrainableIndividualOp],
    prio: Prio,
    call_depth: int,
    trace: Optional[List[_TraceRecord]],
) -> None:
    cls2label = lale.json_operator._get_cls2label(call_depth + 1)
    dot = graphviz.Digraph()
    dot.attr("graph", rankdir="LR", nodesep="0.1")
    dot.attr("node", fontsize="11", margin="0.03,0.03", shape="box", height="0.1")
    next_task = min(tasks.values(), key=lambda t: prio.task_priority(t))
    task_key2trace_id: Dict[_MemoKey, int] = {}
    if trace is not None:
        task_key2trace_id = {r.task.memo_key(): i for i, r in enumerate(trace)}
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
        trace_id = task_key2trace_id.get(task.memo_key(), None)
        task_s = _task_to_string(task, pipeline, cls2label, trace_id=trace_id)
        dot.node(task_s, style=style, fillcolor=color)
    for task in tasks.values():
        trace_id = task_key2trace_id.get(task.memo_key(), None)
        task_s = _task_to_string(task, pipeline, cls2label, trace_id=trace_id)
        for succ in task.succs:
            succ_id = task_key2trace_id.get(succ.memo_key(), None)
            succ_s = _task_to_string(succ, pipeline, cls2label, trace_id=succ_id)
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


def _batch_ids_except(
    folds: List[str],
    n_batches_per_fold: int,
    held_out: Optional[str],
) -> Tuple[str, ...]:
    return tuple(
        _batch_id(fold, idx)
        for fold in folds
        if fold != held_out
        for idx in range(n_batches_per_fold)
    )


def _create_tasks_cross_val(
    pipeline: TrainablePipeline[TrainableIndividualOp],
    folds: List[str],
    n_batches_per_fold: int,
    same_fold: bool,
    keep_estimator: bool,
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
    if keep_estimator:
        for step_id in range(len(pipeline.steps_list())):
            for held_out in folds:
                task = tg.find_or_create(
                    _TrainTask,
                    step_id,
                    _batch_ids_except(folds, n_batches_per_fold, held_out),
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
                    _batch_ids_except(folds, n_batches_per_fold, task.held_out),
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


def _analyze_run_trace(stats: _RunStats, trace: List[_TraceRecord]) -> _RunStats:
    memo_key2critical_count: Dict[_MemoKey, int] = {}
    memo_key2critical_time: Dict[_MemoKey, int] = {}
    for record in trace:
        if isinstance(record.task, _TrainTask):
            stats.train_count += 1
            stats.train_time += record.time
        elif isinstance(record.task, _ApplyTask):
            stats.apply_count += 1
            stats.apply_time += record.time
        elif isinstance(record.task, _MetricTask):
            stats.metric_count += 1
            stats.metric_time += record.time
        else:
            assert False, type(record.task)
        critical_count = 1 + max(
            (memo_key2critical_count[p.memo_key()] for p in record.task.preds),
            default=0,
        )
        stats.critical_count = max(critical_count, stats.critical_count)
        memo_key2critical_count[record.task.memo_key()] = critical_count
        critical_time = record.time + max(
            (memo_key2critical_time[p.memo_key()] for p in record.task.preds), default=0
        )
        stats.critical_time = max(critical_time, stats.critical_time)
        memo_key2critical_time[record.task.memo_key()] = critical_time
    return stats


def _run_tasks_inner(
    tasks: Dict[_MemoKey, _Task],
    pipeline: TrainablePipeline[TrainableIndividualOp],
    batches: Iterable[_RawBatch],
    scoring: Optional[MetricMonoidFactory],
    unique_class_labels: List[Union[str, int, float]],
    all_batch_ids: Tuple[str, ...],
    max_resident: int,
    prio: Prio,
    verbose: int,
    call_depth: int,
    spill_dir: Optional[pathlib.Path],
) -> None:
    for task in tasks.values():
        assert task.status is _TaskStatus.FRESH
        if len(task.preds) == 0:
            task.status = _TaskStatus.READY
        else:
            task.status = _TaskStatus.WAITING
    ready_keys = {k for k, t in tasks.items() if t.status is _TaskStatus.READY}
    stats = _RunStats()

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
                    if task.batch is not None:
                        task.batch.delete_if_spilled()
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
        if isinstance(task, _TrainTask):
            if task.get_operation(pipeline) is _Operation.TO_MONOID:
                if task.monoid is not None and task.monoid.is_absorbing:

                    def is_moot(task2):  # same modulo batch_ids
                        type1, step1, _, hold1 = task.memo_key()
                        type2, step2, _, hold2 = task2.memo_key()
                        return type1 == type2 and step1 == step2 and hold1 == hold2

                    task_monoid = task.monoid  # prevent accidental None assignment
                    for task2 in tasks.values():
                        if task2.status is not _TaskStatus.DONE and is_moot(task2):
                            assert isinstance(task2, _TrainTask)
                            task2.monoid = task_monoid
                            mark_done(task2)

    def ensure_space(amount_needed: int) -> None:
        resident_batches = [
            t.batch
            for t in tasks.values()
            if isinstance(t, _ApplyTask) and t.batch is not None
            if t.batch.status == _BatchStatus.RESIDENT
        ]
        resident_batches.sort(key=lambda b: prio.batch_priority(b))
        resident_batches_size = len(
            resident_batches
        )  # sum(b.size for b in resident_batches)
        while resident_batches_size + amount_needed > max_resident:
            batch = resident_batches.pop()
            assert batch.status == _BatchStatus.RESIDENT and batch.task is not None
            assert spill_dir is not None, max_resident
            batch.spill(spill_dir)
            stats.spill_count += 1
            if verbose >= 2:
                task_string = _task_to_string(batch.task, pipeline, sep=" ")
                print(f"spill {task_string} {batch.X} {batch.y.name}")
            resident_batches_size -= 1  # batch.size

    def load_batch(batch: _Batch) -> None:
        assert batch.status == _BatchStatus.SPILLED and batch.task is not None
        ensure_space(1)  # ensure_space(batch.size)
        batch.load_spilled()
        stats.load_count += 1
        if verbose >= 2:
            print(f"load {_task_to_string(batch.task, pipeline, sep=' ')}")

    def load_input_batches(task: _Task) -> None:
        for batch in (p.batch for p in task.preds if isinstance(p, _ApplyTask)):
            assert batch is not None
            if batch.status == _BatchStatus.SPILLED:
                load_batch(batch)

    def assert_input_batches(task: _Task) -> None:
        batches = (p.batch for p in task.preds if isinstance(p, _ApplyTask))
        assert all(b is not None and b.status == _BatchStatus.RESIDENT for b in batches)

    trace: Optional[List[_TraceRecord]] = [] if verbose >= 2 else None
    batches_iterator = iter(batches)
    while len(ready_keys) > 0:
        if verbose >= 3:
            _visualize_tasks(tasks, pipeline, prio, call_depth + 1, trace)
        task = tasks[min(ready_keys, key=lambda k: prio.task_priority(tasks[k]))]
        operation = task.get_operation(pipeline)
        start_time = time.time() if verbose >= 2 else float("nan")
        if operation is _Operation.SCAN:
            assert isinstance(task, _ApplyTask)
            assert len(task.batch_ids) == 1 and len(task.preds) == 0
            ensure_space(1)  # ensure_space(output batch size)
            X, y = next(batches_iterator)
            task.batch = _Batch(X, y, task)
        elif operation in [_Operation.TRANSFORM, _Operation.PREDICT]:
            assert isinstance(task, _ApplyTask)
            assert len(task.batch_ids) == 1
            train_pred = cast(_TrainTask, find_task(_TrainTask, task.preds))
            trained = train_pred.get_trained(pipeline)
            apply_preds = find_task(_ApplyTask, task.preds)
            load_input_batches(task)
            if isinstance(apply_preds, _Task):
                apply_pred = cast(_ApplyTask, apply_preds)
                assert apply_pred.batch is not None
                input_X, input_y = apply_pred.batch.Xy
            else:  # a list of tasks
                assert not any(apply_pred.batch is None for apply_pred in apply_preds)  # type: ignore
                input_X = [pred.batch.X for pred in apply_preds]  # type: ignore
                # The assumption is that input_y is not changed by the preds, so we can
                # use it from any one of them.
                input_y = apply_preds[0].batch.y  # type: ignore
            ensure_space(1)  # ensure_space(output batch size)
            assert_input_batches(task)
            if operation is _Operation.TRANSFORM:
                task.batch = _Batch(trained.transform(input_X), input_y, task)
            else:
                y_pred = trained.predict(input_X)
                if not isinstance(y_pred, pd.Series):
                    y_pred = pd.Series(y_pred, input_y.index, input_y.dtype, "y_pred")
                task.batch = _Batch(input_X, y_pred, task)
        elif operation is _Operation.FIT:
            assert isinstance(task, _TrainTask)
            assert all(isinstance(p, _ApplyTask) for p in task.preds)
            assert not any(cast(_ApplyTask, p).batch is None for p in task.preds)
            trainable = pipeline.steps_list()[task.step_id]
            if is_pretrained(trainable):
                assert len(task.preds) == 0
                task.trained = cast(TrainedIndividualOp, trainable)
            else:
                load_input_batches(task)
                assert_input_batches(task)
                if len(task.preds) == 1:
                    input_X, input_y = task.preds[0].batch.Xy  # type: ignore
                else:
                    assert not is_incremental(trainable)
                    input_X = pd.concat([p.batch.X for p in task.preds])  # type: ignore
                    input_y = pd.concat([p.batch.y for p in task.preds])  # type: ignore
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
            load_input_batches(task)
            assert_input_batches(task)
            input_X, input_y = apply_pred.batch.Xy
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
            load_input_batches(task)
            assert_input_batches(task)
            if isinstance(task, _TrainTask):
                assert len(task.preds) == 1
                trainable = pipeline.steps_list()[task.step_id]
                input_X, input_y = task.preds[0].batch.Xy  # type: ignore
                task.monoid = trainable.impl.to_monoid((input_X, input_y))
            elif isinstance(task, _MetricTask):
                assert len(task.preds) == 2
                assert task.preds[0].step_id == _DUMMY_INPUT_STEP
                assert scoring is not None
                y_true = task.preds[0].batch.y  # type: ignore
                y_pred = task.preds[1].batch.y  # type: ignore
                task.score = scoring.to_monoid((y_true, y_pred))
            else:
                assert False, type(task)
        elif operation is _Operation.COMBINE:
            assert len(task.batch_ids) > 1
            assert len(task.preds) == len(task.batch_ids)
            load_input_batches(task)
            assert_input_batches(task)
            if isinstance(task, _TrainTask):
                assert all(isinstance(p, _TrainTask) for p in task.preds)
                trainable = pipeline.steps_list()[task.step_id]
                monoids = (cast(_TrainTask, p).monoid for p in task.preds)
                task.monoid = functools.reduce(lambda x, y: x.combine(y), monoids)  # type: ignore
            elif isinstance(task, _MetricTask):
                scores = (cast(_MetricTask, p).score for p in task.preds)
                task.score = functools.reduce(lambda x, y: x.combine(y), scores)  # type: ignore
            else:
                assert False, type(task)
        else:
            assert False, operation
        if verbose >= 2:
            finish_time = time.time()
            assert trace is not None
            trace.append(_TraceRecord(task, finish_time - start_time))
        mark_done(task)
    if verbose >= 2:
        _visualize_tasks(tasks, pipeline, prio, call_depth + 1, trace)
        assert trace is not None
        print(_analyze_run_trace(stats, trace))


def _run_tasks(
    tasks: Dict[_MemoKey, _Task],
    pipeline: TrainablePipeline[TrainableIndividualOp],
    batches: Iterable[_RawBatch],
    scoring: Optional[MetricMonoidFactory],
    unique_class_labels: List[Union[str, int, float]],
    all_batch_ids: Tuple[str, ...],
    max_resident: Optional[int],
    prio: Prio,
    verbose: int,
    call_depth: int,
) -> None:
    if max_resident is None:
        _run_tasks_inner(
            tasks,
            pipeline,
            batches,
            scoring,
            unique_class_labels,
            all_batch_ids,
            sys.maxsize,
            prio,
            verbose,
            call_depth + 1,
            None,
        )
    else:
        with tempfile.TemporaryDirectory() as tmpdirname:
            _run_tasks_inner(
                tasks,
                pipeline,
                batches,
                scoring,
                unique_class_labels,
                all_batch_ids,
                max_resident,
                prio,
                verbose,
                call_depth + 1,
                pathlib.Path(tmpdirname),
            )


def mockup_data_loader(
    X: pd.DataFrame, y: pd.Series, n_splits: int
) -> Iterable[_RawBatch]:
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
    for task in tasks.values():
        # preds form a cycle with succs
        task.preds.clear()
        task.succs.clear()
        # tasks form a cycle with batches
        if isinstance(task, _ApplyTask) and task.batch is not None:
            task.batch.task = None
            task.batch = None
    tasks.clear()


def _extract_trained_pipeline(
    pipeline: TrainablePipeline[TrainableIndividualOp],
    folds: List[str],
    n_batches_per_fold: int,
    tasks: Dict[_MemoKey, _Task],
    held_out: Optional[str],
) -> TrainedPipeline:
    batch_ids = _batch_ids_except(folds, n_batches_per_fold, held_out)

    def extract_trained_step(step_id: int) -> TrainedIndividualOp:
        task = cast(_TrainTask, tasks[(_TrainTask, step_id, batch_ids, held_out)])
        return task.get_trained(pipeline)

    step_map = {
        old_step: extract_trained_step(step_id)
        for step_id, old_step in enumerate(pipeline.steps_list())
    }
    trained_edges = [(step_map[x], step_map[y]) for x, y in pipeline.edges()]
    result = TrainedPipeline(
        list(step_map.values()), trained_edges, ordered=True, _lale_trained=True
    )
    return result


def fit_with_batches(
    pipeline: TrainablePipeline[TrainableIndividualOp],
    batches: Iterable[_RawBatch],
    n_batches: int,
    unique_class_labels: List[Union[str, int, float]],
    max_resident: Optional[int],
    prio: Prio,
    incremental: bool,
    verbose: int,
) -> TrainedPipeline[TrainedIndividualOp]:
    all_batch_ids = tuple(_batch_id("d", idx) for idx in range(n_batches))
    tasks = _create_tasks_batching(pipeline, all_batch_ids, incremental)
    if verbose >= 3:
        _visualize_tasks(tasks, pipeline, prio, call_depth=2, trace=None)
    _run_tasks(
        tasks,
        pipeline,
        batches,
        None,
        unique_class_labels,
        all_batch_ids,
        max_resident,
        prio,
        verbose,
        call_depth=2,
    )
    trained_pipeline = _extract_trained_pipeline(
        pipeline, ["d"], n_batches, tasks, None
    )
    _clear_tasks_dict(tasks)
    return trained_pipeline


def _extract_scores(
    pipeline: TrainablePipeline[TrainableIndividualOp],
    folds: List[str],
    n_batches_per_fold: int,
    scoring: MetricMonoidFactory,
    tasks: Dict[_MemoKey, _Task],
) -> List[float]:
    def extract_score(held_out: str) -> float:
        batch_ids = tuple(_batch_id(held_out, idx) for idx in range(n_batches_per_fold))
        task = tasks[(_MetricTask, _DUMMY_SCORE_STEP, batch_ids, held_out)]
        assert isinstance(task, _MetricTask) and task.score is not None
        return scoring.from_monoid(task.score)

    scores = [extract_score(held_out) for held_out in folds]
    return scores


def cross_val_score(
    pipeline: TrainablePipeline[TrainableIndividualOp],
    batches: Iterable[_RawBatch],
    n_batches: int,
    n_folds: int,
    n_batches_per_fold: int,
    scoring: MetricMonoidFactory,
    unique_class_labels: List[Union[str, int, float]],
    max_resident: Optional[int],
    prio: Prio,
    same_fold: bool,
    verbose: int,
) -> List[float]:
    assert n_batches == n_folds * n_batches_per_fold
    folds = [chr(ord("d") + i) for i in range(n_folds)]
    all_batch_ids = tuple(
        _batch_id(fold, idx) for fold in folds for idx in range(n_batches_per_fold)
    )
    tasks = _create_tasks_cross_val(
        pipeline, folds, n_batches_per_fold, same_fold, False
    )
    if verbose >= 3:
        _visualize_tasks(tasks, pipeline, prio, call_depth=2, trace=None)
    _run_tasks(
        tasks,
        pipeline,
        batches,
        scoring,
        unique_class_labels,
        all_batch_ids,
        max_resident,
        prio,
        verbose,
        call_depth=2,
    )
    scores = _extract_scores(pipeline, folds, n_batches_per_fold, scoring, tasks)
    _clear_tasks_dict(tasks)
    return scores


def cross_validate(
    pipeline: TrainablePipeline[TrainableIndividualOp],
    batches: Iterable[_RawBatch],
    n_batches: int,
    n_folds: int,
    n_batches_per_fold: int,
    scoring: MetricMonoidFactory,
    unique_class_labels: List[Union[str, int, float]],
    max_resident: Optional[int],
    prio: Prio,
    same_fold: bool,
    return_estimator: bool,
    verbose: int,
) -> Dict[str, Union[List[float], List[TrainedPipeline]]]:
    assert n_batches == n_folds * n_batches_per_fold
    folds = [chr(ord("d") + i) for i in range(n_folds)]
    all_batch_ids = tuple(
        _batch_id(fold, idx) for fold in folds for idx in range(n_batches_per_fold)
    )
    tasks = _create_tasks_cross_val(
        pipeline, folds, n_batches_per_fold, same_fold, return_estimator
    )
    if verbose >= 3:
        _visualize_tasks(tasks, pipeline, prio, call_depth=2, trace=None)
    _run_tasks(
        tasks,
        pipeline,
        batches,
        scoring,
        unique_class_labels,
        all_batch_ids,
        max_resident,
        prio,
        verbose,
        call_depth=2,
    )
    result: Dict[str, Union[List[float], List[TrainedPipeline]]] = {}
    result["test_score"] = _extract_scores(
        pipeline, folds, n_batches_per_fold, scoring, tasks
    )
    if return_estimator:
        result["estimator"] = [
            _extract_trained_pipeline(
                pipeline, folds, n_batches_per_fold, tasks, held_out
            )
            for held_out in folds
        ]
    _clear_tasks_dict(tasks)
    return result
