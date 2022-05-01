# Copyright 2019, 2020, 2021 IBM Corporation
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

import inspect
import logging
from typing import Set

from lale.operators import Operator, clone_op, get_op_from_lale_lib

logger = logging.getLogger(__name__)


def _wrap_operators_in_symtab(symtab, exclude_classes=None, wrapper_modules=None):
    for name, impl in symtab.items():
        if (
            inspect.isclass(impl)
            and not issubclass(impl, Operator)
            and (hasattr(impl, "predict") or hasattr(impl, "transform"))
        ):
            if exclude_classes is not None:
                if name in exclude_classes:
                    continue
            operator = get_op_from_lale_lib(impl, wrapper_modules)
            if operator is None:
                # symtab[name] = make_operator(impl=impl, name=name)
                logger.info(f"Lale:Not wrapping unknown operator:{name}")
            else:
                symtab[name] = clone_op(operator, name)
                if operator.class_name().startswith("lale.lib.autogen"):
                    logger.info(f"Lale:Wrapped autogen operator:{name}")
                else:
                    logger.info(f"Lale:Wrapped known operator:{name}")


def wrap_imported_operators(exclude_classes=None, wrapper_modules=None):
    """Wrap the currently imported operators from the symbol table
    to their lale wrappers.

        Parameters
        ----------
        exclude_classes : [str], optional
            List of class names to exclude from wrapping,
            alias names if they are used while importing.
            by default None
    """
    current_frame = inspect.currentframe()
    assert (
        current_frame is not None
    ), "Try to use inspect.stack()[1][0] to get the calling frame"
    calling_frame = current_frame.f_back
    assert (
        calling_frame is not None
    ), "Try to use inspect.stack()[1][0] to get the calling frame"
    if wrapper_modules is not None:
        wrapper_modules.extend(get_lale_wrapper_modules())
    else:
        wrapper_modules = get_lale_wrapper_modules()
    _wrap_operators_in_symtab(
        calling_frame.f_globals, exclude_classes, wrapper_modules=wrapper_modules
    )
    if calling_frame.f_code.co_name == "<module>":  # for testing with exec()
        _wrap_operators_in_symtab(
            calling_frame.f_locals, exclude_classes, wrapper_modules=wrapper_modules
        )


_lale_wrapper_modules: Set[str] = set()


def register_lale_wrapper_modules(m: str) -> None:
    """Register a module with lale's import system
    so that :meth:`lale.helpers.import_from_sklearn_pipeline` will look for replacement classes in that module.

        Example: (in `__init__.py` file for the module):

    .. code-block:: python

        from lale import register_lale_wrapper_modules

        register_lale_wrapper_modules(__name__)

    Parameters
    ----------
    m : [str]
        The module name
    """
    _lale_wrapper_modules.add(m)


def get_lale_wrapper_modules() -> Set[str]:
    return _lale_wrapper_modules


for m in [
    "lale.lib.sklearn",
    "lale.lib.autoai_libs",
    "lale.lib.xgboost",
    "lale.lib.lightgbm",
    "lale.lib.snapml",
    "autoai_ts_libs.lale",
]:
    register_lale_wrapper_modules(m)
