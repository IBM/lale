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

from lale.operators import Operator, clone_op, get_op_from_lale_lib, make_operator

logger = logging.getLogger(__name__)


def _wrap_operators_in_symtab(symtab):
    for name, impl in symtab.items():
        if (
            inspect.isclass(impl)
            and not issubclass(impl, Operator)
            and (hasattr(impl, "predict") or hasattr(impl, "transform"))
        ):
            operator = get_op_from_lale_lib(impl)
            if operator is None:
                symtab[name] = make_operator(impl=impl, name=name)
                logger.info(f"Lale:Wrapped unkwnown operator:{name}")
            else:
                symtab[name] = clone_op(operator, name)
                if operator.class_name().startswith("lale.lib.autogen"):
                    logger.info(f"Lale:Wrapped autogen operator:{name}")
                else:
                    logger.info(f"Lale:Wrapped known operator:{name}")


def wrap_imported_operators():
    calling_frame = inspect.stack()[1][0]
    _wrap_operators_in_symtab(calling_frame.f_globals)
    if calling_frame.f_code.co_name == "<module>":  # for testing with exec()
        _wrap_operators_in_symtab(calling_frame.f_locals)
