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

# A shim for compatibility across 3.7.
# pre 3.7, we need to inherit from the GenericMeta class (which inherits from ABCmeta)
# to use Generic (which we want to do)
# post 3.7, GenericMeta no longer exists
import sys
from abc import ABCMeta


class VisitorMeta(object):
    """ This meta class adds a private _accept method that calls visitCLASSNAME on the visitor.
        It does not currently support inheritance: you need to define the visitC method for subclasses
        explicitly.
        The private _accept method should be called via the Visitor#acccept method
    """

    def __init__(cls, *args, **kwargs):
        super(VisitorMeta, cls).__init__(*args, **kwargs)
        selector = """
        from lale.util import VisitorPathError
        try:
            return visitor.visit{}(self, *args, **kwargs)
        except VisitorPathError as e:
            e.push_parent_path(self)
            raise
        except BaseException as e:
            raise VisitorPathError([self]) from e
        """.format(
            cls.__name__
        )
        _accept_code = "def _accept(self, visitor, *args, **kwargs):\n\t{}".format(
            selector
        )
        ll = {}
        exec(_accept_code, globals(), ll)
        setattr(cls, "_accept", ll["_accept"])


if sys.version_info < (3, 7, 0):
    from typing import GenericMeta  # type: ignore
else:
    global GenericMeta
    GenericMeta = ABCMeta  # type: ignore


class AbstractVisitorMeta(VisitorMeta, GenericMeta):
    """ This meta class adds an _accept method that calls visitCLASSNAME on the visitor.
        It does not currently support inheritance: you need to define the visitC method for subclasses
        explicitly.
        The private _accept method should be called via the Visitor#acccept method.
    """

    def __init__(cls, *args, **kwargs):
        super(AbstractVisitorMeta, cls).__init__(*args, **kwargs)
