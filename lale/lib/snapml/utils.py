from types import ModuleType
from typing import Optional

from packaging import version


def get_snapml_version(snapml: ModuleType) -> Optional[version.Version]:
    version_str = getattr(snapml, "__version__", None)
    if version_str is None:
        version_pkg = getattr(snapml, "version", None)
        if version_pkg is not None:
            version_str = getattr(version_pkg, "__version__", None)
    if isinstance(version_str, str):
        return version.parse(version_str)
    else:
        return None
