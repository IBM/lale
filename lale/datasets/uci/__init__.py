# Note: all imports should be done as
# from .xxx import XXX as XXX
# this ensures that pyright considers them to be publicly available
# and not private imports (this affects lale users that use pyright)

from .uci_datasets import fetch_drugslib as fetch_drugslib
from .uci_datasets import (
    fetch_household_power_consumption as fetch_household_power_consumption,
)
