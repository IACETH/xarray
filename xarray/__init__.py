from .core.alignment import align, broadcast, broadcast_arrays
from .core.combine import concat, auto_combine
from .core.variable import Variable, Coordinate
from .core.dataset import Dataset
from .core.dataarray import DataArray
from .core.options import set_options

from .backends.api import open_dataset, open_mfdataset, save_mfdataset
from .conventions import decode_cf

from .version import version as __version__

from . import tutorial


# import mutils

from mutils import read_netcdfs, _average_da, _average_ds
from mutils import _wrap360

from .mutils import open_cesm

# monkey patch
DataArray.average = _average_da
Dataset.average = _average_ds

DataArray.wrap360 = _wrap360
Dataset.wrap360 = _wrap360
