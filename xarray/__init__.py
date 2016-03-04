from .core.alignment import align, broadcast, broadcast_arrays
from .core.combine import concat, auto_combine
from .core.variable import Variable, Coordinate
from .core.dataset import Dataset
from .core.dataarray import DataArray
from .core.options import set_options

from .backends.api import open_dataset, open_mfdataset, save_mfdataset
from .conventions import decode_cf

try:
    from .version import version as __version__
except ImportError:  # pragma: no cover
    raise ImportError('xarray not properly installed. If you are running from '
                      'the source directory, please instead create a new '
                      'virtual environment (using conda or virtualenv) and '
                      'then install it in-place by running: pip install -e .')

from . import tutorial


# import mutils

from mutils import read_netcdfs, _average_da, _average_ds
from mutils import _wrap360, _wrap180, _cos_wgt

from .mutils import open_cesm, read_netcdfs_cesm

# monkey patch
DataArray.average = _average_da
Dataset.average = _average_ds

DataArray.wrap360 = _wrap360
Dataset.wrap360 = _wrap360

DataArray.wrap180 = _wrap180
Dataset.wrap180 = _wrap180

DataArray.cos_wgt = _cos_wgt
Dataset.cos_wgt = _cos_wgt