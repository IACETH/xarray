#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Author: Mathias Hauser
#Date: 

from glob import glob
import numpy as np

from .core.combine import concat
from .core.dataarray import DataArray
from .backends.api import open_dataset
from .conventions import decode_cf_datetime

# =============================================================================

def read_netcdfs(files, dim, transform_func=None, **kwargs):
    """
    read and combine multiple netcdf files

    Parameters
    ----------
    files : string or list of files
        path with wildchars or iterable of files
    dim : string
        dimension along which to combine, does not have to exist in 
        file (e.g. ensemble)
    transform_func : function
        function to apply for individual datasets, see example
    kwargs : keyword arguments
    	passed to open_dataset

    Returns
    -------
    combined : xarray Dataset
        the combined xarray Dataset with transform_func applied

    Example
    -------
    read_netcdfs('/path/*.nc', dim='ens',
                 transform_func=lambda ds: ds.mean())

    Reference
    ---------
    http://xarray.pydata.org/en/stable/io.html#combining-multiple-files
    """


    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with open_dataset(path, **kwargs) as ds:
            # transform_func should do some sort of selection or
            # aggregation
            if transform_func is not None:
                ds = transform_func(ds)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    if isinstance(files, basestring):
        paths = sorted(glob(files))
    else:
        paths = files

    datasets = [process_one_path(p) for p in paths]
    combined = concat(datasets, dim)
    return combined


# =============================================================================

def read_netcdfs_cesm(files, dim, transform_func=None, **kwargs):
    """
    read and combine multiple netcdf files with open_cesm

    Parameters
    ----------
    files : string or list of files
        path with wildchars or iterable of files
    dim : string
        dimension along which to combine, does not have to exist in 
        file (e.g. ensemble)
    transform_func : function
        function to apply for individual datasets, see example
    kwargs : keyword arguments
        passed to open_cesm

    Returns
    -------
    combined : xarray Dataset
        the combined xarray Dataset with transform_func applied

    Example
    -------
    read_netcdfs('/path/*.nc', dim='ens',
                 transform_func=lambda ds: ds.mean())

    Reference
    ---------
    http://xarray.pydata.org/en/stable/io.html#combining-multiple-files
    """


    def process_one_path(path):
        # use a context manager, to ensure the file gets closed after use
        with open_cesm(path, **kwargs) as ds:
            # transform_func should do some sort of selection or
            # aggregation
            if transform_func is not None:
                ds = transform_func(ds)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    if isinstance(files, basestring):
        paths = sorted(glob(files))
    else:
        paths = files

    datasets = [process_one_path(p) for p in paths]
    combined = concat(datasets, dim)
    return combined


# =============================================================================

def _average_da(self, dim=None, weights=None):
    """
    weighted average for DataArrays

    Parameters
    ----------
    dim : str or sequence of str, optional
        Dimension(s) over which to apply average.
    weights : DataArray
        weights to apply. Shape must be broadcastable to shape of self.

    Returns
    -------
    reduced : DataArray
        New DataArray with average applied to its data and the indicated
        dimension(s) removed.

    """

    if weights is None:
        return self.mean(dim)
    else:
        if not isinstance(weights, DataArray):
            raise ValueError("weights must be a DataArray")

        # if NaNs are present, we need individual weights
        if self.notnull().any():
            total_weights = weights.where(self.notnull()).sum(dim=dim)
        else:
            total_weights = weights.sum(dim)
        
        return (self * weights).sum(dim) / total_weights

# -----------------------------------------------------------------------------

def _average_ds(self, dim=None, weights=None):
    """
    weighted average for Datasets

    Parameters
    ----------
    dim : str or sequence of str, optional
        Dimension(s) over which to apply average.
    weights : DataArray
        weights to apply. Shape must be broadcastable to shape of data.

    Returns
    -------
    reduced : Dataset
        New Dataset with average applied to its data and the indicated
        dimension(s) removed.

    """
    
    if weights is None:
        return self.mean(dim)
    else:
        return self.apply(_average_da, dim=dim, weights=weights)

# =============================================================================

def _wrap360(self, lon='lon'):
    """
    wrap longitude coordinates to 0..360

    Parameters
    ----------
    ds : Dataset
        object with longitude coordinates
    lon : string
        name of the longitude ('lon', 'longitude', ...)

    Returns
    -------
    wrapped : Dataset
        Another dataset array wrapped around.
    """

    # wrap -180..179 to 0..359   

    
    new_lon = np.mod(self[lon], 360)

    self = self.assign_coords(**{lon: new_lon})
    # sort the data
    return self.reindex(**{lon : np.sort(self[lon])})

# =============================================================================

def _wrap180(self, lon='lon'):
    """
    wrap longitude coordinates to -180..180

    Parameters
    ----------
    ds : Dataset
        object with longitude coordinates
    lon : string
        name of the longitude ('lon', 'longitude', ...)

    Returns
    -------
    wrapped : Dataset
        Another dataset array wrapped around.
    """

    # wrap 0..359 to -180..179
    new_lon = self[lon].data

    # only modify values > 180
    sel = new_lon > 180
    
    if np.any(sel):
        # 359 -> -1, 181 -> -179    
        new_lon[sel] = np.mod(new_lon[sel], -180)
        self = self.assign_coords(**{lon: new_lon})
        # sort the data

    self = self.reindex(**{lon : np.sort(self[lon])})
    
    return self

# =============================================================================

def _cos_wgt(self, lat='lat'):
    """cosine-weighted latitude"""
    return np.cos(np.deg2rad(self[lat]))


# =============================================================================


def open_cesm(filename_or_obj, group=None, decode_cf=True,
              mask_and_scale=True, decode_times=True,
              concat_characters=True, decode_coords=True, engine=None,
              chunks=None, lock=None, drop_variables=None, round_latlon=4):
    """Load and decode a dataset from a file or file-like object.

    Parameters
    ----------
    filename_or_obj : str, file or xarray.backends.*DataStore
        Strings are interpreted as a path to a netCDF file or an OpenDAP URL
        and opened with python-netCDF4, unless the filename ends with .gz, in
        which case the file is gunzipped and opened with scipy.io.netcdf (only
        netCDF3 supported). File-like objects are opened with scipy.io.netcdf
        (only netCDF3 supported).
    group : str, optional
        Path to the netCDF4 group in the given file to open (only works for
        netCDF4 files).
    decode_cf : bool, optional
        Whether to decode these variables, assuming they were saved according
        to CF conventions.
    mask_and_scale : bool, optional
        If True, replace array values equal to `_FillValue` with NA and scale
        values according to the formula `original_values * scale_factor +
        add_offset`, where `_FillValue`, `scale_factor` and `add_offset` are
        taken from variable attributes (if they exist).  If the `_FillValue` or
        `missing_value` attribute contains multiple values a warning will be
        issued and all array values matching one of the multiple values will
        be replaced by NA.
    decode_times : bool, optional
        If True, decode times encoded in the standard NetCDF datetime format
        into datetime objects. Otherwise, leave them encoded as numbers.
    concat_characters : bool, optional
        If True, concatenate along the last dimension of character arrays to
        form string arrays. Dimensions will only be concatenated over (and
        removed) if they have no corresponding variable and if they are only
        used as the last dimension of character arrays.
    decode_coords : bool, optional
        If True, decode the 'coordinates' attribute to identify coordinates in
        the resulting dataset.
    engine : {'netcdf4', 'scipy', 'pydap', 'h5netcdf', 'pynio'}, optional
        Engine to use when reading files. If not provided, the default engine
        is chosen based on available dependencies, with a preference for
        'netcdf4'.
    chunks : int or dict, optional
        If chunks is provided, it used to load the new dataset into dask
        arrays. This is an experimental feature; see the documentation for more
        details.
    lock : False, True or threading.Lock, optional
        If chunks is provided, this argument is passed on to
        :py:func:`dask.array.from_array`. By default, a per-variable lock is
        used when reading data from netCDF files with the netcdf4 and h5netcdf
        engines to avoid issues with concurrent access when using dask's
        multithreaded backend.
    drop_variables: string or iterable, optional
        A variable or list of variables to exclude from being parsed from the
        dataset. This may be useful to drop variables with problems or
        inconsistent values.
    round_latlon: int, optional
        The latitude and longitude coordinates are rounded to this number of
        decimals. This is done because there are very small numerical differences
        in the coordinates between the land and atmosphere files.

    Returns
    -------
    dataset : Dataset
        The newly created dataset.

    See Also
    --------
    open_mfdataset
    """

    # always open with decode_times = False
    ds = open_dataset(filename_or_obj, group, decode_cf,
                      mask_and_scale, False,
                      concat_characters, decode_coords, engine,
                      chunks, lock, drop_variables)

    if decode_times:
        if 'time_bnds' in ds.variables.keys() or 'time_bounds' in ds.variables.keys():
            time_name = 'time'
            # read temperature

            if 'time_bnds' in ds.variables.keys():
                num_dates = ds.time_bnds.mean(axis=1)
            else: 
                num_dates = ds.time_bounds.mean(axis=1)
            # 
            units = ds.coords[time_name].units
            calendar = ds.coords[time_name].calendar
            time = decode_cf_datetime(num_dates, units, calendar)
            ds.coords[time_name] = time



    if round_latlon:
        lat_name = 'lat'
        lon_name = 'lon'
        ds.coords[lat_name] = np.round(ds[lat_name], round_latlon)
        ds.coords[lon_name] = np.round(ds[lon_name], round_latlon)

    return ds