#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Author: Mathias Hauser
#Date: 

import matplotlib.pyplot as plt

from .plot import _plot2d, _infer_interval_breaks

@_plot2d
def geocolormesh(x, y, z, ax, **kwargs):
    """
    Pseudocolor plot of 2d DataArray

    Wraps matplotlib.pyplot.pcolormesh
    """

    import cartopy.crs as ccrs

    proj = kwargs.pop('projection', ccrs.PlateCarree())
    trans = kwargs.pop('transform', ccrs.PlateCarree())

    x = _infer_interval_breaks(x)
    y = _infer_interval_breaks(y)

    coastlines = kwargs.pop('coastlines', True)

    primitive = ax.pcolormesh(x, y, z, transform=trans, **kwargs)


    if coastlines:
        ax.coastlines()

    xmin, xmax, ymin, ymax = x.min(), x.max(), y.min(), y.max()
    


    if y.min() <= -90 and  y.max() >= 90:
        ymin, ymax = -89.9, 89.9

    if x.min() % 360 == x.max() % 360:
        xmin += 0.1

    ext = xmin, xmax, ymin, ymax

    ax.set_extent(ext, proj)

    return ax, primitive
