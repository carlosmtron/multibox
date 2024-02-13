"""
Función para construir mapas a partir de campos de valores del clima.
     Input: campo, paleta de colores
     El campo debe ser un xarray 2D (ej: temperaturas, albedo, etc.)
     La paleta de colores es una de estas: https://matplotlib.org/stable/users/explain/colors/colormaps.html
Además grafica el promedio zonal en un gráfico a la derecha del mapa.

Adaptado de "The Climate Laboratory" de Brian E. J. Rose
https://brian-rose.github.io/ClimateLaboratoryBook/courseware/transient-cesm.html
"""

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point

def make_map(field, colores=plt.cm.viridis):
    longitudes = field.coords['lon']
    lon_idx = field.dims.index('lon')
    wrap_field, wrap_lon = add_cyclic_point(field.values, coord=longitudes, axis=lon_idx)
    fig = plt.figure(figsize=(14,6))
    nrows = 10; ncols = 3
    mapax = plt.subplot2grid((nrows,ncols), (0,0), colspan=ncols-1, rowspan=nrows-1, projection=ccrs.Robinson())
    barax = plt.subplot2grid((nrows,ncols), (nrows-1,0), colspan=ncols-1)
    plotax = plt.subplot2grid((nrows,ncols), (0,ncols-1), rowspan=nrows-1)
    cx = mapax.pcolormesh(wrap_lon, field.lat, wrap_field, cmap=colores, transform=ccrs.PlateCarree())
    mapax.set_global(); mapax.coastlines();
    plt.colorbar(cx, cax=barax, orientation='horizontal')
    plotax.plot(field.mean(dim='lon'), field.lat)
    plotax.set_ylabel('Latitud [º]')
    plotax.grid()
    return fig, (mapax, plotax, barax), cx
