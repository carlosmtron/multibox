"""
Este script compara los perfiles de radiación incidente en TOA
según el modelo utilizado en el programa de MEP y los datos de
NCEP/NCAR Reanalysis y CERES.

Autor: Carlos Silva
cislva@fceia.unr.edu.ar
Febrero 2024.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import SWA

# Datos de TOA provinientes de NCEP reanalysis.
# Los saco de un mirror al que nos podemos conectar desde el programa
# en vez de bajarlo manualmente.
url = 'http://apdrc.soest.hawaii.edu:80/dods/public_data/Reanalysis_Data/NCEP/NCEP/clima/'
ncep_dswrf = xr.open_dataset(url + "other_gauss/dswrf")

ceres = xr.open_dataset("data/CERES_EBAF-TOA_Ed4.2_Subset_200003-202310.nc")
incoming_ceres = ceres.zsolar_mon.mean(dim='time')

incoming_ncep = ncep_dswrf.dswrf.mean(dim=('time'))

latitudes = incoming_ncep.lat.values
SW_neto = np.zeros((len(latitudes)))

for ii in range(len(latitudes)):
    SW_neto[ii] = SWA.SWA_calc(latitudes[ii], 0.)

plt.rcParams['text.usetex'] = True
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

plt.plot(latitudes, SW_neto, label='Modelo')
plt.plot(latitudes, incoming_ncep.mean(dim='lon'), label='NCEP Reanalysis 1991-2020')
plt.plot(incoming_ceres.lat, incoming_ceres.values, label='CERES 2020-2023')
plt.xlabel('Latitud [$^\circ$]', fontsize=16)
plt.ylabel('Radiación solar incidente en TOA [W m$^{-2}$]', fontsize=16)
plt.grid()
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("img/comp_SW.pdf")
plt.show()
