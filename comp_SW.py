import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import SWA

# Datos de TOA provinientes de NCEP reanalysis
url = 'http://apdrc.soest.hawaii.edu:80/dods/public_data/Reanalysis_Data/NCEP/NCEP/clima/'
ncep_dswrf = xr.open_dataset(url + "other_gauss/dswrf")

IncomingSWmap = ncep_dswrf.dswrf.mean(dim=('time'))

latitudes = IncomingSWmap.lat.values
SW_neto = np.zeros((len(latitudes)))

for ii in range(len(latitudes)):
    SW_neto[ii] = SWA.SWA_calc(latitudes[ii], 0.)

plt.rcParams['text.usetex'] = True
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

plt.plot(latitudes, SW_neto, label='Teórico')
plt.plot(latitudes, IncomingSWmap.mean(dim='lon'), label='Observado')
plt.xlabel('Latitud [º]', fontsize=16)
plt.ylabel('$SW$ en TOA [W m$^{-2}$]', fontsize=16)
plt.grid()
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("comp_SW.pdf")
plt.show()
