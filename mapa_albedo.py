import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import mapas


plt.rcParams['text.usetex'] = True
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

ceres = xr.open_dataset("data/CERES_EBAF-TOA_Ed4.2_Subset_200003-202310.nc")
ceres

latitudes = ceres.lat
outsw = ceres.ztoa_sw_all_mon.mean(dim='time')
incoming = ceres.zsolar_mon.mean(dim='time')
resta = incoming-outsw

ticks = [-90, -60, -30, 0, 30, 60, 90]

plt.plot(latitudes, resta.values)
plt.xticks(ticks)
plt.grid(True, color='0.95')
plt.xlabel("Latitud [º]", fontsize=16)
plt.ylabel("SWR [W/m²]", fontsize=16)
plt.show()

albedo = outsw/incoming

plt.plot(latitudes, albedo.values)
plt.xticks(ticks)
plt.grid(True, color='0.95')
plt.xlabel("Latitud [º]", fontsize=16)
plt.ylabel("albedo", fontsize=16)
plt.show()

tronix = np.zeros((len(latitudes), 2))
tronix[:,0] = latitudes
tronix[:,1] = albedo.values
np.savetxt('latitudes-new.dat', tronix)

ceresmapa = xr.open_dataset("data/CERES_EBAF-TOA_Ed4.2_Subset_200003-202310-global.nc")
ceresmapa


latitudes = ceresmapa.lat
outsw = ceresmapa.toa_sw_all_mon.mean(dim='time')
incoming = ceresmapa.solar_mon.mean(dim='time')
resta = incoming-outsw
albedo_map = outsw/incoming

ceresmapa.solar_mon.mean(dim='time')

fig, axes, cx = mapas.make_map(outsw)

fig.suptitle('Radiación SW reflejada en TOA', fontsize=16)
axes[1].set_xlabel('Radiación reflejada [W/m²]')

plt.show()



fig, axes, cx = mapas.make_map(incoming)

fig.suptitle('Radiación SW reflejada en TOA', fontsize=16)
axes[1].set_xlabel('Radiación reflejada [W/m²]')

plt.show()

fig, axes, cx = mapas.make_map(albedo_map,plt.cm.Blues)
plt.tight_layout()

# fig.suptitle('Albedo observado', fontsize=16)
axes[1].set_xlabel('Albedo', fontsize=16)
axes[1].set_ylabel('Latitud [$^\circ$]', fontsize=16)

plt.savefig('albedo.png', dpi=300)


plt.show()
