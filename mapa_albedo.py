import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import mapas


plt.rcParams['text.usetex'] = True
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rcParams["grid.color"]= "#DDDDDD"
plt.rcParams["grid.linewidth"] = 0.8

ceres = xr.open_dataset("data/CERES_EBAF-TOA_Ed4.2_Subset_200003-202310.nc")
ceres

latitudes = ceres.lat
outsw = ceres.ztoa_sw_all_mon.mean(dim='time')
incoming = ceres.zsolar_mon.mean(dim='time')
resta = incoming-outsw

ticks = [-90, -60, -30, 0, 30, 60, 90]

plt.plot(latitudes, resta.values)
plt.xticks(ticks)
plt.grid()
plt.xlabel("Latitud [º]", fontsize=16)
plt.ylabel("SWR [W/m²]", fontsize=16)
plt.savefig("SW_neto.pdf")
plt.show()

albedo = outsw/incoming

plt.plot(latitudes, albedo.values)
plt.xticks(ticks)
plt.grid()
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

fig.suptitle('Radiación SW incidente en TOA', fontsize=16)
axes[1].set_xlabel('Radiación incidente [W/m²]')

plt.show()

fig, axes, cx = mapas.make_map(albedo_map,plt.cm.Blues)
plt.tight_layout()

# fig.suptitle('Albedo observado', fontsize=16)
axes[1].set_xlabel('Albedo', fontsize=16)
axes[1].set_ylabel('Latitud [$^\circ$]', fontsize=16)

plt.savefig('albedo.png', dpi=300)


plt.show()


emitedlw = ceresmapa.toa_lw_all_mon.mean(dim='time')

fig, axes, cx = mapas.make_map(emitedlw)

fig.suptitle('Radiación LW emitida en TOA', fontsize=16)
axes[1].set_xlabel('Radiación reflejada [W/m²]')

plt.show()


acels = 208.0         # Parámetro optico a en W/m²ºC
b = 1.9               # Parámetro óptico b
a = acels-b*273.15    # Parámetro óptico a en W/m²K
RT = 6373000.0        # Radio terrestre en m


def T_inferida(LWmap):
    # Devuelve la temperatura inferida a partir de un mapa de LW.
    # Usa los parámetros a y b y el resultado sale en Kelvin.
    return (LWmap - a)/b

mapatemp = T_inferida(emitedlw) - 273.15

fig, axes, cx = mapas.make_map(mapatemp, plt.cm.coolwarm)

fig.suptitle('Temperatura inferida', fontsize=16)
axes[1].set_xlabel('T [$^\circ$C]')

plt.show()


# ncep_Ts = xr.open_dataset('data/skt.sfc.mon.ltm.1991-2020.nc')
# T_obs = ncep_Ts.skt.mean(dim=('lon', 'time'))
T_ceres = mapatemp.mean(dim=('lon'))
T_MEP = np.loadtxt("temperaturasMEP.dat")


plt.plot(T_MEP[:, 0], T_MEP[:, 1], marker="o", ls="", label="MEP", markersize=4)
# plt.plot(T_obs.lat, T_obs.values, label="NCEP Reanalysis 1991-2020")
# plt.plot(T_fuku[:,0], T_fuku[:,1]-273.15, label="MEP Fukumura y Ozawa (2014)",  ls="--", markersize=4)
plt.plot(latitudes, T_ceres, label="CERES 2000-2023")
plt.xticks(ticks)
plt.grid(which='major')
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
plt.minorticks_on()
plt.legend(fontsize=10)
# plt.title("Distribución de temperaturas")
plt.xlabel("Latitud [$^\circ$]", fontsize=16)
plt.ylabel("Temperatura [ºC]", fontsize=16)
plt.tight_layout()
plt.savefig("temperaturas_comp_CERES.pdf")
plt.show()


# Cálculo de la discrepancia relativa

mepkelvin = T_MEP[:,1] + 273.15
cereskelvin = T_ceres.values + 273.15
discrepancia = np.abs(cereskelvin - mepkelvin)/cereskelvin*100

plt.plot(latitudes, discrepancia)
plt.xticks(ticks)
plt.grid(which='major')
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
plt.minorticks_on()
plt.xlabel("Latitud [$^\circ$]", fontsize=16)
plt.ylabel("Discrepancia relativa [\%]", fontsize=16)
plt.tight_layout()
plt.savefig("discrepanciarelativa.pdf")
plt.show()
