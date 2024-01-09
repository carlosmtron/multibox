import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import mapas
import SWA


plt.rcParams['text.usetex'] = True
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)



# Datos de TOA provinientes de NCEP reanalysis
url = 'http://apdrc.soest.hawaii.edu:80/dods/public_data/Reanalysis_Data/NCEP/NCEP/clima/'
ncep_dswrf = xr.open_dataset(url + "other_gauss/dswrf")
ncep_uswrf = xr.open_dataset(url + "other_gauss/uswrf")
IncomingSWmap = ncep_dswrf.dswrf.mean(dim=('time'))
ReflectedSWmap = ncep_uswrf.uswrf.mean(dim='time')

Net_SW_map = IncomingSWmap - ReflectedSWmap
albedo_map = ReflectedSWmap / IncomingSWmap

fig, axes, cx = mapas.make_map(albedo_map,plt.cm.Blues)
plt.tight_layout()

# fig.suptitle('Albedo observado', fontsize=16)
axes[1].set_xlabel('Albedo', fontsize=16)
axes[1].set_ylabel('Latitud [$^\circ$]', fontsize=16)

plt.savefig('albedo.png', dpi=300)
plt.show()

###########
# Comp SW #
###########

latitudes = IncomingSWmap.lat.values
SW_neto = np.zeros((len(latitudes)))

for ii in range(len(latitudes)):
    SW_neto[ii] = SWA.SWA_calc(latitudes[ii], 0.)


plt.plot(latitudes, SW_neto, label='Teórico')
plt.plot(latitudes, IncomingSWmap.mean(dim='lon'), label='Observado')
plt.xlabel('Latitud [$^\circ$]', fontsize=16)
plt.ylabel('$SW$ en TOA [W m$^{-2}$]', fontsize=16)
plt.grid()
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("comp_SW.pdf")
plt.show()


################
# Graf SW neto #
################

# Reinicializo SW
SW_neto = xr.Dataset()
SW_neto['lat'] = latitudes

albedo = albedo_map.mean(dim='lon')
SW_neto['albedo'] = albedo

resultados = []
for jj in SW_neto['lat']:
    valor_albedo = SW_neto['albedo'].sel(lat=jj)
    resultado = SWA.SWA_calc(jj, valor_albedo)
    resultados.append(resultado)
    

SW_neto['data'] = (('lat',), np.array(resultados))


plt.plot(latitudes, SW_neto['data'])
plt.xlabel('Latitud [$^\circ$]', fontsize=16)
plt.ylabel('Insolación $SW$ neta [W m$^{-2}$]', fontsize=16)
plt.grid()
#plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig("SW_neto.pdf")
plt.show()
