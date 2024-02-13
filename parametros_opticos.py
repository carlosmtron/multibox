# Determinación de los parámetros ópticos A y B de
# la atmósfera a partir de un ajuste lineal de los
# datos de NCEP Reanalysis.
# Basado en "The Climate Laboratory" Cap. 20. de Brian E. J. Rose
# https://brian-rose.github.io/ClimateLaboratoryBook/courseware/one-dim-ebm.html

import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import linregress

plt.rcParams['text.usetex'] = True

# Descargamos el dataset manualmente de
# https://downloads.psl.noaa.gov/Datasets/ncep.reanalysis/Monthlies/
# y lo guardamos en la carpeta "data".
# Alternativamente podemos descargarlo del mirror
# http://apdrc.soest.hawaii.edu:80/dods/public_data/Reanalysis_Data/NCEP/NCEP/monthly/
ncep_Ts = xr.open_dataset("data/skt.sfc.mon.ltm.1991-2020.nc", decode_times=False)
lat_ncep = ncep_Ts.lat; lon_ncep = ncep_Ts.lon
print(ncep_Ts)

# Tomamos el promedio zonal anual de temperaturas
Ts_ncep_anual = ncep_Ts.skt.mean(dim=('lon','time'))

# Dataset de radiación en TOA. Se descargan manualmente del mismo sitio.
# Asegurarse que los datos sean del mismo período y resolución temporal.
ncep_ulwrf = xr.open_dataset("data/ulwrf.ntat.mon.ltm.1991-2020.nc")
LW_ncep_anual = ncep_ulwrf.ulwrf.mean(dim=('lon','time'))


# Realizamos la regresión lineal para hallar el mejor ajuste
pendiente, ordenada, r_val, p_val, std_err = linregress(Ts_ncep_anual, LW_ncep_anual)

print('El mejor ajuste es A = %0.0f W/m² y B = %0.1f W/m²/ºC' %(ordenada, pendiente))
print('r = %0.3f y std_err = %0.3f' %(r_val, std_err))

# Voy a extraer los datos de temperaturas mayores a -10ºC
nuevas_temp = []
nuevas_LW = []
for i in range(len(Ts_ncep_anual)):
    if Ts_ncep_anual[i] > -10 and Ts_ncep_anual[i] < 30:
        nuevas_temp.append(Ts_ncep_anual[i])
        nuevas_LW.append(LW_ncep_anual[i])

B, A, r_val, p_val, std_err = linregress(nuevas_temp, nuevas_LW)

print('El nuevo ajuste es A = %0.0f W/m² y B = %0.1f W/m²/ºC' %(A, B))
print('r = %0.3f y std_err = %0.3f' %(r_val, std_err))

plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

fig, ax1 = plt.subplots(figsize=(8, 6))
ax1.plot(Ts_ncep_anual, LW_ncep_anual, 'o', label='Datos de NCEP Reanalysis 1991-2020')
ax1.plot(Ts_ncep_anual, ordenada + pendiente * Ts_ncep_anual, 'k--', label='mejor ajuste')
ax1.plot(Ts_ncep_anual, A + B * Ts_ncep_anual, 'r--', label='ajuste para $-10 \leq T \leq 30$ $^\circ$C')

ax1.set_xlabel('Temperatura de la superficie [$^\circ$C]', fontsize=16)
ax1.set_ylabel('$LW$ en TOA [W m$^{-2}$]', fontsize=16)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid()
plt.savefig("img/ajusteAyB.pdf")
plt.show()
