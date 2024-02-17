# Determinación de los parámetros ópticos A y B de
# la atmósfera a partir de un ajuste lineal de los
# datos de NCEP Reanalysis.
# Basado en "The Climate Laboratory" Cap. 20. de Brian E. J. Rose
# https://brian-rose.github.io/ClimateLaboratoryBook/courseware/one-dim-ebm.html

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.odr import ODR, Model, Data, RealData

plt.rcParams['text.usetex'] = True
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

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
desvios_Ts = ncep_Ts.skt.std(dim=('lon','time')).values

# Dataset de radiación en TOA. Se descargan manualmente del mismo sitio.
# Asegurarse que los datos sean del mismo período y resolución temporal.
ncep_ulwrf = xr.open_dataset("data/ulwrf.ntat.mon.ltm.1991-2020.nc")
LW_ncep_anual = ncep_ulwrf.ulwrf.mean(dim=('lon','time'))
desvios_LW = ncep_ulwrf.std(dim=('lon','time')).ulwrf.values

# Realizamos la regresión lineal para hallar el ajuste preliminar
pendiente, ordenada, r_val, p_val, std_err = linregress(Ts_ncep_anual, LW_ncep_anual)

print('El ajuste preliminar es A = %0.0f W/m² y B = %0.1f W/m²/ºC' %(ordenada, pendiente))
print('r = %0.3f y std_err = %0.3f' %(r_val, std_err))

################################################
###   Ajuste contemplando barras de error    ###
################################################

def f_model(beta, x):
    # Función lineal: Y = A * x + B
    a, b = beta
    return a + b * x


# Gráficas de los desvíos estándar
ticks=[-90, -60, -30, 0, 30, 60, 90]
plt.plot(Ts_ncep_anual.lat, desvios_Ts, 'g', label='$\sigma_T$')
plt.plot(LW_ncep_anual.lat, desvios_LW, 'r', label='$\sigma_{LW}$')
plt.xticks(ticks)
plt.xlabel('Latitud [$^\circ$]', fontsize=16)
plt.ylabel('Desvío Estándar [$^\circ$C, Wm$^{-2}$]', fontsize=16)
plt.ylim([0, 38])
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# Utilizamos el método de ajuste ODR que contempla barras de error en ambos ejes

data = RealData(Ts_ncep_anual.values, LW_ncep_anual.values, desvios_Ts, desvios_LW)
model = Model(f_model)

odr = ODR(data, model, [200, 2])
odr.set_job(fit_type=0)
odr_output = odr.run()
print("Iteración 1:")
print("------------")
print(" Razón de stop:", odr_output.stopreason)
print("        params:", odr_output.beta)
print("          info:", odr_output.info)
print("       sd_beta:", odr_output.sd_beta)
print("sqrt(diag(cov):", np.sqrt(np.diag(odr_output.cov_beta)))

# Si no se alcanzó la convergencia, corre de nuevo el algoritmo
if odr_output.info != 1:
    print("\nSe reinicia ODR hasta que converja")
    i = 1
    while odr_output.info != 1 and i < 100:
        print("Reinicio nro. ", i)
        odr_output = odr.restart()
        i += 1
    print(" Razón de stop:", odr_output.stopreason)
    print("        params:", odr_output.beta)
    print("          info:", odr_output.info)
    print("       sd_beta:", odr_output.sd_beta)
    print("sqrt(diag(cov):", np.sqrt(np.diag(odr_output.cov_beta)))

# Resultados del ajuste contemplando barras de error
a_odr, b_odr = odr_output.beta
print('El nuevo ajuste es A = %0.2f W/m² y B = %0.1f W/m²/ºC' %(a_odr, b_odr))

# Gráfica de LW vs. T y los ajustes

x = Ts_ncep_anual.values
y = LW_ncep_anual.values
xerror = desvios_Ts
yerror = desvios_LW

fig, ax1 = plt.subplots(figsize=(8, 6))
markers, caps, bars = ax1.errorbar(x, y, xerr=xerror, yerr=yerror,
            ecolor='lightblue', capsize=10,
            marker='o', linestyle='none', alpha=0.8,
            label='Datos de NCEP Reanalysis 1991-2020')
[bar.set_alpha(0.5) for bar in bars]
[cap.set_alpha(0.5) for cap in caps]

ax1.plot(x, ordenada + pendiente * x, 'k--', label='Ajuste de los valores promedios')
ax1.plot(x, a_odr + b_odr * x, 'r--', label='Ajuste con pesado del desvío estándar')
ax1.set_xlabel('Temperatura de la superficie [$^\circ$C]', fontsize=16)
ax1.set_ylabel('$LW$ en TOA [W m$^{-2}$]', fontsize=16)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid()
plt.savefig("img/ajusteAyB.pdf")
plt.show()
