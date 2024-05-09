"""
Cálculo de promedios zonales de temperaturas
y radiación de onda corta ponderados por el área de cada zona.
Se admite que dentro de una misma zona latitudinal todas las subceldas son iguales
y por ello el promedio en longitud es directo.

Autor: Carlos Silva
csilva@fceia.unr.edu.ar
Febrero 2024
"""

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np

# Importamos los datos de temperatura
Tsdata_ncep = xr.open_dataset('data/skt.sfc.mon.ltm.1991-2020.nc')
ncep_Ts = Tsdata_ncep
# ncep_Ts['skt'] = ncep_Ts['skt'] - 273.15 # Descomentar si los datos están en K
lat_ncep = ncep_Ts.lat; lon_ncep = ncep_Ts.lon
print(ncep_Ts)

"""
Ahora haremos una función que seleccione los datos que se ingresan en un array,
filtrando entre dos latitudes $\lambda_1$ y $\lambda_2$.
"""

def mascara(datos, lambda1, lambda2):
  lat=datos.lat
  target = datos.where((lat >= lambda1) & (lat <= lambda2))
  datos_filtrados = target.dropna(dim='lat')
  return datos_filtrados

def mascara_excl(datos, lambda1, lambda2):
  # Excluye datos que están entre lambda1 y lambda2
  lat=datos.lat
  target = datos.where((lat <= lambda1) | (lat >= lambda2))
  datos_filtrados = target.dropna(dim='lat')
  return datos_filtrados

# Divisiones zonas
lambda1 = -30
lambda2 = +30

zona_sur = mascara(ncep_Ts, -90, lambda1)
zona_norte = mascara(ncep_Ts, lambda2, +90)

zona_polar = mascara_excl(ncep_Ts, lambda1, lambda2)
zona_ecuat = mascara(ncep_Ts, lambda1, lambda2)

# Histogramas
hist_ecuat = xr.plot.hist(zona_ecuat.skt,bins=100,alpha=0.8)
plt.suptitle("Histograma zona ecuatorial")
plt.grid()
plt.show()
hist_sur = xr.plot.hist(zona_sur.skt,bins=100)
plt.suptitle("Histograma casquete sur")
plt.show()
hist_norte = xr.plot.hist(zona_norte.skt,bins=100,alpha=0.6)
plt.suptitle("Histograma casquete norte")
plt.show()
hist_polos = xr.plot.hist(zona_polar.skt,bins=100)
plt.suptitle("Histograma zonas polares")
plt.grid()
plt.show()


promedio_ecuatorial = zona_ecuat.skt.mean().values+273.15
print("Prom. ecuatorial: ", promedio_ecuatorial, "K")
desveste = zona_ecuat.skt.std().values
print("Intervalo [K]:", (promedio_ecuatorial-desveste, promedio_ecuatorial+desveste))

promedio_polar = zona_polar.skt.mean().values+273.15
print("Prom. polar: ", promedio_polar, "K")
desvestp = zona_polar.skt.std().values
print("Intervalo [K]:", (promedio_polar-desvestp, promedio_polar+desvestp))

"""
¡OJO!
Hasta acá no hicimos ningún pesaje por el área de cada celda.
"""

######################
#### TEMPERATURAS ####
######################

print("\n\n-----------------------------")
print("     Promedios ponderados")
print("-----------------------------")

def pesaje_areas(zona):
  valores_zonales = zona.mean(dim=('time','lon'))
  ancho = np.pi/180.0
  areas = np.cos(np.deg2rad(zona.lat))*ancho
  return valores_zonales*areas/areas.sum()


promedio_sur = pesaje_areas(zona_sur.skt).sum().values
print("\nTemperatura promedio zona sur: %4.2f K" % (promedio_sur+273.15))

promedio_norte = pesaje_areas(zona_norte.skt).sum().values
print("\nTemperatura promedio zona norte: %4.2f K" % (promedio_norte+273.15))

promedio_polar = pesaje_areas(zona_polar.skt).sum().values
print("\nTemperatura promedio zona polar: %4.2f K" % (promedio_polar+273.15))

promedio_ecuador = pesaje_areas(zona_ecuat.skt).sum().values
print("\nTemperatura promedio zona ecuatorial: %4.2f K" % (promedio_ecuador+273.15))

promedio_global = pesaje_areas(ncep_Ts.skt).sum().values
print("\nTemperatura promedio global: %4.2f K" % (promedio_global+273.15))

# Cálculo de los desvíos estándar ponderados
# a priori, just for fun...
def desvios_ponderados(zona):
  latitudes = zona.lat
  ancho = np.pi/180.0
  areas = np.cos(np.deg2rad(latitudes))*ancho
  pesos = (areas/areas.sum()).values
  datos = zona.mean(dim=('lon','time')).values
  desvio = np.sqrt(np.cov(datos, aweights=pesos))
  return desvio

desvpolar = desvios_ponderados(zona_polar.skt)
print("Desvio temp. polares %.2f K" % desvpolar)
print("Intervalo: ( %.1f , %.1f ) ºC" % (promedio_polar-desvpolar, promedio_polar+desvpolar) )
desvecuat = desvios_ponderados(zona_ecuat.skt)
print("Desvio temp. ecuatoriales %.2f K" % desvecuat)
print("Intervalo: ( %.1f , %.1f ) ºC" % (promedio_ecuador-desvecuat, promedio_ecuador+desvecuat) )
print()

######################
#### RADIACIÓN SW ####
######################

# Importamos los datos de radiación incidente y reflejada en TOA
ceresmapa = xr.open_dataset("data/CERES_EBAF-TOA_Ed4.2_Subset_200003-202310-global.nc")
outsw = ceresmapa.toa_sw_all_mon.mean(dim='time')
incoming = ceresmapa.solar_mon.mean(dim='time')
resta = incoming-outsw
SWR_anual = resta


def pesaje_areas_nt(zona):
  # Ya se que no es elegante, pero no tengo tiempo, xD
  valores_zonales = zona.mean(dim=('lon'))
  ancho = np.pi/180.0
  areas = np.cos(np.deg2rad(zona.lat))*ancho
  return valores_zonales*areas/areas.sum()

SWA_data = mascara(SWR_anual, -30, 30)
SWB_data = mascara_excl(SWR_anual, -30, 30)

print("\n\nRADIACIONES")
print("-----------------------------")
SWA = pesaje_areas_nt(SWA_data).sum().values
print("SWA = ", SWA, " W/m²")
SWB = pesaje_areas_nt(SWB_data).sum().values
print("SWB = ", SWB, " W/m²")


SW0_data = mascara(SWR_anual, -90.00, -19.47)
SW1_data = mascara(SWR_anual, -19.47, +19.47)
SW2_data = mascara(SWR_anual, +19.47, +90.00)

print("\n\nRADIACIONES TRES CAJAS")
print("-----------------------------")
SW0 = pesaje_areas_nt(SW0_data).sum().values
print("SW0 = ", SW0, " W/m²")
SW1 = pesaje_areas_nt(SW1_data).sum().values
print("SW1 = ", SW1, " W/m²")
SW2 = pesaje_areas_nt(SW2_data).sum().values
print("SW2 = ", SW2, " W/m²")
