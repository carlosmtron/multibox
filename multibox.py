#!/usr/bin/python3

from math import pi
import numpy as np
import SWA

###########################
# Constantes del problema #
###########################

theta = 0.408407044966673121 # Axial tilt (rad)
t0 = 3.15576e7  # Período de traslación (segundos)
S  = 1360      # Constante Solar (W/m²)
a  = -302       # Parámetro optico a
b  = 1.87       # Parámetro óptico b


def meridional():
    # Meridional Heat Flux
    return 0

# Voy a leer el archivo "latitudes.dat" cuyas columnas son:
# latitud y albedo
albedo_vs_latitud = np.loadtxt('latitudes.dat')
nboxes = albedo_vs_latitud.shape[0]  # Número de cajas
print("Se han detectado ", nboxes, " cajas\n")
print("Latitud Media [rad] \t Albedo \t SWA [W/m²]")

fluxes = np.zeros((nboxes,4))
for ii in np.arange(0,nboxes):
    lat = albedo_vs_latitud[ii,0]*pi/180.0
    albedo = albedo_vs_latitud[ii,1]
    fluxes[ii,:] = [lat, SWA.SWA_calc(lat, albedo), 0,0]
    print(lat,"\t", albedo,"\t", fluxes[ii,1])

# Para arrancar, voy a suponer que Fi=0 en todas partes
print(fluxes)
