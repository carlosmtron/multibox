#!/usr/bin/python3

from math import pi
import numpy as np
import SWA

###########################
# Constantes del problema #
###########################

acels = 203.3      # Parámetro optico a en W/m²ºC
b     = 1.87       # Parámetro óptico b
a     = acels-b*273.15 #Parámetro óptico a en W/m²K 
area  = 5.107083e+13 # Área de cada sector (ojo, sectores de Paltridge)


def LWA(T):
    # Radiación infrarroja que sale al espacio
    return a+b*T

# Voy a leer el archivo "latitudes.dat" cuyas columnas son:
# latitud y albedo
albedo_vs_latitud = np.loadtxt('latitudes.dat')
nboxes = albedo_vs_latitud.shape[0]  # Número de cajas
print("Se han detectado ", nboxes, " cajas\n")
print("Latitud Media \t Albedo \t SWA [W/m²]")

# Creo la matriz 'fluxes', de 'nboxes' filas, cuyas columnas son:
# lat, SWA, LWA, Fi, temp
# # Para arrancar, voy a suponer que Fi=0 en todas partes.
# La distribución de temperatura inicial será uniforme, de 273,15 K 


fluxes = np.zeros((nboxes,5))
divisiones = np.arange(0,nboxes)
T_inicial = 400      # Kelvin
for ii in divisiones:
    lat = albedo_vs_latitud[ii,0]  # Se los paso en grados
    albedo = albedo_vs_latitud[ii,1]
    fluxes[ii,:] = [lat, SWA.SWA_calc(lat, albedo), LWA(T_inicial), 0, T_inicial]
    if ii == 0 or ii == nboxes-1:
        fluxes[ii,3] = 0
    else:
        fluxes[ii,3] = fluxes[ii-1,3] + (fluxes[ii,1]-fluxes[ii,2])*area

    
    print(f"{lat:^+13.2f} \t {albedo:^6.2f} \t {fluxes[ii,1]:^10.4f}")

"""    
def sigma(fluxes):
    suma = 0
    nboxes = fluxes.shape[0]
    for ii in np.arange(1,nboxes):
        suma += (fluxes[ii-1,3] - fluxes[ii,3]) / fluxes[ii,4]
    return suma
"""

def sigmaT(vtemp):
    suma = 0
    nboxes = fluxes.shape[0]
    for ii in np.arange(0,nboxes):
        suma += (fluxes[ii,1] - LWA(vtemp[ii])) / vtemp[ii]
    return -suma


# print(sigmaT(fluxes))

print("    lat \t SWi \t    LWi \t Fi \t     Ti")
with np.printoptions(precision=3, suppress=True):
    print(fluxes)


from scipy.optimize import minimize

# Semilla
semilla = fluxes[:,4]

    # Optimización
solucion =  minimize(sigmaT,semilla,method='nelder-mead', options={'maxiter': 1200, 'xatol': 0.0001, 'return_all': True, 'disp': True})

print(solucion.x)
