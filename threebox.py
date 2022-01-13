#!/usr/bin/python3

import numpy as np
from scipy.optimize import minimize
import SWA


###########################
# Constantes del problema #
###########################

acels = 203.3          # Parámetro optico a en W/m²ºC
b     = 1.87           # Parámetro óptico b
a     = acels-b*273.15 # Parámetro óptico a en W/m²K 
area  = 1              # Área de cada sector


def LWA(T):
    # Radiación infrarroja que sale al espacio
    return a+b*T

# Voy a leer el archivo "latitudes.dat" cuyas columnas son:
# latitud y albedo
albedo_vs_latitud = np.array(([-56.719, 0.4], [0, 0.2], [56.719, 0.41]))
nboxes = albedo_vs_latitud.shape[0]  # Número de cajas
print("Se han detectado ", nboxes, " cajas\n")


# Creo la matriz 'fluxes', de 'nboxes' filas, cuyas columnas son:
# lat, SWA, LWA, temp
# La distribución de temperatura inicial será uniforme, de 400 K 


dcajas = np.zeros((nboxes,4))
divisiones = np.arange(0,nboxes)
T_inicial = 300      # Kelvin
flujos = np.zeros((nboxes+1))

print("Latitud Media \t Albedo \t SWA [W/m²]")

for ii in divisiones:
    lat = albedo_vs_latitud[ii,0]  # Se los paso en grados
    albedo = albedo_vs_latitud[ii,1]
    dcajas[ii,:] = [lat, SWA.SWA_calc(lat, albedo), LWA(T_inicial), T_inicial]
    
    print(f"{lat:^+13.2f} \t {albedo:^6.2f} \t {dcajas[ii,1]:^10.4f}")

    
def calculo_flujos(vtemp):
    flujos[0] = 0
    flujos[nboxes] = 0
    for i in np.arange(1,nboxes):
        flujos[i] = (dcajas[i-1,1] - LWA(vtemp[i-1]))*area + flujos[i-1]
    return flujos

def suma_flujos(vtemp):
    sumaf = 0
    flujos = calculo_flujos(vtemp)
    for ii in np.arange(0,flujos.shape[0]):
        sumaf += flujos[ii]
    return sumaf

def sigmaT(vtemp):
    suma = 0
    factual = calculo_flujos(vtemp)
    for ii in np.arange(0,nboxes):
        suma += (factual[ii] - factual[ii+1])/vtemp[ii]
    return -suma



print([["lat", " SWi ", " LWi ", " Ti "]])
with np.printoptions(precision=3, suppress=True):
    print(dcajas)
    print("\nFlujos meridionales:")
    print(calculo_flujos(dcajas[:,3]))


print("Suma de flujos = ", suma_flujos(dcajas[:,3]))

cons = ({'type': 'eq', 'fun': suma_flujos})

# Semilla
semilla = dcajas[:,3]

bnds=[(3, None) for i in range(nboxes)]

# Optimización
# solucion =  minimize(sigmaT,semilla,method='nelder-mead', options={'maxiter': 1200, 'xatol': 0.0001, 'return_all': True, 'disp': True})

solucion = minimize(sigmaT, semilla, method='SLSQP', bounds=bnds, constraints=cons, options={'maxiter': 1200, 'disp': True})

print("Cálculo de temperaturas:")
print(solucion)

print("\nFlujos meridionales:")
print(calculo_flujos(solucion.x))

print("\nTemperaturas en ºC:")
print(solucion.x-273.15)

print("\nTemp. promedio:", np.average(solucion.x)-273.15)
