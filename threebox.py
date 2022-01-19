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



dcajas = np.zeros((nboxes,4))
flujos = np.zeros((nboxes+1))

divisiones = np.arange(0,nboxes)

# Defino las temperaturas iniciales. Al ser tres cajas lo puedo hacer manualmente.
dcajas[:,3] = [220, 300, 250]  # Kelvin


print("Latitud Media \t Albedo \t SWA [W/m²]")

for ii in divisiones:
    lat = albedo_vs_latitud[ii,0]  # Se los paso en grados
    albedo = albedo_vs_latitud[ii,1]
    dcajas[ii,:3] = [lat, SWA.SWA_calc(lat, albedo), LWA(dcajas[ii,3])]
    
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
    # La función objetivo es el opuesto de la producción de entropía
    suma = 0
    factual = calculo_flujos(vtemp)
    for ii in divisiones:
        suma += (factual[ii] - factual[ii+1])/vtemp[ii]
    return -suma

def gradiente(t):
    # Calcula el gradiente de la función objetivo
    f = calculo_flujos(t)
    df = np.zeros((3))
    df[0] = f[1]/t[0]**2 + b*area*(1/t[0] - 1/t[1])
    df[1] = 1/t[1]**2 * (f[2]-f[1]) + b*area*(1/t[1] - 1/t[2])
    df[2] = -f[2]/t[2]**2
    return df


print("\n", [["lat", " SWi ", " LWi ", " Ti "]])
with np.printoptions(precision=3, suppress=True):
    print(dcajas)
    print("\nFlujos meridionales:")
    print(calculo_flujos(dcajas[:,3]))


print("Suma de flujos = ", suma_flujos(dcajas[:,3]))

cons = ({'type': 'eq', 'fun': suma_flujos})

# Semilla
semilla = dcajas[:,3]

bnds=[(200, 400) for i in range(nboxes)]

# Optimización

solucion = minimize(sigmaT, semilla, method='trust-constr', bounds=bnds, constraints=cons, jac=gradiente, options={'maxiter': 1200, 'disp': True})

print("Cálculo de temperaturas:")
print(solucion)

print("\nFlujos meridionales:")
print(calculo_flujos(solucion.x))

print("\nTemperaturas en ºC:")
print(solucion.x-273.15)

print("\nTemp. promedio:", np.average(solucion.x)-273.15)
