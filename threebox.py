#!/usr/bin/python3

import numpy as np
from scipy.optimize import minimize
import SWA


###########################
# Constantes del problema #
###########################

acels = 207.42          # Parámetro optico a en W/m²ºC
b     = 1.93           # Parámetro óptico b
a     = acels-b*273.15 # Parámetro óptico a en W/m²K 
area  = 1              # Área de cada sector


def LWA(T):
    # Radiación infrarroja que sale al espacio
    return a+b*T

# Voy a cargar manualmente latitud y albedo en un array. Cada caja está en una fila.
albedo_vs_latitud = np.array(([-56.719, 0.428], [0, 0.222], [56.719, 0.398]))
nboxes = albedo_vs_latitud.shape[0]  # Número de cajas
print("Se han detectado ", nboxes, " cajas\n")

# El albedo no lo voy a usar en este problema. Lo dejo porque me sirve para multibox.
# La latitud central tampoco me importa en verdad.

# Creo la matriz 'dcajas', de 'nboxes' filas, cuyas columnas son:
# lat, SWA, LWA, temp


dcajas = np.zeros((nboxes,4))
dseta = np.zeros((nboxes))    # Vector de convergencias

divisiones = np.arange(0,nboxes)
dcajas[:,0] = albedo_vs_latitud[:,0]
dcajas[:,1] = [203.984, 314.096, 208.916]
# Defino las temperaturas iniciales. Al ser tres cajas lo puedo hacer manualmente.
dcajas[:,3] = [300, 300, 300]  # Kelvin
dcajas[:,2] = LWA(dcajas[:,3])


print("Latitud Media \t Albedo \t SWA [W/m²]")

for ii in divisiones:   
    print(f"{lat:^+13.2f} \t {albedo:^6.2f} \t {dcajas[ii,1]:^10.4f}")



def calculo_dseta(vtemp):
    for i in divisiones:
        dseta[i] = (LWA(vtemp[i]) - dcajas[i,1])*area
    return dseta

def suma_dseta(vtemp):
    sumaf = 0
    dseta = calculo_dseta(vtemp)
    for ii in divisiones:
        sumaf += dseta[ii]
    return sumaf

def sigmaT(vtemp):
    # La función objetivo a minimizar es el opuesto de la producción de entropía
    suma = 0
    factual = calculo_dseta(vtemp)
    for ii in divisiones:
        suma += factual[ii]/vtemp[ii]
    return -suma

def gradiente(t):
    # Calcula el gradiente de la función objetivo
    # que es el -grad(sigmaT)
    f = calculo_dseta(t)
    df = np.zeros((divisiones.shape[0]))
    for i in divisiones:
        df[i] = b*area/t[i]-f[i]/t[i]**2
    return -df


print("\n", [["lat", " SWi ", " LWi ", " Ti "]])
with np.printoptions(precision=3, suppress=True):
    print(dcajas)
    print("\nConvergencias meridionales:")
    print(calculo_dseta(dcajas[:,3]))


print("Suma de convergencias = ", suma_dseta(dcajas[:,3]))

cons = ({'type': 'eq', 'fun': suma_dseta})

# Semilla
semilla = dcajas[:,3]

bnds=[(3, 400) for i in range(nboxes)]

# Optimización
solucion = minimize(sigmaT, semilla, method='SLSQP', bounds=bnds, tol=1e-9,
            constraints=cons, jac=gradiente, options={'maxiter': 2000, 'disp': True})


print("Cálculo de temperaturas:")
print(solucion)

Zfin = calculo_dseta(solucion.x)

print("\nConvergencias meridionales:")
print(Zfin)

print("Suma de convergencias = ", suma_dseta(solucion.x))

print("\nTemperaturas en ºC:")
print(solucion.x-273.15)

print("\nTemp. promedio:", np.average(solucion.x)-273.15, "ºC")


####################################
##       Solución analítica       ##
####################################

SW = dcajas[:,1]
SWtot = SW.sum()
sumita = (np.sqrt(SW-a)).sum()
Tf = np.zeros(3)
for i in range(3):
    Tf[i] = (np.sqrt(SW[i]-a)*(SWtot-3*a))/(sumita*b)

print("\nTemperaturas analíticas")
print("-------------------------------")
print(Tf, "[K]")
print(Tf-273.15, "[°C]")

zeta = -SW + a + b * Tf
print("\nConvergencias meridionales")
print("-------------------------------")
print(zeta, "[W/m²]")

F = np.zeros(2)
F[0] = -zeta[0]
F[1] = -zeta[1] + F[0]
print("\nFlujos meridionales")
print("-------------------------------")
print(F, "[W/m²]")
