# maximize.py
#
# Cálculo de las temperaturas que maximizan la producción de entropía
# para un Flujo meridional F constante en el problema de dos cajas.
#
# Autor Carlos M. Silva

import numpy as np
from scipy.optimize import minimize

# Constantes del problema
SWA   = 316.0
SWB   = 220
A_CELSIUS = 203.3
BETA  = 2.09
ALPHA = A_CELSIUS-BETA*273.15

def ta(F):
    return (SWA-F-ALPHA)/BETA

def tb(F):
    return (SWB+F-ALPHA)/(BETA)

def objective(F):
    # La función objetivo es la opuesta de la que queremos maximizar
    return -F*(1/tb(F) - 1/ta(F))
   


# Semilla
semilla = 24.0


# Optimización
solucion =  minimize(objective,semilla,method='nelder-mead', options={'maxiter': 1200, 'xatol': 0.0001, 'return_all': True, 'disp': True})

F = solucion.x[0]

print("sigma = ", -objective(F), "W/m²K") #Recordamos que sigma es el opuesto de la función que minimizamos
print("F = ", F, "W/m²")
print("Ta = ", ta(F), "K")
print("Tb = ", tb(F), "K")
