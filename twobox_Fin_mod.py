# twobox.py VERSIÓN MODIFICADA
# Autor Carlos M. Silva
# csilva@fceia.unr.edu.ar
#
# Programa basado en el modelo de 2 cajas descripto en Kleidon (2009)
# Itera para diferentes valores de kab.
# Resuelve numéricamente las ecuaciones diferenciales de la temperatura vs.
# tiempo para cada caja.
# Finalmente grafica las temperaturas del estado estacionario
# en función de kab.

# NUEVO: En esta versión, Fin es función de Tb [Kleidon, Phil. Trans. R. Soc. B (2010) 365, 1303–1315].

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# Constantes del problema
# F = 96
C = 2e+08
ALPHA = -302
BETA = 2.17
# GAMMA = (F-ALPHA)/C
ZETA = -ALPHA/C

# Funciones
def fab(kab, ta, tb):
    # Flujo de calor pseudomeridional
    return kab*(ta-tb)


def sigma_ab(kab, ta, tb):
    # Producción de entropía por flujo pseudomeridional
    return fab(ta,tb,kab)*(1/(tb+273.15)-1/(ta+273.15))


def sigma_a(ta,tb):
    # Producción de entropía por radiación que ingresa en la caja A
    return fin(tb)*(1/(ta+273.15)-1/5760.0)

def NEE(ta,tb):
    # Intercambio neto de entropía
    return (ALPHA+BETA*ta)/ta + (ALPHA+BETA*tb)/tb - fin(tb)/5760.0

# Defino el sistema de ecuaciones
def df(x,t,delta,epsilon):
    #delta,epsilon = parametros
    Ta, Tb = x[0], x[1]
    gamma = (fin(Tb) - ALPHA)/C
    diffTa = gamma + delta * Ta + epsilon * Tb
    diffTb = ZETA + epsilon * Ta + delta * Tb
    return [diffTa, diffTb]

# Modelizo Fin como función de Tb 
def fin(tb):
    faux = (15-tb)/30
    if faux < 0:
        return 0
    elif faux > 1:
        return 1
    else:
        return 96 + 35 * faux


# Condiciones Iniciales
temp_inicial = [38.7, -5.5]

# Rango de tiempo
npasos = 5000
t = np.linspace(0, 1e+09, npasos)

num_k = 4000 # cantidad de iteraciones de kab
temp_estacionaria = np.zeros((num_k,3))
ii=0
lista =  np.logspace(0,4,num_k)/1e3
for kab in lista:
    delta = -(kab + BETA)/C
    epsilon = kab/C
    # Solucion numérica
    sol = odeint(df, temp_inicial, t, args=(delta, epsilon))
    # Guardamos las temperaturas del estado estacionario en un array
    temp_estacionaria[ii,0] = kab
    temp_estacionaria[ii,1] = sol[npasos-1,0]
    temp_estacionaria[ii,2] = sol[npasos-1,1]
    ii+=1



# Voy a calcular fab y produccion de entropía para cada kab
sigmas = np.zeros((num_k,5))
sigmas[:,0] = temp_estacionaria[:,0]
for jj in range(num_k):
    kab, ta, tb = temp_estacionaria[jj,:]
    sigmas[jj,1] = fab(kab, ta, tb)
    sigmas[jj,2] = sigma_ab(kab, ta, tb)
    sigmas[jj,3] = sigma_a(ta, tb)
    sigmas[jj,4] = NEE(ta, tb)


fig, axs = plt.subplots(3, sharex=True, sharey=False)

axs[0].plot(temp_estacionaria[:,0],temp_estacionaria[:,1])
axs[0].plot(temp_estacionaria[:,0],temp_estacionaria[:,2])
axs[0].set_ylabel('T [°C]')
axs[0].grid(True)

axs[1].plot(sigmas[:,0],sigmas[:,2], "tab:green")
#axs[1].plot(sigmas[:,0],sigmas[:,3], "tab:pink")
#axs[1].plot(sigmas[:,0],sigmas[:,4], "tab:purple")
axs[1].set_ylabel('$\sigma_{AB}$ [mW m$^{-2}$K$^{-1}$]')
axs[1].set_autoscaley_on(True)
axs[1].grid(True)

axs[2].plot(sigmas[:,0],sigmas[:,1], "tab:red")
axs[2].set_ylabel('$F_{AB}$ [W m$^{-2}$]')
axs[2].grid(True)

plt.xscale('log')
plt.xlabel('$k_{AB}$')
plt.show()