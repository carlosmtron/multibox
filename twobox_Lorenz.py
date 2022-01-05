# twobox.py
# Autor Carlos M. Silva
# csilva@fceia.unr.edu.ar
#
# Programa basado en el modelo de 2 cajas descripto en Kleidon (2009)
# Itera para diferentes valores de kab.
# Resuelve numéricamente las ecuaciones diferenciales de la temperatura vs.
# tiempo para cada caja.
# Finalmente grafica las temperaturas del estado estacionario
# en función de kab.

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

# Constantes del problema
SWA = 316.0
SWB = 220
C = 2e+08
A_CELSIUS = 203.3 #(North et.al., 1981) Wm⁻²
BETA = 2.09   #Ibid. Wm⁻2ºC⁻¹
ALPHA = A_CELSIUS-BETA*273.15
#
GAMMA1 = (SWA-ALPHA)/C
GAMMA2 = (SWB-ALPHA)/C
ZETA = -ALPHA/C

# Funciones
def fab(kab, ta, tb):
    # Flujo de calor pseudomeridional
    return kab*(ta-tb)


def sigma_ab(kab, ta, tb):
    # Producción de entropía por flujo pseudomeridional
    return fab(ta,tb,kab)*(1/tb-1/ta)


def sigma_ingress(ta, tb):
    # Producción de entropía por radiación que ingresa en las dos cajas
    return SWA*(1/ta+-1/5760.0) + SWB*(1/tb-1/5760.0)


def NEE(ta,tb):
    # Intercambio neto de entropía
    return (ALPHA+BETA*ta)/ta + (ALPHA+BETA*tb)/tb - (SWA+SWB)/5760.0

# Defino el sistema de ecuaciones
def df(x,t,delta,epsilon):
    #delta,epsilon = parametros
    Ta, Tb = x[0], x[1]
    diffTa = GAMMA1 + delta * Ta + epsilon * Tb
    diffTb = GAMMA2 + epsilon * Ta + delta * Tb
    return [diffTa, diffTb]



# Condiciones Iniciales
temp_inicial = [320, 268]

# Rango de tiempo
npasos = 10000
t = np.linspace(0, 3e+09, npasos)

num_k = 4000 # cantidad de iteraciones de kab
temp_estacionaria = np.zeros((num_k,3))
ii=0
lista =  np.logspace(0,8,num_k)/1e3
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
    sigmas[jj,3] = sigma_ingress(ta,tb)
    sigmas[jj,4] = NEE(ta, tb)

# Voy a determinar el máximo sigma_ab y las temperaturas correspondientes
busqueda = sigmas[:,2]
sigma_max = np.amax(busqueda)
temperatura_a = temp_estacionaria[np.where(np.amax(busqueda)),1]
temperatura_b = temp_estacionaria[np.where(np.amax(busqueda)),2]
print("sigma_max:", sigma_max, temperatura_a, temperatura_b)
                                  
    
# Grafico los para t infinito T, \sigma_{AB} y F_{AB} en función de k_{AB}
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
plt.xlabel('$k_{AB}$ [W m$^{-2}$K$^{-1}$]')
plt.show()