# twobox.py
# Autor Carlos M. Silva
# csilva@fceia.unr.edu.ar
#
# Programa basado en el modelo de dos  cajas descripto por
# Lorenz et. al. (2001) y Kleidon (2009), entre otros.
# Itera para diferentes valores de k (= kab).
# Resuelve numéricamente las ecuaciones diferenciales de la
# temperatura en función del  tiempo para cada caja.
# Finalmente grafica las temperaturas del estado estacionario
# y la producción de entropía en función de k.
#
# En este código, sigma hace referencia a la produccion
# de entropía por unidad de superficie.

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

plt.rcParams['text.usetex'] = True

# Constantes del problema
SWA = 305.71
SWB = 177.38
C = 2e+08
A_CELSIUS = 208.0
BETA = 1.9
ALPHA = A_CELSIUS-BETA*273.15
GAMMA1 = (SWA-ALPHA)/C
GAMMA2 = (SWB-ALPHA)/C

# Funciones
def fab(kab, ta, tb):
    # Flujo de calor pseudomeridional
    return kab*(ta-tb)


def sigma_ab(kab, ta, tb):
    # Producción de entropía por flujo pseudomeridional
    return fab(kab, ta, tb)*(1/tb-1/ta)


def sigma_ingress(ta, tb):
    # Producción de entropía por radiación que ingresa en las dos cajas
    return SWA*(1/ta-1/5760.0) + SWB*(1/tb-1/5760.0)


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
temp_inicial = [312.2, 267.65]

# Rango de tiempo
npasos = 10000
t = np.linspace(0, 3e+09, npasos)

num_k = 4000 # cantidad de iteraciones de kab
temp_estacionaria = np.zeros((num_k,3))
ii = 0
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
    ii += 1


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
k_indice = np.where(sigmas[:,2]==sigma_max)[0][0]
k_inferencia = temp_estacionaria[k_indice,0]
temperatura_a = temp_estacionaria[k_indice,1]
temperatura_b = temp_estacionaria[k_indice,2]
print("P_max/A [W/m²K]:", sigma_max)
print("kab inferido [W/m²K]:", k_inferencia)
print("Ta [K]:", temperatura_a, "=", temperatura_a-273.15, "[ºC]")
print("Tb [K]:", temperatura_b, "=", temperatura_b-273.15, "[ºC]")

# Grafico para t infinito T, P y F_{AB} en función de k.
fig, axs = plt.subplots(2, sharex=True, sharey=False)

plt.xscale('log')
plt.xlabel('$k\ [\mbox{W m}^{-2}\mbox{K}^{-1}]$')

# Bandas de Tº observada
tmax_ecuador = 303*np.ones(num_k)
tmin_ecuador = 293*np.ones(num_k)
tmax_polos = 291*np.ones(num_k)
tmin_polos = 271*np.ones(num_k)

axs[0].plot(temp_estacionaria[:,0],temp_estacionaria[:,1],
            label="$T_{\infty,A}$")
axs[0].plot(temp_estacionaria[:,0],temp_estacionaria[:,2],
            label="$T_{\infty,B}$")
axs[0].fill_between(temp_estacionaria[:,0], tmin_ecuador, tmax_ecuador, alpha=0.4)
axs[0].fill_between(temp_estacionaria[:,0], tmin_polos, tmax_polos, alpha=0.4)
axs[0].set_ylabel('$T_\infty\ [\mbox{K}]$')
axs[0].set_yticks([250, 260, 270, 280, 290, 300, 310, 320, 330])
axs[0].grid()
axs[0].minorticks_on()
axs[0].legend()

axs[1].plot(sigmas[:,0],sigmas[:,2], "tab:green", label="$\mathcal{P}/\mathcal{A}$")
axs[1].set_ylim([0, 0.013])
axs[1].set_ylabel('$\mathcal{P}/\mathcal{A}\ [\mbox{W m}^{-2}\mbox{K}^{-1}]$')
secundario = axs[1].twinx()
secundario.plot(sigmas[:,0],sigmas[:,1], "tab:red", label="$F_{AB}$")
secundario.set_ylabel('$F_{AB}\ [\mbox{W m}^{-2}]$')
secundario.set_ylim([0, 70])
lines = axs[1].get_lines() + secundario.get_lines()
axs[1].legend(lines, [line.get_label() for line in lines], loc='upper left')

axs[1].grid()
axs[1].minorticks_on()
plt.savefig('img/doscajas_MEP_results.pdf')
plt.show()
