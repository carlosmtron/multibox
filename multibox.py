#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import SWA
# import xarray as xr

###########################
# Constantes del problema #
###########################

area = 1              # Área de c/sector (1 -> trabajo por m²)
acels = 208.0         # Parámetro optico a en W/m²ºC
b = 1.9               # Parámetro óptico b
a = acels-b*273.15    # Parámetro óptico a en W/m²K


def LWA(T):
    # Radiación infrarroja que sale al espacio
    return a+b*T


# Voy a leer el archivo "latitudes.dat" cuyas columnas son:
# latitud y albedo
albedo_vs_latitud = np.loadtxt('latitudes-new.dat')
nboxes = albedo_vs_latitud.shape[0]              # Número de cajas
print("Se han detectado ", nboxes, " cajas\n")


# Creo la matriz 'dcajas', de 'nboxes' filas, cuyas columnas son:
# lat, SWA, LWA, temp


dcajas = np.zeros((nboxes,4))
dseta = np.zeros((nboxes))    # Vector de convergencias

divisiones = np.arange(0,nboxes)  # Grilla de números naturales de 0 a nboxes

# Defino las temperaturas iniciales. Empiezo con T homogénea.
for i in divisiones:
    dcajas[i, 3] = 0  # Kelvin


print("Latitud Media \t Albedo \t SWA [W/m²]")

for ii in divisiones:
    lat = albedo_vs_latitud[ii, 0]  # Se los paso en grados
    albedo = albedo_vs_latitud[ii, 1]
    dcajas[ii, :3] = [lat, SWA.SWA_calc(lat, albedo), LWA(dcajas[ii, 3])]
  
    print(f"{lat:^+13.2f} \t {albedo:^6.2f} \t {dcajas[ii,1]:^10.4f}")

    
def calculo_dseta(vtemp):
    for i in divisiones:
        dseta[i] = (LWA(vtemp[i]) - dcajas[i, 1])*area
    return dseta


def suma_dseta(vtemp):
    sumaf = 0
    dseta = calculo_dseta(vtemp)
    for ii in divisiones:
        sumaf += dseta[ii]
    return sumaf


def sigmaT(vtemp):
    # La función objetivo a minimizar es el opuesto
    # de la producción de entropía
    suma = 0
    factual = calculo_dseta(vtemp)
    for ii in divisiones:
        suma += factual[ii]/vtemp[ii]
    return -suma


def gradiente(t):
    # Calcula el gradiente de la función objetivo
    # que es el -grad(sigmaT)
    f = calculo_dseta(t)
    df = np.zeros((nboxes))
    for i in divisiones:
        df[i] = b*area/t[i]-f[i]/t[i]**2
    return -df


print("\n", [["lat", " SWi ", " LWi ", " Ti "]])
with np.printoptions(precision=3, suppress=True):
    print(dcajas)
    print("\nConvergencias meridionales:")
    print(calculo_dseta(dcajas[:, 3]))


print("Suma de convergencias = ", suma_dseta(dcajas[:, 3]))

cons = ({'type': 'eq', 'fun': suma_dseta})

# Semilla
semilla = dcajas[:, 3]

bnds = [(3, 600) for i in range(nboxes)]

# Optimización
solucion = minimize(sigmaT, semilla, method='SLSQP', bounds=bnds,
                    tol=1e-9, constraints=cons, jac=gradiente,
                    options={'maxiter': 2000, 'disp': True})


print("Cálculo de temperaturas:")
print(solucion)

Zfin = calculo_dseta(solucion.x)

print("\nConvergencias meridionales:")
print(Zfin)

print("Suma de convergencias = ", suma_dseta(solucion.x))

print("\nTemperaturas en ºC:")
print(solucion.x-273.15)

print("\nTemp. promedio:", np.average(solucion.x)-273.15)


#################
##  GRÁFICOS   ##
#################

plt.rcParams['text.usetex'] = True
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)

ticks = [-90, -60, -30, 0, 30, 60, 90]

plt.plot(dcajas[:, 0], solucion.x-273.15, marker="o", ls="")
plt.xticks(ticks)
plt.grid(True, color='0.95')
# plt.title("Distribución de temperaturas")
plt.xlabel("Latitud [$^\circ$]", fontsize=16)
plt.ylabel("Temperatura [ºC]", fontsize=16)
plt.savefig("temperaturas.pdf")
plt.show()

plt.plot(dcajas[:, 0], Zfin, marker="o", ls="")
plt.xticks(ticks)
plt.grid(True, color='0.95')
# plt.title("Convergencia de flujos meridionales")
plt.xlabel("Latitud [$^\circ$]", fontsize=16)
plt.ylabel("$\zeta$ [W/m$^2$]", fontsize=16)
plt.savefig("convergencias.pdf")
plt.show()

# Cálculo de LW final
vector_LW = LWA(solucion.x)
print(vector_LW)

# SW de Fukumura
SW_fuku = np.loadtxt("SWfuku.dat")

plt.plot(dcajas[:, 0], vector_LW, label="$LW$")
plt.plot(dcajas[:,0], dcajas[:,1], label="$SW$")
plt.plot(SW_fuku[:,0], SW_fuku[:,1], label="$SW$ Fukumura y Ozawa")
plt.xticks(ticks)
plt.grid(True, color='0.95')
plt.legend(fontsize=10)
# plt.title("LWR en TOA")
plt.xlabel("Latitud [$^\circ$]", fontsize=16)
plt.ylabel("$LW$ y $SW$ en TOA [W/m$^2$]", fontsize=16)
plt.savefig("comp_LWySW.pdf")
plt.show()


######################################
##  Comparación con observaciones   ##
######################################

import xarray as xr
url = 'http://apdrc.soest.hawaii.edu:80/dods/public_data/Reanalysis_Data/NCEP/NCEP/clima/'
#url = 'http://apdrc.soest.hawaii.edu/dods/public_data/Reanalysis_Data/NCEP/NCEP/daily/'
ncep_Ts = xr.open_dataset(url + 'surface_gauss/skt')
#ncep_Ts = xr.open_dataset(url + 'surface_gauss/air')
T_obs = ncep_Ts.skt.mean(dim=('lon', 'time'))
#T_obs = ncep_Ts.air.mean(dim=('lon', 'time'))

T_fuku = np.loadtxt("observaciones.dat")


plt.plot(dcajas[:, 0], solucion.x-273.15, marker="o", ls="")
plt.plot(T_obs.lat, T_obs.values)
plt.plot(T_fuku[:,0], T_fuku[:,1]-273.15)
plt.xticks(ticks)
plt.grid(True, color='0.95')
# plt.title("Distribución de temperaturas")
plt.xlabel("Latitud [$^\circ$]", fontsize=16)
plt.ylabel("Temperatura [ºC]", fontsize=16)
plt.savefig("temperaturas_comp.pdf")
plt.show()
