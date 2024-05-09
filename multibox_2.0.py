#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import SWA
import xarray as xr

###########################
# Constantes del problema #
###########################

acels = 208.0         # Parámetro optico a en W/m²ºC
b = 1.9               # Parámetro óptico b
a = acels-b*273.15    # Parámetro óptico a en W/m²K
RT = 6373000.0        # Radio terrestre en m


def LWA(T):
    # Radiación infrarroja que sale al espacio
    return a+b*T

def area(latgrados, ncajas):
    # lat es la latitud central de la zona.
    # ncajas es la cantidad de celdas del problema.
    lat = np.deg2rad(latgrados)
    ancho = np.pi / ncajas
    phi1 = lat - ancho/2
    phi2 = lat + ancho/2
    return 2*np.pi * np.abs(np.sin(phi2) - np.sin(phi1))
    

# Voy a leer el archivo "latitudes.dat" cuyas columnas son:
# latitud y albedo
albedo_vs_latitud = np.loadtxt('data/latitudes-new.dat')
nboxes = albedo_vs_latitud.shape[0]              # Número de cajas
print("Se han detectado ", nboxes, " cajas\n")


# Creo la matriz 'dcajas', de 'nboxes' filas, cuyas columnas son:
# lat, SWA, LWA, temp, area


dcajas = np.zeros((nboxes,5))
dseta = np.zeros((nboxes))    # Vector de convergencias

divisiones = np.arange(0,nboxes)  # Grilla de números naturales de 0 a nboxes

# Defino las temperaturas iniciales. Empiezo con T homogénea.
for i in divisiones:
    dcajas[i, 3] = 200.  # Kelvin


print("Latitud Media \t Albedo \t SWA [W/m²] \t Área [m²/Radio terr]")

for ii in divisiones:
    lat = albedo_vs_latitud[ii, 0]  # Se los paso en grados
    albedo = albedo_vs_latitud[ii, 1]
    dcajas[ii, :3] = [lat, SWA.SWA_calc(lat, albedo), LWA(dcajas[ii, 3])]
    dcajas[ii, 4] = area(lat, nboxes)
    print(f"{lat:^+13.2f} \t {albedo:^6.2f} \t {dcajas[ii,1]:^10.4f} \t {dcajas[ii,4]:^10.3E}")

def calculo_dseta(vtemp):
    for i in divisiones:
        dseta[i] = (LWA(vtemp[i]) - dcajas[i, 1])*dcajas[i,4]
    return dseta


def suma_dseta(vtemp):
    dseta = calculo_dseta(vtemp)
    sumaf = np.sum(dseta)
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
        df[i] = b*dcajas[i,4]/t[i]-f[i]/t[i]**2
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


print("\nCálculo de temperaturas:\n")
print(solucion)

Zfin = calculo_dseta(solucion.x) * RT * RT

print("\nConvergencias meridionales:\n")
print(Zfin)

print("\nSuma de convergencias = ", suma_dseta(solucion.x) * RT * RT)

print("\nTemperaturas en ºC:\n")
print(solucion.x-273.15)

print("\nTemp. promedio:", np.average(solucion.x)-273.15)


#################
##  GRÁFICOS   ##
#################


plt.rcParams['text.usetex'] = True
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rcParams["grid.color"]= "#DDDDDD"
plt.rcParams["grid.linewidth"] = 0.8


ticks = [-90, -60, -30, 0, 30, 60, 90]

plt.plot(dcajas[:, 0], solucion.x-273.15, marker="o", ls="", markersize=4)
plt.xticks(ticks)
plt.grid(True, color='0.95')
# plt.title("Distribución de temperaturas")
plt.xlabel("Latitud [$^\circ$]", fontsize=16)
plt.ylabel("Temperatura [ºC]", fontsize=16)
plt.savefig("img/temperaturas.pdf")
plt.show()

plt.plot(dcajas[:, 0], np.abs(Zfin), marker="o", ls="", markersize=4)
plt.xticks(ticks)
plt.yscale('log')
plt.grid(True, color='0.95')
# plt.title("Convergencia de flujos meridionales")
plt.xlabel("Latitud [$^\circ$]", fontsize=16)
plt.ylabel("$|\zeta|\cdot\mathcal{A}$ [W]", fontsize=16)
plt.tight_layout()
plt.savefig("img/convergencias.pdf")
plt.show()



# Cálculo de LW final
vector_LW = LWA(solucion.x)
print(vector_LW)
# Observaciones CERES
ceres = xr.open_dataset("data/CERES_EBAF-TOA_Ed4.2_Subset_200003-202310.nc")
LW_ceres = ceres.ztoa_lw_all_mon.mean(dim='time')


plt.plot(dcajas[:, 0], vector_LW, label="$LW$ MEP")
plt.plot(LW_ceres.lat, LW_ceres, label="$LW$ CERES")
plt.plot(dcajas[:,0], dcajas[:,1], label="$SW$ MEP")
plt.xticks(ticks)
plt.grid(True, color='0.95')
plt.legend(fontsize=10)
# plt.title("LWR en TOA")
plt.xlabel("Latitud [$^\circ$]", fontsize=16)
plt.ylabel("$LW$ y $SW$ en TOA [W/m$^2$]", fontsize=16)
plt.tight_layout()
plt.savefig("img/comp_LWySW.pdf")
plt.show()


######################################
##  Comparación con observaciones   ##
######################################


ncep_Ts = xr.open_dataset('data/skt.sfc.mon.ltm.1991-2020.nc')
T_obs = ncep_Ts.skt.mean(dim=('lon', 'time'))
T_desvest = ncep_Ts.skt.std(dim=('lon','time'))

T_fuku = np.loadtxt("data/MEP-Fukumura.dat")


plt.plot(dcajas[:, 0], solucion.x-273.15, marker="o", ls="", label="MEP", markersize=4)
plt.plot(T_obs.lat, T_obs.values, label="NCEP Reanalysis 1991-2020")
plt.fill_between(T_obs.lat, (T_obs-T_desvest).values, (T_obs+T_desvest).values, color="tab:orange", alpha=0.15)
plt.plot(T_fuku[:,0], T_fuku[:,1]-273.15, label="MEP Fukumura y Ozawa (2014)",  ls="--", markersize=4)
plt.xticks(ticks)
plt.grid(which='major')
plt.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5)
plt.minorticks_on()

plt.legend(fontsize=10)
# plt.title("Distribución de temperaturas")
plt.xlabel("Latitud [$^\circ$]", fontsize=16)
plt.ylabel("Temperatura [ºC]", fontsize=16)
plt.tight_layout()
plt.savefig("img/temperaturas_comp.pdf")
plt.show()


########################################
## Cálculo de los Flujos meridionales ##
########################################

def calculo_H(vlat, zetaA):
    # Calcula los flujos meridionales H
    # a partir de las convergencias.
    # vlat: vector de latitudes
    # zetaA: vector de convergencias multiplicadas
    # por el área de la celda

    nboxes = zetaA.size
    puntos = np.arange(0,nboxes-1)
    
    flujosH = []
    flujosH.append(0)
    
    latitudes = []
    latitudes.append(-90)
    
    for i in puntos:
        fluxH = -(zetaA[i]-flujosH[i])
        flujosH.append(fluxH)
        latH = (vlat[i] + vlat[i+1]) * 0.5
        latitudes.append(latH)

    flujosH.append(-(zetaA[nboxes-1]-flujosH[nboxes-1]))
    latitudes.append(90)
    
    tabla = np.zeros((nboxes+1,2))
    tabla[:,0] = latitudes
    tabla[:,1] = flujosH

    return tabla


flujosH = calculo_H(dcajas[:,0], Zfin)

"""
incoming = ceres.zsolar_mon.mean(dim='time')
reflected = ceres.ztoa_sw_all_mon.mean(dim='time')
SW_ceres = incoming - reflected
Z_ceres = LW_ceres - SW_ceres
ZA_ceres = Z_ceres*area(Z_ceres.lat, Z_ceres.lat.size)*RT*RT

flujos_ceres = calculo_H(Z_ceres.lat, ZA_ceres)
"""

trenberth = np.loadtxt("data/trenberth_total.dat")

plt.plot(flujosH[:,0], flujosH[:,1]*1e-15, label="MEP")
plt.plot(trenberth[:,0], trenberth[:,1], label="Fasullo y Trenberth (2008)",  ls="--", markersize=4)
plt.xticks(ticks)
plt.legend()
plt.grid()
plt.xlabel("Latitud [$^\circ$]", fontsize=16)
plt.ylabel("$\mathcal{H}$ [PW]", fontsize=16)
plt.tight_layout()
plt.savefig("img/flujo_meridional.pdf")
plt.show()


########################################
## Exporto temperaturas a un archivo  ##
########################################

tronix = np.zeros((nboxes, 2))
tronix[:,0] = dcajas[:, 0]
tronix[:,1] = solucion.x-273.15
np.savetxt('data/temperaturasMEP.dat', tronix)
