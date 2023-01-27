#!/usr/bin/python3

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import SWA
import mapas
import gc


###########################
# Constantes del problema #
###########################

area = 1              # Área de cada sector
acels = 210           # Parámetro optico a en W/m²ºC
b = 2                 # Parámetro óptico b
a = acels-b*273.15    # Parámetro óptico a en W/m²K


# Datos de TOA provinientes de NCEP reanalysis
url = 'http://apdrc.soest.hawaii.edu:80/dods/public_data/Reanalysis_Data/NCEP/NCEP/clima/'
ncep_dswrf = xr.open_dataset(url + "other_gauss/dswrf")
ncep_uswrf = xr.open_dataset(url + "other_gauss/uswrf")
IncomingSWmap = ncep_dswrf.dswrf.mean(dim=('time'))
ReflectedSWmap = ncep_uswrf.uswrf.mean(dim='time')

Net_SW_map = IncomingSWmap - ReflectedSWmap
albedo_map = ReflectedSWmap / IncomingSWmap

print(albedo_map)

fig, axes, cx = mapas.make_map(albedo_map,plt.cm.Blues)

fig.suptitle('Albedo observado', fontsize=16)
axes[1].set_xlabel('Albedo')

plt.show()

nlats, nlongs = albedo_map.shape[0], albedo_map.shape[1]
nboxes = nlats*nlongs

temperaturas = 273.15 * xr.DataArray(np.ones((nlats,nlongs)),
                             dims=("lat","lon"),
                             coords={"lat": albedo_map.lat, "lon": albedo_map.lon})

print(temperaturas)
dseta = xr.DataArray(np.zeros((nlats,nlongs)), dims=("lat","lon"),
                             coords={"lat": albedo_map.lat, "lon": albedo_map.lon})

SW_map_stacked = Net_SW_map.stack(lls=('lat','lon'))

gc.collect()


def LWA(temp):
    # Radiación infrarroja que sale al espacio
    return a+b*temp

def calculo_dseta(temp):
    # Convergencia de flujos horizontales
    dseta = LWA(temp) - SW_map_stacked           # *area
    return dseta


def suma_dseta(temp):
    # Calcula la suma de flujos horizontales
    Zcal = calculo_dseta(temp)
    sumaf = Zcal.sum().values
    return sumaf


def sigmaT(temp, info={'Nfeval': 0}):
    # Devuelve la tupla (sigma, gradiente)
    # La función objetivo a minimizar es el opuesto
    # de la producción de entropía
    print('entré')
    factual = calculo_dseta(temp)
    entrop_celda = factual / temp
    suma = entrop_celda.sum().values
    # Calculo el gradiente de la función objetivo
    # que es el -grad(sigmaT)
    df = (b*area/temp - factual/temp**2).values
    # display information
    print(info['Nfeval'], -suma, -df)
    info['Nfeval'] += 1
    return -suma, -df




cons = ({'type': 'eq', 'fun': suma_dseta})

# Semilla
semilla = temperaturas.stack(lls=('lat','lon'))

sigma_cero, gradiente_cero = sigmaT(semilla)

print("Semilla -->", semilla)
print("-Producción de entropía de la semilla -->", sigma_cero)


bnds = [(3, 500) for i in range(nboxes)]


# Optimización
solucion = minimize(sigmaT, semilla.values,
                    args=({'Nfeval':0,}),
                    method='SLSQP', jac=True,
                    bounds=bnds, constraints=cons,
                    options={'maxiter': 500, 'iprint': 99, 'disp': True})


reordenando = xr.DataArray(solucion.x, dims=semilla.dims, coords=semilla.coords)
temperaturas_map = reordenando.unstack("lls")










"""
Net_SW_map = IncomingSWmap - ReflectedSWmap
albedo_map = ReflectedSWmap / IncomingSWmap

print(albedo_map)

fig, axes, cx = mapas.make_map(albedo_map,plt.cm.Blues)

fig.suptitle('Albedo observado', fontsize=16)
axes[1].set_xlabel('Albedo')

plt.show()

nlats, nlongs = albedo_map.shape[0], albedo_map.shape[1]
nboxes = nlats*nlongs

temperaturas = 273.15 * xr.DataArray(np.ones((nlats,nlongs)),
                             dims=("lat","lon"),
                             coords={"lat": albedo_map.lat, "lon": albedo_map.lon})

print(temperaturas)
dseta = xr.DataArray(np.zeros((nlats,nlongs)), dims=("lat","lon"),
                             coords={"lat": albedo_map.lat, "lon": albedo_map.lon})

def LWA(temp):
    # Radiación infrarroja que sale al espacio
    return a+b*temp

def calculo_dseta(temp):
    dseta = LWA(temp) - Net_SW_map           # *area
    return dseta


def suma_dseta(temp):
    Zcal = calculo_dseta(temp)
    sumaf = Zcal.sum(dim=('lat','lon'))
    return sumaf


def sigmaT(temp):
    # La función objetivo a minimizar es el opuesto
    # de la producción de entropía
    factual = calculo_dseta(temp)
    entrop_celda = factual / temp
    suma = entrop_celda.sum(dim=('lat','lon')).values
    return -suma

print(sigmaT(temperaturas))

cons = ({'type': 'eq', 'fun': suma_dseta})

# Semilla
semilla = temperaturas.stack(lls=('lat','lon'))

bnds = [(3, 600) for i in range(nboxes)]

# Optimización
solucion = minimize(sigmaT, semilla, method='SLSQP', bounds=bnds,
                    tol=1e-9, constraints=cons,
                    options={'maxiter': 2000, 'disp': True})



# Creo la matriz 'dcajas', de 'nboxes' filas, cuyas columnas son:
# lat, SWA, LWA, temp


dcajas = np.zeros((nboxes,4))
dseta = np.zeros((nboxes))    # Vector de convergencias

divisiones = np.arange(0,nboxes)  # Grilla de números naturales de 0 a nboxes

# Defino las temperaturas iniciales. Empiezo con T homogénea.
for i in divisiones:
    dcajas[i, 3] = 200  # Kelvin


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
"""
