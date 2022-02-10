#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import SWA


###########################
# Constantes del problema #
###########################

area = 1              # Área de cada sector
acels = 210           # Parámetro optico a en W/m²ºC
b = 2                 # Parámetro óptico b
a = acels-b*273.15    # Parámetro óptico a en W/m²K


def LWA(T):
    # Radiación infrarroja que sale al espacio
    return a+b*T


# Voy a leer el archivo "latitudes.dat" cuyas columnas son:
# latitud y albedo
albedo_vs_latitud = np.loadtxt('latitudes.dat')
nboxes = albedo_vs_latitud.shape[0]              # Número de cajas
print("Se han detectado ", nboxes, " cajas\n")


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

############################
##  Flujos a partir de ζ  ##
############################

fluxes = np.zeros((nboxes+1))
fluxes[0] = 0
fluxes[nboxes] = 0
latlim = np.loadtxt('latlim.txt')
for ii in np.arange(1, nboxes):
    fluxes[ii] = fluxes[ii-1] - Zfin[ii-1]


#################
##  GRÁFICOS   ##
#################


plt.plot(dcajas[:, 0], solucion.x-273.15, marker="o", ls="")
plt.grid(True, color='0.95')
plt.title("Distribución de temperaturas")
plt.xlabel("Latitud [º]")
plt.ylabel("Temperatura [ºC]")
plt.show()

plt.plot(dcajas[:, 0], Zfin, marker="o", ls="")
plt.grid(True, color='0.95')
plt.title("Convergencia de flujos meridionales")
plt.xlabel("Latitud [º]")
plt.ylabel("ζ [W/m²]")
plt.show()

plt.plot(latlim, fluxes, marker="o", ls="")
plt.grid(True, color='0.95')
plt.title("Flujo meridional")
plt.xlabel("Latitud [º]")
plt.ylabel("F [W/m²]")
plt.show()



#################################################
##  Cálculo de flujo meridional por integral   ##
#################################################

def inferred_heat_transport(energy_in, lat=None, latax=None):
    '''Compute heat transport as integral of local energy imbalance.
    Required input:
        energy_in: energy imbalance in W/m2, positive in to domain
    As either numpy array or xarray.DataArray
    If using plain numpy, need to supply these arguments:
        lat: latitude in degrees
        latax: axis number corresponding to latitude in the data
            (axis over which to integrate)
    returns the heat transport in PW.
    Will attempt to return data in xarray.DataArray if possible.
    '''
    from scipy import integrate
    from climlab import constants as const
    if lat is None:
        try: lat = energy_in.lat
        except:
            raise InputError('Need to supply latitude array if input data is not self-describing.')
    lat_rad = np.deg2rad(lat)
    coslat = np.cos(lat_rad)
    field = coslat*energy_in
    if latax is None:
        try: latax = field.get_axis_num('lat')
        except:
            raise ValueError('Need to supply axis number for integral over latitude.')
    #  result as plain numpy array
    integral = integrate.cumtrapz(field, x=lat_rad, initial=0., axis=latax)
    result = (1E-15 * 2 * np.math.pi * const.a**2 * integral)
    if isinstance(field, xr.DataArray):
        result_xarray = field.copy()
        result_xarray.values = result
        return result_xarray
    else:
        return result
