import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import SWA
import mapas



###########################
# Constantes del problema #
###########################

area = 1              # Área de cada sector
acels = 210           # Parámetro optico a en W/m²ºC
b = 2                 # Parámetro óptico b
a = acels-b*273.15    # Parámetro óptico a en W/m²K


# Datos de TOA provinientes de NCEP reanalysis
# url = 'http://apdrc.soest.hawaii.edu:80/dods/public_data/Reanalysis_Data/NCEP/NCEP/clima/'
# ncep_dswrf = xr.open_dataset(url + "other_gauss/dswrf")
# ncep_uswrf = xr.open_dataset(url + "other_gauss/uswrf")
# IncomingSWmap = ncep_dswrf.dswrf.mean(dim=('time'))
# ReflectedSWmap = ncep_uswrf.uswrf.mean(dim='time')
latis = [-90.0, -60.0, -40.0, -20.0, 20.0, 40.0, 60.0, 90.0]
longis = [-135, -90, -45, 0, 45, 90, 135, 180]
IncomingSWmap = xr.DataArray([[50, 50, 50, 50, 50, 50, 50, 50],
                              [70, 70, 70, 70, 70, 70, 70, 70],
                              [100, 100, 100, 100, 100, 100, 100, 100],
                              [350, 350, 350, 350, 350, 350, 350, 350],
                              [360, 360, 360, 360, 360, 360, 360, 360],
                              [180, 180, 180, 180, 180, 180, 180, 180],
                              [75, 75, 75, 75, 75, 75, 75, 75],
                              [50, 50, 50, 50, 50, 50, 50, 50]], dims=("lat","lon"),
                             coords={"lat": latis, "lon": longis})
ReflectedSWmap = xr.DataArray([[45, 45, 49, 48, 45, 48, 44, 45],
                              [50, 45, 50, 45, 42, 43, 45, 50],
                              [90, 85, 86, 85, 84, 85, 85, 88],
                              [120, 110, 120, 130, 120, 110, 115, 118],
                              [99, 98, 99, 105, 115, 110, 110, 105],
                              [100, 110, 150, 99, 120, 150, 130, 120],
                              [45, 44, 48, 44, 40, 40, 40, 35],
                              [45, 45, 49, 48, 45, 48, 44, 45]], dims=("lat","lon"),
                             coords={"lat": latis, "lon": longis})
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


def sigmaT(temp):
    # La función objetivo a minimizar es el opuesto
    # de la producción de entropía
    factual = calculo_dseta(temp)
    entrop_celda = factual / temp
    suma = entrop_celda.sum().values
    return -suma


cons = ({'type': 'eq', 'fun': suma_dseta})

# Semilla
semilla = temperaturas.stack(lls=('lat','lon'))


print(semilla)
print("-Producción de entropía de la semilla =", sigmaT(semilla))


bnds = [(3, 600) for i in range(nboxes)]

# Optimización
solucion = minimize(sigmaT, semilla.values, method='SLSQP', bounds=bnds,
                    tol=1e-9, constraints=cons,
                    options={'maxiter': 2000, 'disp': True})


reordenando = xr.DataArray(solucion.x, dims=semilla.dims, coords=semilla.coords)
temperaturas_map = reordenando.unstack("lls")

print(temperaturas_map)

fig, axes, cx = mapas.make_map(temperaturas_map, plt.cm.coolwarm)

fig.suptitle('Temperatura estimada', fontsize=16)
axes[1].set_xlabel('Temperatura')

plt.show()
