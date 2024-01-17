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
# RT = 6373000.0        # Radio terrestre en m


def LWA(T):
    # Radiación infrarroja que sale al espacio
    return a+b*T

"""
def area(lat, ncajas):
    # lat es la latitud central de la zona.
    # ncajas es la cantidad de celdas del problema.
    ancho = 180. / ncajas
    phi1 = lat - ancho/2
    phi2 = lat + ancho/2
    return 2*np.pi * RT**2 * np.abs(np.cos(phi2) - np.cos(phi1))
"""    

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
    dcajas[i, 3] = 200.  # Kelvin


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

plt.plot(dcajas[:, 0], solucion.x-273.15, marker="o", ls="", markersize=4)
plt.xticks(ticks)
plt.grid(True, color='0.95')
# plt.title("Distribución de temperaturas")
plt.xlabel("Latitud [$^\circ$]", fontsize=16)
plt.ylabel("Temperatura [ºC]", fontsize=16)
plt.savefig("temperaturas.pdf")
plt.show()

plt.plot(dcajas[:, 0], Zfin, marker="o", ls="", markersize=4)
plt.xticks(ticks)
plt.grid(True, color='0.95')
# plt.title("Convergencia de flujos meridionales")
plt.xlabel("Latitud [$^\circ$]", fontsize=16)
plt.ylabel("$\zeta$ [W/m$^2$]", fontsize=16)
plt.tight_layout()
plt.savefig("convergencias.pdf")
plt.show()

# Cálculo de LW final
vector_LW = LWA(solucion.x)
print(vector_LW)

# SW de Fukumura
# SW_fuku = np.loadtxt("SWfuku.dat")

plt.plot(dcajas[:, 0], vector_LW, label="$LW$")
plt.plot(dcajas[:,0], dcajas[:,1], label="$SW$")
# plt.plot(SW_fuku[:,0], SW_fuku[:,1], label="$SW$ Fukumura y Ozawa")
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

ncep_Ts = xr.open_dataset('data/skt.sfc.mon.ltm.1991-2020.nc')
T_obs = ncep_Ts.skt.mean(dim=('lon', 'time'))

T_fuku = np.loadtxt("MEP-Fukumura.dat")


plt.plot(dcajas[:, 0], solucion.x-273.15, marker="o", ls="", label="MEP", markersize=4)
plt.plot(T_obs.lat, T_obs.values, label="NCEP Reanalysis 1991-2020")
plt.plot(T_fuku[:,0], T_fuku[:,1]-273.15, label="MEP Fukumura y Ozawa (2014)",  ls="--", markersize=4)
plt.xticks(ticks)
plt.grid(True, color='0.95')
plt.legend(fontsize=10)
# plt.title("Distribución de temperaturas")
plt.xlabel("Latitud [$^\circ$]", fontsize=16)
plt.ylabel("Temperatura [ºC]", fontsize=16)
plt.savefig("temperaturas_comp.pdf")
plt.show()


########################################
## Cálculo de los Flujos meridionales ##
########################################

puntos = np.arange(0,nboxes-1)

flujosF = []
flujosF.append(0)

latitudes = []
latitudes.append(-90)

for i in puntos:
    fluxF = -(Zfin[i]-flujosF[i])
    flujosF.append(fluxF)
    latF = (dcajas[i,0]+dcajas[i+1,0])*0.5
    latitudes.append(latF)

flujosF.append(0)
latitudes.append(90)

tabla = np.zeros((nboxes+1,2))
tabla[:,0] = latitudes
tabla[:,1] = flujosF

print("\n Flujos meridionales")
print(tabla)

plt.plot(latitudes, flujosF)
plt.xticks(ticks)
plt.grid()
plt.xlabel("Latitud [$^\circ$]", fontsize=16)
plt.ylabel("Flujo de energía meridional [W/m$^2$]", fontsize=16)
plt.tight_layout()
plt.savefig("flujo_meridional.pdf")
plt.show()

# Vamos a muliplicar por el área zonal para poderlo comparar

const_a = 6373000.0 #Radio terrestre

fluxH = []
for f_i in flujosF:
    if f_i == 0:
        fluxH.append(0)
    else:
        hi = f_i*2*np.pi*const_a**2 * np.abs(
            np.cos(np.deg2rad(dcajas[i+1,0]))-np.cos(np.deg2rad(dcajas[i,0])))
        fluxH.append(1E-15*hi)


plt.plot(latitudes, fluxH)
plt.xticks(ticks)
plt.grid()
plt.xlabel("Latitud [$^\circ$]", fontsize=16)
plt.ylabel("Flujo de energía meridional [PW]", fontsize=16)
plt.tight_layout()
#plt.savefig("flujo_meridional.pdf")
plt.show()


#################################################
##  Cálculo de flujo meridional por integral   ##
#################################################

def inferred_heat_transport(energy_in, lat=None, latax=None):
    from scipy import integrate

    const_a = 6373000.0 #Radio terrestre
    
    if lat is None:
        lat = energy_in.lat

    lat_rad = np.deg2rad(lat)
    coslat = np.cos(lat_rad)
    field = coslat*energy_in
    if latax is None:
        latax = field.get_axis_num('lat')
    integral = integrate.cumtrapz(field, x=lat_rad, initial=0., axis=latax)
    result = (1E-15 * 2 * np.pi * const_a**2 * integral)
    # Si la entrada era un xarray, devuelve un xarray
    if isinstance(field, xr.DataArray):
        result_xarray = field.copy()
        result_xarray.values = result
        return result_xarray
    else:
        return result


data = xr.DataArray(-Zfin, dims=("lat"), coords={"lat": dcajas[:, 0]})
print(data)

flujo_meridional = inferred_heat_transport(data)

print(flujo_meridional)

plt.plot(flujo_meridional.lat,flujo_meridional.values)
plt.xticks(ticks)
plt.grid(True, color='0.95')
plt.title("Flujo meridional")
plt.xlabel("Latitud [º]")
plt.ylabel("Flujo meridional [PW]")
plt.show()


"""
Las siguientes líneas corrigen un desbalance en los datos que
provienen de NCEP reanalysis.
"""

lat_ncep = flujo_meridional.lat

#  global average of TOA radiation in reanalysis data
weight_ncep = np.cos(np.deg2rad(lat_ncep)) / np.cos(np.deg2rad(lat_ncep)).mean(dim='lat')
imbal_ncep = (data * weight_ncep).mean(dim='lat')
print('The net downward TOA radiation flux in NCEP renalysis data is %0.1f W/m².' %imbal_ncep)

convergencia_balanceada = data - imbal_ncep
newimbalance = float((convergencia_balanceada * weight_ncep).mean(dim='lat'))
print('The net downward TOA radiation flux after balancing the data is %0.2e W/m².' %newimbalance)

fig, ax = plt.subplots()
ax.plot(lat_ncep, inferred_heat_transport(convergencia_balanceada))
ax.set_ylabel('PW')
ax.set_xlabel('Latitud [º]')
ax.set_xticks(ticks)
ax.grid()
ax.set_title('Transporte de energía meridional inferido')
plt.show()
