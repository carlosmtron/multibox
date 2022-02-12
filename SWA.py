#!/usr/bin/python3

from math import pi
import numpy as np
from scipy.integrate import quad


###########################
# Constantes del problema #
###########################

tilt = 0.408407044966673121 # Axial tilt (rad)
t0   = 3.15576e7  # Período de traslación (días)
S    = 1360       # Constante Solar (W/m²)


###############
#  Funciones  # 
###############

def dec(t):
    return np.arcsin(np.sin(tilt)*np.sin(2*pi*t/t0))


def cos_h0(t,lat):
    val=-np.tan(lat)*np.tan(dec(t))
    return -np.tan(lat)*np.tan(dec(t))


def SWA_dia(t, lat, albedo):
    lat_rad = lat * pi/180.0
    s_lat = np.sin(lat_rad)
    c_lat = np.cos(lat_rad)
    coef1 = S*(1-albedo)/pi
    if cos_h0(t,lat_rad) > 1:
      return 0
    elif cos_h0(t,lat_rad) < -1:
      return S*(1-albedo)*np.sin(dec(t))*s_lat
    else:
      valor = coef1*(np.arccos(cos_h0(t,lat_rad))*s_lat*np.sin(dec(t)) + c_lat*np.cos(dec(t))*np.sin(np.arccos(cos_h0(t,lat_rad))))
      return valor


def SWA_calc(lat, albedo):
    integral = quad(SWA_dia, 0, t0, args=(lat, albedo))
    return 1/t0 * integral[0]


# Las siguientes líneas sirven para probar el Script sin necesidad de ejecutar el main.
if __name__ == '__main__':
    lat = -90
    FSWA = SWA_calc(lat, 0.31)
    print("El valor medio de la radiación anual absorbida para la latitud", lat, "grados es:")
    print(FSWA, "W/m²")
