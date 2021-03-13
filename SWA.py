#!/usr/bin/python3

from math import pi
import numpy as np
from scipy.integrate import quad


###########################
# Constantes del problema #
###########################

tilt = 0.408407044966673121 # Axial tilt (rad)
t0   = 3.15576e7  # Período de traslación (segundos)
S    = 1360       # Constante Solar (W/m²)


###############
#  Funciones  # 
###############

def dec(t):
    return np.arcsin(np.sin(tilt)*np.sin(2*pi*t/t0))


def h0(t,lat):
    return np.arccos(-np.tan(lat)*np.tan(dec(t)))

def SWA_calc(lat, albedo):
    ####################
    #  Optimizaciones  #
    ####################
    lat_rad = lat * pi/180.0
    s_lat = np.sin(lat_rad)
    c_lat = np.cos(lat_rad)
    coef1 = S*(1-albedo)/(pi*t0)
    #
    f = lambda t: h0(t,lat_rad)*s_lat*np.sin(dec(t))+c_lat*np.cos(dec(t))*np.sin(h0(t,lat_rad))
    integral = quad(f,0,t0)
    return coef1*integral[0]

# Las siguientes líneas sirven para probar el Script sin necesidad de ejecutar el main.
if __name__ == '__main__':
    lat = -30.0
    FSWA = SWA_calc(lat, 0.34)
    print("La integral de la radiación anual recibida para la latitud", lat, "grados es:")
    print(FSWA)
