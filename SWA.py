#!/usr/bin/python3

from math import pi
import numpy as np
from scipy.integrate import quad


###########################
# Constantes del problema #
###########################

theta = 0.408407044966673121 # Axial tilt (rad)
t0 = 3.15576e7  # Período de traslación (segundos)
S  = 1360      # Constante Solar (W/m²)
a  = -302       # Parámetro optico a
b  = 1.87       # Parámetro óptico b


###############
#  Funciones  # 
###############

def dec(t):
    return np.arcsin(np.sin(theta)*np.sin(2*pi*t/t0))


def h0(t,lat):
    return np.arccos(-np.tan(lat)*np.tan(dec(t)))

def SWA_calc(lat, albedo):
    ####################
    #  Optimizaciones  #
    ####################
    s_lat = np.sin(lat)
    c_lat = np.cos(lat)
    coef1 = S*(1-albedo)/(pi*t0)
    #
    f = lambda t: h0(t,lat)*s_lat*np.sin(dec(t))+c_lat*np.cos(dec(t))*np.sin(h0(t,lat))
    integral = quad(f,0,t0)
    return coef1*integral[0]

