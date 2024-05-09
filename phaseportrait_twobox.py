"""
Este script sirve para graficar retratos de fase. Las ecuaciones del sistema deben ser de la forma
Ta' = f1(Ta, Tb)
Tb' = f2(Ta, Tb)
El sistema se ingresa en el return de la función f del script.
Exporta un archivo pdf

Autor: Carlos Silva
csilva@fceia.unr.edu.ar
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 14})


k = 10
c = 2e+08
SWA = 305.7
SWB = 177.4
a_celsius = 207.42
b_celsius = 1.93
b = b_celsius
a = a_celsius - b*273.15

def f(Temp, t):
    Ta, Tb = Temp
    return [(SWA -(a+b*Ta) - k*(Ta-Tb))/c, (SWB -(a+b*Tb) + k*(Ta-Tb))/c]

y1 = np.linspace(-40.0, 60.0, 20)
y2 = np.linspace(-40.0, 60.0, 20)

Y1, Y2 = np.meshgrid(y1, y2)

t = 0

u, v = np.zeros(Y1.shape), np.zeros(Y2.shape)

NI, NJ = Y1.shape

for i in range(NI):
    for j in range(NJ):
        # La grilla está en ºC pero a la f debemos pasarle T en kelvin
        x = Y1[i, j] + 273.15
        y = Y2[i, j] + 273.15
        yprima = f([x, y], t)
        u[i,j] = yprima[0]
        v[i,j] = yprima[1]
     


speed = np.sqrt(u**2 + v**2)
#lw = 5*speed / speed.max()
# Comentar una de las dos líneas siguientes (o no)
S = plt.streamplot(Y1, Y2, u, v, color=speed, density=0.2, broken_streamlines=False, cmap='autumn_r')
# S = plt.streamplot(Y1, Y2, u, v, color=speed, density=2, cmap='autumn_r')
Q = plt.quiver(Y1, Y2, u, v, color='r')

plt.title('$k={s1}$ Wm$^{-2}$K$^{-1}$'.replace('s1','{}'.format(k)))
plt.xlabel('$T_A$ [ºC]')
plt.ylabel('$T_B$ [ºC]')
plt.axis('square')
plt.xlim([-40, 60])
plt.ylim([-40, 60])


plt.savefig('img/lorenz-retrato-fases-k={}.pdf'.format(k))
plt.show()
