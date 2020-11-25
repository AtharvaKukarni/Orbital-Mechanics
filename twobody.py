# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 22:21:00 2020

@author: Atharva Kulkarni
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from mpl_toolkits.mplot3d import Axes3D
from OrbitPropagator import OrbitPropagator as op
import PlanetaryData as pd
import orbit_tools as ot
# defining constants

Re = 6378 #km
mu = 398600 #km^3/s^2



# defining diffy eqn

def two_body(y0,t,mu):
    rx,ry,rz,vx,vy,vz = y0
    r = np.asarray([rx,ry,rz])
    r_mod = np.linalg.norm(r)
    ax,ay,az = -mu*r/(r_mod**3)
    return vx, vy, vz, ax, ay, az

# defining initial conditions:

y0 = [Re + 500, 0, 0, 0, 8, 3]

# defining time space:

t = np.linspace(0, 90 * 60 , 900)

#solving the diffy eqn
    
sol = odeint(two_body, y0,t,args = (mu,))
rx,ry,rz,vx,vy,vz = sol.T
# plotting everything

def plot(r):
    rx,ry,rz = r
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.set_aspect("equal")
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = Re*np.cos(u)*np.sin(v)
    y = Re*np.sin(u)*np.sin(v)
    z = Re*np.cos(v)
    ax.plot_surface(x, y, z, color="b", linewidth = 0.25)
    ax.plot(rx,ry,rz, label = 'Trajectory', color = 'r')
    ax.scatter(rx[0],ry[0],rz[0],label = 'Starting point', color = 'k')
    plt.title('Trajectory of spacecraft')
    plt.legend()
    plt.show()
 
plot([rx,ry,rz])
    
o1 = op([9878,0,0,1,5,9], pd.earth, tspan = 200)

o1.plot()

sol = o1.states

_,sol1,_,_ = ot.propogate(ot.state_to_orbital([9878,0,0,1,5,9],398600), 398600,200*60)

s1 = ot.state_to_orbital([6878,0,0,0,8.5,0],398600)