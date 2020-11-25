# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 07:11:08 2020

@author: Atharva Kulkarni
"""

import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import PlanetaryData as pd
from mpl_toolkits.mplot3d import Axes3D

class OrbitPropagator:
    def __init__(self,y0, central_body, tspan = 100 , propagator = 'two_body' ):
        self.y0 = y0
        self.t = np.linspace(0,tspan*60, tspan)
        self.cb = central_body
        self.propagator = propagator
        self.r0 = y0[0:3]
        self.v0 = y0[3:]
        if self.propagator == 'two_body':
            self.states = odeint(self.two_body,self.y0,self.t)
            self.rx, self.ry, self.rz, self.vx, self.vy, self.vz = self.states.T
    # diffy eqns:
    
    def two_body(self, y0, t):
        rx,ry,rz, vx, vy, vz = y0
        r = np.asarray([rx,ry,rz])
        r_mod = np.linalg.norm(r)
        ax,ay,az = -self.cb['mu']*r/(r_mod**3) 
        return vx,vy,vz,ax,ay,az
    
    #plotting the orbit
    
    def plot(self):
        fig = plt.figure()
        ax = fig.add_subplot(111,projection='3d')
        ax.set_aspect("equal")
        u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x = self.cb['r']*np.cos(u)*np.sin(v)
        y = self.cb['r']*np.sin(u)*np.sin(v)
        z = self.cb['r']*np.cos(v)
        ax.plot_surface(x, y, z, color="b", linewidth = 0.25)
        ax.plot(self.rx,self.ry,self.rz, label = 'Trajectory', color = 'r')
        ax.scatter(self.rx[0],self.ry[0],self.rz[0],label = 'Starting point', color = 'k')
        plt.title('Trajectory of spacecraft')
        plt.legend()
        plt.show()

