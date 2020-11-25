# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 08:12:53 2020

@author: Atharva Kulkarni
"""

import numpy as np
from numpy import cos as cos
from numpy import sin as sin
import matplotlib.pyplot as plt
from scipy.integrate import odeint


# function to convert given state vector to orbital elements:
Y = [6878,0,0,0,8.5,0]
mu = 398600
def state_to_orbital(Y,mu):
    x,y,z,x_dot,y_dot,z_dot = Y
    r_bar = np.array([x,y,z])
    v_bar = np.array([x_dot,y_dot,z_dot])
    r = np.sqrt(x**2 + y**2 + z**2)
    v= np.sqrt( x_dot**2 + y_dot**2 + z_dot**2)
    a = 1/(2/r - (v**2)/mu)
    
    e_bar = (v**2/mu - 1/r)* r_bar - ((1/mu)*(np.dot(r_bar, v_bar)))*v_bar
    e = np.sqrt(np.dot(e_bar,e_bar))
    
    i = 180/np.pi*np.arccos(np.dot(np.cross(r_bar,v_bar),[0,0,1])/np.sqrt(np.dot(np.cross(r_bar,v_bar),np.cross(r_bar,v_bar))))
    
    raan = 180/np.pi*np.arccos(np.dot([1,0,0],np.cross([0,0,1],np.cross(r_bar,v_bar))/(np.sqrt(np.dot(np.cross([0,0,1],np.cross(r_bar,v_bar)),np.cross([0,0,1],np.cross(r_bar,v_bar))))+1e-8))            
    
    if (np.dot(np.cross([1,0,0],np.cross([0,0,1],np.cross(r_bar,v_bar))/np.sqrt(np.dot(np.cross([0,0,1],np.cross(r_bar,v_bar)),np.cross([0,0,1],np.cross(r_bar,v_bar))))),[0,0,1]) < 0 )):
        raan = 360 - raan
    N_bar = np.cross([0,0,1],np.cross(r_bar,v_bar))
    N_cap = N_bar/(np.sqrt(np.dot(N_bar,N_bar)))
    w = 180/np.pi*np.arccos(np.dot(N_cap, e_bar/e))
    w_cap = np.cross(r_bar, v_bar)/np.sqrt(np.dot(np.cross(r_bar,v_bar),np.cross(r_bar,v_bar)))
    if (np.dot(np.cross(N_cap, e_bar/e), w_cap) < 0):
        w = 360 - w
    
    ta = 180/np.pi*np.arccos(np.dot(r_bar/r,e_bar/e))
    if np.dot(w_cap, np.cross(e_bar/e, r_bar/r)) < 0:
        ta = 360 - ta
    
    return [a, e, i, raan, w, ta]

# function to convert given  orbital elements to state vector:
def orbital_to_state(elements, mu):
    a, e, i, raan, w, ta = elements
    i, raan, w, ta = np.pi/180* np.asarray([i, raan, w, ta])
    H = np.sqrt(mu*a*(1-e**2))
    r = H**2/((1+ e*cos(ta))*mu)
    xp = r*cos(ta)
    yp = r*sin(ta)
    zp = 0
    
    r_dot = mu*e*sin(ta)/H
    ta_dot = mu*(1+e*cos(ta))/(H*r)
    xp_dot = r_dot*cos(ta) - r* sin(ta)*ta_dot
    yp_dot = r_dot*sin(ta) + r* cos(ta)*ta_dot
    zp_dot = 0
    
    R = np.array([[cos(raan)*cos(w) - sin(raan)*sin(w)*cos(i), -cos(raan)*sin(w) - sin(raan)*cos(w)*cos(i), sin(raan)*sin(i)],
         [sin(raan)*cos(w) + cos(raan)*sin(w)*cos(i), -sin(raan)*sin(w) + cos(raan)*cos(w)*cos(i), -cos(raan)*sin(i)],
         [sin(i)*sin(w), sin(i)*cos(w), cos(i)]])
    
    x, y, z = np.dot(R, [xp, yp, zp ])
    x_dot, y_dot, z_dot = np.dot(R, [xp_dot, yp_dot, zp_dot])
    
    return x, y, z, x_dot, y_dot, z_dot

# function to find eccentric anomaly given mean anomaly:
def mean_to_eccentric(M,e):
    E = M
    def f(M,E,e): 
        return E - e* sin(E) - M
        
    def f_dot(E,e): 
        return 1 - e*cos(E)
    
    while( abs(f(M,E,e)) > 1e-7):
        E = E - f(M,E,e)/f_dot(E,e)
    return E


# function to find orbital elements after time t for given orbital elements at current epoch:
def propogate(elements, mu, t):
    a, e, i, raan, w, ta = elements
    E0 = 2*np.arctan(np.sqrt((1-e)/(1+e))*np.tan(np.pi/180*ta/2))
    M0 = E0 - e*sin(E0)

    n = np.sqrt(mu/a**3)
    M = M0 + n*t
    E = mean_to_eccentric(M,e)
    ta = 180/np.pi*2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(E/2))
    if ta<0:
        ta = 360 + ta
    elements = a,e,i,raan,w,ta
    Y = orbital_to_state(elements,mu)
    return elements, Y, E, M

