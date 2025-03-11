
import numpy as np


def deg2rad(d):

    return d*np.pi/180

def rad2deg(r):

    return r*180/np.pi


def rad_sec_to_rpm(rs):

    return 60*rs/(2*np.pi)

def deg_sec_to_rpm(ds):
    
    return 60*ds/360

def rpm_to_deg_sec(rpm):

    return rpm*360/60
    
