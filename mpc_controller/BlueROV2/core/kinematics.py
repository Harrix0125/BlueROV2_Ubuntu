import numpy as np
import casadi as cas

def get_J1(phi, theta, psi):
    """ Rotation Matrix (Body -> World) using CasADi SX """
    cphi, sphi = cas.cos(phi), cas.sin(phi)
    cth, sth  = cas.cos(theta), cas.sin(theta)
    cpsi, spsi = cas.cos(psi), cas.sin(psi)
    
    return cas.vertcat(
        cas.horzcat(cpsi*cth, -spsi*cphi + sphi*sth*cpsi, spsi*sphi + sth*cpsi*cphi),
        cas.horzcat(spsi*cth,  cpsi*cphi + sphi*sth*spsi, -cpsi*sphi + sth*spsi*cphi),
        cas.horzcat(-sth,      sphi*cth,                  cphi*cth)
    )

def get_J2(phi, theta, psi):
    """ Angular Velocity Transformation Matrix (CasADi) """
    ttheta = cas.tan(theta)
    cphi, sphi = cas.cos(phi), cas.sin(phi)
    cth = cas.cos(theta)
    
    return cas.vertcat(
        cas.horzcat(1, sphi*ttheta, cphi*ttheta),
        cas.horzcat(0, cphi,       -sphi),
        cas.horzcat(0, sphi/cth,    cphi/cth)
    )

def get_J1_np(phi, theta, psi):
    """ Numpy implementation of Rotation Matrix (Body -> World). """
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    cpsi, spsi = np.cos(psi), np.sin(psi)

    return np.array([
        [cpsi*cth, -spsi*cphi + sphi*sth*cpsi, spsi*sphi + sth*cpsi*cphi],
        [spsi*cth,  cpsi*cphi + sphi*sth*spsi, -cpsi*sphi + sth*spsi*cphi],
        [-sth,      sphi*cth,                  cphi*cth]
    ])

def get_J2_np(phi, theta, psi):
    """ Numpy implementation of Angular Velocity Transformation. """
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth = np.cos(theta)
    ttheta = np.tan(theta)

    if abs(cth) < 0.01: cth = np.sign(cth) * 0.01

    return np.array([
        [1, sphi * ttheta, cphi * ttheta],
        [0, cphi,         -sphi],
        [0, sphi / cth,    cphi / cth]
    ])

def force_w2b(rov_state, force_world):
    """ Rotate Force from World frame into Body frame """
    phi, theta, psi = rov_state[3], rov_state[4], rov_state[5]
    R_B2W = get_J1_np(phi, theta, psi)
    return R_B2W.T @ force_world[0:3]