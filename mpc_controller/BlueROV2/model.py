from acados_template import AcadosModel
import casadi as cas
import numpy as np

def export_vehicle_model(params):
    model_name = 'vehicle_model'

    # States: [x, y, z, phi, theta, psi, u, v, w, p, q, r] (12)
    x_state = cas.SX.sym('x_state', params.nx)
    eta = x_state[0:6]
    nu  = x_state[6:12]
    
    # Parameter disturbances: External forces and moments (6) [fx, fy, fz, mx, my, mz]
    p_dist = cas.SX.sym('p_dist', 6)

    # Controls: Thruster inputs (8)
    u_ctrl = cas.SX.sym('u_ctrl', params.nu)

    # dynamics    
    # Load Constants
    M_inv = cas.SX(params.M_INV)
    D_lin = cas.SX(params.D_LIN)
    TAM   = cas.SX(params.TAM)
    W = params.W
    B = params.B
    zg = params.zg

    # quadr damping
    dq_diag = cas.vertcat(
        params.D_QUAD_COEFFS[0] * cas.fabs(nu[0]),
        params.D_QUAD_COEFFS[1] * cas.fabs(nu[1]),
        params.D_QUAD_COEFFS[2] * cas.fabs(nu[2]),
        params.D_QUAD_COEFFS[3] * cas.fabs(nu[3]),
        params.D_QUAD_COEFFS[4] * cas.fabs(nu[4]),
        params.D_QUAD_COEFFS[5] * cas.fabs(nu[5])
    )
    D_quad = cas.diag(dq_diag)

    phi, theta, psi = eta[3], eta[4], eta[5]
    
    diff = B-W
    g_vec = cas.vertcat(
        -diff * cas.sin(theta),
        diff * cas.cos(theta) * cas.sin(phi),
        diff * cas.cos(theta) * cas.cos(phi),
        zg * W * cas.cos(theta) * cas.sin(phi),
        zg * W * cas.sin(theta),
        0
    )

    # 
    tau = cas.mtimes(TAM, u_ctrl)
    D_total = D_lin + D_quad
    
    #coriolis_matrix = get_C_SX(nu, params) (not in Gazebo?)

    # Forces sum : tau - cas.mtimes((D_total + coriolis_matrix), nu) - g_vec+ p_dist
    forces_sum = tau - cas.mtimes((D_total ), nu) - g_vec + p_dist
    nu_dot = cas.mtimes(M_inv, forces_sum)

    J1 = get_J1(phi, theta, psi) # Ensure utils returns SX compatible logic
    J2 = get_J2(phi, theta, psi)
    
    pos_dot = cas.mtimes(J1, nu[0:3])
    att_dot = cas.mtimes(J2, nu[3:6])

    # explicit dynamics expression
    f_expl = cas.vertcat(pos_dot, att_dot, nu_dot)

     #  Pack into AcadosModel
    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.x = x_state
    model.p = p_dist
    model.u = u_ctrl
    model.name = model_name

    return model




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
    """ Angular Velocity Transformation Matrix """
    ttheta = cas.tan(theta)
    cphi, sphi = cas.cos(phi), cas.sin(phi)
    cth = cas.cos(theta)
    
    return cas.vertcat(
        cas.horzcat(1, sphi*ttheta, cphi*ttheta),
        cas.horzcat(0, cphi,       -sphi),
        cas.horzcat(0, sphi/cth,    cphi/cth)
    )

def get_C_SX(nu, params):
    """
    Computes Coriolis matrix using CasADi SX and params.
    """
    C_rb = cas.SX.zeros(6, 6)
    
    m = params.m
    Ix, Iy, Iz = params.Ix, params.Iy, params.Iz
    
    u, v, w = nu[0], nu[1], nu[2]
    p, q, r = nu[3], nu[4], nu[5]

    # Rigid Body Coriolis
    # Rows 0-2 (Linear Momentum)
    C_rb[0, 4] =  m * w;   C_rb[0, 5] = -m * v
    C_rb[1, 3] = -m * w;   C_rb[1, 5] =  m * u
    C_rb[2, 3] =  m * v;   C_rb[2, 4] = -m * u
    # Rows 3-5 (Angular Momentum)
    C_rb[3, 1] = -m * w;   C_rb[3, 2] =  m * v;   C_rb[3, 4] =  Iz * r;   C_rb[3, 5] = -Iy * q
    C_rb[4, 0] =  m * w;   C_rb[4, 2] = -m * u;   C_rb[4, 3] = -Iz * r;   C_rb[4, 5] =  Ix * p
    C_rb[5, 0] = -m * v;   C_rb[5, 1] =  m * u;   C_rb[5, 3] =  Iy * q;   C_rb[5, 4] = -Ix * p
    
    # Added Mass Coriolis (Approximated Diagonal)
    C_a = cas.SX.zeros(6, 6)
    
    Xud, Yvd, Zwd = params.X_ud, params.Y_vd, params.Z_wd
    Kpd, Mqd, Nrd = params.K_pd, params.M_qd, params.N_rd

    # Precompute terms
    a1, a2, a3 = Xud*u, Yvd*v, Zwd*w
    b1, b2, b3 = Kpd*p, Mqd*q, Nrd*r

    C_a[0, 5] = -a2;   C_a[0, 4] =  a3
    C_a[1, 5] =  a1;   C_a[1, 3] = -a3
    C_a[2, 4] = -a1;   C_a[2, 3] =  a2
    
    C_a[3, 5] = -b2;   C_a[3, 4] =  b3;  C_a[3, 2] = -a2;  C_a[3, 1] =  a3
    C_a[4, 5] = -b1;   C_a[4, 3] = -b3;  C_a[4, 2] =  a1;  C_a[4, 0] = -a3
    C_a[5, 4] =  b1;   C_a[5, 3] =  b2;  C_a[5, 1] = -a1;  C_a[5, 0] =  a2

    return  C_rb + C_a