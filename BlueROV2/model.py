from acados_template import AcadosModel
import casadi as cas
import numpy as np
from nmpc_params import NMPC_params as MPCC
import utils as utils

def export_bluerov_model():
    model_name = 'bluerov2'
    # States: [x, y, z, phi, theta, psi, u, v, w, p, q, r] (12)
    x_state = cas.SX.sym('x_state', 12)
    eta = x_state[0:6]
    nu  = x_state[6:12]
    
    # Controls: Thruster inputs (8)
    u_ctrl = cas.SX.sym('u_ctrl', 8)

    # dynamics    
    # Load Constants
    M_inv = cas.SX(MPCC.M_INV)
    D_lin = cas.SX(MPCC.D_LIN)
    TAM   = cas.SX(MPCC.TAM)
    W = MPCC.W
    B = MPCC.B
    zg = MPCC.zg

    # quadr damping
    dq_diag = cas.vertcat(
        MPCC.D_QUAD_COEFFS[0] * cas.fabs(nu[0]),
        MPCC.D_QUAD_COEFFS[1] * cas.fabs(nu[1]),
        MPCC.D_QUAD_COEFFS[2] * cas.fabs(nu[2]),
        MPCC.D_QUAD_COEFFS[3] * cas.fabs(nu[3]),
        MPCC.D_QUAD_COEFFS[4] * cas.fabs(nu[4]),
        MPCC.D_QUAD_COEFFS[5] * cas.fabs(nu[5])
    )
    D_quad = cas.diag(dq_diag)

    phi, theta, psi = eta[3], eta[4], eta[5]
    
    diff = W - B
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
    
    # force sum (Ignoring Coriolis?)
    coriolis_matrix = utils.get_C_SX(nu)
    forces_sum = tau - cas.mtimes((D_total), nu) - g_vec

    forces_sum = tau - cas.mtimes((D_total+coriolis_matrix), nu) - g_vec
    nu_dot = cas.mtimes(M_inv, forces_sum)

    J1 = utils.get_J1(phi, theta, psi) # Ensure utils returns SX compatible logic
    J2 = utils.get_J2(phi, theta, psi)
    
    pos_dot = cas.mtimes(J1, nu[0:3])
    att_dot = cas.mtimes(J2, nu[3:6])

    # explicit dynamics expression
    f_expl = cas.vertcat(pos_dot, att_dot, nu_dot)

     #  Pack into AcadosModel
    model = AcadosModel()
    model.f_expl_expr = f_expl
    model.x = x_state
    model.u = u_ctrl
    model.name = model_name

    return model