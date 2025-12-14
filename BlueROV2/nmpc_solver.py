import casadi as cas
import numpy as np
from nmpc_params import NMPC_params as MPCC
import utils as utils

class NMPC_solver:
    def __init__(self):
        self.opti = cas.Opti()

        self.X = self.opti.variable(12, MPCC.N + 1)

        self.U = self.opti.variable(8, MPCC.N)

        self.opt_x0 = self.opti.parameter(12)
        self.opt_xref = self.opti.parameter(12)
        self.opt_u0 = self.opti.parameter(8)
        self.opt_uref = self.opti.parameter(8)

        J = 0
        # they are python lists not numpy array, so we get a 12dim list
        # [1,1,1]+[2,2,9] = [1,1,1,2,2,9]

        for k in range(MPCC.N):
            e = self.X[:,k] - self.opt_xref
            J += cas.mtimes([e.T, MPCC.Q, e])

            J += MPCC.R_THRUST * cas.dot(self.U[:,k], self.U[:,k])
        e = self.X[:,MPCC.N] - self.opt_xref
        J += cas.mtimes([e.T, MPCC.Q_N, e])
        self.opti.minimize(J)

        M_inv = cas.DM(MPCC.M_INV)
        D_lin = cas.DM(MPCC.D_LIN)
        TAM = cas.DM(MPCC.TAM)

        for k in range(MPCC.N):
            state_k = self.X[:,k]
            eta = state_k[0:6]
            nu = state_k[6:12]
            
            phi = eta[3]
            theta = eta[4]
            psi = eta[5]
            
            dq_diag = cas.vertcat(
                MPCC.D_QUAD_COEFFS[0] * cas.fabs(nu[0]),
                MPCC.D_QUAD_COEFFS[1] * cas.fabs(nu[1]),
                MPCC.D_QUAD_COEFFS[2] * cas.fabs(nu[2]),
                MPCC.D_QUAD_COEFFS[3] * cas.fabs(nu[3]),
                MPCC.D_QUAD_COEFFS[4] * cas.fabs(nu[4]),
                MPCC.D_QUAD_COEFFS[5] * cas.fabs(nu[5])
            )
            D_quad = cas.diag(dq_diag)

            W = MPCC.W
            B = MPCC.B
            zg = MPCC.zg
            diff = W - B
            g_vec = cas.vertcat(
                -diff * cas.sin(theta),
                diff * cas.cos(theta) * cas.sin(phi),
                diff * cas.cos(theta) * cas.cos(phi),
                zg * W * cas.cos(theta) * cas.sin(phi),
                zg * W * cas.sin(theta),
                0
            )
            # J1: Linear velocity rotation
            J1 = utils.get_J1(phi, theta, psi)

            # J2: (body rates -> Euler rates)
            # !!! warning: singularity at theta = +/- 90 degrees
            J2 = utils.get_J2(phi, theta, psi)

            tau = cas.mtimes(TAM, self.U[:, k])
        
            D_total = D_lin + D_quad
            
            #coriolis_sum = utils.get_C_cas(nu)
            forces_sum = tau  - cas.mtimes(D_total, nu) - g_vec
            nu_dot = cas.mtimes(M_inv, forces_sum)

            pos_dot = cas.mtimes(J1, nu[0:3])
            att_dot = cas.mtimes(J2, nu[3:6])

            x_dot = cas.vertcat(pos_dot, att_dot, nu_dot)

            # Euler integration : remember to change to RK4 or IRK3 !!!!
            x_next = state_k + x_dot * MPCC.T_s
            self.opti.subject_to(self.X[:, k+1] == x_next)

        self.opti.subject_to(self.X[:, 0] == self.opt_x0)
        self.opti.subject_to(self.opti.bounded(MPCC.THRUST_MIN, self.U, MPCC.THRUST_MAX))
        
        z_pos = self.X[2, :] # select row 2 so the z-row, all columns
        self.opti.subject_to(self.opti.bounded(-0.1, z_pos, 200.0))

        theta_angle = self.X[4, :]
        self.opti.subject_to(self.opti.bounded(-0.78, theta_angle, 0.78))

        # solver setup
        opts = {
            'ipopt.print_level': 0, 
            'print_time': 0,
            'ipopt.sb': 'yes',
            'ipopt.max_iter': 100 # limit iterations for real-time safety
        }
        self.opti.solver('ipopt', opts)

        

    def solve(self, x0, target):
        self.opti.set_value(self.opt_x0, x0)
        self.opti.set_value(self.opt_xref, target)

        # if i want to warm start : here

        try:
            sol = self.opti.solve()
            return sol.value(self.U)[:, 0]
        except:
            # if solver fails, return zeros to stop motors
            return np.zeros(8)