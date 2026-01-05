from acados_template import AcadosOcp, AcadosOcpSolver
from model import export_bluerov_model
import numpy as np
import scipy.linalg
from nmpc_params import NMPC_params as MPCC
import os

# Wrapper class for Acados NMPC Solver
class Acados_Solver_Wrapper:
    def __init__(self):
        # Load Model
        self.model = export_bluerov_model()
        self.ocp = AcadosOcp()
        self.ocp.model = self.model

        # Dimensions
        nx = 12
        nu = 8
        ny = nx + nu 
        self.ocp.dims.N = MPCC.N

        # Cost Setup (Linear Least Squares)
        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'

        self.ocp.cost.Vx = np.zeros((ny, nx))
        self.ocp.cost.Vx[:nx, :] = np.eye(nx)
        self.ocp.cost.Vu = np.zeros((ny, nu))
        self.ocp.cost.Vu[nx:, :] = np.eye(nu)

        self.ocp.cost.Vx_e = np.eye(nx)

        # Weight Matrix
        R_mat = np.eye(nu) * MPCC.R_THRUST
        Q_mat = np.diag(MPCC.Q_diag)
        self.ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        self.ocp.cost.W_e = np.diag(MPCC.Q_diag_N)

        # References (Init to zero)
        self.ocp.cost.yref = np.zeros((ny, ))
        self.ocp.cost.yref_e = np.zeros((nx, ))

        # Constraints
        self.ocp.constraints.lbu = np.array([MPCC.THRUST_MIN] * nu)
        self.ocp.constraints.ubu = np.array([MPCC.THRUST_MAX] * nu)
        self.ocp.constraints.idxbu = np.array(range(nu))
        
        # State Constraints (z and theta)
        self.ocp.constraints.lbx = np.array([-100, -1.4])
        self.ocp.constraints.ubx = np.array([0.5, 1.4])
        self.ocp.constraints.idxbx = np.array([2, 4])

        self.ocp.constraints.x0 = np.zeros((nx, ))

        # Solver Options
        self.ocp.solver_options.tf = MPCC.T_s * MPCC.N
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'ERK' 

        # Set IRK stages (4 is standard for accuracy/stability)
        #self.ocp.solver_options.sim_method_num_stages = 4 
        # Integration steps per shooting interval (Increase to 2 or 3 if still unstable)
        #self.ocp.solver_options.sim_method_num_steps = 3

        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        # 6. Generate
        self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json')

    def solve(self, x0, target_state):
        # Set Initial Condition
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        # Set Reference
        y_ref = np.concatenate((target_state, np.zeros(8)))
        for i in range(MPCC.N):
            self.solver.set(i, "yref", y_ref)
        # Using as terminal cost yref, be careful
        self.solver.set(MPCC.N, "yref", target_state)

        for i in range(MPCC.N + 1):
            self.solver.set(i, "x", x0)
        # Solve
        status = self.solver.solve()
        if status != 0 :
            print(f"Acados NMPC solver returned status {status}!")
        # Return first control
        return self.solver.get(0, "u")