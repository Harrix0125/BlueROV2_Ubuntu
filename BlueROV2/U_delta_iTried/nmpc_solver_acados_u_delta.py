from acados_template import AcadosOcp, AcadosOcpSolver
from model_u_delta import export_bluerov_model
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
        nx = 20
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
        Q_mat = np.diag(MPCC.Q_diag) 
        # R_mat penalizes THRUST MAGNITUDE (now part of state)
        R_mat = np.eye(8) * MPCC.R_THRUST
        # R_rate penalizes CHANGE RATE (control input)
        R_rate = np.eye(nu) * MPCC.R_RATE

        self.ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat, R_rate)
        self.ocp.cost.W_e = scipy.linalg.block_diag(np.diag(MPCC.Q_diag_N), R_mat)
        # References (Init to zero)
        self.ocp.cost.yref = np.zeros((ny, ))
        self.ocp.cost.yref_e = np.zeros((nx, ))

        rate_limit = MPCC.DELTA_THRUST_LIMIT
        self.ocp.constraints.lbu = np.array([-rate_limit] * nu)
        self.ocp.constraints.ubu = np.array([+rate_limit] * nu)
        self.ocp.constraints.idxbu = np.array(range(nu))
        
        # State Constraints (z and theta)
        # z, theta and u_ctrl
        self.ocp.constraints.idxbx = np.array([2, 4] + list(range(12, 20)))
        
        # Lower bounds
        lb_phys = [-100, -1.4]
        lb_thrust = [MPCC.THRUST_MIN] * 8
        self.ocp.constraints.lbx = np.array(lb_phys + lb_thrust)
        
        # Upper bounds
        ub_phys = [0.5, 1.4]
        ub_thrust = [MPCC.THRUST_MAX] * 8
        self.ocp.constraints.ubx = np.array(ub_phys + ub_thrust)

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

    def solve(self, x0_physics, u_prev, target_state):
        # Combine physics state and previous thrust for full x0
        x0_augmented = np.concatenate((x0_physics, u_prev))
        
        # Set Initial Condition
        self.solver.set(0, "lbx", x0_augmented)
        self.solver.set(0, "ubx", x0_augmented)

        # Set Reference
        # y_ref structure: [12 physics, 8 thrust, 8 rate]
        # Target: physics=target, thrust=0 (minimize energy), rate=0 (steady)
        y_ref = np.concatenate((target_state, np.zeros(8), np.zeros(8)))
        
        for i in range(MPCC.N):
            self.solver.set(i, "yref", y_ref)
            
        # Terminal Ref (no rate)
        y_ref_e = np.concatenate((target_state, np.zeros(8)))
        self.solver.set(MPCC.N, "yref", y_ref_e)

        # Solve
        status = self.solver.solve()
        if status != 0 :
            print(f"Acados NMPC solver returned status {status}!")
            
        # Get the first CONTROL input (which is now the rate)
        u_rate = self.solver.get(0, "u")
        
        # To get the actual thrust to send to the robot, we must predict the next state
        # or simply apply: u_send = u_prev + u_rate * Ts
        # However, Acados has already computed the next state x1. 
        # The most robust way is to read the Thrust State at index 1
        x1 = self.solver.get(1, "x")
        u_next_thrust = x1[12:20]
        
        return u_next_thrust