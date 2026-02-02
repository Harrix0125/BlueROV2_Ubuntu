from acados_template import AcadosOcp, AcadosOcpSolver

from core.model import export_vehicle_model

# from model import export_vehicle_model

import numpy as np
import scipy.linalg

# Wrapper class for Acados NMPC Solver
class Acados_Solver_Wrapper:
    def __init__(self, vehicle_params):
        self.params = vehicle_params
        self.model = export_vehicle_model(self.params)

        self.ocp = AcadosOcp()
        self.ocp.model = self.model


        nx = self.params.nx
        nu = self.params.nu  # 8 for BlueROV2, 2 for BlueBoat
        ny = nx + nu 
        self.ocp.dims.N = self.params.N

        self.ocp.parameter_values = np.zeros(6)  # Disturbance parameter (i forgot before ops)

        # Cost Setup : Linear Least Squares
        self.ocp.cost.cost_type = 'LINEAR_LS'
        self.ocp.cost.cost_type_e = 'LINEAR_LS'

        self.ocp.cost.Vx = np.zeros((ny, nx))
        self.ocp.cost.Vx[:nx, :] = np.eye(nx)
        self.ocp.cost.Vu = np.zeros((ny, nu))
        self.ocp.cost.Vu[nx:, :] = np.eye(nu)

        self.ocp.cost.Vx_e = np.eye(nx)

        # Weight Matrix
        R_mat = np.eye(nu) * self.params.R_THRUST
        Q_mat = np.diag(self.params.Q_diag)
        self.ocp.cost.W = scipy.linalg.block_diag(Q_mat, R_mat)
        self.ocp.cost.W_e = np.diag(self.params.Q_diag_N)

        # References (Init to zero)
        self.ocp.cost.yref = np.zeros((ny, ))
        self.ocp.cost.yref_e = np.zeros((nx, ))

        # Trying to include disturbances
        self.ocp.dims.np = 6
        self.ocp.parameter_values = np.zeros(6)


        # Constraints
        self.ocp.constraints.lbu = np.array([self.params.THRUST_MIN] * nu)
        self.ocp.constraints.ubu = np.array([self.params.THRUST_MAX] * nu)
        self.ocp.constraints.idxbu = np.array(range(nu))
        

        # State Constraints (z and theta)
        self.ocp.constraints.lbx = np.array([self.params.z_min, -1.4])
        self.ocp.constraints.ubx = np.array([self.params.z_max, 1.4])
        self.ocp.constraints.idxbx = np.array([2, 4])

        self.ocp.constraints.x0 = np.zeros((nx, ))

        # Solver Options
        self.ocp.solver_options.tf = self.params.T_s * self.params.N
        self.ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
        self.ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
        self.ocp.solver_options.integrator_type = 'ERK' 
        self.ocp.solver_options.nlp_solver_type = 'SQP_RTI'

        self.solver = AcadosOcpSolver(self.ocp, json_file='acados_ocp.json',generate=True, build=True)

    def solve_legacy_version(self, x0, target_state, disturbance = None):
        if disturbance is None:
            disturbance = np.zeros(6)
        
        self.solver.set(0, "lbx", x0)
        self.solver.set(0, "ubx", x0)

        # Set Reference
        y_ref = np.concatenate((target_state, np.zeros(self.params.nu)))
        for i in range(self.params.N):
            self.solver.set(i, "yref", y_ref)
        self.solver.set(self.params.N, "yref", target_state)

        for i in range(self.params.N + 1):
            self.solver.set(i, "x", x0)

        # Passing disturbance as parameter for every step
        for i in range(self.params.N):
            self.solver.set(i, "p", disturbance)


        status = self.solver.solve()
        if status != 0 :
            print(f"Acados NMPC solver returned status {status}!")
        # Return first control
        return self.solver.get(0, "u")
    

    def solve(self, x0, target_ref, disturbance=None):
            """
            target_ref: Can be 1D array (constant reference) OR 2D array (trajectory).
                        If 2D, shape must be (N+1, 12).
            """
            if disturbance is None:
                disturbance = np.zeros(6)
            
            #Initial State Constraint
            self.solver.set(0, "lbx", x0)
            self.solver.set(0, "ubx", x0)

            for i in range(self.params.N):
                self.solver.set(i, "p", disturbance)

            # If: single point reference
            if target_ref.ndim == 1:
                y_ref = np.concatenate((target_ref, np.zeros(self.params.nu)))
                # Set for 0 to N-1
                for i in range(self.params.N):
                    self.solver.set(i, "yref", y_ref)
                # Set for N (Terminal, no controls)
                self.solver.set(self.params.N, "yref", target_ref)

            # If: Trajectory Reference (New behavior)
            else:                
                for i in range(self.params.N):
                    # Stick to last point if trajectory is too short
                    idx = min(i, target_ref.shape[0] - 1)
                    
                    state_ref = target_ref[idx, :]
                    u_ref = np.zeros(self.params.nu)
                    
                    y_ref_k = np.concatenate((state_ref, u_ref))
                    self.solver.set(i, "yref", y_ref_k)

                # Terminal Node N
                idx_N = min(self.params.N, target_ref.shape[0] - 1)
                state_ref_e = target_ref[idx_N, :]
                self.solver.set(self.params.N, "yref", state_ref_e)

            for i in range(self.params.N + 1):
                self.solver.set(i, "x", x0)

            status = self.solver.solve()
            if status != 0 :
                print(f"Acados NMPC solver returned status {status}!")
                
            return self.solver.get(0, "u")