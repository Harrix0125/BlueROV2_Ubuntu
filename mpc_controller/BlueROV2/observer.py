import numpy as np
from nmpc_params import NMPC_params as MPCC
import utils

class DisturbanceObserver:
    def __init__(self):
        self.dt = MPCC.T_s
        
        self.nu_hat = np.zeros(6)  # Estimated velocities
        self.d_hat = np.zeros(6)  # Estimated disturbances

        