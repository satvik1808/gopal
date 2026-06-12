import math
import numpy as np

class FlexibleSatelliteDynamics:
    def __init__(self, inertia, flex_modes_params):
        """
        flex_modes_params: list of dicts with 'freq_hz', 'zeta', 'coupling'
        """
        self.I_rigid = inertia
        
        self.num_modes = len(flex_modes_params)
        self.wn = np.array([2.0 * math.pi * m['freq_hz'] for m in flex_modes_params])
        self.zeta = np.array([m['zeta'] for m in flex_modes_params])
        self.F = np.array([m['coupling'] for m in flex_modes_params])
        
        # Effective inertia
        sum_F_sq = np.sum(self.F**2)
        if sum_F_sq >= self.I_rigid:
            raise ValueError("Coupling terms too large! Effective inertia is <= 0")
        self.I_eff = self.I_rigid - sum_F_sq
        
        # States
        self.theta = 0.0
        self.theta_dot = 0.0
        
        self.eta = np.zeros(self.num_modes)
        self.eta_dot = np.zeros(self.num_modes)

    def step(self, torque, dt):
        # 1. Compute theta_ddot
        # (I - sum(F^2)) * theta_ddot = T + sum(F * (2*zeta*wn*eta_dot + wn^2*eta))
        flex_restoring_forces = 2.0 * self.zeta * self.wn * self.eta_dot + (self.wn**2) * self.eta
        theta_ddot = (torque + np.sum(self.F * flex_restoring_forces)) / self.I_eff
        
        # 2. Compute eta_ddot for each mode
        # eta_ddot = -2*zeta*wn*eta_dot - wn^2*eta - F*theta_ddot
        eta_ddot = -flex_restoring_forces - self.F * theta_ddot
        
        # 3. Euler integration
        self.theta_dot += theta_ddot * dt
        self.theta += self.theta_dot * dt
        
        self.eta_dot += eta_ddot * dt
        self.eta += self.eta_dot * dt
        
        # The true measured attitude might just be theta (assuming sensor is on the rigid hub)
        # If the sensor is on the flexible appendage, it would be theta + mode_shape * eta
        # We assume sensor is on the hub, but the modes affect the rigid hub dynamics directly.
        return self.theta, self.theta_dot
