import numpy as np

class SatelliteDynamics:
    def __init__(self, inertia):
        self.inertia = inertia
        self.angle = 0.0
        self.rate = 0.0

    def step(self, torque, dt):
        # Using simple Euler integration due to the small time step and discrete nature of thrusters
        # RK4 could be used but Euler is generally sufficient for dt=0.001 with this simple plant.
        
        # Angular acceleration
        accel = torque / self.inertia
        
        # Integrate to get new rate and angle
        self.rate += accel * dt
        self.angle += self.rate * dt
        
        return self.angle, self.rate
