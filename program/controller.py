import math
import collections

class PIDController:
    def __init__(self, inertia, bandwidth_hz, damping):
        self.inertia = inertia
        
        # Calculate natural frequency (omega_n) from bandwidth
        self.omega_n = 2.0 * math.pi * bandwidth_hz
        self.damping = damping
        
        # Calculate PID gains
        # For a 1/Is^2 plant, Kp = I * w_n^2, Kd = I * 2 * zeta * w_n
        self.Kp = self.inertia * (self.omega_n ** 2)
        self.Kd = self.inertia * 2.0 * self.damping * self.omega_n
        
        # Add a small integral gain for steady-state error elimination
        self.Ki = 0.0 # Pure PD for spacecraft slew maneuver
        
        self.integral_error = 0.0

    def compute(self, target_angle, current_angle, current_rate, dt):
        error = target_angle - current_angle
        self.integral_error += error * dt
        
        # Anti-windup
        if self.integral_error > 5.0: self.integral_error = 5.0
        if self.integral_error < -5.0: self.integral_error = -5.0
        
        # PD/PID control law
        commanded_torque = (self.Kp * error) + (self.Ki * self.integral_error) - (self.Kd * current_rate)
        return commanded_torque

class PWPFM:
    def __init__(self, Km, Tm, Uon, Uoff, Um):
        self.Km = Km
        self.Tm = Tm
        self.Uon = Uon
        self.Uoff = Uoff
        self.Um = Um
        
        self.filter_state = 0.0
        self.output_state = 0.0

    def update(self, command_input, dt):
        # Limit command input to not exceed max possible required or let it be saturated?
        # Usually command is limited
        if command_input > 1.5 * self.Um:
            command_input = 1.5 * self.Um
        if command_input < -1.5 * self.Um:
            command_input = -1.5 * self.Um

        # Update filter state: dx/dt = (Km * (input - output) - x) / Tm
        derivative = (self.Km * (command_input - self.output_state) - self.filter_state) / self.Tm
        self.filter_state += derivative * dt
        
        # Schmitt Trigger logic
        if self.output_state == 0.0:
            if self.filter_state > self.Uon:
                self.output_state = self.Um
            elif self.filter_state < -self.Uon:
                self.output_state = -self.Um
        elif self.output_state == self.Um:
            if self.filter_state < self.Uoff:
                self.output_state = 0.0
        elif self.output_state == -self.Um:
            if self.filter_state > -self.Uoff:
                self.output_state = 0.0
                
        return self.output_state

class DelayBuffer:
    def __init__(self, delay_ms, dt_ms):
        self.delay_steps = int(delay_ms / dt_ms)
        self.buffer = collections.deque([0.0] * self.delay_steps, maxlen=self.delay_steps if self.delay_steps > 0 else 1)
        
    def step(self, value):
        if self.delay_steps <= 0:
            return value
        delayed_value = self.buffer[0]
        self.buffer.append(value)
        return delayed_value
