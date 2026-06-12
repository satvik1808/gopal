import math
import collections

class PIDController:
    def __init__(self, inertia, bandwidth_hz, damping):
        self.inertia = inertia
        self.omega_n = 2.0 * math.pi * bandwidth_hz
        self.damping = damping
        self.Kp = self.inertia * (self.omega_n ** 2)
        self.Kd = self.inertia * 2.0 * self.damping * self.omega_n
        self.Ki = 0.0 # Pure PD
        self.integral_error = 0.0

    def compute(self, target_angle, current_angle, current_rate, dt):
        error = target_angle - current_angle
        self.integral_error += error * dt
        if self.integral_error > 5.0: self.integral_error = 5.0
        if self.integral_error < -5.0: self.integral_error = -5.0
        return (self.Kp * error) + (self.Ki * self.integral_error) - (self.Kd * current_rate)

class NotchFilter:
    def __init__(self, freq_hz, zeta_zero, zeta_pole):
        self.wn = 2.0 * math.pi * freq_hz
        self.zeta_z = zeta_zero
        self.zeta_p = zeta_pole
        
        # State space coefficients
        self.a1 = 2.0 * self.zeta_p * self.wn
        self.a0 = self.wn ** 2
        self.b1 = 2.0 * self.zeta_z * self.wn
        
        self.x1 = 0.0
        self.x2 = 0.0

    def step(self, u, dt):
        # State derivatives
        dx1 = -self.a1 * self.x1 + self.x2 + (self.b1 - self.a1) * u
        dx2 = -self.a0 * self.x1
        
        # Euler integration
        self.x1 += dx1 * dt
        self.x2 += dx2 * dt
        
        # Output equation
        y = self.x1 + u
        return y

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
        if command_input > 1.5 * self.Um:
            command_input = 1.5 * self.Um
        if command_input < -1.5 * self.Um:
            command_input = -1.5 * self.Um

        derivative = (self.Km * (command_input - self.output_state) - self.filter_state) / self.Tm
        self.filter_state += derivative * dt
        
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
