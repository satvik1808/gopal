import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt
from controller import PIDController, PWPFM, DelayBuffer
from simulation import SatelliteDynamics

def load_params(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_simulation():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'satellite_params.json')
    result_dir = os.path.join(base_dir, 'result')
    
    # Load parameters
    params = load_params(data_path)
    
    dt = params['simulation']['dt']
    t_end = params['simulation']['t_end']
    target_angle_deg = params['simulation']['step_target_deg']
    target_angle_rad = math.radians(target_angle_deg)
    
    dt_ms = dt * 1000.0
    
    # Initialize components
    satellite = SatelliteDynamics(inertia=params['satellite']['inertia'])
    
    pid = PIDController(
        inertia=params['satellite']['inertia'],
        bandwidth_hz=params['controller']['bandwidth_hz'],
        damping=params['controller']['damping']
    )
    
    pwpfm = PWPFM(
        Km=params['pwpfm']['Km'],
        Tm=params['pwpfm']['Tm'],
        Uon=params['pwpfm']['Uon'] * params['satellite']['max_torque'],
        Uoff=params['pwpfm']['Uoff'] * params['satellite']['max_torque'],
        Um=params['satellite']['max_torque']
    )
    
    # Delays
    comp_delay = DelayBuffer(params['controller']['comp_delay_ms'], dt_ms)
    rcs_delay = DelayBuffer(params['controller']['rcs_delay_ms'], dt_ms)
    
    # Storage for plotting
    time_log = []
    angle_log = []
    rate_log = []
    cmd_torque_log = []
    rcs_torque_log = []
    filter_state_log = []
    
    # Initial states
    current_angle = 0.0
    current_rate = 0.0
    actual_rcs_torque = 0.0
    
    # Simulation loop
    time_steps = int(t_end / dt)
    for step in range(time_steps):
        current_time = step * dt
        
        # 1. Sensor measurement (assuming perfect sensors, but can add noise)
        meas_angle = current_angle
        meas_rate = current_rate
        
        # 2. PID Control logic (with computational delay)
        cmd_torque_raw = pid.compute(target_angle_rad, meas_angle, meas_rate, dt)
        cmd_torque_delayed = comp_delay.step(cmd_torque_raw)
        
        # 3. PWPFM Modulation
        pwpfm_cmd = pwpfm.update(cmd_torque_delayed, dt)
        
        # 4. RCS System delay
        actual_rcs_torque = rcs_delay.step(pwpfm_cmd)
        
        # 5. Dynamics update
        current_angle, current_rate = satellite.step(actual_rcs_torque, dt)
        
        # Logging
        time_log.append(current_time)
        angle_log.append(math.degrees(current_angle))
        rate_log.append(math.degrees(current_rate))
        cmd_torque_log.append(cmd_torque_raw)
        rcs_torque_log.append(actual_rcs_torque)
        filter_state_log.append(pwpfm.filter_state)

    # Plotting results
    fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    
    axs[0].plot(time_log, angle_log, label='Attitude')
    axs[0].axhline(y=target_angle_deg, color='r', linestyle='--', label='Target')
    axs[0].set_ylabel('Attitude (deg)')
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].plot(time_log, rate_log, label='Rate', color='orange')
    axs[1].set_ylabel('Rate (deg/s)')
    axs[1].legend()
    axs[1].grid(True)
    
    axs[2].plot(time_log, cmd_torque_log, label='Continuous Cmd Torque (raw)')
    axs[2].plot(time_log, filter_state_log, label='PWPFM Filter State', alpha=0.6)
    axs[2].axhline(y=pwpfm.Uon, color='g', linestyle='--', alpha=0.5, label='U_on')
    axs[2].axhline(y=pwpfm.Uoff, color='m', linestyle='--', alpha=0.5, label='U_off')
    axs[2].set_ylabel('Torque / State')
    axs[2].legend()
    axs[2].grid(True)
    
    axs[3].plot(time_log, rcs_torque_log, label='Actual RCS Torque', color='red')
    axs[3].set_ylabel('Torque (Nm)')
    axs[3].set_xlabel('Time (s)')
    axs[3].legend()
    axs[3].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(result_dir, 'simulation_results.png')
    plt.savefig(plot_path)
    print(f"Results saved to {plot_path}")
    
    # Save a JSON summary of the run
    summary = {
        "final_angle_deg": angle_log[-1],
        "final_rate_deg_s": rate_log[-1],
        "max_overshoot_deg": max(angle_log) - target_angle_deg,
        "thruster_firings": int(sum(np.abs(np.diff(rcs_torque_log)) > 0.5) / 2)
    }
    with open(os.path.join(result_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    run_simulation()
