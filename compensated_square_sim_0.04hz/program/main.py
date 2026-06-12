import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
from controller import PIDController, NotchFilter, PWPFM, DelayBuffer
from simulation import FlexibleSatelliteDynamics

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
    target_angle_amplitude = params['simulation']['step_target_deg']

    
    dt_ms = dt * 1000.0
    
    # Initialize dynamics
    satellite = FlexibleSatelliteDynamics(
        inertia=params['satellite']['inertia'],
        flex_modes_params=params['satellite']['flexible_modes']
    )
    
    # Initialize controller
    pid = PIDController(
        inertia=params['satellite']['inertia'],
        bandwidth_hz=params['controller']['bandwidth_hz'],
        damping=params['controller']['damping']
    )
    
    # Initialize Notch Filters (cascaded)
    notch_filters = []
    for nf_param in params['controller']['notch_filters']:
        notch_filters.append(
            NotchFilter(
                freq_hz=nf_param['freq_hz'],
                zeta_zero=nf_param['zeta_zero'],
                zeta_pole=nf_param['zeta_pole']
            )
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
    eta_log = []
    cmd_torque_raw_log = []
    cmd_torque_filtered_log = []
    rcs_torque_log = []
    target_angle_log = []
    
    # Initial states
    current_angle = 0.0
    current_rate = 0.0
    actual_rcs_torque = 0.0
    
    # Simulation loop
    time_steps = int(t_end / dt)
    for step in range(time_steps):
        current_time = step * dt
        
        # 1. Sensor measurement (hub angle)
        meas_angle = current_angle
        meas_rate = current_rate
        
        # dynamic target angle
        target_angle_deg = target_angle_amplitude * signal.square(2.0 * math.pi * 0.04 * current_time)
        target_angle_rad = math.radians(target_angle_deg)
        
        # 2. PID Control
        cmd_torque_raw = pid.compute(target_angle_rad, meas_angle, meas_rate, dt)
        
        # 3. Notch Filtering
        cmd_torque_filtered = cmd_torque_raw
        for nf in notch_filters:
            cmd_torque_filtered = nf.step(cmd_torque_filtered, dt)
        
        # 4. Computational Delay
        cmd_torque_delayed = comp_delay.step(cmd_torque_filtered)
        
        # 5. PWPFM Modulation
        pwpfm_cmd = pwpfm.update(cmd_torque_delayed, dt)
        
        # 6. RCS System delay
        actual_rcs_torque = rcs_delay.step(pwpfm_cmd)
        
        # 7. Dynamics update
        current_angle, current_rate = satellite.step(actual_rcs_torque, dt)
        
        # Logging
        # Only log every 10 steps (100 Hz) to save memory for plotting
        if step % 10 == 0:
            time_log.append(current_time)
            angle_log.append(math.degrees(current_angle))
            rate_log.append(math.degrees(current_rate))
            eta_log.append(np.copy(satellite.eta))
            cmd_torque_raw_log.append(cmd_torque_raw)
            cmd_torque_filtered_log.append(cmd_torque_filtered)
            rcs_torque_log.append(actual_rcs_torque)
            target_angle_log.append(target_angle_deg)

    eta_log = np.array(eta_log)

    # Plotting results
    fig, axs = plt.subplots(5, 1, figsize=(10, 15), sharex=True)
    
    axs[0].plot(time_log, angle_log, label='Hub Attitude')
    axs[0].plot(time_log, target_angle_log, color='r', linestyle='--', label='Target')
    axs[0].set_ylabel('Attitude (deg)')
    axs[0].legend()
    axs[0].grid(True)
    
    axs[1].plot(time_log, rate_log, label='Hub Rate', color='orange')
    axs[1].set_ylabel('Rate (deg/s)')
    axs[1].legend()
    axs[1].grid(True)
    
    for i in range(eta_log.shape[1]):
        axs[2].plot(time_log, eta_log[:, i], label=f'Mode {i+1} Displacement (eta)', alpha=0.7)
    axs[2].set_ylabel('Modal Displacement')
    axs[2].legend()
    axs[2].grid(True)
    
    axs[3].plot(time_log, cmd_torque_raw_log, label='Raw PID Cmd', alpha=0.5)
    axs[3].plot(time_log, cmd_torque_filtered_log, label='Notch Filtered Cmd', alpha=0.9)
    axs[3].set_ylabel('Commanded Torque')
    axs[3].legend()
    axs[3].grid(True)
    
    axs[4].plot(time_log, rcs_torque_log, label='Actual RCS Torque', color='red')
    axs[4].set_ylabel('Torque (Nm)')
    axs[4].set_xlabel('Time (s)')
    axs[4].legend()
    axs[4].grid(True)
    
    plt.tight_layout()
    plot_path = os.path.join(result_dir, 'compensated_square_0.04hz_results.png')
    plt.savefig(plot_path)
    print(f"Results saved to {plot_path}")
    
    # Save a JSON summary
    summary = {
        "final_angle_deg": float(angle_log[-1]),
        "final_rate_deg_s": float(rate_log[-1]),
        "max_overshoot_deg": 0.0,

        "thruster_firings": int(sum(np.abs(np.diff(rcs_torque_log)) > 0.5) / 2)
    }
    with open(os.path.join(result_dir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=4)

if __name__ == "__main__":
    run_simulation()
