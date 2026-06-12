# Satellite Attitude Control with PID and PWPFM

This document outlines the plan to design and simulate a PID controller and Pulse-Width Pulse-Frequency Modulator (PWPFM) for controlling the Reaction Control System (RCS) of a satellite.

## Background and Data

Based on open literature for typical small satellites (e.g., Spacecraft Dynamics and Control by Sidi), we will assume the following baseline parameters:
*   **Moment of Inertia ($I$)**: $10 \text{ kg} \cdot \text{m}^2$ (for a single axis, e.g., pitch)
*   **Max Thruster Torque ($U_m$)**: $1 \text{ Nm}$

The user-specified parameters are:
*   **PID Bandwidth ($f_b$)**: $0.05 \text{ Hz}$ ($\omega_n \approx 0.314 \text{ rad/s}$)
*   **PID Damping ($\zeta$)**: $0.9$
*   **Computational Delay**: $16 \text{ ms}$
*   **RCS System Delay**: $64 \text{ ms}$
*   **Total Delay**: $80 \text{ ms}$

## Open Questions

> [!IMPORTANT]  
> 1. Is the assumed moment of inertia ($10 \text{ kg}\cdot\text{m}^2$) and maximum thruster torque ($1 \text{ Nm}$) acceptable, or would you like to use values for a specific satellite mission?
> 2. Do you want to simulate a single axis (e.g., pitch only) or full 3-axis attitude dynamics? I will start with a single-axis simulation to clearly demonstrate the PWPFM and PID behavior if not specified.
> 3. Should the integral term of the PID be tuned for a specific disturbance rejection, or would a standard PD controller (which handles the rigid body double-integrator plant well) with a small integral gain suffice?

## Proposed Changes

We will create the required directory structure: `program`, `reference`, `data`, and `result`.

### 1. Data and References
#### [NEW] `data/satellite_params.json`
Will store the satellite inertia, thruster torque, delays, and PID specifications.

#### [NEW] `reference/references.txt`
Will contain citations to literature regarding PWPFM design principles.

### 2. Controller Design
#### [NEW] `program/controller.py`
Will implement:
*   **PID Controller**: Calculates commanded torque based on attitude error. Gains $K_p$, $K_d$, and $K_i$ will be calculated analytically to achieve $0.05 \text{ Hz}$ bandwidth and $0.9$ damping ratio for the given inertia.
*   **PWPFM**: A class simulating the first-order filter and Schmitt trigger. It will take the continuous torque command and output discrete thruster firing signals ($\pm U_m$ or $0$).
*   **Delay Buffer**: A simple ring buffer or queue to simulate the total $80 \text{ ms}$ delay ($16 \text{ ms}$ computational + $64 \text{ ms}$ RCS).

### 3. Simulation
#### [NEW] `program/simulation.py`
Will implement the 1D rotational dynamics ($I \ddot{\theta} = \tau_{RCS}$) using a fixed-step integration method (e.g., RK4) to properly capture the discrete nature of the PWPFM switching and the exact delays.
The simulation will run a step-response scenario (e.g., a $10^\circ$ attitude maneuver).

### 4. Results
#### [NEW] `program/main.py`
The entry point. It will:
1. Load parameters from `data/satellite_params.json`.
2. Run the simulation.
3. Save plots (Attitude, Angular Velocity, Commanded Torque, and actual RCS Torque pulses) to the `result` directory.

## Verification Plan

### Automated Tests
*   Verify that the calculated PID gains mathematically yield the requested bandwidth and damping for the rigid-body plant.

### Manual Verification
*   Run `python program/main.py` and inspect the generated plots in the `result` directory.
*   Check that the rise time and overshoot in the attitude plot match the expected theoretical response for $\omega_n = 0.314 \text{ rad/s}$ and $\zeta = 0.9$.
*   Verify that the RCS thruster only outputs discrete pulses of magnitude $0$ or $\pm U_m$.
*   Verify the $80 \text{ ms}$ delay between the PID command crossing the Schmitt trigger threshold and the thruster actually firing.
