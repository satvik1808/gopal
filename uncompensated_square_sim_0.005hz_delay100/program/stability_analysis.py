import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

def load_params(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_stability_analysis():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'satellite_params.json')
    result_dir = os.path.join(base_dir, 'result')
    
    # Load parameters
    params = load_params(data_path)
    
    I_rigid = params['satellite']['inertia']
    flex_modes = params['satellite']['flexible_modes']
    
    bw = params['controller']['bandwidth_hz']
    damping = params['controller']['damping']
    omega_n_ctrl = 2.0 * math.pi * bw
    Kp = I_rigid * (omega_n_ctrl ** 2)
    Kd = I_rigid * 2.0 * damping * omega_n_ctrl
    
    notch_filters = params['controller']['notch_filters']
    
    total_delay = (params['controller']['comp_delay_ms'] + params['controller']['rcs_delay_ms']) / 1000.0
    
    # Frequency vector
    w = np.logspace(-2, 2, 10000)
    s = 1j * w
    
    # 1. Plant Frequency Response P(s)
    # P(s) = 1 / ( s^2 * ( I - sum( F_i^2 * s^2 / (s^2 + 2*zeta*wn*s + wn^2) ) ) )
    flex_term = np.zeros_like(s, dtype=complex)
    for fm in flex_modes:
        wn = 2.0 * math.pi * fm['freq_hz']
        zeta = fm['zeta']
        F = fm['coupling']
        den = s**2 + 2*zeta*wn*s + wn**2
        flex_term += (F**2 * s**2) / den
        
    P_s = 1.0 / (s**2 * (I_rigid - flex_term))
    
    # 2. Controller Frequency Response C(s) = Kp + Kd*s
    C_s = Kp + Kd * s
    
    # 3. Notch Filters Frequency Response N(s)
    N_s = np.ones_like(s, dtype=complex)
    for nf in notch_filters:
        wn = 2.0 * math.pi * nf['freq_hz']
        zeta_z = nf['zeta_zero']
        zeta_p = nf['zeta_pole']
        num = s**2 + 2*zeta_z*wn*s + wn**2
        den = s**2 + 2*zeta_p*wn*s + wn**2
        N_s *= (num / den)
        
    # 4. Delay Frequency Response D(s) = e^(-s * Td)
    D_s = np.exp(-s * total_delay)
    
    # 5. Open Loop Transfer Function L(s)
    L_s = P_s * C_s * N_s * D_s
    
    mag = np.abs(L_s)
    mag_dB = 20 * np.log10(mag)
    phase = np.angle(L_s)
    phase_deg = np.degrees(np.unwrap(phase))
    
    # Calculate Margins
    # Phase Margin
    # Find index where magnitude crosses 1 (0 dB)
    crossings_0dB = np.where(np.diff(np.sign(mag_dB)))[0]
    PMs = []
    for idx in crossings_0dB:
        # linear interpolate for precise crossing
        w_gc = np.interp(0, [mag_dB[idx+1], mag_dB[idx]], [w[idx+1], w[idx]])
        phase_gc = np.interp(w_gc, w, phase_deg)
        PM = 180.0 + phase_gc
        # normalize PM
        PM = (PM + 180) % 360 - 180
        PMs.append((w_gc, PM))
        
    # Gain Margin
    # Find index where phase crosses -180 deg
    crossings_180deg = np.where(np.diff(np.sign(phase_deg + 180.0)))[0]
    GMs = []
    for idx in crossings_180deg:
        w_pc = np.interp(-180.0, [phase_deg[idx+1], phase_deg[idx]], [w[idx+1], w[idx]])
        mag_pc = np.interp(w_pc, w, mag_dB)
        GM = -mag_pc
        if GM > 0: # Only care about positive GM for stability
            GMs.append((w_pc, GM))
            
    # Primary PM and GM (lowest frequency crossing typically)
    pm_val = PMs[0][1] if PMs else float('nan')
    w_gc_val = PMs[0][0] if PMs else float('nan')
    gm_val = GMs[0][1] if GMs else float('nan')
    w_pc_val = GMs[0][0] if GMs else float('nan')

    # Plotting Bode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.semilogx(w, mag_dB, color='#1f77b4', linewidth=2)
    ax1.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Magnitude (dB)', fontsize=12)
    ax1.set_title('Open-Loop Bode Plot (Linear Logic)', fontsize=14, fontweight='bold')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    if not math.isnan(w_gc_val):
        ax1.plot(w_gc_val, 0, 'ro', markersize=8)
        ax1.annotate(f'w_gc={w_gc_val:.2f} rad/s', (w_gc_val, 5), color='r')
        
    ax2.semilogx(w, phase_deg, color='#ff7f0e', linewidth=2)
    ax2.axhline(-180, color='r', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Phase (deg)', fontsize=12)
    ax2.set_xlabel('Frequency (rad/s)', fontsize=12)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    
    if not math.isnan(w_gc_val):
        ax2.plot(w_gc_val, pm_val - 180, 'ro', markersize=8)
        ax2.plot([w_gc_val, w_gc_val], [-180, pm_val - 180], 'r-')
        ax2.annotate(f' PM={pm_val:.1f} deg', (w_gc_val, pm_val - 180 + 10), color='r')
        
    if not math.isnan(w_pc_val):
        ax2.plot(w_pc_val, -180, 'go', markersize=8)
        ax1.plot([w_pc_val, w_pc_val], [0, -gm_val], 'g-')
        ax1.plot(w_pc_val, -gm_val, 'go', markersize=8)
        ax1.annotate(f' GM={gm_val:.1f} dB', (w_pc_val, -gm_val - 10), color='g')
    
    plt.tight_layout()
    plot_path = os.path.join(result_dir, 'bode_plot.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    # Plotting Nyquist
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Positive frequencies
    ax.plot(np.real(L_s), np.imag(L_s), color='#1f77b4', linewidth=2, label='$\\omega > 0$')
    # Negative frequencies
    ax.plot(np.real(L_s), -np.imag(L_s), color='#1f77b4', linewidth=2, linestyle='--', alpha=0.7, label='$\\omega < 0$')
    
    # Plot -1 point
    ax.plot(-1, 0, 'r+', markersize=15, markeredgewidth=2, label='-1+j0')
    
    # Customize plot
    ax.set_title('Nyquist Plot', fontsize=14, fontweight='bold')
    ax.set_xlabel('Real Axis', fontsize=12)
    ax.set_ylabel('Imaginary Axis', fontsize=12)
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)
    ax.axvline(0, color='black', linewidth=1, alpha=0.5)
    ax.grid(True, linestyle=':', alpha=0.7)
    
    # Determine appropriate limits (avoiding poles at origin if any)
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.legend()
    
    plt.tight_layout()
    nyquist_path = os.path.join(result_dir, 'nyquist_plot.png')
    plt.savefig(nyquist_path, dpi=150)
    plt.close()
    
    return pm_val, w_gc_val, gm_val, w_pc_val, plot_path, nyquist_path

if __name__ == "__main__":
    pm, w_gc, gm, w_pc, path, nyquist_path = run_stability_analysis()
    print(f"Bode plot saved to {path}")
    print(f"Nyquist plot saved to {nyquist_path}")
    print(f"Phase Margin: {pm:.2f} deg at {w_gc:.2f} rad/s")
    if not math.isnan(gm):
        print(f"Gain Margin: {gm:.2f} dB at {w_pc:.2f} rad/s")
    else:
        print("Gain Margin: Infinite")
