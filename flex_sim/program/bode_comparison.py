import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt

def load_params(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_bode_comparison():
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
    w = np.logspace(-1, 2, 10000)
    s = 1j * w
    
    # 1. Plant Frequency Response P(s)
    flex_term = np.zeros_like(s, dtype=complex)
    for fm in flex_modes:
        wn = 2.0 * math.pi * fm['freq_hz']
        zeta = fm['zeta']
        F = fm['coupling']
        den = s**2 + 2*zeta*wn*s + wn**2
        flex_term += (F**2 * s**2) / den
        
    P_s = 1.0 / (s**2 * (I_rigid - flex_term))
    
    # 2. Controller Frequency Response C(s)
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
        
    # 4. Delay Frequency Response D(s)
    D_s = np.exp(-s * total_delay)
    
    # Open Loop Transfer Functions
    L_uncomp = P_s * C_s * D_s
    L_comp = P_s * C_s * N_s * D_s
    
    mag_uncomp = 20 * np.log10(np.abs(L_uncomp))
    phase_uncomp = np.degrees(np.unwrap(np.angle(L_uncomp)))
    
    mag_comp = 20 * np.log10(np.abs(L_comp))
    phase_comp = np.degrees(np.unwrap(np.angle(L_comp)))
    
    # Plotting Bode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Magnitude
    ax1.semilogx(w, mag_uncomp, color='#f59e0b', linewidth=2, label='Uncompensated')
    ax1.semilogx(w, mag_comp, color='#4ade80', linewidth=2, label='Compensated (Notch Filters)')
    ax1.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Magnitude (dB)', fontsize=12)
    ax1.set_title('Open-Loop Bode Plot Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    ax1.legend()
    
    # Phase
    ax2.semilogx(w, phase_uncomp, color='#f59e0b', linewidth=2, label='Uncompensated')
    ax2.semilogx(w, phase_comp, color='#4ade80', linewidth=2, label='Compensated (Notch Filters)')
    ax2.axhline(-180, color='r', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Phase (deg)', fontsize=12)
    ax2.set_xlabel('Frequency (rad/s)', fontsize=12)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    ax2.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(result_dir, 'bode_comparison.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Bode comparison plot saved to {plot_path}")
    plt.close()
    
    # Plotting Nyquist Comparison
    fig_nyq, ax_nyq = plt.subplots(figsize=(8, 8))
    
    # Uncompensated (positive and negative frequencies)
    ax_nyq.plot(np.real(L_uncomp), np.imag(L_uncomp), color='#f59e0b', linewidth=2, label='Uncompensated')
    ax_nyq.plot(np.real(L_uncomp), -np.imag(L_uncomp), color='#f59e0b', linewidth=2, linestyle=':')
    
    # Compensated (positive and negative frequencies)
    ax_nyq.plot(np.real(L_comp), np.imag(L_comp), color='#4ade80', linewidth=2, label='Compensated (Notch Filters)')
    ax_nyq.plot(np.real(L_comp), -np.imag(L_comp), color='#4ade80', linewidth=2, linestyle=':')
    
    # Critical point
    ax_nyq.plot(-1, 0, 'ro', markersize=8, label='Critical Point (-1, 0)')
    
    ax_nyq.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax_nyq.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    ax_nyq.set_xlim([-2.5, 1.5])
    ax_nyq.set_ylim([-2.0, 2.0])
    
    ax_nyq.set_xlabel('Real Axis', fontsize=12)
    ax_nyq.set_ylabel('Imaginary Axis', fontsize=12)
    ax_nyq.set_title('Nyquist Plot Comparison', fontsize=14, fontweight='bold')
    ax_nyq.grid(True, linestyle='-', alpha=0.2)
    ax_nyq.legend(loc='upper right')
    
    plt.tight_layout()
    nyq_path = os.path.join(result_dir, 'nyquist_comparison.png')
    plt.savefig(nyq_path, dpi=150)
    print(f"Nyquist comparison plot saved to {nyq_path}")
    plt.close()

if __name__ == "__main__":
    run_bode_comparison()
