import json
import math
import os
import numpy as np
import matplotlib.pyplot as plt

def load_params(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_compensator_analysis():
    # Setup paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'satellite_params.json')
    result_dir = os.path.join(base_dir, 'result')
    
    # Load parameters
    params = load_params(data_path)
    notch_filters = params['controller']['notch_filters']
    
    # Frequency vector
    w = np.logspace(-1, 2, 5000)
    s = 1j * w
    
    # 1. Notch Filters Frequency Response N(s)
    N_s = np.ones_like(s, dtype=complex)
    for nf in notch_filters:
        wn = 2.0 * math.pi * nf['freq_hz']
        zeta_z = nf['zeta_zero']
        zeta_p = nf['zeta_pole']
        num = s**2 + 2*zeta_z*wn*s + wn**2
        den = s**2 + 2*zeta_p*wn*s + wn**2
        N_s *= (num / den)
        
    mag = np.abs(N_s)
    mag_dB = 20 * np.log10(mag)
    phase = np.angle(N_s)
    phase_deg = np.degrees(np.unwrap(phase))
    
    # Plotting Bode
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.semilogx(w, mag_dB, color='#9467bd', linewidth=2)
    ax1.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax1.set_ylabel('Magnitude (dB)', fontsize=12)
    ax1.set_title('Bode Plot of Flexible Mode Compensator (Cascaded Notch Filters)', fontsize=14, fontweight='bold')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    # Annotate the notches
    for nf in notch_filters:
        wn = 2.0 * math.pi * nf['freq_hz']
        mag_at_wn = 20 * math.log10(nf['zeta_zero'] / nf['zeta_pole'])
        ax1.plot(wn, mag_at_wn, 'mo', markersize=6)
        ax1.annotate(f'Notch at {nf["freq_hz"]} Hz', (wn, mag_at_wn - 5), color='m', horizontalalignment='center')
        
    ax2.semilogx(w, phase_deg, color='#e377c2', linewidth=2)
    ax2.axhline(0, color='r', linestyle='--', alpha=0.5)
    ax2.set_ylabel('Phase (deg)', fontsize=12)
    ax2.set_xlabel('Frequency (rad/s)', fontsize=12)
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    plot_path = os.path.join(result_dir, 'compensator_bode_plot.png')
    plt.savefig(plot_path, dpi=150)
    plt.close()
    
    return plot_path

if __name__ == "__main__":
    path = run_compensator_analysis()
    print(f"Compensator Bode plot saved to {path}")
