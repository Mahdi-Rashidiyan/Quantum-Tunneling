#!/usr/bin/env python
"""Quick test of fixed RTD simulator against experimental data"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import monte

print("="*80)
print("FIXED RTD SIMULATOR - QUICK TEST")
print("="*80)

# Create simulator with realistic Brown 1991 device
print("\n1. Creating simulator with realistic Brown et al. (1991) parameters...")
sim = monte.Advanced_Quantum_Tunneling_Simulator(
    n_points=800, 
    n_particles=5000, 
    temperature=300,
    device='brown_1991'
)

print(f"   Device: {sim.device_name}")
print(f"   Barrier height: {sim.V0/1.602e-19:.2f} eV")
print(f"   Barrier 1: {sim.barrier_config['b1_width']*1e9:.2f} nm at {sim.barrier_config['b1_start']*1e9:.1f} nm")
print(f"   Barrier 2: {sim.barrier_config['b2_width']*1e9:.2f} nm at {sim.barrier_config['b2_start']*1e9:.1f} nm")
print(f"   Energy range: {sim.E_min/1.602e-19:.2f} - {sim.E_max/1.602e-19:.2f} eV")

# Run QUICK simulation
print("\n2. Running quick Monte Carlo simulation (no heavy effects)...")
print("   Physics: Clean resonant tunneling (no phonons, no many-body)")

for i, E in enumerate(sim.E_incident):
    T, R = sim.monte_carlo_advanced(E, include_phonons=False, include_many_body=False)
    sim.results['transmission'].append(T)
    sim.results['reflection'].append(R)
    
    if i % 20 == 0:
        print(f"   Progress: {i}/{len(sim.E_incident)}")

print("   DONE!")

# Check results
print("\n3. Checking simulation results...")
transmission = np.array(sim.results['transmission'])
print(f"   Transmission range: {np.min(transmission):.6f} - {np.max(transmission):.6f}")
print(f"   Mean transmission: {np.mean(transmission):.6f}")
print(f"   Std deviation: {np.std(transmission):.6f}")

# Check if we have peaks
from scipy.signal import find_peaks
peaks, props = find_peaks(transmission, height=0.1, distance=3)
print(f"   Peaks detected: {len(peaks)}")
if len(peaks) > 0:
    print(f"   Peak heights: {transmission[peaks]}")
    print(f"   Peak energies: {sim.E_incident[peaks]/1.602e-19} eV")

# Validate against Brown 1991
print("\n4. Validating against Brown et al. (1991) experimental data...")
dataset = sim.experimental_data.get_dataset('brown_1991')
exp_data = dataset['temperature_300K']

exp_voltage = exp_data['voltage']
exp_current = exp_data['current']
sim_voltage = sim.E_incident / 1.602e-19

print(f"   Experimental voltage range: {np.min(exp_voltage):.2f} - {np.max(exp_voltage):.2f} V")
print(f"   Experimental current range: {np.min(exp_current):.2f} - {np.max(exp_current):.2f} kA/cm²")
print(f"   Experimental peak: {np.max(exp_current):.2f} kA/cm² at {exp_voltage[np.argmax(exp_current)]:.2f} V")

# Scale simulation properly
if np.max(transmission) > 0:
    scale_factor = np.max(exp_current) / (np.max(transmission) + 1e-10)
else:
    scale_factor = np.max(exp_current)

print(f"   Scale factor: {scale_factor:.3e}")

sim_current = transmission * scale_factor

# Interpolate
sim_interp = interp1d(sim_voltage, sim_current, kind='cubic', 
                     bounds_error=False, fill_value='extrapolate')
sim_current_matched = sim_interp(exp_voltage)
sim_current_matched = np.maximum(sim_current_matched, 0)

print(f"   Scaled sim current range: {np.min(sim_current_matched):.2f} - {np.max(sim_current_matched):.2f} kA/cm²")

# Calculate metrics
mae = np.mean(np.abs(sim_current_matched - exp_current))
rmse = np.sqrt(np.mean((sim_current_matched - exp_current)**2))
if np.std(sim_current_matched) > 0 and np.std(exp_current) > 0:
    corr = np.corrcoef(sim_current_matched, exp_current)[0, 1]
else:
    corr = 0.0

print(f"\n5. Validation Metrics:")
print(f"   MAE: {mae:.2f} kA/cm²")
print(f"   RMSE: {rmse:.2f} kA/cm²")
print(f"   Correlation: {corr:.4f}")

# Plot
print("\n6. Creating comparison plot...")
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(exp_voltage, exp_current, 'ro-', linewidth=2.5, markersize=8, 
        label='Brown et al. (1991) - Real Data', alpha=0.7)
ax.plot(exp_voltage, sim_current_matched, 'b--', linewidth=2, 
        label='Our Fixed Simulation', alpha=0.8)
ax.fill_between(exp_voltage, exp_current, sim_current_matched, alpha=0.2, color='gray')

ax.set_xlabel('Voltage (V)', fontweight='bold', fontsize=12)
ax.set_ylabel('Current (kA/cm²)', fontweight='bold', fontsize=12)
ax.set_title(f'RTD Validation: Simulation vs Brown 1991\nMAE: {mae:.2f}, Corr: {corr:.4f}', 
            fontweight='bold', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('quick_test_comparison.png', dpi=300, bbox_inches='tight')
print("   Saved: quick_test_comparison.png")

print("\n" + "="*80)
print("QUICK TEST COMPLETE!")
print("="*80)
print("\nThe fixes have been applied:")
print("  [X] Realistic RTD device parameters (Brown 1991)")
print("  [X] Clean resonant tunneling (no damping effects)")
print("  [X] Proper current scaling")
print("  [X] Better interpolation")
print("\nNext: Run 'python monte.py' for full simulation with all features.")
print("="*80 + "\n")
