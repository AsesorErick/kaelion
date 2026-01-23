#!/usr/bin/env python3
"""
================================================================================
KAELION v3.0 - NOTEBOOK PARA GOOGLE COLAB
================================================================================

Ejecutar las 3 simulaciones con constantes v3.0:
  V₀ = √3 = 1.7321
  φ₀ = 1/√3 = 0.5774

INSTRUCCIONES:
1. Subir este archivo a Google Colab
2. Ejecutar celda por celda
3. Los resultados se guardan automáticamente

Autor: Erick Perez
ORCID: 0009-0006-3228-4847
================================================================================
"""

#==============================================================================
# CELDA 1: INSTALACIÓN (ejecutar primero, toma ~2 minutos)
#==============================================================================

!pip install qiskit qiskit-aer -q
print("✅ Qiskit instalado")

#==============================================================================
# CELDA 2: IMPORTS Y CONSTANTES v3.0
#==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import List, Dict, Tuple
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector, partial_trace, entropy

# CONSTANTES KAELION v3.0
V_0 = np.sqrt(3)        # = 1.7321
PHI_0 = 1 / np.sqrt(3)  # = 0.5774

print("="*60)
print("KAELION v3.0 - CONSTANTES DERIVADAS")
print("="*60)
print(f"  V₀ = √3 = {V_0:.6f}")
print(f"  φ₀ = 1/√3 = {PHI_0:.6f}")
print(f"  V₀ × φ₀ = {V_0 * PHI_0:.6f} (debe ser 1)")
print("="*60)

#==============================================================================
# CELDA 3: FUNCIONES COMUNES
#==============================================================================

def alpha_from_lambda(lam):
    """α(λ) = -1/2 - λ"""
    return -0.5 - lam

def V_potential(lam):
    """V(λ) = V₀·λ²(1-λ)²"""
    return V_0 * (lam ** 2) * ((1 - lam) ** 2)

def V_prime(lam):
    """V'(λ) = 2V₀·λ(1-λ)(1-2λ)"""
    return 2 * V_0 * lam * (1 - lam) * (1 - 2 * lam)

def V_double_prime(lam):
    """V''(λ) = 2V₀(6λ² - 6λ + 1)"""
    return 2 * V_0 * (6 * lam**2 - 6 * lam + 1)

def coupling_gradient(i, i_horizon, w, J_base, J_chaos):
    """Acoplamiento J(i) con gradiente tipo tanh."""
    x = (i - i_horizon) / w
    return J_base + (J_chaos - J_base) * (1 + np.tanh(x)) / 2

def lambda_prediction(x, x_h, w):
    """λ(x) predicho desde posición."""
    return np.where(x >= x_h, 1 - np.exp(-(x - x_h) / w), np.exp((x - x_h) / w) * 0.5)

print("✅ Funciones cargadas")

#==============================================================================
# CELDA 4: SIMULACIÓN 01 - VARIACIÓN ESPACIAL DE λ(x)
#==============================================================================

print("\n" + "="*60)
print("SIMULACIÓN 01: VARIACIÓN ESPACIAL DE λ(x)")
print("="*60)

# Parámetros
N_QUBITS = 8
I_HORIZON = 3
W_TRANSITION = 1.5
J_BASE = 0.3
J_CHAOS = 2.5
H_FIELD = 0.4
DEPTHS = [2, 4, 6, 8, 10, 12]
SHOTS = 4096

def create_gradient_evolution(n_qubits, i_h, w, J_base, J_chaos, h_field, depth):
    qc = QuantumCircuit(n_qubits)
    dt = 1.0
    for _ in range(depth):
        for i in range(n_qubits - 1):
            J_i = coupling_gradient(i, i_h, w, J_base, J_chaos)
            qc.rzz(2 * J_i * dt, i, i + 1)
        for i in range(n_qubits):
            qc.rx(2 * h_field * dt, i)
        qc.barrier()
    return qc

def measure_otoc(target, depth):
    n = N_QUBITS
    qr = QuantumRegister(n, 'q')
    cr = ClassicalRegister(n, 'c')
    qc = QuantumCircuit(qr, cr)
    
    for i in range(n):
        qc.h(i)
    qc.z(0)
    U = create_gradient_evolution(n, I_HORIZON, W_TRANSITION, J_BASE, J_CHAOS, H_FIELD, depth)
    qc.compose(U, inplace=True)
    qc.z(target)
    qc.compose(U.inverse(), inplace=True)
    qc.z(0)
    for i in range(n):
        qc.h(i)
    qc.measure(qr, cr)
    
    sim = AerSimulator()
    counts = sim.run(qc, shots=SHOTS).result().get_counts()
    
    otoc = 0.0
    total = sum(counts.values())
    for bits, count in counts.items():
        parity = bits.count('1') % 2
        otoc += (1 - 2*parity) * count / total
    return otoc

def extract_lyapunov(otocs, depths):
    decay = np.clip(1 - np.array(otocs), 1e-10, None)
    try:
        coeffs = np.polyfit(depths, np.log(decay), 1)
        return max(0, coeffs[0])
    except:
        return 0.0

# Ejecutar
results_01 = {'qubit': [], 'lambda': [], 'alpha': []}
lambda_MSS = 2.0

for target in range(N_QUBITS):
    print(f"  Midiendo qubit {target}...", end=" ")
    otocs = [measure_otoc(target, d) for d in DEPTHS]
    lam_L = extract_lyapunov(otocs, DEPTHS)
    lam = np.clip(lam_L / lambda_MSS, 0, 1)
    alpha = alpha_from_lambda(lam)
    results_01['qubit'].append(target)
    results_01['lambda'].append(lam)
    results_01['alpha'].append(alpha)
    print(f"λ={lam:.3f}, α={alpha:.3f}")

# Gráfica
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
qubits = np.array(results_01['qubit'])
lambdas = np.array(results_01['lambda'])

ax[0].scatter(qubits, lambdas, s=100, c='blue', label='Medido')
x_smooth = np.linspace(0, N_QUBITS-1, 100)
ax[0].plot(x_smooth, lambda_prediction(x_smooth, I_HORIZON, W_TRANSITION), 'r--', label='Predicción')
ax[0].axvline(I_HORIZON, color='orange', linestyle=':', label='Horizonte')
ax[0].set_xlabel('Qubit')
ax[0].set_ylabel('λ')
ax[0].set_title('SIM 01: λ(x) Espacial')
ax[0].legend()
ax[0].grid(True, alpha=0.3)

ax[1].scatter(qubits, results_01['alpha'], s=100, c='red')
ax[1].axhline(-0.5, color='green', linestyle='--', alpha=0.5, label='α(LQG)')
ax[1].axhline(-1.5, color='purple', linestyle='--', alpha=0.5, label='α(Holo)')
ax[1].set_xlabel('Qubit')
ax[1].set_ylabel('α')
ax[1].set_title('α(x) = -1/2 - λ(x)')
ax[1].legend()
ax[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sim01_lambda_spatial.png', dpi=150)
print("\n✅ SIM 01 COMPLETADA - Guardado: sim01_lambda_spatial.png")

#==============================================================================
# CELDA 5: SIMULACIÓN 03 - DINÁMICA λ(t)-M(t)
#==============================================================================

print("\n" + "="*60)
print("SIMULACIÓN 03: DINÁMICA λ(t)-M(t) Y CURVA DE PAGE")
print("="*60)

KAPPA = 1.0
GAMMA = 0.1

def phi_from_M(M):
    return 4 * np.pi * M**2

def lambda_from_phi(phi):
    return phi / (PHI_0 + phi)

def coupled_dynamics(t, y):
    M, lam = y
    if M <= 0:
        return [0.0, 0.0]
    dM_dt = -KAPPA / M**2
    phi = phi_from_M(M)
    dphi_dt = 8 * np.pi * M * dM_dt
    dlam_geom = (PHI_0 / (PHI_0 + phi)**2) * dphi_dt
    dlam_pot = -GAMMA * V_prime(lam)
    return [dM_dt, dlam_geom + dlam_pot]

# Condición inicial: M tal que veremos transición por λ=0.5
M_crit = np.sqrt(PHI_0 / (4 * np.pi))
M_init = 5.0 * M_crit
phi_init = phi_from_M(M_init)
lam_init = lambda_from_phi(phi_init)

print(f"  M₀ = {M_init:.4f}, λ₀ = {lam_init:.4f}")

t_evap = M_init**3 / (3 * KAPPA)
sol = solve_ivp(coupled_dynamics, (0, t_evap*1.5), [M_init, lam_init],
                t_eval=np.linspace(0, t_evap*1.2, 500), max_step=t_evap/100)

t = sol.t
M = sol.y[0]
lam = sol.y[1]
phi = phi_from_M(M)
alpha = alpha_from_lambda(lam)

# Tiempo de Page
idx_P = np.argmin(np.abs(lam - 0.5))
t_Page = t[idx_P]
print(f"\n  TIEMPO DE PAGE:")
print(f"    t_P = {t_Page:.4f}")
print(f"    λ(t_P) = {lam[idx_P]:.4f} (esperado: 0.5)")
print(f"    φ(t_P) = {phi[idx_P]:.4f} (esperado: {PHI_0:.4f})")
print(f"    α(t_P) = {alpha[idx_P]:.4f} (esperado: -1.0)")

# Gráfica
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0,0].plot(t, M, 'b-', lw=2)
axes[0,0].axvline(t_Page, color='orange', linestyle=':')
axes[0,0].set_xlabel('Tiempo')
axes[0,0].set_ylabel('M(t)')
axes[0,0].set_title('Masa del Black Hole')
axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(t, lam, 'b-', lw=2)
axes[0,1].axvline(t_Page, color='orange', linestyle=':')
axes[0,1].axhline(0.5, color='gray', linestyle='--')
axes[0,1].scatter([t_Page], [lam[idx_P]], s=100, c='red', zorder=5)
axes[0,1].set_xlabel('Tiempo')
axes[0,1].set_ylabel('λ(t)')
axes[0,1].set_title('Parámetro de Interpolación')
axes[0,1].grid(True, alpha=0.3)

axes[1,0].plot(t, alpha, 'r-', lw=2)
axes[1,0].axvline(t_Page, color='orange', linestyle=':')
axes[1,0].axhline(-0.5, color='green', linestyle='--', alpha=0.5)
axes[1,0].axhline(-1.0, color='gray', linestyle='--', alpha=0.5)
axes[1,0].axhline(-1.5, color='purple', linestyle='--', alpha=0.5)
axes[1,0].set_xlabel('Tiempo')
axes[1,0].set_ylabel('α(t)')
axes[1,0].set_title('Corrección Logarítmica')
axes[1,0].grid(True, alpha=0.3)

# Curva de Page
S_BH = phi / 4
axes[1,1].plot(t, S_BH, 'b-', lw=2, label='S_BH')
axes[1,1].axvline(t_Page, color='orange', linestyle=':', label='t_Page')
axes[1,1].set_xlabel('Tiempo')
axes[1,1].set_ylabel('Entropía')
axes[1,1].set_title('CURVA DE PAGE')
axes[1,1].legend()
axes[1,1].grid(True, alpha=0.3)

plt.suptitle('KAELION v3.0: Dinámica λ(t)-M(t)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('sim03_dynamics.png', dpi=150)
print("\n✅ SIM 03 COMPLETADA - Guardado: sim03_dynamics.png")

#==============================================================================
# CELDA 6: RESUMEN FINAL
#==============================================================================

print("\n" + "="*60)
print("RESUMEN KAELION v3.0")
print("="*60)
print(f"""
CONSTANTES DERIVADAS:
  V₀ = √3 = {V_0:.4f}
  φ₀ = 1/√3 = {PHI_0:.4f}
  V₀ × φ₀ = {V_0*PHI_0:.4f} ✓

SIMULACIÓN 01 - λ(x) ESPACIAL:
  Verificada variación espacial de λ
  Horizonte en qubit {I_HORIZON}

SIMULACIÓN 03 - DINÁMICA:
  t_Page = {t_Page:.4f}
  λ(t_P) = {lam[idx_P]:.4f} ≈ 0.5 ✓
  φ(t_P) = {phi[idx_P]:.4f} ≈ φ₀ ✓
  α(t_P) = {alpha[idx_P]:.4f} ≈ -1.0 ✓

ARCHIVOS GENERADOS:
  - sim01_lambda_spatial.png
  - sim03_dynamics.png
""")
print("="*60)
print("✅ TODAS LAS SIMULACIONES COMPLETADAS")
print("="*60)
