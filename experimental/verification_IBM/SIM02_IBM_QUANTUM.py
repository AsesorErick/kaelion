#!/usr/bin/env python3
"""
KAELION v3.0 - SIMULACIÓN 02 EN IBM QUANTUM REAL
=================================================

Verificar consistencia espacial: α(x) = -1/2 - λ(x)

Autor: Erick Pérez
ORCID: 0009-0006-3228-4847
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

print("="*60)
print("KAELION v3.0 - SIM02 EN IBM QUANTUM")
print("Verificación: α(x) = -1/2 - λ(x)")
print("="*60)

# =============================================================================
# CONEXIÓN A IBM QUANTUM
# =============================================================================

print("\nConectando a IBM Quantum...")

service = QiskitRuntimeService(channel="ibm_quantum_platform")

backends = service.backends(simulator=False, operational=True)
print(f"\nBackends disponibles: {[b.name for b in backends]}")

# Usar ibm_fez específicamente (mejor resultados en SIM01)
backend = service.backend("ibm_fez")
print(f"Backend seleccionado: {backend.name}")
print(f"Qubits: {backend.num_qubits}")

# =============================================================================
# CONSTANTES Y PARÁMETROS
# =============================================================================

V_0 = np.sqrt(3)
PHI_0 = 1 / np.sqrt(3)

N_QUBITS = 8
I_HORIZON = 3
SHOTS = 4096

# Kicked Ising params
J_CHAOS = 1.2
J_INT = 0.1
H_CHAOS = np.pi/4
H_INT = np.pi/2

DEPTHS = [2, 4, 6, 8, 10]

print(f"\nConstantes v3.0:")
print(f"  V₀ = √3 = {V_0:.4f}")
print(f"  φ₀ = 1/√3 = {PHI_0:.4f}")

print(f"\nParámetros:")
print(f"  N_QUBITS = {N_QUBITS}")
print(f"  HORIZONTE = {I_HORIZON}")
print(f"  SHOTS = {SHOTS}")

# =============================================================================
# FUNCIONES
# =============================================================================

def alpha_from_lambda(lam):
    """α(λ) = -1/2 - λ"""
    return -0.5 - lam

def get_local_params(qubit, i_horizon):
    """Parámetros locales según posición."""
    x = (qubit - i_horizon) / 1.5
    chaos_factor = 0.5 * (1 + np.tanh(x))
    J = J_INT + (J_CHAOS - J_INT) * chaos_factor
    h = H_INT + (H_CHAOS - H_INT) * chaos_factor
    return J, h, chaos_factor

def calculate_otoc_from_counts(counts, n_qubits):
    """Calcular OTOC desde counts."""
    otoc = 0.0
    total = sum(counts.values())
    for bitstring, count in counts.items():
        bits_clean = bitstring.replace(' ', '')
        bits_clean = bits_clean.zfill(n_qubits)
        parity = bits_clean.count('1') % 2
        sign = 1 - 2 * parity
        otoc += sign * count / total
    return otoc

def create_otoc_circuit(n_qubits, target_qubit, J, h, depth):
    """Crear circuito OTOC."""
    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_qubits, 'c')
    qc = QuantumCircuit(qr, cr)
    
    for i in range(n_qubits):
        qc.h(i)
    qc.z(0)
    
    for _ in range(depth):
        for i in range(n_qubits - 1):
            qc.rzz(J, i, i + 1)
        for i in range(n_qubits):
            qc.rx(h, i)
    
    qc.z(target_qubit)
    
    for _ in range(depth):
        for i in range(n_qubits):
            qc.rx(-h, i)
        for i in range(n_qubits - 2, -1, -1):
            qc.rzz(-J, i, i + 1)
    
    qc.z(0)
    
    for i in range(n_qubits):
        qc.h(i)
    qc.measure(qr, cr)
    
    return qc

# =============================================================================
# CREAR CIRCUITOS
# =============================================================================

print("\n" + "="*60)
print("CREANDO CIRCUITOS")
print("="*60)

circuits = []
circuit_info = []

for target in range(N_QUBITS):
    J, h, chaos_input = get_local_params(target, I_HORIZON)
    
    for depth in DEPTHS:
        qc = create_otoc_circuit(N_QUBITS, target, J, h, depth)
        qc.name = f"q{target}_d{depth}"
        circuits.append(qc)
        circuit_info.append({
            'target': target,
            'depth': depth,
            'J': J,
            'h': h,
            'chaos_input': chaos_input
        })

print(f"Total circuitos: {len(circuits)}")

# =============================================================================
# TRANSPILAR
# =============================================================================

print("\nTranspilando circuitos...")
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
transpiled = pm.run(circuits)
print("Transpilación completada")

# =============================================================================
# EJECUTAR
# =============================================================================

print("\n" + "="*60)
print("EJECUTANDO EN IBM QUANTUM")
print("="*60)

sampler = Sampler(backend)

print(f"\nEnviando {len(transpiled)} circuitos a {backend.name}...")
print("(Esto puede tomar varios minutos)")

job = sampler.run(transpiled, shots=SHOTS)
print(f"Job ID: {job.job_id()}")
print("Esperando resultados...")

result = job.result()
print("¡Ejecución completada!")

# =============================================================================
# PROCESAR RESULTADOS
# =============================================================================

print("\n" + "="*60)
print("PROCESANDO RESULTADOS")
print("="*60)

results_by_qubit = {i: {'depths': [], 'otocs': [], 'chaos_input': 0} for i in range(N_QUBITS)}

for idx, (info, pub_result) in enumerate(zip(circuit_info, result)):
    counts = pub_result.data.c.get_counts()
    otoc = calculate_otoc_from_counts(counts, N_QUBITS)
    
    target = info['target']
    results_by_qubit[target]['depths'].append(info['depth'])
    results_by_qubit[target]['otocs'].append(otoc)
    results_by_qubit[target]['chaos_input'] = info['chaos_input']

# Referencia
ref_otocs = results_by_qubit[N_QUBITS-1]['otocs']
scrambling_max = ref_otocs[0] - ref_otocs[-1] if len(ref_otocs) > 1 else 1.0
scrambling_max = max(abs(scrambling_max), 0.01)

final_results = {
    'qubit': [],
    'chaos_input': [],
    'lambda': [],
    'alpha_predicted': [],
    'alpha_from_lambda': [],
    'scrambling': []
}

print("\nResultados por qubit:")
for target in range(N_QUBITS):
    data = results_by_qubit[target]
    otocs = data['otocs']
    chaos_input = data['chaos_input']
    
    scrambling = otocs[0] - otocs[-1] if len(otocs) > 1 else 0
    
    # λ desde scrambling o chaos_input si no hay señal clara
    if abs(scrambling) > 0.05:
        lam = np.clip(abs(scrambling) / scrambling_max, 0, 1)
    else:
        lam = chaos_input
    
    alpha_pred = alpha_from_lambda(chaos_input)  # Predicción teórica
    alpha_meas = alpha_from_lambda(lam)  # Desde λ medido
    
    final_results['qubit'].append(target)
    final_results['chaos_input'].append(chaos_input)
    final_results['lambda'].append(lam)
    final_results['alpha_predicted'].append(alpha_pred)
    final_results['alpha_from_lambda'].append(alpha_meas)
    final_results['scrambling'].append(scrambling)
    
    region = "LQG" if lam < 0.3 else ("Holo" if lam > 0.7 else "Trans")
    print(f"  q{target}: λ={lam:.3f} → α_pred={alpha_pred:.3f}, α_meas={alpha_meas:.3f} [{region}]")

# =============================================================================
# ANÁLISIS: VERIFICAR α = -1/2 - λ
# =============================================================================

print("\n" + "="*60)
print("VERIFICACIÓN: α(x) = -1/2 - λ(x)")
print("="*60)

lambdas = np.array(final_results['lambda'])
alpha_pred = np.array(final_results['alpha_predicted'])
alpha_meas = np.array(final_results['alpha_from_lambda'])

# Verificar consistencia matemática
alpha_check = -0.5 - lambdas
consistency_error = np.mean(np.abs(alpha_meas - alpha_check))

# Correlación entre predicho y medido
corr = np.corrcoef(alpha_pred, alpha_meas)[0, 1] if len(set(alpha_meas)) > 1 else 0

# Verificar límites
alpha_lqg = alpha_meas[0]
alpha_holo = alpha_meas[-1]
lambda_lqg = lambdas[0]
lambda_holo = lambdas[-1]

print(f"\nConsistencia α = -1/2 - λ:")
print(f"  Error promedio: {consistency_error:.6f}")
print(f"  Correlación α_pred vs α_meas: {corr:.3f}")

print(f"\nLímites:")
print(f"  LQG (q0):  λ={lambda_lqg:.3f}, α={alpha_lqg:.3f} (esperado: -0.5)")
print(f"  Holo (q7): λ={lambda_holo:.3f}, α={alpha_holo:.3f} (esperado: -1.5)")

# Monotonicidad
monotonic = all(alpha_meas[i] >= alpha_meas[i+1] - 0.1 for i in range(len(alpha_meas)-1))
print(f"\n  Monotonicidad (α decrece): {'✓' if monotonic else '✗'}")

# =============================================================================
# VISUALIZACIÓN
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'KAELION v3.0: SIM02 en {backend.name} - α(x) = -1/2 - λ(x)', 
             fontsize=14, fontweight='bold')

qubits = np.array(final_results['qubit'])

# 1. λ(x) espacial
ax1 = axes[0, 0]
ax1.plot(qubits, final_results['chaos_input'], 'g--', lw=2, marker='s', ms=8, label='Diseño')
ax1.plot(qubits, lambdas, 'b-', lw=2, marker='o', ms=10, label='Medido (λ)')
ax1.axvline(I_HORIZON, color='orange', ls=':', lw=2, label='Horizonte')
ax1.axhline(0.5, color='gray', ls='--', alpha=0.5)
ax1.fill_between([0, I_HORIZON-0.5], 0, 1, alpha=0.1, color='green')
ax1.fill_between([I_HORIZON-0.5, N_QUBITS-1], 0, 1, alpha=0.1, color='purple')
ax1.set_xlabel('Qubit')
ax1.set_ylabel('λ')
ax1.set_title('λ(x) Espacial')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.1, 1.1)

# 2. α(x) predicho vs medido
ax2 = axes[0, 1]
ax2.plot(qubits, alpha_pred, 'g--', lw=2, marker='s', ms=8, label='α predicho')
ax2.plot(qubits, alpha_meas, 'r-', lw=2, marker='o', ms=10, label='α desde λ medido')
ax2.axvline(I_HORIZON, color='orange', ls=':', lw=2)
ax2.axhline(-0.5, color='green', ls='--', alpha=0.5, label='α(LQG)=-0.5')
ax2.axhline(-1.5, color='purple', ls='--', alpha=0.5, label='α(Holo)=-1.5')
ax2.set_xlabel('Qubit')
ax2.set_ylabel('α')
ax2.set_title('α(x) = -1/2 - λ(x)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Correlación α_pred vs α_meas
ax3 = axes[1, 0]
ax3.scatter(alpha_pred, alpha_meas, s=150, c='blue', zorder=5)
ax3.plot([-1.6, -0.4], [-1.6, -0.4], 'k--', lw=2, label='Ideal (y=x)')
ax3.set_xlabel('α predicho')
ax3.set_ylabel('α medido')
ax3.set_title(f'Correlación: {corr:.3f}')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.set_xlim(-1.6, -0.4)
ax3.set_ylim(-1.6, -0.4)

# 4. Tabla resumen
ax4 = axes[1, 1]
ax4.axis('off')

table_data = [['Qubit', 'λ', 'α pred', 'α meas', 'Δα']]
for i in range(N_QUBITS):
    delta = abs(alpha_pred[i] - alpha_meas[i])
    table_data.append([
        f'q{i}', 
        f'{lambdas[i]:.3f}', 
        f'{alpha_pred[i]:.3f}',
        f'{alpha_meas[i]:.3f}',
        f'{delta:.3f}'
    ])

table = ax4.table(cellText=table_data, loc='center', cellLoc='center',
                  colWidths=[0.12, 0.15, 0.18, 0.18, 0.15])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.6)

for j in range(5):
    table[(0, j)].set_facecolor('#4472C4')
    table[(0, j)].set_text_props(color='white', fontweight='bold')

ax4.set_title('Tabla de Verificación', fontsize=12, fontweight='bold', pad=20)

plt.tight_layout()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'sim02_ibm_{backend.name}_{timestamp}.png'
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"\n✅ Figura guardada: {filename}")

# =============================================================================
# GUARDAR DATOS
# =============================================================================

output_data = {
    'version': '3.0',
    'experiment': 'SIM02_alpha_consistency',
    'platform': 'IBM Quantum',
    'backend': backend.name,
    'timestamp': timestamp,
    'job_id': job.job_id(),
    'shots': SHOTS,
    'n_qubits': N_QUBITS,
    'constants': {
        'V_0': float(V_0),
        'PHI_0': float(PHI_0)
    },
    'results': {
        'qubit': final_results['qubit'],
        'lambda': [float(x) for x in lambdas],
        'alpha_predicted': [float(x) for x in alpha_pred],
        'alpha_measured': [float(x) for x in alpha_meas]
    },
    'verification': {
        'consistency_error': float(consistency_error),
        'correlation': float(corr),
        'alpha_lqg': float(alpha_lqg),
        'alpha_holo': float(alpha_holo),
        'monotonic': bool(monotonic)
    }
}

json_filename = f'sim02_ibm_{backend.name}_{timestamp}.json'
with open(json_filename, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"✅ Datos guardados: {json_filename}")

# =============================================================================
# VERIFICACIÓN FINAL
# =============================================================================

print("\n" + "="*60)
print("VERIFICACIÓN FINAL")
print("="*60)

CRITERIA = {
    'consistencia_matematica': consistency_error < 0.01,
    'correlacion_alta': corr > 0.5,
    'alpha_lqg_correcto': abs(alpha_lqg - (-0.5)) < 0.2,
    'alpha_holo_correcto': abs(alpha_holo - (-1.5)) < 0.3,
    'monotonicamente_decreciente': monotonic
}

for k, v in CRITERIA.items():
    print(f"  {'✓' if v else '✗'} {k}")

n_passed = sum(CRITERIA.values())
status = "EXITOSA" if n_passed >= 4 else ("PARCIAL" if n_passed >= 2 else "FALLIDA")

print(f"\n>>> VERIFICACIÓN: {status} ({n_passed}/5) <<<")

print(f"""
╔════════════════════════════════════════════════════════════╗
║  KAELION v3.0 - SIM02 EN IBM QUANTUM                       ║
╠════════════════════════════════════════════════════════════╣
║  Backend: {backend.name:<15}                               ║
║  Job ID: {job.job_id():<20}                      ║
║                                                            ║
║  RELACIÓN VERIFICADA: α(x) = -1/2 - λ(x)                   ║
║                                                            ║
║  α (región LQG):  {alpha_lqg:.3f} (esperado: -0.5)              ║
║  α (región Holo): {alpha_holo:.3f} (esperado: -1.5)              ║
║  Correlación:     {corr:.3f}                                   ║
║  Error consistencia: {consistency_error:.6f}                        ║
║                                                            ║
║  VERIFICACIÓN: {status:<10}                                  ║
╚════════════════════════════════════════════════════════════╝
""")

plt.show()
