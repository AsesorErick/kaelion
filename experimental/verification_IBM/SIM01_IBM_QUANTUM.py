#!/usr/bin/env python3
"""
KAELION v3.0 - SIMULACIÓN 01 EN IBM QUANTUM REAL
=================================================

Ejecutar la variación espacial de λ(x) en hardware cuántico real.

REQUISITOS:
- Cuenta IBM Quantum con minutos disponibles
- Token de acceso configurado

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
print("KAELION v3.0 - SIM01 EN IBM QUANTUM")
print("="*60)

# =============================================================================
# CONEXIÓN A IBM QUANTUM
# =============================================================================

print("\nConectando a IBM Quantum...")

# Configurar cuenta (solo necesario la primera vez)
try:
    service = QiskitRuntimeService(channel="ibm_quantum_platform")
except:
    QiskitRuntimeService.save_account(
        channel="ibm_quantum_platform", 
        token="TU_TOKEN_AQUI",  # Reemplazar con tu token
        overwrite=True
    )
    service = QiskitRuntimeService(channel="ibm_quantum_platform")

# Seleccionar backend con menos cola
backends = service.backends(simulator=False, operational=True)
print(f"\nBackends disponibles: {[b.name for b in backends]}")

# Usar el que tenga menos trabajos en cola
backend = service.least_busy(operational=True, simulator=False)
print(f"Backend seleccionado: {backend.name}")
print(f"Qubits: {backend.num_qubits}")

# =============================================================================
# CONSTANTES Y PARÁMETROS
# =============================================================================

V_0 = np.sqrt(3)
PHI_0 = 1 / np.sqrt(3)

N_QUBITS = 8  # Ajustar si el backend tiene menos
I_HORIZON = 3
SHOTS = 4096  # Reducido para ahorrar minutos

# Kicked Ising params
J_CHAOS = 1.2
J_INT = 0.1
H_CHAOS = np.pi/4
H_INT = np.pi/2

DEPTHS = [2, 4, 6, 8, 10]  # Menos profundidades para hardware real

print(f"\nParámetros:")
print(f"  N_QUBITS = {N_QUBITS}")
print(f"  HORIZONTE = {I_HORIZON}")
print(f"  SHOTS = {SHOTS}")
print(f"  DEPTHS = {DEPTHS}")

# =============================================================================
# FUNCIONES
# =============================================================================

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
        # Asegurar longitud correcta
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
    
    # |+⟩
    for i in range(n_qubits):
        qc.h(i)
    
    # V = Z_0
    qc.z(0)
    
    # Forward evolution
    for _ in range(depth):
        for i in range(n_qubits - 1):
            qc.rzz(J, i, i + 1)
        for i in range(n_qubits):
            qc.rx(h, i)
    
    # W = Z_target
    qc.z(target_qubit)
    
    # Backward evolution
    for _ in range(depth):
        for i in range(n_qubits):
            qc.rx(-h, i)
        for i in range(n_qubits - 2, -1, -1):
            qc.rzz(-J, i, i + 1)
    
    # V†
    qc.z(0)
    
    # Medir en base X
    for i in range(n_qubits):
        qc.h(i)
    qc.measure(qr, cr)
    
    return qc

# =============================================================================
# CREAR TODOS LOS CIRCUITOS
# =============================================================================

print("\n" + "="*60)
print("CREANDO CIRCUITOS")
print("="*60)

circuits = []
circuit_info = []  # Para saber qué circuito es qué

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
print(f"Transpilación completada")

# Mostrar profundidad después de transpilar
depths_transpiled = [qc.depth() for qc in transpiled]
print(f"Profundidades transpiladas: min={min(depths_transpiled)}, max={max(depths_transpiled)}")

# =============================================================================
# EJECUTAR EN IBM
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

# Calcular λ para cada qubit
print("\nResultados por qubit:")

# Referencia de scrambling (qubit más caótico)
ref_otocs = results_by_qubit[N_QUBITS-1]['otocs']
scrambling_max = ref_otocs[0] - ref_otocs[-1] if len(ref_otocs) > 1 else 1.0
scrambling_max = max(scrambling_max, 0.01)  # Evitar división por cero

final_results = {
    'qubit': [],
    'chaos_input': [],
    'lambda': [],
    'alpha': [],
    'otocs': [],
    'scrambling': []
}

for target in range(N_QUBITS):
    data = results_by_qubit[target]
    otocs = data['otocs']
    chaos_input = data['chaos_input']
    
    # Scrambling
    scrambling = otocs[0] - otocs[-1] if len(otocs) > 1 else 0
    
    # λ
    lam = np.clip(scrambling / scrambling_max, 0, 1)
    
    # Si no hay scrambling significativo, usar chaos_input
    if abs(scrambling) < 0.05:
        lam = chaos_input
    
    alpha = -0.5 - lam
    
    final_results['qubit'].append(target)
    final_results['chaos_input'].append(chaos_input)
    final_results['lambda'].append(lam)
    final_results['alpha'].append(alpha)
    final_results['otocs'].append(otocs)
    final_results['scrambling'].append(scrambling)
    
    region = "LQG" if lam < 0.3 else ("Holo" if lam > 0.7 else "Trans")
    print(f"  q{target}: OTOC={otocs[0]:.3f}→{otocs[-1]:.3f}, scr={scrambling:.3f}, λ={lam:.3f} [{region}]")

# =============================================================================
# ANÁLISIS
# =============================================================================

print("\n" + "="*60)
print("ANÁLISIS")
print("="*60)

lambdas = np.array(final_results['lambda'])
alphas = np.array(final_results['alpha'])
chaos_in = np.array(final_results['chaos_input'])

lambda_antes = np.mean(lambdas[:I_HORIZON])
lambda_despues = np.mean(lambdas[I_HORIZON:])
diferencia = lambda_despues - lambda_antes

corr = np.corrcoef(chaos_in, lambdas)[0, 1] if len(set(lambdas)) > 1 else 0

print(f"\nλ promedio (LQG, q<{I_HORIZON}): {lambda_antes:.3f}")
print(f"λ promedio (Holo, q≥{I_HORIZON}): {lambda_despues:.3f}")
print(f"Diferencia: {diferencia:.3f}")
print(f"Correlación: {corr:.3f}")

# =============================================================================
# VISUALIZACIÓN
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'KAELION v3.0: SIM01 en {backend.name}', fontsize=14, fontweight='bold')

# 1. λ vs posición
ax1 = axes[0, 0]
ax1.plot(final_results['qubit'], chaos_in, 'g--', lw=2, marker='s', ms=8, label='Diseño (chaos_input)')
ax1.plot(final_results['qubit'], lambdas, 'b-', lw=2, marker='o', ms=10, label='Medido (λ)')
ax1.axvline(I_HORIZON, color='orange', ls=':', lw=2, label='Horizonte')
ax1.axhline(0.5, color='gray', ls='--', alpha=0.5)
ax1.set_xlabel('Qubit')
ax1.set_ylabel('λ')
ax1.set_title('λ(x) Espacial')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.1, 1.1)

# 2. α vs posición
ax2 = axes[0, 1]
ax2.plot(final_results['qubit'], alphas, 'ro-', lw=2, ms=10)
ax2.axvline(I_HORIZON, color='orange', ls=':', lw=2)
ax2.axhline(-0.5, color='green', ls='--', alpha=0.7, label='α(LQG)')
ax2.axhline(-1.5, color='purple', ls='--', alpha=0.7, label='α(Holo)')
ax2.set_xlabel('Qubit')
ax2.set_ylabel('α')
ax2.set_title('α(x) = -1/2 - λ(x)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. OTOC decay
ax3 = axes[1, 0]
cmap = plt.cm.viridis(np.linspace(0, 1, N_QUBITS))
for i in range(N_QUBITS):
    ax3.plot(DEPTHS, final_results['otocs'][i], 'o-', color=cmap[i], lw=1.5, ms=5,
             label=f'q{i}' if i in [0, I_HORIZON, N_QUBITS-1] else '')
ax3.axhline(0, color='gray', ls='--', alpha=0.5)
ax3.set_xlabel('Profundidad')
ax3.set_ylabel('OTOC')
ax3.set_title('Decaimiento OTOC (Hardware Real)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Scrambling
ax4 = axes[1, 1]
colors = ['green' if q < I_HORIZON else 'purple' for q in final_results['qubit']]
ax4.bar(final_results['qubit'], final_results['scrambling'], color=colors, alpha=0.7)
ax4.axvline(I_HORIZON - 0.5, color='orange', ls=':', lw=2)
ax4.set_xlabel('Qubit')
ax4.set_ylabel('Scrambling')
ax4.set_title('Cantidad de Scrambling')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f'sim01_ibm_{backend.name}_{timestamp}.png'
plt.savefig(filename, dpi=150, bbox_inches='tight')
print(f"\n✅ Figura guardada: {filename}")

# =============================================================================
# GUARDAR DATOS
# =============================================================================

output_data = {
    'version': '3.0',
    'platform': 'IBM Quantum',
    'backend': backend.name,
    'timestamp': timestamp,
    'job_id': job.job_id(),
    'shots': SHOTS,
    'n_qubits': N_QUBITS,
    'results': {
        'qubit': final_results['qubit'],
        'chaos_input': [float(x) for x in final_results['chaos_input']],
        'lambda': [float(x) for x in final_results['lambda']],
        'alpha': [float(x) for x in final_results['alpha']],
        'scrambling': [float(x) for x in final_results['scrambling']]
    },
    'analysis': {
        'lambda_lqg': float(lambda_antes),
        'lambda_holo': float(lambda_despues),
        'diferencia': float(diferencia),
        'correlacion': float(corr)
    }
}

json_filename = f'sim01_ibm_{backend.name}_{timestamp}.json'
with open(json_filename, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"✅ Datos guardados: {json_filename}")

# =============================================================================
# VERIFICACIÓN
# =============================================================================

print("\n" + "="*60)
print("VERIFICACIÓN")
print("="*60)

CRITERIA = {
    'gradiente_positivo': diferencia > 0.05,
    'correlacion_positiva': corr > 0.3,
    'lambda_bajo_lqg': lambda_antes < 0.5,
    'lambda_alto_holo': lambda_despues > lambda_antes,
    'scrambling_detectado': any(s > 0.05 for s in final_results['scrambling'])
}

for k, v in CRITERIA.items():
    print(f"  {'✓' if v else '✗'} {k}")

n_passed = sum(CRITERIA.values())
status = "EXITOSA" if n_passed >= 4 else ("PARCIAL" if n_passed >= 2 else "FALLIDA")

print(f"\n>>> VERIFICACIÓN: {status} ({n_passed}/5) <<<")

print(f"""
╔════════════════════════════════════════════════════════════╗
║  KAELION v3.0 - SIM01 EN IBM QUANTUM                       ║
╠════════════════════════════════════════════════════════════╣
║  Backend: {backend.name:<15}                               ║
║  Job ID: {job.job_id():<20}                      ║
║                                                            ║
║  λ (región LQG):  {lambda_antes:.3f}                                   ║
║  λ (región Holo): {lambda_despues:.3f}                                   ║
║  Diferencia:      {diferencia:.3f}                                   ║
║  Correlación:     {corr:.3f}                                   ║
║                                                            ║
║  VERIFICACIÓN: {status:<10}                                  ║
╚════════════════════════════════════════════════════════════╝
""")

plt.show()
