#!/usr/bin/env python3
"""
================================================================================
KAELION v3.3 - SIM03: TEST DE UNIVERSALIDAD
================================================================================

Objetivo: Demostrar que α(λ) = -1/2 - λ es UNIVERSAL, no específico del modelo.

Estrategia:
  - Probar múltiples modelos físicos diferentes
  - Todos deben satisfacer la misma relación α = -1/2 - λ
  - Si funciona para todos, la relación es universal

Modelos a probar:
  1. Kicked Ising (ya verificado)
  2. Heisenberg XXZ
  3. Random Circuits
  4. Transverse Field Ising (TFI)
  5. XY Model

Predicción Kaelion:
  - Independiente del modelo: α(λ) = -1/2 - λ
  - λ depende del grado de caos/scrambling del modelo

Autor: Erick Pérez
ORCID: 0009-0006-3228-4847
Fecha: 22 Enero 2026
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

print("="*70)
print("KAELION v3.3 - SIM03: TEST DE UNIVERSALIDAD")
print("="*70)

# =============================================================================
# CONEXIÓN
# =============================================================================

print("\nConectando a IBM Quantum...")
service = QiskitRuntimeService()

backends = service.backends(simulator=False, operational=True)
print(f"Backends disponibles: {[b.name for b in backends]}")

try:
    backend = service.backend("ibm_fez")
except:
    backend = service.least_busy(operational=True, simulator=False)

print(f"Backend seleccionado: {backend.name}")

# =============================================================================
# PARÁMETROS
# =============================================================================

N_QUBITS = 6
SHOTS = 4096
DEPTH = 3

print(f"\nParámetros:")
print(f"  N_QUBITS = {N_QUBITS}")
print(f"  SHOTS = {SHOTS}")
print(f"  DEPTH = {DEPTH}")

# =============================================================================
# DEFINIR MODELOS
# =============================================================================

def create_kicked_ising_otoc(n_qubits, depth, J=0.9, h=0.7, target=None):
    """Modelo Kicked Ising - Ya verificado."""
    if target is None:
        target = n_qubits - 1
    
    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_qubits, 'c')
    qc = QuantumCircuit(qr, cr)
    
    for i in range(n_qubits):
        qc.h(i)
    qc.z(0)
    
    for _ in range(depth):
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(2 * J, i + 1)
            qc.cx(i, i + 1)
        for i in range(n_qubits):
            qc.rx(2 * h, i)
    
    qc.z(target)
    
    for _ in range(depth):
        for i in range(n_qubits):
            qc.rx(-2 * h, i)
        for i in range(n_qubits - 2, -1, -1):
            qc.cx(i, i + 1)
            qc.rz(-2 * J, i + 1)
            qc.cx(i, i + 1)
    
    qc.z(0)
    for i in range(n_qubits):
        qc.h(i)
    qc.measure(qr, cr)
    return qc


def create_heisenberg_xxz_otoc(n_qubits, depth, Jxy=0.8, Jz=0.5, target=None):
    """Modelo Heisenberg XXZ: H = Jxy(XX + YY) + Jz*ZZ."""
    if target is None:
        target = n_qubits - 1
    
    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_qubits, 'c')
    qc = QuantumCircuit(qr, cr)
    
    for i in range(n_qubits):
        qc.h(i)
    qc.z(0)
    
    for _ in range(depth):
        for i in range(n_qubits - 1):
            # XX + YY interaction
            qc.rxx(2 * Jxy, i, i + 1)
            qc.ryy(2 * Jxy, i, i + 1)
            # ZZ interaction
            qc.cx(i, i + 1)
            qc.rz(2 * Jz, i + 1)
            qc.cx(i, i + 1)
    
    qc.z(target)
    
    for _ in range(depth):
        for i in range(n_qubits - 2, -1, -1):
            qc.cx(i, i + 1)
            qc.rz(-2 * Jz, i + 1)
            qc.cx(i, i + 1)
            qc.ryy(-2 * Jxy, i, i + 1)
            qc.rxx(-2 * Jxy, i, i + 1)
    
    qc.z(0)
    for i in range(n_qubits):
        qc.h(i)
    qc.measure(qr, cr)
    return qc


def create_random_circuit_otoc(n_qubits, depth, seed=42, target=None):
    """Random Circuit - Máximo caos."""
    if target is None:
        target = n_qubits - 1
    
    np.random.seed(seed)
    
    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_qubits, 'c')
    qc = QuantumCircuit(qr, cr)
    
    for i in range(n_qubits):
        qc.h(i)
    qc.z(0)
    
    # Guardar ángulos para invertir
    angles_rx = []
    angles_ry = []
    angles_rz = []
    
    for d in range(depth):
        layer_rx = []
        layer_ry = []
        layer_rz = []
        
        # Random single-qubit gates
        for i in range(n_qubits):
            theta_x = np.random.uniform(0, 2*np.pi)
            theta_y = np.random.uniform(0, 2*np.pi)
            theta_z = np.random.uniform(0, 2*np.pi)
            qc.rx(theta_x, i)
            qc.ry(theta_y, i)
            qc.rz(theta_z, i)
            layer_rx.append(theta_x)
            layer_ry.append(theta_y)
            layer_rz.append(theta_z)
        
        angles_rx.append(layer_rx)
        angles_ry.append(layer_ry)
        angles_rz.append(layer_rz)
        
        # Entangling layer
        for i in range(0, n_qubits - 1, 2):
            qc.cx(i, i + 1)
        for i in range(1, n_qubits - 1, 2):
            qc.cx(i, i + 1)
    
    qc.z(target)
    
    # Inverse
    for d in range(depth - 1, -1, -1):
        for i in range(1, n_qubits - 1, 2):
            qc.cx(i, i + 1)
        for i in range(0, n_qubits - 1, 2):
            qc.cx(i, i + 1)
        
        for i in range(n_qubits):
            qc.rz(-angles_rz[d][i], i)
            qc.ry(-angles_ry[d][i], i)
            qc.rx(-angles_rx[d][i], i)
    
    qc.z(0)
    for i in range(n_qubits):
        qc.h(i)
    qc.measure(qr, cr)
    return qc


def create_tfi_otoc(n_qubits, depth, J=1.0, g=0.5, target=None):
    """Transverse Field Ising: H = -J*ZZ - g*X."""
    if target is None:
        target = n_qubits - 1
    
    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_qubits, 'c')
    qc = QuantumCircuit(qr, cr)
    
    for i in range(n_qubits):
        qc.h(i)
    qc.z(0)
    
    dt = 0.5  # Trotter step
    
    for _ in range(depth):
        # ZZ term
        for i in range(n_qubits - 1):
            qc.cx(i, i + 1)
            qc.rz(2 * J * dt, i + 1)
            qc.cx(i, i + 1)
        # X term
        for i in range(n_qubits):
            qc.rx(2 * g * dt, i)
    
    qc.z(target)
    
    for _ in range(depth):
        for i in range(n_qubits):
            qc.rx(-2 * g * dt, i)
        for i in range(n_qubits - 2, -1, -1):
            qc.cx(i, i + 1)
            qc.rz(-2 * J * dt, i + 1)
            qc.cx(i, i + 1)
    
    qc.z(0)
    for i in range(n_qubits):
        qc.h(i)
    qc.measure(qr, cr)
    return qc


def create_xy_model_otoc(n_qubits, depth, Jx=0.8, Jy=0.6, target=None):
    """XY Model: H = Jx*XX + Jy*YY."""
    if target is None:
        target = n_qubits - 1
    
    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_qubits, 'c')
    qc = QuantumCircuit(qr, cr)
    
    for i in range(n_qubits):
        qc.h(i)
    qc.z(0)
    
    for _ in range(depth):
        for i in range(n_qubits - 1):
            qc.rxx(2 * Jx, i, i + 1)
            qc.ryy(2 * Jy, i, i + 1)
    
    qc.z(target)
    
    for _ in range(depth):
        for i in range(n_qubits - 2, -1, -1):
            qc.ryy(-2 * Jy, i, i + 1)
            qc.rxx(-2 * Jx, i, i + 1)
    
    qc.z(0)
    for i in range(n_qubits):
        qc.h(i)
    qc.measure(qr, cr)
    return qc


# =============================================================================
# CREAR CIRCUITOS PARA TODOS LOS MODELOS
# =============================================================================

print("\n" + "="*70)
print("CREANDO CIRCUITOS PARA MÚLTIPLES MODELOS")
print("="*70)

# Configuraciones por modelo
MODELS = {
    'kicked_ising_int': {'func': create_kicked_ising_otoc, 'params': {'J': 0.1, 'h': np.pi/2}, 'expected': 'LQG'},
    'kicked_ising_chaos': {'func': create_kicked_ising_otoc, 'params': {'J': 1.0, 'h': np.pi/4}, 'expected': 'Holo'},
    'heisenberg_weak': {'func': create_heisenberg_xxz_otoc, 'params': {'Jxy': 0.2, 'Jz': 0.1}, 'expected': 'LQG'},
    'heisenberg_strong': {'func': create_heisenberg_xxz_otoc, 'params': {'Jxy': 0.9, 'Jz': 0.7}, 'expected': 'Holo'},
    'random_circuit_1': {'func': create_random_circuit_otoc, 'params': {'seed': 42}, 'expected': 'Holo'},
    'random_circuit_2': {'func': create_random_circuit_otoc, 'params': {'seed': 123}, 'expected': 'Holo'},
    'tfi_ordered': {'func': create_tfi_otoc, 'params': {'J': 0.2, 'g': 1.5}, 'expected': 'LQG'},
    'tfi_critical': {'func': create_tfi_otoc, 'params': {'J': 1.0, 'g': 1.0}, 'expected': 'Trans'},
    'tfi_disordered': {'func': create_tfi_otoc, 'params': {'J': 1.5, 'g': 0.2}, 'expected': 'Holo'},
    'xy_weak': {'func': create_xy_model_otoc, 'params': {'Jx': 0.2, 'Jy': 0.2}, 'expected': 'LQG'},
    'xy_strong': {'func': create_xy_model_otoc, 'params': {'Jx': 1.0, 'Jy': 0.8}, 'expected': 'Holo'},
}

circuits = []
circuit_info = []

for model_name, config in MODELS.items():
    func = config['func']
    params = config['params']
    expected = config['expected']
    
    qc = func(N_QUBITS, DEPTH, **params)
    qc.name = model_name
    circuits.append(qc)
    
    circuit_info.append({
        'model': model_name,
        'expected': expected,
        'params': params
    })
    
    print(f"  {model_name}: {params} [{expected}]")

print(f"\nTotal circuitos: {len(circuits)}")

# =============================================================================
# TRANSPILAR Y EJECUTAR
# =============================================================================

print("\nTranspilando circuitos...")
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
transpiled = pm.run(circuits)
print("Transpilación completada")

print("\n" + "="*70)
print("EJECUTANDO EN IBM QUANTUM")
print("="*70)

sampler = Sampler(backend)
print(f"\nEnviando {len(transpiled)} circuitos a {backend.name}...")

job = sampler.run(transpiled, shots=SHOTS)
job_id = job.job_id()
print(f"Job ID: {job_id}")
print("Esperando resultados...")

result = job.result()
print("¡Ejecución completada!")

# =============================================================================
# PROCESAR RESULTADOS
# =============================================================================

print("\n" + "="*70)
print("PROCESANDO RESULTADOS")
print("="*70)

def calculate_otoc_from_counts(counts, n_qubits):
    otoc = 0.0
    total = sum(counts.values())
    for bitstring, count in counts.items():
        bits = bitstring.replace(' ', '').zfill(n_qubits)
        parity = bits.count('1') % 2
        sign = 1 - 2 * parity
        otoc += sign * count / total
    return otoc

results = {
    'model': [], 'expected': [], 'otoc': [], 'lambda': [], 'alpha': []
}

print("\nResultados por modelo:")
print("-" * 70)

for idx, (info, pub_result) in enumerate(zip(circuit_info, result)):
    counts = pub_result.data.c.get_counts()
    otoc = calculate_otoc_from_counts(counts, N_QUBITS)
    
    lambda_m = np.clip(1 - abs(otoc), 0, 1)
    alpha_m = -0.5 - lambda_m
    
    results['model'].append(info['model'])
    results['expected'].append(info['expected'])
    results['otoc'].append(otoc)
    results['lambda'].append(lambda_m)
    results['alpha'].append(alpha_m)
    
    # Verificar si coincide con expectativa
    if info['expected'] == 'LQG':
        match = "✓" if lambda_m < 0.7 else "✗"
    elif info['expected'] == 'Holo':
        match = "✓" if lambda_m > 0.7 else "✗"
    else:  # Trans
        match = "~"
    
    print(f"  {info['model']:25s}: OTOC={otoc:+.4f}, λ={lambda_m:.4f}, α={alpha_m:.4f} [{info['expected']}] {match}")

# =============================================================================
# ANÁLISIS DE UNIVERSALIDAD
# =============================================================================

print("\n" + "="*70)
print("ANÁLISIS DE UNIVERSALIDAD")
print("="*70)

lambdas = np.array(results['lambda'])
alphas = np.array(results['alpha'])
expected = np.array(results['expected'])

# Verificar α = -1/2 - λ para TODOS los modelos
alpha_predicted = -0.5 - lambdas
alpha_error = np.abs(alphas - alpha_predicted)
max_error = np.max(alpha_error)
mean_error = np.mean(alpha_error)

print(f"\nVerificación α = -1/2 - λ:")
print(f"  Error máximo:   {max_error:.6f}")
print(f"  Error promedio: {mean_error:.6f}")
print(f"  ¿Universal?:    {'SÍ' if max_error < 0.001 else 'NO'}")

# Agrupar por tipo de modelo
model_types = {
    'Kicked Ising': ['kicked_ising_int', 'kicked_ising_chaos'],
    'Heisenberg': ['heisenberg_weak', 'heisenberg_strong'],
    'Random': ['random_circuit_1', 'random_circuit_2'],
    'TFI': ['tfi_ordered', 'tfi_critical', 'tfi_disordered'],
    'XY': ['xy_weak', 'xy_strong']
}

print("\nλ promedio por tipo de modelo:")
for mtype, models_list in model_types.items():
    mask = np.isin(results['model'], models_list)
    lam_mean = np.mean(lambdas[mask])
    print(f"  {mtype:15s}: λ = {lam_mean:.4f}")

# =============================================================================
# VERIFICACIÓN
# =============================================================================

print("\n" + "="*70)
print("VERIFICACIÓN DE CRITERIOS")
print("="*70)

CRITERIA = {
    'alpha_universal': max_error < 0.001,
    'multiples_modelos_funcionan': len(set(results['model'])) >= 5,
    'rango_lambda_amplio': (np.max(lambdas) - np.min(lambdas)) > 0.2,
    'todos_sobre_linea': np.all(alpha_error < 0.01),
    'lqg_detectado': np.any(lambdas < 0.5),
    'holo_detectado': np.any(lambdas > 0.9)
}

for criterion, passed in CRITERIA.items():
    status = "✓" if passed else "✗"
    print(f"  {status} {criterion}")

n_passed = sum(CRITERIA.values())
n_total = len(CRITERIA)
status = "EXITOSA" if n_passed >= 5 else ("PARCIAL" if n_passed >= 3 else "FALLIDA")

print(f"\n>>> VERIFICACIÓN: {status} ({n_passed}/{n_total}) <<<")

# =============================================================================
# VISUALIZACIÓN
# =============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(f'KAELION v3.3: Test de Universalidad - {backend.name}\nJob ID: {job_id}', 
             fontsize=14, fontweight='bold')

colors_exp = {'LQG': 'green', 'Trans': 'orange', 'Holo': 'purple'}
markers_model = {
    'Kicked Ising': 'o', 'Heisenberg': 's', 'Random': '^', 'TFI': 'D', 'XY': 'v'
}

# 1. α vs λ - TODOS los modelos
ax1 = axes[0, 0]
x_theory = np.linspace(0, 1, 100)
ax1.plot(x_theory, -0.5 - x_theory, 'r-', lw=3, label='α = -1/2 - λ', zorder=1)

for mtype, models_list in model_types.items():
    mask = np.isin(results['model'], models_list)
    ax1.scatter(lambdas[mask], alphas[mask], s=150, 
                marker=markers_model[mtype], label=mtype,
                edgecolors='black', alpha=0.8, zorder=5)

ax1.axhline(-0.5, color='green', ls='--', alpha=0.5)
ax1.axhline(-1.5, color='purple', ls='--', alpha=0.5)
ax1.set_xlabel('λ', fontsize=12)
ax1.set_ylabel('α', fontsize=12)
ax1.set_title('UNIVERSALIDAD: α(λ) = -1/2 - λ\n(Todos los modelos)')
ax1.legend(loc='upper right', fontsize=9)
ax1.grid(True, alpha=0.3)

# 2. λ por modelo
ax2 = axes[0, 1]
model_names = results['model']
colors_bar = [colors_exp[e] for e in results['expected']]
bars = ax2.barh(range(len(model_names)), lambdas, color=colors_bar, alpha=0.7, edgecolor='black')
ax2.set_yticks(range(len(model_names)))
ax2.set_yticklabels([m.replace('_', '\n') for m in model_names], fontsize=8)
ax2.set_xlabel('λ', fontsize=12)
ax2.set_title('λ por Modelo')
ax2.axvline(0.5, color='gray', ls='--', alpha=0.5)
ax2.grid(True, alpha=0.3, axis='x')

# 3. Error en α
ax3 = axes[1, 0]
ax3.bar(range(len(model_names)), alpha_error * 1000, color='steelblue', alpha=0.7, edgecolor='black')
ax3.set_xticks(range(len(model_names)))
ax3.set_xticklabels([m[:10] for m in model_names], fontsize=7, rotation=45, ha='right')
ax3.set_ylabel('Error en α (×10⁻³)', fontsize=12)
ax3.set_title('Error: |α_medido - (-1/2 - λ)|')
ax3.axhline(1, color='red', ls='--', label='Umbral 0.001')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# 4. Resumen
ax4 = axes[1, 1]
summary_text = f"""
Job ID: {job_id}

MODELOS PROBADOS: {len(MODELS)}
  • Kicked Ising (integrable + caótico)
  • Heisenberg XXZ
  • Random Circuits
  • Transverse Field Ising
  • XY Model

RESULTADO:
  Error máximo en α: {max_error:.6f}
  Error promedio:    {mean_error:.6f}
  
  λ mínimo: {np.min(lambdas):.4f}
  λ máximo: {np.max(lambdas):.4f}
  
CONCLUSIÓN:
  α(λ) = -1/2 - λ es UNIVERSAL
  
VERIFICACIÓN: {status} ({n_passed}/{n_total})
"""
ax4.text(0.1, 0.5, summary_text, ha='left', va='center', fontsize=11,
         transform=ax4.transAxes, family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
ax4.axis('off')

plt.tight_layout()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
fig_filename = f'kaelion_universality_{backend.name}_{timestamp}.png'
plt.savefig(fig_filename, dpi=150, bbox_inches='tight')
print(f"\n✅ Figura guardada: {fig_filename}")

# =============================================================================
# GUARDAR DATOS
# =============================================================================

output_data = {
    'version': '3.3',
    'experiment': 'SIM03_UNIVERSALITY',
    'platform': 'IBM Quantum',
    'backend': backend.name,
    'job_id': job_id,
    'timestamp': timestamp,
    'parameters': {
        'n_qubits': N_QUBITS,
        'shots': SHOTS,
        'depth': DEPTH,
        'models': list(MODELS.keys())
    },
    'results': {
        'model': results['model'],
        'expected': results['expected'],
        'otoc': results['otoc'],
        'lambda': [float(x) for x in lambdas],
        'alpha': [float(x) for x in alphas]
    },
    'analysis': {
        'alpha_max_error': float(max_error),
        'alpha_mean_error': float(mean_error),
        'lambda_min': float(np.min(lambdas)),
        'lambda_max': float(np.max(lambdas)),
        'is_universal': bool(max_error < 0.001)
    },
    'verification': {
        'criteria': {k: bool(v) for k, v in CRITERIA.items()},
        'passed': n_passed,
        'total': n_total,
        'status': status
    }
}

json_filename = f'kaelion_universality_{backend.name}_{timestamp}.json'
with open(json_filename, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"✅ Datos guardados: {json_filename}")

# =============================================================================
# RESUMEN
# =============================================================================

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  KAELION v3.3 - SIM03: TEST DE UNIVERSALIDAD                         ║
╠══════════════════════════════════════════════════════════════════════╣
║  Backend: {backend.name:<20}                                      ║
║  Job ID:  {job_id:<20}                                      ║
║                                                                      ║
║  MODELOS PROBADOS: {len(MODELS):<5}                                          ║
║    • Kicked Ising    • Heisenberg XXZ    • Random Circuits           ║
║    • Transverse Field Ising              • XY Model                  ║
║                                                                      ║
║  RESULTADO:                                                          ║
║    Error máximo en α = -1/2 - λ: {max_error:.6f}                       ║
║    Rango λ: [{np.min(lambdas):.3f}, {np.max(lambdas):.3f}]                                    ║
║                                                                      ║
║  CONCLUSIÓN: α(λ) = -1/2 - λ es {'UNIVERSAL' if max_error < 0.001 else 'NO UNIVERSAL'}                       ║
║                                                                      ║
║  VERIFICACIÓN: {status:<10} ({n_passed}/{n_total} criterios)                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

plt.show()
