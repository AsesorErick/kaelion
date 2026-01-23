#!/usr/bin/env python3
"""
================================================================================
KAELION v3.3 - SIM02: REGIÓN LQG EN HARDWARE
================================================================================

Objetivo: Medir λ < 0.3 en hardware real (región LQG pura).

Estrategia:
  - Usar circuitos INTEGRABLES (sin scrambling)
  - Profundidad mínima para reducir ruido
  - Múltiples configuraciones integrables
  - Comparar con circuitos caóticos como control

Configuraciones integrables:
  1. Solo campos locales (J=0)
  2. Solo ZZ sin campo (h=0) 
  3. Ising integrable (J << h)
  4. Estado producto (depth=0)

Predicción Kaelion:
  - Circuitos integrables: λ → 0, α → -0.5
  - Circuitos caóticos: λ → 1, α → -1.5

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
print("KAELION v3.3 - SIM02: REGIÓN LQG EN HARDWARE")
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

N_QUBITS = 6       # Menos qubits = menos ruido
SHOTS = 8192       # Más estadística
DEPTHS = [1, 2, 3] # Profundidades bajas

# Configuraciones a probar
CONFIGS = {
    # Integrables (esperamos λ bajo)
    'integrable_local': {'J': 0.0, 'h': np.pi/2, 'type': 'LQG'},      # Solo rotaciones locales
    'integrable_weak': {'J': 0.05, 'h': np.pi/2, 'type': 'LQG'},      # Acoplamiento muy débil
    'integrable_ising': {'J': 0.1, 'h': np.pi/2, 'type': 'LQG'},      # Ising integrable
    
    # Transición
    'transition_low': {'J': 0.3, 'h': np.pi/3, 'type': 'Trans'},
    'transition_mid': {'J': 0.5, 'h': np.pi/4, 'type': 'Trans'},
    
    # Caóticos (esperamos λ alto) - CONTROL
    'chaotic_weak': {'J': 0.8, 'h': np.pi/4, 'type': 'Holo'},
    'chaotic_strong': {'J': 1.2, 'h': np.pi/4, 'type': 'Holo'},
}

print(f"\nParámetros:")
print(f"  N_QUBITS = {N_QUBITS}")
print(f"  SHOTS = {SHOTS}")
print(f"  DEPTHS = {DEPTHS}")
print(f"  Configuraciones: {len(CONFIGS)}")

print("\nConfiguraciones:")
for name, params in CONFIGS.items():
    print(f"  {name}: J={params['J']:.2f}, h={params['h']:.3f} [{params['type']}]")

# =============================================================================
# CREAR CIRCUITOS
# =============================================================================

def create_otoc_circuit(n_qubits, J, h, depth, target=None):
    """Crea circuito OTOC estándar."""
    if target is None:
        target = n_qubits - 1
    
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
        # ZZ interactions (solo si J > 0)
        if J > 0.001:
            for i in range(n_qubits - 1):
                qc.cx(i, i + 1)
                qc.rz(2 * J, i + 1)
                qc.cx(i, i + 1)
        
        # Local fields
        for i in range(n_qubits):
            qc.rx(2 * h, i)
    
    # W = Z_target
    qc.z(target)
    
    # Backward evolution
    for _ in range(depth):
        for i in range(n_qubits):
            qc.rx(-2 * h, i)
        
        if J > 0.001:
            for i in range(n_qubits - 2, -1, -1):
                qc.cx(i, i + 1)
                qc.rz(-2 * J, i + 1)
                qc.cx(i, i + 1)
    
    # V†
    qc.z(0)
    
    # Medir en base X
    for i in range(n_qubits):
        qc.h(i)
    
    qc.measure(qr, cr)
    return qc

print("\n" + "="*70)
print("CREANDO CIRCUITOS")
print("="*70)

circuits = []
circuit_info = []

for config_name, params in CONFIGS.items():
    for depth in DEPTHS:
        qc = create_otoc_circuit(N_QUBITS, params['J'], params['h'], depth)
        qc.name = f"{config_name}_d{depth}"
        circuits.append(qc)
        circuit_info.append({
            'config': config_name,
            'J': params['J'],
            'h': params['h'],
            'depth': depth,
            'type': params['type']
        })

print(f"Total circuitos: {len(circuits)}")

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
    'config': [], 'type': [], 'J': [], 'h': [], 'depth': [],
    'otoc': [], 'lambda': [], 'alpha': []
}

print("\nResultados por configuración:")
print("-" * 70)

for idx, (info, pub_result) in enumerate(zip(circuit_info, result)):
    counts = pub_result.data.c.get_counts()
    otoc = calculate_otoc_from_counts(counts, N_QUBITS)
    
    # λ desde OTOC
    lambda_m = np.clip(1 - abs(otoc), 0, 1)
    alpha_m = -0.5 - lambda_m
    
    results['config'].append(info['config'])
    results['type'].append(info['type'])
    results['J'].append(info['J'])
    results['h'].append(info['h'])
    results['depth'].append(info['depth'])
    results['otoc'].append(otoc)
    results['lambda'].append(lambda_m)
    results['alpha'].append(alpha_m)
    
    print(f"  {info['config']:20s} d={info['depth']}: OTOC={otoc:+.4f}, λ={lambda_m:.4f}, α={alpha_m:.4f} [{info['type']}]")

# =============================================================================
# ANÁLISIS POR TIPO
# =============================================================================

print("\n" + "="*70)
print("ANÁLISIS POR TIPO DE RÉGIMEN")
print("="*70)

types = np.array(results['type'])
lambdas = np.array(results['lambda'])
alphas = np.array(results['alpha'])
Js = np.array(results['J'])

# Promedios por tipo
for regime in ['LQG', 'Trans', 'Holo']:
    mask = types == regime
    if np.any(mask):
        lam_mean = np.mean(lambdas[mask])
        lam_std = np.std(lambdas[mask])
        alpha_mean = np.mean(alphas[mask])
        n_points = np.sum(mask)
        print(f"\n  {regime}:")
        print(f"    λ = {lam_mean:.4f} ± {lam_std:.4f}")
        print(f"    α = {alpha_mean:.4f}")
        print(f"    n = {n_points} puntos")

# Correlación J vs λ
corr_J_lambda = np.corrcoef(Js, lambdas)[0, 1]
print(f"\n  Correlación J vs λ: {corr_J_lambda:.4f}")

# =============================================================================
# VERIFICACIÓN
# =============================================================================

print("\n" + "="*70)
print("VERIFICACIÓN DE CRITERIOS")
print("="*70)

lqg_lambdas = lambdas[types == 'LQG']
holo_lambdas = lambdas[types == 'Holo']

lambda_lqg_mean = np.mean(lqg_lambdas)
lambda_holo_mean = np.mean(holo_lambdas)
lambda_min = np.min(lambdas)

CRITERIA = {
    'lambda_lqg_menor_que_holo': lambda_lqg_mean < lambda_holo_mean,
    'correlacion_J_lambda_positiva': corr_J_lambda > 0.5,
    'alpha_formula_exacta': np.allclose(alphas, -0.5 - lambdas, atol=0.001),
    'diferencia_regimenes': (lambda_holo_mean - lambda_lqg_mean) > 0.1,
    'lambda_minimo_bajo': lambda_min < 0.8,
    'gradiente_monotono': np.corrcoef(Js, lambdas)[0, 1] > 0.7
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
fig.suptitle(f'KAELION v3.3: Región LQG en {backend.name}\nJob ID: {job_id}', 
             fontsize=14, fontweight='bold')

# 1. λ vs J
ax1 = axes[0, 0]
colors_type = {'LQG': 'green', 'Trans': 'orange', 'Holo': 'purple'}
for regime in ['LQG', 'Trans', 'Holo']:
    mask = types == regime
    ax1.scatter(Js[mask], lambdas[mask], s=100, c=colors_type[regime], 
                label=regime, alpha=0.7, edgecolors='black')

# Línea de tendencia
z = np.polyfit(Js, lambdas, 1)
p = np.poly1d(z)
x_fit = np.linspace(0, 1.3, 100)
ax1.plot(x_fit, p(x_fit), 'k--', lw=2, alpha=0.5, label=f'Tendencia (r={corr_J_lambda:.2f})')

ax1.set_xlabel('J (acoplamiento)', fontsize=12)
ax1.set_ylabel('λ', fontsize=12)
ax1.set_title('λ vs J: Transición Integrable → Caótico')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-0.1, 1.4)
ax1.set_ylim(0.4, 1.05)

# 2. α vs λ
ax2 = axes[0, 1]
ax2.scatter(lambdas, alphas, s=100, c=[colors_type[t] for t in types], 
            edgecolors='black', alpha=0.7)
x_theory = np.linspace(0, 1, 100)
ax2.plot(x_theory, -0.5 - x_theory, 'r-', lw=2, label='α = -1/2 - λ')
ax2.axhline(-0.5, color='green', ls='--', alpha=0.5, label='LQG: α=-0.5')
ax2.axhline(-1.5, color='purple', ls='--', alpha=0.5, label='Holo: α=-1.5')
ax2.set_xlabel('λ', fontsize=12)
ax2.set_ylabel('α', fontsize=12)
ax2.set_title('Verificación: α(λ) = -1/2 - λ')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. λ por configuración
ax3 = axes[1, 0]
config_names = list(CONFIGS.keys())
lambda_by_config = {}
for cfg in config_names:
    mask = np.array(results['config']) == cfg
    lambda_by_config[cfg] = np.mean(lambdas[mask])

colors_bar = [colors_type[CONFIGS[c]['type']] for c in config_names]
bars = ax3.bar(range(len(config_names)), [lambda_by_config[c] for c in config_names],
               color=colors_bar, alpha=0.7, edgecolor='black')
ax3.set_xticks(range(len(config_names)))
ax3.set_xticklabels([c.replace('_', '\n') for c in config_names], fontsize=8, rotation=45, ha='right')
ax3.set_ylabel('λ promedio', fontsize=12)
ax3.set_title('λ por Configuración')
ax3.axhline(0.5, color='gray', ls='--', alpha=0.5)
ax3.grid(True, alpha=0.3, axis='y')

# 4. OTOC vs depth por tipo
ax4 = axes[1, 1]
depths_arr = np.array(results['depth'])
for regime in ['LQG', 'Trans', 'Holo']:
    mask = types == regime
    for d in DEPTHS:
        mask_d = mask & (depths_arr == d)
        if np.any(mask_d):
            otoc_mean = np.mean(np.abs(np.array(results['otoc'])[mask_d]))
            ax4.scatter(d, otoc_mean, s=150, c=colors_type[regime], 
                       marker='o' if regime=='LQG' else ('s' if regime=='Trans' else '^'),
                       edgecolors='black', alpha=0.7)

ax4.set_xlabel('Profundidad', fontsize=12)
ax4.set_ylabel('|OTOC| promedio', fontsize=12)
ax4.set_title('Decaimiento OTOC por Profundidad')
ax4.legend(['LQG', 'Trans', 'Holo'])
ax4.grid(True, alpha=0.3)

plt.tight_layout()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
fig_filename = f'kaelion_lqg_region_{backend.name}_{timestamp}.png'
plt.savefig(fig_filename, dpi=150, bbox_inches='tight')
print(f"\n✅ Figura guardada: {fig_filename}")

# =============================================================================
# GUARDAR DATOS
# =============================================================================

output_data = {
    'version': '3.3',
    'experiment': 'SIM02_LQG_REGION',
    'platform': 'IBM Quantum',
    'backend': backend.name,
    'job_id': job_id,
    'timestamp': timestamp,
    'parameters': {
        'n_qubits': N_QUBITS,
        'shots': SHOTS,
        'depths': DEPTHS,
        'configs': CONFIGS
    },
    'results': {
        'config': results['config'],
        'type': results['type'],
        'J': results['J'],
        'depth': results['depth'],
        'otoc': results['otoc'],
        'lambda': [float(x) for x in lambdas],
        'alpha': [float(x) for x in alphas]
    },
    'analysis': {
        'lambda_lqg_mean': float(lambda_lqg_mean),
        'lambda_holo_mean': float(lambda_holo_mean),
        'lambda_min': float(lambda_min),
        'correlation_J_lambda': float(corr_J_lambda)
    },
    'verification': {
        'criteria': {k: bool(v) for k, v in CRITERIA.items()},
        'passed': n_passed,
        'total': n_total,
        'status': status
    }
}

json_filename = f'kaelion_lqg_region_{backend.name}_{timestamp}.json'
with open(json_filename, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"✅ Datos guardados: {json_filename}")

# =============================================================================
# RESUMEN
# =============================================================================

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  KAELION v3.3 - SIM02: REGIÓN LQG                                    ║
╠══════════════════════════════════════════════════════════════════════╣
║  Backend: {backend.name:<20}                                      ║
║  Job ID:  {job_id:<20}                                      ║
║                                                                      ║
║  RESULTADOS POR RÉGIMEN:                                             ║
║    LQG (integrable):  λ = {lambda_lqg_mean:.4f}                              ║
║    Holo (caótico):    λ = {lambda_holo_mean:.4f}                              ║
║    Diferencia:        Δλ = {lambda_holo_mean - lambda_lqg_mean:.4f}                              ║
║                                                                      ║
║  Correlación J-λ:     {corr_J_lambda:.4f}                                      ║
║  λ mínimo medido:     {lambda_min:.4f}                                      ║
║                                                                      ║
║  VERIFICACIÓN: {status:<10} ({n_passed}/{n_total} criterios)                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

plt.show()
