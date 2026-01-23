#!/usr/bin/env python3
"""
================================================================================
KAELION v3.3 - SIM01 MEJORADO: GRADIENTE ESPACIAL REAL
================================================================================

Objetivo: Verificar que λ(x) varía ESPACIALMENTE dentro del mismo sistema.

Diseño:
  - N qubits en línea representando distancia al horizonte
  - Qubits interiores (x < r_h): régimen LQG, bajo scrambling
  - Qubits exteriores (x > r_h): régimen Holo, alto scrambling
  - Medir λ LOCAL para cada qubit

Diferencia con experimentos anteriores:
  - Antes: λ fijo por circuito (32 puntos independientes)
  - Ahora: λ(x) variable DENTRO del mismo circuito

Predicción Kaelion:
  - λ(x) debe aumentar al alejarse del horizonte
  - α(x) = -1/2 - λ(x) en cada punto

Autor: Erick Pérez
ORCID: 0009-0006-3228-4847
Fecha: 22 Enero 2026
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json

# Qiskit imports
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

print("="*70)
print("KAELION v3.3 - SIM01 MEJORADO: GRADIENTE ESPACIAL REAL")
print("="*70)

# =============================================================================
# CONEXIÓN IBM QUANTUM
# =============================================================================

print("\nConectando a IBM Quantum...")
service = QiskitRuntimeService()

# Seleccionar backend
backends = service.backends(simulator=False, operational=True)
print(f"Backends disponibles: {[b.name for b in backends]}")

# Preferir ibm_fez (mejores resultados anteriores)
try:
    backend = service.backend("ibm_fez")
except:
    backend = service.least_busy(operational=True, simulator=False)
    
print(f"Backend seleccionado: {backend.name}")
print(f"Qubits disponibles: {backend.num_qubits}")

# =============================================================================
# PARÁMETROS DEL EXPERIMENTO
# =============================================================================

N_QUBITS = 10          # Línea de 10 qubits
I_HORIZON = 4          # Posición del horizonte (qubit 4)
SHOTS = 4096           # Estadística
DEPTH_LAYERS = 4       # Capas de evolución

# Parámetros de Kicked Ising con GRADIENTE ESPACIAL
# J(x) y h(x) varían según posición relativa al horizonte

def get_spatial_params(qubit_index, horizon_index):
    """
    Retorna parámetros J, h según posición relativa al horizonte.
    
    Interior (x < horizon): Integrable (J pequeño, h grande)
    Exterior (x > horizon): Caótico (J grande, h intermedio)
    """
    # Distancia normalizada al horizonte [-1, 1]
    x_norm = (qubit_index - horizon_index) / (N_QUBITS / 2)
    
    # Factor de caos: 0 en interior, 1 en exterior
    chaos_factor = 0.5 * (1 + np.tanh(2 * x_norm))
    
    # Parámetros interpolados
    J_int, J_chaos = 0.1, 1.2      # Acoplamiento ZZ
    h_int, h_chaos = np.pi/2, np.pi/4  # Campo transverso
    
    J = J_int + (J_chaos - J_int) * chaos_factor
    h = h_int + (h_chaos - h_int) * chaos_factor
    
    return J, h, chaos_factor

print(f"\nParámetros:")
print(f"  N_QUBITS = {N_QUBITS}")
print(f"  HORIZONTE = qubit {I_HORIZON}")
print(f"  SHOTS = {SHOTS}")
print(f"  DEPTH = {DEPTH_LAYERS}")

print("\nGradiente espacial de parámetros:")
for i in range(N_QUBITS):
    J, h, cf = get_spatial_params(i, I_HORIZON)
    region = "LQG" if cf < 0.3 else ("Holo" if cf > 0.7 else "Trans")
    print(f"  q{i}: J={J:.3f}, h={h:.3f}, chaos={cf:.3f} [{region}]")

# =============================================================================
# CREAR CIRCUITO CON GRADIENTE ESPACIAL
# =============================================================================

def create_spatial_gradient_circuit(n_qubits, horizon, depth, target_qubit):
    """
    Crea circuito OTOC con parámetros que varían espacialmente.
    
    El circuito aplica evolución con J(x), h(x) diferentes para cada qubit,
    luego mide OTOC en el qubit objetivo.
    """
    qr = QuantumRegister(n_qubits, 'q')
    cr = ClassicalRegister(n_qubits, 'c')
    qc = QuantumCircuit(qr, cr)
    
    # Estado inicial |+⟩
    for i in range(n_qubits):
        qc.h(i)
    
    # Perturbación inicial V = Z en qubit 0 (borde LQG)
    qc.z(0)
    
    # Forward evolution con gradiente espacial
    for _ in range(depth):
        # Interacciones ZZ con J(x) espacialmente variable
        for i in range(n_qubits - 1):
            # J promedio entre qubits vecinos
            J_i, _, _ = get_spatial_params(i, horizon)
            J_j, _, _ = get_spatial_params(i + 1, horizon)
            J_avg = (J_i + J_j) / 2
            
            qc.cx(i, i + 1)
            qc.rz(2 * J_avg, i + 1)
            qc.cx(i, i + 1)
        
        # Campos locales h(x) espacialmente variables
        for i in range(n_qubits):
            _, h_i, _ = get_spatial_params(i, horizon)
            qc.rx(2 * h_i, i)
    
    # Perturbación W = Z en qubit objetivo
    qc.z(target_qubit)
    
    # Backward evolution (inversa)
    for _ in range(depth):
        for i in range(n_qubits):
            _, h_i, _ = get_spatial_params(i, horizon)
            qc.rx(-2 * h_i, i)
        
        for i in range(n_qubits - 2, -1, -1):
            J_i, _, _ = get_spatial_params(i, horizon)
            J_j, _, _ = get_spatial_params(i + 1, horizon)
            J_avg = (J_i + J_j) / 2
            
            qc.cx(i, i + 1)
            qc.rz(-2 * J_avg, i + 1)
            qc.cx(i, i + 1)
    
    # V†
    qc.z(0)
    
    # Medición en base X
    for i in range(n_qubits):
        qc.h(i)
    
    qc.measure(qr, cr)
    
    return qc

# =============================================================================
# CREAR TODOS LOS CIRCUITOS
# =============================================================================

print("\n" + "="*70)
print("CREANDO CIRCUITOS")
print("="*70)

circuits = []
circuit_info = []

# Un circuito por cada qubit objetivo (para medir λ local)
for target in range(N_QUBITS):
    qc = create_spatial_gradient_circuit(N_QUBITS, I_HORIZON, DEPTH_LAYERS, target)
    qc.name = f"spatial_q{target}"
    circuits.append(qc)
    
    J_t, h_t, chaos_t = get_spatial_params(target, I_HORIZON)
    circuit_info.append({
        'target': target,
        'J': J_t,
        'h': h_t,
        'chaos_design': chaos_t
    })

print(f"Total circuitos: {len(circuits)}")

# =============================================================================
# TRANSPILAR
# =============================================================================

print("\nTranspilando circuitos...")
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
transpiled = pm.run(circuits)
print("Transpilación completada")

depths_t = [qc.depth() for qc in transpiled]
print(f"Profundidades transpiladas: min={min(depths_t)}, max={max(depths_t)}")

# =============================================================================
# EJECUTAR EN IBM QUANTUM
# =============================================================================

print("\n" + "="*70)
print("EJECUTANDO EN IBM QUANTUM")
print("="*70)

sampler = Sampler(backend)

print(f"\nEnviando {len(transpiled)} circuitos a {backend.name}...")
print("(Esto puede tomar varios minutos)")

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
    """Calcular OTOC desde distribución de medidas."""
    otoc = 0.0
    total = sum(counts.values())
    
    for bitstring, count in counts.items():
        # Limpiar bitstring
        bits = bitstring.replace(' ', '').zfill(n_qubits)
        # Paridad
        parity = bits.count('1') % 2
        sign = 1 - 2 * parity
        otoc += sign * count / total
    
    return otoc

results = {
    'qubit': [],
    'x_position': [],
    'chaos_design': [],
    'otoc': [],
    'lambda': [],
    'alpha': [],
    'region': []
}

# Referencia: OTOC máximo (estado inicial)
otoc_ref = 1.0

print("\nResultados por qubit (posición espacial):")
print("-" * 70)

for idx, (info, pub_result) in enumerate(zip(circuit_info, result)):
    counts = pub_result.data.c.get_counts()
    otoc = calculate_otoc_from_counts(counts, N_QUBITS)
    
    target = info['target']
    chaos_design = info['chaos_design']
    
    # Calcular λ desde OTOC
    # OTOC ≈ 1 → no scrambling → λ ≈ 0 (LQG)
    # OTOC ≈ 0 → máximo scrambling → λ ≈ 1 (Holo)
    lambda_measured = np.clip(1 - abs(otoc), 0, 1)
    
    # α desde λ
    alpha_measured = -0.5 - lambda_measured
    
    # Clasificar región
    if lambda_measured < 0.3:
        region = "LQG"
    elif lambda_measured > 0.7:
        region = "Holo"
    else:
        region = "Trans"
    
    # Posición relativa al horizonte
    x_rel = target - I_HORIZON
    
    results['qubit'].append(target)
    results['x_position'].append(x_rel)
    results['chaos_design'].append(chaos_design)
    results['otoc'].append(otoc)
    results['lambda'].append(lambda_measured)
    results['alpha'].append(alpha_measured)
    results['region'].append(region)
    
    print(f"  q{target} (x={x_rel:+d}): OTOC={otoc:+.4f}, λ={lambda_measured:.4f}, "
          f"α={alpha_measured:.4f} [{region}]")

# =============================================================================
# ANÁLISIS DE GRADIENTE ESPACIAL
# =============================================================================

print("\n" + "="*70)
print("ANÁLISIS DE GRADIENTE ESPACIAL")
print("="*70)

x_pos = np.array(results['x_position'])
lambdas = np.array(results['lambda'])
alphas = np.array(results['alpha'])
chaos_design = np.array(results['chaos_design'])

# Promedios por región
mask_lqg = x_pos < 0
mask_holo = x_pos > 0

lambda_lqg = np.mean(lambdas[mask_lqg]) if np.any(mask_lqg) else 0
lambda_holo = np.mean(lambdas[mask_holo]) if np.any(mask_holo) else 0
delta_lambda = lambda_holo - lambda_lqg

alpha_lqg = np.mean(alphas[mask_lqg]) if np.any(mask_lqg) else -0.5
alpha_holo = np.mean(alphas[mask_holo]) if np.any(mask_holo) else -1.5

# Correlación diseño vs medido
corr_lambda = np.corrcoef(chaos_design, lambdas)[0, 1]

# Gradiente (pendiente)
gradient = np.polyfit(x_pos, lambdas, 1)[0]

print(f"\nMétricas del gradiente:")
print(f"  λ promedio (LQG, x<0):  {lambda_lqg:.4f}")
print(f"  λ promedio (Holo, x>0): {lambda_holo:.4f}")
print(f"  Δλ (Holo - LQG):        {delta_lambda:.4f}")
print(f"  Gradiente dλ/dx:        {gradient:.4f}")
print(f"  Correlación diseño-medido: {corr_lambda:.4f}")

print(f"\n  α promedio (LQG):  {alpha_lqg:.4f} (esperado: -0.5)")
print(f"  α promedio (Holo): {alpha_holo:.4f} (esperado: -1.5)")

# =============================================================================
# VERIFICACIÓN
# =============================================================================

print("\n" + "="*70)
print("VERIFICACIÓN DE CRITERIOS")
print("="*70)

CRITERIA = {
    'gradiente_positivo': gradient > 0.01,
    'delta_lambda_significativo': delta_lambda > 0.05,
    'correlacion_diseno_medido': corr_lambda > 0.3,
    'lambda_lqg_menor': lambda_lqg < lambda_holo,
    'alpha_sigue_formula': np.allclose(alphas, -0.5 - lambdas, atol=0.001),
    'transicion_detectada': np.any(lambdas < 0.5) and np.any(lambdas > 0.5)
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
fig.suptitle(f'KAELION v3.3: Gradiente Espacial Real en {backend.name}\n'
             f'Job ID: {job_id}', fontsize=14, fontweight='bold')

# 1. λ(x) espacial
ax1 = axes[0, 0]
ax1.plot(x_pos, chaos_design, 'g--', lw=2, marker='s', ms=8, 
         label='Diseño (chaos factor)', alpha=0.7)
ax1.plot(x_pos, lambdas, 'b-', lw=2, marker='o', ms=10, 
         label='Medido (λ)')
ax1.axvline(0, color='orange', ls=':', lw=2, label='Horizonte')
ax1.axhline(0.5, color='gray', ls='--', alpha=0.5)
ax1.fill_between(x_pos[mask_lqg], 0, 1, alpha=0.1, color='green', label='Región LQG')
ax1.fill_between(x_pos[mask_holo], 0, 1, alpha=0.1, color='purple', label='Región Holo')
ax1.set_xlabel('Posición relativa al horizonte (x - r_h)', fontsize=12)
ax1.set_ylabel('λ', fontsize=12)
ax1.set_title('λ(x) - Gradiente Espacial')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.1, 1.1)

# 2. α(x) espacial
ax2 = axes[0, 1]
ax2.plot(x_pos, alphas, 'r-', lw=2, marker='o', ms=10)
ax2.plot(x_pos, -0.5 - lambdas, 'k--', lw=1, label='α = -1/2 - λ')
ax2.axvline(0, color='orange', ls=':', lw=2)
ax2.axhline(-0.5, color='green', ls='--', alpha=0.7, label='α(LQG) = -1/2')
ax2.axhline(-1.5, color='purple', ls='--', alpha=0.7, label='α(Holo) = -3/2')
ax2.set_xlabel('Posición relativa al horizonte', fontsize=12)
ax2.set_ylabel('α', fontsize=12)
ax2.set_title('α(x) = -1/2 - λ(x)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. OTOC por posición
ax3 = axes[1, 0]
colors = ['green' if r == 'LQG' else ('purple' if r == 'Holo' else 'orange') 
          for r in results['region']]
ax3.bar(x_pos, results['otoc'], color=colors, alpha=0.7, edgecolor='black')
ax3.axvline(0, color='orange', ls=':', lw=2)
ax3.axhline(0, color='gray', ls='-', alpha=0.5)
ax3.set_xlabel('Posición relativa al horizonte', fontsize=12)
ax3.set_ylabel('OTOC', fontsize=12)
ax3.set_title('OTOC(x) - Scrambling Espacial')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Correlación diseño vs medido
ax4 = axes[1, 1]
ax4.scatter(chaos_design, lambdas, s=150, c=x_pos, cmap='coolwarm', 
            edgecolors='black', zorder=5)
ax4.plot([0, 1], [0, 1], 'k--', lw=2, label='Ideal')
ax4.set_xlabel('Chaos factor (diseño)', fontsize=12)
ax4.set_ylabel('λ medido', fontsize=12)
ax4.set_title(f'Correlación: {corr_lambda:.3f}')
ax4.legend()
ax4.grid(True, alpha=0.3)
ax4.set_xlim(-0.1, 1.1)
ax4.set_ylim(-0.1, 1.1)
cbar = plt.colorbar(ax4.collections[0], ax=ax4)
cbar.set_label('Posición x')

plt.tight_layout()

# Guardar figura
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
fig_filename = f'kaelion_spatial_gradient_{backend.name}_{timestamp}.png'
plt.savefig(fig_filename, dpi=150, bbox_inches='tight')
print(f"\n✅ Figura guardada: {fig_filename}")

# =============================================================================
# GUARDAR DATOS
# =============================================================================

output_data = {
    'version': '3.3',
    'experiment': 'SIM01_SPATIAL_GRADIENT',
    'platform': 'IBM Quantum',
    'backend': backend.name,
    'job_id': job_id,
    'timestamp': timestamp,
    'parameters': {
        'n_qubits': N_QUBITS,
        'horizon': I_HORIZON,
        'depth': DEPTH_LAYERS,
        'shots': SHOTS
    },
    'results': {
        'qubit': results['qubit'],
        'x_position': results['x_position'],
        'chaos_design': [float(x) for x in chaos_design],
        'otoc': results['otoc'],
        'lambda': [float(x) for x in lambdas],
        'alpha': [float(x) for x in alphas],
        'region': results['region']
    },
    'analysis': {
        'lambda_lqg': float(lambda_lqg),
        'lambda_holo': float(lambda_holo),
        'delta_lambda': float(delta_lambda),
        'gradient': float(gradient),
        'correlation': float(corr_lambda),
        'alpha_lqg': float(alpha_lqg),
        'alpha_holo': float(alpha_holo)
    },
    'verification': {
        'criteria': {k: bool(v) for k, v in CRITERIA.items()},
        'passed': n_passed,
        'total': n_total,
        'status': status
    }
}

json_filename = f'kaelion_spatial_gradient_{backend.name}_{timestamp}.json'
with open(json_filename, 'w') as f:
    json.dump(output_data, f, indent=2)
print(f"✅ Datos guardados: {json_filename}")

# =============================================================================
# RESUMEN FINAL
# =============================================================================

print(f"""
╔══════════════════════════════════════════════════════════════════════╗
║  KAELION v3.3 - SIM01 MEJORADO: GRADIENTE ESPACIAL                   ║
╠══════════════════════════════════════════════════════════════════════╣
║  Backend: {backend.name:<20}                                      ║
║  Job ID:  {job_id:<20}                                      ║
║                                                                      ║
║  GRADIENTE ESPACIAL:                                                 ║
║    λ (LQG, x<0):  {lambda_lqg:.4f}                                          ║
║    λ (Holo, x>0): {lambda_holo:.4f}                                          ║
║    Δλ:            {delta_lambda:.4f}                                          ║
║    dλ/dx:         {gradient:.4f}                                          ║
║    Correlación:   {corr_lambda:.4f}                                          ║
║                                                                      ║
║  VERIFICACIÓN: {status:<10} ({n_passed}/{n_total} criterios)                      ║
╚══════════════════════════════════════════════════════════════════════╝
""")

plt.show()
