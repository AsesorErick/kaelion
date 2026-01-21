#!/usr/bin/env python3
"""
KAELION EXPERIMENTS v3.0: Test de Histéresis - Energía de Barrera
==================================================================

ACTUALIZACIÓN v3.0:
- V₀ = √3 ≈ 1.732 (derivado de Λ cosmológica)
- Barrera: ΔE = V₀/16 = √3/16 ≈ 0.108 (antes: 0.44/16 ≈ 0.028)
- La barrera es ~4× más alta que en v2.0

PREDICCIÓN P3: Si V(λ) = V₀λ²(1-λ)² es correcta, entonces:

    ΔE = V₀/16 = √3/16 ≈ 0.108 (energía de barrera en λ = 0.5)

La existencia de una barrera implica HISTÉRESIS en transiciones:
- Al aumentar "caoticidad": transición en λ_c⁺
- Al disminuir "caoticidad": transición en λ_c⁻
- Histéresis: Δλ = λ_c⁺ - λ_c⁻ ∝ √(ΔE/T)

PROTOCOLO:
1. Preparar sistema en régimen integrable (λ ≈ 0)
2. Aumentar gradualmente la caoticidad del Hamiltoniano
3. Medir λ durante el aumento (forward sweep)
4. Disminuir gradualmente la caoticidad
5. Medir λ durante la disminución (backward sweep)
6. Si hay histéresis → evidencia de barrera → evidencia de V(λ)

CAMBIOS DESDE v2.0:
- Barrera: 0.028 → 0.108 (×3.94)
- Histéresis esperada más pronunciada
- Transición de fase de primer orden más clara

Autor: Erick Francisco Pérez Eugenio
ORCID: 0009-0006-3228-4847
Fecha: Enero 2026
Proyecto: Kaelion v3.0 - Verificación de V(λ)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import json
from datetime import datetime

# ==============================================================================
# CONSTANTES KAELION v3.0
# ==============================================================================

PHI_0 = 1 / np.sqrt(3)      # ≈ 0.5774 - Escala de transición
V_0 = np.sqrt(3)            # ≈ 1.7321 - Escala del potencial
KAPPA = 1.0                 # Inercia canónica
BETA = V_0 + PHI_0          # ≈ 2.3094 - Clausura de ciclos
M_LAMBDA = np.sqrt(2 * V_0) # ≈ 1.8612 - Masa del campo λ
BARRIER = V_0 / 16          # ≈ 0.1083 - Altura de barrera

# Verificaciones
assert abs(V_0 * PHI_0 - 1.0) < 1e-10, "Invariante V₀×φ₀=1 violado"

print(f"KAELION v3.0 - Constantes de Histéresis:")
print(f"  V₀ = {V_0:.6f}")
print(f"  Barrera ΔE = V₀/16 = {BARRIER:.6f}")
print(f"  Comparación v2.0: ΔE = 0.44/16 = {0.44/16:.6f}")
print(f"  Ratio: {BARRIER/(0.44/16):.2f}× mayor")


# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

@dataclass
class HysteresisConfig:
    """Configuración para experimento de histéresis."""
    n_qubits: int = 4                      # Número de qubits
    chaos_steps: int = 30                  # Pasos en el barrido
    chaos_min: float = 0.0                 # Caoticidad mínima
    chaos_max: float = 1.0                 # Caoticidad máxima
    depth_per_step: int = 10               # Profundidad de circuito por paso
    shots: int = 8192                      # Mediciones por punto
    repetitions: int = 3                   # Repeticiones para estadística
    use_hardware: bool = False             # True para IBM Quantum real
    backend_name: str = "ibm_brisbane"     # Backend de IBM


# ==============================================================================
# PREDICCIÓN TEÓRICA (KAELION v3.0)
# ==============================================================================

def theoretical_hysteresis_width(temperature: float, V0: float = V_0) -> float:
    """
    Calcula el ancho de histéresis teórico.
    
    Para un potencial de doble pozo con barrera ΔE = V₀/16,
    la histéresis en presencia de ruido térmico es:
    
    Δλ ∝ √(ΔE/T) = √(V₀/(16T))
    
    KAELION v3.0: Con V₀ = √3, ΔE = √3/16 ≈ 0.108
    
    Args:
        temperature: Temperatura efectiva (ruido)
        V0: Altura del potencial (default: √3)
        
    Returns:
        Ancho de histéresis Δλ
    """
    delta_E = V0 / 16
    return np.sqrt(delta_E / temperature)


def barrier_height(V0: float = V_0) -> float:
    """
    Calcula altura de barrera en λ = 0.5.
    
    V(0.5) = V₀ · (0.5)² · (0.5)² = V₀/16
    
    KAELION v3.0: V(0.5) = √3/16 ≈ 0.108
    """
    return V0 / 16


def potential_V(lambda_val: float, V0: float = V_0) -> float:
    """Potencial V(λ) = V₀λ²(1-λ)²"""
    return V0 * lambda_val**2 * (1 - lambda_val)**2


def print_hysteresis_predictions():
    """Imprime predicciones de histéresis para v3.0."""
    print("\n" + "="*60)
    print("PREDICCIONES DE HISTÉRESIS - KAELION v3.0")
    print("="*60)
    print(f"Potencial: V(λ) = √3·λ²(1-λ)² = {V_0:.4f}·λ²(1-λ)²")
    print(f"Barrera: V(0.5) = V₀/16 = {BARRIER:.4f}")
    print("-"*60)
    print("COMPARACIÓN CON v2.0:")
    barrier_v2 = 0.44 / 16
    print(f"  Barrera v2.0 = {barrier_v2:.4f}")
    print(f"  Barrera v3.0 = {BARRIER:.4f}")
    print(f"  Ratio: {BARRIER/barrier_v2:.2f}× (barrera más alta)")
    print("-"*60)
    print("Implicaciones:")
    print("  - Histéresis más pronunciada")
    print("  - Transición de fase más clara")
    print("  - Mayor separación entre λ_c⁺ y λ_c⁻")
    print("="*60 + "\n")


# ==============================================================================
# CIRCUITOS CUÁNTICOS
# ==============================================================================

def create_variable_chaos_circuit_qiskit(config: HysteresisConfig, 
                                          chaos_parameter: float):
    """
    Crea circuito con caoticidad variable.
    
    chaos_parameter = 0: Completamente integrable (λ ≈ 0)
    chaos_parameter = 1: Máximamente caótico (λ ≈ 1)
    
    Interpolamos entre Hamiltoniano integrable y Kicked Ising.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        return None
    
    n = config.n_qubits
    qc = QuantumCircuit(n, n)
    
    # Parámetros que interpolan
    J = 0.9 * chaos_parameter          # Acoplamiento (0 = libre, 0.9 = fuerte)
    h = 0.5 * (1 - chaos_parameter)    # Campo transverso (fuerte = integrable)
    kick = np.pi/4 * chaos_parameter   # Kick (0 = nada, π/4 = fuerte)
    
    for layer in range(config.depth_per_step):
        # Términos de campo (parte integrable)
        for i in range(n):
            qc.rz(h, i)
            qc.rx(h * 0.5, i)
        
        # Términos de acoplamiento (parte caótica)
        if J > 0.01:
            for i in range(n - 1):
                qc.rzz(J, i, i + 1)
        
        # Kick periódico (induce caos)
        if kick > 0.01:
            for i in range(n):
                qc.rx(kick, i)
            qc.barrier()
    
    # Medición
    qc.measure(range(n), range(n))
    
    return qc


def extract_lambda_from_counts(counts: dict, n_qubits: int) -> float:
    """
    Extrae λ desde distribución de conteos.
    
    Método: Entropía de Shannon normalizada como proxy de λ.
    """
    total = sum(counts.values())
    probs = np.array(list(counts.values())) / total
    
    # Entropía de Shannon
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    # Normalizar
    S_max = n_qubits
    lambda_val = entropy / S_max
    
    return np.clip(lambda_val, 0, 1)


# ==============================================================================
# SIMULACIÓN ANALÍTICA
# ==============================================================================

def run_analytical_hysteresis(config: HysteresisConfig) -> Dict:
    """
    Simulación analítica de histéresis con modelo de Landau.
    
    Usa dinámica de Kramers para transiciones sobre barrera.
    """
    print("\n" + "="*60)
    print("SIMULACIÓN ANALÍTICA DE HISTÉRESIS - KAELION v3.0")
    print("="*60)
    
    # Parámetros de barrido
    chaos_values = np.linspace(config.chaos_min, config.chaos_max, config.chaos_steps)
    
    # Ruido efectivo (temperatura)
    T_eff = 0.05  # Temperatura efectiva baja para ver histéresis
    
    # Inicializar
    lambda_forward = np.zeros(config.chaos_steps)
    lambda_backward = np.zeros(config.chaos_steps)
    
    # Estado inicial
    lambda_current = 0.1  # Cerca del polo LQG
    
    print(f"Parámetros v3.0:")
    print(f"  V₀ = {V_0:.4f}")
    print(f"  Barrera = {BARRIER:.4f}")
    print(f"  T_eff = {T_eff:.4f}")
    print(f"  Histéresis esperada ∝ √(ΔE/T) = {np.sqrt(BARRIER/T_eff):.4f}")
    
    # Forward sweep (aumentar caoticidad)
    print("\nBarrido forward...")
    lambda_current = 0.1
    for i, chaos in enumerate(chaos_values):
        # El "chaos parameter" actúa como campo externo que favorece λ→1
        # Punto de equilibrio efectivo se desplaza con chaos
        lambda_eq = chaos  # Simplificación: equilibrio sigue a chaos
        
        # Dinámica con barrera
        # Probabilidad de transición depende de barrera efectiva
        barrier_eff = potential_V(0.5, V_0) * (1 - 2*abs(lambda_current - 0.5))
        rate = np.exp(-barrier_eff / T_eff)
        
        # Relajación hacia equilibrio con rate limitado por barrera
        d_lambda = 0.1 * (lambda_eq - lambda_current) * rate
        lambda_current += d_lambda + np.random.normal(0, 0.01)
        lambda_current = np.clip(lambda_current, 0, 1)
        
        lambda_forward[i] = lambda_current
    
    # Backward sweep (disminuir caoticidad)
    print("Barrido backward...")
    lambda_current = 0.9  # Cerca del polo holográfico
    for i, chaos in enumerate(reversed(chaos_values)):
        lambda_eq = chaos
        
        barrier_eff = potential_V(0.5, V_0) * (1 - 2*abs(lambda_current - 0.5))
        rate = np.exp(-barrier_eff / T_eff)
        
        d_lambda = 0.1 * (lambda_eq - lambda_current) * rate
        lambda_current += d_lambda + np.random.normal(0, 0.01)
        lambda_current = np.clip(lambda_current, 0, 1)
        
        lambda_backward[config.chaos_steps - 1 - i] = lambda_current
    
    # Calcular métricas de histéresis
    hysteresis_area = np.trapz(lambda_forward - lambda_backward, chaos_values)
    hysteresis_area_abs = np.abs(hysteresis_area)  # Usar valor absoluto
    max_diff = np.max(np.abs(lambda_forward - lambda_backward))
    
    # Detección con valor absoluto (el signo depende del orden de barrido)
    hysteresis_detected = hysteresis_area_abs > 0.02 or max_diff > 0.05
    
    print(f"\nResultados:")
    print(f"  Área de histéresis = {hysteresis_area:.4f} (|área| = {hysteresis_area_abs:.4f})")
    print(f"  Máxima diferencia = {max_diff:.4f}")
    print(f"  Histéresis detectada: {'✓ SÍ' if hysteresis_detected else '✗ NO'}")
    
    results = {
        "version": "3.0",
        "chaos_values": chaos_values.tolist(),
        "lambda_forward": lambda_forward.tolist(),
        "lambda_backward": lambda_backward.tolist(),
        "hysteresis_area": hysteresis_area,
        "hysteresis_area_abs": hysteresis_area_abs,
        "max_difference": max_diff,
        "barrier_height": BARRIER,
        "V0": V_0,
        "T_eff": T_eff,
        "hysteresis_detected": hysteresis_detected,
        "mode": "analytical_simulation",
        "timestamp": datetime.now().isoformat()
    }
    
    return results


# ==============================================================================
# EXPERIMENTO PRINCIPAL
# ==============================================================================

def run_hysteresis_experiment(config: HysteresisConfig) -> Dict:
    """
    Ejecuta experimento completo de histéresis.
    """
    
    print("=" * 60)
    print("EXPERIMENTO DE HISTÉRESIS - KAELION v3.0")
    print("=" * 60)
    print(f"Configuración:")
    print(f"  - Qubits: {config.n_qubits}")
    print(f"  - Pasos de barrido: {config.chaos_steps}")
    print(f"  - V₀ = {V_0:.4f}")
    print(f"  - Barrera = V₀/16 = {BARRIER:.4f}")
    print(f"  - Hardware: {'Real' if config.use_hardware else 'Simulador'}")
    print("=" * 60)
    
    # Verificar Qiskit
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        
        if config.use_hardware:
            from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
            service = QiskitRuntimeService()
            backend = service.backend(config.backend_name)
        else:
            backend = AerSimulator()
            
    except ImportError as e:
        print(f"Error importando Qiskit: {e}")
        print("Ejecutando simulación analítica...")
        return run_analytical_hysteresis(config)
    
    # Arrays para resultados
    chaos_values = np.linspace(config.chaos_min, config.chaos_max, config.chaos_steps)
    lambda_forward = np.zeros(config.chaos_steps)
    lambda_backward = np.zeros(config.chaos_steps)
    lambda_forward_err = np.zeros(config.chaos_steps)
    lambda_backward_err = np.zeros(config.chaos_steps)
    
    # Forward sweep
    print("\nBarrido FORWARD (aumentando caoticidad)...")
    for i, chaos in enumerate(chaos_values):
        samples = []
        for rep in range(config.repetitions):
            qc = create_variable_chaos_circuit_qiskit(config, chaos)
            qc_transpiled = transpile(qc, backend)
            
            if config.use_hardware:
                sampler = Sampler(backend)
                job = sampler.run([qc_transpiled], shots=config.shots)
                result = job.result()
                counts = result[0].data.meas.get_counts()
            else:
                job = backend.run(qc_transpiled, shots=config.shots)
                result = job.result()
                counts = result.get_counts()
            
            lambda_val = extract_lambda_from_counts(counts, config.n_qubits)
            samples.append(lambda_val)
        
        lambda_forward[i] = np.mean(samples)
        lambda_forward_err[i] = np.std(samples) / np.sqrt(len(samples))
        print(f"  chaos={chaos:.2f}: λ = {lambda_forward[i]:.3f} ± {lambda_forward_err[i]:.3f}")
    
    # Backward sweep
    print("\nBarrido BACKWARD (disminuyendo caoticidad)...")
    for i, chaos in enumerate(reversed(chaos_values)):
        samples = []
        for rep in range(config.repetitions):
            qc = create_variable_chaos_circuit_qiskit(config, chaos)
            qc_transpiled = transpile(qc, backend)
            
            if config.use_hardware:
                sampler = Sampler(backend)
                job = sampler.run([qc_transpiled], shots=config.shots)
                result = job.result()
                counts = result[0].data.meas.get_counts()
            else:
                job = backend.run(qc_transpiled, shots=config.shots)
                result = job.result()
                counts = result.get_counts()
            
            lambda_val = extract_lambda_from_counts(counts, config.n_qubits)
            samples.append(lambda_val)
        
        idx = config.chaos_steps - 1 - i
        lambda_backward[idx] = np.mean(samples)
        lambda_backward_err[idx] = np.std(samples) / np.sqrt(len(samples))
        print(f"  chaos={chaos:.2f}: λ = {lambda_backward[idx]:.3f} ± {lambda_backward_err[idx]:.3f}")
    
    # Calcular métricas
    hysteresis_area = np.trapz(np.abs(lambda_forward - lambda_backward), chaos_values)
    max_diff = np.max(np.abs(lambda_forward - lambda_backward))
    
    results = {
        "version": "3.0",
        "config": {
            "n_qubits": config.n_qubits,
            "chaos_steps": config.chaos_steps,
            "shots": config.shots
        },
        "chaos_values": chaos_values.tolist(),
        "lambda_forward": lambda_forward.tolist(),
        "lambda_backward": lambda_backward.tolist(),
        "lambda_forward_err": lambda_forward_err.tolist(),
        "lambda_backward_err": lambda_backward_err.tolist(),
        "hysteresis_area": hysteresis_area,
        "max_difference": max_diff,
        "barrier_height": BARRIER,
        "V0": V_0,
        "hysteresis_detected": hysteresis_area > 0.02 or max_diff > 0.05,
        "mode": "hardware" if config.use_hardware else "simulator",
        "timestamp": datetime.now().isoformat()
    }
    
    return results


# ==============================================================================
# VISUALIZACIÓN
# ==============================================================================

def plot_hysteresis_results(results: Dict, save_path: Optional[str] = None):
    """
    Genera gráficos de histéresis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    chaos = np.array(results["chaos_values"])
    lambda_fwd = np.array(results["lambda_forward"])
    lambda_bwd = np.array(results["lambda_backward"])
    
    # === Panel 1: Curvas de histéresis ===
    ax1 = axes[0, 0]
    ax1.plot(chaos, lambda_fwd, 'b-o', markersize=4, label='Forward (↑ caos)')
    ax1.plot(chaos, lambda_bwd, 'r-s', markersize=4, label='Backward (↓ caos)')
    ax1.fill_between(chaos, lambda_fwd, lambda_bwd, alpha=0.3, color='purple',
                     label=f'|Área| = {abs(results["hysteresis_area"]):.3f}')
    
    ax1.set_xlabel('Parámetro de caoticidad', fontsize=12)
    ax1.set_ylabel('λ', fontsize=12)
    ax1.set_title('Curvas de Histéresis', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === Panel 2: Potencial V(λ) v3.0 ===
    ax2 = axes[0, 1]
    lambda_range = np.linspace(0, 1, 100)
    V_lambda = V_0 * lambda_range**2 * (1 - lambda_range)**2
    
    ax2.plot(lambda_range, V_lambda, 'b-', linewidth=2)
    ax2.axhline(y=BARRIER, color='red', linestyle='--', 
                label=f'Barrera ΔE = V₀/16 = {BARRIER:.4f}')
    ax2.scatter([0.5], [BARRIER], color='red', s=100, zorder=5)
    
    # Comparar con v2.0
    V_lambda_v2 = 0.44 * lambda_range**2 * (1 - lambda_range)**2
    ax2.plot(lambda_range, V_lambda_v2, 'g--', linewidth=1, alpha=0.5,
             label=f'v2.0: V₀=0.44, ΔE={0.44/16:.4f}')
    
    ax2.set_xlabel('λ', fontsize=12)
    ax2.set_ylabel('V(λ)', fontsize=12)
    ax2.set_title(f'Potencial KAELION v3.0 (V₀={V_0:.3f})', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # === Panel 3: Diferencia ===
    ax3 = axes[1, 0]
    diff = lambda_fwd - lambda_bwd
    ax3.plot(chaos, diff, 'purple', linewidth=2)
    ax3.fill_between(chaos, 0, diff, alpha=0.3, color='purple')
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    ax3.set_xlabel('Parámetro de caoticidad', fontsize=12)
    ax3.set_ylabel('λ_forward - λ_backward', fontsize=12)
    ax3.set_title(f'Diferencia (máx = {results["max_difference"]:.3f})', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # === Panel 4: Resumen ===
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    KAELION v3.0 - Análisis de Histéresis
    =====================================
    
    Parámetros:
      V₀ = √3 = {V_0:.4f}
      Barrera ΔE = V₀/16 = {BARRIER:.4f}
    
    Resultados:
      |Área de histéresis| = {abs(results['hysteresis_area']):.4f}
      Máxima diferencia = {results['max_difference']:.4f}
    
    Comparación con v2.0:
      Barrera v2.0 = 0.44/16 = {0.44/16:.4f}
      Barrera v3.0 = {BARRIER:.4f}
      Ratio = {BARRIER/(0.44/16):.2f}×
    
    Conclusión:
    """
    
    if results['hysteresis_detected']:
        summary_text += """  ✓ HISTÉRESIS DETECTADA
      → Evidencia de barrera
      → Compatible con V(λ) = √3·λ²(1-λ)²"""
    else:
        summary_text += """  ✗ HISTÉRESIS NO DETECTADA
      → Barrera no observable a esta temperatura
      → Aumentar resolución o reducir ruido"""
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes,
             fontsize=11, family='monospace', verticalalignment='top')
    
    plt.suptitle('KAELION v3.0: Test de Predicción P3 - Histéresis\n'
                 f'V(λ) = √3·λ²(1-λ)², Barrera = {BARRIER:.4f}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada en: {save_path}")
    
    plt.show()
    
    return fig


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Función principal."""
    
    print("\n" + "=" * 70)
    print("KAELION v3.0: TEST DE PREDICCIÓN P3 - HISTÉRESIS")
    print("=" * 70)
    
    print_hysteresis_predictions()
    
    # Configuración
    config = HysteresisConfig(
        n_qubits=4,
        chaos_steps=30,
        chaos_min=0.0,
        chaos_max=1.0,
        depth_per_step=10,
        shots=8192,
        repetitions=3,
        use_hardware=False
    )
    
    # Ejecutar
    results = run_hysteresis_experiment(config)
    
    # Visualizar
    plot_hysteresis_results(results, save_path="hysteresis_test_P3_v3.png")
    
    # Guardar
    with open("hysteresis_results_P3_v3.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResultados guardados en: hysteresis_results_P3_v3.json")
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN FINAL - KAELION v3.0")
    print("=" * 70)
    
    if results['hysteresis_detected']:
        print("✓ HISTÉRESIS DETECTADA")
        print(f"  |Área| = {abs(results['hysteresis_area']):.4f}")
        print(f"  Δλ_max = {results['max_difference']:.4f}")
        print(f"  Barrera V₀/16 = {BARRIER:.4f}")
        print("  → V(λ) = √3·λ²(1-λ)² es compatible con los datos")
    else:
        print("✗ HISTÉRESIS NO DETECTADA CLARAMENTE")
        print("  → Aumentar sensibilidad o reducir temperatura efectiva")
    
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()
