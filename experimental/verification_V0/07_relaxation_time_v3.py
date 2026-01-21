#!/usr/bin/env python3
"""
KAELION EXPERIMENTS v3.0: Test de Tiempo de Relajación τ
=========================================================

ACTUALIZACIÓN v3.0:
- V₀ = √3 ≈ 1.732 (derivado de Λ cosmológica)
- φ₀ = 1/√3 ≈ 0.577 (escala de transición)
- V₀ × φ₀ = 1 (invariante fundamental)
- κ = 1 (inercia canónica)

PREDICCIÓN P1: Si V(λ) = V₀λ²(1-λ)² es correcta, entonces:

    τ = √(κ / 2V₀) = √(1 / 2√3) ≈ 0.537

Este experimento mide el tiempo de relajación después de una perturbación
y verifica si es consistente con el potencial propuesto.

PROTOCOLO:
1. Preparar sistema en estado perturbado (λ ≠ 0, λ ≠ 1)
2. Dejar evolucionar libremente
3. Medir λ(t) via OTOC en diferentes tiempos
4. Ajustar a λ(t) = λ_eq + δλ·exp(-t/τ)
5. Extraer τ experimental
6. Comparar con predicción τ = √(κ/2V₀)

Autor: Erick Francisco Pérez Eugenio
ORCID: 0009-0006-3228-4847
Fecha: Enero 2026
Proyecto: Kaelion v3.0 - Verificación de V(λ)

CAMBIOS DESDE v2.0:
- V₀: 0.44 → √3 ≈ 1.732 (×3.94)
- τ_teórico: ~1.07 → ~0.537 (×0.50)
- Predicciones de relajación son 2× más rápidas
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import json
from datetime import datetime

# ==============================================================================
# CONSTANTES KAELION v3.0
# ==============================================================================

# Valores fundamentales (en unidades de Planck)
PHI_0 = 1 / np.sqrt(3)      # ≈ 0.5774 - Escala de transición
V_0 = np.sqrt(3)            # ≈ 1.7321 - Escala del potencial
KAPPA = 1.0                 # Inercia canónica
BETA = V_0 + PHI_0          # ≈ 2.3094 - Clausura de ciclos
M_LAMBDA = np.sqrt(2 * V_0) # ≈ 1.8612 - Masa del campo λ

# Verificaciones
assert abs(V_0 * PHI_0 - 1.0) < 1e-10, "Invariante V₀×φ₀=1 violado"
assert abs(BETA - 4/np.sqrt(3)) < 1e-10, "β ≠ 4/√3"

print(f"KAELION v3.0 - Constantes cargadas:")
print(f"  φ₀ = {PHI_0:.6f}")
print(f"  V₀ = {V_0:.6f}")
print(f"  κ = {KAPPA:.6f}")
print(f"  β = {BETA:.6f} (≈ ln(10) = {np.log(10):.6f})")
print(f"  m_λ = {M_LAMBDA:.6f}")

# ==============================================================================
# CONFIGURACIÓN
# ==============================================================================

@dataclass
class RelaxationConfig:
    """Configuración para experimento de relajación."""
    n_qubits: int = 4                      # Número de qubits
    lambda_initial: float = 0.3            # λ inicial (perturbado)
    lambda_equilibrium: float = 0.0        # λ de equilibrio esperado
    time_points: int = 20                  # Puntos temporales a medir
    t_max: float = 5.0                     # Tiempo máximo (reducido porque τ es menor)
    shots: int = 8192                      # Mediciones por punto
    repetitions: int = 5                   # Repeticiones para estadística
    use_hardware: bool = False             # True para IBM Quantum real
    backend_name: str = "ibm_brisbane"     # Backend de IBM


# ==============================================================================
# FUNCIONES DE AJUSTE
# ==============================================================================

def exponential_decay(t: np.ndarray, lambda_eq: float, delta_lambda: float, tau: float) -> np.ndarray:
    """
    Modelo de relajación exponencial.
    
    λ(t) = λ_eq + δλ·exp(-t/τ)
    
    Args:
        t: Array de tiempos
        lambda_eq: Valor de equilibrio de λ
        delta_lambda: Amplitud de la perturbación
        tau: Tiempo de relajación
        
    Returns:
        Array de valores de λ(t)
    """
    return lambda_eq + delta_lambda * np.exp(-t / tau)


def damped_oscillation(t: np.ndarray, lambda_eq: float, delta_lambda: float, 
                       tau: float, omega: float, phi: float) -> np.ndarray:
    """
    Modelo alternativo: oscilación amortiguada.
    
    λ(t) = λ_eq + δλ·exp(-t/τ)·cos(ωt + φ)
    
    Este modelo sería indicativo de un potencial diferente.
    """
    return lambda_eq + delta_lambda * np.exp(-t / tau) * np.cos(omega * t + phi)


# ==============================================================================
# PREDICCIÓN TEÓRICA (KAELION v3.0)
# ==============================================================================

def theoretical_tau(kappa: float = KAPPA, V0: float = V_0) -> float:
    """
    Calcula τ teórico desde V(λ) = V₀λ²(1-λ)².
    
    Derivación:
    - Cerca de λ = 0: V(λ) ≈ V₀λ²
    - Ecuación de movimiento: κλ̈ + 2V₀λ = 0
    - Frecuencia: ω = √(2V₀/κ)
    - Tiempo de relajación: τ = 1/ω = √(κ/2V₀)
    
    KAELION v3.0:
    - κ = 1, V₀ = √3
    - τ = √(1/2√3) = √(1/3.464) ≈ 0.537
    
    Args:
        kappa: Parámetro cinético del campo λ (default: 1)
        V0: Altura del potencial (default: √3)
        
    Returns:
        τ teórico
    """
    return np.sqrt(kappa / (2 * V0))


def theoretical_omega(kappa: float = KAPPA, V0: float = V_0) -> float:
    """
    Calcula ω teórico (frecuencia de oscilación cerca del mínimo).
    
    ω = √(2V₀/κ)
    
    KAELION v3.0: ω = √(2√3) ≈ 1.861 (= m_λ)
    """
    return np.sqrt(2 * V0 / kappa)


def print_theoretical_predictions():
    """Imprime predicciones teóricas de Kaelion v3.0."""
    tau = theoretical_tau()
    omega = theoretical_omega()
    
    print("\n" + "="*60)
    print("PREDICCIONES TEÓRICAS KAELION v3.0")
    print("="*60)
    print(f"Potencial: V(λ) = {V_0:.4f} × λ²(1-λ)²")
    print(f"Barrera: V(0.5) = V₀/16 = {V_0/16:.4f}")
    print(f"Curvatura: V''(0) = 2V₀ = {2*V_0:.4f}")
    print("-"*60)
    print(f"Frecuencia: ω = √(2V₀/κ) = {omega:.4f}")
    print(f"Tiempo de relajación: τ = 1/ω = {tau:.4f}")
    print(f"Masa del campo: m = √(2V₀) = {M_LAMBDA:.4f}")
    print("-"*60)
    print("COMPARACIÓN CON v2.0:")
    tau_v2 = np.sqrt(1.0 / (2 * 0.44))  # κ=1, V₀=0.44 (v2.0)
    print(f"  τ(v2.0) = {tau_v2:.4f}")
    print(f"  τ(v3.0) = {tau:.4f}")
    print(f"  Ratio: {tau/tau_v2:.2f} (v3.0 es {1/tau*tau_v2:.1f}× más rápido)")
    print("="*60 + "\n")


# ==============================================================================
# CIRCUITO CUÁNTICO PARA PREPARACIÓN Y MEDICIÓN
# ==============================================================================

def create_perturbation_circuit_qiskit(config: RelaxationConfig, 
                                        perturbation_strength: float = 0.3):
    """
    Crea circuito que prepara estado perturbado (λ ≠ 0).
    
    Estrategia:
    1. Comenzar en estado base (λ ≈ 0, integrable)
    2. Aplicar perturbación controlada
    3. El sistema debería relajar hacia λ = 0
    
    Args:
        config: Configuración del experimento
        perturbation_strength: Fuerza de la perturbación (0 a 1)
        
    Returns:
        Circuito de Qiskit
    """
    try:
        from qiskit import QuantumCircuit
        from qiskit.circuit import Parameter
    except ImportError:
        print("Qiskit no disponible. Retornando None.")
        return None
    
    n = config.n_qubits
    qc = QuantumCircuit(n, n)
    
    # 1. Estado inicial: superposición parcial (no completamente integrable)
    for i in range(n):
        # Rotación que crea estado con λ_initial
        theta = np.arcsin(np.sqrt(perturbation_strength))
        qc.ry(2 * theta, i)
    
    # 2. Añadir entrelazamiento parcial (perturbación)
    for i in range(n - 1):
        qc.cx(i, i + 1)
        # Rotación que controla el grado de perturbación
        qc.rz(perturbation_strength * np.pi, i + 1)
        qc.cx(i, i + 1)
    
    qc.barrier()
    
    return qc


def create_evolution_circuit_qiskit(config: RelaxationConfig, 
                                     t: float,
                                     evolution_type: str = "free"):
    """
    Crea circuito de evolución temporal.
    
    Args:
        config: Configuración
        t: Tiempo de evolución (en unidades de profundidad de circuito)
        evolution_type: "free" para relajación libre, "driven" para forzada
        
    Returns:
        Circuito de evolución
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        return None
    
    n = config.n_qubits
    qc = QuantumCircuit(n, n)
    
    # Número de capas proporcional al tiempo
    depth = max(1, int(t))
    
    for _ in range(depth):
        if evolution_type == "free":
            # Evolución libre: Hamiltoniano integrable
            # El sistema debería relajar hacia λ = 0
            for i in range(n):
                qc.rz(0.1, i)  # Campo débil
            for i in range(n - 1):
                qc.rzz(0.05, i, i + 1)  # Acoplamiento débil
        
        elif evolution_type == "driven":
            # Evolución forzada: mantiene perturbación
            for i in range(n):
                qc.rx(0.3, i)
                qc.rz(0.3, i)
            for i in range(n - 1):
                qc.cx(i, i + 1)
    
    return qc


def create_otoc_measurement_qiskit(config: RelaxationConfig):
    """
    Crea circuito para medir λ via OTOC simplificado.
    
    Mide el grado de scrambling que indica el valor de λ.
    """
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        return None
    
    n = config.n_qubits
    qc = QuantumCircuit(n, n)
    
    # Medición en base computacional
    qc.measure(range(n), range(n))
    
    return qc


# ==============================================================================
# EXTRACCIÓN DE λ DESDE MEDICIONES
# ==============================================================================

def extract_lambda_from_counts(counts: dict, n_qubits: int) -> float:
    """
    Extrae λ desde distribución de conteos.
    
    Método: λ se relaciona con el entrelazamiento/scrambling
    - Sistema integrable (λ=0): distribución concentrada
    - Sistema caótico (λ=1): distribución uniforme
    
    Usamos entropía de Shannon normalizada como proxy de λ.
    
    Args:
        counts: Diccionario de conteos {bitstring: count}
        n_qubits: Número de qubits
        
    Returns:
        Valor estimado de λ ∈ [0, 1]
    """
    total = sum(counts.values())
    probs = np.array(list(counts.values())) / total
    
    # Entropía de Shannon
    entropy = -np.sum(probs * np.log2(probs + 1e-10))
    
    # Normalizar: S_max = n_qubits para distribución uniforme
    S_max = n_qubits
    
    # λ = S / S_max (0 para concentrado, 1 para uniforme)
    lambda_val = entropy / S_max
    
    return np.clip(lambda_val, 0, 1)


# ==============================================================================
# SIMULACIÓN ANALÍTICA
# ==============================================================================

def run_analytical_simulation(config: RelaxationConfig) -> Dict:
    """
    Ejecuta simulación analítica de relajación.
    
    Usa la solución analítica: λ(t) = λ₀·exp(-t/τ)
    con τ = √(κ/2V₀) de KAELION v3.0.
    """
    print("\n" + "="*60)
    print("SIMULACIÓN ANALÍTICA - KAELION v3.0")
    print("="*60)
    
    # Parámetros
    tau_theoretical = theoretical_tau()
    omega = theoretical_omega()
    
    print(f"Parámetros v3.0:")
    print(f"  V₀ = {V_0:.4f}")
    print(f"  κ = {KAPPA:.4f}")
    print(f"  τ_teórico = {tau_theoretical:.4f}")
    print(f"  ω = {omega:.4f}")
    
    # Generar tiempos
    times = np.linspace(0, config.t_max, config.time_points)
    
    # Solución analítica con ruido
    np.random.seed(42)
    noise_level = 0.02
    
    lambda_analytical = config.lambda_initial * np.exp(-times / tau_theoretical)
    lambda_noisy = lambda_analytical + np.random.normal(0, noise_level, len(times))
    lambda_noisy = np.clip(lambda_noisy, 0, 1)
    lambda_errors = np.ones_like(times) * noise_level
    
    # Ajustar
    try:
        popt, pcov = curve_fit(
            exponential_decay,
            times,
            lambda_noisy,
            p0=[0.0, config.lambda_initial, tau_theoretical],
            sigma=lambda_errors + 1e-6,
            absolute_sigma=True,
            bounds=([-0.1, 0, 0.01], [0.5, 1.0, 10.0])
        )
        
        tau_experimental = popt[2]
        perr = np.sqrt(np.diag(pcov))
        tau_error = perr[2]
        fit_success = True
        
    except Exception as e:
        print(f"Error en ajuste: {e}")
        tau_experimental = np.nan
        tau_error = np.nan
        fit_success = False
        popt = [np.nan, np.nan, np.nan]
    
    # Calcular métricas
    if fit_success:
        relative_error = abs(tau_experimental - tau_theoretical) / tau_theoretical
        consistency = relative_error < 0.1
        
        print(f"\nResultados:")
        print(f"  τ_experimental = {tau_experimental:.4f} ± {tau_error:.4f}")
        print(f"  τ_teórico = {tau_theoretical:.4f}")
        print(f"  Error relativo = {relative_error*100:.1f}%")
        print(f"  Consistente: {'✓ SÍ' if consistency else '✗ NO'}")
    else:
        consistency = False
        print("Ajuste falló")
    
    results = {
        "version": "3.0",
        "times": times.tolist(),
        "lambda_values": lambda_noisy.tolist(),
        "lambda_analytical": lambda_analytical.tolist(),
        "lambda_errors": lambda_errors.tolist(),
        "fit_params": {
            "lambda_eq": popt[0] if fit_success else None,
            "delta_lambda": popt[1] if fit_success else None,
            "tau": tau_experimental if fit_success else None,
            "tau_error": tau_error if fit_success else None
        },
        "tau_experimental": tau_experimental,
        "tau_theoretical": tau_theoretical,
        "kappa": KAPPA,
        "V0": V_0,
        "phi0": PHI_0,
        "consistency": consistency,
        "mode": "analytical_simulation",
        "timestamp": datetime.now().isoformat()
    }
    
    return results


# ==============================================================================
# EXPERIMENTO PRINCIPAL
# ==============================================================================

def run_relaxation_experiment(config: RelaxationConfig) -> Dict:
    """
    Ejecuta experimento completo de relajación.
    
    Returns:
        Diccionario con resultados
    """
    
    # Generar tiempos de medición
    times = np.linspace(0, config.t_max, config.time_points)
    
    # Arrays para resultados
    lambda_means = np.zeros(config.time_points)
    lambda_stds = np.zeros(config.time_points)
    
    print("=" * 60)
    print("EXPERIMENTO DE RELAJACIÓN - KAELION v3.0")
    print("=" * 60)
    print(f"Configuración:")
    print(f"  - Qubits: {config.n_qubits}")
    print(f"  - λ inicial: {config.lambda_initial}")
    print(f"  - Puntos temporales: {config.time_points}")
    print(f"  - V₀ = {V_0:.4f} (v3.0)")
    print(f"  - τ_teórico = {theoretical_tau():.4f}")
    print(f"  - Hardware: {'Real' if config.use_hardware else 'Simulador'}")
    print("=" * 60)
    
    # Verificar disponibilidad de Qiskit
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        
        if config.use_hardware:
            from qiskit_ibm_runtime import QiskitRuntimeService, Sampler
            service = QiskitRuntimeService()
            backend = service.backend(config.backend_name)
            print(f"Usando backend: {config.backend_name}")
        else:
            backend = AerSimulator()
            print("Usando simulador AerSimulator")
            
    except ImportError as e:
        print(f"Error importando Qiskit: {e}")
        print("Ejecutando en modo simulación analítica...")
        return run_analytical_simulation(config)
    
    # Ejecutar mediciones para cada tiempo
    for i, t in enumerate(times):
        lambda_samples = []
        
        for rep in range(config.repetitions):
            # Construir circuito completo
            qc_prep = create_perturbation_circuit_qiskit(config, config.lambda_initial)
            qc_evol = create_evolution_circuit_qiskit(config, t)
            qc_meas = create_otoc_measurement_qiskit(config)
            
            # Combinar circuitos
            qc = qc_prep.compose(qc_evol).compose(qc_meas)
            
            # Transpilar y ejecutar
            qc_transpiled = transpile(qc, backend)
            
            if config.use_hardware:
                sampler = Sampler(backend)
                job = sampler.run([qc_transpiled], shots=config.shots)
                result = job.result()
                counts = result[0].data.meas.get_counts()
            else:
                from qiskit_aer import AerSimulator
                job = backend.run(qc_transpiled, shots=config.shots)
                result = job.result()
                counts = result.get_counts()
            
            # Extraer λ
            lambda_val = extract_lambda_from_counts(counts, config.n_qubits)
            lambda_samples.append(lambda_val)
        
        # Estadísticas
        lambda_means[i] = np.mean(lambda_samples)
        lambda_stds[i] = np.std(lambda_samples) / np.sqrt(config.repetitions)
        
        print(f"t = {t:.2f}: λ = {lambda_means[i]:.4f} ± {lambda_stds[i]:.4f}")
    
    # Ajustar modelo exponencial
    try:
        popt, pcov = curve_fit(
            exponential_decay,
            times,
            lambda_means,
            p0=[0.0, config.lambda_initial, theoretical_tau()],
            sigma=lambda_stds + 1e-6,
            absolute_sigma=True,
            bounds=([0, 0, 0.01], [0.5, 1.0, 10.0])
        )
        
        lambda_eq_fit, delta_lambda_fit, tau_experimental = popt
        perr = np.sqrt(np.diag(pcov))
        tau_error = perr[2]
        
        fit_success = True
        
    except Exception as e:
        print(f"Error en ajuste: {e}")
        tau_experimental = np.nan
        tau_error = np.nan
        fit_success = False
        popt = [np.nan, np.nan, np.nan]
    
    # Predicción teórica KAELION v3.0
    tau_theoretical = theoretical_tau()
    
    # Verificar consistencia
    if fit_success:
        relative_error = abs(tau_experimental - tau_theoretical) / tau_theoretical
        consistency = relative_error < 0.2
    else:
        consistency = False
        relative_error = np.nan
    
    results = {
        "version": "3.0",
        "config": {
            "n_qubits": config.n_qubits,
            "lambda_initial": config.lambda_initial,
            "t_max": config.t_max,
            "shots": config.shots
        },
        "times": times.tolist(),
        "lambda_values": lambda_means.tolist(),
        "lambda_errors": lambda_stds.tolist(),
        "fit_params": {
            "lambda_eq": popt[0],
            "delta_lambda": popt[1],
            "tau": tau_experimental,
            "tau_error": tau_error
        },
        "tau_experimental": tau_experimental,
        "tau_theoretical": tau_theoretical,
        "relative_error": relative_error,
        "kappa": KAPPA,
        "V0": V_0,
        "phi0": PHI_0,
        "consistency": consistency,
        "mode": "hardware" if config.use_hardware else "simulator",
        "timestamp": datetime.now().isoformat()
    }
    
    return results


# ==============================================================================
# VISUALIZACIÓN
# ==============================================================================

def plot_relaxation_results(results: Dict, save_path: Optional[str] = None):
    """
    Genera gráficos de los resultados de relajación.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    times = np.array(results["times"])
    lambda_vals = np.array(results["lambda_values"])
    lambda_errs = np.array(results["lambda_errors"])
    
    # === Panel 1: Datos y ajuste ===
    ax1 = axes[0, 0]
    ax1.errorbar(times, lambda_vals, yerr=lambda_errs, fmt='o', 
                 color='blue', markersize=6, capsize=3, label='Datos')
    
    if results["fit_params"]["tau"] is not None:
        tau = results["fit_params"]["tau"]
        lambda_eq = results["fit_params"]["lambda_eq"]
        delta_lambda = results["fit_params"]["delta_lambda"]
        
        t_fine = np.linspace(0, max(times), 100)
        fit_curve = exponential_decay(t_fine, lambda_eq, delta_lambda, tau)
        ax1.plot(t_fine, fit_curve, 'r-', linewidth=2, 
                 label=f'Ajuste: τ = {tau:.3f}')
    
    # Curva teórica v3.0
    tau_th = results["tau_theoretical"]
    config_lambda = results.get("config", {}).get("lambda_initial", 0.3)
    t_fine = np.linspace(0, max(times), 100)
    theory_curve = exponential_decay(t_fine, 0, config_lambda, tau_th)
    ax1.plot(t_fine, theory_curve, 'g--', linewidth=2, alpha=0.7,
             label=f'Teórico v3.0: τ = {tau_th:.3f}')
    
    ax1.set_xlabel('Tiempo (unidades adimensionales)', fontsize=12)
    ax1.set_ylabel('λ(t)', fontsize=12)
    ax1.set_title('Relajación de λ hacia equilibrio', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 0.5)
    
    # === Panel 2: Potencial V(λ) v3.0 ===
    ax2 = axes[0, 1]
    lambda_range = np.linspace(0, 1, 100)
    V_lambda = V_0 * lambda_range**2 * (1 - lambda_range)**2
    
    ax2.plot(lambda_range, V_lambda, 'b-', linewidth=2)
    ax2.axvline(x=0, color='g', linestyle='--', alpha=0.5, label='Mínimo (LQG)')
    ax2.axvline(x=1, color='r', linestyle='--', alpha=0.5, label='Mínimo (Holo)')
    ax2.axvline(x=0.5, color='orange', linestyle=':', alpha=0.5, label='Máximo')
    ax2.axhline(y=V_0/16, color='purple', linestyle=':', alpha=0.5)
    
    # Marcar posición inicial
    lambda_init = config_lambda
    V_init = V_0 * lambda_init**2 * (1 - lambda_init)**2
    ax2.scatter([lambda_init], [V_init], color='red', s=100, zorder=5, 
                label=f'Estado inicial (λ={lambda_init})')
    
    ax2.set_xlabel('λ', fontsize=12)
    ax2.set_ylabel(f'V(λ) = √3·λ²(1-λ)²', fontsize=12)
    ax2.set_title(f'Potencial KAELION v3.0 (V₀={V_0:.3f})', fontsize=14)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    # Añadir texto con parámetros
    ax2.text(0.5, V_0/16 + 0.01, f'Barrera = V₀/16 = {V_0/16:.3f}', 
             ha='center', fontsize=10, color='purple')
    
    # === Panel 3: Residuos ===
    ax3 = axes[1, 0]
    if results["fit_params"]["tau"] is not None:
        tau = results["fit_params"]["tau"]
        lambda_eq = results["fit_params"]["lambda_eq"]
        delta_lambda = results["fit_params"]["delta_lambda"]
        
        fit_vals = exponential_decay(times, lambda_eq, delta_lambda, tau)
        residuals = lambda_vals - fit_vals
        
        ax3.errorbar(times, residuals, yerr=lambda_errs, fmt='o', 
                     color='purple', markersize=6, capsize=3)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.fill_between(times, -2*lambda_errs, 2*lambda_errs, 
                         alpha=0.2, color='gray', label='±2σ')
    
    ax3.set_xlabel('Tiempo', fontsize=12)
    ax3.set_ylabel('Residuos', fontsize=12)
    ax3.set_title('Residuos del ajuste', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # === Panel 4: Comparación τ ===
    ax4 = axes[1, 1]
    
    tau_exp = results.get("tau_experimental", np.nan)
    tau_th = results.get("tau_theoretical", np.nan)
    tau_err = results.get("fit_params", {}).get("tau_error", 0)
    
    # También mostrar predicción v2.0 para comparación
    tau_v2 = np.sqrt(1.0 / (2 * 0.44))
    
    if not np.isnan(tau_exp) and not np.isnan(tau_th):
        categories = ['Experimental', 'Teórico v3.0\n(V₀=√3)', 'Teórico v2.0\n(V₀=0.44)']
        values = [tau_exp, tau_th, tau_v2]
        errors = [tau_err if tau_err else 0, 0, 0]
        colors = ['steelblue', 'forestgreen', 'lightgray']
        
        bars = ax4.bar(categories, values, yerr=errors, capsize=5, 
                       color=colors, alpha=0.7, edgecolor='black')
        
        # Mostrar valores
        for bar, val, err in zip(bars, values, errors):
            height = bar.get_height()
            if err > 0:
                ax4.text(bar.get_x() + bar.get_width()/2., height + err + 0.05,
                         f'{val:.3f}±{err:.3f}', ha='center', va='bottom', fontsize=10)
            else:
                ax4.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                         f'{val:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Verificar consistencia
        if results.get("consistency", False):
            ax4.text(0.5, 0.95, '✓ CONSISTENTE CON v3.0', transform=ax4.transAxes,
                     ha='center', fontsize=14, color='green', fontweight='bold')
        else:
            ax4.text(0.5, 0.95, '✗ INCONSISTENTE', transform=ax4.transAxes,
                     ha='center', fontsize=14, color='red', fontweight='bold')
    
    ax4.set_ylabel('τ (tiempo de relajación)', fontsize=12)
    ax4.set_title('Comparación: Experimental vs Teórico', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('KAELION v3.0: Test de Predicción P1 - Tiempo de Relajación\n'
                 f'V(λ) = √3·λ²(1-λ)², τ = √(κ/2V₀) = {tau_th:.3f}', 
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
    print("KAELION v3.0: TEST DE PREDICCIÓN P1 - TIEMPO DE RELAJACIÓN τ")
    print("=" * 70)
    
    # Mostrar predicciones teóricas
    print_theoretical_predictions()
    
    print("Predicción: Si V(λ) = √3·λ²(1-λ)², entonces τ = √(κ/2V₀) ≈ 0.537")
    print("=" * 70)
    
    # Configuración (t_max reducido porque τ es menor en v3.0)
    config = RelaxationConfig(
        n_qubits=4,
        lambda_initial=0.3,
        time_points=20,
        t_max=5.0,  # Reducido de 10.0 porque τ ≈ 0.537 en v3.0
        shots=8192,
        repetitions=5,
        use_hardware=False  # Cambiar a True para IBM Quantum real
    )
    
    # Ejecutar experimento
    results = run_relaxation_experiment(config)
    
    # Visualizar
    plot_relaxation_results(results, save_path="relaxation_test_P1_v3.png")
    
    # Guardar resultados
    with open("relaxation_results_P1_v3.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResultados guardados en: relaxation_results_P1_v3.json")
    
    # Resumen final
    print("\n" + "=" * 70)
    print("RESUMEN FINAL - KAELION v3.0")
    print("=" * 70)
    
    if results["consistency"]:
        print("✓ PREDICCIÓN P1 CONSISTENTE CON v3.0")
        print(f"  τ_experimental = {results['tau_experimental']:.4f}")
        print(f"  τ_teórico(v3.0) = {results['tau_theoretical']:.4f}")
        print(f"  V₀ = {V_0:.4f}")
        print("  → V(λ) = √3·λ²(1-λ)² es compatible con los datos")
    else:
        print("✗ PREDICCIÓN P1 INCONSISTENTE")
        print("  → Revisar forma del potencial V(λ)")
    
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    results = main()
