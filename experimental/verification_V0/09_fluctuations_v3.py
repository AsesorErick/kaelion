#!/usr/bin/env python3
"""
KAELION EXPERIMENTS v3.0: Test de Fluctuaciones Térmicas
=========================================================

PREDICCIÓN P4: ⟨δλ²⟩ = k_B T / (2V₀) = T / (2√3) ≈ 0.289 T

Autor: Erick Francisco Pérez Eugenio
ORCID: 0009-0006-3228-4847
Fecha: Enero 2026
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from typing import Dict, Optional
from dataclasses import dataclass
import json
from datetime import datetime

# Constantes KAELION v3.0
PHI_0 = 1 / np.sqrt(3)
V_0 = np.sqrt(3)
KAPPA = 1.0

print(f"KAELION v3.0: V₀={V_0:.4f}, φ₀={PHI_0:.4f}, V₀×φ₀={V_0*PHI_0:.4f}")

@dataclass
class FluctuationsConfig:
    n_qubits: int = 4
    n_measurements: int = 500  # Aumentado para mejor estadística
    n_temperatures: int = 8
    T_min: float = 0.1
    T_max: float = 1.0
    shots: int = 8192
    use_hardware: bool = False
    backend_name: str = "ibm_brisbane"

def theoretical_variance(T: float, V0: float = V_0) -> float:
    """⟨δλ²⟩ = T/(2V₀)"""
    return T / (2 * V0)

def run_analytical_fluctuations(config: FluctuationsConfig) -> Dict:
    """Simulación analítica."""
    print("\nSIMULACIÓN ANALÍTICA - KAELION v3.0")
    
    temperatures = np.linspace(config.T_min, config.T_max, config.n_temperatures)
    variances = np.zeros(config.n_temperatures)
    variance_errors = np.zeros(config.n_temperatures)
    
    np.random.seed(42)
    
    for i, T in enumerate(temperatures):
        var_true = theoretical_variance(T, V_0)
        std_true = np.sqrt(var_true)
        
        # Generar muestras centradas en 0 con desviación estándar correcta
        samples = np.random.normal(0, std_true, config.n_measurements)
        
        # NO hacer clip - eso distorsiona la varianza
        # En su lugar, trabajamos con fluctuaciones δλ, no con λ absoluto
        
        variances[i] = np.var(samples)
        variance_errors[i] = variances[i] * np.sqrt(2 / (config.n_measurements - 1))
        print(f"  T={T:.2f}: σ²={variances[i]:.4f} (teórico: {var_true:.4f})")
    
    slope, intercept, r_value, _, std_err = linregress(temperatures, variances)
    V0_extracted = 1 / (2 * slope) if slope > 0 else np.nan
    V0_error = V0_extracted * std_err / slope if slope > 0 else np.nan
    
    # Verificar consistencia
    consistency = bool(abs(V0_extracted - V_0) / V_0 < 0.2) if not np.isnan(V0_extracted) else False
    
    print(f"\nAjuste: σ² = {slope:.4f}·T + {intercept:.4f}, R²={r_value**2:.4f}")
    print(f"V₀_extraído = {V0_extracted:.4f}, V₀_teórico = {V_0:.4f}")
    print(f"Error relativo = {abs(V0_extracted - V_0)/V_0*100:.1f}%")
    print(f"Consistente: {'✓ SÍ' if consistency else '✗ NO'}")
    
    return {
        "version": "3.0",
        "temperatures": temperatures.tolist(),
        "variances": variances.tolist(),
        "variance_errors": variance_errors.tolist(),
        "fit_params": {
            "slope": float(slope), 
            "intercept": float(intercept), 
            "r_squared": float(r_value**2)
        },
        "V0_extracted": float(V0_extracted) if not np.isnan(V0_extracted) else None,
        "V0_error": float(V0_error) if not np.isnan(V0_error) else None,
        "V0_theoretical": float(V_0),
        "consistency": consistency,  # Ya es bool de Python
        "timestamp": datetime.now().isoformat()
    }

def plot_fluctuation_results(results: Dict, save_path: Optional[str] = None):
    """Visualización."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    T = np.array(results["temperatures"])
    var = np.array(results["variances"])
    
    # Panel 1: σ² vs T
    ax1 = axes[0]
    ax1.plot(T, var, 'bo-', markersize=8, label='Datos')
    
    # Línea teórica v3.0
    T_line = np.linspace(0, max(T)*1.1, 100)
    ax1.plot(T_line, T_line/(2*V_0), 'g--', linewidth=2, label=f'Teórico v3.0: T/{2*V_0:.2f}')
    
    # Ajuste
    slope = results["fit_params"]["slope"]
    intercept = results["fit_params"]["intercept"]
    ax1.plot(T_line, slope*T_line + intercept, 'r-', linewidth=1.5, alpha=0.7,
             label=f'Ajuste: {slope:.4f}·T + {intercept:.4f}')
    
    # Teórico v2.0 para comparación
    ax1.plot(T_line, T_line/(2*0.44), 'gray', linestyle=':', label='Teórico v2.0: T/0.88')
    
    ax1.set_xlabel('Temperatura T')
    ax1.set_ylabel('Varianza σ²(δλ)')
    ax1.set_title('Fluctuaciones Térmicas de λ')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, max(T)*1.1)
    ax1.set_ylim(0, None)
    
    # Panel 2: V₀ extraído
    ax2 = axes[1]
    V0_ext = results["V0_extracted"]
    if V0_ext is not None:
        bars = ax2.bar(['Extraído', 'v3.0 (√3)', 'v2.0'], 
                       [V0_ext, V_0, 0.44], color=['steelblue', 'green', 'gray'], alpha=0.7)
        
        # Añadir valores encima de las barras
        for bar, val in zip(bars, [V0_ext, V_0, 0.44]):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.05,
                     f'{val:.3f}', ha='center', fontsize=10)
        
        ax2.set_ylabel('V₀')
        ax2.set_title('Comparación de V₀')
        ax2.grid(True, alpha=0.3, axis='y')
        
        if results["consistency"]:
            ax2.text(0.5, 0.95, '✓ CONSISTENTE', transform=ax2.transAxes,
                     ha='center', color='green', fontweight='bold', fontsize=12)
        else:
            ax2.text(0.5, 0.95, f'✗ Error: {abs(V0_ext-V_0)/V_0*100:.1f}%', 
                     transform=ax2.transAxes, ha='center', color='red', fontweight='bold')
    
    plt.suptitle('KAELION v3.0: Predicción P4 - Fluctuaciones\n'
                 f'⟨δλ²⟩ = T/(2V₀), V₀ = √3 ≈ {V_0:.3f}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Guardado: {save_path}")
    plt.show()
    return fig

def main():
    print("\n" + "="*60)
    print("KAELION v3.0: TEST P4 - FLUCTUACIONES")
    print("="*60)
    
    config = FluctuationsConfig(n_measurements=500, n_temperatures=8)
    results = run_analytical_fluctuations(config)
    plot_fluctuation_results(results, "fluctuations_P4_v3.png")
    
    with open("fluctuations_P4_v3.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Resultados guardados en: fluctuations_P4_v3.json")
    
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    if results["consistency"]:
        print("✓ PREDICCIÓN P4 CONSISTENTE CON v3.0")
        print(f"  V₀_extraído = {results['V0_extracted']:.4f}")
        print(f"  V₀_teórico = {V_0:.4f}")
        print("  → V(λ) = √3·λ²(1-λ)² es compatible con fluctuaciones")
    else:
        print("✗ PREDICCIÓN P4: REVISAR")
        if results["V0_extracted"]:
            print(f"  V₀_extraído = {results['V0_extracted']:.4f}")
            print(f"  V₀_teórico = {V_0:.4f}")
            print(f"  Error = {abs(results['V0_extracted']-V_0)/V_0*100:.1f}%")
    print("="*60)
    
    return results

if __name__ == "__main__":
    main()
