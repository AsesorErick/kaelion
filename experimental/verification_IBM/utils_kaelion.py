"""
KAELION v3.0 - Utilidades para cálculo de OTOC
===============================================

Funciones comunes usadas en las simulaciones.
"""

import numpy as np

# =============================================================================
# CONSTANTES KAELION v3.0
# =============================================================================

V_0 = np.sqrt(3)        # = 1.7321
PHI_0 = 1 / np.sqrt(3)  # = 0.5774

# =============================================================================
# FUNCIONES FUNDAMENTALES
# =============================================================================

def alpha_from_lambda(lam):
    """
    Corrección logarítmica α(λ) = -1/2 - λ
    
    Interpola entre:
      λ = 0 → α = -1/2 (LQG)
      λ = 1 → α = -3/2 (Holografía)
    """
    return -0.5 - lam


def V_potential(lam):
    """
    Potencial efectivo V(λ) = V₀·λ²(1-λ)²
    
    Propiedades:
      V(0) = V(1) = 0 (mínimos)
      V(0.5) = V₀/16 (barrera)
    """
    return V_0 * (lam ** 2) * ((1 - lam) ** 2)


def V_prime(lam):
    """Primera derivada V'(λ) = 2V₀·λ(1-λ)(1-2λ)"""
    return 2 * V_0 * lam * (1 - lam) * (1 - 2 * lam)


def V_double_prime(lam):
    """Segunda derivada V''(λ) = 2V₀(6λ² - 6λ + 1)"""
    return 2 * V_0 * (6 * lam**2 - 6 * lam + 1)


def f_correlation(lam, epsilon=0.01):
    """
    Corrección a termalidad f(λ)
    
    f(λ) = (1-λ)·[1 + 1/max(|V''(λ)/V₀|, ε)]
    
    Límites:
      f(0) = 2 (máxima desviación)
      f(1) = 0 (térmica perfecta)
    """
    Vpp_normalized = abs(V_double_prime(lam) / V_0)
    Vpp_reg = max(Vpp_normalized, epsilon)
    return (1 - lam) * (1 + 1 / Vpp_reg)


def phi_from_lambda(lam):
    """Dilatón φ = φ₀·λ/(1-λ)"""
    if lam >= 1:
        return np.inf
    return PHI_0 * lam / (1 - lam)


def lambda_from_phi(phi):
    """Inversa: λ = φ/(φ₀ + φ)"""
    return phi / (PHI_0 + phi)


def phi_from_M(M):
    """Área del horizonte φ = 4πM²"""
    return 4 * np.pi * M**2


def entropy_BH(phi, lam):
    """
    Entropía de Bekenstein-Hawking corregida
    S = φ/4 + α(λ)·ln(φ)
    """
    if phi <= 0:
        return 0.0
    alpha = alpha_from_lambda(lam)
    return phi / 4 + alpha * np.log(phi)


# =============================================================================
# FUNCIONES PARA OTOC
# =============================================================================

def calculate_otoc_from_counts(counts):
    """
    Calcular OTOC desde counts de Qiskit.
    
    OTOC = Σ_s (-1)^(paridad de s) × P(s)
    
    Maneja correctamente el formato de bitstrings de Qiskit
    (que puede incluir espacios).
    """
    otoc = 0.0
    total = sum(counts.values())
    
    for bitstring, count in counts.items():
        # Limpiar espacios
        bits_clean = bitstring.replace(' ', '').replace('\n', '')
        
        # Paridad
        n_ones = bits_clean.count('1')
        parity = n_ones % 2
        
        # (-1)^paridad
        sign = 1 - 2 * parity
        
        otoc += sign * count / total
    
    return otoc


def extract_lambda_from_scrambling(otocs_initial, otocs_final, scrambling_max=1.0):
    """
    Extraer λ desde cantidad de scrambling.
    
    scrambling = OTOC_inicial - OTOC_final
    λ = scrambling / scrambling_max
    """
    scrambling = otocs_initial - otocs_final
    lam = np.clip(scrambling / scrambling_max, 0, 1)
    return lam, scrambling


# =============================================================================
# PERFILES ESPACIALES
# =============================================================================

def chaos_profile(i, i_horizon, width=1.5):
    """
    Perfil de caos espacial: transición suave de 0 a 1.
    
    chaos(i) = 0.5·(1 + tanh((i - i_h)/w))
    """
    x = (i - i_horizon) / width
    return 0.5 * (1 + np.tanh(x))


def lambda_prediction_spatial(x, x_horizon, width):
    """
    Predicción teórica de λ(x) espacial.
    """
    return np.where(
        x >= x_horizon,
        1 - np.exp(-(x - x_horizon) / width),
        np.exp((x - x_horizon) / width) * 0.5
    )


# =============================================================================
# VERIFICACIÓN
# =============================================================================

if __name__ == "__main__":
    print("KAELION v3.0 - Verificación de constantes")
    print("=" * 50)
    print(f"V₀ = √3 = {V_0:.6f}")
    print(f"φ₀ = 1/√3 = {PHI_0:.6f}")
    print(f"V₀ × φ₀ = {V_0 * PHI_0:.6f}")
    print()
    print("Verificación de límites:")
    print(f"  α(0) = {alpha_from_lambda(0):.1f} (esperado: -0.5)")
    print(f"  α(1) = {alpha_from_lambda(1):.1f} (esperado: -1.5)")
    print(f"  α(0.5) = {alpha_from_lambda(0.5):.1f} (esperado: -1.0)")
    print()
    print(f"  V(0) = {V_potential(0):.4f} (esperado: 0)")
    print(f"  V(1) = {V_potential(1):.4f} (esperado: 0)")
    print(f"  V(0.5) = {V_potential(0.5):.4f} (esperado: {V_0/16:.4f})")
    print()
    print(f"  f(0) = {f_correlation(0):.2f} (esperado: 2)")
    print(f"  f(1) = {f_correlation(1):.2f} (esperado: 0)")
