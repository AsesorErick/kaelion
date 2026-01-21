#!/usr/bin/env python3
"""
KAELION v3.0 - Physical Constants Module
=========================================

This module provides the fundamental constants of the Kaelion framework
version 3.0, derived from cosmological constraints.

Usage:
    from KAELION_V3_CONSTANTS import V_0, PHI_0, KAPPA, BETA, M_LAMBDA

Author: Erick Francisco Pérez Eugenio
ORCID: 0009-0006-3228-4847
Date: January 2026
"""

import numpy as np

# =============================================================================
# FUNDAMENTAL CONSTANTS (Planck units)
# =============================================================================

# Transition scale (from cosmological matching)
PHI_0 = 1 / np.sqrt(3)  # ≈ 0.5774

# Potential scale (from cosmological constant)
V_0 = np.sqrt(3)  # ≈ 1.7321

# Canonical inertia (normalized)
KAPPA = 1.0

# Cycle closure parameter
BETA = V_0 + PHI_0  # ≈ 2.3094 ≈ ln(10)

# Field mass
M_LAMBDA = np.sqrt(2 * V_0)  # ≈ 1.8612

# =============================================================================
# DERIVED QUANTITIES
# =============================================================================

# Barrier height at λ = 0.5
BARRIER = V_0 / 16  # ≈ 0.1083

# Relaxation time near minimum
TAU = np.sqrt(KAPPA / (2 * V_0))  # ≈ 0.5373

# Oscillation frequency near minimum
OMEGA = np.sqrt(2 * V_0 / KAPPA)  # ≈ 1.8612

# Curvature at minimum
V_DOUBLE_PRIME_0 = 2 * V_0  # ≈ 3.4641

# =============================================================================
# VERIFICATION
# =============================================================================

# Fundamental invariant
INVARIANT = V_0 * PHI_0  # Must equal 1.0

# Verify invariant
assert abs(INVARIANT - 1.0) < 1e-10, f"Invariant V₀×φ₀ = {INVARIANT} ≠ 1"

# =============================================================================
# COMPARISON WITH PREVIOUS VERSIONS
# =============================================================================

V0_V1 = 0.125   # v1.0 (arbitrary)
V0_V2 = 0.44    # v2.0 (phenomenological)
V0_V3 = V_0     # v3.0 (cosmological)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def potential_V(lambda_val: float, V0: float = V_0) -> float:
    """
    Kaelion potential V(λ) = V₀·λ²·(1-λ)²
    
    Args:
        lambda_val: Interpolation parameter λ ∈ [0, 1]
        V0: Potential scale (default: √3)
    
    Returns:
        V(λ)
    """
    return V0 * lambda_val**2 * (1 - lambda_val)**2


def theoretical_tau(kappa: float = KAPPA, V0: float = V_0) -> float:
    """
    Relaxation time τ = √(κ/2V₀)
    """
    return np.sqrt(kappa / (2 * V0))


def theoretical_variance(T: float, V0: float = V_0) -> float:
    """
    Thermal fluctuation variance ⟨δλ²⟩ = T/(2V₀)
    """
    return T / (2 * V0)


def alpha_lambda(lambda_val: float) -> float:
    """
    Logarithmic correction coefficient α(λ) = -1/2 - λ
    
    Note: This is INDEPENDENT of V₀
    """
    return -0.5 - lambda_val


# =============================================================================
# PRINT SUMMARY IF RUN DIRECTLY
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("KAELION v3.0 - Physical Constants")
    print("=" * 60)
    print(f"\nFundamental Constants:")
    print(f"  φ₀ = 1/√3     = {PHI_0:.6f}")
    print(f"  V₀ = √3       = {V_0:.6f}")
    print(f"  κ  = 1        = {KAPPA:.6f}")
    print(f"  β  = V₀ + φ₀  = {BETA:.6f}")
    print(f"  m_λ = √(2V₀)  = {M_LAMBDA:.6f}")
    
    print(f"\nDerived Quantities:")
    print(f"  Barrier ΔE    = V₀/16    = {BARRIER:.6f}")
    print(f"  Relaxation τ  = √(κ/2V₀) = {TAU:.6f}")
    print(f"  Frequency ω   = √(2V₀/κ) = {OMEGA:.6f}")
    print(f"  Curvature V'' = 2V₀      = {V_DOUBLE_PRIME_0:.6f}")
    
    print(f"\nInvariant Check:")
    print(f"  V₀ × φ₀ = {INVARIANT:.10f} {'✓' if abs(INVARIANT-1)<1e-10 else '✗'}")
    
    print(f"\nVersion Comparison:")
    print(f"  v1.0: V₀ = {V0_V1:.4f}")
    print(f"  v2.0: V₀ = {V0_V2:.4f}")
    print(f"  v3.0: V₀ = {V0_V3:.4f} (current)")
    
    print(f"\nRatios (v3.0 / v2.0):")
    print(f"  V₀ ratio:      {V0_V3/V0_V2:.2f}×")
    print(f"  τ ratio:       {theoretical_tau(V0=V0_V3)/theoretical_tau(V0=V0_V2):.2f}×")
    print(f"  Barrier ratio: {(V0_V3/16)/(V0_V2/16):.2f}×")
    
    print("=" * 60)
