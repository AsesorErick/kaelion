"""
CHSH State-Dependent Verification Code
=======================================
Supplementary material for:
"Clarifying State Dependence in CHSH: Optimal Angular Configurations for the Four Bell States"

Author: Erick Francisco Pérez Eugenio
Project: Kaelion v3.0
Date: January 2026

This code demonstrates that the standard angular configuration {0, π/2, π/4, -π/4}
yields S=0 for |Φ⁺⟩ when using CHSH Form I, revealing a common pedagogical omission.
"""

import numpy as np

def correlation(theta_a, theta_b, phi_a=0, phi_b=0, delta=0):
    """
    Quantum correlation for Bell states.
    
    Parameters:
    -----------
    theta_a, theta_b : float
        Polar angles for Alice and Bob measurements
    phi_a, phi_b : float
        Azimuthal angles (default 0 for XZ plane measurements)
    delta : float
        Phase parameter: 0 for |Φ⁺⟩ and |Ψ⁻⟩, π for |Φ⁻⟩ and |Ψ⁺⟩
    
    Returns:
    --------
    float : Expected correlation value E(a,b)
    """
    return (np.cos(theta_a) * np.cos(theta_b) + 
            np.sin(theta_a) * np.sin(theta_b) * np.cos(phi_a - phi_b + delta))

def chsh_form1(theta_a, theta_ap, theta_b, theta_bp, 
               phi_a=0, phi_ap=0, phi_b=0, phi_bp=0, delta=0):
    """
    CHSH Form I (Historical): S = |E(a,b) - E(a,b') + E(a',b) + E(a',b')|
    """
    E_ab = correlation(theta_a, theta_b, phi_a, phi_b, delta)
    E_abp = correlation(theta_a, theta_bp, phi_a, phi_bp, delta)
    E_apb = correlation(theta_ap, theta_b, phi_ap, phi_b, delta)
    E_apbp = correlation(theta_ap, theta_bp, phi_ap, phi_bp, delta)
    return abs(E_ab - E_abp + E_apb + E_apbp)

def chsh_form2(theta_a, theta_ap, theta_b, theta_bp, 
               phi_a=0, phi_ap=0, phi_b=0, phi_bp=0, delta=0):
    """
    CHSH Form II (Alternative): S = |E(a,b) + E(a,b') + E(a',b) - E(a',b')|
    """
    E_ab = correlation(theta_a, theta_b, phi_a, phi_b, delta)
    E_abp = correlation(theta_a, theta_bp, phi_a, phi_bp, delta)
    E_apb = correlation(theta_ap, theta_b, phi_ap, phi_b, delta)
    E_apbp = correlation(theta_ap, theta_bp, phi_ap, phi_bp, delta)
    return abs(E_ab + E_abp + E_apb - E_apbp)

if __name__ == "__main__":
    # Angular configurations
    angles_std = {
        'theta_a': 0,
        'theta_ap': np.pi/2,
        'theta_b': np.pi/4,
        'theta_bp': -np.pi/4  # Standard textbook configuration
    }
    
    angles_opt = {
        'theta_a': 0,
        'theta_ap': np.pi/2,
        'theta_b': np.pi/4,
        'theta_bp': 3*np.pi/4  # Optimal for |Φ⁺⟩ with Form I
    }
    
    # Demonstration of pedagogical error
    print("="*70)
    print("DEMONSTRATION OF PEDAGOGICAL OMISSION IN CHSH")
    print("="*70)
    print("\nState: |Φ⁺⟩ = (|00⟩ + |11⟩)/√2")
    print("-"*50)
    
    S_wrong = chsh_form1(delta=0, **angles_std)
    S_correct = chsh_form1(delta=0, **angles_opt)
    
    print(f"\n1. Standard config (b'=-45°) with Form I:")
    print(f"   S = {S_wrong:.6f}  → ERROR (expected 2√2)")
    
    print(f"\n2. Optimal config (b'=135°) with Form I:")
    print(f"   S = {S_correct:.6f}  → CORRECT")
    
    print(f"\nTsirelson bound (2√2): {2*np.sqrt(2):.6f}")
    
    # Demonstration of CHSH duality
    print("\n" + "="*70)
    print("CHSH DUALITY DEMONSTRATION")
    print("="*70)
    print("\nSame angles (b'=-45°), different CHSH form:")
    
    S1_std = chsh_form1(delta=0, **angles_std)
    S2_std = chsh_form2(delta=0, **angles_std)
    
    print(f"  Form I:  S = {S1_std:.6f}  → ZERO!")
    print(f"  Form II: S = {S2_std:.6f}  → 2√2 (quantum maximum)")
    
    # Summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print("""
┌─────────────┬──────────────┬─────────┬──────────┬───────────┐
│ State       │ Config       │ Form    │ S        │ Result    │
├─────────────┼──────────────┼─────────┼──────────┼───────────┤
│ |Φ⁺⟩        │ std (b'=-45°)│ Form I  │ 0.000000 │ ✗ ERROR   │
│ |Φ⁺⟩        │ std (b'=-45°)│ Form II │ 2.828427 │ ✓ 2√2     │
│ |Φ⁺⟩        │ opt (b'=135°)│ Form I  │ 2.828427 │ ✓ 2√2     │
└─────────────┴──────────────┴─────────┴──────────┴───────────┘
""")
    print("="*70)
    print("✓ Code verification complete")
    print("="*70)
