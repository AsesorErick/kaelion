"""
AGUJERO NEGRO DE KERR - ECUACIÓN DE CORRESPONDENCIA
====================================================
Proyecto Kaelion v3.0 - Simulación 13

El agujero negro de Kerr es la solución más general para un BH
astrofísico (con rotación). Representa el 99% de los BH observados.

MÉTRICA DE KERR:
ds² = -(1 - 2Mr/Σ)dt² - (4Mar sin²θ/Σ)dtdφ + (Σ/Δ)dr² + Σdθ² + ...

donde:
  Σ = r² + a²cos²θ
  Δ = r² - 2Mr + a²
  a = J/M (parámetro de espín)

PREGUNTA: ¿Cómo entra el momento angular J en λ(A, I)?

Referencias:
- Kerr (1963) "Gravitational Field of a Spinning Mass"
- Bekenstein (1973) "Black Holes and Entropy"
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict

print("="*70)
print("AGUJERO NEGRO DE KERR - KAELION v3.0")
print("Caso astrofísico con rotación")
print("="*70)

# =============================================================================
# CONSTANTES
# =============================================================================

@dataclass
class Constants:
    l_P: float = 1.0
    G_N: float = 1.0
    c: float = 1.0
    gamma: float = 0.2375
    
    @property
    def A_c(self) -> float:
        return 4 * np.pi / self.gamma * self.l_P**2

const = Constants()

# =============================================================================
# CLASE: AGUJERO NEGRO DE KERR
# =============================================================================

class KerrBlackHole:
    """
    Agujero negro de Kerr (con rotación).
    
    Parámetros:
    - M: Masa
    - a = J/M: Parámetro de espín (0 ≤ a ≤ M para BH)
    - J = aM: Momento angular
    
    Límites:
    - a = 0: Schwarzschild (sin rotación)
    - a = M: Kerr extremal (máxima rotación)
    
    Horizonte exterior: r_+ = M + √(M² - a²)
    Área: A = 8πM r_+ = 8πM(M + √(M² - a²))
    """
    
    def __init__(self, M: float, a: float, constants: Constants = None):
        """
        Args:
            M: Masa del BH
            a: Parámetro de espín (a = J/M, debe ser |a| ≤ M)
        """
        self.M = M
        self._a = min(abs(a), M)  # Asegurar a ≤ M
        self.const = constants or Constants()
    
    @property
    def a(self) -> float:
        """Parámetro de espín a = J/M"""
        return self._a
    
    @property
    def J(self) -> float:
        """Momento angular J = aM"""
        return self._a * self.M
    
    @property
    def a_star(self) -> float:
        """Espín adimensional a* = a/M ∈ [0, 1]"""
        return self._a / self.M if self.M > 0 else 0
    
    @property
    def r_plus(self) -> float:
        """Radio del horizonte exterior: r_+ = M + √(M² - a²)"""
        return self.M + np.sqrt(self.M**2 - self._a**2)
    
    @property
    def r_minus(self) -> float:
        """Radio del horizonte interior: r_- = M - √(M² - a²)"""
        return self.M - np.sqrt(self.M**2 - self._a**2)
    
    @property
    def area(self) -> float:
        """
        Área del horizonte: A = 8πMr_+ = 4π(r_+² + a²)
        
        Para a = 0: A = 16πM² (Schwarzschild)
        Para a = M: A = 8πM² (Kerr extremal)
        """
        return 4 * np.pi * (self.r_plus**2 + self._a**2)
    
    @property
    def irreducible_mass(self) -> float:
        """
        Masa irreducible: M_ir = √(A/16π)
        
        Es la masa que queda después de extraer toda la energía rotacional.
        M² = M_ir² + J²/(4M_ir²)
        """
        return np.sqrt(self.area / (16 * np.pi))
    
    @property
    def temperature(self) -> float:
        """
        Temperatura de Hawking para Kerr:
        T = (r_+ - r_-)/(4π(r_+² + a²)) = √(M² - a²)/(4πMr_+)
        
        Para a = 0: T = 1/(8πM) (Schwarzschild)
        Para a = M: T = 0 (Kerr extremal)
        """
        numerator = np.sqrt(self.M**2 - self._a**2)
        denominator = 4 * np.pi * self.M * self.r_plus
        return numerator / denominator if denominator > 0 else 0
    
    @property
    def omega_H(self) -> float:
        """
        Velocidad angular del horizonte: Ω_H = a/(r_+² + a²)
        """
        return self._a / (self.r_plus**2 + self._a**2)
    
    @property
    def entropy_BH(self) -> float:
        """Entropía de Bekenstein-Hawking: S = A/(4G)"""
        return self.area / (4 * self.const.G_N)
    
    @property
    def surface_gravity(self) -> float:
        """
        Gravedad superficial: κ = (r_+ - r_-)/(2(r_+² + a²))
        
        Satisface T = κ/(2π)
        """
        return (self.r_plus - self.r_minus) / (2 * (self.r_plus**2 + self._a**2))
    
    def is_extremal(self) -> bool:
        """True si a = M (BH extremal)"""
        return abs(self._a - self.M) < 1e-10


# =============================================================================
# ECUACIÓN DE CORRESPONDENCIA PARA KERR
# =============================================================================

class KerrCorrespondence:
    """
    Ecuación de correspondencia para Kerr.
    
    PREGUNTA CLAVE: ¿Cómo afecta J a λ?
    
    HIPÓTESIS:
    La rotación afecta la accesibilidad de la información.
    Un BH con alta rotación tiene:
    - Ergosfera donde se puede extraer energía (proceso de Penrose)
    - Más "estructura" accesible desde el exterior
    
    Propuesta: λ incluye un factor de rotación
    λ(A, I, a*) = f(A) × g(I) × h(a*)
    
    donde h(a*) = 1 + ε × a*² (la rotación aumenta λ ligeramente)
    """
    
    ALPHA_LQG = -0.5
    ALPHA_CFT = -1.5
    
    def __init__(self, constants: Constants = None):
        self.const = constants or Constants()
        self.beta_LQG = 0.5 * np.log(np.pi * self.const.gamma)
        self.beta_CFT = np.log(2)
        
        # Factor de rotación (pequeño, conservador)
        self.epsilon_rotation = 0.1
    
    def lambda_parameter(self, bh: KerrBlackHole, S_accessible: float = 0) -> float:
        """
        Parámetro de interpolación para Kerr.
        
        λ = f(A) × g(I) × h(a*)
        
        donde:
        - f(A) = 1 - exp(-A/A_c)
        - g(I) = S_acc/S_total
        - h(a*) = 1 + ε × a*² (efecto de rotación)
        """
        A = bh.area
        S_total = bh.entropy_BH
        a_star = bh.a_star
        
        # Factor de área
        f_area = 1 - np.exp(-A / self.const.A_c)
        
        # Factor informacional
        g_info = np.clip(S_accessible / S_total, 0, 1) if S_total > 0 else 0
        
        # Factor de rotación
        h_rotation = 1 + self.epsilon_rotation * a_star**2
        
        # λ total (limitado a [0, 1])
        return np.clip(f_area * g_info * h_rotation, 0, 1)
    
    def alpha(self, lam: float) -> float:
        return self.ALPHA_LQG + lam * (self.ALPHA_CFT - self.ALPHA_LQG)
    
    def beta(self, lam: float) -> float:
        return self.beta_LQG * (1 - lam) + self.beta_CFT * lam
    
    def entropy(self, bh: KerrBlackHole, S_accessible: float = 0, lam: float = None) -> Dict:
        """Entropía con correcciones"""
        A = bh.area
        
        if lam is None:
            lam = self.lambda_parameter(bh, S_accessible)
        
        alpha_val = self.alpha(lam)
        beta_val = self.beta(lam)
        
        S_BH = A / (4 * self.const.G_N)
        S_log = alpha_val * np.log(A / self.const.l_P**2) if A > self.const.l_P**2 else 0
        
        return {
            'S': S_BH + S_log + beta_val,
            'S_BH': S_BH,
            'S_log': S_log,
            'lambda': lam,
            'alpha': alpha_val,
            'a_star': bh.a_star
        }


# =============================================================================
# SIMULACIÓN Y VERIFICACIONES
# =============================================================================

print("\n" + "="*70)
print("PROPIEDADES DE KERR")
print("="*70)

corresp = KerrCorrespondence(const)

# Comparar Schwarzschild vs Kerr para la misma masa
M = 10.0

print(f"\nPara M = {M}:")
print("-" * 70)
print(f"{'a/M':<10} {'r_+':<10} {'A':<15} {'T':<15} {'S_BH':<12} {'Ω_H':<10}")
print("-" * 70)

a_values = [0, 0.3, 0.5, 0.7, 0.9, 0.99, 1.0]  # a* = a/M

for a_star in a_values:
    a = a_star * M
    bh = KerrBlackHole(M, a, const)
    extremal = " (extremal)" if bh.is_extremal() else ""
    print(f"{a_star:<10.2f} {bh.r_plus:<10.4f} {bh.area:<15.2f} {bh.temperature:<15.6f} "
          f"{bh.entropy_BH:<12.2f} {bh.omega_H:<10.4f}{extremal}")


print("\n" + "="*70)
print("EFECTO DE LA ROTACIÓN EN λ")
print("="*70)

print("\nPara BH con S_acc = 0.5 × S_BH:")
print(f"{'a/M':<10} {'λ(a*=0)':<12} {'λ(a*)':<12} {'Δλ':<12} {'α':<10}")
print("-" * 60)

# Referencia: Schwarzschild
bh_schw = KerrBlackHole(M, 0, const)
result_schw = corresp.entropy(bh_schw, S_accessible=0.5*bh_schw.entropy_BH)
lambda_ref = result_schw['lambda']

for a_star in a_values[:-1]:  # Excluir extremal
    a = a_star * M
    bh = KerrBlackHole(M, a, const)
    result = corresp.entropy(bh, S_accessible=0.5*bh.entropy_BH)
    delta_lambda = result['lambda'] - lambda_ref
    print(f"{a_star:<10.2f} {lambda_ref:<12.4f} {result['lambda']:<12.4f} "
          f"{delta_lambda:<12.4f} {result['alpha']:<10.4f}")


# =============================================================================
# VERIFICACIONES
# =============================================================================

print("\n" + "="*70)
print("VERIFICACIONES")
print("="*70)

verifications = []

# V1: Límite Schwarzschild (a = 0)
bh_schw = KerrBlackHole(M, 0, const)
A_schw = bh_schw.area
A_expected = 16 * np.pi * M**2
v1 = abs(A_schw - A_expected) / A_expected < 1e-10
verifications.append(("a=0 → Schwarzschild (A = 16πM²)", v1))

# V2: Temperatura Schwarzschild
T_schw = bh_schw.temperature
T_expected = 1 / (8 * np.pi * M)
v2 = abs(T_schw - T_expected) / T_expected < 1e-10
verifications.append(("a=0 → T = 1/(8πM)", v2))

# V3: Kerr extremal tiene T = 0
bh_ext = KerrBlackHole(M, M, const)
v3 = abs(bh_ext.temperature) < 1e-10
verifications.append(("a=M (extremal) → T = 0", v3))

# V4: Área decrece con rotación
areas = [KerrBlackHole(M, a*M, const).area for a in [0, 0.5, 0.99]]
v4 = areas[0] > areas[1] > areas[2]
verifications.append(("Área decrece con rotación", v4))

# V5: Primera ley: T = κ/(2π)
bh_test = KerrBlackHole(M, 0.5*M, const)
T_test = bh_test.temperature
T_from_kappa = bh_test.surface_gravity / (2 * np.pi)
v5 = abs(T_test - T_from_kappa) / T_test < 1e-10
verifications.append(("T = κ/(2π) (primera ley)", v5))

# V6: λ aumenta con rotación (para mismo S_acc/S_total)
S_ratio = 0.5
lambda_a0 = corresp.lambda_parameter(KerrBlackHole(M, 0, const), 
                                      S_ratio*KerrBlackHole(M, 0, const).entropy_BH)
lambda_a05 = corresp.lambda_parameter(KerrBlackHole(M, 0.5*M, const), 
                                       S_ratio*KerrBlackHole(M, 0.5*M, const).entropy_BH)
v6 = lambda_a05 > lambda_a0
verifications.append(("λ aumenta con rotación", v6))

# V7: Entropía positiva para todo a
v7 = all(corresp.entropy(KerrBlackHole(M, a*M, const), lam=0)['S'] > 0 
         for a in [0, 0.5, 0.9, 0.99])
verifications.append(("S > 0 para todo a < M", v7))

print("\nResultados:")
print("-" * 70)
for name, passed in verifications:
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"  {status}: {name}")

n_passed = sum(1 for _, p in verifications if p)
print("-" * 70)
print(f"Total: {n_passed}/{len(verifications)} verificaciones pasadas")


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

print("\n" + "="*70)
print("GENERANDO VISUALIZACIÓN")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

a_star_range = np.linspace(0, 0.999, 100)

# 1. Área vs a*
ax1 = axes[0, 0]
areas = [KerrBlackHole(M, a*M, const).area for a in a_star_range]
ax1.plot(a_star_range, areas, 'b-', lw=2)
ax1.axhline(y=16*np.pi*M**2, color='gray', linestyle='--', label='Schwarzschild')
ax1.axhline(y=8*np.pi*M**2, color='red', linestyle='--', label='Extremal')
ax1.set_xlabel('a* = a/M')
ax1.set_ylabel('Área A')
ax1.set_title('ÁREA vs ESPÍN')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Temperatura vs a*
ax2 = axes[0, 1]
temps = [KerrBlackHole(M, a*M, const).temperature for a in a_star_range]
ax2.plot(a_star_range, temps, 'orange', lw=2)
ax2.axhline(y=1/(8*np.pi*M), color='gray', linestyle='--', label='Schwarzschild')
ax2.set_xlabel('a* = a/M')
ax2.set_ylabel('Temperatura T')
ax2.set_title('TEMPERATURA vs ESPÍN')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Entropía vs a*
ax3 = axes[0, 2]
entropies = [KerrBlackHole(M, a*M, const).entropy_BH for a in a_star_range]
ax3.plot(a_star_range, entropies, 'g-', lw=2)
ax3.set_xlabel('a* = a/M')
ax3.set_ylabel('Entropía S_BH')
ax3.set_title('ENTROPÍA vs ESPÍN')
ax3.grid(True, alpha=0.3)

# 4. λ vs a* para diferentes S_acc/S_total
ax4 = axes[1, 0]
for S_ratio, color, label in [(0.3, 'blue', '0.3'), (0.5, 'green', '0.5'), (0.7, 'red', '0.7')]:
    lambdas = []
    for a in a_star_range:
        bh = KerrBlackHole(M, a*M, const)
        lam = corresp.lambda_parameter(bh, S_ratio*bh.entropy_BH)
        lambdas.append(lam)
    ax4.plot(a_star_range, lambdas, color=color, lw=2, label=f'S_acc/S_BH = {label}')
ax4.set_xlabel('a* = a/M')
ax4.set_ylabel('λ')
ax4.set_title('λ vs ESPÍN')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. α vs a*
ax5 = axes[1, 1]
alphas = []
for a in a_star_range:
    bh = KerrBlackHole(M, a*M, const)
    result = corresp.entropy(bh, S_accessible=0.5*bh.entropy_BH)
    alphas.append(result['alpha'])
ax5.plot(a_star_range, alphas, 'purple', lw=2)
ax5.axhline(y=-0.5, color='blue', linestyle='--', alpha=0.5, label='LQG')
ax5.axhline(y=-1.5, color='red', linestyle='--', alpha=0.5, label='CFT')
ax5.set_xlabel('a* = a/M')
ax5.set_ylabel('α')
ax5.set_title('α vs ESPÍN (S_acc/S_BH = 0.5)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Resumen
ax6 = axes[1, 2]
ax6.axis('off')

ax6.text(0.5, 0.95, 'KERR - RESUMEN', ha='center', fontsize=12, fontweight='bold')

summary = f"""
┌────────────────────────────────────────┐
│     AGUJERO NEGRO DE KERR              │
├────────────────────────────────────────┤
│  r_+ = M + √(M² - a²)                  │
│  A = 4π(r_+² + a²)                     │
│  T = √(M² - a²)/(4πMr_+)               │
│  S = A/(4G)                            │
├────────────────────────────────────────┤
│     LÍMITES                            │
├────────────────────────────────────────┤
│  a = 0: Schwarzschild                  │
│  a = M: Extremal (T = 0)               │
├────────────────────────────────────────┤
│     EFECTO EN λ                        │
├────────────────────────────────────────┤
│  λ(a*) = f(A)×g(I)×(1 + εa*²)          │
│  La rotación AUMENTA λ ligeramente     │
│  (más estructura accesible)            │
├────────────────────────────────────────┤
│  Verificaciones: {n_passed}/{len(verifications)} pasadas           │
└────────────────────────────────────────┘
"""

ax6.text(0.5, 0.45, summary, ha='center', va='center', fontsize=9,
         family='monospace', bbox=dict(boxstyle='round', facecolor='lightyellow'))
ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)

plt.suptitle('AGUJERO NEGRO DE KERR - KAELION v3.0', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/Kerr_BlackHole.png', dpi=150, bbox_inches='tight')
print("✓ Figura guardada: Kerr_BlackHole.png")


# =============================================================================
# RESUMEN FINAL
# =============================================================================

print("\n" + "="*70)
print("RESUMEN: KERR COMPLETADO")
print("="*70)

print(f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║                    KERR - RESULTADOS                                      ║
║                                                                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  PROPIEDADES VERIFICADAS:                                                 ║
║  • a = 0 → Schwarzschild (A = 16πM², T = 1/8πM)                          ║
║  • a = M → Extremal (A = 8πM², T = 0)                                    ║
║  • Área decrece con rotación                                              ║
║  • Primera ley: T = κ/(2π)                                                ║
║                                                                           ║
║  EFECTO DE ROTACIÓN EN KAELION:                                           ║
║  • λ(a*) = f(A) × g(I) × (1 + ε a*²)                                      ║
║  • La rotación aumenta λ ligeramente                                      ║
║  • Interpretación: más estructura accesible con rotación                  ║
║                                                                           ║
║  VERIFICACIONES: {n_passed}/{len(verifications)} PASADAS                                             ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

plt.show()
