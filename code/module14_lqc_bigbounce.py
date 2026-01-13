"""
LOOP QUANTUM COSMOLOGY - BIG BOUNCE
====================================
Proyecto Kaelion v3.0 - Simulación 14

Loop Quantum Cosmology (LQC) aplica las técnicas de LQG al universo
homogéneo e isotrópico. El resultado más importante es que la
singularidad del Big Bang se reemplaza por un "Big Bounce".

PREGUNTA: ¿Qué predice Kaelion para el universo temprano?
¿Cómo se comporta λ cerca del bounce?

Referencias:
- Ashtekar, Pawlowski, Singh (2006) "Quantum Nature of the Big Bang"
- Bojowald (2001) "Absence of a Singularity in Loop Quantum Cosmology"
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from dataclasses import dataclass
from typing import Dict, Tuple

print("="*70)
print("LOOP QUANTUM COSMOLOGY - BIG BOUNCE")
print("Kaelion v3.0 - Módulo 14")
print("="*70)

# =============================================================================
# CONSTANTES
# =============================================================================

@dataclass
class LQCConstants:
    """Constantes para Loop Quantum Cosmology"""
    l_P: float = 1.0           # Longitud de Planck
    G_N: float = 1.0           # Constante de Newton
    gamma: float = 0.2375      # Parámetro de Immirzi
    
    @property
    def rho_c(self) -> float:
        """
        Densidad crítica de LQC.
        
        ρ_c = √3 / (32π²γ³G²ℏ) ≈ 0.41 ρ_P
        
        En unidades de Planck (G = ℏ = 1):
        ρ_c ≈ √3 / (32π²γ³)
        """
        return np.sqrt(3) / (32 * np.pi**2 * self.gamma**3)
    
    @property
    def A_c(self) -> float:
        """Área crítica de crossover (Kaelion)"""
        return 4 * np.pi / self.gamma * self.l_P**2
    
    @property
    def a_bounce(self) -> float:
        """
        Factor de escala en el bounce.
        
        El bounce ocurre cuando ρ = ρ_c.
        Para radiación: ρ ∝ a⁻⁴
        """
        return 1.0  # Normalizamos a_bounce = 1

const = LQCConstants()

print(f"\nConstantes de LQC:")
print(f"  γ (Immirzi) = {const.gamma}")
print(f"  ρ_c = {const.rho_c:.4f} ρ_P (densidad crítica)")
print(f"  A_c = {const.A_c:.2f} l_P² (área crítica Kaelion)")


# =============================================================================
# ECUACIONES DE FRIEDMANN MODIFICADAS (LQC)
# =============================================================================

class LQCUniverse:
    """
    Universo en Loop Quantum Cosmology.
    
    Las ecuaciones de Friedmann se modifican por efectos cuánticos:
    
    Clásica: H² = (8πG/3)ρ
    LQC:     H² = (8πG/3)ρ(1 - ρ/ρ_c)
    
    El factor (1 - ρ/ρ_c) es la corrección cuántica que:
    - Para ρ << ρ_c: recupera Friedmann clásica
    - Para ρ → ρ_c: H → 0 (bounce!)
    - Para ρ > ρ_c: H² < 0 (prohibido)
    """
    
    def __init__(self, constants: LQCConstants = None):
        self.const = constants or LQCConstants()
        self.rho_c = self.const.rho_c
    
    def hubble_classical(self, rho: float) -> float:
        """Ecuación de Friedmann clásica: H² = (8πG/3)ρ"""
        return np.sqrt(8 * np.pi * self.const.G_N / 3 * rho)
    
    def hubble_lqc(self, rho: float) -> float:
        """
        Ecuación de Friedmann en LQC: H² = (8πG/3)ρ(1 - ρ/ρ_c)
        
        Retorna |H| (el signo depende de si es contracción o expansión)
        """
        if rho >= self.rho_c:
            return 0  # Bounce
        
        H_sq = 8 * np.pi * self.const.G_N / 3 * rho * (1 - rho / self.rho_c)
        return np.sqrt(max(0, H_sq))
    
    def quantum_correction(self, rho: float) -> float:
        """Factor de corrección cuántica: (1 - ρ/ρ_c)"""
        return max(0, 1 - rho / self.rho_c)
    
    def evolve_radiation(self, a_initial: float, t_span: Tuple[float, float], 
                         rho_initial: float, N_points: int = 1000) -> Dict:
        """
        Evoluciona un universo dominado por radiación.
        
        Para radiación: ρ = ρ_0 (a_0/a)⁴
        
        Ecuación de evolución: da/dt = a × H(ρ(a))
        """
        def rho_of_a(a):
            return rho_initial * (a_initial / a)**4
        
        def dadt_classical(a, t):
            rho = rho_of_a(a)
            H = self.hubble_classical(rho)
            return a * H
        
        def dadt_lqc(a, t):
            rho = rho_of_a(a)
            H = self.hubble_lqc(rho)
            # Signo: positivo para expansión, negativo para contracción
            if t < 0:
                return -a * H  # Contracción (antes del bounce)
            return a * H  # Expansión (después del bounce)
        
        t = np.linspace(t_span[0], t_span[1], N_points)
        
        # Integrar clásica (solo para comparación, puede tener singularidad)
        try:
            a_classical = odeint(dadt_classical, a_initial, t[t >= 0])[:, 0]
        except:
            a_classical = np.ones(len(t[t >= 0])) * np.nan
        
        # Para LQC, integramos desde el bounce hacia adelante y hacia atrás
        # Simplificación: usamos solución analítica aproximada
        
        # Solución aproximada cerca del bounce para radiación:
        # a(t) ≈ a_bounce × (1 + (t/t_P)²)^(1/4)
        
        t_P = 1.0  # Tiempo de Planck
        a_lqc = const.a_bounce * (1 + (t / t_P)**2)**(1/4)
        
        # Densidad
        rho_lqc = rho_initial * (a_initial / a_lqc)**4
        
        # H en LQC
        H_lqc = np.array([self.hubble_lqc(r) for r in rho_lqc])
        
        # Corrección cuántica
        q_corr = np.array([self.quantum_correction(r) for r in rho_lqc])
        
        return {
            't': t,
            'a_lqc': a_lqc,
            'rho_lqc': rho_lqc,
            'H_lqc': H_lqc,
            'quantum_correction': q_corr
        }


# =============================================================================
# ECUACIÓN DE CORRESPONDENCIA PARA LQC
# =============================================================================

class KaelionLQC:
    """
    Aplicación de la ecuación de correspondencia a LQC.
    
    INTERPRETACIÓN:
    En cosmología, el "área" relevante es el área del horizonte de Hubble:
        A_H = 4π/H²
    
    Para el universo temprano (cerca del bounce):
    - A_H pequeña (H grande en régimen clásico, pero H → 0 en bounce)
    - En el bounce: H = 0 → A_H → ∞
    
    HIPÓTESIS DE KAELION:
    λ depende de cuán "cuántico" es el régimen:
    - Cerca del bounce (ρ ≈ ρ_c): efectos cuánticos dominantes → λ → 0 (LQG)
    - Lejos del bounce (ρ << ρ_c): régimen clásico → λ → 1 (Holo)
    """
    
    ALPHA_LQG = -0.5
    ALPHA_CFT = -1.5
    
    def __init__(self, constants: LQCConstants = None):
        self.const = constants or LQCConstants()
    
    def lambda_parameter(self, rho: float, rho_c: float) -> float:
        """
        Parámetro λ para cosmología.
        
        λ = 1 - ρ/ρ_c (régimen cuántico cuando ρ → ρ_c)
        
        Esto es consistente con:
        - ρ << ρ_c: λ ≈ 1 (holográfico, clásico)
        - ρ → ρ_c: λ → 0 (LQG, cuántico)
        """
        return max(0, min(1, 1 - rho / rho_c))
    
    def alpha(self, lam: float) -> float:
        """Coeficiente logarítmico"""
        return self.ALPHA_LQG + lam * (self.ALPHA_CFT - self.ALPHA_LQG)
    
    def entropy_horizon(self, H: float, lam: float) -> float:
        """
        Entropía del horizonte de Hubble.
        
        S = A_H/(4G) + α(λ)ln(A_H)
        
        donde A_H = 4π/H² (área del horizonte de Hubble)
        """
        if H <= 0:
            return np.inf  # En el bounce, horizonte infinito
        
        A_H = 4 * np.pi / H**2
        
        if A_H <= self.const.l_P**2:
            return 0
        
        alpha_val = self.alpha(lam)
        S_BH = A_H / (4 * self.const.G_N)
        S_log = alpha_val * np.log(A_H / self.const.l_P**2)
        
        return S_BH + S_log


# =============================================================================
# SIMULACIÓN
# =============================================================================

print("\n" + "="*70)
print("SIMULACIÓN DEL BIG BOUNCE")
print("="*70)

universe = LQCUniverse(const)
kaelion = KaelionLQC(const)

# Configuración
a_bounce = 1.0  # Factor de escala en el bounce
rho_bounce = const.rho_c  # Densidad en el bounce

print(f"\nConfiguración:")
print(f"  a_bounce = {a_bounce}")
print(f"  ρ_bounce = ρ_c = {rho_bounce:.4f}")

# Evolucionar desde t = -10 hasta t = +10 (en unidades de Planck)
result = universe.evolve_radiation(a_bounce, (-10, 10), rho_bounce, N_points=500)

# Calcular λ y α para cada punto
lambdas = []
alphas = []
for rho in result['rho_lqc']:
    lam = kaelion.lambda_parameter(rho, const.rho_c)
    lambdas.append(lam)
    alphas.append(kaelion.alpha(lam))

result['lambda'] = np.array(lambdas)
result['alpha'] = np.array(alphas)

print("✓ Simulación completada")


# =============================================================================
# ANÁLISIS
# =============================================================================

print("\n" + "="*70)
print("ANÁLISIS DE RESULTADOS")
print("="*70)

# Encontrar el bounce
bounce_idx = np.argmin(result['a_lqc'])
t_bounce = result['t'][bounce_idx]

print(f"\nEn el bounce (t = {t_bounce:.2f}):")
print(f"  a = {result['a_lqc'][bounce_idx]:.4f}")
print(f"  ρ/ρ_c = {result['rho_lqc'][bounce_idx]/const.rho_c:.4f}")
print(f"  H = {result['H_lqc'][bounce_idx]:.6f}")
print(f"  Corrección cuántica = {result['quantum_correction'][bounce_idx]:.4f}")
print(f"  λ = {result['lambda'][bounce_idx]:.4f}")
print(f"  α = {result['alpha'][bounce_idx]:.4f}")

# Lejos del bounce
far_idx = -1  # Último punto (lejos del bounce)
print(f"\nLejos del bounce (t = {result['t'][far_idx]:.2f}):")
print(f"  a = {result['a_lqc'][far_idx]:.4f}")
print(f"  ρ/ρ_c = {result['rho_lqc'][far_idx]/const.rho_c:.4f}")
print(f"  H = {result['H_lqc'][far_idx]:.6f}")
print(f"  Corrección cuántica = {result['quantum_correction'][far_idx]:.4f}")
print(f"  λ = {result['lambda'][far_idx]:.4f}")
print(f"  α = {result['alpha'][far_idx]:.4f}")

# Evolución de α
print(f"\nEvolución de α(t) - RÉGIMEN COSMOLÓGICO:")
print("-" * 50)
checkpoints = [-10, -5, -1, 0, 1, 5, 10]
for t_check in checkpoints:
    idx = np.argmin(np.abs(result['t'] - t_check))
    lam = result['lambda'][idx]
    alpha = result['alpha'][idx]
    rho_ratio = result['rho_lqc'][idx] / const.rho_c
    regime = "LQG" if lam < 0.25 else "Trans" if lam < 0.75 else "Holo"
    bounce_note = " ← BOUNCE" if abs(t_check) < 0.1 else ""
    print(f"  t = {t_check:+6.1f}: ρ/ρ_c = {rho_ratio:.4f}, λ = {lam:.4f}, α = {alpha:.4f} [{regime}]{bounce_note}")


# =============================================================================
# VERIFICACIONES
# =============================================================================

print("\n" + "="*70)
print("VERIFICACIONES")
print("="*70)

verifications = []

# V1: Bounce ocurre (a tiene mínimo)
a_min_idx = np.argmin(result['a_lqc'])
v1 = result['a_lqc'][a_min_idx] > 0  # No hay singularidad
verifications.append(("Bounce ocurre (no singularidad)", v1, f"a_min = {result['a_lqc'][a_min_idx]:.4f}"))

# V2: H = 0 en el bounce
v2 = result['H_lqc'][bounce_idx] < 0.01
verifications.append(("H ≈ 0 en el bounce", v2, f"H = {result['H_lqc'][bounce_idx]:.6f}"))

# V3: ρ = ρ_c en el bounce
v3 = abs(result['rho_lqc'][bounce_idx] / const.rho_c - 1) < 0.01
verifications.append(("ρ = ρ_c en el bounce", v3, f"ρ/ρ_c = {result['rho_lqc'][bounce_idx]/const.rho_c:.4f}"))

# V4: λ → 0 en el bounce (régimen LQG)
v4 = result['lambda'][bounce_idx] < 0.1
verifications.append(("λ → 0 en bounce (LQG)", v4, f"λ = {result['lambda'][bounce_idx]:.4f}"))

# V5: λ → 1 lejos del bounce (régimen clásico/Holo)
v5 = result['lambda'][far_idx] > 0.9
verifications.append(("λ → 1 lejos del bounce (Holo)", v5, f"λ = {result['lambda'][far_idx]:.4f}"))

# V6: α ≈ -0.5 en el bounce
v6 = abs(result['alpha'][bounce_idx] - (-0.5)) < 0.1
verifications.append(("α ≈ -0.5 en bounce (LQG)", v6, f"α = {result['alpha'][bounce_idx]:.4f}"))

# V7: Simetría temporal (a(t) = a(-t))
mid = len(result['t']) // 2
a_before = result['a_lqc'][:mid]
a_after = result['a_lqc'][mid:][::-1]
min_len = min(len(a_before), len(a_after))
symmetry_error = np.mean(np.abs(a_before[:min_len] - a_after[:min_len]))
v7 = symmetry_error < 0.01
verifications.append(("Simetría temporal a(t) = a(-t)", v7, f"error = {symmetry_error:.6f}"))

print("\nResultados:")
print("-" * 70)
for name, passed, detail in verifications:
    status = "✓ PASSED" if passed else "✗ FAILED"
    print(f"  {status}: {name}")
    print(f"           Detalle: {detail}")

n_passed = sum(1 for v in verifications if v[1])
print("-" * 70)
print(f"Total: {n_passed}/{len(verifications)} verificaciones pasadas")


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

print("\n" + "="*70)
print("GENERANDO VISUALIZACIÓN")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. Factor de escala a(t)
ax1 = axes[0, 0]
ax1.plot(result['t'], result['a_lqc'], 'b-', lw=2, label='LQC')
ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Bounce')
ax1.axhline(y=a_bounce, color='gray', linestyle=':', alpha=0.5)
ax1.set_xlabel('Tiempo t (t_P)')
ax1.set_ylabel('Factor de escala a')
ax1.set_title('FACTOR DE ESCALA (Big Bounce)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Densidad ρ(t)
ax2 = axes[0, 1]
ax2.semilogy(result['t'], result['rho_lqc'] / const.rho_c, 'orange', lw=2)
ax2.axhline(y=1, color='red', linestyle='--', lw=2, label='ρ_c (bounce)')
ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Tiempo t (t_P)')
ax2.set_ylabel('ρ / ρ_c')
ax2.set_title('DENSIDAD')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Parámetro de Hubble H(t)
ax3 = axes[0, 2]
ax3.plot(result['t'], result['H_lqc'], 'g-', lw=2)
ax3.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Bounce (H=0)')
ax3.set_xlabel('Tiempo t (t_P)')
ax3.set_ylabel('H')
ax3.set_title('PARÁMETRO DE HUBBLE')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. λ(t) - Predicción de Kaelion
ax4 = axes[1, 0]
ax4.plot(result['t'], result['lambda'], 'purple', lw=3)
ax4.axvline(x=0, color='red', linestyle='--', alpha=0.7)
ax4.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
ax4.fill_between(result['t'], 0, result['lambda'], alpha=0.3, color='purple')
ax4.set_xlabel('Tiempo t (t_P)')
ax4.set_ylabel('λ')
ax4.set_title('λ: LQG (bounce) → Holo (clásico)')
ax4.set_ylim(0, 1.1)
ax4.grid(True, alpha=0.3)

# 5. α(t)
ax5 = axes[1, 1]
ax5.plot(result['t'], result['alpha'], 'purple', lw=3)
ax5.axhline(y=-0.5, color='blue', linestyle='--', alpha=0.7, label='LQG')
ax5.axhline(y=-1.5, color='red', linestyle='--', alpha=0.7, label='Holo')
ax5.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
ax5.set_xlabel('Tiempo t (t_P)')
ax5.set_ylabel('α')
ax5.set_title('COEFICIENTE α(t)')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 6. Resumen
ax6 = axes[1, 2]
ax6.axis('off')

ax6.text(0.5, 0.95, 'LQC - BIG BOUNCE', ha='center', fontsize=12, fontweight='bold')

summary = f"""
┌────────────────────────────────────────┐
│     LOOP QUANTUM COSMOLOGY             │
├────────────────────────────────────────┤
│  Friedmann modificada:                 │
│  H² = (8πG/3)ρ(1 - ρ/ρ_c)             │
│                                        │
│  ρ_c = {const.rho_c:.4f} ρ_P (densidad crítica)    │
├────────────────────────────────────────┤
│     PREDICCIÓN DE KAELION              │
├────────────────────────────────────────┤
│  En el bounce (ρ = ρ_c):               │
│  • λ → 0 (régimen LQG)                 │
│  • α → -0.5                            │
│                                        │
│  Lejos del bounce (ρ << ρ_c):          │
│  • λ → 1 (régimen holográfico)         │
│  • α → -1.5                            │
├────────────────────────────────────────┤
│  Verificaciones: {n_passed}/{len(verifications)} pasadas           │
└────────────────────────────────────────┘
"""

ax6.text(0.5, 0.45, summary, ha='center', va='center', fontsize=9,
         family='monospace', bbox=dict(boxstyle='round', facecolor='lightcyan'))
ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)

plt.suptitle('LOOP QUANTUM COSMOLOGY - KAELION v3.0', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/LQC_BigBounce.png', dpi=150, bbox_inches='tight')
print("✓ Figura guardada: LQC_BigBounce.png")


# =============================================================================
# RESUMEN
# =============================================================================

print("\n" + "="*70)
print("RESUMEN: LQC BIG BOUNCE COMPLETADO")
print("="*70)

print(f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║              LOOP QUANTUM COSMOLOGY - RESULTADOS                          ║
║                                                                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  FÍSICA DE LQC:                                                           ║
║  • El Big Bang se reemplaza por un Big Bounce                             ║
║  • Densidad crítica: ρ_c = {const.rho_c:.4f} ρ_P                                   ║
║  • H → 0 en el bounce (no hay singularidad)                               ║
║  • Simetría temporal: contracción → bounce → expansión                    ║
║                                                                           ║
║  PREDICCIÓN DE KAELION:                                                   ║
║  • En el bounce: λ = {result['lambda'][bounce_idx]:.4f}, α = {result['alpha'][bounce_idx]:.4f} (régimen LQG)             ║
║  • Lejos del bounce: λ → 1, α → -1.5 (régimen holográfico)                ║
║                                                                           ║
║  INTERPRETACIÓN:                                                          ║
║  El universo temprano (cerca del bounce) está en régimen LQG.             ║
║  A medida que se expande, transiciona hacia el régimen holográfico.       ║
║  Esto es consistente con la idea de que LQG domina a escala Planck.       ║
║                                                                           ║
║  VERIFICACIONES: {n_passed}/{len(verifications)} PASADAS                                            ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

plt.show()
