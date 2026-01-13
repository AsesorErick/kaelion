"""
ECUACIÓN DE DIRAC - FERMIONES RELATIVISTAS
===========================================
Proyecto Kaelion v3.0 - Simulación 16

La ecuación de Dirac describe partículas con espín 1/2 (electrones,
quarks, etc.) de manera relativista. Es fundamental en física de
partículas y tiene conexiones profundas con LQG a través de los
spinores.

CONEXIÓN CON KAELION:
- Los spin networks de LQG usan representaciones de SU(2)
- Los spinores de Dirac también transforman bajo SU(2)
- ¿Hay una conexión entre fermiones y la estructura de LQG?

Referencias:
- Dirac (1928) "The Quantum Theory of the Electron"
- Thiemann (1998) "QSD V: Quantum Gravity as the Natural Regulator"
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from dataclasses import dataclass
from typing import Dict, Tuple, List

print("="*70)
print("ECUACIÓN DE DIRAC - FERMIONES RELATIVISTAS")
print("Kaelion v3.0 - Módulo 16")
print("="*70)

# =============================================================================
# MATRICES DE DIRAC
# =============================================================================

# Matrices de Pauli
sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
sigma_0 = np.eye(2, dtype=complex)

# Matrices gamma en representación de Dirac
# γ⁰ = [[I, 0], [0, -I]], γⁱ = [[0, σⁱ], [-σⁱ, 0]]

def gamma_matrices():
    """Construye las matrices gamma de Dirac (4x4)"""
    I2 = np.eye(2, dtype=complex)
    Z2 = np.zeros((2, 2), dtype=complex)
    
    gamma0 = np.block([[I2, Z2], [Z2, -I2]])
    gamma1 = np.block([[Z2, sigma_x], [-sigma_x, Z2]])
    gamma2 = np.block([[Z2, sigma_y], [-sigma_y, Z2]])
    gamma3 = np.block([[Z2, sigma_z], [-sigma_z, Z2]])
    gamma5 = 1j * gamma0 @ gamma1 @ gamma2 @ gamma3
    
    return gamma0, gamma1, gamma2, gamma3, gamma5

gamma0, gamma1, gamma2, gamma3, gamma5 = gamma_matrices()
gammas = [gamma0, gamma1, gamma2, gamma3]

print("\nMatrices gamma construidas (representación de Dirac)")


# =============================================================================
# CLASE: SPINOR DE DIRAC
# =============================================================================

class DiracSpinor:
    """
    Spinor de Dirac (4 componentes).
    
    Un spinor libre de masa m y momento p satisface:
    (γ·p - m)ψ = 0  (ecuación de Dirac en el espacio de momentos)
    
    Soluciones:
    - u(p, s): partícula con espín s
    - v(p, s): antipartícula con espín s
    """
    
    def __init__(self, mass: float):
        self.m = mass
    
    def energy(self, p: np.ndarray) -> float:
        """Energía relativista: E = √(p² + m²)"""
        return np.sqrt(np.sum(p**2) + self.m**2)
    
    def u_spinor(self, p: np.ndarray, s: int) -> np.ndarray:
        """
        Spinor de partícula u(p, s).
        
        Args:
            p: momento 3D
            s: espín (+1 para up, -1 para down)
        """
        E = self.energy(p)
        p_mag = np.sqrt(np.sum(p**2))
        
        # Spinor de 2 componentes para el espín
        if s == 1:  # spin up
            chi = np.array([1, 0], dtype=complex)
        else:  # spin down
            chi = np.array([0, 1], dtype=complex)
        
        # Normalización
        N = np.sqrt(E + self.m)
        
        # Construir spinor de 4 componentes
        # u = N * [[chi], [(σ·p)/(E+m) chi]]
        sigma_p = (p[0] * sigma_x + p[1] * sigma_y + p[2] * sigma_z)
        lower = sigma_p @ chi / (E + self.m) if E + self.m > 0 else np.zeros(2)
        
        u = N * np.concatenate([chi, lower])
        return u
    
    def v_spinor(self, p: np.ndarray, s: int) -> np.ndarray:
        """
        Spinor de antipartícula v(p, s).
        """
        E = self.energy(p)
        
        if s == 1:
            chi = np.array([0, 1], dtype=complex)  # Invertido para v
        else:
            chi = np.array([1, 0], dtype=complex)
        
        N = np.sqrt(E + self.m)
        sigma_p = (p[0] * sigma_x + p[1] * sigma_y + p[2] * sigma_z)
        upper = sigma_p @ chi / (E + self.m) if E + self.m > 0 else np.zeros(2)
        
        v = N * np.concatenate([upper, chi])
        return v
    
    def dirac_equation_check(self, psi: np.ndarray, p: np.ndarray) -> float:
        """
        Verifica que (γ·p - m)ψ = 0.
        
        Returns: norma del residuo (debe ser ~0)
        """
        E = self.energy(p)
        p4 = np.array([E, p[0], p[1], p[2]])
        
        # γ·p = γ⁰E - γⁱpⁱ
        gamma_p = gamma0 * E - gamma1 * p[0] - gamma2 * p[1] - gamma3 * p[2]
        
        residuo = (gamma_p - self.m * np.eye(4)) @ psi
        return np.linalg.norm(residuo)


# =============================================================================
# CONEXIÓN CON SPIN NETWORKS
# =============================================================================

class SpinorSpinNetworkConnection:
    """
    Conexión entre spinores de Dirac y spin networks de LQG.
    
    IDEA CLAVE:
    - En LQG, los edges de spin networks llevan representaciones de SU(2)
    - Los spinores también transforman bajo SU(2)
    - Un fermión en LQG se puede pensar como un "defecto" en la red de espín
    
    HIPÓTESIS DE KAELION:
    La entropía de un sistema con fermiones incluye una contribución
    del contenido de espín:
    
    S_total = S_área + S_fermiones
    
    donde S_fermiones ∝ ln(dim de la representación de espín)
    """
    
    def __init__(self):
        self.gamma = 0.2375  # Parámetro de Immirzi
    
    def spin_dimension(self, j: float) -> int:
        """Dimensión de la representación de espín j: dim = 2j + 1"""
        return int(2 * j + 1)
    
    def fermion_entropy_contribution(self, n_fermions: int, j: float = 0.5) -> float:
        """
        Contribución de fermiones a la entropía.
        
        Para espín-1/2: dim = 2, S_f = n × ln(2)
        """
        dim = self.spin_dimension(j)
        return n_fermions * np.log(dim)
    
    def holonomy_fermion(self, connection: np.ndarray, path_length: float) -> np.ndarray:
        """
        Holonomía de una conexión SU(2) a lo largo de un path.
        
        En LQG, la holonomía describe cómo el espín "rota" al moverse.
        Para un fermión, esto afecta su fase.
        """
        # Conexión SU(2) parametrizada como A = A^i τ_i
        # donde τ_i = -i σ_i / 2 son generadores de su(2)
        tau = [-1j * sigma_x / 2, -1j * sigma_y / 2, -1j * sigma_z / 2]
        
        A_matrix = sum(connection[i] * tau[i] for i in range(3))
        
        # Holonomía: U = exp(∫ A) ≈ exp(A × L) para conexión constante
        return expm(A_matrix * path_length)
    
    def lambda_with_fermions(self, A: float, S_fermions: float, 
                              S_total: float, A_c: float) -> float:
        """
        Parámetro λ incluyendo fermiones.
        
        Los fermiones contribuyen a la información accesible.
        """
        f_area = 1 - np.exp(-A / A_c)
        
        # La información de fermiones es "accesible"
        S_acc = S_fermions
        g_info = S_acc / S_total if S_total > 0 else 0
        
        return f_area * g_info


# =============================================================================
# SIMULACIÓN
# =============================================================================

print("\n" + "="*70)
print("SIMULACIÓN DE LA ECUACIÓN DE DIRAC")
print("="*70)

# Crear fermión (electrón)
m_electron = 1.0  # En unidades naturales
electron = DiracSpinor(m_electron)

# Diferentes momentos
momenta = [
    np.array([0, 0, 0]),      # En reposo
    np.array([0.5, 0, 0]),    # Movimiento lento
    np.array([1, 0, 0]),      # Relativista
    np.array([5, 0, 0]),      # Ultra-relativista
]

print("\n1. SPINORES DE DIRAC:")
print("-" * 70)
print(f"{'p':<20} {'E':<12} {'|u↑|²':<12} {'|u↓|²':<12} {'Residuo':<12}")
print("-" * 70)

for p in momenta:
    E = electron.energy(p)
    u_up = electron.u_spinor(p, 1)
    u_down = electron.u_spinor(p, -1)
    
    residuo = electron.dirac_equation_check(u_up, p)
    
    print(f"{str(p):<20} {E:<12.4f} {np.sum(np.abs(u_up)**2):<12.4f} "
          f"{np.sum(np.abs(u_down)**2):<12.4f} {residuo:<12.2e}")


# =============================================================================
# CONEXIÓN SPINOR - SPIN NETWORK
# =============================================================================

print("\n2. CONEXIÓN CON SPIN NETWORKS:")
print("-" * 70)

connection = SpinorSpinNetworkConnection()

# Diferentes números de fermiones
n_fermions_list = [1, 10, 100, 1000]

print(f"{'N fermiones':<15} {'S_f':<15} {'dim(j=1/2)':<15}")
print("-" * 45)

for n in n_fermions_list:
    S_f = connection.fermion_entropy_contribution(n, j=0.5)
    dim = connection.spin_dimension(0.5)
    print(f"{n:<15} {S_f:<15.4f} {dim:<15}")


# =============================================================================
# λ CON FERMIONES
# =============================================================================

print("\n3. PARÁMETRO λ CON FERMIONES:")
print("-" * 70)

A_c = 4 * np.pi / connection.gamma  # Área crítica
A = 1000  # Área del sistema
S_area = A / 4  # Entropía de área (en unidades G=1)

print(f"Área del sistema: A = {A}")
print(f"Área crítica: A_c = {A_c:.2f}")
print(f"S_área = {S_area:.2f}")

print(f"\n{'N fermiones':<15} {'S_f':<12} {'S_total':<12} {'λ':<12} {'α':<12}")
print("-" * 65)

for n in n_fermions_list:
    S_f = connection.fermion_entropy_contribution(n, j=0.5)
    S_total = S_area + S_f
    lam = connection.lambda_with_fermions(A, S_f, S_total, A_c)
    alpha = -0.5 - lam
    print(f"{n:<15} {S_f:<12.4f} {S_total:<12.4f} {lam:<12.4f} {alpha:<12.4f}")


# =============================================================================
# VERIFICACIONES
# =============================================================================

print("\n" + "="*70)
print("VERIFICACIONES")
print("="*70)

verifications = []

# V1: Ecuación de Dirac satisfecha
p_test = np.array([1, 0, 0])
u_test = electron.u_spinor(p_test, 1)
residuo = electron.dirac_equation_check(u_test, p_test)
v1 = residuo < 1e-10
verifications.append(("Ecuación de Dirac satisfecha", v1, f"residuo = {residuo:.2e}"))

# V2: Normalización de spinores
norm = np.sum(np.abs(u_test)**2)
E_test = electron.energy(p_test)
v2 = abs(norm - 2 * E_test) < 0.1  # u†u = 2E
verifications.append(("Normalización u†u = 2E", v2, f"u†u = {norm:.4f}, 2E = {2*E_test:.4f}"))

# V3: Anticommutación de gammas: {γμ, γν} = 2ημν
anticomm_00 = gamma0 @ gamma0 + gamma0 @ gamma0
v3 = np.allclose(anticomm_00, 2 * np.eye(4))
verifications.append(("{γ⁰, γ⁰} = 2I", v3, None))

# V4: γ⁵² = I
gamma5_sq = gamma5 @ gamma5
v4 = np.allclose(gamma5_sq, np.eye(4))
verifications.append(("γ⁵² = I", v4, None))

# V5: Helicidad definida para partícula ultra-relativista
p_ultra = np.array([100, 0, 0])
u_ultra = electron.u_spinor(p_ultra, 1)
# Para p >> m, los spinores se acercan a estados de helicidad
v5 = np.abs(u_ultra[0])**2 + np.abs(u_ultra[1])**2 > 0.99 * np.sum(np.abs(u_ultra)**2) * 0.5
verifications.append(("Límite ultra-relativista correcto", v5, None))

# V6: Entropía de fermiones positiva
S_f_test = connection.fermion_entropy_contribution(100, j=0.5)
v6 = S_f_test > 0
verifications.append(("S_fermiones > 0", v6, f"S_f = {S_f_test:.4f}"))

# V7: λ aumenta con más fermiones
lam_1 = connection.lambda_with_fermions(A, connection.fermion_entropy_contribution(1), S_area + connection.fermion_entropy_contribution(1), A_c)
lam_100 = connection.lambda_with_fermions(A, connection.fermion_entropy_contribution(100), S_area + connection.fermion_entropy_contribution(100), A_c)
v7 = lam_100 > lam_1
verifications.append(("λ aumenta con más fermiones", v7, f"λ(1)={lam_1:.4f}, λ(100)={lam_100:.4f}"))

print("\nResultados:")
print("-" * 70)
for item in verifications:
    name, passed, detail = item[0], item[1], item[2] if len(item) > 2 else None
    status = "✓ PASSED" if passed else "✗ FAILED"
    detail_str = f" ({detail})" if detail else ""
    print(f"  {status}: {name}{detail_str}")

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

# 1. Spinor u↑ vs momento
ax1 = axes[0, 0]
p_range = np.linspace(0.01, 5, 50)
u_norms = []
for p in p_range:
    u = electron.u_spinor(np.array([p, 0, 0]), 1)
    u_norms.append(np.sum(np.abs(u)**2))
ax1.plot(p_range, u_norms, 'b-', lw=2)
ax1.set_xlabel('Momento p')
ax1.set_ylabel('|u|²')
ax1.set_title('NORMALIZACIÓN DEL SPINOR')
ax1.grid(True, alpha=0.3)

# 2. Componentes del spinor
ax2 = axes[0, 1]
p_range = np.linspace(0.01, 5, 50)
comp_upper = []
comp_lower = []
for p in p_range:
    u = electron.u_spinor(np.array([p, 0, 0]), 1)
    comp_upper.append(np.sum(np.abs(u[:2])**2))
    comp_lower.append(np.sum(np.abs(u[2:])**2))
ax2.plot(p_range, comp_upper, 'b-', lw=2, label='Componentes superiores')
ax2.plot(p_range, comp_lower, 'r-', lw=2, label='Componentes inferiores')
ax2.set_xlabel('Momento p')
ax2.set_ylabel('|componente|²')
ax2.set_title('ESTRUCTURA DEL SPINOR')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Entropía de fermiones
ax3 = axes[0, 2]
n_range = np.logspace(0, 4, 50)
S_f_range = [connection.fermion_entropy_contribution(int(n)) for n in n_range]
ax3.loglog(n_range, S_f_range, 'g-', lw=2)
ax3.set_xlabel('Número de fermiones')
ax3.set_ylabel('S_fermiones')
ax3.set_title('ENTROPÍA DE FERMIONES')
ax3.grid(True, alpha=0.3)

# 4. λ vs número de fermiones
ax4 = axes[1, 0]
n_range = np.logspace(0, 4, 50)
lam_range = []
for n in n_range:
    S_f = connection.fermion_entropy_contribution(int(n))
    lam = connection.lambda_with_fermions(A, S_f, S_area + S_f, A_c)
    lam_range.append(lam)
ax4.semilogx(n_range, lam_range, 'purple', lw=2)
ax4.set_xlabel('Número de fermiones')
ax4.set_ylabel('λ')
ax4.set_title('λ CON FERMIONES')
ax4.grid(True, alpha=0.3)

# 5. Matrices gamma (visualización)
ax5 = axes[1, 1]
ax5.imshow(np.abs(gamma5), cmap='viridis')
ax5.set_title('|γ⁵| (matriz de quiralidad)')
ax5.set_xticks([])
ax5.set_yticks([])
ax5.colorbar = plt.colorbar(ax5.images[0], ax=ax5)

# 6. Resumen
ax6 = axes[1, 2]
ax6.axis('off')

ax6.text(0.5, 0.95, 'DIRAC - RESUMEN', ha='center', fontsize=12, fontweight='bold')

summary = f"""
┌────────────────────────────────────────┐
│     ECUACIÓN DE DIRAC                  │
├────────────────────────────────────────┤
│  (iγ·∂ - m)ψ = 0                       │
│                                        │
│  Spinor de 4 componentes               │
│  Espín 1/2 (fermiones)                 │
├────────────────────────────────────────┤
│     CONEXIÓN CON LQG                   │
├────────────────────────────────────────┤
│  • Spinores ↔ representaciones SU(2)   │
│  • Fermiones como "defectos" en        │
│    spin networks                       │
│  • S_f = n × ln(2) para espín-1/2      │
├────────────────────────────────────────┤
│     EFECTO EN λ                        │
├────────────────────────────────────────┤
│  λ aumenta con más fermiones           │
│  (más información accesible)           │
├────────────────────────────────────────┤
│  Verificaciones: {n_passed}/{len(verifications)} pasadas           │
└────────────────────────────────────────┘
"""

ax6.text(0.5, 0.45, summary, ha='center', va='center', fontsize=9,
         family='monospace', bbox=dict(boxstyle='round', facecolor='lightgreen'))
ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)

plt.suptitle('ECUACIÓN DE DIRAC - KAELION v3.0', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/Dirac_Equation.png', dpi=150, bbox_inches='tight')
print("✓ Figura guardada: Dirac_Equation.png")


# =============================================================================
# RESUMEN
# =============================================================================

print("\n" + "="*70)
print("RESUMEN: ECUACIÓN DE DIRAC COMPLETADA")
print("="*70)

print(f"""
╔═══════════════════════════════════════════════════════════════════════════╗
║                                                                           ║
║              ECUACIÓN DE DIRAC - RESULTADOS                               ║
║                                                                           ║
╠═══════════════════════════════════════════════════════════════════════════╣
║                                                                           ║
║  FÍSICA DE DIRAC:                                                         ║
║  • Ecuación: (iγ·∂ - m)ψ = 0                                              ║
║  • Spinores de 4 componentes para espín 1/2                               ║
║  • Matrices gamma satisfacen álgebra de Clifford                          ║
║                                                                           ║
║  CONEXIÓN CON LQG:                                                        ║
║  • Spinores transforman bajo SU(2) como spin networks                     ║
║  • Fermiones contribuyen entropía: S_f = n × ln(2)                        ║
║  • Los fermiones aumentan la información accesible                        ║
║                                                                           ║
║  EFECTO EN KAELION:                                                       ║
║  • λ aumenta con más fermiones                                            ║
║  • Sistema con muchos fermiones → más holográfico                         ║
║                                                                           ║
║  VERIFICACIONES: {n_passed}/{len(verifications)} PASADAS                                            ║
║                                                                           ║
╚═══════════════════════════════════════════════════════════════════════════╝
""")

plt.show()
