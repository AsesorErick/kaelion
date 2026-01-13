"""
¿QUÉ DETERMINA λ? - EXPLORACIÓN DEL PARÁMETRO DE INTERPOLACIÓN
================================================================

Hemos identificado que la corrección logarítmica a la entropía difiere:
    S = A/(4G) + α ln(A) + ...
    
    LQG: α = -1/2
    CFT: α = -3/2

Proponemos parametrizar:
    α(λ) = -1/2 - λ
    
    λ = 0 → LQG
    λ = 1 → CFT

PREGUNTA CENTRAL: ¿Qué determina λ?

Exploramos 5 hipótesis:
1. λ depende del ÁREA (régimen de escala)
2. λ depende de la TOPOLOGÍA del horizonte
3. λ depende del ACOPLAMIENTO (fuerte/débil)
4. λ depende de la DIMENSIÓN del espacio-tiempo
5. λ es un OBSERVABLE EMERGENTE (información/complejidad)

Proyecto Kaelion v3.0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma as gamma_func
from scipy.optimize import minimize_scalar

print("="*70)
print("¿QUÉ DETERMINA λ? - PARÁMETRO DE INTERPOLACIÓN LQG ↔ HOLOGRAFÍA")
print("="*70)

# =============================================================================
# CONSTANTES
# =============================================================================

l_P = 1.0
G_N = 1.0
gamma_immirzi = 0.2375

# Coeficientes conocidos
alpha_LQG = -0.5
alpha_CFT = -1.5


def alpha_interpolated(lam):
    """Coeficiente logarítmico interpolado"""
    return -0.5 - lam  # α = -1/2 - λ


def entropy_with_alpha(A, alpha):
    """Entropía con corrección logarítmica general"""
    S_0 = A / (4 * G_N)
    S_log = alpha * np.log(A / l_P**2) if A > l_P**2 else 0
    return S_0 + S_log


# =============================================================================
# HIPÓTESIS 1: λ DEPENDE DEL ÁREA (ESCALA)
# =============================================================================

print("\n" + "="*70)
print("HIPÓTESIS 1: λ = λ(A) - DEPENDENCIA DE ESCALA")
print("="*70)

print("""
IDEA FÍSICA:
- A pequeñas escalas (A ~ l_P²), dominan efectos cuánticos → LQG
- A grandes escalas (A >> l_P²), domina la descripción efectiva → CFT/Holo
- λ interpola suavemente entre ambos regímenes

PROPUESTA:
    λ(A) = 1 - exp(-A/A_c)
    
donde A_c es un área crítica de crossover.

    A << A_c: λ ≈ 0 → α ≈ -1/2 (LQG)
    A >> A_c: λ ≈ 1 → α ≈ -3/2 (CFT)
""")

def lambda_area_dependent(A, A_c=100):
    """λ como función del área"""
    return 1 - np.exp(-A / A_c)


def alpha_area_dependent(A, A_c=100):
    """α que interpola según el área"""
    lam = lambda_area_dependent(A, A_c)
    return alpha_interpolated(lam)


# Explorar diferentes valores de A_c
A_range = np.logspace(0, 5, 100)
A_c_values = [10, 100, 1000]

print("\nComportamiento de α(A) para diferentes A_c:")
print("-" * 60)

for A_c in A_c_values:
    print(f"\n  A_c = {A_c} l_P²:")
    for A_test in [1, 10, 100, 1000, 10000]:
        alpha_val = alpha_area_dependent(A_test, A_c)
        regime = "LQG" if alpha_val > -1.0 else "CFT" if alpha_val < -1.0 else "transición"
        print(f"    A = {A_test:5d}: α = {alpha_val:.3f} ({regime})")


# =============================================================================
# HIPÓTESIS 2: λ DEPENDE DE LA TOPOLOGÍA
# =============================================================================

print("\n" + "="*70)
print("HIPÓTESIS 2: λ = λ(χ) - DEPENDENCIA TOPOLÓGICA")
print("="*70)

print("""
IDEA FÍSICA:
- La topología del horizonte afecta el conteo de microestados
- Horizontes con diferente género g tienen diferentes correcciones
- La característica de Euler χ = 2 - 2g podría entrar en λ

PROPUESTA:
    λ(χ) = (2 - χ) / 2 = g
    
    Esfera (g=0, χ=2):  λ = 0 → α = -1/2 (LQG dominante)
    Toro (g=1, χ=0):    λ = 1 → α = -3/2 (CFT dominante)
    
JUSTIFICACIÓN:
- En LQG, el conteo sobre la esfera da α = -1/2
- En CFT, la función de partición en el toro tiene correcciones adicionales
- La topología "conecta" los dos regímenes
""")

def lambda_topological(genus):
    """λ como función del género topológico"""
    return min(genus, 1.0)  # Saturar en 1


def alpha_topological(genus):
    """α dependiente de la topología"""
    return alpha_interpolated(lambda_topological(genus))


print("\nα para diferentes topologías:")
print("-" * 40)
topologies = [(0, "Esfera S²"), (1, "Toro T²"), (2, "Género 2")]
for g, name in topologies:
    alpha_val = alpha_topological(g)
    print(f"  {name} (g={g}): α = {alpha_val:.3f}")


# =============================================================================
# HIPÓTESIS 3: λ DEPENDE DEL ACOPLAMIENTO
# =============================================================================

print("\n" + "="*70)
print("HIPÓTESIS 3: λ = λ(g_s) - DEPENDENCIA DEL ACOPLAMIENTO")
print("="*70)

print("""
IDEA FÍSICA:
- En teoría de cuerdas, g_s (acoplamiento de cuerdas) controla
  la transición entre diferentes regímenes
- Acoplamiento débil → descripción perturbativa (tipo CFT)
- Acoplamiento fuerte → descripción no perturbativa (tipo LQG)

PROPUESTA:
    λ(g_s) = g_s² / (1 + g_s²)
    
    g_s → 0: λ → 0 → α = -1/2 (régimen de cuerdas débil ~ LQG)
    g_s → ∞: λ → 1 → α = -3/2 (régimen de cuerdas fuerte ~ CFT)
    
NOTA: Esta es una analogía. En realidad:
- LQG no tiene g_s
- La correspondencia es conceptual, no literal
""")

def lambda_coupling(g_s):
    """λ como función del acoplamiento"""
    return g_s**2 / (1 + g_s**2)


def alpha_coupling(g_s):
    """α dependiente del acoplamiento"""
    return alpha_interpolated(lambda_coupling(g_s))


print("\nα para diferentes acoplamientos:")
print("-" * 40)
g_s_values = [0.1, 0.5, 1.0, 2.0, 10.0]
for g_s in g_s_values:
    lam = lambda_coupling(g_s)
    alpha_val = alpha_coupling(g_s)
    print(f"  g_s = {g_s:4.1f}: λ = {lam:.3f}, α = {alpha_val:.3f}")


# =============================================================================
# HIPÓTESIS 4: λ DEPENDE DE LA DIMENSIÓN
# =============================================================================

print("\n" + "="*70)
print("HIPÓTESIS 4: λ = λ(D) - DEPENDENCIA DIMENSIONAL")
print("="*70)

print("""
IDEA FÍSICA:
- Las correcciones logarítmicas dependen de la dimensión D
- En D=3 (BTZ): la holografía es exacta (AdS₃/CFT₂)
- En D=4 (Schwarzschild): LQG está mejor desarrollado
- En D→∞: límite de campo medio, holografía domina

PROPUESTA:
    λ(D) = (D - 3) / (D - 2)
    
    D = 3: λ = 0 → α = -1/2 (BTZ, ambos coinciden más)
    D = 4: λ = 1/2 → α = -1 (mezcla)
    D → ∞: λ → 1 → α = -3/2 (holografía pura)
    
NOTA: Los coeficientes reales dependen de D de forma más compleja.
""")

def lambda_dimensional(D):
    """λ como función de la dimensión"""
    if D <= 2:
        return 0
    return (D - 3) / (D - 2)


def alpha_dimensional(D):
    """α dependiente de la dimensión"""
    return alpha_interpolated(lambda_dimensional(D))


print("\nα para diferentes dimensiones:")
print("-" * 40)
for D in [3, 4, 5, 6, 10, 26]:
    lam = lambda_dimensional(D)
    alpha_val = alpha_dimensional(D)
    print(f"  D = {D:2d}: λ = {lam:.3f}, α = {alpha_val:.3f}")


# =============================================================================
# HIPÓTESIS 5: λ ES UN OBSERVABLE EMERGENTE (INFORMACIÓN)
# =============================================================================

print("\n" + "="*70)
print("HIPÓTESIS 5: λ = λ(I) - DEPENDENCIA INFORMACIONAL")
print("="*70)

print("""
IDEA FÍSICA (CONEXIÓN KAELION):
- La información es la sustancia fundamental (Pilar 1)
- λ podría depender de la "complejidad" o "accesibilidad" de la información
- Información completamente accesible → CFT (borde)
- Información oculta/codificada → LQG (bulk)

PROPUESTA:
    λ = S_accesible / S_total
    
donde:
- S_total = A/(4G) (entropía de Bekenstein-Hawking)
- S_accesible = entropía visible desde el borde

Para un agujero negro:
- S_accesible ≈ S_radiación (lo que escapa en radiación Hawking)
- Durante evaporación: S_accesible crece, λ crece

OTRA INTERPRETACIÓN:
    λ = 1 - I_mutual(bulk, borde) / S_total
    
- Alta correlación bulk-borde → λ pequeño → LQG
- Baja correlación → λ grande → CFT
""")

def lambda_informational(S_accessible, S_total):
    """λ como ratio de información accesible"""
    if S_total <= 0:
        return 0
    return min(S_accessible / S_total, 1.0)


# Simular evaporación de agujero negro
print("\nEvolución de λ durante evaporación de agujero negro:")
print("-" * 60)

# Modelo simplificado: S_total decrece, S_radiación crece
S_initial = 100  # Entropía inicial
times = np.linspace(0, 1, 11)  # Tiempo normalizado (0 = inicio, 1 = evaporación completa)

print(f"{'Tiempo':<10} {'S_total':<12} {'S_rad':<12} {'λ':<10} {'α':<10}")
print("-" * 60)

for t in times:
    S_total = S_initial * (1 - t)**2  # Decrece cuadráticamente
    S_rad = S_initial * t  # Crece linealmente (simplificación)
    
    if S_total > 0:
        lam = lambda_informational(S_rad, S_initial)
        alpha_val = alpha_interpolated(lam)
    else:
        lam = 1.0
        alpha_val = -1.5
    
    print(f"{t:<10.2f} {S_total:<12.2f} {S_rad:<12.2f} {lam:<10.3f} {alpha_val:<10.3f}")


# =============================================================================
# COMPARACIÓN DE HIPÓTESIS
# =============================================================================

print("\n" + "="*70)
print("COMPARACIÓN DE LAS 5 HIPÓTESIS")
print("="*70)

print("""
┌──────────────────────────────────────────────────────────────────────────┐
│  HIPÓTESIS      │ VARIABLE   │ RANGO λ │ FÍSICAMENTE │ KAELION         │
├──────────────────────────────────────────────────────────────────────────┤
│ 1. Área         │ A/A_c      │ [0,1]   │ Natural     │ Escala          │
│ 2. Topología    │ género g   │ {0,1,2} │ Discreto    │ Estructura      │
│ 3. Acoplamiento │ g_s        │ [0,1]   │ String only │ Interacción     │
│ 4. Dimensión    │ D          │ [0,1]   │ Fijo en D=4 │ Emergente       │
│ 5. Información  │ S_acc/S_t  │ [0,1]   │ Dinámico    │ Sustancia       │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  EVALUACIÓN:                                                             │
│                                                                          │
│  • Hipótesis 1 (Área): Más natural físicamente. El crossover             │
│    LQG→Holo debería depender de la escala.                               │
│                                                                          │
│  • Hipótesis 2 (Topología): Interesante pero discreto. Solo              │
│    aplicable cuando la topología varía.                                  │
│                                                                          │
│  • Hipótesis 3 (Acoplamiento): Requiere teoría de cuerdas.               │
│    No directamente aplicable a LQG puro.                                 │
│                                                                          │
│  • Hipótesis 4 (Dimensión): La dimensión es fija (D=4). No               │
│    proporciona dinámica.                                                 │
│                                                                          │
│  • Hipótesis 5 (Información): Más profunda conceptualmente.              │
│    Conecta con Kaelion. Dinámica durante evolución.                      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
""")


# =============================================================================
# HIPÓTESIS COMBINADA: λ = λ(A, I)
# =============================================================================

print("\n" + "="*70)
print("PROPUESTA: HIPÓTESIS COMBINADA")
print("="*70)

print("""
La opción más rica combina ÁREA e INFORMACIÓN:

    λ(A, I) = f(A/A_c) × g(I)

donde:
    f(A/A_c) = 1 - exp(-A/A_c)     [factor de escala]
    g(I) = S_accesible/S_total     [factor informacional]

INTERPRETACIÓN:
- f(A/A_c): "Cuán clásico" es el sistema (área grande = más clásico)
- g(I): "Cuán holográfico" es el sistema (más info en borde = más holo)

FORMA FINAL:
    λ = [1 - exp(-A/A_c)] × [S_acc/S_total]

CASOS LÍMITE:
    A pequeño, info oculta:     λ → 0 → LQG
    A grande, info en borde:    λ → 1 → Holografía
    A grande, info oculta:      λ intermedio
    A pequeño, info en borde:   λ intermedio (¿paradoja?)
""")

def lambda_combined(A, S_accessible, S_total, A_c=100):
    """λ combinando área e información"""
    f_area = 1 - np.exp(-A / A_c)
    g_info = S_accessible / S_total if S_total > 0 else 0
    return f_area * g_info


# Mapa 2D de λ
A_grid = np.logspace(0, 4, 50)
ratio_grid = np.linspace(0, 1, 50)

lambda_map = np.zeros((len(ratio_grid), len(A_grid)))

for i, ratio in enumerate(ratio_grid):
    for j, A in enumerate(A_grid):
        S_total = A / (4 * G_N)
        S_acc = ratio * S_total
        lambda_map[i, j] = lambda_combined(A, S_acc, S_total, A_c=100)


# =============================================================================
# VISUALIZACIÓN
# =============================================================================

print("\n" + "="*70)
print("GENERANDO VISUALIZACIÓN")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# 1. λ vs Área
ax1 = axes[0, 0]
for A_c in [10, 100, 1000]:
    lam_values = [lambda_area_dependent(A, A_c) for A in A_range]
    ax1.semilogx(A_range, lam_values, lw=2, label=f'A_c = {A_c}')
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('Área A (l_P²)')
ax1.set_ylabel('λ')
ax1.set_title('HIPÓTESIS 1: λ(A)')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. α vs Área
ax2 = axes[0, 1]
for A_c in [10, 100, 1000]:
    alpha_values = [alpha_area_dependent(A, A_c) for A in A_range]
    ax2.semilogx(A_range, alpha_values, lw=2, label=f'A_c = {A_c}')
ax2.axhline(y=-0.5, color='blue', linestyle=':', alpha=0.7, label='LQG (α=-1/2)')
ax2.axhline(y=-1.5, color='red', linestyle=':', alpha=0.7, label='CFT (α=-3/2)')
ax2.set_xlabel('Área A (l_P²)')
ax2.set_ylabel('α')
ax2.set_title('COEFICIENTE α(A)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. λ vs Acoplamiento
ax3 = axes[0, 2]
g_s_range = np.logspace(-2, 2, 100)
lam_coupling = [lambda_coupling(g) for g in g_s_range]
ax3.semilogx(g_s_range, lam_coupling, 'g-', lw=2)
ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax3.axvline(x=1, color='gray', linestyle='--', alpha=0.5)
ax3.set_xlabel('Acoplamiento g_s')
ax3.set_ylabel('λ')
ax3.set_title('HIPÓTESIS 3: λ(g_s)')
ax3.grid(True, alpha=0.3)

# 4. λ durante evaporación
ax4 = axes[1, 0]
times_fine = np.linspace(0, 0.99, 100)
lam_evap = []
for t in times_fine:
    S_rad = S_initial * t
    lam_evap.append(lambda_informational(S_rad, S_initial))
ax4.plot(times_fine, lam_evap, 'm-', lw=2)
ax4.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax4.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Page time')
ax4.set_xlabel('Tiempo de evaporación (normalizado)')
ax4.set_ylabel('λ')
ax4.set_title('HIPÓTESIS 5: λ(t) EVAPORACIÓN')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 5. Mapa 2D de λ combinada
ax5 = axes[1, 1]
im = ax5.imshow(lambda_map, extent=[0, 4, 0, 1], aspect='auto', 
                origin='lower', cmap='viridis')
ax5.set_xlabel('log₁₀(A/l_P²)')
ax5.set_ylabel('S_accesible / S_total')
ax5.set_title('HIPÓTESIS COMBINADA: λ(A, I)')
plt.colorbar(im, ax=ax5, label='λ')

# 6. Resumen visual
ax6 = axes[1, 2]
ax6.axis('off')

# Dibujar diagrama de flujo conceptual
ax6.text(0.5, 0.95, '¿QUÉ DETERMINA λ?', ha='center', fontsize=14, fontweight='bold')

# Cajas
boxes = [
    (0.15, 0.7, 'ÁREA\nA/A_c', 'lightblue'),
    (0.5, 0.7, 'INFORMACIÓN\nS_acc/S_tot', 'lightgreen'),
    (0.85, 0.7, 'TOPOLOGÍA\ngénero g', 'lightyellow'),
]

for x, y, text, color in boxes:
    ax6.add_patch(plt.Rectangle((x-0.12, y-0.1), 0.24, 0.2, 
                                 facecolor=color, edgecolor='black', lw=2))
    ax6.text(x, y, text, ha='center', va='center', fontsize=9)

# Flechas hacia λ
for x, _, _, _ in boxes:
    ax6.annotate('', xy=(0.5, 0.4), xytext=(x, 0.6),
                arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

# λ central
ax6.add_patch(plt.Circle((0.5, 0.35), 0.1, facecolor='orange', edgecolor='black', lw=2))
ax6.text(0.5, 0.35, 'λ', ha='center', va='center', fontsize=16, fontweight='bold')

# Flechas hacia α
ax6.annotate('', xy=(0.5, 0.15), xytext=(0.5, 0.25),
            arrowprops=dict(arrowstyle='->', color='black', lw=2))

# α resultado
ax6.text(0.5, 0.08, 'α(λ) = -1/2 - λ', ha='center', fontsize=12,
         bbox=dict(boxstyle='round', facecolor='white', edgecolor='black'))

# Límites
ax6.text(0.15, 0.08, 'λ→0: LQG\nα→-1/2', ha='center', fontsize=9, color='blue')
ax6.text(0.85, 0.08, 'λ→1: CFT\nα→-3/2', ha='center', fontsize=9, color='red')

ax6.set_xlim(0, 1)
ax6.set_ylim(0, 1)

plt.suptitle('EXPLORACIÓN: ¿QUÉ DETERMINA λ?', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/claude/Lambda_Determination.png', dpi=150, bbox_inches='tight')
print("✓ Figura guardada: Lambda_Determination.png")


# =============================================================================
# CONCLUSIONES Y SIGUIENTE PASO
# =============================================================================

print("\n" + "="*70)
print("CONCLUSIONES")
print("="*70)

print("""
┌──────────────────────────────────────────────────────────────────────────┐
│                         HALLAZGOS PRINCIPALES                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  1. λ NO ES ÚNICO: Múltiples factores pueden determinarlo                │
│                                                                          │
│  2. HIPÓTESIS MÁS PROMETEDORAS:                                          │
│     • Área (H1): Natural, continuo, físicamente motivado                 │
│     • Información (H5): Profundo, dinámico, conecta con Kaelion          │
│                                                                          │
│  3. HIPÓTESIS COMBINADA es la más rica:                                  │
│     λ(A, I) = [1 - exp(-A/A_c)] × [S_acc/S_total]                        │
│                                                                          │
│  4. PARÁMETRO LIBRE: A_c (área de crossover)                             │
│     • A_c ~ 100 l_P² parece razonable                                    │
│     • Podría fijarse con input adicional                                 │
│                                                                          │
│  5. CONEXIÓN KAELION:                                                    │
│     • Pilar 1 (Sustancia): S_acc/S_total es ratio informacional          │
│     • Pilar 2 (1/-1): LQG (discreto) ↔ Holo (continuo)                   │
│     • Alteridad: La interpolación λ conecta ambos polos                  │
│                                                                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  PREGUNTA ABIERTA:                                                       │
│                                                                          │
│  ¿Puede derivarse A_c desde principios fundamentales?                    │
│  ¿O es un parámetro fenomenológico?                                      │
│                                                                          │
│  PROPUESTA: A_c podría relacionarse con:                                 │
│  • El parámetro de Immirzi: A_c ~ 1/γ² ~ 17 l_P²                         │
│  • La carga central: A_c ~ c × l_P²                                      │
│  • La entropía de Bekenstein-Hawking mínima: A_c ~ A_min                  │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
""")

print("\n" + "="*70)
print("SIGUIENTE PASO SUGERIDO")
print("="*70)

print("""
Con λ caracterizado, podemos ahora:

    A) Formular la ECUACIÓN DE CORRESPONDENCIA completa:
       
       S = A/(4G) + α(λ) ln(A/l_P²) + β(λ) + O(l_P²/A)
       
       con λ = λ(A, I) especificado

    B) Intentar DERIVAR A_c desde los pilares de Kaelion

    C) Verificar si la ecuación tiene PREDICCIONES NUEVAS
       que difieran de LQG puro o Holografía pura

¿Cuál es el siguiente paso?
""")

plt.show()
