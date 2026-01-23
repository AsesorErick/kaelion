# Kaelion v3.3

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18238030.svg)](https://doi.org/10.5281/zenodo.18238030)

**A Phenomenological Correspondence Between Loop Quantum Gravity and Holographic Entropy**

---

## Overview

Kaelion proposes a correspondence equation that interpolates between Loop Quantum Gravity (LQG) and holographic entropy calculations:

```
S(A,I) = A/(4G) + α(λ)ln(A/l_P²) + β(λ) + γ(λ)(l_P²/A)
```

where the interpolation parameter λ ∈ [0,1] controls the transition:
- λ = 0: Pure LQG regime (α = -0.5)
- λ = 1: Pure holographic regime (α = -1.5)

**Key prediction:** α(λ) = -1/2 - λ. This is falsifiable and unique to Kaelion.

---

## 🎯 NEW in v3.3: Experimental Verification Complete

## 📊 Experimental Data

**Complete experimental data is maintained in [kaelion-experiments](https://github.com/AsesorErick/kaelion-experiments)**

Summary:
- **136+ data points** from IBM Quantum hardware
- **3 backends:** ibm_fez, ibm_torino, ibm_marrakesh
- **λ range:** [0.006, 1.000] - complete coverage
- **All Job IDs documented** for reproducibility

See: `kaelion-experiments/data/EXPERIMENTAL_DATA_MASTER.json`

---

## 🔬 Theoretical Foundation

**Want to know WHY α(λ) = -0.5 - λ?**

See: **[kaelion-derivation](https://github.com/AsesorErick/kaelion-derivation)**

The derivation shows that λ emerges from:
1. Tensor network coarse-graining
2. Holographic quantum error correction

Both approaches independently give α(λ) = -0.5 - λ.

---

## Statistics

- **Modules:** 25
- **Theoretical Verifications:** 156/164 passed (95.1%)
- **Experimental Points:** 74+ (IBM Quantum)
- **Domains:** 22 physics areas covered

---

## Module Overview

### Core (1-8): 38/38 tests passed
Fundamental building blocks: CHSH, Klein-Gordon, Ryu-Takayanagi, LQG spin networks, volume operator, 6j symbols, BTZ black hole, LQG-Holography connection.

### Extended (9-16): 53/56 tests passed
Applications: Page curve, Schwarzschild 4D, de Sitter, GSL, Kerr, LQC Big Bounce, Hayden-Preskill, Dirac equation.

### Advanced (17-20): 30/32 tests passed
Extensions: Reissner-Nordström, wormholes (ER=EPR), quantum error correction, topological entropy.

### Implications (21-25): 35/39 tests passed
Key questions answered:

| Module | Question | Answer |
|--------|----------|--------|
| 21 | Does Kaelion resolve the information paradox? | YES (partially) |
| 22 | Compatible with entropy islands (2019+)? | YES |
| 23 | Does Kaelion predict a firewall? | NO |
| 24 | How does λ relate to complexity? | Higher C → higher λ |
| 25 | How does λ affect scrambling? | Higher λ → faster |

---

## Key Constants

| Constant | Value | Source |
|----------|-------|--------|
| γ (Immirzi) | 0.2375 | Meissner (2004) |
| α_LQG | -0.5 | Kaul-Majumdar (2000) |
| α_CFT | -1.5 | Sen (2012) |
| A_c | 52.91 l_P² | Derived |
| V₀ | √3 | v3.0 Constants |
| φ₀ | 1/√3 | v3.0 Constants |

---

## Repository Structure

```
kaelion/
├── code/                    # 25 theoretical modules
├── experimental/            # Local verification scripts
│   └── → Full data at kaelion-experiments
├── figures/
├── paper/
├── README.md
├── CITATION.cff
└── LICENSE
```

---

## Installation

```bash
git clone https://github.com/AsesorErick/kaelion.git
cd kaelion/code
python3 summary_v31.py
```

### For IBM Quantum Experiments

```bash
pip install qiskit qiskit-ibm-runtime numpy matplotlib
cd experimental/verification_IBM
python SIM03_UNIVERSALITY_IBM.py
```

---

## Citation

```bibtex
@software{perez_kaelion_2026,
  author = {Pérez Eugenio, Erick Francisco},
  title = {Kaelion v3.3: Experimental Verification of the LQG-Holography Correspondence},
  year = {2026},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.18238030}
}
```

---

## Related Repositories

| Repository | Description |
|------------|-------------|
| **kaelion** (this) | Main model and simulations |
| [kaelion-derivation](https://github.com/AsesorErick/kaelion-derivation) | Theoretical foundation (Modules 26-38) |
| [kaelion-experiments](https://github.com/AsesorErick/kaelion-experiments) | **Complete experimental data (136+ points)** |
| [kaelion-formal](https://github.com/AsesorErick/kaelion-formal) | Formal verification (Lean/Coq) |

---

## License

MIT License

---

## Author

**Erick Francisco Pérez Eugenio**  
ORCID: [0009-0006-3228-4847](https://orcid.org/0009-0006-3228-4847)
