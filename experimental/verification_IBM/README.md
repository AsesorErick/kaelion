# IBM Quantum Verification - Kaelion v3.3

Experimental verification of α(λ) = -1/2 - λ on IBM Quantum hardware.

## Results Summary

| Experiment | Job ID | Status | Key Result |
|------------|--------|--------|------------|
| 32 points (2 backends) | 27 jobs | ✅ | p < 10⁻¹⁰ |
| SIM01 Spatial Gradient | d5p8ij0r0v5s739nkph0 | ✅ 5/6 | Correlation 0.932 |
| SIM02 LQG Region | d5p9289dgvjs73dbe2r0 | ✅ 4/6 | λ = 0.24 detected |
| SIM03 Universality | d5p9gk8h0i0s73eov7r0 | ✅ 4/4 | Error = 0, 11 models |

**Total: 74+ experimental data points**

## Hardware

- **ibm_fez:** 156 qubits (Heron processor)
- **ibm_torino:** 133 qubits

## Scripts

| Script | Description |
|--------|-------------|
| `SIM01_SPATIAL_GRADIENT_IBM.py` | λ(x) spatial profile measurement |
| `SIM02_LQG_REGION_IBM.py` | Pure LQG regime detection (J=0) |
| `SIM03_UNIVERSALITY_IBM.py` | Multi-model universality test |
| `SIM01_IBM_QUANTUM.py` | Basic OTOC verification |
| `SIM02_IBM_QUANTUM.py` | Cross-platform consistency |
| `KAELION_COLAB.py` | Combined Colab notebook |

## Requirements

```bash
pip install qiskit qiskit-ibm-runtime numpy matplotlib
```

## Quick Start

```python
from qiskit_ibm_runtime import QiskitRuntimeService

# Save token (once)
QiskitRuntimeService.save_account(
    channel="ibm_quantum_platform",
    token="YOUR_TOKEN"
)

# Run any script
python SIM03_UNIVERSALITY_IBM.py
```

## Key Finding: Universality

α(λ) = -1/2 - λ verified across 5 different Hamiltonians:

1. **Kicked Ising** - Standard chaos model
2. **Heisenberg XXZ** - Spin chain
3. **Random Circuits** - Maximum scrambling
4. **Transverse Field Ising** - Phase transitions
5. **XY Model** - Anisotropic interactions

**Result:** Zero error across all models.

## Date

22 January 2026
