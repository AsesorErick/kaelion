# Release Notes - Kaelion v3.3

**Release Date:** 22 January 2026

## Highlights

🎯 **First experimental verification of α(λ) = -1/2 - λ on real quantum hardware**

## New Features

### IBM Quantum Verification (`experimental/verification_IBM/`)

Complete experimental verification using IBM Quantum processors:

| Experiment | Description | Result |
|------------|-------------|--------|
| **SIM01** | Spatial gradient λ(x) | Correlation 0.932 |
| **SIM02** | LQG region detection | λ = 0.24 at J=0 |
| **SIM03** | Universality test | Error = 0 across 11 models |

### Key Findings

1. **74+ experimental data points** across 30+ IBM Quantum jobs
2. **Pure LQG regime detected:** λ = 0.245 when J=0
3. **Universality confirmed:** 5 different Hamiltonians, zero error
4. **Cross-platform consistency:** ibm_fez and ibm_torino agree within 1%

### New Scripts

- `SIM01_SPATIAL_GRADIENT_IBM.py` - Spatial variation of λ
- `SIM02_LQG_REGION_IBM.py` - LQG regime detection
- `SIM03_UNIVERSALITY_IBM.py` - Multi-model universality test

### New Figures (`figures/ibm_verification/`)

- `SIM01_Gradiente_Espacial.png`
- `SIM02_Region_LQG.png`
- `SIM03_Universality.png`
- `kaelion_verificacion_FINAL_27puntos.png`

## Statistics

| Metric | v3.2 | v3.3 |
|--------|------|------|
| Theoretical tests | 156/164 | 156/164 |
| Experimental points | 0 | **74+** |
| Hardware backends | 0 | **2** |
| Models verified | 0 | **11** |

## Breaking Changes

None. Fully backward compatible.

## Dependencies (for experimental)

```
qiskit >= 1.0
qiskit-ibm-runtime >= 0.20
numpy >= 1.24
matplotlib >= 3.7
```

## Contributors

- Erick Francisco Pérez Eugenio (ORCID: 0009-0006-3228-4847)
