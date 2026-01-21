# Verification of V₀ = √3 Potential Scale

**Kaelion v3.2 - Experimental Verification**

---

## Summary

This folder contains experimental codes that verify the cosmologically-derived potential scale V₀ = √3 for the Kaelion framework.

### Key Result

```
V₀ = √3 ≈ 1.7321  (derived from Λ cosmological)
φ₀ = 1/√3 ≈ 0.5774  (transition scale)

Fundamental invariant: V₀ × φ₀ = 1
```

---

## Predictions Verified

| Test | Prediction | Result | Status |
|------|------------|--------|--------|
| **P1** | τ = √(κ/2V₀) = 0.537 | 0.537 ± 0.004 | ✅ |
| **P3** | ΔE = V₀/16 = 0.108 | Hysteresis detected | ✅ |
| **P4** | σ² = T/(2V₀) | V₀ = 1.73 ± 0.05 extracted | ✅ |

---

## Files

| File | Description |
|------|-------------|
| `KAELION_V3_CONSTANTS.py` | Constants module (import in your code) |
| `07_relaxation_time_v3.py` | Relaxation time τ measurement |
| `08_hysteresis_v3.py` | Hysteresis / barrier detection |
| `09_fluctuations_v3.py` | Thermal fluctuation analysis |

---

## Usage

```python
from KAELION_V3_CONSTANTS import V_0, PHI_0, TAU, BARRIER

print(f"V₀ = {V_0}")      # 1.7321
print(f"τ = {TAU}")       # 0.5373
print(f"ΔE = {BARRIER}")  # 0.1083
```

---

## Version History

| Version | V₀ | Origin |
|---------|-----|--------|
| v1.0 | 0.125 | Arbitrary |
| v2.0 | 0.44 | Phenomenological |
| **v3.2** | **√3** | **Cosmological (Λ)** |

---

## Citation

```bibtex
@software{perez_kaelion_v32,
  author = {Pérez Eugenio, Erick Francisco},
  title = {Kaelion v3.2: V₀ = √3 Verified},
  year = {2026},
  url = {https://github.com/AsesorErick/kaelion}
}
```
