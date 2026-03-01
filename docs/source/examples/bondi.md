# Bondi Accretion

The `athenakit.app.bondi` module provides analysis tools for Bondi (spherically-symmetric) accretion simulations.

## Setup

```python
import athenakit as ak
from athenakit.app.bondi import BondiData

ad = ak.load("bondi.out1.00100.bin")
```

## Radial profiles

```python
# Radial profile of density and inflow velocity
prof = ad.get_profile('r', ['dens', 'velr', 'temp'], bins=256, weights='vol')

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, var in zip(axes, ['dens', 'velr', 'temp']):
    ax.loglog(prof['r'], abs(prof[var]))
    ax.set_xlabel('r')
    ax.set_ylabel(var)
plt.tight_layout()
```

## Accretion rate

```python
# Mass flux through a sphere at radius r_s
r_s = 10.0
mask = (ad.data('r') > 0.95*r_s) & (ad.data('r') < 1.05*r_s)
mdot = ad.sum('mflxrin', where=mask)
print(f"Accretion rate: {mdot:.4e}")
```
