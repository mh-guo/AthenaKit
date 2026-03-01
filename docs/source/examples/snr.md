# Supernova Remnant

The `athenakit.app.snr` module provides tools for analyzing supernova remnant (SNR) simulations.

## Load and explore

```python
import athenakit as ak

ad = ak.load("snr.out1.00050.bin")
print(f"Domain: [{ad.x1min}, {ad.x1max}]^3,  t = {ad.time:.2f}")
```

## Shock front radius

```python
import numpy as np
import matplotlib.pyplot as plt

# Locate the shock by the density jump
prof = ad.get_profile('r', ['dens', 'pres', 'vtot'], bins=512, weights='vol')
r_shock = prof['r'][np.argmax(np.gradient(prof['dens']))]
print(f"Shock radius: {r_shock:.3f}")
```

## Sedov-Taylor comparison

```python
from athenakit.physics.snr import sedov_taylor

r_st, rho_st, p_st = sedov_taylor(E=1.0, rho0=1.0, t=ad.time)

fig, ax = plt.subplots()
ax.semilogy(prof['r'], prof['pres'], label='simulation')
ax.semilogy(r_st, p_st, '--', label='Sedov-Taylor')
ax.set_xlabel('r')
ax.set_ylabel('pressure')
ax.legend()
plt.show()
```
