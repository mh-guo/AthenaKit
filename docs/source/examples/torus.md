# Magnetized Torus (GRMHD)

The `athenakit.app.torus` module supports analysis of magnetized accretion torus simulations in general relativity.

## Load and inspect

```python
import athenakit as ak

ad = ak.load("torus.out1.00200.bin")
print(f"GR: {ad.is_gr},  MHD: {ad.is_mhd},  Spin: {ad.spin}")
```

## Slice plots

```python
import matplotlib.pyplot as plt

# Density slice in the x-z plane
fig = ad.plot_slice('dens', axis='y', level=1, norm='log', cmap='inferno')
plt.title(f"t = {ad.time:.1f}")
plt.show()
```

## Phase diagrams

```python
# Density-temperature phase diagram weighted by volume
ad.set_hist2d([['dens', 'temp']], bins=128, scales=[['log','log']], weights='vol')
fig = ad.plot_phase('dens,temp', key='vol', cmap='viridis', norm='log')
plt.show()
```

## GR-specific variables

When `ad.is_gr` is `True`, additional derived variables are available:

| Variable | Description |
|---|---|
| `u^t` | Contravariant time component of 4-velocity |
| `u_t` | Covariant time component (related to specific energy) |
| `b^2` | Magnetic field invariant |
| `sigma` | Magnetization σ = b²/ρ |
