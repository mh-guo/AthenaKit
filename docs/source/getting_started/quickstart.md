# Quickstart

This guide shows the most common workflow: load a simulation snapshot, compute derived quantities, and make a plot.

## Load a snapshot

AthenaKit supports `.bin` (native AthenaK binary), `.athdf` (HDF5), and cached `.h5`/`.pkl` formats:

```python
import athenakit as ak

# Load a .bin file
ad = ak.load("my_simulation.out1.00042.bin")

# Or load an athdf file
ad = ak.load("my_simulation.out1.00042.athdf")
```

## Inspect available variables

```python
# See all variables (coordinates, raw, and derived)
print(ad.data_list)

# Key attributes from the header
print(f"Time = {ad.time}")
print(f"Grid: {ad.Nx1} x {ad.Nx2} x {ad.Nx3}")
print(f"MHD: {ad.is_mhd},  GR: {ad.is_gr}")
```

## Access data

`ad.data(var)` returns a per-meshblock array (shape `[n_mb, nz, ny, nx]`).
Raw variables from the dump (e.g. `dens`, `velx`, `eint`, `bcc1`) and 50+ derived quantities are all accessible by name:

```python
rho  = ad.data('dens')   # density
pres = ad.data('pres')   # thermal pressure  (= (γ-1)*eint)
temp = ad.data('temp')   # temperature       (= pres/dens)
vtot = ad.data('vtot')   # total velocity magnitude
btot = ad.data('btot')   # total magnetic field magnitude
beta = ad.data('beta')   # plasma β = pgas/pmag

# Math expressions also work
sound_speed = ad.data('(gamma*pres/dens)**0.5')
```

## Compute a radial profile

```python
prof = ad.get_profile('r', ['dens', 'temp'], bins=128, weights='vol')

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.loglog(prof['r'], prof['dens'])
ax.set_xlabel('r')
ax.set_ylabel('density')
plt.show()
```

## Make a slice plot

```python
# Project along z-axis (default)
fig = ad.plot_slice('dens', zoom=0, level=0, axis='z', norm='log', cmap='viridis')
plt.show()
```

## Save and reload processed data

```python
# Save expensive reductions to HDF5
ad.set_profile('r', ['dens', 'temp', 'pres'])
ad.save('snapshot_042.h5')

# Reload later without re-reading the binary
ad2 = ak.load('snapshot_042.h5')
```

## Next steps

- See {doc}`../examples/index` for worked astrophysical examples
- See {doc}`../api/index` for the full API reference
