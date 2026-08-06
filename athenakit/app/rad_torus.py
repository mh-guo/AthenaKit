'''
Application module for GR radiation-MHD torus runs (rad_torus campaign).

Provides:
    setup(ad)            : units and common derived variables
    reduce_column(ad)    : (r,theta) slice maps + polar-column scalars
    reduce_profiles(ad)  : tilt-aware radial profiles from a 3D (coarsened) dump
    reduce_tasks(ctx)    : task registry for the generic `athenakit-reduce` driver
        c : column diagnostics from 2D slices -> data/pkl/column_<slice>.NNNNN.pkl
        p : radial profiles from cbin dumps   -> data/pkl/Base.NNNNN.pkl
'''
import os
import pickle
import numpy as np

from .. import units as units_mod
from .. import load


def tilt_of(ad):
    """Torus tilt angle in radians (tilted toward +x from +z)."""
    return np.deg2rad(ad.header('problem', 'tilt_angle', float, 0.0))


def setup(ad):
    '''Attach cgs units and common data funcs (mirrors the notebook boilerplate).'''
    u = units_mod
    bhmass_msun = ad.header('units', 'bhmass_msun', float)
    density_cgs = ad.header('units', 'density_cgs', float)
    mu = ad.header('units', 'mu', float, 0.618)
    bhmass_cgs = bhmass_msun * u.msun_cgs
    length_cgs = u.grav_constant_cgs * bhmass_cgs / u.speed_of_light_cgs**2
    time_cgs = length_cgs / u.speed_of_light_cgs
    mass_cgs = density_cgs * length_cgs**3
    ad.unit = u.Units(lunit=length_cgs, munit=mass_cgs, tunit=time_cgs, mu=mu)
    ad.dunit = density_cgs
    ad.lunit = length_cgs
    ad.tempunit = ad.unit.temperature_cgs
    kappa_s = ad.header('radiation', 'kappa_s', float, 0.34)
    ad.ledd = 4.0 * np.pi / (kappa_s * density_cgs * length_cgs)
    ad.add_data_func('dunit', lambda d: d.ad.dunit)
    ad.add_data_func('lunit', lambda d: d.ad.lunit)
    ad.add_data_func('tempunit', lambda d: d.ad.tempunit)
    ad.add_data_func('Ledd', lambda d: d.ad.ledd)
    return ad


def load_snapshot(binfile, sigmafile=None, add_gr=True, do_setup=True):
    ad = load(binfile)
    if sigmafile and os.path.isfile(sigmafile):
        ad.load(sigmafile)
    if add_gr:
        ad.add_gr_data()
    if do_setup:
        setup(ad)
    return ad


# ------------------------------------------------------------------ column

def reduce_column(ad, slice_axis='x2', nth=360, nr=96, rmin=20.0):
    '''(r,theta) maps in the slice plane plus polar-column scalars.

    theta is measured from +z toward the in-plane transverse coordinate
    (x for slice_x2, y for slice_x1), range (-pi, pi].  For a run tilted
    by 30 deg toward +x, the torus axis sits at theta=+30 deg (north) and
    theta=-150 deg (south) in slice_x2.  The per-shell scalars locate the
    density peak so one can tell whether the polar column follows the
    grid axis (theta=0) or the torus axis (theta=tilt).
    '''
    trans = {'x1': 'y', 'x2': 'x'}[slice_axis]
    t = ad.data(trans).ravel()
    z = ad.data('z').ravel()
    dx = ad.data('dx').ravel()
    dens = ad.data('dens').ravel()
    r = np.sqrt(t**2 + z**2)
    theta = np.arctan2(t, z)
    area = dx**2
    velt = ad.data('vel' + trans).ravel()
    velz = ad.data('velz').ravel()
    vr = (t * velt + z * velz) / np.maximum(r, 1e-30)
    try:
        kap = (ad.data('sigma_a') + ad.data('sigma_s')).ravel()
    except Exception:
        kap = None
    erad = ad.data('erad').ravel()

    rmax = float(np.min(np.abs([ad.x1min, ad.x1max, ad.x2min, ad.x2max,
                                ad.x3min, ad.x3max])))
    redge = np.logspace(np.log10(rmin), np.log10(rmax), nr + 1)
    tedge = np.linspace(-np.pi, np.pi, nth + 1)
    sel = (r >= redge[0]) & (r < redge[-1])

    def h2(w):
        return np.histogram2d(r[sel], theta[sel], bins=[redge, tedge],
                              weights=w[sel])[0]

    A = h2(area)
    M = h2(area * dens)

    def area_mean(q):
        return np.divide(h2(area * q), A, out=np.full_like(A, np.nan),
                         where=A > 0)

    def mass_mean(q):
        return np.divide(h2(area * dens * q), M, out=np.full_like(A, np.nan),
                         where=M > 0)

    out = {
        'r_edge': redge, 'theta_edge': tedge,
        'area': A,
        'dens': np.divide(M, A, out=np.full_like(A, np.nan), where=A > 0),
        'velr': mass_mean(vr),
        'erad': area_mean(erad),
        'dxmin': np.histogram(r[sel], bins=redge, weights=dx[sel])[0] /
                 np.maximum(np.histogram(r[sel], bins=redge)[0], 1),
    }
    if kap is not None:
        out['kappa'] = mass_mean(kap)

    # ---- per-shell scalars: column direction and width
    rc = 0.5 * (redge[1:] + redge[:-1])
    tc = 0.5 * (tedge[1:] + tedge[:-1])
    tilt = tilt_of(ad)
    dmap = out['dens']
    scal = {k: np.full(nr, np.nan) for k in
            ['th_peak_N', 'th_peak_S', 'fwhm_N', 'fwhm_S',
             'dens_gaxis_N', 'dens_gaxis_S', 'dens_taxis_N', 'dens_taxis_S',
             'dens_peak_N', 'dens_peak_S', 'dens_bg_N', 'dens_bg_S']}
    for i in range(nr):
        prof = dmap[i]
        if np.all(~np.isfinite(prof)):
            continue
        for hemi, sgn in (('N', +1), ('S', -1)):
            ax = tilt if sgn > 0 else tilt - np.pi
            dth = (tc - ax + np.pi) % (2 * np.pi) - np.pi
            cone = np.abs(dth) < np.deg2rad(75)
            p = np.where(cone, prof, np.nan)
            if np.all(~np.isfinite(p)):
                continue
            ipk = np.nanargmax(p)
            pk = p[ipk]
            bgsel = (np.abs(dth) > np.deg2rad(45)) & (np.abs(dth) < np.deg2rad(75))
            bg = np.nanmedian(np.where(bgsel, prof, np.nan))
            half = bg + 0.5 * (pk - bg) if np.isfinite(bg) else 0.5 * pk
            above = np.isfinite(p) & (p > half)
            il = ir = ipk
            while il - 1 >= 0 and above[il - 1]:
                il -= 1
            while ir + 1 < nth and above[ir + 1]:
                ir += 1
            g = np.argmin(np.abs((tc - (0 if sgn > 0 else -np.pi) + np.pi)
                                 % (2 * np.pi) - np.pi))
            ta = np.argmin(np.abs(dth))
            scal['th_peak_' + hemi][i] = tc[ipk]
            scal['fwhm_' + hemi][i] = tc[ir] - tc[il]
            scal['dens_peak_' + hemi][i] = pk
            scal['dens_bg_' + hemi][i] = bg
            scal['dens_gaxis_' + hemi][i] = prof[g]
            scal['dens_taxis_' + hemi][i] = prof[ta]
    out.update(scal)
    out['r_center'] = rc
    out['theta_center'] = tc
    return out


# ------------------------------------------------------------------ profiles

def reduce_profiles(ad, bins=128):
    ad.rmin = max(2.0, 4.0 * float(ad.mb_dx.min()))
    ad.rmax = float(np.min(np.abs([ad.x1min, ad.x1max, ad.x2min, ad.x2max,
                                   ad.x3min, ad.x3max])))
    tilt = tilt_of(ad)
    # height along the (tilted) torus axis; theta_t measured from that axis
    ad.add_data_func('zt', lambda d: np.sin(tilt) * d('x') + np.cos(tilt) * d('z'))
    ad.add_data_func('theta_t', lambda d: np.arccos(
        np.clip(d('zt') / np.maximum(d('r'), 1e-30), -1, 1)))
    ad.add_data_func('mdot', lambda d: 4 * np.pi * d('r^2') * d('dens') * d('ur'))

    rng = [[ad.rmin, ad.rmax]]
    varl = ['dens', 'temp', 'pres', 'erad', 'b^2', 'beta', 'Begas',
            'mdot', 'dens*lor', 'ur', 'uph', 'vr_rel']
    common_kw = dict(bins=bins, scales='log', weights='vol', range=rng)
    ad.set_profile('r', varl=varl, key='r_vol', **common_kw)
    # wedges around the *torus* axis
    pol = ad.data('theta_t') < np.deg2rad(30)
    pols = ad.data('theta_t') > np.deg2rad(150)
    eqt = np.abs(ad.data('theta_t') - np.pi / 2) < np.deg2rad(30)
    ad.set_profile('r', varl=varl, key='r_pol_N', where=pol, **common_kw)
    ad.set_profile('r', varl=varl, key='r_pol_S', where=pols, **common_kw)
    ad.set_profile('r', varl=varl, key='r_eqt', where=eqt, **common_kw)
    # in/outflow decomposition of mdot
    ad.set_profile('r', varl=['mdot'], key='r_out', where=ad.data('ur') > 0,
                   **common_kw)
    ad.set_profile('r', varl=['mdot'], key='r_in', where=ad.data('ur') < 0,
                   **common_kw)
    # 2D (r, theta_t): where does the fast dense outflow sit in angle?
    varl2 = ['dens', 'temp', 'erad', 'mdot', 'dens*ur', 'b^2', 'Begas']
    ad.set_profile2d(['r', 'theta_t'], varl=varl2, key='rtheta_vol',
                     bins=[bins, bins // 2], weights='vol',
                     scales=['log', 'linear'],
                     range=[[ad.rmin, ad.rmax], [0.0, np.pi]])
    return ad


# ------------------------------------------------------------------ tasks

def reduce_tasks(ctx):
    '''Task registry for the generic reduce driver.'''
    slc = ctx.args.slice
    prefix = ctx.prefix()

    def column_out(num):
        return f'{ctx.pklpath}/column_{slc}.{num:05d}.pkl'

    def profile_out(num):
        return f'{ctx.pklpath}/Base.{num:05d}.pkl'

    def run_column(num):
        binfile = f'{ctx.binpath}/{prefix}.slice_{slc}.{num:05d}.bin'
        sigmafile = f'{ctx.binpath}/{prefix}.slice_{slc}_sigma.{num:05d}.bin'
        ad = load_snapshot(binfile, sigmafile, add_gr=False, do_setup=False)
        out = reduce_column(ad, slc)
        out.update(time=ad.time, num=num, slice_axis=slc,
                   tilt_angle_deg=float(np.rad2deg(tilt_of(ad))))
        fname = column_out(num)
        with open(fname, 'wb') as f:
            pickle.dump(out, f)
        print(f'[column] n={num} t={ad.time:g} -> {fname}', flush=True)

    def run_profile(num):
        binfile = f'{ctx.cbinpath}/{prefix}.full.{num:05d}.cbin'
        sigmafile = binfile.replace('full', 'rad_sigma')
        ad = load_snapshot(binfile, sigmafile, add_gr=True)
        reduce_profiles(ad, ctx.args.bins)
        fname = profile_out(num)
        ad.save(fname)
        print(f'[profile] n={num} t={ad.time:g} -> {fname}', flush=True)

    return {
        'c': ctx.task(name='column',
                      nums=lambda: ctx.snapshots(ctx.binpath,
                                                 f'{prefix}.slice_{slc}.', '.bin'),
                      outfile=column_out, run=run_column),
        'p': ctx.task(name='profile',
                      nums=lambda: ctx.snapshots(ctx.cbinpath,
                                                 f'{prefix}.full.', '.cbin'),
                      outfile=profile_out, run=run_profile),
    }
