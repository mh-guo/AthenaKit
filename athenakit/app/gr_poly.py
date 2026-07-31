'''
Application module for GR polytrope runs (gr_poly campaign).

Non-rotating spherical polytrope around a BH (fm_torus with l_peak = 0),
run as pure hydro, rad-hydro, or rad-MHD; all quantities are physics-aware.

Provides:
    setup(ad)            : units and common derived variables
    analytic_rho(ad, r)  : initial capped l=0 polytrope density profile
    reduce_profiles(ad)  : spherical radial profiles from a 3D (coarsened) dump
    reduce_tasks(ctx)    : task registry for the generic `athenakit-reduce` driver
        p : radial profiles -> data/pkl/Base.NNNNN.pkl
'''
import os
import numpy as np

from .. import units as units_mod
from .. import load


def has_mhd(ad):
    return ad.header('mhd', 'eos', str, '') != ''


def has_rad(ad):
    return ad.header('radiation', 'kappa_s', str, '') != ''


def setup(ad):
    '''Attach cgs units and common data funcs (mirrors app.rad_torus.setup).'''
    u = units_mod
    bhmass_msun = ad.header('units', 'bhmass_msun', float, 1.0e6)
    density_cgs = ad.header('units', 'density_cgs', float, 1.0)
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
    ad.mdotedd = 10.0 * ad.ledd  # Mdot_Edd = L_Edd/(0.1 c^2), as app.rad_torus
    ad.add_data_func('pi', lambda d: np.pi)
    ad.add_data_func('dunit', lambda d: d.ad.dunit)
    ad.add_data_func('lunit', lambda d: d.ad.lunit)
    ad.add_data_func('tempunit', lambda d: d.ad.tempunit)
    ad.add_data_func('Ledd', lambda d: d.ad.ledd)
    ad.add_data_func('Mdotedd', lambda d: d.ad.mdotedd)
    return ad


def analytic_rho(ad, r):
    '''Initial density of the capped l=0 polytrope (Schwarzschild, static).'''
    r_edge = ad.header('problem', 'r_edge', float)
    r_peak = ad.header('problem', 'r_peak', float)
    rho_max = ad.header('problem', 'rho_max', float)
    gam = ad.header('mhd' if has_mhd(ad) else 'hydro', 'gamma', float, 5.0 / 3.0)
    rc = np.maximum(r, r_peak)
    h = np.sqrt((1.0 - 2.0 / r_edge) / (1.0 - 2.0 / rc))
    h_pk = np.sqrt((1.0 - 2.0 / r_edge) / (1.0 - 2.0 / r_peak))
    rho = rho_max * ((h - 1.0) / (h_pk - 1.0))**(1.0 / (gam - 1.0))
    return np.where(r < r_edge, rho, 0.0)


def load_snapshot(binfile, sigmafile=None, add_gr=True, do_setup=True):
    ad = load(binfile)
    if sigmafile and os.path.isfile(sigmafile):
        ad.load(sigmafile)
    if add_gr:
        ad.add_gr_data()
    if do_setup:
        setup(ad)
    return ad


# ------------------------------------------------------------------ profiles

def reduce_profiles(ad, bins=128):
    ad.rmin = max(1.5, 2.0 * float(ad.mb_dx.min()))
    ad.rmax = float(np.min(np.abs([ad.x1min, ad.x1max, ad.x2min, ad.x2max,
                                   ad.x3min, ad.x3max])))
    ad.add_data_func('mdot', lambda d: 4 * np.pi * d('r^2') * d('dens') * d('ur'))

    varl = ['dens', 'temp', 'pres', 'entropy', 'mdot', 'ur', 'dens*lor', 'lor']
    if has_rad(ad):
        varl += ['erad', 'rr']
        if ('sigma_a' in ad.data_list):
            varl += ['sigma_a', 'sigma_s']
    if has_mhd(ad):
        varl += ['b^2', 'beta']

    rng = [[ad.rmin, ad.rmax]]
    common_kw = dict(bins=bins, scales='log', range=rng)
    ad.set_profile('r', varl=varl, key='r_vol', weights='vol', **common_kw)
    ad.set_profile('r', varl=varl, key='r_mass', weights='mass', **common_kw)
    # in/outflow decomposition of mdot
    ad.set_profile('r', varl=['mdot'], key='r_out', where=ad.data('ur') > 0,
                   weights='vol', **common_kw)
    ad.set_profile('r', varl=['mdot'], key='r_in', where=ad.data('ur') < 0,
                   weights='vol', **common_kw)
    # (r, theta) map to monitor departures from spherical symmetry
    varl2 = ['dens', 'temp', 'mdot', 'dens*ur']
    if has_rad(ad):
        varl2 += ['erad']
    if has_mhd(ad):
        varl2 += ['b^2']
    ad.set_profile2d(['r', 'theta'], varl=varl2, key='rtheta_vol',
                     bins=[bins, bins // 2], weights='vol',
                     scales=['log', 'linear'],
                     range=[[ad.rmin, ad.rmax], [0.0, np.pi]])
    return ad


# ------------------------------------------------------------------ tasks

def reduce_tasks(ctx):
    '''Task registry for the generic reduce driver.'''
    prefix = ctx.prefix()

    def profile_out(num):
        return f'{ctx.pklpath}/Base.{num:05d}.pkl'

    def run_profile(num):
        binfile = f'{ctx.cbinpath}/{prefix}.full.{num:05d}.cbin'
        sigmafile = binfile.replace('full', 'rad_sigma')
        ad = load_snapshot(binfile, sigmafile)
        reduce_profiles(ad, ctx.args.bins)
        fname = profile_out(num)
        ad.save(fname)
        print(f'[profile] n={num} t={ad.time:g} -> {fname}', flush=True)

    return {
        'p': ctx.task(name='profile',
                      nums=lambda: ctx.snapshots(ctx.cbinpath,
                                                 f'{prefix}.full.', '.cbin'),
                      outfile=profile_out, run=run_profile),
    }
