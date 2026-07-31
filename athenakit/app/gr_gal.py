'''
Application module for GR galactic-scale accretion runs (gr_gal campaign).

Builds on app.acc (units, cooling, initial-condition tools) and provides
reduce tasks for the generic `athenakit-reduce` driver:
    p : radial profiles (vol/mass weighted, cold/hot phases, in/outflow,
        polar/equatorial wedges) and (r, X) phase histograms
        -> data/pkl/Base.NNNNN.pkl
'''
import numpy as np

from .. import load
from . import acc

# cold/hot split for profile decomposition
TEMP_SPLIT_K = 1.0e5

# thin-shell (theta,phi) sphere maps: nominal radii; per frame only those
# resolved (r >= 8 dx_min, inside domain) are reduced
SPHERE_RADII = (4.0, 1.0e1, 1.0e2, 1.0e3, 1.0e4, 1.0e5, 4.0e5)
SPHERE_WIDTH = 1.2      # shell spans [r/SPHERE_WIDTH, r*SPHERE_WIDTH]


def setup(ad):
    '''Units + cooling/potential data funcs; robust to missing IC headers.'''
    acc.add_tools(ad)   # sets ad.unit, ad.mu_h, ad.rin/rmin/rmax, tries IC solve
    acc.add_data(ad)    # cooling_rate/cooling_time, stresses, etc. (lazy funcs)
    # gr_gal uses an isothermal sigma^2 potential; override vkep if the
    # entropy-profile initial condition is unavailable
    sigma2 = ad.header('problem', 'sigma2', float, 0.0)
    if not hasattr(ad, 'rad_initial') and sigma2 > 0.0:
        ad.add_data_func('vkep', lambda d: np.sqrt(1.0 / d('r') + sigma2))
        ad.add_data_func('Omega', lambda d: d('vkep') / d('r'))
    ad.add_data_func('tff', lambda d: d('r') / d('vkep'))
    ad.add_data_func('mdot', lambda d: 4 * np.pi * d('r^2') * d('dens') * d('ur'))
    ad.add_data_func('temp_K', lambda d: d('temp') * d.ad.unit.temperature_cgs)
    # feedback power (positive = outward), notebook convention:
    #   Edot_fbk = -4 pi r^2 (rho u^r + T^r_t);  EM part from Tr_t_mhd-Tr_t_hydro
    ad.add_data_func('pfbk', lambda d: -4 * np.pi * d('r^2') *
                     (d('dens') * d('ur') + d('Tr_t_mhd')))
    ad.add_data_func('pfbk_hyd', lambda d: -4 * np.pi * d('r^2') *
                     (d('dens') * d('ur') + d('Tr_t_hydro')))
    ad.add_data_func('pfbk_mag', lambda d: -4 * np.pi * d('r^2') *
                     (d('Tr_t_mhd') - d('Tr_t_hydro')))
    return ad


def load_snapshot(binfile, add_gr=True, do_setup=True):
    ad = load(binfile)
    if add_gr:
        ad.add_gr_data()
    if do_setup:
        setup(ad)
    return ad


def reduce_profiles(ad, bins=128):
    ad.rmin = max(2.0, 4.0 * float(ad.mb_dx.min()))
    ad.rmax = float(np.min(np.abs([ad.x1min, ad.x1max, ad.x2min, ad.x2max,
                                   ad.x3min, ad.x3max])))
    rng = [[ad.rmin, ad.rmax]]
    varl = ['dens', 'temp', 'pres', 'entropy', 'mdot', 'ur', 'vr_rel', 'lor',
            'uph', 'b^2', 'beta', 'sigma_rel', 'Begas',
            'pfbk', 'pfbk_hyd', 'pfbk_mag',
            'cooling_rate', 'cooling_time', 'cooling_time/tff']
    common_kw = dict(bins=bins, scales='log', weights='vol', range=rng)
    ad.set_profile('r', varl=varl, key='r_vol', **common_kw)
    ad.set_profile('r', varl=varl, key='r_mass',
                   bins=bins, scales='log', weights='mass', range=rng)
    # cold / hot phases
    cold = ad.data('temp_K') < TEMP_SPLIT_K
    ad.set_profile('r', varl=varl, key='r_cold', where=cold, **common_kw)
    ad.set_profile('r', varl=varl, key='r_hot', where=~cold, **common_kw)
    # in/outflow decomposition
    flux = ['mdot', 'dens', 'ur', 'pfbk', 'pfbk_hyd', 'pfbk_mag', 'Begas']
    ad.set_profile('r', varl=flux, key='r_out',
                   where=ad.data('ur') > 0, **common_kw)
    ad.set_profile('r', varl=flux, key='r_in',
                   where=ad.data('ur') < 0, **common_kw)
    # polar / equatorial wedges (grid axis)
    pol = np.abs(np.cos(ad.data('theta'))) > np.cos(np.deg2rad(30))
    eqt = np.abs(ad.data('theta') - np.pi / 2) < np.deg2rad(30)
    ad.set_profile('r', varl=varl, key='r_pol', where=pol, **common_kw)
    ad.set_profile('r', varl=varl, key='r_eqt', where=eqt, **common_kw)
    # (r,theta): spatial/angular distribution of flow and feedback
    varl2 = ['dens', 'temp', 'velr', 'vrot', 'b^2', 'beta',
             'mdot', 'pfbk', 'pfbk_hyd', 'pfbk_mag']
    ad.set_profile2d(['r', 'theta'], varl=varl2, key='rtheta_vol',
                     bins=[bins, bins // 2], weights='vol',
                     scales=['log', 'linear'],
                     range=[[ad.rmin, ad.rmax], [0.0, np.pi]])
    # phase diagrams
    phase = [['r', 'dens'], ['r', 'temp'], ['r', 'entropy'],
             ['r', 'cooling_time'], ['r', 'sigma_rel+1e-12']]
    ad.set_hist2d(phase, bins=bins, weights='vol', scales='log')
    ad.set_hist2d(phase, bins=bins, weights='mass', scales='log')
    # thin-shell sphere maps (Mollweide-ready)
    dxmin = float(ad.mb_dx.min())
    r = ad.data('r')
    svarl = ['dens', 'temp', 'ur', 'beta', 'b^2', 'pfbk', 'pfbk_mag']
    for rs in SPHERE_RADII:
        if rs < 8 * dxmin or rs * SPHERE_WIDTH > ad.rmax:
            continue
        shell = (r > rs / SPHERE_WIDTH) & (r < rs * SPHERE_WIDTH)
        ad.set_profile2d(['theta', 'phi'], varl=svarl, key=f'sph_{rs:g}',
                         bins=[64, 128], weights='vol', scales='linear',
                         where=shell,
                         range=[[0.0, np.pi], [-np.pi, np.pi]])
    return ad


def reduce_tasks(ctx):
    prefix = ctx.prefix()

    def profile_out(num):
        return f'{ctx.pklpath}/Base.{num:05d}.pkl'

    def run_profile(num):
        binfile = f'{ctx.cbinpath}/{prefix}.full.{num:05d}.cbin'
        ad = load_snapshot(binfile, add_gr=True)
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
