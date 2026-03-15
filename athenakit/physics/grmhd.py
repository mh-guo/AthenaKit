### GRMHD physics module: metrics, variables, etc.

from .. import global_vars
if (global_vars.cupy_enabled):
    import cupy as xp
else:
    import numpy as xp

##################################################################
### Functions for calculating quantities related to CKS metric ###
##################################################################

# Function for calculating quantities related to CKS metric
def cks_geometry(x, y, z, a):
    a2 = a ** 2
    z2 = z ** 2
    rr2 = x ** 2 + y ** 2 + z2
    # Kerr-Schild radius
    r2 = 0.5 * (rr2 - a2 + xp.sqrt((rr2 - a2) ** 2 + 4.0 * a2 * z2))
    r = xp.sqrt(r2)
    f = 2.0 * r2 * r / (r2 ** 2 + a2 * z2)
    lx = (r * x + a * y) / (r2 + a2)
    ly = (r * y - a * x) / (r2 + a2)
    lz = z / r
    gtt = -1.0 - f
    alpha2 = -1.0 / gtt
    alpha = xp.sqrt(alpha2)
    g_tt = -1.0 + f
    g_tx = f * lx
    g_ty = f * ly
    g_tz = f * lz
    g_xx = 1.0 + f * lx ** 2
    g_xy = f * lx * ly
    g_xz = f * lx * lz
    g_yy = 1.0 + f * ly ** 2
    g_yz = f * ly * lz
    g_zz = 1.0 + f * lz ** 2
    return alpha, g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz

# Function for calculating normal-frame Lorentz factor
def normal_lorentz(uux, uuy, uuz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz):
    uut = xp.sqrt(1.0 + g_xx * uux ** 2 + 2.0 * g_xy * uux * uuy
                  + 2.0 * g_xz * uux * uuz + g_yy * uuy ** 2
                  + 2.0 * g_yz * uuy * uuz + g_zz * uuz ** 2)
    return uut

# Function for transforming velocity from normal frame to coordinate frame
def norm_to_coord(uut, uux, uuy, uuz, alpha, g_tx, g_ty, g_tz):
    ut = uut / alpha
    ux = uux - alpha * uut * g_tx
    uy = uuy - alpha * uut * g_ty
    uz = uuz - alpha * uut * g_tz
    return ut, ux, uy, uz

# Function for transforming vector from contravariant to covariant components
def lower_vector(at, ax, ay, az,
                 g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz):
    a_t = g_tt * at + g_tx * ax + g_ty * ay + g_tz * az
    a_x = g_tx * at + g_xx * ax + g_xy * ay + g_xz * az
    a_y = g_ty * at + g_xy * ax + g_yy * ay + g_yz * az
    a_z = g_tz * at + g_xz * ax + g_yz * ay + g_zz * az
    return a_t, a_x, a_y, a_z

# Function for converting contravariant vector CKS components to SKS
def cks_to_sks_vec_con(ax, ay, az, x, y, z, a):
    a2 = a ** 2
    x2 = x ** 2
    y2 = y ** 2
    z2 = z ** 2
    rr2 = x2 + y2 + z2
    r2 = 0.5 * (rr2 - a2 + xp.sqrt((rr2 - a2) ** 2 + 4.0 * a2 * z2))
    r = xp.sqrt(r2)
    dr_dx = r * x / (2.0 * r2 - rr2 + a2)
    dr_dy = r * y / (2.0 * r2 - rr2 + a2)
    dr_dz = r * z * (1.0 + a2 / r2) / (2.0 * r2 - rr2 + a2)
    dth_dx = z / r * dr_dx / xp.sqrt(r2 - z2)
    dth_dy = z / r * dr_dy / xp.sqrt(r2 - z2)
    dth_dz = (z / r * dr_dz - 1.0) / xp.sqrt(r2 - z2)
    dph_dx = -y / (x2 + y2) + a / (r2 + a2) * dr_dx
    dph_dy = x / (x2 + y2) + a / (r2 + a2) * dr_dy
    dph_dz = a / (r2 + a2) * dr_dz
    ar = dr_dx * ax + dr_dy * ay + dr_dz * az
    ath = dth_dx * ax + dth_dy * ay + dth_dz * az
    aph = dph_dx * ax + dph_dy * ay + dph_dz * az
    return ar, ath, aph

# Function for converting covariant covector CKS components to SKS
def cks_to_sks_vec_cov(a_x, a_y, a_z, x, y, z, a):
    a2 = a ** 2
    z2 = z ** 2
    rr2 = x ** 2 + y ** 2 + z2
    r2 = 0.5 * (rr2 - a2 + xp.sqrt((rr2 - a2) ** 2 + 4.0 * a2 * z2))
    r = xp.sqrt(r2)
    th = xp.arccos(z / r)
    sth = xp.sin(th)
    cth = xp.cos(th)
    ph = xp.arctan2(y, x) - xp.arctan2(a, r)
    sph = xp.sin(ph)
    cph = xp.cos(ph)
    dx_dr = sth * cph
    dy_dr = sth * sph
    dz_dr = cth
    dx_dth = cth * (r * cph - a * sph)
    dy_dth = cth * (r * sph + a * cph)
    dz_dth = -r * sth
    dx_dph = sth * (-r * sph - a * cph)
    dy_dph = sth * (r * cph - a * sph)
    dz_dph = 0.0
    a_r = dx_dr * a_x + dy_dr * a_y + dz_dr * a_z
    a_th = dx_dth * a_x + dy_dth * a_y + dz_dth * a_z
    a_ph = dx_dph * a_x + dy_dph * a_y + dz_dph * a_z
    return a_r, a_th, a_ph

# Function for converting 3-magnetic field to 4-magnetic field
def three_to_four_field(bbx, bby, bbz, ut, ux, uy, uz, u_x, u_y, u_z):
    bt = u_x * bbx + u_y * bby + u_z * bbz
    bx = (bbx + bt * ux) / ut
    by = (bby + bt * uy) / ut
    bz = (bbz + bt * uz) / ut
    return bt, bx, by, bz

# Function for converting contravariant rank-2 tensor CKS components to SKS
def cks_to_sks_tens_con(axx, axy, axz, ayx, ayy, ayz, azx, azy, azz, x, y, z, a):
    axr, axth, axph = cks_to_sks_vec_con(axx, axy, axz, a, x, y, z)
    ayr, ayth, ayph = cks_to_sks_vec_con(ayx, ayy, ayz, a, x, y, z)
    azr, azth, azph = cks_to_sks_vec_con(azx, azy, azz, a, x, y, z)
    arr, athr, aphr = cks_to_sks_vec_con(axr, ayr, azr, a, x, y, z)
    arth, athth, aphth = cks_to_sks_vec_con(axth, ayth, azth, a, x, y, z)
    arph, athph, aphph = cks_to_sks_vec_con(axph, ayph, azph, a, x, y, z)
    return arr, arth, arph, athr, athth, athph, aphr, aphth, aphph

# Function to calculate the horizon radius
def r_horizon(a):
    return 1.0 + (1.0 - a ** 2) ** 0.5

##################################################################

# calculate the all the variables, slow but easy to understand and efficient
def variables(f, a):
    """
    Calculate all the variables used in the GRMHD simulation
    
    Parameters
    ----------
    f : function 
        A function that takes a string as input and returns the value of the variable
    a : float
        The spin parameter of the black hole

    Returns
    -------
    v : dict
        A dictionary containing all the variables
    """
    v = {}
    x, y, z = f('x'), f('y'), f('z')
    alpha, g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz = \
        cks_geometry(x, y, z, a)

    # Calculate relativistic velocity
    uux, uuy, uuz = f('velx'), f('vely'), f('velz')
    uut = normal_lorentz(uux, uuy, uuz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
    ut, ux, uy, uz = norm_to_coord(uut, uux, uuy, uuz, alpha, g_tx, g_ty, g_tz)
    u_t, u_x, u_y, u_z = lower_vector(ut, ux, uy, uz,\
        g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
    ur, uth, uph = cks_to_sks_vec_con(ux, uy, uz, x, y, z, a)
    u_r, u_th, u_ph = cks_to_sks_vec_cov(u_x, u_y, u_z, x, y, z, a)
    
    # Calculate relativistic magnetic field
    bbx, bby, bbz = f('bcc1'), f('bcc2'), f('bcc3')
    bt, bx, by, bz = three_to_four_field(bbx, bby, bbz, ut, ux, uy, uz, u_x, u_y, u_z)
    b_t, b_x, b_y, b_z = lower_vector(bt, bx, by, bz,\
        g_tt, g_tx, g_ty, g_tz, g_xx, g_xy, g_xz, g_yy, g_yz, g_zz)
    br, bth, bph = cks_to_sks_vec_con(bx, by, bz, x, y, z, a)
    b_r, b_th, b_ph = cks_to_sks_vec_cov(b_x, b_y, b_z, x, y, z, a)

    # vx, vy, vz = ux / ut, uy / ut, uz / ut
    # vr_rel, vth_rel, vph_rel = ur / ut, uth / ut, uph / ut
    # Br_rel = br * ut - bt * ur
    # Bth_rel = bth * ut - bt * uth
    # Bph_rel = bph * ut - bt * uph

    # # Calculate relativistic related quantity
    # dens = f('dens')
    # ugas = f('eint')
    # b2   = b_t * bt + b_x * bx + b_y * by + b_z * bz
    # pmag_rel = 0.5 * b2
    # pgas = (gamma_adi - 1.0) * ugas
    # beta_inv_rel = pmag_rel / pgas
    # sigma_rel = b2 / dens
    # wgas = dens + ugas + pgas
    # sigmah_rel = b2 / wgas
    # wmhd = wgas + b2
    # va_rel = xp.sqrt(b2 / wmhd)

    # # Calculate relativistic enthalpy density or Bernoulli parameter
    # Begas = -u_t * wgas / dens - 1.0
    # Bemhd = -u_t * wmhd / dens - 1.0

    # # Calculate relativistic conserved quantity
    # Tt_t_hydro = wgas * ut * u_t + pgas
    # Tt_x_hydro = wgas * ut * u_x
    # Tt_y_hydro = wgas * ut * u_y
    # Tt_z_hydro = wgas * ut * u_z
    # Tt_t_mhd   = wmhd * ut * u_t + pgas + pmag_rel - bt * b_t
    # Tt_x_mhd   = wmhd * ut * u_x - bt * b_x
    # Tt_y_mhd   = wmhd * ut * u_y - bt * b_y
    # Tt_z_mhd   = wmhd * ut * u_z - bt * b_z

    # # Calculate relativistic fluxes
    # Tr_t_hydro  = wgas * ur * u_t
    # Tr_ph_hydro = wgas * ur * u_ph
    # Tr_th_hydro = wgas * ur * u_th
    # Tr_t_mhd    = wmhd * ur * u_t - br * b_t
    # Tr_ph_mhd   = wmhd * ur * u_ph - br * b_ph
    # Tr_th_mhd   = wmhd * ur * u_th - br * b_th

    # Phi_flux    = 0.5 * xp.abs(Br_rel)

    # # Mask horizon for purposes of calculating colorbar limits
    # a2 = a ** 2
    # rr2 = x ** 2 + y ** 2 + z ** 2
    # r_horizon = 1.0 + (1.0 - a2) ** 0.5
    # rks = xp.sqrt(0.5 * (rr2 - a2 + xp.sqrt((rr2 - a2) ** 2 + 4.0 * a2 * z ** 2)))
    # horizon_mask = rks < r_horizon

    # TODO(@mhguo): Add radiation related quantities

    # TODO(@mhguo): consider what to include as raw data
    # TODO(@mhguo): also consider the name of the variables
    # Add necessary variables to the dictionary
    v['alpha'] = alpha
    v['g_tt'] = g_tt
    v['g_tx'] = g_tx
    v['g_ty'] = g_ty
    v['g_tz'] = g_tz
    v['g_xx'] = g_xx
    v['g_xy'] = g_xy
    v['g_xz'] = g_xz
    v['g_yy'] = g_yy
    v['g_yz'] = g_yz
    v['g_zz'] = g_zz
    v['uut'] = uut
    v['ut'] = ut
    v['ux'] = ux
    v['uy'] = uy
    v['uz'] = uz
    v['u_t'] = u_t
    v['u_x'] = u_x
    v['u_y'] = u_y
    v['u_z'] = u_z
    v['ur'] = ur
    v['uth'] = uth
    v['uph'] = uph
    v['u_r'] = u_r
    v['u_th'] = u_th
    v['u_ph'] = u_ph
    v['bt'] = bt
    v['bx'] = bx
    v['by'] = by
    v['bz'] = bz
    v['b_t'] = b_t
    v['b_x'] = b_x
    v['b_y'] = b_y
    v['b_z'] = b_z
    v['br'] = br
    v['bth'] = bth
    v['bph'] = bph
    v['b_r'] = b_r
    v['b_th'] = b_th
    v['b_ph'] = b_ph

    return v

#TODO: in principle, we can use the following function to calculate all the variables
def functions(a):
    # assuming we have x, y, z, dens, velx, vely, velz, eint, bcc1, bcc2, bcc3
    f = {}
    
    # coordinate transformation
    f['r^2'] = lambda d : d('x') ** 2 + d('y') ** 2 + d('z') ** 2
    f['rks'] = lambda d : xp.sqrt(0.5 * (d('r^2') - a ** 2
        + xp.sqrt((d('r^2') - a ** 2) ** 2 + 4.0 * a ** 2 * d('z') ** 2)))
    f['horizon_mask'] = lambda d : d('rks') < r_horizon(a)
    # if we calculate flux on constant rks surface, we need to use sqrtmdet as weight
    f['sqrtmdet'] = lambda d : d('rks') ** 2 + a ** 2 * (d('z')/d('rks')) ** 2

    # get geometry
    # TODO(@mhguo): the calculation is repeated, should be optimized
    for i,k in enumerate(['alpha', 'g_tt', 'g_tx', 'g_ty', 'g_tz',\
                          'g_xx', 'g_xy', 'g_xz', 'g_yy', 'g_yz', 'g_zz']):
        f[k] = lambda d, a=a, i=i : cks_geometry(a, d('x'), d('y'), d('z'))[i]
    f['uux'] =  lambda d : d('velx')
    f['uuy'] =  lambda d : d('vely')
    f['uuz'] =  lambda d : d('velz')
    f['bbx'] =  lambda d : d('bcc1')
    f['bby'] =  lambda d : d('bcc2')
    f['bbz'] =  lambda d : d('bcc3')
    f['lor'] =  lambda d : d('uut')
    f['vx']  =  lambda d : d('ux') / d('ut')
    f['vy']  =  lambda d : d('uy') / d('ut')
    f['vz']  =  lambda d : d('uz') / d('ut')
    # TODO(@mhguo): think about the name of the variables
    f['vr_rel'] = lambda d : d('ur') / d('ut')
    f['vth_rel'] = lambda d : d('uth') / d('ut')
    f['vph_rel'] = lambda d : d('uph') / d('ut')
    f['Br_rel'] = lambda d : d('br') * d('ut') - d('bt') * d('ur')
    f['Bth_rel'] = lambda d : d('bth') * d('ut') - d('bt') * d('uth')
    f['Bph_rel'] = lambda d : d('bph') * d('ut') - d('bt') * d('uph')

    f['b^2'] = lambda d : d('b_t') * d('bt') + d('b_x') * d('bx')\
                         + d('b_y') * d('by') + d('b_z') * d('bz')
    f['pmag_rel'] = lambda d : 0.5 * d('b^2')
    f['beta_inv_rel'] = lambda d : d('pmag_rel') / d('pgas')
    f['sigma_rel'] = lambda d : d('b^2') / d('dens')
    f['wgas'] = lambda d : d('dens') + d('eint') + d('pgas')
    f['sigmah_rel'] = lambda d : d('b^2') / d('wgas')
    f['wmhd'] = lambda d : d('wgas') + d('b^2')
    f['va_rel'] = lambda d : xp.sqrt(d('b^2') / d('wmhd'))
    f['Begas'] = lambda d : -d('u_t') * d('wgas') / d('dens') - 1.0
    f['Bemhd'] = lambda d : -d('u_t') * d('wmhd') / d('dens') - 1.0
    f['Tt_t_hydro'] = lambda d : d('wgas') * d('ut') * d('u_t') + d('pgas')
    f['Tt_x_hydro'] = lambda d : d('wgas') * d('ut') * d('u_x')
    f['Tt_y_hydro'] = lambda d : d('wgas') * d('ut') * d('u_y')
    f['Tt_z_hydro'] = lambda d : d('wgas') * d('ut') * d('u_z')
    f['Tt_r_hydro'] = lambda d : d('wgas') * d('ut') * d('u_r')
    f['Tt_th_hydro'] = lambda d : d('wgas') * d('ut') * d('u_th')
    f['Tt_ph_hydro'] = lambda d : d('wgas') * d('ut') * d('u_ph')
    f['Tt_t_mag'] = lambda d : d('b^2') * d('ut') * d('u_t') + d('pmag_rel') - d('bt') * d('b_t')
    f['Tt_x_mag'] = lambda d : d('b^2') * d('ut') * d('u_x') - d('bt') * d('b_x')
    f['Tt_y_mag'] = lambda d : d('b^2') * d('ut') * d('u_y') - d('bt') * d('b_y')
    f['Tt_z_mag'] = lambda d : d('b^2') * d('ut') * d('u_z') - d('bt') * d('b_z')
    f['Tt_r_mag'] = lambda d : d('b^2') * d('ut') * d('u_r') - d('bt') * d('b_r')
    f['Tt_th_mag'] = lambda d : d('b^2') * d('ut') * d('u_th') - d('bt') * d('b_th')
    f['Tt_ph_mag'] = lambda d : d('b^2') * d('ut') * d('u_ph') - d('bt') * d('b_ph')
    f['Tt_t_mhd'] = lambda d : d('wmhd') * d('ut') * d('u_t') + d('pgas')\
                             + d('pmag_rel') - d('bt') * d('b_t')
    f['Tt_x_mhd'] = lambda d : d('wmhd') * d('ut') * d('u_x') - d('bt') * d('b_x')
    f['Tt_y_mhd'] = lambda d : d('wmhd') * d('ut') * d('u_y') - d('bt') * d('b_y')
    f['Tt_z_mhd'] = lambda d : d('wmhd') * d('ut') * d('u_z') - d('bt') * d('b_z')
    f['Tt_r_mhd'] = lambda d : d('wmhd') * d('ut') * d('u_r') - d('bt') * d('b_r')
    f['Tt_th_mhd'] = lambda d : d('wmhd') * d('ut') * d('u_th') - d('bt') * d('b_th')
    f['Tt_ph_mhd'] = lambda d : d('wmhd') * d('ut') * d('u_ph') - d('bt') * d('b_ph')
    f['Tr_t_hydro'] = lambda d : d('wgas') * d('ur') * d('u_t')
    f['Tr_ph_hydro'] = lambda d : d('wgas') * d('ur') * d('u_ph')
    f['Tr_th_hydro'] = lambda d : d('wgas') * d('ur') * d('u_th')
    f['Tr_t_mag'] = lambda d : d('b^2') * d('ur') * d('u_t') - d('br') * d('b_t')
    f['Tr_ph_mag'] = lambda d : d('b^2') * d('ur') * d('u_ph') - d('br') * d('b_ph')
    f['Tr_th_mag'] = lambda d : d('b^2') * d('ur') * d('u_th') - d('br') * d('b_th')
    f['Tr_t_mhd'] = lambda d : d('wmhd') * d('ur') * d('u_t') - d('br') * d('b_t')
    f['Tr_ph_mhd'] = lambda d : d('wmhd') * d('ur') * d('u_ph') - d('br') * d('b_ph')
    f['Tr_th_mhd'] = lambda d : d('wmhd') * d('ur') * d('u_th') - d('br') * d('b_th')
    f['Phi_flux'] = lambda d : 0.5 * xp.abs(d('Br_rel'))

    return f
