import numpy as np
try:
    import cupy as xp
    xp.array(0)
except:
    import numpy as xp
from numpy.linalg import inv
from .. import units
from .. import kit
from ..athena_data import asnumpy

def add_tools(ad):
    ad.rmin = float(asnumpy(np.min(ad.data('r').min())))
    ad.rmax = float(np.min(np.abs([ad.x1min,ad.x1max,ad.x2min,ad.x2max,ad.x3min,ad.x3max])))
            
    return

def add_data(ad,add_bcc=True):
    if ('bcc1' not in ad.data_raw.keys()) and add_bcc:
        ad.add_data_func('bcc1', lambda sf : sf.data('zeros'))
        ad.add_data_func('bcc2', lambda sf : sf.data('zeros'))
        ad.add_data_func('bcc3', lambda sf : sf.data('zeros'))

    for var in ['mdot','mdotin','mdotout','momdot','momdotin','momdotout','ekdot','ekdotin','ekdotout']:
        ad.add_data_func(var, lambda sf, var=var : 4.0*xp.pi*sf.data('r')**2*sf.data(var.replace('dot','flxr')))

    return