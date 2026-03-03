import os
from pathlib import Path
import numpy as np
import h5py
import pickle
import warnings
from packaging.version import parse as version_parse

from matplotlib import pyplot as plt

from .io import read_binary
from .utils import eval_expr, save_dict_to_hdf5, load_dict_from_hdf5
from .physics import grmhd
from . import global_vars
if (global_vars.cupy_enabled):
    import cupy as xp
    xp.cuda.set_allocator(xp.cuda.MemoryPool().malloc)
else:
    xp = np
if (global_vars.mpi_enabled):
    from . import mpi

def load(filename,**kwargs):
    ad = AthenaData(filename)
    ad.load(filename,**kwargs)
    return ad

def asnumpy(arr):
    if (type(arr) is dict):
        return {k:asnumpy(v) for k,v in arr.items()}
    if (type(arr) is list):
        return [asnumpy(a) for a in arr]
    if (global_vars.cupy_enabled):
        return (xp.asnumpy(arr)).astype(float)
    else:
        return arr

class AthenaData:
    def __init__(self,num=0,version='1.0'):
        self.num=num
        self.version=version
        self._header={}
        self.binary={}
        self.coord={}
        self.data_raw={}
        self.data_func={}
        self.sums={}
        self.avgs={}
        self.hists={}
        self.profs={}
        self.slices={}
        self.spectra={}
        return

    @property
    def n(self):
        return self.num

    def load(self,filename,config=True,add_gr=False,**kwargs):
        self.filename=filename
        self.add_gr=add_gr # temporary flag to add GR data
        if (filename.endswith('.bin')):
            self.binary_name = filename
            self.load_binary(filename,**kwargs)
        elif (filename.endswith('.athdf')):
            self.athdf_name = filename
            self.load_athdf(filename,**kwargs)
        elif (filename.endswith(('.h5','.hdf5'))):
            self.hdf5_name = filename
            self.load_hdf5(filename,**kwargs)
        elif (filename.endswith('.pkl')):
            self.pkl_name = filename
            self.load_pickle(filename,**kwargs)
        else:
            raise ValueError(f"Unsupported file type: {filename.split('.')[-1]}")
        if (config):
            self.config()
        return

    def save(self,filename,except_keys=[],
             default_except_keys=['binary', 'h5file', 'h5dic', 'coord', 'data_raw', 'data_func', 'mb_list'],
             **kwargs):
        if (global_vars.rank!=0): return
        dic={}
        for k,v in self.__dict__.items():
            if (k not in except_keys+default_except_keys and not callable(v)):
                if(type(v) in [xp.ndarray]):
                    dic[k]=asnumpy(v)
                else:
                    dic[k]=v
        if (filename.endswith(('.h5','.hdf5'))):
            self.save_hdf5(dic,filename,**kwargs)
        elif (filename.endswith(('.p','.pkl'))):
            self.save_pickle(dic,filename,**kwargs)
        else:
            raise ValueError(f"Unsupported file type: {filename.split('.')[-1]}")
        return

    def load_binary(self,filename):
        self._load_from_binary(read_binary(filename))
        return

    def load_athdf(self,filename):
        self._load_from_athdf(filename)
        return
    
    def load_pickle(self,filename,**kwargs):
        self._load_from_dic(pickle.load(open(filename,'rb')),**kwargs)
        self._config_attrs_from_header()
        return

    def load_hdf5(self,filename,**kwargs):
        self._load_from_dic(load_dict_from_hdf5(filename),**kwargs)
        self._config_attrs_from_header()
        return

    def save_hdf5(self,dic,filename):
        save_dict_to_hdf5(dic,filename)
        return
    
    def save_pickle(self,dic,filename):
        pickle.dump(dic,open(filename,'wb'))
        return

    def _load_from_dic(self,dic,except_keys=['header', 'data', 'binary', 'coord', 'data_raw', 'mb_data']):
        for k,v in dic.items():
            if (k not in except_keys):
                self.__dict__[k]=v
        return

    def _load_from_binary(self,binary):
        self.binary = binary
        self._load_from_dic(self.binary)
        for var in self.binary['var_names']:
            self.data_raw[var]=xp.asarray(binary['mb_data'][var])
        self._config_header(self.binary['header'])
        self._config_attrs_from_header()
        return
    
    def _load_from_athdf(self,filename):
        h5file = h5py.File(filename, mode='r')
        self.h5file = h5file
        self.h5dic = load_dict_from_hdf5(filename)
        self._config_header(self.h5file.attrs['Header'])
        self._config_attrs_from_header()
        self.time = self.h5file.attrs['Time']
        self.cycle = self.h5file.attrs['NumCycles']
        self.n_mbs = self.h5file.attrs['NumMeshBlocks']
        # @mhguo: using numpy here because cupy would be very slow when
        # accessing mb_logical and mb_geometry later (in _data_raw_uniform())
        rank, size = global_vars.rank, global_vars.size
        my_mb_beg = rank * self.n_mbs // size + min(rank, self.n_mbs % size)
        my_mb_end = my_mb_beg + self.n_mbs // size + (rank < self.n_mbs % size)
        self.mb_list = np.arange(my_mb_beg, my_mb_end)
        self.mb_logical = np.append(self.h5dic['LogicalLocations'],self.h5dic['Levels'].reshape(-1,1),axis=1)
        self.mb_geometry = np.asarray([self.h5dic['x1f'][:,0],self.h5dic['x1f'][:,-1],
                                       self.h5dic['x2f'][:,0],self.h5dic['x2f'][:,-1],
                                       self.h5dic['x3f'][:,0],self.h5dic['x3f'][:,-1],]).T
        n_var_read = 0
        for ds_n,num in enumerate(self.h5file.attrs['NumVariables']):
            for i in range(num):
                var = self.h5file.attrs['VariableNames'][n_var_read+i].decode("utf-8")
                self.data_raw[var] = xp.asarray(self.h5dic[self.h5file.attrs['DatasetNames'][ds_n].decode("utf-8")][i][self.mb_list])
            n_var_read += num
        return

    def config(self):
        if (self.data_raw and not self.coord): self.config_coord()
        if (self.data_raw): self._config_data()
        self._config_data_func()
        self.path = str(Path(self.filename).parent)
        self.num = int(self.filename.split('.')[-2])
        # assuming use_e=True
        # assuming we have dens, velx, vely, velz, eint
        # TODO(@mhguo): add support for arbitrary variables
        return

    def _config_header(self, header):
        for line in [entry for entry in header]:
            if line.startswith('<'):
                block = line.strip('<').strip('>')
                self._header[block]={}
                continue
            key, value = line.split('=')
            self._header[block][key.strip()] = value

    def header(self, blockname, keyname, astype=str, default=None):
        blockname = blockname.strip()
        keyname = keyname.strip()
        if blockname in self._header.keys():
            if keyname in self._header[blockname].keys():
                if (astype==bool):
                    return self._header[blockname][keyname].lower().replace(' ','') not in ['f','false','0']
                return astype(self._header[blockname][keyname])
        warnings.warn(f'Warning: no parameter called {blockname}/{keyname}, return default value: {default}')
        return default
    
    def _config_attrs_from_header(self,include_ghost=False):
        self.nghost = self.header( 'mesh', 'nghost', int)
        self.Nx1 = self.header( 'mesh', 'nx1', int)
        self.Nx2 = self.header( 'mesh', 'nx2', int)
        self.Nx3 = self.header( 'mesh', 'nx3', int)
        self.nx1 = self.header( 'meshblock', 'nx1', int)
        self.nx2 = self.header( 'meshblock', 'nx2', int)
        self.nx3 = self.header( 'meshblock', 'nx3', int)
        self.x1min = self.header( 'mesh', 'x1min', float)
        self.x1max = self.header( 'mesh', 'x1max', float)
        self.x2min = self.header( 'mesh', 'x2min', float)
        self.x2max = self.header( 'mesh', 'x2max', float)
        self.x3min = self.header( 'mesh', 'x3min', float)
        self.x3max = self.header( 'mesh', 'x3max', float)

        self.is_gr = self.header('coord','general_rel',bool,False)
        self.spin = self.header('coord','a',float,0.0)
        self.is_mhd = 'mhd' in self._header.keys()
        self.is_rad = 'radiation' in self._header.keys()
        self.gamma=self.header('mhd','gamma',float,5/3) if self.is_mhd else self.header('hydro','gamma',float,5/3)
        # self.use_e=self.header('mhd','use_e',bool,True) if self.is_mhd else self.header('hydro','use_e',bool,True)
        # self.iso_cs=self.header('mhd','iso_sound_speed',float,0.0) if self.is_mhd else self.header('hydro','iso_sound_speed',float,0.0)

        return

    # squeeze the ghost cells into the physical mesh for diagnostics
    def include_ghost(self):
        # reset the mesh size to the original size
        self.Nx1 = self.header( 'mesh', 'nx1', int)
        self.Nx2 = self.header( 'mesh', 'nx2', int)
        self.Nx3 = self.header( 'mesh', 'nx3', int)
        self.nx1 = self.header( 'meshblock', 'nx1', int)
        self.nx2 = self.header( 'meshblock', 'nx2', int)
        self.nx3 = self.header( 'meshblock', 'nx3', int)
        # squeeze the ghost cells into the physical mesh
        self.Nx1 += 2*self.nghost*(self.Nx1//self.nx1)
        self.Nx2 += 2*self.nghost*(self.Nx2//self.nx2)
        self.Nx3 += 2*self.nghost*(self.Nx3//self.nx3)
        self.nx1 += 2*self.nghost
        self.nx2 += 2*self.nghost
        self.nx3 += 2*self.nghost
        return

    def config_coord(self):
        mb_geo, mb_list = self.mb_geometry, self.mb_list
        nc1, nc2, nc3 = self.nx1, self.nx2, self.nx3
        x=xp.swapaxes(xp.linspace(mb_geo[mb_list,0],mb_geo[mb_list,1],nc1+1),0,1)
        y=xp.swapaxes(xp.linspace(mb_geo[mb_list,2],mb_geo[mb_list,3],nc2+1),0,1)
        z=xp.swapaxes(xp.linspace(mb_geo[mb_list,4],mb_geo[mb_list,5],nc3+1),0,1)
        x,y,z=0.5*(x[:,:-1]+x[:,1:]),0.5*(y[:,:-1]+y[:,1:]),0.5*(z[:,:-1]+z[:,1:])
        ZYX=xp.swapaxes(xp.asarray([xp.meshgrid(z[i],y[i],x[i],indexing='ij') for i in range(len(mb_list))]),0,1)
        self.coord['x'],self.coord['y'],self.coord['z']=ZYX[2],ZYX[1],ZYX[0]
        dx=xp.asarray([xp.full((nc3,nc2,nc1),(mb_geo[i,1]-mb_geo[i,0])/nc1) for i in mb_list])
        dy=xp.asarray([xp.full((nc3,nc2,nc1),(mb_geo[i,3]-mb_geo[i,2])/nc2) for i in mb_list])
        dz=xp.asarray([xp.full((nc3,nc2,nc1),(mb_geo[i,5]-mb_geo[i,4])/nc3) for i in mb_list])
        self.coord['dx'],self.coord['dy'],self.coord['dz']=dx,dy,dz
        return

    ### data handling ###
    def add_data(self,name,data):
        self.data_raw[name]=data
        return

    def add_gr_data(self):
        self.data_raw.update(grmhd.variables(self.data,self.spin))
        return

    def add_data_func(self,name,func):
        self.data_func[name]=func
        return
    
    # add extra raw data
    def _config_data(self):
        if (self.is_gr and self.add_gr):
            # print('Adding GR data')
            self.add_gr_data()
        return

    def _config_data_func(self):
        self.data_func['zeros'] = lambda d : xp.zeros(d(list(d.ad.data_raw.keys())[0]).shape)
        self.data_func['ones'] = lambda d : xp.ones(d(list(d.ad.data_raw.keys())[0]).shape)
        self.data_func['vol'] = lambda d : d('dx')*d('dy')*d('dz')
        self.data_func['r'] = lambda d : xp.sqrt(d('x')**2+d('y')**2+d('z')**2)
        self.data_func['R'] = lambda d : xp.sqrt(d('x')**2+d('y')**2)
        self.data_func['theta'] = lambda d : xp.arccos(d('z')/d('r'))
        self.data_func['phi'] = lambda d : xp.arctan2(d('y'),d('x'))
        self.data_func['mass'] = lambda d : d('vol')*d('dens')
        self.data_func['pres'] = lambda d : (d.ad.gamma-1)*d('eint')
        self.data_func['pgas'] = lambda d : d('pres')
        self.data_func['temp'] = lambda d : d('pres')/d('dens')
        self.data_func['entropy'] = lambda d : d('pres')/d('dens')**d.ad.gamma
        self.data_func['c_s^2'] = lambda d : d.ad.gamma*d('pres')/d('dens')
        self.data_func['c_s'] = lambda d : xp.sqrt(d('c_s^2'))
        self.data_func['momx'] = lambda d : d('dens')*d('velx')
        self.data_func['momy'] = lambda d : d('dens')*d('vely')
        self.data_func['momz'] = lambda d : d('dens')*d('velz')
        self.data_func['velr'] = lambda d : (d('velx')*d('x')+d('vely')*d('y')+d('velz')*d('z'))/d('r')
        self.data_func['velR'] =  lambda d : (d('x')*d('velx')+d('y')*d('vely'))/d('R')
        self.data_func['vtheta'] = lambda d : (d('z')*d('velr')-d('r')*d('velz'))/d('R')
        self.data_func['vphi'] = lambda d : (d('x')*d('vely')-d('y')*d('velx'))/d('R')
        self.data_func['momr'] = lambda d : d('dens')*d('velr')
        self.data_func['velrin'] = lambda d : xp.minimum(d('velr'),0.0)
        self.data_func['velrout'] = lambda d : xp.maximum(d('velr'),0.0)
        self.data_func['velin'] = lambda d : d('velrin')
        self.data_func['velout'] = lambda d : d('velrout')
        self.data_func['vtot^2'] = lambda d : d('velx')**2+d('vely')**2+d('velz')**2
        self.data_func['vtot'] = lambda d : xp.sqrt(d('vtot^2'))
        self.data_func['vrot'] = lambda d : xp.sqrt(d('vtot^2')-d('velr')**2)
        self.data_func['momtot'] = lambda d : d('dens')*d('vtot')
        self.data_func['ekin'] = lambda d : 0.5*d('dens')*d('vtot^2')
        self.data_func['egas'] = lambda d : d('ekin')+d('eint')
        self.data_func['amx'] = lambda d : d('y')*d('velz')-d('z')*d('vely')
        self.data_func['amy'] = lambda d : d('z')*d('velx')-d('x')*d('velz')
        self.data_func['amz'] = lambda d : d('x')*d('vely')-d('y')*d('velx')
        self.data_func['amtot'] = lambda d : d('r')*d('vrot')
        self.data_func['mflxr'] = lambda d : d('dens')*d('velr')
        self.data_func['mflxrin'] = lambda d : d('dens')*d('velin')
        self.data_func['mflxrout'] = lambda d : d('dens')*d('velout')
        self.data_func['momflxr'] = lambda d : d('dens')*d('velr')**2
        self.data_func['momflxrin'] = lambda d : d('dens')*d('velr')*d('velin')
        self.data_func['momflxrout'] = lambda d : d('dens')*d('velr')*d('velout')
        self.data_func['eiflxr'] = lambda d : d('eint')*d('velr')
        self.data_func['eiflxrin'] = lambda d : d('eint')*d('velin')
        self.data_func['eiflxrout'] = lambda d : d('eint')*d('velout')
        self.data_func['ekflxr'] = lambda d : d('dens')*.5*d('vtot^2')*d('velr')
        self.data_func['ekflxrin'] = lambda d : d('dens')*.5*d('vtot^2')*d('velin')
        self.data_func['ekflxrout'] = lambda d : d('dens')*.5*d('vtot^2')*d('velout')
        if not self.is_mhd:
            for var in ('bcc1','bcc2','bcc3'):
                self.data_func[var] = lambda d : d('zeros')
        self.data_func['bccx'] = lambda d : d('bcc1')
        self.data_func['bccy'] = lambda d : d('bcc2')
        self.data_func['bccz'] = lambda d : d('bcc3')
        self.data_func['bccr'] = lambda d : (d('bccx')*d('x')+d('bccy')*d('y')+d('bccz')*d('z'))/d('r')
        self.data_func['bccR'] = lambda d : (d('x')*d('bccx')+d('y')*d('bccy'))/d('R')
        self.data_func['btheta'] = lambda d : (d('z')*d('bccr')-d('r')*d('bccz'))/d('R')
        self.data_func['bphi'] = lambda d : (d('x')*d('bccy')-d('y')*d('bccx'))/d('R')
        self.data_func['btot^2'] = lambda d : d('bccx')**2+d('bccy')**2+d('bccz')**2
        self.data_func['btot'] = lambda d : xp.sqrt(d('btot^2'))
        self.data_func['brot'] = lambda d : xp.sqrt(d('btot^2')-d('bccr')**2)
        self.data_func['v_A^2'] = lambda d : d('btot^2')/d('dens')
        self.data_func['v_A'] = lambda d : xp.sqrt(d('v_A^2'))
        self.data_func['pmag'] = lambda d : 0.5*d('btot^2')
        self.data_func['emag'] = lambda d : 0.5*d('btot^2')
        self.data_func['beta'] = lambda d : d('pgas')/d('pmag')
        self.data_func['1/beta'] = lambda d : d('pmag')/d('pgas')
        self.data_func['ptot'] = lambda d : d('pres')+d('pmag') if d.ad.is_mhd else d('pres')
        self.data_func['etot'] = lambda d : d('ekin')+d('eint')+d('emag') if d.ad.is_mhd\
                                               else d('ekin')+d('eint')
        # radiaton
        if (self.is_rad and 'r00' in self.data_raw.keys()):
            # radial flux
            self.data_func['rr'] = lambda d : (d('r01')*d('x')+d('r02')*d('y')+d('r03')*d('z'))/d('r')
            # radial flux in fluid frame
            self.data_func['rr_ff'] = lambda d : (d('r01_ff')*d('x')+d('r02_ff')*d('y')+d('r03_ff')*d('z'))/d('r')
        if (self.is_gr):
            self.data_func.update(grmhd.functions(self.spin))
        return

    @property
    def data_list(self):
        return list(self.coord.keys())+list(self.data_raw.keys())+list(self.data_func.keys())

    def data(self,var,**kwargs):
        if (type(var) is str):
            # coordinate
            if (var in self.coord.keys()):
                if (kwargs.get('dtype')=='uniform'):
                    return self._coord_uniform(var,**kwargs)
                return self.coord[var]#[kwargs.get('mbs')]
            # raw data
            elif (var in self.data_raw.keys()):
                if (kwargs.get('dtype')=='uniform'):
                    return self._data_raw_uniform(var,**kwargs)
                return self.data_raw[var]#[kwargs.get('mbs')]
            # derived data
            elif (var in self.data_func.keys()):
                data = lambda v:self.data(v,**kwargs)
                data.ad = self
                data.kwargs = kwargs
                return self.data_func[var](data)
            elif (var.isidentifier()):
                raise ValueError(f"No variable callled '{var}' ")
            # math expression
            else:
                # Replace '^' with '**' for exponentiation
                # The reason is we never use '^' for XOR in math but we may need to change it
                expr = var.replace('^', '**')
                # Evaluate the expression
                return eval_expr(expr,lambda v:self.data(v,**kwargs))
        elif (type(var) in [list,tuple]):
            return [self.data(v,**kwargs) for v in var]
        elif (type(var) in [int,float,xp.ndarray]):
            return var # the variable itself, useful to interface with other functions
        else:
            raise ValueError(f"var '{var}' not supported")

    # an alias for data
    def d(self,var,**kwargs):
        return self.data(var,**kwargs)

    @property
    def mb_dx(self):
        mb_geo = self.mb_geometry
        return np.asarray([(mb_geo[:,1]-mb_geo[:,0])/self.nx1,
                           (mb_geo[:,3]-mb_geo[:,2])/self.nx2,
                           (mb_geo[:,5]-mb_geo[:,4])/self.nx3]).T

    ### get data in a single array ###
    def _cell_info(self,level=0,xyz=[]):
        if (not xyz):
            xyz = [self.x1min,self.x1max,self.x2min,self.x2max,self.x3min,self.x3max]
        # level is physical level
        nx1_fac = 2**level*self.Nx1/(self.x1max-self.x1min)
        nx2_fac = 2**level*self.Nx2/(self.x2max-self.x2min)
        nx3_fac = 2**level*self.Nx3/(self.x3max-self.x3min)
        i_min = int((xyz[0]-self.x1min)*nx1_fac)
        i_max = int(np.ceil((xyz[1]-self.x1min)*nx1_fac))
        j_min = int((xyz[2]-self.x2min)*nx2_fac)
        j_max = int(np.ceil((xyz[3]-self.x2min)*nx2_fac))
        k_min = int((xyz[4]-self.x3min)*nx3_fac)
        k_max = int(np.ceil((xyz[5]-self.x3min)*nx3_fac))
        dx = (xyz[1]-xyz[0])/(i_max-i_min)
        dy = (xyz[3]-xyz[2])/(j_max-j_min)
        dz = (xyz[5]-xyz[4])/(k_max-k_min)
        xf=xp.linspace(xyz[0],xyz[1],i_max-i_min+1)
        yf=xp.linspace(xyz[2],xyz[3],j_max-j_min+1)
        zf=xp.linspace(xyz[4],xyz[5],k_max-k_min+1)
        return xf,yf,zf,dx,dy,dz
    
    def _cell_faces(self,level=0,xyz=[]):
        return self._cell_info(level,xyz)[:3]
    
    def _cell_centers(self,level=0,xyz=[]):
        xf,yf,zf=self._cell_faces(level,xyz)
        xc,yc,zc=0.5*(xf[:-1]+xf[1:]),0.5*(yf[:-1]+yf[1:]),0.5*(zf[:-1]+zf[1:])
        return xc,yc,zc
    
    def _cell_length(self,level=0,xyz=[]):
        return self._cell_info(level,xyz)[3:]

    def _xyz_uniform(self,level=0,xyz=[]):
        xc,yc,zc=self._cell_centers(level,xyz)
        ZYX=xp.meshgrid(zc,yc,xc,indexing='ij')
        return ZYX[2],ZYX[1],ZYX[0]
    
    def _coord_uniform(self,var,level=0,xyz=[],**kwargs):
        xc,yc,zc=self._cell_centers(level,xyz)
        dx,dy,dz=self._cell_length(level,xyz)
        if (var=='x'): return xp.meshgrid(zc,yc,xc,indexing='ij')[2]
        if (var=='y'): return xp.meshgrid(zc,yc,xc,indexing='ij')[1]
        if (var=='z'): return xp.meshgrid(zc,yc,xc,indexing='ij')[0]
        if (var=='dx'): return xp.full((zc.size,yc.size,xc.size),dx)
        if (var=='dy'): return xp.full((zc.size,yc.size,xc.size),dy)
        if (var=='dz'): return xp.full((zc.size,yc.size,xc.size),dz)
        raise ValueError(f"var '{var}' not supported")

    def _data_raw_uniform(self,var,level=0,xyz=[],**kwargs):
        if (not xyz):
            xyz = [self.x1min,self.x1max,self.x2min,self.x2max,self.x3min,self.x3max]
        # block_level is physical level of mesh refinement
        physical_level = level
        nx1_fac = 2**level*self.Nx1/(self.x1max-self.x1min)
        nx2_fac = 2**level*self.Nx2/(self.x2max-self.x2min)
        nx3_fac = 2**level*self.Nx3/(self.x3max-self.x3min)
        i_min = int((xyz[0]-self.x1min)*nx1_fac)
        i_max = int(np.ceil((xyz[1]-self.x1min)*nx1_fac))
        j_min = int((xyz[2]-self.x2min)*nx2_fac)
        j_max = int(np.ceil((xyz[3]-self.x2min)*nx2_fac))
        k_min = int((xyz[4]-self.x3min)*nx3_fac)
        k_max = int(np.ceil((xyz[5]-self.x3min)*nx3_fac))
        data = xp.zeros((k_max-k_min, j_max-j_min, i_max-i_min))
        raw = self.data(var)
        for nraw,nmb in enumerate(self.mb_list):
            block_level = self.mb_logical[nmb,-1]
            block_loc = self.mb_logical[nmb,:3]
            block_data = raw[nraw]
            dimz, dimy, dimx = np.array(block_data.shape) > 1

            # Prolongate coarse data and copy same-level data
            if (block_level <= physical_level):
                s = int(2**(physical_level - block_level))
                # Calculate destination indices, without selection
                il_d = block_loc[0] * self.nx1 * s if dimx else 0
                jl_d = block_loc[1] * self.nx2 * s if dimy else 0
                kl_d = block_loc[2] * self.nx3 * s if dimz else 0
                iu_d = il_d + self.nx1 * s if dimx else 1
                ju_d = jl_d + self.nx2 * s if dimy else 1
                ku_d = kl_d + self.nx3 * s if dimz else 1
                # Calculate (prolongated) source indices, with selection
                il_s = max(il_d, i_min) - il_d
                jl_s = max(jl_d, j_min) - jl_d
                kl_s = max(kl_d, k_min) - kl_d
                iu_s = min(iu_d, i_max) - il_d
                ju_s = min(ju_d, j_max) - jl_d
                ku_s = min(ku_d, k_max) - kl_d
                if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                    continue
                # Account for selection in destination indices
                il_d = max(il_d, i_min) - i_min
                jl_d = max(jl_d, j_min) - j_min
                kl_d = max(kl_d, k_min) - k_min
                iu_d = min(iu_d, i_max) - i_min
                ju_d = min(ju_d, j_max) - j_min
                ku_d = min(ku_d, k_max) - k_min
                if s > 1:
                    # TODO(@mhguo): this seems to be the bottleneck of performance
                    # Only prolongate selected data
                    kl_r = kl_s // s
                    ku_r = min(ku_s // s + 1, self.nx3)
                    jl_r = jl_s // s
                    ju_r = min(ju_s // s + 1, self.nx2)
                    il_r = il_s // s
                    iu_r = min(iu_s // s + 1, self.nx1)
                    kl_s = kl_s - kl_r * s
                    ku_s = ku_s - kl_r * s
                    jl_s = jl_s - jl_r * s
                    ju_s = ju_s - jl_r * s
                    il_s = il_s - il_r * s
                    iu_s = iu_s - il_r * s
                    block_data = block_data[kl_r:ku_r,jl_r:ju_r,il_r:iu_r]
                    if dimx:
                        block_data = xp.repeat(block_data, s, axis=2)
                    if dimy:
                        block_data = xp.repeat(block_data, s, axis=1)
                    if dimz:
                        block_data = xp.repeat(block_data, s, axis=0)
                data[kl_d:ku_d,jl_d:ju_d,il_d:iu_d]=block_data[kl_s:ku_s,jl_s:ju_s,il_s:iu_s]
            # Restrict fine data, volume average
            else:
                # Calculate scale
                s = int(2 ** (block_level - physical_level))
                # Calculate destination indices, without selection
                il_d = int(block_loc[0] * self.nx1 / s) if dimx else 0
                jl_d = int(block_loc[1] * self.nx2 / s) if dimy else 0
                kl_d = int(block_loc[2] * self.nx3 / s) if dimz else 0
                iu_d = int(il_d + self.nx1 / s) if dimx else 1
                ju_d = int(jl_d + self.nx2 / s) if dimy else 1
                ku_d = int(kl_d + self.nx3 / s) if dimz else 1
                #print(kl_d,ku_d,jl_d,ju_d,il_d,iu_d)
                # Calculate (restricted) source indices, with selection
                il_s = max(il_d, i_min) - il_d
                jl_s = max(jl_d, j_min) - jl_d
                kl_s = max(kl_d, k_min) - kl_d
                iu_s = min(iu_d, i_max) - il_d
                ju_s = min(ju_d, j_max) - jl_d
                ku_s = min(ku_d, k_max) - kl_d
                if il_s >= iu_s or jl_s >= ju_s or kl_s >= ku_s:
                    continue
                # Account for selection in destination indices
                il_d = max(il_d, i_min) - i_min
                jl_d = max(jl_d, j_min) - j_min
                kl_d = max(kl_d, k_min) - k_min
                iu_d = min(iu_d, i_max) - i_min
                ju_d = min(ju_d, j_max) - j_min
                ku_d = min(ku_d, k_max) - k_min
                
                # Account for restriction in source indices
                num_extended_dims = 0
                if dimx:
                    il_s *= s
                    iu_s *= s
                if dimy:
                    jl_s *= s
                    ju_s *= s
                if dimz:
                    kl_s *= s
                    ku_s *= s
                
                # Calculate fine-level offsets
                io_s = s if block_data.shape[-1] > 1 else 1
                jo_s = s if block_data.shape[-2] > 1 else 1
                ko_s = s if block_data.shape[-3] > 1 else 1

                # Assign values
                # TODO(@mhguo): arithmetic mean may fail when fine-level is much finer than coarse-level
                data[kl_d:ku_d,jl_d:ju_d,il_d:iu_d] = block_data[kl_s:ku_s,jl_s:ju_s,il_s:iu_s]\
                    .reshape(ku_d-kl_d,ko_s,ju_d-jl_d,jo_s,iu_d-il_d,io_s)\
                    .mean(axis=(1,3,5))
                continue

                for ko in ko_vals:
                    for jo in jo_vals:
                        for io in io_vals:
                            data[kl_d:ku_d,jl_d:ju_d,il_d:iu_d] += block_data[
                                                                kl_s+ko:ku_s:s,
                                                                jl_s+jo:ju_s:s,
                                                                il_s+io:iu_s:s]\
                                                                /(s**num_extended_dims)
        # TODO(@mhguo): may change to device-to-device communication in the future
        if (global_vars.mpi_enabled):
            # TODO(@mhguo): assuming the mesh blocks are not overlapped
            return xp.asarray(mpi.sum(np.ascontiguousarray(asnumpy(data))))
        return data

    def _axis_index(self,axis):
        if (axis is None): return None
        if (type(axis) is int): return axis
        if (axis=='z'): return -3
        if (axis=='y'): return -2
        if (axis=='x'): return -1
        if (axis=='x3'): return -3
        if (axis=='x2'): return -2
        if (axis=='x1'): return -1
        raise ValueError(f"axis '{axis}' not supported")

    # helper functions similar to numpy/cupy
    def sum(self,var,weights=1.0,where=None,**kwargs):
        arr = asnumpy(xp.sum((self.data(var)*self.data(weights))[where],**kwargs))
        # TODO(@mhguo): make a better type conversion
        #print(var, arr, arr.dtype)
        if (global_vars.mpi_enabled):
            arr = np.array(float(arr))
            arr = mpi.sum(np.ascontiguousarray(arr))
            arr = float(arr)
        return arr
    def average(self,var,weights='ones',where=None,**kwargs):
        if (global_vars.mpi_enabled):
            raise NotImplementedError("average with MPI is not supported yet")
        return asnumpy(xp.average(self.data(var)[where],weights=self.data(weights)[where],**kwargs))

    # def derivative(self,f,x,axis=None,edge_order=1,**kwargs):
    #     f = self.data(f) if (type(f) is str) else f
    #     x = self.data(x) if (type(x) is str) else x
    #     axis = self._axis_index(axis)
    #     return xp.gradient(f,x,axis=axis,edge_order=edge_order,**kwargs)

    def gradient(self,f,axis=None,edge_order=1,**kwargs):
        f = self.data(f,**kwargs)
        axis = self._axis_index(axis)
        mb_dx, mb_list = self.mb_dx[:,::-1], self.mb_list
        if (kwargs.get('dtype')=='uniform'):
            dx = self._cell_length(kwargs.get('level',0),kwargs.get('xyz',[]))[::-1]
            if (axis is None):
                return xp.asarray(xp.gradient(f,*dx,edge_order=edge_order))
            else:
                return xp.gradient(f,dx[axis],axis=axis,edge_order=edge_order)
        if (axis is None):
            return xp.asarray([xp.gradient(d,*dx,edge_order=edge_order) for d,dx in zip(f,mb_dx[mb_list])]).swapaxes(0,1)
        else:
            return xp.asarray([xp.gradient(d,dx[axis],axis=axis,edge_order=edge_order) for d,dx in zip(f,mb_dx[mb_list])])
        
    def divergence(self,fx,fy,fz,**kwargs):
        fx,fy,fz = self.data(fx,**kwargs),self.data(fy,**kwargs),self.data(fz,**kwargs)
        return self.gradient(fx,'x',**kwargs)+self.gradient(fy,'y',**kwargs)+self.gradient(fz,'z',**kwargs)

    # kernal functions for histograms and profiles
    def _set_bins(self,var,bins,range,scale,where):
        if (type(bins) is int):
            if (scale=='linear'):
                if (range is None):
                    dat = self.data(var)[where]
                    dat = dat[xp.isfinite(dat)]
                    dmin,dmax = np.array(1e200),np.array(-1e200)
                    if (dat.size>0):
                        dmin,dmax = asnumpy(dat.min()),asnumpy(dat.max())
                    if (global_vars.mpi_enabled): dmin, dmax = mpi.min(dmin), mpi.max(dmax)
                    if (dmin>dmax):
                        warnings.warn(f"Warning: no bins for {var}, setting to default")
                        dmin,dmax = 0,1
                    return xp.linspace(float(dmin),float(dmax),bins+1)
                else:
                    return xp.linspace(range[0],range[1],bins+1)
            elif (scale=='log'):
                if (range is None):
                    dat = self.data(var)[where]
                    dat = dat[xp.isfinite(dat)]
                    dat = dat[dat>0.0]
                    dmin,dmax = np.array(1e200),np.array(1e-200)
                    if (dat.size>0):
                        dmin,dmax = asnumpy(dat.min()),asnumpy(dat.max())
                    if (global_vars.mpi_enabled): dmin, dmax = mpi.min(dmin), mpi.max(dmax)
                    if (dmin>dmax):
                        warnings.warn(f"Warning: no bins for {var}, setting to default")
                        dmin,dmax = 1,10
                    return xp.logspace(xp.log10(float(dmin)),xp.log10(float(dmax)),bins+1)
                else:
                    return xp.logspace(xp.log10(range[0]),xp.log10(range[1]),bins+1)
            else:
                raise ValueError(f"scale '{scale}' not supported")
        return bins

    def _histograms(self,varll,bins=10,range=None,weights=None,scales='linear',where=None,**kwargs):
        """
        Compute the histogram of a list of variables

        Parameters
        ----------
        varll : list of list of str
            varaible list

        Returns
        -------
        hist : dict
            dictionary of distributions
        """
        if (type(varll) is str):
            varll = [varll]
        for i,varl in enumerate(varll):
            if (type(varl) is str):
                varll[i] = [varl]
        if (type(weights) is str):
            weights = self.data(weights)[where].ravel()
        if (type(scales) is str):
            scales = [scales]*len(varll)

        hists = {}
        bins_0 = bins
        range_0 = range
        for varl,scale in zip(varll,scales):
            arr = [self.data(v)[where].ravel() for v in varl]
            histname = ','.join(varl)
            # get bins
            if (type(bins_0) is int):
                bins = [bins_0]*len(varl)
            scale = [scale]*len(varl) if (type(scale) is str) else scale
            range = [range_0]*len(varl) if (range_0 is None) else range_0
            for i,v in enumerate(varl):
                bins[i] = self._set_bins(v,bins[i],range[i],scale[i],where)
            # bins = xp.asarray(bins)
            # get histogram
            hist = xp.histogramdd(arr,bins=bins,weights=weights,**kwargs)
            hists[histname] = {'dat':xp.asarray(hist[0]),'edges':{v:_ for v,_ in zip(varl,hist[1])}}
            hists[histname]['centers'] = {v:(hists[histname]['edges'][v][:-1]+hists[histname]['edges'][v][1:])/2 for v in varl}
        hists = asnumpy(hists)
        if (global_vars.mpi_enabled):
            for k in hists.keys():
                hists[k]['dat'] = mpi.sum(np.ascontiguousarray(hists[k]['dat']))
        return hists

    # TODO(@mhguo): add a parameter data_func=self.data
    def _profiles(self,bin_varl,varl,bins=10,range=None,weights=None,scales='linear',where=None,data=None,**kwargs):
        """
        Compute the profile of a (list of) variable with respect to one or more bin variables.

        Parameters
        ----------
        bin_varl : str or list of str
            bin varaible list
        
        varl : str or list of str
            variable list

        Returns
        -------
        profs : dict
            dictionary of profiles
        """
        if (data is None):
            data = lambda sf,var : sf.data(var)
        if (type(bin_varl) is str):
            bin_varl = [bin_varl]
        if (type(varl) is str):
            varl = [varl]
        varl = list(dict.fromkeys(varl)) # remove duplicates
        if (type(bins) is int):
            bins = [bins]*len(bin_varl)
        if (range is None):
            range = [None]*len(bin_varl)
        if (type(weights) is str):
            weights = data(self,weights)[where].ravel()
        if (type(scales) is str):
            scales = [scales]*len(bin_varl)
        for i,v in enumerate(bin_varl):
            bins[i] = self._set_bins(v,bins[i],range[i],scales[i],where)
        bin_arr = [data(self,v)[where].ravel() for v in bin_varl]
        norm = xp.histogramdd(bin_arr,bins=bins,weights=weights,**kwargs)
        profs = {'edges':{v:_ for v,_ in zip(bin_varl,norm[1])},'norm':norm[0]}
        profs['centers'] = {v:(edge[:-1]+edge[1:])/2 for v,edge in profs['edges'].items()}
        for var in bin_varl:
            profs[var] = profs['centers'][var]
        for var in varl:
            if (weights is None):
                data_weights = data(self,var)[where].ravel()
            else:
                data_weights = data(self,var)[where].ravel()*weights
            profs[var] = xp.histogramdd(bin_arr,bins=bins,weights=data_weights,**kwargs)[0]
        profs = asnumpy(profs)
        if (global_vars.mpi_enabled):
            for k in varl+['norm',]:
                profs[k] = mpi.sum(np.ascontiguousarray(profs[k]))
        for k in varl:
            profs[k] = profs[k]/profs['norm']
        return profs

    ### get data in a dictionary ###
    def histogram(self,*args,**kwargs):
        hists = self._histograms(*args,**kwargs)
        return hists
    def histogram2d(self,*args,**kwargs):
        return self._histograms(*args,**kwargs)
    
    def get_sum(self,varl,*args,**kwargs):
        varl = [varl] if (type(varl) is str) else varl
        return {var : self.sum(var,*args,**kwargs) for var in varl}
    def get_avg(self,varl,*args,**kwargs):
        varl = [varl] if (type(varl) is str) else varl
        return {var : self.average(var,*args,**kwargs) for var in varl}
    def set_sum(self,*args,**kwargs):
        self.sums.update(self.get_sum(*args,**kwargs))
    def set_avg(self,*args,**kwargs):
        self.avgs.update(self.get_avg(*args,**kwargs))

    def get_hist(self,varl,bins=128,scales='log',weights='vol',**kwargs):
        return self.histogram(varl,bins=bins,scales=scales,weights=weights,**kwargs)
    def get_hist2d(self,varl,bins=128,scales='log',weights='vol',**kwargs):
        return self.histogram2d(varl,bins=bins,scales=scales,weights=weights,**kwargs)
    def get_profile(self,bin_var,varl,bins=256,weights='vol',**kwargs):
        return self._profiles(bin_var,varl,bins=bins,weights=weights,**kwargs)
    def get_profile2d(self,bin_varl,varl,bins=256,weights='vol',**kwargs):
        return self._profiles(bin_varl,varl,bins=bins,weights=weights,**kwargs)

    def set_hist(self,varl,key=None,bins=128,scales='log',weights='vol',**kwargs):
        key = weights if key is None else key
        if key not in self.hists.keys():
            self.hists[key] = {}
        self.hists[key].update(self.get_hist(varl,bins=bins,scales=scales,weights=weights,**kwargs))
    def set_hist2d(self,varl,key=None,bins=128,scales='log',weights='vol',**kwargs):
        key = weights if key is None else key
        if key not in self.hists.keys():
            self.hists[key] = {}
        self.hists[key].update(self.get_hist2d(varl,bins=bins,scales=scales,weights=weights,**kwargs))

    def set_profile(self,bin_var,varl,key=None,bins=256,weights='vol',**kwargs):
        key = bin_var if key is None else key
        if key not in self.profs.keys():
            self.profs[key] = {}
        self.profs[key].update(self.get_profile(bin_var,varl,bins=bins,weights=weights,**kwargs))
    def set_profile2d(self,bin_varl,varl,key=None,bins=256,weights='vol',**kwargs):
        key = ','.join(bin_varl) if key is None else key
        if key not in self.profs.keys():
            self.profs[key] = {}
        self.profs[key].update(self.get_profile2d(bin_varl,varl,bins=bins,weights=weights,**kwargs))

    # TODO(@mhguo): remove later when the new version is stable
    def set_slice_by_profile(self,bin_varl,varl,key=None,**kwargs):
        key = 'z' if key is None else key
        if key not in self.slices.keys():
            self.slices[key] = {}
        self.slices[key].update(self.get_profile2d(bin_varl,varl,**kwargs))

    #def get_radial(self,varl=['dens','temp','velr','mflxr'],bins=256,scales='log',weights='vol',**kwargs):
    #    return self._profiles(['r'],varl,bins=bins,scales=scales,weights=weights,**kwargs)
    #def set_radial(self,varl=['dens','temp','velr','mflxr'],bins=256,scales='log',weights='vol',**kwargs):
    #    self.profs.update(self.get_radial(varl,bins=bins,scales=scales,weights=weights,**kwargs))

    def xyz(self,zoom=0,level=None,axis=None):
        level = zoom if (level is None) else level
        if (axis=='x'):
            xyz = [self.x1min/2**level/self.Nx1,self.x1max/2**level/self.Nx1,
                    self.x2min/2**zoom,self.x2max/2**zoom,
                    self.x3min/2**zoom,self.x3max/2**zoom]
        elif (axis=='y'):
            xyz = [self.x1min/2**zoom,self.x1max/2**zoom,
                    self.x2min/2**level/self.Nx2,self.x2max/2**level/self.Nx2,
                    self.x3min/2**zoom,self.x3max/2**zoom]
        elif (axis=='z'):
            xyz = [self.x1min/2**zoom,self.x1max/2**zoom,
                    self.x2min/2**zoom,self.x2max/2**zoom,
                    self.x3min/2**level/self.Nx3,self.x3max/2**level/self.Nx3]
        else:
            xyz = [self.x1min/2**zoom,self.x1max/2**zoom,
                    self.x2min/2**zoom,self.x2max/2**zoom,
                    self.x3min/2**zoom,self.x3max/2**zoom]
        return xyz

    def get_slice_faces(self,zoom=0,level=0,xyz=[],axis='z'):
        xyz = self.xyz(zoom=zoom,level=level,axis=axis) if (not xyz) else xyz
        x,y,z=self._cell_faces(level=level,xyz=xyz)
        if (axis=='z'): return asnumpy({'x':x,'y':y})
        elif (axis=='y'): return asnumpy({'x':x,'z':z})
        else: return asnumpy({'y':y,'z':z})
    
    def get_slice_centers(self,zoom=0,level=0,xyz=[],axis='z'):
        dic = self.get_slice_faces(zoom=zoom,level=level,xyz=xyz,axis=axis)
        return asnumpy({v:(edge[:-1]+edge[1:])/2 for v,edge in dic.items()})

    def get_slice_coord(self,zoom=0,level=0,xyz=[],axis='z'):
        xyz = self.xyz(zoom=zoom,level=level,axis=axis) if (not xyz) else xyz
        x,y,z=self._xyz_uniform(level=level,xyz=xyz)
        axis = self._axis_index(axis)
        return asnumpy({'x':xp.average(x,axis=axis),'y':xp.average(y,axis=axis),'z':xp.average(z,axis=axis)})

    # TODO(@mhguo): we should have the ability to get slice at any position with any direction
    def slice(self,var='dens',zoom=0,level=0,xyz=[],axis='z'):
        xyz = self.xyz(zoom=zoom,level=level,axis=axis) if (not xyz) else xyz
        axis = self._axis_index(axis)
        return asnumpy({var:xp.average(self.data(var,dtype='uniform',level=level,xyz=xyz),axis=axis)})
    
    def get_slice(self,varl,**kwargs):
        if (type(varl) is str):
            varl = [varl]
        slices = {}
        slices['edges'] = self.get_slice_faces(**kwargs)
        slices['centers'] = self.get_slice_centers(**kwargs)
        #slices.update(self.get_slice_coord(**kwargs))
        for var in varl:
            slices.update(self.slice(var,**kwargs))
        return slices
    
    def set_slice(self,varl,key='z',**kwargs):
        if key not in self.slices.keys():
            self.slices[key] = {}
        self.slices[key].update(self.get_slice(varl,**kwargs))

    # TODO(@mhguo): not compatible with MPI for now I think
    def interpolate(self,varl='dens',points=[[0.,0.,0.]]):
        from scipy.interpolate import RegularGridInterpolator
        if (type(varl) is str):
            varl = [varl]
        varl = list(dict.fromkeys(varl)) # remove duplicates
        raws = {var : asnumpy(self.data(var)) for var in varl}
        points = np.array(points)
        shape = points.shape
        points = points.reshape(-1,3)
        results = {var : np.full(points.shape[0], np.nan) for var in varl}
        # result = np.full(points.shape[0], np.nan)
        # for nraw,nmb in enumerate(self.mb_list):
        for nraw,nmb in enumerate(self.mb_list):
            print(f"Interpolating block {nmb}...")
            x0, x1, y0, y1, z0, z1 = self.mb_geometry[nmb]
            nx1, nx2, nx3 = self.nx1, self.nx2, self.nx3
            x = np.linspace(x0, x1, nx1+1)
            y = np.linspace(y0, y1, nx2+1)
            z = np.linspace(z0, z1, nx3+1)
            # Get cell centers
            x = 0.5 * (x[:-1] + x[1:])
            y = 0.5 * (y[:-1] + y[1:])
            z = 0.5 * (z[:-1] + z[1:])
            # Interpolate (trilinear)
            # block_data = raw[self.mb_list.tolist().index(nmb)]
            interps = {var : RegularGridInterpolator((z, y, x), raws[var][nraw], bounds_error=False, fill_value=None) for var in varl}
            # interp = RegularGridInterpolator((z, y, x), block_data)
            # Get block bounds
            # find location of each point that is inside the block
            locs = (points[:,0]>=x0) & (points[:,0]<=x1) &  (points[:,1]>=y0) & (points[:,1]<=y1) & (points[:,2]>=z0) & (points[:,2]<=z1)
            # result[locs] = interp(points[locs,::-1])  # (z, y, x) order
            for var in varl:
                results[var][locs] = interps[var](points[locs,::-1])  # (z, y, x) order
            # for i, pt in enumerate(points):
            #     if locs[i]:
            #         # Found the finest block containing the point
            #         # Get block data and local grid
            #         result[i] = interp(pt[::-1])  # (z, y, x) order
            #         break  # Stop at the finest block
        # reshape back to original shape
        for var in varl:
            results[var] = results[var].reshape(shape[:-1])
        return results
    
    #def slice(self,var='dens',normal='z',north='y',center=[0.,0.,0.],width=1,height=1,zoom=0,level=0):
    #    return

    def _figax(self,fig=None,ax=None,dpi=135):
        if (ax is not None): fig = ax.get_figure()
        else: 
            fig = plt.figure(dpi=dpi) if fig is None else fig
            ax = fig.axes[0] if len(fig.axes)>0 else plt.axes()
        return fig,ax

    # plot is only for plot, accept the data array
    def plot_image(self,x,y,img,title='',label='',xlabel='X',ylabel='Y',xscale='linear',yscale='linear',\
                   cmap='viridis',norm='log',save=False,figfolder=None,figlabel='',figname='',\
                   dpi=135,fig=None,ax=None,colorbar=True,returnall=False,aspect='auto',\
                   xticks=None,yticks=None,xticklabels=None,yticklabels=None, **kwargs):
        fig,ax = self._figax(fig,ax,dpi)
        img = asnumpy(img[:,:])
        im=ax.pcolormesh(x,y,img,norm=norm,cmap=cmap,**kwargs)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_aspect(aspect)
        if (xticks is not None): ax.set_xticks(xticks)
        if (yticks is not None): ax.set_yticks(yticks)
        if (xticklabels is not None): ax.set_xticklabels(xticklabels)
        if (yticklabels is not None): ax.set_yticklabels(yticklabels)
        if (title != None): ax.set_title(f"Time = {self.time}" if not title else title)
        if (colorbar):
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.02)
            fig.colorbar(im,ax=ax,cax=cax, orientation='vertical',label=label)
        if (save):
            if (not os.path.isdir(figfolder)):
                os.mkdir(figfolder)
            fig.savefig(f"{figfolder}/fig_{figlabel}_{self.num:04d}.png"\
                        if not figname else figname, bbox_inches='tight')
        if (returnall):
            return fig,ax,im
        return im

    def plot_stream(self,dpi=135,fig=None,ax=None,x=None,y=None,u=None,v=None,
                    xyunit=1.0,color='k',linewidth=None,arrowsize=None):
        ax = plt.axes() if ax is None else ax
        fig = ax.get_figure() if fig is None else fig
        strm = ax.streamplot(x*xyunit, y*xyunit, u, v, color=color,linewidth=linewidth,arrowsize=arrowsize)
        if (returnall):
            return fig,ax,strm
        return strm

    def plot_phase(self,var='dens,temp',key='vol',bins=128,range=None,weights='vol',where=None,title='',label='',xlabel=None,ylabel=None,xscale='log',yscale='log',\
                   unit=1.0,cmap='viridis',norm='log',extent=None,density=False,save=False,savepath='',figdir='../figure/Simu_',\
                   figpath='',x=None,y=None,xshift=0.0,xunit=1.0,yshift=0.0,yunit=1.0,fig=None,ax=None,dpi=135,returnall=False,**kwargs):
        fig,ax = self._figax(fig,ax,dpi)
        try:
            dat = self.hists[key][var]
        except:
            varl = var.split(',') if ',' in var else var.split('-')
            dat = self.get_hist2d([varl],bins=bins,range=range,scales=[[xscale,yscale]],weights=weights,where=where)[var]
        x,y = dat['edges'].values()
        im_arr = asnumpy(dat['dat'])
        extent = [x.min(),x.max(),y.min(),y.max()] if extent is None else extent
        if (density):
            xlength = (extent[1]-extent[0] if xscale=='linear' else np.log10(extent[1]/extent[0]))/(x.shape[0]-1)
            ylength = (extent[3]-extent[2] if yscale=='linear' else np.log10(extent[3]/extent[2]))/(y.shape[0]-1)
            unit /= xlength*ylength
        #im = ax.imshow(dat['dat'].swapaxes(0,1)[::-1,:]*unit,extent=extent,norm=norm,cmap=cmap,aspect=aspect,**kwargs)
        im_arr = im_arr.T*unit
        x =  x*xunit+xshift
        y =  y*yunit+yshift
        if (xlabel is None): xlabel = var.split(',')[0]
        if (ylabel is None): ylabel = var.split(',')[1]
        im=self.plot_image(x,y,im_arr,title=title,label=label,xlabel=xlabel,ylabel=ylabel,xscale=xscale,yscale=yscale,\
                    cmap=cmap,norm=norm,save=save,figfolder=figdir,figlabel=var,figname=savepath,fig=fig,ax=ax,**kwargs)
        if (title != None): ax.set_title(f"Time = {self.time}" if not title else title)
        if (save):
            figpath=figdir+Path(self.path).parts[-1]+'/'+self.label+"/" if not figpath else figpath
            if not os.path.isdir(figpath):
                os.mkdir(figpath)
            fig.savefig(f"{figpath}fig_{var}_{self.num:04d}.png"\
                        if not savepath else savepath, bbox_inches='tight')
        if (returnall):
            return fig,ax,im
        return ax
    
    # TODO(@mhguo): maybe remove later when the new version is stable
    def plot_slice_by_prof(self,var='dens',key=None,data=None,zoom=0,level=0,xyz=[],unit=1.0,bins=None,\
                   title='',label='',xlabel='X',ylabel='Y',cmap='viridis',\
                   norm='log',save=False,figdir='../figure/Simu_',figpath=None,\
                   savepath='',savelabel='',figlabel='',dpi=135,vec=None,stream=None,circle=True,\
                   fig=None,ax=None,xyunit=1.0,colorbar=True,returnall=False,stream_color='k',stream_linewidth=1.0,\
                   stream_arrowsize=1.0,vecx='velx',vecy='vely',vel_method='ave',aspect='equal',**kwargs):
        fig,ax = self._figax(fig,ax,dpi)
        bins=int(xp.min(xp.asarray([self.Nx1,self.Nx2,self.Nx3]))) if not bins else bins
        if var in self.slices[key].keys():
            slc = self.slices[key][var]
        else:
            slc = self.get_slice(['x,y'],var,weights='vol',bins=128,where=xp.abs(self.data('z'))<self.x3max/2**zoom/self.nx3,
                     range=[[self.x1min/2**zoom,self.x1max/2**zoom],[self.x2min/2**zoom,self.x2max/2**zoom]])
        x,y = self.slices[key]['edges'].values()
        xc,yc = self.slices[key]['centers'].values()
        im_arr = asnumpy(slc.T)*unit
        if (stream):
            u,v = self.slices[key][vecx].T,self.slices[key][vecy].T
            ax.streamplot(xc*xyunit, yc*xyunit, u, v,color=stream_color,linewidth=stream_linewidth,arrowsize=stream_arrowsize)
        im=self.plot_image(x*xyunit,y*xyunit,im_arr,title=title,label=label,xlabel=xlabel,ylabel=ylabel,aspect=aspect,\
                     cmap=cmap,norm=norm,save=save,figfolder=figdir,figlabel=var,figname=savepath,fig=fig,ax=ax,**kwargs)
        if (vec):
            u,v = self.slices[key][vecx].T,self.slices[key][vecy].T
            ax.quiver(xc*xyunit, yc*xyunit, u, v)
        if(circle and self.header('problem','r_in')):
            ax.add_patch(plt.Circle((0,0),float(self.header('problem','r_in')),ec='k',fc='#00000000'))
        if (title != None): ax.set_title(f"Time = {self.time}" if not title else title)
        if (colorbar):
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.02)
            fig.colorbar(im,ax=ax,cax=cax, orientation='vertical',label=label)
        if (returnall):
            return fig,ax,im
        return ax

    # get snapshot data
    def get_slice_for_plot(self,var='dens',key=None,vec=None,stream=None,vecx='velx',vecy='vely',zoom=0,level=0,xyz=[],unit=1.0,xyunit=1.0,axis='z'):
        if key is None:
            slice=self.get_slice(varl=[var,vecx,vecy],zoom=zoom,level=level,xyz=xyz,axis=axis)
        else:
            slice=self.slices[key]
        x,y = slice['centers'].values()
        x,y = x*xyunit,y*xyunit
        c = slice[var]*unit
        u,v = None,None
        if (vec is not None or stream is not None):
            u,v = slice[vecx],slice[vecy]
        return x,y,c,u,v

    def plot_slice(self,var='dens',key=None,vec=None,stream=None,vecx='velx',vecy='vely',
                   zoom=0,level=0,xyz=[],unit=1.0,xyunit=1.0,axis='z',
                   fig=None,ax=None,dpi=135,norm='log',cmap='viridis',aspect='equal',
                   xlabel=None,ylabel=None,title='',label='',colorbar=True,
                   quiver_para=dict(),stream_para={},
                   returnall=False,**kwargs):
        fig,ax = self._figax(fig,ax,dpi)
        x,y,c,u,v = self.get_slice_for_plot(var=var,key=key,vec=vec,stream=stream,vecx=vecx,vecy=vecy,
                                            zoom=zoom,level=level,xyz=xyz,unit=unit,xyunit=xyunit,axis=axis)
        quiver,strm = None,None
        if (vec is not None):
            quiver = ax.quiver(x,y,u,v,**quiver_para)
        if (stream is not None):
            strm_para=dict(color='k',linewidth=1.0,arrowsize=1.0)
            strm_para.update(stream_para)
            strm = ax.streamplot(x,y,u,v,**strm_para)
        #im=ax.pcolormesh(x,y,c,norm=norm,cmap=cmap,shading=shading,**kwargs)
        im = self.plot_image(x,y,c,fig=fig,ax=ax,dpi=dpi,norm=norm,cmap=cmap,aspect=aspect,
                                 xlabel=xlabel,ylabel=ylabel,title=title,label=label,
                                 colorbar=colorbar,**kwargs)
        if (colorbar):
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.02)
            fig.colorbar(im,ax=ax,cax=cax, orientation='vertical',label=label)
        if (returnall):
            return fig,ax,im,quiver,strm
        return ax

    def plot_profile(self,var='r,dens',key=None,unit=1.0,xunit=1.0,bins=256,weights='vol',range=None,where=None,fig=None,ax=None,dpi=135,xscale='log',yscale='log',xlabel=None,ylabel=None,returnall=False,**kwargs):
        fig,ax = self._figax(fig,ax,dpi)
        binv, v = var.split(',')
        xlabel = binv if (xlabel is None) else xlabel
        ylabel = v if (ylabel is None) else ylabel
        if (key is not None):
            prof = self.profs[key]
        elif (binv in self.profs.keys()):
            prof = self.profs[binv]
        else:
            prof = self.get_profile(bin_var=binv,varl=[v],bins=bins,weights=weights,scales=[xscale,yscale],range=range,where=where)
        line=ax.plot(prof[binv]*xunit,prof[v]*unit,**kwargs)
        if (returnall):
            return fig,ax,line
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        return ax

    def plot(self,*args,**kwargs):
        return self.plot_profile(*args,**kwargs)

    def plot_hist2d(self,*args,**kwargs):
        return self.plot_phase(*args,**kwargs)

class AthenaDataSet:
    def __init__(self,version='1.0'):
        self.version=version
        self.ads={}
        #self._config_func()
        return

    """
    def _config_func(self):
        #ad_methods = [method_name for method_name in dir(AthenaData) if callable(getattr(AthenaData, method_name)) and method_name[0]!='-']
        ad_methods = [method_name for method_name in dir(AthenaData) if method_name[0]!='-']
        for method_name in ad_methods:
            self.__dict__[method_name] = lambda *args, **kwargs: [getattr(self.alist[i], method_name)(*args, **kwargs) for i in self.ilist]
            #self.__dict__[method_name] = lambda *args, **kwargs: [self.alist[i].__dict__[method_name](*args, **kwargs) for i in self.ilist]
        return
    """

    def load(self,ns,path=None,dtype=None,verbose=False,**kwargs):
        for n in ns:
            if(verbose): print("load:",n)
            if n not in self.ads.keys():
                self.ads[n]=AthenaData(num=n,version=self.version)
            self.ads[n].load(path+f".{n:05d}."+dtype,**kwargs)
        return

    @property
    def ns(self):
        return sorted(list(set(self.ads.keys())))

    @property
    def ad(self):
        return self.ads[self.ns[0]]

    def __call__(self, n=None):
        if (n is None):
            n = self.ns[0]
        return self.ads[n]

    def __getitem__(self, n=None):
        if (n is None):
            n = self.ns[0]
        return self.ads[n]
    
    def keys(self):
        return self.ns
    
    def values(self):
        return [self.ads[n] for n in self.ns]
    
    def items(self):
        return [(n,self.ads[n]) for n in self.ns]

    def pop(self,n):
        return self.ads.pop(n)
    
    def popitem(self):
        return self.ads.popitem()
    
    def clear(self):
        return self.ads.clear()
