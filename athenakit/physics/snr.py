import numpy as np
from scipy.integrate import ode as sp_ode
from scipy.integrate import odeint as sp_odeint
from .. import units
##########################################################################################
## SNR
##########################################################################################

# analytical function

class SedovTaylor():
    def __init__(self,E,rho,gamma=5/3) -> None:
        self.E=E
        self.rho=rho
        self.epsilon=1.15167
    def r_s(self,t):
        return self.epsilon*self.E**(1/5)*self.rho**(-1/5)*t**(2/5)
    def v_s(self,t):
        return 2/5*self.epsilon*self.E**(1/5)*self.rho**(-1/5)*t**(-3/5)

class SNR_evo:
    # TODO(@mhguo): add units!!!
    # Note that everything is in code units
    def __init__(self,n=1,M=3,E=1,mu=0.618,gamma=5./3.,config=True):
        #E [erg]
        #n cm^-3
        #M Msun
        self.n0=1
        self.M0=units.msun_cgs
        self.E0=1e51
        self.mu=mu
        self.gamma=gamma
        self.unit=units.Units(lunit=units.pc_cgs,munit=mu*units.atomic_mass_unit_cgs*units.pc_cgs**3,tunit=units.myr_cgs,mu=mu)
        self.n=n*self.n0/self.unit.number_density_cgs
        self.M=M*self.M0/self.unit.mass_cgs
        self.E=E*self.E0/self.unit.energy_cgs
        self.Ei=0.72*self.E
        self.Ek=0.28*self.E
        self.n_shock=self.n*(self.gamma+1)/(self.gamma-1)
        self.v_free=np.sqrt(2*self.E/self.M)
        self.r_free=(self.M/(4/3*np.pi*self.n))**(1/3)
        self.t_free=self.r_free/self.v_free

        #self.t_free=4.64e-4*E**(-1/2)*n**(-1/3)
        #self.r_free=2.75*n**(-1/3)
        
        self.epsilon=1.15167
        self.t_sf=0.030*E**0.22*n**-0.55
        self.r_sf=self._r_st(self.t_sf)
        self.v_sf=self._v_st(self.t_sf)
        self.mom_sf=2.69*self.n*self.v_sf*self.r_sf**3
        #self.r_sf=22.6*E**0.29*n**-0.42
        #self.mom_sf=2.17e5*E**0.93*n**-0.13
        self.evo={}
        if(config): self.config()
        return
    
    def _r_st(self,t):
        return self.epsilon*self.E**(1/5)*self.n**(-1/5)*t**(2/5)
    def _v_st(self,t):
        return 2/5*self.epsilon*self.E**(1/5)*self.n**(-1/5)*t**(-3/5)
    def _temp_st(self,t):
        return 3.0/16.0*self._v_st(t)**2
    @np.vectorize
    def _r(self,t):
        if(t<=self.t_free):
            return np.sqrt(2*self.E/self.M)*t
        elif(t<=self.t_sf):
            #return 5*self.E**(1/5)*self.n**(-1/5)*(t/1e-3)**(2/5)
            return self._r_st(t)
        else:
            return self.r_sf*(t/self.t_sf)**(2/7)
        #else:
        #    return 30*(t/0.1)**(2/7)
    @np.vectorize
    def _v(self,t):
        if(t<=self.t_free):
            return np.sqrt(2*self.E/self.M)
        elif(t<=self.t_sf):
            #return 5*self.E**(1/5)*self.n**(-1/5)*(t/1e-3)**(2/5)
            return self._v_st(t)
        else:
            return 2/7*self.r_sf/self.t_sf*(t/self.t_sf)**(-5/7)
    @np.vectorize
    def _momr(self,t):
        if(t<=self.t_free):
            return self.M*np.sqrt(2*self.E/self.M)
        elif(t<=self.t_sf):
            #return 5*self.E**(1/5)*self.n**(-1/5)*(t/1e-3)**(2/5)
            return 2.69*self.n*self._v_st(t)*self._r_st(t)**3
        else:
            return self.mom_sf*(1+4.6*((t/self.t_sf)**(1/7)-1.0))
    # This is averaged pressure, not shocked pressure
    @np.vectorize
    def _pres(self,t):
        if(t<=self.t_free):
            #return self.M*np.sqrt(2*self.E/self.M)
            return self.Ei/2.0/np.pi/self._r_st(t)**3
        elif(t<=self.t_sf):
            #return 5*self.E**(1/5)*self.n**(-1/5)*(t/1e-3)**(2/5)
            return self.Ei/2.0/np.pi/self._r_st(t)**3
        else:
            return self.Ei/2.0/np.pi/self._r_st(self.t_sf)**3*(t/self.t_sf)**(-10/7)
    def r(self,t):
        return self._r(self,t)
    def v(self,t):
        return self._v(self,t)
    def momr(self,t):
        return self._momr(self,t)
    def pres(self,t):
        return self._pres(self,t)

    def config(self,t=np.logspace(-4,0,300)):
        self.evo['t']=t
        self.evo['r']=self.r(t)
        self.evo['v']=self.v(t)
        self.evo['momr']=self.momr(t)
        self.evo['pres']=self.pres(t)
        # Make sure this is correct!
        self.evo['temp']=self.evo['pres']/self.n_shock
        self.evo['m']=4/3*np.pi*self.n*self.evo['r']**3+self.M
        self.evo['eta']=self.evo['v']/(self.evo['r']/self.evo['t'])
        self.evo['etot']=np.full(self.evo['t'].shape,self.E)
        self.evo['ei']=0.72*self.evo['etot']
        self.evo['ek']=0.28*self.evo['etot']
        #self.evo['momr']=2.69/(4/3*np.pi)*self.evo['m']*self.evo['v']

class SNR_SedovTaylor():
    def __init__(self,gamma=5/3,E0=1e51,n0=1.0,mu=0.618) -> None:
        self.gamma=gamma
        self.ny=3
        self.E0=E0
        self.n0=n0
        self.mu=mu
        self.rho0=self.n0*self.mu*units.mp_cgs
        self._config()
    def _config(self):
        self.a10=2*(self.gamma-1)/(self.gamma+1)**2
    def func(self,Y,x):
        #Y=[f,g,h]
        #a*Yprime=b
        gamma=self.gamma
        a10=self.a10
        f,g,h=Y[0],Y[1],Y[2]
        a=np.array([[0.0,     h-x,            g      ],
                    [a10,     0.0,            g*(h-x)],
                    [g*(h-x), -gamma*f*(h-x), 0.0    ]])
        b=np.array([-2*g*h/x,
                    h*(1.5*g),
                    3*f*g])
        Yprime = np.linalg.solve(a,b)
        return Yprime
    def solve(self,xs):
        y0=np.array([1,1,2.0/(self.gamma+1.0)])
        self.xs=np.asarray(xs)
        xs=self.xs
        self.ys=sp_odeint(self.func,y0,xs)
        return self.ys
    def post_solve(self,ts):
        self.set_K()
        self.ts=ts
        self.rs=self._rs(self.ts)
        self.Ps=self._Ps(self.rs)
        self.vs=self._vs(self.rs)
    def set_K(self):
        xs=self.xs[::-1]
        fs=self.ys[::-1,0]
        gs=self.ys[::-1,1]
        hs=self.ys[::-1,2]
        self.K=(self.gamma-1.0)/(np.trapz(xs**2*(2*fs+0.5*(self.gamma+1)**2*gs*hs**2),xs))
    def _rs(self,t):
        return (25*(self.gamma+1)*self.K*self.E0/16/np.pi/self.rho0)**0.2*t**0.4
    def _Ps(self,rs):
        return self.K*self.E0/(2.0*np.pi*rs**3)
    def _vs(self,rs):
        return np.sqrt((self.gamma+1)*self.K*self.E0/(4.0*np.pi*self.rho0*rs**3))
    def __call__(self,xs,ts):
        self.solve(xs)
        self.post_solve(ts)

class SNR_WhiteLong():
    def __init__(self,tau1,mucl,gamma=5/3,E0=1e51,n0=1.0,mu=0.618) -> None:
        self.mucl=mucl
        self.tau1=tau1
        self.gamma=gamma
        self.ny=4
        self.E0=E0
        self.n0=n0
        self.mu=mu
        self.rho0=self.n0*self.mu*units.mp_cgs
        self._config()
    def _config(self):
        self.a10=2*(self.gamma-1)/(self.gamma+1)**2
        #self.A=j_s*r_s/v_s/rho_s
        self.A=2.5*self.mucl*self.tau1*(self.gamma-1.0)/(self.gamma+1.0)
        self.K=1.0
    def func(self,x,Y):
        #Y=[f,g,h]
        #a*Yprime=b
        gamma=self.gamma
        tau1=self.tau1
        A=self.A
        a10=self.a10
        f,g,h,l=Y[0],Y[1],Y[2],Y[3]
        # TODO(@mhguo) remove below
        # do integral
        '''
        where=np.where(self.xs>=x)
        self.fs=self.ys[:,0]
        if (where[0].shape[0]<2):
            l=np.exp(-2.5*tau1*(sp_quad(lambda _ : self.fs[0]**(5/6)/_,x,1.0)[0]))
        else:
            _x2f=sp_interp1d(self.xs[where],self.fs[where],fill_value='extrapolate')
            #l=np.exp(-2.5*tau1*(sp_quad(lambda _ : _x2f(_)**(5/6)/_,x,1.0)[0]))
            tmp_xs=np.append(self.xs[where],x)[::-1]
            tmp_fs=np.append(self.fs[where],_x2f(x))[::-1]
            tmp_integral=np.trapz(tmp_fs**(5/6)/tmp_xs,tmp_xs)
            l=np.exp(-2.5*tau1*tmp_integral)
        '''
        #tmp
        #l=0.0
        k=l*f**(5/6)
        a=np.array([[0.0,     h-x,            g      , 0.0],
                    [a10,     0.0,            g*(h-x), 0.0],
                    [g*(h-x), -gamma*f*(h-x), 0.0    , 0.0],
                    [0.0,     0.0,            0.0    , 1.0]])
        b=np.array([A*k-2*g*h/x,
                    h*(1.5*g-A*k),
                    3*f*g+A*k*((gamma+1)**2/4.0*g*h**2-gamma*f),
                    2.5*tau1*l*f**(5/6)/x,])
        
        Yprime = np.linalg.solve(a,b)
        '''
        print("x:", x)
        print("Y:", Y)
        print("a:", a)
        print("b:", b)
        print("Yprime:", Yprime)
        '''
        return Yprime
    def solve(self,xs):
        y0=np.array([1.0,1.0,2.0/(self.gamma+1.0),1.0])
        self.xs=np.asarray(xs)
        xs=self.xs
        self.ys=np.zeros((self.xs.shape[0],self.ny))
        self.ys[0]=y0
        solver=sp_ode(self.func)
        solver.set_initial_value(y0,xs[0])
        for i in range(len(xs)-1):
            #print(i)
            self.ys[i+1]=solver.integrate(xs[i+1])
        return self.ys
    def post_solve(self,ts):
        self.set_K()
        self.ts=ts
        self.rs=self._rs(self.ts)
        self.Ps=self._Ps(self.rs)
        self.vs=self._vs(self.rs)
    def set_K(self):
        xs=self.xs[::-1]
        fs=self.ys[::-1,0]
        gs=self.ys[::-1,1]
        hs=self.ys[::-1,2]
        self.K=(self.gamma-1.0)/(np.trapz(xs**2*(2*fs+0.5*(self.gamma+1)**2*gs*hs**2),xs))
    def _rs(self,t):
        return (25*(self.gamma+1)*self.K*self.E0/16/np.pi/self.rho0)**0.2*t**0.4
    def _Ps(self,rs):
        return self.K*self.E0/(2.0*np.pi*rs**3)
    def _vs(self,rs):
        return np.sqrt((self.gamma+1)*self.K*self.E0/(4.0*np.pi*self.rho0*rs**3))
    def __call__(self,xs,ts):
        self.solve(xs)
        self.post_solve(ts)

# inculding rhodot, pdot, edot
class SNR_mpe():
    def __init__(self,tau1,mucl,B=1.0,C=1.0,gamma=5/3,E0=1e51,n0=1.0,mu=0.618,psi_x=0.2) -> None:
        self.mucl=mucl
        self.tau1=tau1
        self.gamma=gamma
        self.ny=4#6
        self.E0=E0
        self.n0=n0
        self.mu=mu
        self.rho0=self.n0*self.mu*units.mp_cgs
        self.B=B
        self.C=C
        self.psi_x=psi_x
        self._config()
    def _config(self):
        self.a10=2*(self.gamma-1)/(self.gamma+1)**2
        #self.A=j_s*r_s/v_s/rho_s
        self.A=2.5*self.mucl*self.tau1*(self.gamma-1.0)/(self.gamma+1.0)
        self.K=1.0
    def func(self,x,Y):
        #Y=[f,g,h]
        #a*Yprime=b
        gamma=self.gamma
        tau1=self.tau1
        A,B,C=self.A,self.B,self.C,
        a10=self.a10
        f,g,h,l=Y[0],Y[1],Y[2],Y[3]
        k=l*f**(5/6)
        # TODO(@mhguo): check this!
        phi=g*h
        psi=1.0/(1.0+(self.psi_x/x)**4)
        ###
        '''
        a=np.array([[0.0,     h-x,            g      , 0.0, 0.0, 0.0],
                    [a10,     0.0,            g*(h-x), 0.0, 0.0, 0.0],
                    [g*(h-x), -gamma*f*(h-x), 0.0    , 0.0, 0.0, 0.0],
                    [0.0,     0.0,            0.0    , 1.0, 0.0, 0.0],
                    [0.0,     0.0,            0.0    , 0.0, 1.0, 0.0],
                    [0.0,     0.0,            0.0    , 0.0, 0.0, 1.0],
                    ])
        b=np.array([A*k-2*g*h/x,
                    1.5*h*g+B*phi-A*k*h,
                    3*f*g+0.5*(gamma+1.0)**2*g*(C*psi-B*phi*h)+A*k*((gamma+1.0)**2/4.0*g*h**2-gamma*f),
                    2.5*tau1*l*f**(5/6)/x,
                    2.5*tau1*phi/x,#0.0,#
                    2.5*tau1*psi/x,#0.0,#
                    ])
        '''
        '''
        a=np.array([[0.0,     h-x,            g      , 0.0,],
                    [a10,     0.0,            g*(h-x), 0.0,],
                    [g*(h-x), -gamma*f*(h-x), 0.0    , 0.0,],
                    [0.0,     0.0,            0.0    , 1.0,],
                    ])
        b=np.array([A*k-2*g*h/x,
                    1.5*h*g+B*phi-A*k*h,
                    3*f*g+0.5*(gamma+1.0)**2*g*(C*psi-B*phi*h)+A*k*((gamma+1.0)**2/4.0*g*h**2-gamma*f),
                    2.5*tau1*l*f**(5/6)/x,
                    ])
        Yprime = np.linalg.solve(a,b)
        '''
        i = h - x
        # TODO(@mhguo): tmp psi!
        psi=1.0/(1.0+(self.psi_x/x)**4)
        psi=0.5+0.5*np.tanh(100.0*(x-self.psi_x))
        #print(psi)
        #psi = (C*i**2-3.0*f)/(0.5*(gamma+1.0)**2*C)
        #psi += (-0.5*(gamma+1.0)**2*C*psi)*np.exp(-(0.0/0.03)**2)/(0.5*(gamma+1.0)**2*C)
        psi += (-3.0*f-0.5*(gamma+1.0)**2*C*psi)*np.exp(-(i/0.01)**2)/(0.5*(gamma+1.0)**2*C)
        #print(psi)
        b=np.array([A*k-2*g*h/x,
                    1.5*h*g+B*phi-A*k*h,
                    3*f*g+0.5*(gamma+1.0)**2*g*(C*psi-B*phi*h)+A*k*((gamma+1.0)**2/4.0*g*h**2-gamma*f),
                    2.5*tau1*l*f**(5/6)/x,
                    ])
        Yprime = np.zeros(Y.shape)
        Yprime[0] = -b[0]*gamma*f*i     + b[1]*gamma*f - b[2]*i     
        #Yprime[1] = -b[0]*g*i           + b[1]*g       - b[2]*a10/i
        Yprime[1] = -b[0]*g*i           + b[1]*g       - b[2]*a10/i
        #Yprime[2] =  b[0]*a10*gamma*f/g - b[1]*i       + b[2]*a10/g
        Yprime[2] = -2*h/x*a10*gamma*f - b[1]*i  + ( 3*f+0.5*(gamma+1.0)**2*(C*psi-B*phi*h)+A*k*((gamma+1.0)**2/4.0*h**2) )*a10
        Yprime   *= 1.0/(a10*gamma*f-g*i**2)
        # last for l
        Yprime[3] =  b[3]
        '''
        print("x:", x)
        print("Y:", Y)
        print("a:", a)
        print("b:", b)
        print("Yprime:", Yprime)
        '''
        return Yprime
    def solve(self,xs):
        y0=np.array([1.0,1.0,2.0/(self.gamma+1.0),1.0])
        self.xs=np.asarray(xs)
        xs=self.xs
        self.ys=np.zeros((self.xs.shape[0],self.ny))
        self.ys[0]=y0
        solver=sp_ode(self.func)
        solver.set_initial_value(y0,xs[0])
        for i in range(len(xs)-1):
            #print(i)
            self.ys[i+1]=solver.integrate(xs[i+1])
        return self.ys
    def post_solve(self,ts):
        self.set_K()
        self.ts=ts
        self.rs=self._rs(self.ts)
        self.Ps=self._Ps(self.rs)
        self.vs=self._vs(self.rs)
    def set_K(self):
        xs=self.xs[::-1]
        fs=self.ys[::-1,0]
        gs=self.ys[::-1,1]
        hs=self.ys[::-1,2]
        self.K=(self.gamma-1.0)/(np.trapz(xs**2*(2*fs+0.5*(self.gamma+1)**2*gs*hs**2),xs))
    def _rs(self,t):
        return (25*(self.gamma+1)*self.K*self.E0/16/np.pi/self.rho0)**0.2*t**0.4
    def _Ps(self,rs):
        return self.K*self.E0/(2.0*np.pi*rs**3)
    def _vs(self,rs):
        return np.sqrt((self.gamma+1)*self.K*self.E0/(4.0*np.pi*self.rho0*rs**3))
    def call(self,xs,ts):
        self.solve(xs)
        self.post_solve(ts)
    def __call__(self,xs,ts):
        print("test")
        self.solve(xs)
        self.post_solve(ts)

# TODO(@mhguo): this is not complete!
class SNR_ABC():
    def __init__(self,A,B=1.0,C=1.0,gamma=5/3,E0=1e51,n0=1.0,mu=0.618,psi_x=0.2) -> None:
        self.a10=2*(self.gamma-1)/(self.gamma+1)**2
        #self.A=j_s*r_s/v_s/rho_s
        self.A=A
        self.B=B
        self.C=C
        self.K=1.0
        self.gamma=gamma
        self.ny=4#6
        self.E0=E0
        self.n0=n0
        self.mu=mu
        self.rho0=self.n0*self.mu*units.mp_cgs
        self.psi_x=psi_x
    def func(self,x,Y):
        #Y=[f,g,h]
        #a*Yprime=b
        gamma=self.gamma
        tau1=self.tau1
        A,B,C=self.A,self.B,self.C,
        a10=self.a10
        f,g,h,l=Y[0],Y[1],Y[2],Y[3]
        k=l*f**(5/6)
        # TODO(@mhguo): check this!
        phi=g*h
        psi=1.0/(1.0+(self.psi_x/x)**4)
        ###
        i = h - x
        # TODO(@mhguo): tmp psi!
        psi=1.0/(1.0+(self.psi_x/x)**4)
        #psi=0.5+0.5*np.tanh(100.0*(x-self.psi_x))
        #print(psi)
        #psi = (C*i**2-3.0*f)/(0.5*(gamma+1.0)**2*C)
        #psi += (-0.5*(gamma+1.0)**2*C*psi)*np.exp(-(0.0/0.03)**2)/(0.5*(gamma+1.0)**2*C)
        #psi += (-3.0*f-0.5*(gamma+1.0)**2*C*psi)*np.exp(-(i/0.01)**2)/(0.5*(gamma+1.0)**2*C)
        #print(psi)
        b=np.array([A*k-2*g*h/x,
                    1.5*h*g+B*phi-A*k*h,
                    3*f*g+0.5*(gamma+1.0)**2*g*(C*psi-B*phi*h)+A*k*((gamma+1.0)**2/4.0*g*h**2-gamma*f),
                    2.5*tau1*l*f**(5/6)/x,
                    ])
        Yprime = np.zeros(Y.shape)
        Yprime[0] = -b[0]*gamma*f*i     + b[1]*gamma*f - b[2]*i     
        #Yprime[1] = -b[0]*g*i           + b[1]*g       - b[2]*a10/i
        Yprime[1] = -b[0]*g*i           + b[1]*g       - b[2]*a10/i
        #Yprime[2] =  b[0]*a10*gamma*f/g - b[1]*i       + b[2]*a10/g
        Yprime[2] = -2*h/x*a10*gamma*f - b[1]*i  + ( 3*f+0.5*(gamma+1.0)**2*(C*psi-B*phi*h)+A*k*((gamma+1.0)**2/4.0*h**2) )*a10
        Yprime   *= 1.0/(a10*gamma*f-g*i**2)
        # last for l
        Yprime[3] =  b[3]
        '''
        print("x:", x)
        print("Y:", Y)
        print("a:", a)
        print("b:", b)
        print("Yprime:", Yprime)
        '''
        return Yprime
