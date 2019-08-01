"""Functions and Classes to return the transport coefficients
for the Navier Stokes equations based on a BGK kinetic closure
as derived in [1].

[1]Haack, J., Hauck, C., & Murillo, M. (2017). 
A Conservative, Entropic Multispecies BGK Model. 
Journal of Statistical Physics, 168(4), 826--856. 
https://doi.org/10.1007/s10955-017-1824-9
"""
import numpy as np
import scipy.linalg as sla
from sm_pproperties import get_calk, get_leff
from zbar import ZbarSolver



class BGK_Htransport(object):
    """A class for generating transport coefficients based off of a single 
    temperature. 
    
    This class generates the Navier-Stokes transport terms for different
    systems that are assumed to all have the same species with invariant
    masses/nuclear charges and variable thermodynamic properties. However, 
    they are all assumed to be described by a single temperature.
    
    Parameters
    ----------
    s : {'T', 'M'}
        String dictating whether temperature or momentum relaxation collision 
        operators are used.
    o : {3, 1, 2}
        Order of approximation to use for electron wavelength. 1->appx with 
        max 2.5% relative error, 2->appx with max 1.1% relative error, 
        and 3->'exact' (with fermi integral fits).
    cd : dict
        A dictionary holding all of the fundamental constants in the same
        units as the other input args (`ndens`, `T`, `masses`). Must have
        the following keys: na, kb, e, g, hbar, c, kc, me, mp.
        mass 
    mass : (N,) nd.array
        Mass of each species. 
    znuc : (N,) nd.array
        Atomic number of each species.
    T : Float
        Temperature of initial system.
    rho : Float
        Density of initial system
    mfracs : (N,) nd.array
        Mass fraction of each species for initial system.
   
    Attributes
    ----------
    s : {'T', 'M'}
        String dictating whether temperature or momentum relaxation collision 
        operators are used.
    o : {3, 1, 2}
        Order of approximation to use for electron wavelength. 1->appx with 
        max 2.5% relative error, 2->appx with max 1.1% relative error, 
        and 3->'exact' (with fermi integral fits).
    cd : dict
        A dictionary holding all of the fundamental constants in the same
        units as the other input args (`ndens`, `T`, `masses`). Must have
        the following keys: na, kb, e, g, hbar, c, kc, me, mp.
        mass 
    znuc : (N,) nd.array
        Atomic number of each species for any system.
    mass : (N,) nd.array
        Mass of each species for any system.
    zsolver : ZbarSolver
        Class holding the current effective charge. 
    nu : (N, N) nd.array
        Pairwise collision frequencies for each species.
    tdict : dict
        Dictionary holding the current transport coefficients, has keys `mus`, 
        `mub`, `k`, `dmat`, for shear and bulk viscosity, thermal conductivity,
        and species diffusion. 
     """

    
    def __init__(self, cd, mass, znuc, T, rho, mfracs, s='T', o=3):
        super(BGK_Htransport, self).__init__()
        self.s = s
        self.o = o
        self.cd = cd
        self.znuc = znuc
        self.mass = mass
        self.nu = np.zeros((mass.shape[0], mass.shape[0]), dtype=np.double)
        ndens = rho*mfracs/mass
        self.zsolver = ZbarSolver(ndens, znuc, T)
        self.tdict = {'mus':0.0, 'mub':0.0, 'k':0.0, 
                      'dmat':np.zeros((mass.shape[0], mass.shape[0]), 
                      dtype=np.double)}
        self.set_transport(T, rho, mfracs)
    
    
    def set_hydro_num(self, T, ndens):
        """Sets the  matrix of momentum relaxation collsion rates formulated 
        in the above reference for a system described by hydrodynamics, i.e.
        only a single temperature is avaliable.
    
        Parameters
        ----------
        T : float 
            System temperature in units of energy/kb
        ndens : (N,) np.ndarray
            Number density of each ion species in units of 1/length^{3}.

        """
        leff = get_leff(T, T, self.zsolver.zbars, ndens, self.cd, o=self.o)
        #set each upper triangular nu to g
        for i in range(ndens.shape[0]):
            for j in range(i, ndens.shape[0]):
                self.nu[i, j] = self.zsolver.zbars[i]*self.zsolver.zbars[j]
                self.nu[i,j] = self.nu[i,j]*self.cd['e']*self.cd['e']
                self.nu[i,j] = self.nu[i,j]/leff/(self.cd['kb']*T)
        self.nu = 128.0*np.pi*np.pi*get_calk(1, 1, self.nu)/3.0/np.power(2*np.pi, 1.5)
        #set each lower triangular element to its upper triangular equivalent
        for i in range(ndens.shape[0]):
            for j in range(i, ndens.shape[0]):
                self.nu[i,j] = self.nu[i,j]*np.power(self.zsolver.zbars[i]*\
                                    self.zsolver.zbars[j]*self.cd['e']*self.cd['e'], 2.0)
                self.nu[i,j] = self.nu[i,j]*np.sqrt(self.mass[i]*self.mass[j])
                self.nu[i,j] = self.nu[i,j]/np.sqrt((self.mass[i] + self.mass[j]))
                self.nu[i,j] = ndens[i]*ndens[j]*self.nu[i,j]
                self.nu[i,j] = self.nu[i,j]/np.power(self.cd['kb']*T, 1.5)
                self.nu[j,i] = self.nu[i,j]
        for i in range(ndens.shape[0]):
            self.nu[i, :] = self.cd['kc']*self.cd['kc']*self.nu[i, :]/(self.mass[i]*ndens[i])
    

    def set_hydro_nut(self, T, ndens):
        """Returns the  matrix of temperature relaxation collsion rates formulated 
        in the above reference for a system described by hydrodynamics, i.e.
        only a single temperature is avaliable.

        Parameters
        ----------
        T : float 
            System temperature in units of energy/kb
        ndens : (N,) np.ndarray
            Number density of each ion species in units of 1/length^{3}.
        """
        leff = get_leff(T, T, self.zsolver.zbars, ndens, self.cd, o=self.o)
        #set each upper triangular nu to g
        for i in range(ndens.shape[0]):
            for j in range(i, ndens.shape[0]):
                self.nu[i, j] = self.zsolver.zbars[i]*self.zsolver.zbars[j]
                self.nu[i,j] = self.nu[i,j]*self.cd['e']*self.cd['e']/leff
                self.nu[i,j] = self.cd['kc']*self.nu[i,j]/(self.cd['kb']*T)
        self.nu = 256.0*np.pi*np.pi*get_calk(1, 1, self.nu)/3.0/np.power(2*np.pi, 1.5)
        #set each lower triangular element to its upper triangular equivalent
        for i in range(ndens.shape[0]):
            for j in range(i, ndens.shape[0]):
                self.nu[i,j] = self.nu[i,j]*np.power(self.zsolver.zbars[i]*\
                               self.zsolver.zbars[j]*self.cd['e']*self.cd['e'],
                                2.0)
                self.nu[i,j] = self.nu[i,j]*np.sqrt(self.mass[i]*self.mass[j])
                self.nu[i,j] = self.nu[i,j]*np.power(self.mass[i] + self.mass[j],-1.5)
                self.nu[i,j] = ndens[i]*ndens[j]*self.nu[i,j]/np.power(self.cd['kb']*T, 1.5)
                self.nu[j,i] = self.nu[i,j]
        for i in range(ndens.shape[0]):
            self.nu[i, :] = self.cd['kc']*self.cd['kc']*self.nu[i, :]/ndens[i]

    def set_mus(self, T, ndens):
        """Set the shear viscosity coefficient in `self.tdict`.
        
        Parameters
        ----------
        T : float
            System temperature.
        ndens : (N,) np.ndarray
            Number density of each species.
        """
        self.tdict['mus'] = np.sum(ndens/np.sum(self.nu, axis=1))
        self.tdict['mus'] = self.cd['kb']*T*self.tdict['mus']


    def set_k(self, T, ndens):
        """Set the thermal conductivity coefficient in `self.tdict`.
        
        Parameters
        ----------
        T : float
            System temperature.
        ndens : (N,) np.ndarray
            Number density of each species.
        """
        self.tdict['k'] = np.sum(ndens/self.mass/np.sum(self.nu, axis=1))
        self.tdict['k'] = 0.5*5*self.cd['kb']*T*self.tdict['k']


    def set_dmat(self, T, ndens, mfracs):
        """Set the diffusion matrix coefficients in  `self.tdict`.

        Parameters
        ----------
        T : float
            System temperature.
        ndens : (N,) np.ndarray
            Number density of each species in the system.
        mfracs : (N,) np.ndarray
            Mass fractions of each species, should sum to 1.
        """
        if mfracs.shape[0] == 2:
            self.tdict['dmat'][:,:] = mfracs[0]*self.nu[0,1] + \
                                     mfracs[1]*self.nu[1,0]
            self.tdict['dmat'][:,:] = self.tdict['dmat'][:,:]/(self.nu[0,1]*\
                                        self.nu[1,0])
            self.tdict['dmat'][:,:] = ndens*self.cd['kb']*T*self.tdict['dmat'][:,:]
            self.tdict['dmat'][:,:] = self.tdict['dmat'][:,:]/np.sum(ndens*self.mass)
            self.tdict['dmat'][0,0] = -1*mfracs[1]*self.tdict['dmat'][0,0]/mfracs[0]
            self.tdict['dmat'][1,1] = -1*mfracs[0]*self.tdict['dmat'][1,1]/mfracs[1]
        elif mfracs.shape[0] > 2:
            for i in range(mfracs.shape[0]):
                for j in range(mfracs.shape[0]):
                    self.tdict['dmat'][i,j] = mfracs[i]*mfracs[j]*\
                                             self.nu[i,j]*self.nu[j,i]
                    self.tdict['dmat'][i,j] = self.tdict['dmat'][i,j]/\
                                    (mfracs[i]*self.nu[i,j] + mfracs[j]*self.nu[j,i])
                    self.tdict['dmat'][i,j] = self.tdict['dmat'][i,j]*np.sum(selfm.mass*ndens)
            self.tdict['dmat'] = np.sum(self.mass*ndens)*self.tdict['dmat'] 
            self.tdict['dmat'] = self.tdict['dmat']/ndens/(self.cd['kb']*T)
            self.tdict['dmat'] = sla.pinvh(self.tdict['dmat'])


    def set_transport(self, T, rho, mfracs):
        ndens = rho*mfracs/self.mass
        #self.zsolver.reset(ndens*self.cd['V2cm3'], self.znuc, T*self.cd['K2eV'])
        self.zsolver.zbars = self.znuc
        if self.s == 'T':
            self.set_hydro_nut(T, ndens)
        elif self.s == 'M':
            self.set_hydro_num(T, ndens)
        self.set_mus(T, ndens)
        self.set_k(T, ndens)
        self.set_dmat(T, ndens, mfracs)
            


def get_kinetic_num(Te, T, Zbar, ndens, mass, cd, o=3):
    """Returns the  matrix of momentum relaxation collsion rates formulated 
    in the above reference for a system described by kinetics, i.e.
    each ion species has its own temperature.
    
    Parameters
    ----------
    Te : float
        System electron temperature.
    T : (N,) np.ndarray 
        System ion temperatures in units of energy/kb
    Zbar : (N,) np.ndarray
        The effective charge of each ion species in units of electron
        charge.
    ndens : (N,) np.ndarray
        Number density of each ion species in units of 1/length^{3}.
    cd : dict
        A dictionary holding all of the fundamental constants in the same
        units as the other input args (`ndens`, `T`, `masses`). Must have
        the following keys: na, kb, e, g, hbar, c, kc, me, mp.
    mass : (N,) np.ndarray
        Mass of each species.
    o : {3, 1, 2}, optional
        Order of approximation to use. 1->appx with max 2.5% relative error,
        2->appx with max 1.1% relative error, and 3->'exact' (with fermi 
        integral fits).

    Returns
    -------
    nu : (N, N) np.ndarray
        The collision frequencies between each species. 
    """
    leff = get_leff(Te, T, Zbar, ndens, cd, o=o)
    nu = np.zeros((ndens.shape[0], ndens.shape[0]), dtype=ndens.dtype)
    #set each upper triangular nu to g
    for i in range(ndens.shape[0]):
        for j in range(i, ndens.shape[0]):
            nu[i, j] = Zbar[i]*Zbar[j]*cd['e']*cd['e']*(mass[i] + mass[j])
            nu[i,j] = nu[i,j]/leff/(cd['kb']*(mass[i]*T[j] + mass[j]*T[i]))
    nu = 128.0*np.pi*np.pi*get_calk(1, 1, nu)/3.0/np.power(2*np.pi, 1.5)
    #set each lower triangular element to its upper triangular equivalent
    for i in range(ndens.shape[0]):
        for j in range(i, ndens.shape[0]):
            nu[i,j] = nu[i, j]*np.power(Zbar[i]*Zbar[j]*cd['e']*cd['e'], 2.0)
            nu[i,j] = nu[i,j]*np.sqrt(mass[i]*mass[j])*(mass[i] + mass[j])
            nu[i,j] = ndens[i]*ndens[j]*nu[i,j]
            nu[i,j] = nu[i,j]/np.power(cd['kb']*(mass[i]*T[j] + mass[j]*T[i]), 1.5)
            nu[j,i] = nu[i,j]
    for i in range(ndens.shape[0]):
        nu[i, :] = nu[i, :]/(mass[i]*ndens[i])
    return nu


def get_kinetic_nut(Te, T, Zbar, ndens, mass, cd, o=3):
    """Returns the  matrix of temperature relaxation collsion rates formulated 
    in the above reference for a system described by hydrodynamics, i.e.
    only a single temperature is avaliable.
    
    Parameters
    ----------
    Te : float 
        System electron temperature in units of energy/kb
    T : (N,) np.ndarray
        System ion temperatures in units of energy/kb
    Zbar : (N,) np.ndarray
        The effective charge of each ion species in units of electron
        charge.
    ndens : (N,) np.ndarray
        Number density of each ion species in units of 1/length^{3}.
    cd : dict
        A dictionary holding all of the fundamental constants in the same
        units as the other input args (`ndens`, `T`, `masses`). Must have
        the following keys: na, kb, e, g, hbar, c, kc, me, mp.
    mass : (N,) np.ndarray
        Mass of each species.
    o : {3, 1, 2}, optional
        Order of approximation to use. 1->appx with max 2.5% relative error,
        2->appx with max 1.1% relative error, and 3->'exact' (with fermi 
        integral fits).

    Returns
    -------
    nu : (N, N) np.ndarray
        The collision frequencies between each species. 
    """
    leff = get_leff(Te, T, Zbar, ndens, cd, o=o)
    nu = np.zeros((ndens.shape[0], ndens.shape[0]), dtype=ndens.dtype)
    #set each upper triangular nu to g
    for i in range(ndens.shape[0]):
        for j in range(i, ndens.shape[0]):
            nu[i, j] = Zbar[i]*Zbar[j]*cd['e']*cd['e']*(mass[i] + mass[j])
            nu[i, j] = nu[i,j]/leff/(cd['kb']*(mass[i]*T[j] + mass[j]*T[i]))
    nu = 256.0*np.pi*np.pi*get_calk(1, 1, nu)/3.0/np.power(2*np.pi, 1.5)
    #set each lower triangular element to its upper triangular equivalent
    for i in range(ndens.shape[0]):
        for j in range(i, ndens.shape[0]):
            nu[i,j] = nu[i,j]*np.power(Z[i]*Z[j]*cd['e']*cd['e'], 2.0)
            nu[i,j] = nu[i,j]*np.sqrt(mass[i]*mass[j])
            nu[i,j] = ndens[i]*ndens[j]*nu[i,j]
            nu[i,j] = nu[i,j]/np.power(cd['kb']*(mass[i]*T[j] + mass[j]*T[i]), 1.5)
            nu[j,i] = nu[i,j]
    for i in range(ndens.shape[0]):
        nu[i, :] = nu[i, :]/ndens[i]
    return nu



