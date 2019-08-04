"""Classes and functions needed to generate  different properties
of effectively screened plasmas, as defined in [1].

[1]Stanton, L. G., & Murillo, M. S. (2016). 
Ionic transport in high-energy-density matter. 
Physical Review E, 93(4), 043203. 
https://doi.org/10.1103/PhysRevE.93.043203
"""
import numpy as np
import math
from fermi_funcs import fint, ifint

#a & b are indexed by (n, m)
a = {1:{1:np.array([1.4660, -1.7836, 1.4313, -0.55833, 0.061162], 
          dtype=np.double),
        2:np.array([0.52094, 0.25153, -1.1337, 1.2155, -0.43784], 
          dtype=np.double),
        3:np.array([0.30346, 0.23739, -0.62167, 0.56110, -0.18046],
          dtype=np.double)},
    2:{2:np.array([0.85401, -0.22898, -0.60059, 0.80591, -0.30555],
          dtype=np.double)}
    }

b = {1:{1:np.array([0.081033, -0.091336, 0.051760, -0.50026, 0.17044],
          dtype=np.double),
        2:np.array([0.20572, -0.16536, 0.061572, -0.12770, 0.066993],
          dtype=np.double),
        3:np.array([0.68375, -0.38459, 0.10711, 0.10649, 0.028760],
          dtype=np.double)},
     2:{2:np.double([0.43475, -0.21147, 0.11116, 0.19665, 0.15195],
          dtype=np.double)}
     }


#c & d are indexed by n only
c = {1: np.array([0.30031, -0.69161, 0.59607, -0.39822, -0.20685], dtype=np.double),
     2: np.array([0.40688, -0.86425, 0.77461, -0.34471, -0.27626], dtype=np.double)}

d = {1: np.array([0.48516, 1.66045, -0.88687, 0.55990, 1.65798, -1.02457], dtype=np.double),
     2: np.array([0.83061, 2.05229, -0.59902, 1.41500, 0.78874, -0.48155], dtype=np.double)}


def get_iradius(Zbar, ndens, cd):
    """Return the Wigner-Seitz ion sphere radius for each mixture component.
    
    Parameters
    -----------
    Zbar : (N,) np.ndarray
        The effective charge of each ion species in units of electron
        charge.
    ndens : (N,) np.ndarray
        Number density of each ion species in units of 1/length^{3}.
    cd : dict
        A dictionary holding all of the fundamental constants in the same
        units as the other input args (`ndens`,`T`). Must have the following
        keys: na, kb, e, g, hbar, c, kc, me, mp.

    Returns
    -------
    ai : (N,) np.ndarray
        Ion sphere radius for each species.
    """
    ai = np.power(3*Zbar/(4*np.pi*np.sum(ndens*Zbar)), 1.0/3.0)
    return ai


def get_lambdai(T, Zbar, ndens, cd):
    """Generate the Debye-Huckel screening length for each ion species. 
    
    Parameters
    ----------
    T : float, (N,) np.ndarray
        System ion temperature(s) in units of energy/kb.
    Zbar : (N,) np.ndarray
        The effective charge of each ion species in units of electron
        charge.
    ndens : (N,) np.ndarray
        Number density of each ion species in units of 1/length^{3}.
    cd : dict
        A dictionary holding all of the fundamental constants in the same
        units as the other input args (`ndens`,`T`). Must have the following
        keys: na, kb, e, g, hbar, c, kc, me, mp.
        
    Returns
    --------
    lambdai : (N,) np.ndarray
        Debye-Huckel screening length for each ion species, in units of length.
    """
    lambdai = 4.0*np.pi*cd['kc']*ndens*(cd['e']*Zbar)**2
    lambdai = cd['kb']*T/lambdai
    lambdai = np.sqrt(lambdai)
    return lambdai


def get_lambdae(T, Zbar, ndens, cd, o=3):
    """Get the electron screening length (Thomas-Fermi length) for the system using
    an approximation or fermi integral fits.
    
    Parameters
    ----------
    T : float
        System temperature in units of energy/kb.
    Zbar : (N,) np.ndarray
        The effective charge of each ion species in units of electron
        charge.
    ndens : (N,) np.ndarray
        Number density of each ion species in units of 1/length^{3}.
    cd : dict
        A dictionary holding all of the fundamental constants in the same
        units as the other input args (`ndens`,`T`). Must have the following
        keys: na, kb, e, g, hbar, c, kc, me, mp.
    o : {3,1,2}, optional
        Order of approximation to use. 1->appx with max 2.5% relative error,
        2->appx with max 1.1% relative error, and 3->'exact'(with fermi integral
        fits).

    Returns
    -------
    le : float
        Electron screening length.
    """
    le = np.sum(Zbar*ndens)
    if o != 3:
        le = 3.0*np.pi*np.pi*le
        le = cd['hbar']*cd['hbar']*np.power(le, 2.0/3.0)/(3.0*cd['me'])
        if o == 1:
            le = np.sqrt((cd['kb']*T)**2 + le**2)
        elif o==2:
            le = np.power(np.power(cd['kb']*T, 9.0/5.0) + np.power(le, 9.0/5.0), 5.0/9.0)
        le= 4.0*np.pi*cd['e']*cd['e']*np.sum(Zbar*ndens)/le
    else:
        le = np.pi*np.pi*le/np.sqrt(2.0*(cd['kb']*T)**3)
        le = ifint(le, 0.5)
        le = np.sqrt(8.0*cd['kb']*T)*fint(le, -0.5)/np.pi
    le = np.sqrt(1/le)
    return le


def get_gamma(T, Zbar, ndens, cd):
    """Get the plasma coupling parameters for each species.

    Parameters
    ----------
    T : float
        System temperature in units of energy/kb.
    Zbar : (N,) np.ndarray
        The effective charge of each ion species in units of electron
        charge.
    ndens : (N,) np.ndarray
        Number density of each ion species in units of 1/length^{3}.
    cd : dict
        A dictionary holding all of the fundamental constants in the same
        units as the other input args (`ndens`,`T`). Must have the following
        keys: na, kb, e, g, hbar, c, kc, me, mp.

    Returns
    -------
    gamma : (N,) np.ndarray
        Coupling parameter of each ion species.
    """
    gamma = get_iradius(Zbar, ndens, cd)
    gamma = (cd['kc']*(Zbar*cd['e'])**2)/gamma/(cd['kb']*T)
    return gamma


def get_kappa(T, Zbar, ndens, cd, o=3):
    """Returns the Wigner-Seitz radius in units of the electron screening length.
    
    Parameters
    ----------
    T : float
        System temperature in units of energy/kb.
    Zbar : (N,) np.ndarray
        The effective charge of each ion species in units of electron
        charge.
    ndens : (N,) np.ndarray
        Number density of each ion species in units of 1/length^{3}.
    cd : dict
        A dictionary holding all of the fundamental constants in the same
        units as the other input args (`ndens`,`T`). Must have the following
        keys: na, kb, e, g, hbar, c, kc, me, mp.
    o : {3,1,2}, optional
        Order of approximation to use when calculating lambdae. 1->appx with 
        max 2.5% relative error, 2->appx with max 1.1% relative error, 
        and 3->'exact'(with fermi integralfits).

    Returns
    ---------- 
    kappa : (N,) nd.array
        Ion radii in units of electron screening length.
    """
    kappa = get_iradius(Zbar, ndens, cd)/get_lambdae(T, Zbar, ndens, cd, o=o)
    return kappa


def get_leff(Te, T, Zbar, ndens, cd, o=3):
    """Obtain the effective screening length for the plasma system.
    
    Parameters
    ----------
    Te : float
        System electron temp in units of energy/kb.
    T : float, (N,) nd.array
        System ion temperature(s) in units of energy/kb.
    Zbar : (N,) np.ndarray
        The effective charge of each ion species in units of electron
        charge.
    ndens : (N,) np.ndarray
        Number density of each ion species in units of 1/length^{3}.
    cd : dict
        A dictionary holding all of the fundamental constants in the same
        units as the other input args (`ndens`,`T`). Must have the following
        keys: na, kb, e, g, hbar, c, kc, me, mp.
    o : {3,1,2}, optional
        Order of approximation to use. 1->appx with max 2.5% relative error,
        2->appx with max 1.1% relative error, and 3->'exact'(with fermi integral
        fits).

    Returns
    -------
    leff : float
        Effective screening length for the system.
    """
    leff = get_lambdai(T, Zbar, ndens, cd)**2
    leff = leff*(1+ 3.0*get_gamma(T, Zbar, ndens, cd))
    leff = np.sum(1/leff)
    leff = leff + (1.0/get_lambdae(Te, Zbar, ndens, cd, o=o)**2)
    leff = np.power(leff, -0.5)
    return leff


def get_phi_sc(n, w):
    """Return the fit function for the cross section of a strongly coupled plasma
    
    Parameters
    ---------- 
    n : {1, 2}
        Order of cross section to find. 
    w : np.ndarray, float
        Dimensionless Velocity
    
    Returns
    -------
    out : fit function phi 
    """
    out = c[n][3]
    for i in range(2, -1, -1):
        out = c[n][i] + np.log(w)*out
    out = out/(1 + c[n][4]*np.log(w))
    return out


def get_phi_wc(n, w):
    """Return the fit function for the cross section  of a weakly coupled plasma.
    
    Parameters
    ----------
    n : {1, 2}
        Order of the cross section to find.
    w : np.ndarray or float
        Dimensionless Velocity
    
    Return
    ------
    out : fit function phi
    """
    num = np.log(w)
    den = np.log(w)
    for i in range(2, -1, -1):
        num = d[n][i] + np.log(w)*num
        den = d[n][3+i] + np.log(w)*den
    num = 0.5*n*np.log(1 + w*w)*num/den/np.power(w, 4)
    return num


def get_momentum_csections(n, v, Te, T, Zbar, ndens, m, cd, leo=3):
    """Get numerical fits for the momentum cross sections of order `n`
    for each pairwise interaction in the system. 

    Parameters
    ----------
    n : {1, 2}
        Order of cross section to compute.
    v : float
        Velocity at which to evaluate the cross sections.
    Te : float
        Electron temperature of the system. 
    T : float, (N,) np.ndarray
        System ion temperature(s) in units of energy/kb.
    Zbar : (N,) np.ndarray
        The effective charge of each ion species in units of electron
        charge.
    ndens : (N,) np.ndarray
        Number density of each ion species in units of 1/length^{3}.
    m : (N,) np.ndarray
        Mass of each ion species.
    cd : dict
        A dictionary holding all of the fundamental constants in the same
        units as the other input args (`ndens`,`T`). Must have the following
        keys: na, kb, e, g, hbar, c, kc, me, mp.
    leo : {3,1,2}, optional
        Order of approximation to use for electron screening length. 1->appx 
        with max 2.5% relative error, 2->appx with max 1.1% relative error, 
        and 3->'exact'(with fermi integral fits).

    Returns
    -------
    s : (0.5*N*(N+1),) np.ndarray
        Pairwise momentum cross sections, a flattened array of the non-zero values
        in an upper triangular matrix.
    """
    leff = get_leff(Te, T, Zbar, ndens, cd, o=leo)
    s = np.zeros(Zbar.shape[0]*(Zbar.shape[0]+1)/2, dtype=ndens.dtype)
    for i in range(len(Zbar)):
        for j in range(i, len(Zbar)):
            s[i,j] = leff*v*vm[i]*m[j]/(m[i] + m[j])
            s[i,j] = 0.5*s[i,j]/(Zbar[i]*Zbar[j]*cd['e']*cd['e'])
    s = leff*s 
    s = np.sqrt(s)
    c = s<=1.0
    s[c==True] = get_phi_sc(n, s[c==True])
    s[c==False] = get_phi_wc(n, s[c==False])
    s = 2*np.pi*leff*leff*s
    return s


def get_calk_wc(n, m, g):
    """Return the weakly coupled fit to the dimensionless integral 
    caligraphic K.
    
    Parameters
    ----------
    n : {1,2}
        One of the order parameters for the fit function.
    m : {1,2,3}
        One of the order parameters for the fit function.
    g : np.ndarray, float
        Coupling parameters.
    
    -------
    out : np.ndarray, float
        The fit function. 
    """
    out = a[n][m][-1]
    for i in range(3, -1, -1):
        out = a[n][m][i] + out*g
    out = g*out
    out = -0.25*n*float(math.factorial(m-1))*np.log(out)
    return out


def get_calk_sc(n, m, g):
    """Return the strongly coupled fit to the dimensionless integral 
    caligraphic K.
    
    Parameters
    ----------
    n : {1,2}
        One of the order parameters for the fit function.
    m : {1,2,3}
        One of the order parameters for the fit function.
    g : np.ndarray, float
        Coupling parameters.
    
    -------
    out : np.ndarray, float
        The fit function.
    """
    out = b[n][m][0] + np.log(g)*(b[n][m][1] + np.log(g)*b[n][m][2])
    out = out/(1 + g*(b[n][m][3] + b[n][m][4]*g))
    return out


def get_calk(n, m, g):
    """Convenience function for calculation of caligraphic K in other modules. 

    Parameters
    ----------
    n : {1,2}
        One of the order parameters for the fit function.
    m : {1,2,3}
        One of the order parameters for the fit function.
    g : np.ndarray, float
        Coupling parameters.
    
    -------
    out : np.ndarray, float
        The fit function.
    
    """
    out = np.zeros_like(g)
    c = g<=1.0
    out[c==True] = get_calk_wc(n, m, g[c==True])
    out[c==False] = get_calk_sc(n, m, g[c==False])
    return out



def get_cints(n, m, T, Zbar, ndens, mass, cd, leo=3):
    """Get numerical fits for the collision integral of order (`n`,`m`)
    for each pairwise interaction in the system. 

    Parameters
    ----------
    n : {1, 2}
        An order parameter of the collision integral.
    m : {1, 2, 3}
        An order parameter of the collision integral.
    T : float
        System temperature in units of energy/kb.
    Zbar : (N,) np.ndarray
        The effective charge of each ion species in units of electron
        charge.
    ndens : (N,) np.ndarray
        Number density of each ion species in units of 1/length^{3}.
    mass : (N,) np.ndarray
        Mass of each ion species.
    cd : dict
        A dictionary holding all of the fundamental constants in the same
        units as the other input args (`ndens`,`T`). Must have the following
        keys: na, kb, e, g, hbar, c, kc, me, mp.
    leo : {3,1,2}, optional
        Order of approximation to use for electron screening length. 1->appx 
        with max 2.5% relative error, 2->appx with max 1.1% relative error, 
        and 3->'exact'(with fermi integral fits).

    Returns
    -------
    omega : (0.5*N*(N+1),) np.ndarray
        Pairwise collision integrals, a flattened array of the non-zero values
        in an upper triangular matrix.
    """
    leff = get_leff(T, Zbar, ndens, cd, o=leo)
    omega = np.zeros(Zbar.shape[0]*(Zbar.shape[0]+1)/2, dtype=ndens.dtype)
    #set g in each entry
    for i in range(ndens.shape[0]):
        for j in range(i, ndens.shape[0]):
            omega[i,j] = Zbar[i]*Zbar[j]*cd['e']*cd['e']/leff/(cd['kb']*T)
    #Evaluate each entry.
    c = omega <= 1.0
    omega[c==True] = get_calk_wc(n, m, omega[c==True])
    omega[c==False] = get_calk_sc(n, m, omega[c==False])
    #Multiply by pre-factors.
    for i in range(ndens.shape[0]):
        for j in range(i, ndens.shape[0]):
            omega[i,j] = omega[i,j]*(Zbar[i]*Zbar[j]*cd['e']*cd['e'])**2
            omega[i,j] = omega[i,j]/np.power(cd['kb']*T, 3.0/2.0)
            omega[i,j] = omega[i, j]*np.sqrt(2*np.pi*(mass[i] + mass[j])/(mass[i]*mass[j]))
    return omega