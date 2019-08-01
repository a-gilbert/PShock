"""Fundamental constants in some common unit systems. 

Constants in each dict indexed by following naming convention:

na - Avagadro's constant
kb - Boltzmann's constant
e - elementary charge
g - newtonian gravitational constant
hbar - Planck's Constant/2*pi
c - speed of light
kc - Coulomb's constant.
me - electron mass
mp - proton mass"""

si_const = {'na':6.02214076e23,
            'kb':1.380649e-23,
            'e':1.602176634e-19,
            'g':6.67430e-11,
            'hbar':1.054572e-34,
            'c':2.99792458e8,
            'kc':8.9875517923e9,
            'me':9.1093837015e-31,
            'mp':1.67262192369e-27,
            'K2eV':1/(1.160451812e4),
            'V2cm3':1e-2*1e-2*1e-2}

cgs_const = {'na':6.02214076e23,
             'kb':1.3806504e-16,
             'e':4.80320427e-10,
             'g':6.67430e-8,
             'hbar':1.0545716e-27,
             'c':2.99792458e10,
             'kc':1.0,
             'me':9.1093837015e-28,
             'mp':1.67262192369e-27}

#hartree units, but with temp in units of eV
ha_const = {'na':6.02214076e23,
            'kb':1.0/27.211386245988,
            'e':1.0,
            'g':2.40061e-43,
            'hbar':1.0,
            'c':137.035999084,
            'kc':1.0,
            'me':1.0,
            'mp':1.83615267343e3}