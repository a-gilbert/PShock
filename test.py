from fermi_funcs import fint_arr, ifint_arr
from fdint import *
from zbar import ZbarSolver
from sm_pproperties import get_phi_wc, get_phi_sc
from bgk_transport import BGK_Htransport
from constants import si_const
import numpy as np
import matplotlib.pyplot as plt

test_zbar = True
test_fermi = True
test_csection = True
test_bmus = True


def get_thermovars(cd, zbar, kappa, gamma):
   T = cd['hbar']*cd['hbar']*np.power(0.25*9*np.pi*zbar, 2.0/3.0)
   T = T/(cd['e']*cd['e']*zbar*zbar*3.0*cd['me'])
   ndens = 9/(kappa**4) - (1.0/gamma/gamma)
   ndens = 1/np.sqrt(ndens)
   ndens = T*ndens
   T = cd['kc']*cd['e']*cd['e']*zbar*zbar/(ndens*gamma*cd['kb'])
   ndens = 0.75/(np.pi * (ndens**3))
   return ndens, T


#get plasma frequency
def get_pfreq(cd, zbar, n, m):
   out = 4*np.pi*cd['kc']*n*zbar*zbar*cd['e']*cd['e']/m
   out = np.sqrt(out)
   return out

if test_zbar:
   #Test ZBar functionality
   Znuc = np.array([1.0, 6.0, 8.0], dtype=np.double)
   ndens = np.array([2.663e21, 6.305e22, 5.511e20], dtype=np.double)

   ZTest = ZbarSolver(ndens, Znuc, 12.6)

   print("Test ZBar:")
   print(ZTest.zbars)
   print("Test ZBar Charge Densities (should all be equal):")
   print(ZTest.zbars/ZTest.s1)

if test_fermi:
   #Test Antia fermi integral functionality
   ns = [-0.5, 0.5, 1.5, 2.5]
   for i in range(len(ns)):
      print("Testing n=%e" % ns[i])
      x = np.linspace(-10, 50, num=200, dtype=np.float64)
      xorig = np.copy(x)
      fval = fint_arr(x, ns[i], order=3)
      ex_fval = fdk(ns[i], xorig)
      fval2 = np.copy(ex_fval)
      xinv = ifint_arr(fval2, ns[i], order=1)
      fval2 = fint_arr(xinv, ns[i], order=3)
      print("Max rdiff for n=%e:"%ns[i])
      print(np.max(np.abs(ex_fval-fval2)/np.abs(ex_fval)))
      plt.close('all')
      plt.semilogy(xorig, np.abs(ex_fval-fval)/np.abs(ex_fval), label=r'rel_err, $F_{n}(x)$')
      plt.semilogy(xorig, np.abs(ex_fval - fval2)/np.abs(ex_fval), label=r'rel_err, $F_{n}(x) - F_{n}(F^{inv}_{n}(F_{n}(x)))$')
      plt.legend()
      plt.title('n=%.1e'%ns[i])
      plt.savefig("fint%d.png"%i)


if test_csection:
   #Test SM cross section
   w = np.logspace(-8, 2.5, num=200)
   c = w<=1.0
   p1 = np.zeros_like(w)
   p2 = np.zeros_like(w)
   p1[c==True] = get_phi_sc(1, w[c==True])
   p1[c==False] = get_phi_wc(1, w[c==False])
   p2[c==True] = get_phi_sc(2, w[c==True])
   p2[c==False] = get_phi_wc(2, w[c==False])

   plt.close('all')
   plt.loglog(w, p1)
   plt.ylabel(r'$\phi_{1}(w)$')
   plt.xlabel(r'dimensionless velocity $w$')
   plt.ylim(1e-10, 1e5)
   plt.tight_layout()
   plt.savefig('phi1.png')
   plt.close('all')
   plt.loglog(w, p2)
   plt.ylabel(r'$\phi_{2}(w)$')
   plt.xlabel(r'dimensionless velocity $w$')
   plt.tight_layout()
   plt.savefig('phi2.png')


if test_bmus:
   #Test BGK Transport
   #mh in kg
   mh = 1.673823378528e-27
   m = np.array([mh], dtype=np.double)
   z = np.array([1.0], dtype=np.double)
   mfrac = np.array([1.0], dtype=np.double)
   #dens in 1/m^{3}
   mus = []
   gammas = np.logspace(0, 2, num=100)
   ndens, T = get_thermovars(si_const, 1.0, 1.0, gammas[0])
   t = BGK_Htransport(si_const, m, z, T, 
                      mh*ndens, mfrac, o=2)
   for i in range(gammas.shape[0]):
      ndens, T = get_thermovars(si_const, 1.0, 1.0, gammas[i])
      t.set_transport(T, mh*ndens, mfrac)
      a = np.power(3.0/4.0/np.pi/ndens, 1.0/3.0)
      wp = get_pfreq(t.cd, 1.0, ndens, mh)
      wp = mh*ndens*wp*a*a
      mus.append(t.tdict['mus']/wp)
   mus = np.array(mus, dtype=np.double)
   plt.close('all')
   plt.semilogx(gammas, mus, label='BGK')
   plt.ylabel(r'$\eta^{*}$')
   plt.xlabel(r'$\Gamma$')
   plt.savefig('bgk_visc.png')

