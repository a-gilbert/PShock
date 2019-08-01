from fermi_funcs import fint_arr, ifint_arr
from fdint import *
from zbar import ZbarSolver
from sm_pproperties import get_phi_wc, get_phi_sc
import numpy as np
import matplotlib.pyplot as plt


#Test ZBar functionality
Znuc = np.array([1.0, 6.0, 8.0], dtype=np.double)
ndens = np.array([2.663e21, 6.305e22, 5.511e20], dtype=np.double)

ZTest = ZbarSolver(ndens, Znuc, 12.6)

print("Test ZBar:")
print(ZTest.zbars)
print("Test ZBar Charge Densities (should all be equal):")
print(ZTest.zbars/ZTest.s1)

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
