import sys
import numpy as np
sys.path.insert(0, 'protocols')
import curvature_audit_lib as cal
from pyscf import gto
from pyscf.hessian import thermo as pyscf_thermo

MASSES = np.array([15.999, 15.999, 15.999, 15.999, 1.008, 1.008])
rng = np.random.default_rng(11)
coords = rng.normal(scale=1.5, size=(6, 3))
V = cal.tr_subspace(MASSES, coords)
# internal_subspace lives in subspace_revision_analyze; replicate via QR
B = np.hstack([V, np.eye(18)])
Q, _ = np.linalg.qr(B)
U = Q[:, 6:]
eigs_B = np.array([-1.234e-4, 5e-5, 1e-4, 3e-4, 7e-4, 2e-3, 5e-3, 1e-2,
                   3e-2, 1e-1, 3e-1, 6e-1])
H18 = U @ np.diag(eigs_B) @ U.T
hblock = np.zeros((6, 6, 3, 3))
for i in range(6):
    for j in range(6):
        hblock[i, j] = H18[3*i:3*i+3, 3*j:3*j+3]
s = "; ".join("%s %.10f %.10f %.10f" % (s_, x, y, z)
              for s_, (x, y, z) in zip(['O']*4 + ['H']*2, coords))
mol = gto.M(atom=s, basis='sto-3g', charge=0, spin=0, verbose=0)
ha = pyscf_thermo.harmonic_analysis(mol, hblock, imaginary_freq=False)
pf = np.sort(np.asarray(ha['freq_wavenumber'], float))
NU = np.sqrt(4.3597447222071e-18 / (1.66053906660e-27 * 0.52917721092e-10**2)) \
    / (2*np.pi*2.99792458e10)
mine = np.sort(np.array([np.sign(x)*np.sqrt(abs(x))*NU for x in eigs_B]))
print('pyscf:', [round(f, 2) for f in pf])
print('mine :', [round(f, 2) for f in mine])
print('max diff:', np.abs(pf - mine).max())
print('pyscf masses:', mol.atom_mass_list(isotope_avg=True))
