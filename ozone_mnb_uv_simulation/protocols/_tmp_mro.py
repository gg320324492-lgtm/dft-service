import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from pyscf import gto
from grad_factory import make_mf_d2_gr

SYMS = ['O', 'O', 'O', 'O', 'H', 'H']
coords = np.zeros((6, 3))
coords[0] = [0, 0, 0]; coords[1] = [0, 0, 1.2]; coords[2] = [0, 0, -1.2]
coords[3] = [3.0, 0, 0]; coords[4] = [3.757, 0.587, 0]; coords[5] = [2.243, 0.587, 0]
atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                 for s, (x, y, z) in zip(SYMS, coords))
mol = gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0, verbose=0,
            max_memory=4000)
mf = make_mf_d2_gr(mol, solvent='water', grid_response=True)
print('MRO of mf:')
for i, c in enumerate(type(mf).__mro__):
    print('  %2d %s' % (i, c.__name__))
undo = mf.undo_solvent()
print('MRO of undo_solvent():')
for i, c in enumerate(type(undo).__mro__):
    print('  %2d %s' % (i, c.__name__))
