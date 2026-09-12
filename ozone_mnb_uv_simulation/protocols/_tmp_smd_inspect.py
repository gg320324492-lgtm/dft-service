import inspect, os
import pyscf
import pyscf.solvent.smd as smd
from pyscf.solvent import _attach_solvent

print('pyscf version:', pyscf.__version__)
print('smd module file:', smd.__file__)
print()
print('== SMD class attributes/params ==')
sig = inspect.signature(smd.SMD.__init__)
print('SMD.__init__:', sig)
src = inspect.getsource(smd.SMD)
print(src[:3000])
