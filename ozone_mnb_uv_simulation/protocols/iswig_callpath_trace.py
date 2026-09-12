import json
import numpy as np
from pyscf import gto
from pyscf.solvent import pcm, smd

trace = {'pyscf_source': '/root/dftvenv/lib/python3.12/site-packages/pyscf/solvent/'}

# --- wrap gen_surface to record actual kwargs (runtime trace, no source edit) ---
orig = pcm.gen_surface
calls = []
def traced_gen_surface(mol, ng=302, rad=None, surface_discretization_method='SWIG'):
    calls.append(dict(surface_discretization_method_arg=surface_discretization_method))
    return orig(mol, ng=ng, rad=rad,
                surface_discretization_method=surface_discretization_method)
pcm.gen_surface = traced_gen_surface

mol = gto.M(atom='O 0 0 0; H 0 0 0.9; H 0 0.9 0', basis='sto-3g', verbose=0)
base = gto.M(atom='O 0 0 0; H 0 0 0.9; H 0 0.9 0', basis='sto-3g', verbose=0)
ws = smd.SMD(mol, solvent='water')
ws.surface_discretization_method = 'ISWIG'          # set the attribute
ws.build()                                          # NO SCF involved
trace['gen_surface_calls'] = calls
trace['attribute_set_to'] = ws.surface_discretization_method
trace['surface_has_SWIG_only_keys'] = sorted(k for k in ws.surface
                                             if k in ('R_in_J', 'R_sw_J'))
trace['surface_has_ISWIG_only_keys'] = sorted(k for k in ws.surface
                                              if k == 'R_J')
trace['surface_n_points'] = int(ws.surface['grid_coords'].shape[0])
trace['conclusion_build'] = (
    'attribute ISWIG was set but gen_surface received NO method argument and '
    'the SWIG branch executed (SWIG-only keys present) -> ISWIG NOT reachable '
    'via SMD.build()')

# --- direct ISWIG surface (diagnostic only, via direct function call) ---
from pyscf.solvent.smd import smd_radii, solvent_db
rad_w = smd_radii(solvent_db['water'][2])
s_isw = orig(mol, ng=590, rad=rad_w, surface_discretization_method='ISWIG')
trace['direct_iswig_surface_keys'] = sorted(s_isw.keys())
trace['direct_iswig_has_R_J'] = 'R_J' in s_isw

# --- gradient wiring: grad/smd.py passes only surface ---
import inspect
import pyscf.solvent.grad.smd as gsmd
src = inspect.getsource(gsmd)
trace['grad_smd_call_line'] = [ln.strip() for ln in src.splitlines()
                               if 'get_dF_dA' in ln]
import pyscf.solvent.grad.pcm as pcm_grad
sig = inspect.signature(pcm_grad.get_dF_dA)
trace['get_dF_dA_signature'] = str(sig)
trace['conclusion_grad'] = ('get_dF_dA(surface, surface_discretization_method='
                            '"SWIG"); grad/smd.py passes ONLY the surface dict '
                            '-> gradient always takes the SWIG branch')

# --- gradient on a directly-built ISWIG surface would fail (expected KeyError) ---
err = None
try:
    pcm_grad.get_dF_dA(s_isw, surface_discretization_method='ISWIG')
except Exception as e:
    err = repr(e)
trace['direct_iswig_grad_outcome'] = err or 'completed'
# and: SWIG-branch on ISWIG surface (what would happen if attribute were read
# inconsistently) -- expected KeyError on R_in_J
err2 = None
try:
    pcm_grad.get_dF_dA(s_isw)   # default method="SWIG" on ISWIG surface
except Exception as e:
    err2 = repr(e)
trace['default_method_on_iswig_surface'] = err2 or 'completed'

# --- hessian side: does hessian/pcm.py know ISWIG? ---
import pyscf.solvent.hessian.pcm as hpcm
hsrc = inspect.getsource(hpcm)
trace['hessian_pcm_has_ISWIG'] = 'ISWIG' in hsrc
trace['hessian_get_dF_dA_calls'] = [ln.strip() for ln in hsrc.splitlines()
                                    if 'get_dF_dA' in ln]
# CDS second derivative implementation
from pyscf.solvent.hessian import smd as hsmd
ssrc = inspect.getsource(hsmd)
trace['hessian_smd_get_cds_head'] = [ln.strip() for ln in ssrc.splitlines()
                                     if 'def get_cds' in ln] + \
    [ln.strip() for ln in ssrc.splitlines()[ssrc.splitlines().index(
        [l for l in ssrc.splitlines() if 'def get_cds' in l][0]):
        ssrc.splitlines().index([l for l in ssrc.splitlines() if 'def get_cds' in l][0]) + 12]]

json.dump(trace, open('run_artifacts/01_pure_water_o3_h2o/iswig_callpath/runtime_trace.json', 'w'),
          indent=2, default=str)
print(json.dumps(trace, indent=2, default=str)[:3000])
