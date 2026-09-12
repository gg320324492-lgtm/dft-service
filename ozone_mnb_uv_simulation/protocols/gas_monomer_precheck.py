import os, sys, json
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
import d2_full
from pyscf import gto

checks = {}

# 1) D2 parameter coverage for N/O/H
checks['d2_C6'] = {k: d2_full.C6_TABLE[k] for k in ('H', 'N', 'O')}
checks['d2_RvdW'] = {k: d2_full.RVDW_TABLE[k] for k in ('H', 'N', 'O')}
checks['d2_coverage'] = all(k in d2_full.C6_TABLE and k in d2_full.RVDW_TABLE
                            for k in ('H', 'N', 'O'))

# 2) units: d2 uses coords in angstrom internally (check _coords_ang usage)
import inspect
src = inspect.getsource(d2_full.d2_energy)
checks['d2_energy_takes_bohr_coords'] = ('coords_bohr' in src)

# 3) D2 energy-gradient consistency (NO SCF): analytic d2_grad vs FD of d2_energy
mol = gto.M(atom='N 0 0 0.11; H 0.94 0 0.25; H -0.47 0.815 0.25; H -0.47 -0.815 0.25',
            basis='def2-tzvp', charge=0, spin=0, verbose=0)
g_an = d2_full.d2_grad(mol).reshape(-1)
h = 1e-4
g_fd = np.zeros(3 * mol.natm)
c0 = mol.atom_coords(unit='Bohr').copy()
for i in range(mol.natm):
    for a in range(3):
        cp = c0.copy(); cm = c0.copy()
        cp[i, a] += h; cm[i, a] -= h
        mp = mol.copy(); mp.set_geom_(cp, unit='Bohr')
        mm = mol.copy(); mm.set_geom_(cm, unit='Bohr')
        ep = d2_full.d2_energy(mp)
        em = d2_full.d2_energy(mm)
        g_fd[3 * i + a] = (ep - em) / (2 * h)  # FD in Bohr, Eh/Bohr
err = float(np.abs(g_an - g_fd).max())
rel = err / (np.abs(g_an).max() + 1e-300)
checks['d2_grad_consistency'] = dict(analytic_maxabs=float(np.abs(g_an).max()),
                                     fd_maxabs=float(np.abs(g_fd).max()),
                                     abs_err=err, rel_err=rel,
                                     pass_=bool(err < 1e-6))

# 4) monomer inputs: NH3 (non-planar C3v guess, sourced) & O3 (from baseline, traceable)
r, ang = 1.012, 106.7          # CRC-style standard guess (source: standard molecular geometry)
import math
half = math.radians(ang / 2)
H1 = (r * math.sin(half), 0.0, r * math.cos(half))
H2 = (r * math.sin(half) * math.cos(2 * math.pi / 3),
      r * math.sin(half) * math.sin(2 * math.pi / 3), r * math.cos(half))
H3 = (r * math.sin(half) * math.cos(4 * math.pi / 3),
      r * math.sin(half) * math.sin(4 * math.pi / 3), r * math.cos(half))
nh3 = [['N', 0.0, 0.0, 0.0], ['H', *H1], ['H', *H2], ['H', *H3]]
o3_base = [['O', -0.00000042, 0.01374723, 0.00000000],
           ['O', 1.06178981, 0.65009697, 0.00000000],
           ['O', -1.06178939, 0.65009672, 0.00000000]]
inputs = dict(
    nh3=dict(atoms=nh3,
             source='standard non-planar C3v guess, r(N-H)=1.012 Ang, '
                    'angle(HNH)=106.7 deg (standard molecular geometry values; '
                    'non-planarity = pyramidal C3v)'),
    o3=dict(atoms=o3_base,
            source='models/00_baseline/o3/optimized_gas.xyz (historical '
                   'wb97X-D/def2-TZVP era baseline geometry; historical source '
                   'label retained; NOT the current-batch method)'),
    atom_order=dict(nh3=['N', 'H', 'H', 'H'], o3=['O', 'O', 'O']),
    charge=0, spin=0, coords_unit='Angstrom')
os.makedirs('/mnt/e/dft-service/ozone_mnb_uv_simulation/inputs/nh3o3_phase2', exist_ok=True)
for sid, d in (('nh3', inputs['nh3']), ('o3', inputs['o3'])):
    lines = [str(len(d['atoms']))] + \
        ['%s %.10f %.10f %.10f' % tuple(a) for a in d['atoms']]
    open('/mnt/e/dft-service/ozone_mnb_uv_simulation/inputs/nh3o3_phase2/%s_gas.xyz' % sid,
         'w').write('\n'.join(lines) + '\n')

# 5) build mols and verify atom order/charge/spin/coords round-trip
def build(atoms):
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, x, y, z in atoms)
    return gto.M(atom=atom, basis='def2-TZVP', charge=0, spin=0, verbose=0)

mol_nh3 = build(nh3); mol_o3 = build(o3_base)
rt = {}
for sid, m0, atoms in (('nh3', mol_nh3, nh3), ('o3', mol_o3, o3_base)):
    rt[sid] = dict(
        elements=list(m0.elements), charge=m0.charge, spin=m0.spin,
        natm=m0.natm,
        coords_bohr_first=m0.atom_coords(unit='Bohr')[0].tolist(),
        coords_match_input=bool(np.abs(np.asarray(
            [a[1:] for a in atoms], float) - m0.atom_coords(unit='Angstrom')).max() < 1e-10))
checks['inputs_roundtrip'] = rt
# C3v NH3: non-planarity = H atoms off the N plane (z != 0 here); the earlier
# check compared two H z-coords (always equal by symmetry) -- fixed:
checks['non_planarity_nh3'] = bool(abs(nh3[1][3]) > 1e-3)
checks['inputs_source_labels'] = dict(nh3=inputs['nh3']['source'],
                                      o3=inputs['o3']['source'])

json.dump(checks, open('/mnt/e/dft-service/ozone_mnb_uv_simulation/run_artifacts/02_nh3o3_reference/precheck_noSCF.json', 'w'),
          indent=2)
print(json.dumps(checks, indent=2)[:1600])
