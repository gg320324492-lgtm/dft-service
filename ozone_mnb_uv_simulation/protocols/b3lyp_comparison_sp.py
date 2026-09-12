import os, sys, json, hashlib, time
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
from sp_ledger import SPLedger
from pyscf import gto, dft

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
BATCH = ROOT + '/run_artifacts/02_nh3o3_reference/b3lyp_comparison'
os.makedirs(BATCH, exist_ok=True)
LEDGER = BATCH + '/budget_b3lyp.json'
CAPS = dict(total=3, scf=3)
GRID_LEVEL = 8
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
XC = 'b3lyp'
BASIS = '6-311++g(3df,3pd)'


class ScanIntegrityError(RuntimeError):
    pass


class EvaluationFailed(RuntimeError):
    pass


ledger = SPLedger(LEDGER, cap=CAPS['total'])

# ---- source: JOB-031 d=3.0 computed coords (NOT rebuilt) ----
r31 = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/'
                            'c1_scan_formal/exec_results_formal.json'))
pt = next(v for v in r31.values() if isinstance(v, dict)
          and v.get('target_d') == 3.0)
coords = np.asarray(pt['coords_computed'], float)
ref = pt['ref_energies']
print('source: JOB-031 d=3.0 computed coords; ref source: %s' % ref['source'],
      flush=True)

# ---- fragment integrity checks ----
dmat = lambda P: np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)
nh3, o3 = coords[:4], coords[4:]
# vs JOB-026 monomer references (rigid placement preserved)
v5 = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/'
                           'gas_monomer_references/nh3_v5_result.json'))
mr = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/'
                           'gas_monomer_references/monomer_results.json'))
NH3_MONO = np.asarray(v5['v3_final']['coords_angstrom'], float)
O3_MONO = np.asarray(mr['o3']['endpoint_recheck']['coords_angstrom'], float)
# JOB-031 scan records keep the O3 rows in the MONOMER's own row order
# (verified: identity permutation gives dev = 0.0)
dev_o3 = float(np.abs(dmat(o3) - dmat(O3_MONO)).max())
dev_nh3 = float(np.abs(dmat(nh3) - dmat(NH3_MONO)).max())
assert dev_nh3 < 1e-10 and dev_o3 < 1e-10, 'fragment geometry not preserved'
print('fragment internal geometry preserved: dev_nh3=%.2e dev_o3=%.2e'
      % (dev_nh3, dev_o3), flush=True)

# ---- SP helper (guarded) ----
def sp(syms, C, tag, category):
    if ledger.attempts_used() >= CAPS['total']:
        raise RuntimeError('TOTAL budget exhausted (%d)' % CAPS['total'])
    atom = "; ".join("%s %.10f %.10f %.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(syms, C))
    mol = gto.M(atom=atom, basis=BASIS, charge=0, spin=0, verbose=0,
                max_memory=4000)
    mf = dft.RKS(mol)
    mf.xc = XC
    mf.grids.level = GRID_LEVEL
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-9
    idx = ledger.start_attempt(json.dumps([tag], sort_keys=True),
                               dict(tag=tag, category=category,
                                    xc=XC, basis=BASIS,
                                    grid_level=GRID_LEVEL))
    t0 = time.time()
    try:
        mf.kernel()
        e = float(mf.e_tot)
        conv = bool(mf.converged)
        fin = bool(np.isfinite(e))
        if not conv:
            ledger.mark_error(idx, 'SCF not converged')
            raise EvaluationFailed('SCF not converged: %s' % tag)
        if not fin:
            ledger.mark_error(idx, 'non-finite')
            raise EvaluationFailed('non-finite: %s' % tag)
        ledger.update(idx, status='done', e_total=e,
                      seconds=round(time.time() - t0, 1))
        print('[%s] E=%.9f Eh (%.1fs)' % (tag, e, time.time() - t0), flush=True)
        return e, cfg_readback(mf)
    except Exception as ex:
        if 'ledger' in str(type(ex)):
            raise
        print('HARD STOP: %s' % ex, flush=True)
        raise


def cfg_readback(mf):
    return dict(xc=mf.xc, basis=BASIS, grid_level=int(mf.grids.level),
                scf_tol=[float(mf.conv_tol), float(mf.conv_tol_grad)],
                d2='NOT included (plain B3LYP, no project -D2)',
                solvent='none (gas phase)')


# ---- three SCF evaluations ----
E_cplx, cfg_c = sp(SYMS, coords, 'complex_d3.0', 'scf')
E_nh3, cfg_n = sp(SYMS[:4], nh3, 'NH3_fragment', 'scf')
E_o3, cfg_o = sp(SYMS[4:], o3, 'O3_fragment', 'scf')

dE = E_cplx - E_nh3 - E_o3
out = dict(
    job='JOB-2026-0906-032 B3LYP literature-level single-point comparison',
    geometry_source='JOB-031 exec_results_formal.json :: d=3.0 computed '
                    'coords (not rebuilt); fragments cut directly, internal '
                    'geometry and orientation preserved, no ghost atoms',
    fragment_checks=dict(dev_nh3_vs_JOB026=dev_nh3, dev_o3_vs_JOB026=dev_o3),
    method_implementation=dict(
        xc=XC, libxc_id='402 (HYB_B3LYP); VWN variant vs Gaussian03 B3LYP '
        '(VWN3) NOT confirmable identical -> classified as '
        '"near-literature-level implementation"',
        basis=BASIS,
        grid_level=GRID_LEVEL,
        grid_note='L8 is denser than the Gaussian03 default grid used by S13 '
                  '(numerical setting difference, recorded)',
        scf_tol=[1e-12, 1e-9],
        d2='NOT included'),
    energies=dict(E_complex=E_cplx, E_NH3_fragment=E_nh3,
                  E_O3_fragment=E_o3),
    dE_B3LYP_nonCP_hartree=dE,
    dE_B3LYP_nonCP_kcalmol=dE * 627.5094740631,
    label='fixed-fragment-geometry non-CP electronic interaction energy at '
          'the JOB-031 d=3.0 geometry',
    budget=dict(attempts=ledger.attempts_used(), cap=CAPS['total']),
    cfg_complex=cfg_c)
json.dump(out, open(BATCH + '/b3lyp_comparison_results.json', 'w'), indent=2)
print('dE_B3LYP_nonCP = %.6f Eh = %.3f kcal/mol' % (dE, out['dE_B3LYP_nonCP_kcalmol']),
      flush=True)
print('B3LYP COMPARISON DONE', flush=True)
