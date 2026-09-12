import os, sys, json, hashlib
import numpy as np
HERE = '/mnt/e/dft-service/ozone_mnb_uv_simulation/protocols'
sys.path.insert(0, HERE)
from c1_scan_exec_v2 import write_standard_xyz, read_xyz_primary, \
    read_xyz_independent
from c1_scan_geometry_v2 import load_monomers, build_scan_geometries

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
OLD_DIR = ROOT + '/inputs/nh3o3_phase2/c1_scan_v2'
NEW_DIR = ROOT + '/inputs/nh3o3_phase2/c1_scan_v2_std'
os.makedirs(NEW_DIR, exist_ok=True)

TARGETS = [2.6, 2.8, 3.0, 3.2, 3.6, 4.2]
NH3, O3 = load_monomers()
m = build_scan_geometries(NH3, O3, TARGETS)
SYMS = m['atom_order']

points = []
for pt in m['points']:
    d_t = pt['target_d']
    coords = np.asarray(pt['coords'], float)
    # physical coords must be unchanged vs the old-format file
    old_p = os.path.join(OLD_DIR, 'c1_scan_d%.1f.xyz' % d_t)
    old_lines = open(old_p).read().splitlines()
    old_C = np.asarray([[float(x) for x in l.split()[1:4]]
                        for l in old_lines[1:1 + int(old_lines[0])]], float)
    assert np.abs(coords - old_C).max() < 1e-12, 'physical coords changed!'
    comment = ('C1-type NH3-O3 scan point d=%.1f A (N-terminal average); '
               'JOB-2026-0906-029/030 rigid placement of JOB-026 monomers; '
               'initial guess convention, not author coordinates' % d_t)
    new_p = os.path.join(NEW_DIR, 'c1_scan_std_d%.1f.xyz' % d_t)
    write_standard_xyz(new_p, SYMS, coords, comment)
    # independent reader cross-check on the NEW file
    s1, C1 = read_xyz_primary(new_p)
    s2, C2 = read_xyz_independent(new_p)
    assert np.abs(C1 - C2).max() < 1e-12, 'reader disagreement'
    assert np.abs(C1 - coords).max() < 1e-12
    fsha = hashlib.sha256(open(new_p, 'rb').read()).hexdigest()[:16]
    points.append(dict(target_d=d_t, actual_avg_d=pt['actual_avg_d'],
                       s=pt['s'], N_O_term1=pt['N_O_term1'],
                       N_O_term2=pt['N_O_term2'],
                       N_O_central=pt['N_O_central'],
                       min_interfragment=pt['min_interfragment'],
                       o3_row_mapping=pt['o3_row_mapping'],
                       xyz_file='c1_scan_std_d%.1f.xyz' % d_t,
                       xyz_sha16=fsha,
                       old_format_file='c1_scan_d%.1f.xyz (retained; '
                                       'missing comment line)' % d_t,
                       coords=coords.tolist()))

manifest = dict(
    job='JOB-2026-0906-030 standard XYZ regeneration',
    xyz_format='standard: line1=atom count, line2=comment, then coordinates '
               '(Angstrom)',
    version_relation=dict(
        old_files='c1_scan_v2/c1_scan_d*.xyz (retained, no comment line)',
        new_files='c1_scan_v2_std/c1_scan_std_d*.xyz',
        physical_coords='identical to old files (verified < 1e-12 A)',
        hash_basis='new hashes refer to the standard-format files'),
    atom_order=SYMS, fragments=dict(NH3=[0, 1, 2, 3], O3=[4, 5, 6]),
    unit='Angstrom', points=points, all_assertions_pass=True)
json.dump(manifest, open(os.path.join(NEW_DIR, 'scan_geometry_manifest_std.json'),
                         'w'), indent=2)
print('STANDARD XYZ x%d regenerated; physical coords unchanged; manifest_std '
      'written' % len(points))
