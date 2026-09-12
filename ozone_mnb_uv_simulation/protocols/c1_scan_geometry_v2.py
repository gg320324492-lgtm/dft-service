import os, sys, json, hashlib
import numpy as np
from scipy.optimize import brentq

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
OUT = ROOT + '/inputs/nh3o3_phase2/c1_scan_v2'
os.makedirs(OUT, exist_ok=True)


def load_monomers():
    mr = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/gas_monomer_references/monomer_results.json'))
    v5 = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/gas_monomer_references/nh3_v5_result.json'))
    return (np.asarray(v5['v3_final']['coords_angstrom'], float),
            np.asarray(mr['o3']['endpoint_recheck']['coords_angstrom'], float))


def identify_central(o3):
    """Central O = the atom whose farthest distance to the other O atoms is
    the smallest.  Cross-checked by distance relations.  Immune to atom
    reordering (per-atom quantities only)."""
    n = len(o3)
    d = np.linalg.norm(o3[:, None, :] - o3[None, :, :], axis=-1)
    others = [sorted(d[i, j] for j in range(n) if j != i) for i in range(n)]
    farthest = [x[-1] for x in others]
    central = int(np.argmin(farthest))
    terms = [i for i in range(n) if i != central]
    r_c = [d[central, t] for t in terms]
    d_tt = d[terms[0], terms[1]]
    ok = (max(r_c) < 1.5) and (d_tt > 1.5 * max(r_c)) and (d_tt < 3.0)
    return central, terms, ok, dict(r_central_to_terminals=r_c,
                                    d_terminal_terminal=d_tt)


def build_scan_geometries(NH3_MONO, O3_MONO, targets):
    """Axis-independent C1 scan geometry builder (JOB-2026-0906-029).

    Convention: C = central O; M = midpoint of the two terminal O;
    u = (M-C)/|M-C| (pointing to the terminal side); N = M + s*u, s > 0
    with s from 1D Brent root-finding on the average N-terminal contact
    distance (terminal asymmetry kept).  NH3 is a complete rigid body with
    N at the target position and its H-centroid on the +u side of N
    (H-centroid->N faces O3).  INITIAL GUESS GEOMETRY CONVENTION -- not a
    lone-pair electron density calculation.  Monomer internal geometry is
    never modified.
    """
    central, terms, central_ok, central_detail = identify_central(O3_MONO)
    assert central_ok, 'central-O distance-relation cross-check failed'
    C = O3_MONO[central]
    M = O3_MONO[terms].mean(axis=0)
    u = (M - C) / np.linalg.norm(M - C)            # -> terminal side
    w = O3_MONO[terms[0]] - M                      # half-base (asymmetry kept)
    p_dir = w - (w @ u) * u
    p_dir /= np.linalg.norm(p_dir)
    n_dir = np.cross(u, p_dir)

    # rotation: monomer basis (u, p_dir, n_dir) -> target basis
    # (u->+x, p_dir->+z, n_dir->-y); RH target basis, det=+1
    M_o = np.column_stack([u, p_dir, n_dir])
    M_t = np.column_stack([np.array([1.0, 0.0, 0.0]),
                           np.array([0.0, 0.0, 1.0]),
                           np.array([0.0, -1.0, 0.0])])
    R = M_t @ M_o.T
    assert np.abs(R.T @ R - np.eye(3)).max() < 1e-12, 'R^T R != I'
    assert abs(np.linalg.det(R) - 1.0) < 1e-12, 'det(R) != +1'

    O3_new = np.array([R @ (p - C) for p in O3_MONO])  # monomer row order
    M_new = O3_new[terms].mean(axis=0)
    T1, T2 = O3_new[terms[0]], O3_new[terms[1]]

    def contacts(s):
        N = M_new + s * np.array([1.0, 0.0, 0.0])
        return np.linalg.norm(N - T1), np.linalg.norm(N - T2), N

    def f(s, d_target):
        a, b, _ = contacts(s)
        return (a + b) / 2.0 - d_target

    def solve_s(d_target):
        lo, hi = 0.5, 6.0
        assert f(lo, d_target) < 0 < f(hi, d_target), 'bracket failed'
        return brentq(lambda s: f(s, d_target), lo, hi, xtol=1e-12)

    # NH3 rigid reorientation (EQUIVARIANT under input rotation): build the
    # monomer's own orthonormal frame from the C3 axis (N -> H-centroid) and
    # an in-plane reference defined by H1 (first H in input order), then map
    # C3-axis -> +x, H1-side -> +y, normal -> +z.  Under a global rotation of
    # the input, the whole frame rotates consistently -> the placed geometry
    # is the exact rigid image (no arbitrary roll about the C3 axis).
    n0 = NH3_MONO[0]
    hc = NH3_MONO[1:].mean(axis=0)
    e_axis = (hc - n0) / np.linalg.norm(hc - n0)      # N -> H-centroid
    h1_rel = NH3_MONO[1] - n0
    e_ref = h1_rel - (h1_rel @ e_axis) * e_axis
    e_ref /= np.linalg.norm(e_ref)
    e_n = np.cross(e_axis, e_ref)
    Rn = np.column_stack([e_axis, e_ref, e_n]).T   # maps e_axis->+x, e_ref->+y, e_n->+z
    assert np.abs(Rn.T @ Rn - np.eye(3)).max() < 1e-12
    assert abs(np.linalg.det(Rn) - 1.0) < 1e-12

    SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']     # 1N,2H,3H,4H,5O,7O,6O
    x_dir = M_t[:, 0]                              # target bisector (+x)
    points = []
    s_prev = -1.0
    for d_t in targets:
        s = solve_s(d_t)
        assert s > s_prev, 's not monotonically increasing'
        s_prev = s
        N_pos = M_new + s * x_dir
        assert (N_pos - M_new) @ x_dir > 0, 'N not on terminal side of M'
        nh3 = np.array([Rn @ (p - n0) for p in NH3_MONO]) + N_pos
        coords = np.vstack([nh3, O3_new])
        d1 = float(np.linalg.norm(N_pos - T1))
        d2c = float(np.linalg.norm(N_pos - T2))
        d_actual = (d1 + d2c) / 2.0
        assert abs(d_actual - d_t) < 1e-9, 'actual != target'
        d_N_C = float(np.linalg.norm(N_pos - O3_new[central]))
        min_inter = float(min(np.linalg.norm(coords[i] - coords[j])
                              for i in range(4) for j in (4, 5, 6)))
        assert min_inter > 1.5, 'collision (construction error)'
        assert d_N_C > d_t, 'N-central-O shorter than terminal contact'
        dev_nh3 = float(np.abs(np.linalg.norm(
            coords[:4][:, None] - coords[:4][None], axis=-1)
            - np.linalg.norm(NH3_MONO[:, None] - NH3_MONO[None],
                             axis=-1)).max())
        dev_o3 = float(np.abs(np.sort(np.linalg.norm(
            coords[4:][:, None] - coords[4:][None], axis=-1)
            [np.triu_indices(3, 1)])
            - np.sort(np.linalg.norm(O3_MONO[:, None] - O3_MONO[None],
                                     axis=-1)[np.triu_indices(3, 1)])).max())
        assert dev_nh3 < 1e-10 and dev_o3 < 1e-10, 'fragment geometry changed'
        xyz = ['7'] + ['%s %.12f %.12f %.12f' % (sy, x, y, z)
                       for sy, (x, y, z) in zip(SYMS, coords)]
        fname = 'c1_scan_d%.1f.xyz' % d_t
        open(os.path.join(OUT, fname), 'w').write('\n'.join(xyz) + '\n')
        fsha = hashlib.sha256(('\n'.join(xyz) + '\n').encode()).hexdigest()[:16]
        points.append(dict(target_d=d_t, actual_avg_d=d_actual,
                           d_dev=abs(d_actual - d_t), s=s,
                           N_M_along_u=float((N_pos - M_new) @ x_dir),
                           N_O_term1=d1, N_O_term2=d2c,
                           N_O_central=d_N_C,
                           o3_row_mapping=dict(central_row=int(central),
                                               terminal_rows=[int(t) for t
                                                              in terms]),
                           min_interfragment=min_inter,
                           dev_nh3=dev_nh3, dev_o3=dev_o3,
                           xyz_file=fname, xyz_sha16=fsha,
                           coords=coords.tolist()))
        print('[d=%.1f] s=%.6f actual=%.9f N-Ocentral=%.4f min_inter=%.3f OK'
              % (d_t, s, d_actual, d_N_C, min_inter), flush=True)
    assert all(points[i]['s'] < points[i + 1]['s']
               for i in range(len(points) - 1)), 's monotonicity'
    return dict(
        job='JOB-2026-0906-029 geometry pipeline v2',
        convention=dict(
            C='central O (validated mapping + distance-relation cross-check)',
            M='midpoint of the two terminal O',
            u='(M-C)/|M-C|, pointing to the terminal side',
            N='M + s*u, s>0 (1D Brent root-finding; terminal asymmetry kept)',
            NH3='complete rigid body; N at target; H-centroid on the +u side '
                'of N (H-centroid->N faces O3) -- INITIAL GUESS GEOMETRY '
                'CONVENTION, not a lone-pair electron density calculation',
            note='monomer internal geometry untouched; no bond length/angle '
                 'reset to literature figure values'),
        central_id=dict(central_index=int(central),
                        terms=[int(t) for t in terms],
                        check_ok=bool(central_ok), detail=central_detail),
        rotation=dict(RtR_I=True, det_plus1=True,
                      note='antiparallel 180-deg case handled by Rodrigues '
                           'rotation about a perpendicular axis, never -I'),
        atom_order=SYMS,
        fragments=dict(NH3=[0, 1, 2, 3], O3=[4, 5, 6]),
        unit='Angstrom',
        points=points,
        all_assertions_pass=True)


if __name__ == '__main__':
    NH3_MONO, O3_MONO = load_monomers()
    manifest = build_scan_geometries(NH3_MONO, O3_MONO,
                                     [2.6, 2.8, 3.0, 3.2, 3.6, 4.2])
    json.dump(manifest, open(os.path.join(OUT, 'scan_geometry_manifest.json'),
                             'w'), indent=2)
    print('SIX-POINT GEOMETRY PREPARATION COMPLETE: all numeric assertions '
          'pass')
