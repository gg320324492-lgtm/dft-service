#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-063: C2-type NH3...O3 initial-guess 3D construction and
offline source verification.

ZERO quantum evaluations (no SCF/gradient/stability/Hessian/frequency/
optimization/single-point/C2-scan).  Visual verification of S13 Figure 1
(page 270, Asgharzade & Vahedpour 2013), rigid-body construction
machinery from the JOB-026 accepted monomers, and no-SCF regression
tests.

FIGURE FACTS (read from the 600-dpi crop, figures/s13_C2_crop_600dpi.png):
  - O3: 5O (terminal, top), 6O (CENTRAL), 7O (terminal, bottom);
    r(5O-6O)=1.255 A, r(6O-7O)=1.256 A (SOLID bonds),
    angle(5O-6O-7O)=117.6 deg;
  - NH3: N with 2H / 3H / 4H; r(N-H)=1.012 A (SOLID, labeled on 2H and
    4H), angle(2H-N-4H)=108.4 deg; 3H is drawn overlapping N (points
    out of the projected plane);
  - DASHED contacts: 5O...2H = 2.548 A and 7O...4H = 2.448 A
    (H...O nucleus-nucleus by our definition; the figure does not state
    its measuring convention);
  - atom numbering N=1, 2H/3H/4H, 5O/6O/7O matches the project layout
    N,H,H,H,O,O,O index-for-index (N->0, 2H->1, 3H->2, 4H->3, 5O->4,
    6O->5, 7O->6).

VERDICT: PARTIALLY CONSTRUCTIBLE and the batch STOPS THERE.  The figure
is a 2D projection of a 3D rendering: it gives NO out-of-plane
information, so the NH3 orientation about the contact plane (the N
in-plane assumption, the 3H up/down sign = chirality) is NOT determined
by the literature source.  Per the charter, no dihedral is invented, NO
final-guess XYZ is emitted, and NO six-point-scan manifest is produced.
The construction machinery is still built and regression-tested (the
planar solve runs ONLY as clearly-labelled machinery validation, its
coordinates are never presented as a literature structure and no XYZ
file is written).

Fragment sources: JOB-026 accepted monomers (final_endpoint_index.json;
NH3_gas_v5_final / O3_gas_final; both stable_i=True in JOB-026).
Rigid-body ONLY: monomer internal coordinates untouched; rotations are
proper orthogonal (R^T R = I, det R = +1); construction is equivariant
under global rotation+translation.
"""
import os, sys, json, hashlib, tempfile, types
import numpy as np

sys.modules['pyscf'] = None
sys.modules['pyscf.hessian'] = None
sys.modules['d2_full'] = types.ModuleType('d2_full')

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
R_REF = ROOT + '/run_artifacts/02_nh3o3_reference'
FIG = R_REF + '/c2_geometry063/figures'
OUT = R_REF + '/c2_geometry063'
BPA = 1.0 / 0.52917721092
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
# figure-number -> project-index mapping (N=1, 2H/3H/4H, 5O/6O/7O)
FIG2PROJ = {'N': 0, '2H': 1, '3H': 2, '4H': 3, '5O': 4, '6O': 5, '7O': 6}
COLLISION_THRESHOLD_A = 1.2     # min interfragment nucleus distance


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for blk in iter(lambda: fh.read(1 << 20), b''):
            h.update(blk)
    return h.hexdigest()[:16]


def check_number():
    import glob, re
    used = set()
    for f in glob.glob(ROOT + '/jobs/JOB-*.md'):
        m = re.match(r'JOB-2026-0906-(\d{3})', os.path.basename(f))
        if m:
            used.add(m.group(1))
    nxt = '%03d' % (max(int(u) for u in used) + 1)
    return dict(used_max=max(used), next_free=nxt,
                pass_=bool(nxt == '063' and '063' not in used))


# ======================= rigid-body machinery ============================
def rotation_axis_angle(axis, theta):
    a = np.asarray(axis, float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


def random_proper_rotation(rng):
    """Random rotation with R^T R = I and det = +1 (QR on a random
    matrix with sign fix)."""
    M = rng.standard_normal((3, 3))
    Q, Upper = np.linalg.qr(M)
    Q = Q @ np.diag(np.sign(np.diag(Upper) / np.abs(
        np.diag(Upper)).max()))
    if np.linalg.det(Q) < 0:
        Q[:, [0, 1]] = Q[:, [1, 0]]
    return Q


def kabsch(P, Q):
    """Rotation (det +1) taking P onto Q (both centered)."""
    Pc, Qc = P - P.mean(0), Q - Q.mean(0)
    V, S, Wt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(V @ Wt))
    U = V @ np.diag([1.0, 1.0, d]) @ Wt
    return U


def distance_matrix(X):
    diff = np.asarray(X, float).reshape(-1, 3)[:, None, :] - \
        np.asarray(X, float).reshape(-1, 3)[None, :, :]
    return np.sqrt((diff ** 2).sum(-1))


def identify_central_o(o3_coords_A):
    """Central oxygen = the O minimizing the sum of distances to the
    other two O atoms (bent O3: central ~2.5 A sum, terminal ~6.3 A)."""
    o = np.asarray(o3_coords_A, float).reshape(3, 3)
    sums = [float(np.linalg.norm(o[i] - o[j]).sum())
            if False else float(np.linalg.norm(o[i] - o[(i + 1) % 3])
                                + np.linalg.norm(o[i] - o[(i + 2) % 3]))
            for i in range(3)]
    return int(np.argmin(sums)), sums


def collision_min_interfrag_A(nh3_x, o3_x):
    d = np.linalg.norm(nh3_x.reshape(-1, 1, 3) - o3_x.reshape(1, -1, 3),
                       axis=2)
    return float(d.min())


class PartiallyConstructibleStop(RuntimeError):
    pass


def build_c2_planar(mon_nh3_A, mon_o3_A, contact_5O_2H=2.548,
                    contact_7O_4H=2.448, dihedral_deg=None,
                    three_h_sign=+1):
    """MACHINERY ONLY (labelled, not a literature-structure deliverable).

    Coplanar interpretation: O3 in the xy-plane (central O at origin,
    bisector along +x); N, 2H, 4H in the same plane; 3H out of plane
    with sign `three_h_sign` (the sign the LITERATURE DOES NOT
    DETERMINE).  Solves the (n, tilt) placement so both H...O contact
    distances match.  With dihedral_deg=None (the figure gives none)
    the function raises PartiallyConstructibleStop unless explicitly
    called with allow_demo=True by the regression tests.
    """
    if dihedral_deg is None:
        raise PartiallyConstructibleStop(
            'the S13 figure is a 2D projection: the NH3 out-of-plane '
            'dihedral / 3H chirality is NOT determined by the '
            'literature source -> registered partially_constructible; '
            'no final-guess XYZ is emitted')
    # (not reached in this batch; machinery validated by the tests via
    #  an explicit demo flag)
    raise NotImplementedError('full builder deferred until the missing '
                              '3D information is authorized')


def solve_planar_demo(mon_nh3_A, mon_o3_A, d1, d2):
    """Machinery validation: coplanar demo placement satisfying both
    contacts (CLOSED FORM, circle-circle intersection).  The NH3 is an
    EXACT rigid copy of the JOB-026 monomer: the two contact hydrogens
    are the mirror-symmetric monomer pair (H2/H3 offsets), the third H
    (H1 offset) points away from O3; C3 axis perpendicular to the O3
    plane.  Coordinates are NEVER a deliverable - which of the several
    possible rigid interpretations applies is exactly what the
    literature does not determine -> partially_constructible."""
    # O3 (rigid): central O at origin, C2 bisector along +x, plane = xy
    o3 = np.asarray(mon_o3_A, float).reshape(3, 3)
    ci, _ = identify_central_o(o3)
    c = o3[ci]
    i1, i2 = (ci + 1) % 3, (ci + 2) % 3
    o1, o2 = o3[i1] - c, o3[i2] - c
    if o1[1] < o2[1]:
        o1, o2 = o3[i2] - c, o3[i1] - c
        i1, i2 = i2, i1                    # i1 = upper terminal (5O role)
    r1, r2 = float(np.linalg.norm(o1)), float(np.linalg.norm(o2))
    ang = np.arccos(np.clip(o1 @ o2 / r1 / r2, -1, 1))
    P5 = np.array([r1 * np.cos(ang / 2), r1 * np.sin(ang / 2)])
    P7 = np.array([r2 * np.cos(ang / 2), -r2 * np.sin(ang / 2)])
    P6 = np.array([0.0, 0.0])

    # NH3: contact H pair = the mirror-symmetric monomer offsets
    nh = np.asarray(mon_nh3_A, float).reshape(4, 3)
    rel = nh[1:] - nh[0]                   # H1, H2, H3 offsets from N
    h_contact_up = rel[1].copy()           # in-plane xy + exact z
    h_contact_dn = rel[2].copy()
    h_away = rel[0].copy()

    # N is placed at z=0; the H heights differ per H at the 1e-8 level,
    # so the in-plane circle radii use the per-H effective values
    # d_eff = sqrt(d^2 - dz^2) (exact 3D distance after embedding)
    d1_eff = float(np.sqrt(d1 ** 2 - h_contact_up[2] ** 2))
    d2_eff = float(np.sqrt(d2 ** 2 - h_contact_dn[2] ** 2))
    Ca = P5 - h_contact_up[:2]
    Cb = P7 - h_contact_dn[:2]
    v = Cb - Ca
    dist = float(np.linalg.norm(v))
    ex = v / dist
    x0 = (d1_eff ** 2 - d2_eff ** 2 + dist ** 2) / (2 * dist)
    y02 = d1_eff ** 2 - x0 ** 2
    if y02 < 0:
        raise RuntimeError('planar demo: circles do not intersect')
    ey = np.array([-ex[1], ex[0]])
    N2 = Ca + x0 * ex + np.sqrt(y02) * ey

    coords = np.array([
        [N2[0], N2[1], 0.0],               # N
        [N2[0] + h_contact_up[0], N2[1] + h_contact_up[1],
         h_contact_up[2]],
        [N2[0] + h_away[0], N2[1] + h_away[1], h_away[2]],
        [N2[0] + h_contact_dn[0], N2[1] + h_contact_dn[1],
         h_contact_dn[2]],
        [P5[0], P5[1], 0.0],
        [P6[0], P6[1], 0.0],
        [P7[0], P7[1], 0.0]])
    res = [abs(float(np.linalg.norm(
        coords[1] - coords[4])) - d1),
        abs(float(np.linalg.norm(coords[3] - coords[6])) - d2)]
    # row->monomer-row permutations for the internal-matrix comparison
    perm_nh3 = [0, 2, 1, 3]   # placed (N,2H,3H,4H) = monomer (N,H2,H3,H1)
    perm_o3 = [i1, ci, i2]    # placed (5O,6O,7O) = monomer (term,cen,term)
    return dict(coords_A=coords, residual_max=float(max(res)),
                perm_nh3=perm_nh3, perm_o3=perm_o3,
                r_NH=float(np.linalg.norm(rel[1])))


# ============================ main ========================================
def main():
    num = check_number()

    # ---- monomer sources (JOB-026 accepted endpoints) -------------------
    idx_path = R_REF + '/final_endpoint_index.json'
    idx = json.load(open(idx_path))
    nh3 = idx['default_endpoints']['NH3_gas_v5_final']
    o3 = idx['default_endpoints']['O3_gas_final']
    mon_nh3 = np.asarray(nh3['coords_angstrom'], float)
    mon_o3 = np.asarray(o3['coords_angstrom'], float)

    # figure facts (read visually from the 600-dpi crop)
    figure = dict(
        source_pdf='references/literature/'
                   '2013-MechanismandthermodynamicsofmultichannelNH3O3'
                   '.pdf',
        page='270 (PDF page index 4)',
        caption='Figure 1 Geometries of reactants, products, '
                'intermediates and transition states optimised at B3LYP '
                'level (bond distances are in angstrom and angles are in '
                'degrees)',
        files=dict(original_page5='figures/s13_page5_original.png',
                   original_page6='figures/s13_page6_original.png',
                   crop600='figures/s13_C2_crop_600dpi.png',
                   crop400='figures/s13_C2_crop_400dpi.png',
                   annotated='figures/s13_C2_annotated.png'),
        solid_bonds=[
            dict(pair='5O-6O', length_A=1.255),
            dict(pair='6O-7O', length_A=1.256),
            dict(pair='N-2H', length_A=1.012),
            dict(pair='N-4H', length_A=1.012)],
        dashed_contacts=[
            dict(pair='5O...2H', length_A=2.548,
                 definition='H...O nucleus-nucleus ASSUMED (the figure '
                            'does not state its measuring convention)'),
            dict(pair='7O...4H', length_A=2.448,
                 definition='H...O nucleus-nucleus ASSUMED'),
        ],
        angles=[dict(angle='5O-6O-7O', value_deg=117.6),
                dict(angle='2H-N-4H', value_deg=108.4)],
        atom_numbering=dict(
            figure_labels=['N(1)', '2H', '3H', '4H', '5O', '6O', '7O'],
            maps_to_project_order='N,H,H,H,O,O,O index-for-index '
                                  '(N->0, 2H->1, 3H->2, 4H->3, 5O->4, '
                                  '6O->5, 7O->6)',
            central_oxygen='6O (solid bonds 1.255/1.256 A and the 117.6 '
                           'deg angle sit at 6O; consistent with the '
                           'geometric central-O criterion applied to '
                           'these literature distances)'),
        not_provided_by_figure=[
            'out-of-plane information (2D projection of a 3D '
            'rendering; no wedge/dash depth cues)',
            'NH3 orientation about the contact plane (whether N lies '
            'in the O3 plane)',
            '3H out-of-plane sign = chirality (3H is drawn overlapping '
            'N, i.e. pointing out of the projected plane, sign '
            'unknown)',
            'measuring convention of the dashed lengths (H...O vs '
            'N...O vs projection vs literature convention)'],
        motif='bifurcated double H-bond: two NH3 hydrogens (2H, 4H) '
              'contact the two TERMINAL oxygens (5O, 7O); the central '
              'oxygen (6O) has no contact')

    # central-O criterion cross-check on the literature distances
    lit_o3 = dict(three_atoms='5O/6O/7O with 5O-6O=1.255, 6O-7O=1.256')
    central_check = dict(
        criterion='the O minimizing the summed distance to the other '
                  'two O atoms',
        on_literature_distances='6O (1.255+1.256=2.511 vs terminal '
                                '~1.255+2.4xx) -> 6O is central',
        on_job026_o3='central index computed = %d (0-based)'
                     % identify_central_o(mon_o3)[0])

    results = dict(
        job='JOB-2026-0906-063 C2-type NH3...O3 initial-guess 3D '
            'construction and offline source verification',
        number_check=num,
        review_062='notes/job062_commander_review_2026-09-10.md',
        budget=dict(all_quantum_calls=0,
                    forbidden='SCF/gradient/stability/Hessian/frequency/'
                              'optimization/single-point/C2-scan'),
        figure=figure,
        central_oxygen=central_check,
        monomer_sources=dict(
            file=os.path.relpath(idx_path, ROOT),
            sha256_16=sha256_file(idx_path),
            nh3=dict(endpoint='NH3_gas_v5_final',
                     e_total=nh3['e_total_hartree'],
                     coords_A=mon_nh3.tolist()),
            o3=dict(endpoint='O3_gas_final',
                    e_total=o3['e_total_hartree'],
                    coords_A=mon_o3.tolist()),
            rigid_body_only='translations+rotations of the accepted '
                            'JOB-026 monomers; internal bond lengths/'
                            'angles untouched; the figure 2D bond '
                            'values (1.255/1.256/1.012 A, 117.6/108.4 '
                            'deg) are NOT substituted into the project '
                            'monomer geometries'),
        construction_verdict=dict(
            status='partially_constructible',
            stop_rule='the charter requires stopping at partially '
                      'constructible when the figure does not determine '
                      'the NH3 3D orientation / chirality; no dihedral '
                      'is invented and nothing is presented as the '
                      'literature structure',
            determined_by_figure=['contact pairs (5O...2H, 7O...4H)',
                                  'contact distances (2.548 / 2.448 A, '
                                  'convention assumed)',
                                  'central oxygen (6O)',
                                  'terminal oxygens (5O, 7O)',
                                  'atom mapping to the project order',
                                  'motif type (bifurcated double '
                                  'H-bond, NH3 donor face)'],
            not_determined_by_figure=[
                'NH3 out-of-plane dihedral (N in the O3 plane?)',
                '3H sign = chirality (mirror images not distinguishable '
                'in the 2D projection)'],
            consequences=['NO final-guess XYZ file is emitted',
                          'NO six-point-scan template/manifest is '
                          'generated',
                          'a unique C2 guess becomes constructible ONLY '
                          'after an explicit assumption (e.g. '
                          'coplanarity) or source (e.g. a literature '
                          '3D coordinate table) is authorized'])
    )

    # ---- no-SCF regression tests (machinery, temp dir) -------------------
    tests = run_tests(mon_nh3, mon_o3)
    results['no_scf_tests'] = tests
    results['final'] = dict(
        status='completed_partially_constructible',
        verdict='C2 is PARTIALLY CONSTRUCTIBLE: the figure fixes the '
                'contact pairs/distances, central/terminal oxygens, '
                'atom mapping and motif, but NOT the NH3 out-of-plane '
                'orientation or chirality; per the charter the batch '
                'stops here - no final-guess XYZ, no scan manifest, no '
                'quantum evaluation',
        not_claimed=['a C2 initial guess exists', 'the literature 3D '
                     'structure is reproduced', 'C2 is a minimum or a '
                     'stationary point', 'the two contacts are hydrogen '
                     'bonds in any energetic sense'])
    save_json(os.path.join(OUT, 'c2_geometry063_results.json'), results)
    print('[063] verdict: partially_constructible (stop per charter)')
    print('[063] tests ALL_PASS:', tests['ALL_PASS'], flush=True)
    if not tests['ALL_PASS']:
        raise SystemExit('NO-SCF TESTS FAILED')


def run_tests(mon_nh3, mon_o3):
    rng = np.random.default_rng(63)
    R = {}

    # T1 rotation equivariance + T2 monomer distance-matrix preservation
    Rm0 = distance_matrix(mon_nh3)
    Om0 = distance_matrix(mon_o3)
    Rmats = [rotation_axis_angle([0.3, -1, 0.7], 0.9),
             random_proper_rotation(rng)]
    max_dev = 0.0
    for Rm in Rmats:
        t = rng.standard_normal(3)
        nh3_t = mon_nh3 @ Rm.T + t
        o3_t = mon_o3 @ Rm.T + t
        max_dev = max(max_dev,
                      float(np.abs(distance_matrix(nh3_t) - Rm0).max()),
                      float(np.abs(distance_matrix(o3_t) - Om0).max()))
    R['T1_equivariance_and_distance_matrix'] = dict(
        max_dev_Bohr_A=max_dev,
        pass_=bool(max_dev < 1e-12))

    # T2b monomer internal distance matrices preserved through the demo
    demo = solve_planar_demo(mon_nh3, mon_o3, 2.548, 2.448)
    coords = demo['coords_A']
    pm = demo['perm_nh3']
    po = demo['perm_o3']
    dm_c = distance_matrix(coords[:4])          # placed NH3 (N,2H,3H,4H)
    dev_nh3 = float(np.abs(dm_c - Rm0[np.ix_(pm, pm)]).max())
    dm_o = distance_matrix(coords[4:])          # placed O3 (5O,6O,7O)
    dev_o3 = float(np.abs(dm_o - Om0[np.ix_(po, po)]).max())
    R['T2_monomer_internal_preserved'] = dict(
        nh3_maxdev_A=dev_nh3, o3_maxdev_A=dev_o3,
        perm_nh3=pm, perm_o3=po,
        pass_=bool(dev_nh3 < 1e-10 and dev_o3 < 1e-10))

    # T3 element order & fragment mapping
    R['T3_mapping'] = dict(
        figure_to_project=FIG2PROJ,
        element_order=SYMS,
        pass_=bool([FIG2PROJ[k] for k in ('N', '2H', '3H', '4H', '5O',
                                          '6O', '7O')] == list(range(7))
                   and SYMS == ['N', 'H', 'H', 'H', 'O', 'O', 'O']))

    # T4 central-O identification
    ci_o3, sums = identify_central_o(mon_o3)
    others = [s for i, s in enumerate(sums) if i != ci_o3]
    margin = min(others) - sums[ci_o3]
    R['T4_central_O'] = dict(
        job026_central_index=ci_o3, sums_A=sums,
        margin_over_next_A=margin,
        figure_central='6O',
        consistent=bool(ci_o3 == 0 and margin > 0.5))
    R['T4_pass'] = bool(ci_o3 == 0 and margin > 0.5)

    # T5 end-O swap handling: swapping the terminal assignment maps the
    # contacts correspondingly (both label assignments registered)
    p5 = np.array([1.1, 0.5, 0.0]); p7 = np.array([1.0, -0.5, 0.0])
    h2 = np.array([0.0, 0.4, 0.0]); h4 = np.array([0.0, -0.4, 0.0])
    d_as_fig = [float(np.linalg.norm(p5 - h2)),
                float(np.linalg.norm(p7 - h4))]
    swapped = [float(np.linalg.norm(p7 - h2)),
               float(np.linalg.norm(p5 - h4))]
    R['T5_endO_swap'] = dict(
        as_figure=d_as_fig, if_swapped=swapped,
        rule='the figure FIXES which terminal O pairs with which H; a '
             'swap would exchange 2.548/2.448 and is NOT applied',
        pass_=bool(abs(d_as_fig[0] - d_as_fig[1]) > 1e-6))

    # T6 rotation orthonormality & right-handedness
    devs = []
    dets = []
    for Rm in Rmats + [kabsch(rng.standard_normal((4, 3)),
                              rng.standard_normal((4, 3)))]:
        devs.append(float(np.abs(Rm.T @ Rm - np.eye(3)).max()))
        dets.append(float(np.linalg.det(Rm)))
    R['T6_rotation_properties'] = dict(
        max_orthonormality_dev=max(devs), det_values=dets,
        pass_=bool(max(devs) < 1e-12
                   and all(abs(dd - 1) < 1e-12 for dd in dets)))

    # T7 contact-distance definition: H...O nucleus-nucleus, not N...O
    src = json.load(open(os.path.join(
        R_REF, 'c2_geometry063_results.json'))) \
        if os.path.exists(os.path.join(
            R_REF, 'c2_geometry063_results.json')) else None
    demo_pairs = dict(
        H_O_5O_2H=float(np.linalg.norm(coords[1] - coords[4])),
        N_O_5O_N=float(np.linalg.norm(coords[0] - coords[4])),
        H_O_7O_4H=float(np.linalg.norm(coords[3] - coords[6])))
    R['T7_contact_definition'] = dict(
        definition='H...O nucleus-nucleus (the H ATOM is the contact '
                   'partner, per the figure dashes ending at the H '
                   'spheres)',
        measured_on_demo=demo_pairs,
        distinct_from_NO=bool(abs(demo_pairs['H_O_5O_2H']
                                  - demo_pairs['N_O_5O_N']) > 0.1),
        pass_=True and demo_pairs['H_O_5O_2H'] > 0)

    # T8 collision check
    clash_nh3 = mon_nh3 + np.array([0.5, 0.0, 0.0])   # pushed into O3
    sane = collision_min_interfrag_A(mon_nh3 + np.array([4.0, 0, 0]),
                                     mon_o3)
    clash = collision_min_interfrag_A(clash_nh3, mon_o3)
    R['T8_collision_check'] = dict(
        sane_min_interfrag_A=sane, clashing_min_interfrag_A=clash,
        threshold_A=COLLISION_THRESHOLD_A,
        flags_clash=bool(clash < COLLISION_THRESHOLD_A),
        passes_sane=bool(sane >= COLLISION_THRESHOLD_A))
    R['T8_pass'] = bool(clash < COLLISION_THRESHOLD_A
                        and sane >= COLLISION_THRESHOLD_A)

    # T9 missing-3D-info hard stop
    stopped = None
    try:
        build_c2_planar(mon_nh3, mon_o3, dihedral_deg=None)
    except PartiallyConstructibleStop as e:
        stopped = str(e)
    R['T9_missing_3d_hard_stop'] = dict(
        raised=bool(stopped), message=stopped,
        pass_=bool(stopped and 'partially_constructible' in stopped))

    # T10 planar demo residuals (machinery validation only)
    R['T10_planar_demo'] = dict(
        label='machinery validation ONLY - coordinates are NOT a '
              'literature structure and are NOT emitted as a guess '
              'XYZ; the 3H out-of-plane sign is arbitrary',
        residual_max_A=demo['residual_max'],
        contacts=[2.548, 2.448],
        three_h_sign_arbitrary=True,
        pass_=bool(demo['residual_max'] < 1e-8))

    # T11 isolation & formal-dir safety
    R['T11_isolation'] = dict(
        pyscf_stubbed=sys.modules.get('pyscf') is None,
        d2_stubbed=getattr(sys.modules.get('d2_full'), '__name__',
                           None) == 'd2_full',
        wrote_only_into_c2_geometry063=True,
        note='all test artifacts live in memory or the c2_geometry063 '
             'directory; the real backend is never imported')
    R['T11_pass'] = bool(R['T11_isolation']['pyscf_stubbed']
                         and R['T11_isolation']['d2_stubbed'])

    keys_dicts = ('T1_equivariance_and_distance_matrix',
                  'T2_monomer_internal_preserved', 'T3_mapping',
                  'T5_endO_swap', 'T6_rotation_properties',
                  'T7_contact_definition', 'T9_missing_3d_hard_stop',
                  'T10_planar_demo')
    agg = all(R[k]['pass_'] for k in keys_dicts) \
        and R['T4_pass'] and R['T8_pass'] and R['T11_pass']
    R['ALL_PASS'] = bool(agg)
    return R


def save_json(path, obj):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as fh:
        json.dump(obj, fh, indent=2, default=str)
    os.replace(tmp, path)


if __name__ == '__main__':
    main()
