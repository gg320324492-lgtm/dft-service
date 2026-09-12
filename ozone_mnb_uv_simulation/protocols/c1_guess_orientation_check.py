import json, hashlib
import numpy as np
ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'

# JOB-026 monomer references (for rigid-placement verification)
mr = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/gas_monomer_references/monomer_results.json'))
v5 = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/gas_monomer_references/nh3_v5_result.json'))
NH3_MONO = np.asarray(v5['v3_final']['coords_angstrom'], float)      # N,H,H,H
O3_MONO = np.asarray(mr['o3']['endpoint_recheck']['coords_angstrom'], float)

guess_lines = open(ROOT + '/inputs/nh3o3_phase2/c1_guess_S13fig1.xyz').read().splitlines()
G = np.asarray([[float(x) for x in l.split()[1:4]] for l in guess_lines[1:]], float)
GS = [l.split()[0] for l in guess_lines[1:]]

def dmat(P):
    return np.linalg.norm(P[:, None, :] - P[None, :, :], axis=-1)

def rigid_equivalence(A, B, tol=1e-8):
    """A and B represent the same rigid body if their internal distance
    matrices match exactly (translation+rotation invariant)."""
    return bool(np.abs(dmat(A) - dmat(B)).max() < tol)

# fragment internal distance-matrix verification
nh3_g, o3_g = G[:4], G[4:]
# O3 fragment rows in the guess = [5O, 7O, 6O]; the build script derived
# them from O3_MONO rows [terms[0], terms[1], central].  Recover that
# permutation from connectivity (central = smallest max-distance row).
d_mono = dmat(O3_MONO)
np.fill_diagonal(d_mono, 1e9)
central_m = int(np.argmin(d_mono.max(axis=1)))
perm = [i for i in range(3) if i != central_m] + [central_m]
O3_MONO_P = O3_MONO[perm]          # rows: 5O, 7O, 6O (same as guess)
checks = dict(
    nh3_internal_matrix_matches_monomer=rigid_equivalence(nh3_g, NH3_MONO),
    o3_internal_matrix_matches_monomer=rigid_equivalence(o3_g, O3_MONO_P),
    nh3_internal_maxdev=float(np.abs(dmat(nh3_g) - dmat(NH3_MONO)).max()),
    o3_internal_maxdev=float(np.abs(dmat(o3_g) - dmat(O3_MONO_P)).max()),
    o3_perm_applied=[int(i) for i in perm],
    atom_order=GS)

# orientation quantities
N = G[0]
Hc = G[1:4].mean(axis=0)                    # H-triangle centroid
lp_dir = N - Hc                             # H-centroid -> N = lone-pair direction
lp_dir /= np.linalg.norm(lp_dir)
o6 = G[6]
term_mid = G[4:6].mean(axis=0)              # terminal-O midpoint (5O,7O)
to_mid = term_mid - N; to_mid /= np.linalg.norm(to_mid)
to_c = o6 - N; to_c /= np.linalg.norm(to_c)
plane_n = np.cross(G[1] - N, G[2] - N)
plane_n /= np.linalg.norm(plane_n)
ang = lambda a, b: float(np.degrees(np.arccos(np.clip(a @ b, -1, 1))))
orient = dict(
    lone_pair_dir_vs_terminal_midpoint_deg=ang(lp_dir, to_mid),
    lone_pair_dir_vs_central_O_deg=ang(lp_dir, to_c),
    NH3_plane_normal_vs_terminal_midpoint_deg=ang(plane_n, to_mid),
    NH3_plane_normal_vs_central_O_deg=ang(plane_n, to_c),
    H_centroid=(N - Hc).tolist(),
    N_to_central_O=float(np.linalg.norm(o6 - N)),
    N_to_terminals=[float(np.linalg.norm(G[4] - N)),
                    float(np.linalg.norm(G[5] - N))],
    min_interfragment=float(min(np.linalg.norm(G[i] - G[j])
                                for i in range(4) for j in (4, 5, 6))))
# geometry-only claim check: lone-pair direction (H-centroid -> N) vs terminal midpoint
orient['lone_pair_toward_terminals_geometry'] = bool(
    orient['lone_pair_dir_vs_terminal_midpoint_deg'] < 30.0)
checks['orientation'] = orient

json.dump(checks, open(ROOT + '/run_artifacts/02_nh3o3_reference/c1_adduct/guess_orientation_check.json', 'w'),
          indent=2)
print(json.dumps(checks, indent=2))
