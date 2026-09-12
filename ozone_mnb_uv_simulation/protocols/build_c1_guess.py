import json, math
import numpy as np
import os
ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
mr = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/gas_monomer_references/monomer_results.json'))
v5 = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/gas_monomer_references/nh3_v5_result.json'))

o3_c = np.asarray(mr['o3']['endpoint_recheck']['coords_angstrom'], float)
nh3_c = np.asarray(v5['v3_final']['coords_angstrom'], float)

# ---- O3 fragment: central atom = the O bonded to both others ----
d = np.linalg.norm(o3_c[:, None, :] - o3_c[None, :, :], axis=-1)
np.fill_diagonal(d, 1e9)
central = int(np.argmin(d.max(axis=1)))          # smallest max-dist to others
terms = [i for i in range(3) if i != central]
d_pair = float(np.linalg.norm(o3_c[terms[0]] - o3_c[terms[1]]))
r_o3 = [float(np.linalg.norm(o3_c[central] - o3_c[t])) for t in terms]

# reorient: central at origin, bisector along -x, terminals symmetric in xz-plane
v1 = o3_c[terms[0]] - o3_c[central]; v1 /= np.linalg.norm(v1)
v2 = o3_c[terms[1]] - o3_c[central]; v2 /= np.linalg.norm(v2)
bis = (v1 + v2); bis /= np.linalg.norm(bis)
if bis[0] > 0: bis = -bis                          # bisector along -x
perp = np.cross(bis, np.array([0.0, 1.0, 0.0])); perp /= np.linalg.norm(perp)
v1 = o3_c[terms[0]] - o3_c[central]; v1 /= np.linalg.norm(v1)
v2 = o3_c[terms[1]] - o3_c[central]; v2 /= np.linalg.norm(v2)
bis = (v1 + v2); bis /= np.linalg.norm(bis)
perp = (v1 - v2); perp /= np.linalg.norm(perp)
norm_v = np.cross(bis, perp)
# JOB-028 correction: PURE rigid placement via proper orthonormal-basis
# rotation (no projection squash, no bond-length/angle reset): map the
# monomer frame (bis, perp, norm_v) onto (-x, +z, +y).
b_t = np.array([-1.0, 0.0, 0.0]); p_t = np.array([0.0, 0.0, 1.0])
n_t = np.cross(b_t, p_t)
M_t = np.column_stack([b_t, p_t, n_t])
M_o = np.column_stack([bis, perp, norm_v])
R = M_t @ M_o.T
o3_new = np.zeros((3, 3))
for j, t in enumerate(terms):
    o3_new[t] = R @ (o3_c[t] - o3_c[central])
o3_new[central] = [0.0, 0.0, 0.0]
# atom order in O3 fragment: figure 5O(top,-z), 7O(bottom,+z), 6O(central)
o3_order = [terms[0], terms[1], central]           # [5O, 7O, 6O]
o3_final = o3_new[o3_order]                        # rows: 5O, 7O, 6O

# ---- N position on bisector at average contact distance ----
r_contact = (2.967 + 2.963) / 2.0
# terminals symmetric in z after the rotation; solve |N - 5O| = r_contact:
xN = o3_final[0, 0] + math.sqrt(max(r_contact ** 2 - o3_final[0, 2] ** 2, 0.0))
N_pos = np.array([xN, 0.0, 0.0])

# ---- NH3 fragment: orient C3 axis (N -> H centroid) along +x, translate to N_pos ----
h_idx = [i for i in range(4) if i != 0]
n0 = nh3_c[0]
hc = nh3_c[h_idx].mean(axis=0)
axis = hc - n0; axis /= np.linalg.norm(axis)       # current C3 axis (N->H centroid)
target = np.array([1.0, 0.0, 0.0])
v = np.cross(axis, target); s = np.linalg.norm(v); c = axis @ target
if s < 1e-12:
    R = np.eye(3) if c > 0 else -np.eye(3)
else:
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + vx + vx @ vx * ((1 - c) / s ** 2)
nh3_new = np.array([R @ (p - n0) for p in nh3_c]) + N_pos

# ---- assemble in S13 numbering: 1N, 2H, 3H, 4H, 5O, 7O, 6O ----
order_syms = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
coords = np.vstack([nh3_new, o3_final])
out = []
for s, p in zip(order_syms, coords):
    out.append([s] + [float(x) for x in p])
os.makedirs(ROOT + '/inputs/nh3o3_phase2', exist_ok=True)
lines = ['7'] + ['%s %.12f %.12f %.12f' % tuple(a) for a in out]
open(ROOT + '/inputs/nh3o3_phase2/c1_guess_S13fig1.xyz', 'w').write('\n'.join(lines) + '\n')

# ---- sanity distances ----
C = coords
def dist(i, j): return float(np.linalg.norm(C[i] - C[j]))
d5N, d7N = dist(4, 0), dist(5, 0)   # 5O-N, 7O-N  (rows 4=5O, 5=7O, 6=6O)
d56 = dist(4, 6); d76 = dist(5, 6); d57 = dist(4, 5)
nh = [dist(0, j) for j in (1, 2, 3)]
min_inter = min(min(dist(4, j) for j in range(4)),
                min(dist(5, j) for j in range(4)),
                min(dist(6, j) for j in range(4)))
info = dict(
    guess_label='初猜：结合 S13 Figure 1 图示与 JOB-026 项目单体几何的刚体放置（非作者原始坐标）',
    fig_values=dict(oO_1_252_x2='5O-6O, 6O-7O', angle_6O_deg=117.7,
                    NH=1.012, contact_5O_1N=2.967, contact_7O_1N=2.963,
                    dashed='non-covalent contacts; C1 is a CONTACT complex, '
                           'not a new-bond structure'),
    guess_choice=dict(contact_avg=r_contact,
                      note='2.967/2.963 slight asymmetry averaged to '
                           'symmetric bisector placement in the guess'),
    distances=dict(O5_N=d5N, O7_N=d7N, O5_O6=d56, O7_O6=d76, O5_O7=d57,
                   NH3_bonds=nh, min_interfragment=min_inter),
    atom_order='S13 numbering: 1N, 2H, 3H, 4H, 5O, 7O, 6O',
    fragments=dict(NH3=[0, 1, 2, 3], O3=[4, 5, 6]),
    n_atoms=7, charge=0, electrons=34, overall='singlet, RKS start')
json.dump(info, open(ROOT + '/inputs/nh3o3_phase2/c1_guess_info.json', 'w'),
          indent=2, ensure_ascii=False)
print(json.dumps(info['distances'], indent=2))
print('O3 rows (5O,7O,6O):', np.round(o3_final, 4).tolist())
print('NH3 rows:', np.round(nh3_new, 4).tolist())
