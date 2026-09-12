#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""JOB-2026-0906-047 step A (offline, no SCF):
  * centre = 045 actual centre record; q = 045 saved Cartesian direction
    (unit Euclidean norm), hash verified against charter 59599f2a2c0be260;
  * generate R+ = R0 + 0.10 q and R- = R0 - 0.10 q (0.10 Bohr is the norm
    of the full 21-component displacement vector, NOT per-atom amplitude);
  * verify units, displacement norms, element order, interatomic
    distances and fragment composition (NH3 = N+3H, O3 = 3O);
  * draw a two-view displacement figure (PNG) for collision / build-error
    inspection;
  * save the input manifest.
"""
import os, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
S45 = ROOT + '/run_artifacts/02_nh3o3_reference/c1_negmode_scan045'
OUT = ROOT + '/run_artifacts/02_nh3o3_reference/c1_bidir_opt047'
os.makedirs(OUT, exist_ok=True)
SYMS = ['N', 'H', 'H', 'H', 'O', 'O', 'O']
ANG_PER_BOHR = 0.52917721092
BOHR_PER_A = 1.0 / ANG_PER_BOHR
T_STEP = 0.10                      # Bohr, norm of the 21-vector displacement
CHARTER_Q_HASH = '59599f2a2c0be260'
RADII = {'N': 0.65, 'H': 0.35, 'O': 0.60}   # Angstrom, schematic
COLORS = {'N': '#3050c8', 'H': '#e0e0e0', 'O': '#e03030'}


def sha_arr(a):
    return hashlib.sha256(np.asarray(a, float).tobytes()).hexdigest()[:16]


def pair_distances(R):
    n = len(R)
    dm = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
    return dm


def fragment_checks(R):
    dm = pair_distances(R)
    nh = [dm[0, i] for i in (1, 2, 3)]            # N-H
    oo = sorted([dm[4, 5], dm[5, 6], dm[4, 6]])   # O-O (2 bonds + 1 diagonal)
    inter = [dm[i, j] for i in range(4) for j in range(4, 7)]
    dm_nodiag = dm.copy()
    np.fill_diagonal(dm_nodiag, 9e9)
    return dict(N_H_bohr=[float(x) for x in nh],
                O_O_sorted_bohr=[float(x) for x in oo],
                fragment_min_inter_bohr=float(min(inter)),
                global_min_pair_bohr=float(dm_nodiag.min()),
                NH3_intact=bool(max(nh) < 2.5),
                # bent triatomic ozone: two short bonds + one 1-3 diagonal
                O3_intact=bool(oo[0] < 3.0 and oo[1] < 3.0 and oo[2] < 5.5),
                no_collision=bool(dm_nodiag.min() > 1.5),
                fragments_separated=bool(min(inter) > 4.0))


# ---------- centre + direction ----------
man45 = json.load(open(S45 + '/input_manifest.json'))
q = np.asarray(man45['direction']['q_vector'], float)
assert abs(np.linalg.norm(q) - 1.0) < 1e-12
q_hash = hashlib.sha256(q.tobytes()).hexdigest()[:16]
if q_hash != CHARTER_Q_HASH:
    raise RuntimeError('HARD STOP: q hash mismatch (%s)' % q_hash)

c45 = json.load(open(S45 + '/eval_eval_centre_centre.json'))
R0 = np.asarray(c45['coords_actual_angstrom'], float) / ANG_PER_BOHR

# ---------- two displacement geometries ----------
geoms = {}
for tag, sgn in (('p10', +1.0), ('m10', -1.0)):
    R = R0 + sgn * T_STEP * q.reshape(7, 3)
    d_actual = (R - R0).reshape(-1)
    fc = fragment_checks(R)
    geoms[tag] = dict(
        t_Bohr=sgn * T_STEP,
        coords_bohr=[list(map(float, r)) for r in R],
        coords_angstrom=[list(map(float, r * ANG_PER_BOHR)) for r in R],
        coords_bohr_sha=sha_arr(R),
        displacement_norm_Bohr=float(np.linalg.norm(d_actual)),
        displacement_norm_expected=float(T_STEP),
        norm_check=bool(abs(np.linalg.norm(d_actual) - T_STEP) < 1e-12),
        per_atom_displacement_Bohr=[float(np.linalg.norm(
            d_actual[3 * i:3 * i + 3])) for i in range(7)],
        max_atom_displacement_Bohr=float(max(
            np.linalg.norm(d_actual[3 * i:3 * i + 3]) for i in range(7))),
        element_order=SYMS,
        unit='Bohr (primary) / Angstrom (derived, x0.52917721092)',
        fragment_checks=fc,
        all_checks_pass=bool(fc['NH3_intact'] and fc['O3_intact']
                             and fc['no_collision']
                             and fc['fragments_separated']))

# ---------- two-view displacement figure ----------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
views = [('xy view', 0, 1), ('xz view', 0, 2)]
for ax, (title, ia, ib) in zip(axes, views):
    for tag, base_col, marker in (('m10', 'tab:blue', 'o'),
                                  ('p10', 'tab:red', 's')):
        R = np.asarray(geoms[tag]['coords_bohr'])
        for i, s in enumerate(SYMS):
            ax.scatter(R[i, ia] * ANG_PER_BOHR, R[i, ib] * ANG_PER_BOHR,
                       s=260 * RADII[s] ** 2, c=COLORS[s], alpha=0.75,
                       marker=marker, edgecolors='k', zorder=3,
                       label=('%s (%s)' % (tag, {'p10': 'R+', 'm10': 'R-'}[
                           tag])) if i == 0 else None)
    R0a = R0 * ANG_PER_BOHR
    for i, s in enumerate(SYMS):
        ax.scatter(R0a[i, ia], R0a[i, ib], s=260 * RADII[s] ** 2,
                   c='none', marker='o', edgecolors='gray',
                   linewidths=1.8, zorder=2,
                   label='R0 (centre, outline)' if i == 0 else None)
    # intramolecular bonds of R+ for readability (N-H bonds + O-O chain;
    # the ozone 1-3 diagonal is NOT a bond and is not drawn)
    Rp = np.asarray(geoms['p10']['coords_bohr']) * ANG_PER_BOHR
    for a, b in ((0, 1), (0, 2), (0, 3), (4, 5), (5, 6)):
        ax.plot([Rp[a, ia], Rp[b, ia]], [Rp[a, ib], Rp[b, ib]],
                color='gray', lw=0.9, zorder=1)
    ax.set_xlabel('%s (Angstrom)' % 'xyz'[ia])
    ax.set_ylabel('%s (Angstrom)' % 'xyz'[ib])
    ax.set_title(title + '  |  R0 vs R(+0.10q) vs R(-0.10q)')
    ax.legend(fontsize=8, loc='best')
    ax.set_aspect('equal')
    ax.grid(alpha=0.25)
fig.suptitle('JOB-047: bidirectional displacement inputs along the 045 '
             'negative-mode direction (norm 0.10 Bohr)')
fig.tight_layout()
fig.savefig(OUT + '/displacement_two_views.png', dpi=150)

# ---------- manifest ----------
out = dict(job='JOB-2026-0906-047 step A: centre + bidirectional inputs',
           centre_source=dict(
               file='c1_negmode_scan045/eval_eval_centre_centre.json',
               e_total=c45['e_total'], grad_max=c45['grad_max'],
               coords_sha=c45['coords_actual_sha']),
           direction=dict(q_hash=q_hash, charter_hash=CHARTER_Q_HASH,
                          match=True,
                          norm=float(np.linalg.norm(q)),
                          note='Cartesian unit vector, NOT re-derived from '
                               'current gradients, mass NOT re-applied'),
           t_step_Bohr=T_STEP,
           displacement_definition='R(+/-) = R0 +/- 0.10 * q, 0.10 Bohr is '
                                   'the norm of the full 21-component '
                                   'displacement vector',
           geometries=geoms,
           figure='displacement_two_views.png')
json.dump(out, open(OUT + '/input_manifest.json', 'w'), indent=2)

for tag in ('p10', 'm10'):
    g = geoms[tag]
    fc = g['fragment_checks']
    print('%s: |disp|=%.6f (expect %.2f) max_atom=%.4f Bohr  '
          'NH3=%s O3=%s min_pair=%.3f min_inter=%.3f -> checks %s'
          % (tag, g['displacement_norm_Bohr'],
             g['displacement_norm_expected'], g['max_atom_displacement_Bohr'],
             fc['NH3_intact'], fc['O3_intact'], fc['global_min_pair_bohr'],
             fc['fragment_min_inter_bohr'], g['all_checks_pass']))
print('centre: E=%.9f (045 record)  q hash %s OK'
      % (c45['e_total'], q_hash))
print('figure ->', OUT + '/displacement_two_views.png')
print('saved ->', OUT + '/input_manifest.json')
