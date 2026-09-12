#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Acceptance-revision step #2 (JOB-2026-0905-009): re-deduplicate all gas and
SMD structures with the REVISED equivalence/H-bond logic.

Inputs (read-only): gas_freq/*.json (14 originals), mode_follow/*.json (14
mode-follow endpoints), smd_freq/*.json (SMD re-optimisations).
No quantum chemistry is re-run here -- pure geometry analysis.

Outputs (run_artifacts/01_pure_water_o3_h2o/acceptance_revision/):
  revision_rededup.json  -- full record:
    * per-structure revised topology + n_imaginary + provenance;
    * gas candidate set: clustered minima (n_imag==0) with pairwise RMSD,
      old->new correspondence, eliminated (saddles / duplicates / window);
    * same for SMD structures.
Terminology per the revision brief: the outcome is a "待核验候选集"
(pending-verification candidate set) and the lowest-energy structure among
the searched candidates ("已搜索候选中的最低能结构"), NOT final distinct
minima / a global minimum.
"""
import os
import sys
import json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import conformer_utils as cu

ROOT = os.path.dirname(HERE)
ART = os.path.join(ROOT, 'run_artifacts', '01_pure_water_o3_h2o')
REV = os.path.join(ART, 'acceptance_revision')
os.makedirs(REV, exist_ok=True)

RMSD_TOL = 0.10
ETOL_KCAL = 0.10
WINDOW_KCAL = 3.0


def load_gas_pool():
    """14 original conformers + 14 mode-follow endpoints."""
    pool = {}
    for tid in ['c%02d' % i for i in range(1, 15)]:
        d = json.load(open(os.path.join(ART, 'gas_freq', '%s.json' % tid)))
        pool[tid] = dict(coords=d['geometry_angstrom'],
                         e_total=d['e_total_hartree'],
                         n_imag=int(d['n_imaginary']),
                         source='original trial conformer')
    for fn in sorted(os.listdir(os.path.join(ART, 'mode_follow'))):
        if not fn.endswith('.json'):
            continue
        d = json.load(open(os.path.join(ART, 'mode_follow', fn)))
        pool[d['label']] = dict(coords=d['geometry_angstrom'],
                                e_total=d['e_total_hartree'],
                                n_imag=int(d['n_imaginary']),
                                source='mode-follow endpoint of %s (%s)'
                                       % (d['source'], d['direction']))
    return pool


def load_smd_pool():
    pool = {}
    for fn in sorted(os.listdir(os.path.join(ART, 'smd_freq'))):
        if not fn.endswith('.json'):
            continue
        d = json.load(open(os.path.join(ART, 'smd_freq', fn)))
        if 'e_total_hartree' not in d:
            continue
        pool[d['id']] = dict(coords=d['optimised_coords_angstrom'],
                             e_total=d['e_total_hartree'],
                             n_imag=int(d.get('n_imaginary', -1)),
                             source='SMD re-optimisation of %s' % d['id'])
    return pool


def analyse(pool):
    out = {}
    for tid, r in pool.items():
        coords = np.asarray(r['coords'], dtype=float)
        topo = cu.hbond_topology(coords)
        out[tid] = dict(coords=r['coords'], e_total=r['e_total'],
                        n_imag=r['n_imag'], source=r['source'],
                        topology=topo['description'],
                        topology_signature=[list(topo['hbonds']),
                                            sorted(topo['ow_contacts'])],
                        sig_hash=str(cu.topology_signature(coords)))
    return out


def dedup(info, names):
    """Cluster minima by revised topology signature; within a cluster remove
    duplicates by RMSD + energy; window-filter vs the lowest minimum."""
    minima = [i for i in names if info[i]['n_imag'] == 0]
    saddles = [i for i in names if info[i]['n_imag'] > 0]
    minima.sort(key=lambda i: info[i]['e_total'])
    e_low = info[minima[0]]['e_total']
    for i in names:
        info[i]['rel_kcal'] = (info[i]['e_total'] - e_low) * cu.HARTREE2KCAL

    kept, elim = [], {}
    for sig in sorted(set(info[i]['sig_hash'] for i in minima)):
        group = [i for i in minima if info[i]['sig_hash'] == sig]
        rep = group[0]
        kept.append(rep)
        for other in group[1:]:
            rms = cu.best_rmsd(info[rep]['coords'], info[other]['coords'])
            de = (info[other]['e_total'] - info[rep]['e_total']) * cu.HARTREE2KCAL
            if rms < RMSD_TOL and abs(de) < ETOL_KCAL:
                elim[other] = dict(reason='duplicate of %s (revised RMSD=%.3f A, '
                                          'dE=%.3f kcal/mol, same revised topology)'
                                          % (rep, rms, de), duplicate_of=rep,
                                   rmsd_to_rep=rms)
            else:
                kept.append(other)
    final = []
    for i in sorted(kept, key=lambda i: info[i]['e_total']):
        if info[i]['rel_kcal'] > WINDOW_KCAL:
            elim[i] = dict(reason='energy %.3f kcal/mol above lowest candidate (> %.1f window)'
                                  % (info[i]['rel_kcal'], WINDOW_KCAL), duplicate_of=None,
                           rmsd_to_rep=None)
        else:
            final.append(i)
    return final, elim, saddles


def pairwise_rmsd(info, names):
    m = {}
    for a in names:
        for b in names:
            if a < b:
                m['%s|%s' % (a, b)] = round(cu.best_rmsd(info[a]['coords'],
                                                         info[b]['coords']), 4)
    return m


def main():
    # ---------------- gas phase
    gas = analyse(load_gas_pool())
    gnames = sorted(gas)
    final, elim, saddles = dedup(gas, gnames)
    gas_out = dict(
        n_structures=len(gnames),
        n_minima=len(gnames) - len(saddles), n_saddles=len(saddles),
        saddles=sorted(saddles),
        candidates_pending_verification=final,
        lowest_among_searched_candidates=min(final, key=lambda i: gas[i]['e_total']),
        lowest_energy_hartree=min(gas[i]['e_total'] for i in final),
        eliminated={i: elim[i] for i in elim},
        pairwise_rmsd_angstrom=pairwise_rmsd(gas, final + saddles),
        per_structure={i: {k: v for k, v in gas[i].items() if k != 'coords'}
                       for i in gnames})

    # old->new correspondence: old stage2 kept list vs revised candidate set
    old_kept = json.load(open(os.path.join(ART, 'stage2_minima.json'))).get('kept_minima', [])
    gas_out['old_to_new'] = {i: ('kept as candidate' if i in final else
                                 elim.get(i, {}).get('reason', 'not in revised pool'))
                             for i in old_kept}

    # ---------------- SMD
    smd = analyse(load_smd_pool())
    snames = sorted(smd)
    s_final, s_elim, s_saddles = dedup(smd, snames)
    smd_out = dict(
        n_structures=len(snames),
        n_minima=len(snames) - len(s_saddles), n_saddles=len(s_saddles),
        saddles=sorted(s_saddles),
        candidates_pending_verification=s_final,
        lowest_among_searched_candidates=min(s_final, key=lambda i: smd[i]['e_total']),
        eliminated={i: s_elim[i] for i in s_elim},
        pairwise_rmsd_angstrom=pairwise_rmsd(smd, s_final),
        per_structure={i: {k: v for k, v in smd[i].items() if k != 'coords'}
                       for i in snames})

    out = dict(
        job='JOB-2026-0905-009',
        step='revision_rededup (revised H-bond criterion + full equivalence set)',
        terminology=dict(candidates='待核验候选集 (pending-verification candidate set)',
                         lowest='已搜索候选中的最低能结构 (lowest-energy among searched candidates)'),
        criteria=dict(rmsd_tol_angstrom=RMSD_TOL, energy_tol_kcal=ETOL_KCAL,
                      window_kcal=WINDOW_KCAL,
                      hbond='d(H..O)<=2.5 A and O_w-H..O angle (vectors H->O_w, H->O) >= 120 deg',
                      equivalence='O3 terminal swap x water-H swap'),
        gas_phase=gas_out, smd_phase=smd_out)
    with open(os.path.join(REV, 'revision_rededup.json'), 'w') as fh:
        json.dump(out, fh, indent=2)

    print('GAS: %d structures, %d minima / %d saddles -> %d candidates (pending verification)'
          % (len(gnames), len(gnames) - len(saddles), len(saddles), len(final)))
    print('  candidates:', final)
    print('  lowest among searched:', gas_out['lowest_among_searched_candidates'])
    for i, e in elim.items():
        print('  eliminated %s: %s' % (i, e['reason'][:80]))
    print('SMD: %d structures -> %d candidates' % (len(snames), len(s_final)))
    print('  candidates:', s_final)
    print('saved ->', os.path.join(REV, 'revision_rededup.json'))


if __name__ == '__main__':
    main()
