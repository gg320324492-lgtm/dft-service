#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Inspect fragment distances in the 047 inputs (read-only)."""
import json
import numpy as np

ROOT = '/mnt/e/dft-service/ozone_mnb_uv_simulation'
m = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/'
                        'c1_bidir_opt047/input_manifest.json'))
g = m['geometries']['p10']['fragment_checks']
print('p10 N-H:', [round(x, 3) for x in g['N_H_bohr']])
print('p10 O-O:', [round(x, 3) for x in g['O_O_bohr']])
c = json.load(open(ROOT + '/run_artifacts/02_nh3o3_reference/'
                        'c1_negmode_scan045/eval_eval_centre_centre.json'))
R = np.asarray(c['coords_actual_angstrom']) / 0.52917721092
dm = np.linalg.norm(R[:, None, :] - R[None, :, :], axis=-1)
print('centre O-O:', [round(dm[4, 5], 3), round(dm[5, 6], 3),
                      round(dm[4, 6], 3)])
print('centre N-H:', [round(dm[0, i], 3) for i in (1, 2, 3)])
print('centre full distance matrix (Bohr):')
print(np.round(dm, 2))
