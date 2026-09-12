#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Baseline driver: H2O / O3 / O2 at wB97X-D/def2-TZVP, gas phase and SMD(H2O).

Runs, for each species:
    gas-phase geometry optimisation + analytic Hessian + thermochemistry
    SMD(water) geometry optimisation + Hessian + thermochemistry

Writes per-species/phase JSON + XYZ into results/00_baseline/.
Runs under WSL2 (Ubuntu 24.04) with PySCF 2.14.0 + pyberny 0.7.0.

NOTE on the functional: PySCF routes xc='wb97xd' to libxc's
XC_HYB_GGA_XC_wB97X_D, which contains the *DFT part only* -- the empirical
Chai-Head-Gordon -D2 dispersion term is NOT included (verified: the
wb97x -> wb97xd energy difference is distance-independent for Kr2 from
6 to 20 A, i.e. no R^-6 tail).  The -D2 term is therefore added explicitly
below (s6 = 1.0, sr6 = 1.1, damping a = 6.0, exponent 12; Grimme-2006 C6
and vdW radii), per Chai & Head-Gordon, PCCP 10, 6615 (2008) and confirmed
against the Q-Chem and Gaussian manuals.  For H2O/O3/O2 it is <= 2e-3
kcal/mol and is reported separately rather than folded into the SCF.
"""
import os, sys, json, time, math
import numpy as np
from pyscf import gto, dft, data, __version__ as PYSCF_VERSION
from pyscf.solvent import smd
from pyscf.hessian import thermo as pyscf_thermo
from pyscf.geomopt.berny_solver import optimize

# ----------------------------------------------------------------- settings
XC = 'wb97xd'                     # libxc XC_HYB_GGA_XC_wB97X_D (DFT part)
BASIS = 'def2-TZVP'
GRID_LEVEL = 5                    # fine Lebedev/Treutler grid (~90k pts for H2O)
SCF_CONV = 1e-11
OPT_CONV = dict(gradientmax=2e-5, gradientrms=1e-5,
                stepmax=1e-4, steprms=6e-5, maxsteps=200)
SOLVENT = 'water'                 # SMD(H2O)
TEMP = 298.15                     # K
PRESS = 101325.0                  # Pa (1 atm)
FD_STEP = 0.01                    # Bohr, numerical-Hessian fallback step

OUT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(OUT, 'results', '00_baseline')
os.makedirs(RES, exist_ok=True)

HARTREE2KCAL = 627.5094740631
HA2JMOL = data.nist.HARTREE2J * data.nist.AVOGADRO
AMU = data.nist.ATOMIC_MASS       # 1 amu in kg (pyscf names it ATOMIC_MASS)

# ------------------------------------------------- Chai-Head-Gordon D2 (-D2)
C6_TABLE = {'H': 0.14, 'C': 1.75, 'N': 1.23, 'O': 0.70, 'F': 0.75,
            'P': 7.84, 'S': 5.57, 'Cl': 5.07, 'Br': 12.47, 'I': 19.81}
RVDW_TABLE = {'H': 1.30, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
              'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98}
CHG_S6, CHG_SR6, CHG_A, CHG_M = 1.0, 1.1, 6.0, 12


def chg_d2(mol):
    """wB97X-D empirical dispersion, returns Hartree."""
    sym = [mol.atom_symbol(i) for i in range(mol.natm)]
    r = mol.atom_coords(unit='Angstrom')
    e = 0.0
    for i in range(mol.natm):
        for j in range(i + 1, mol.natm):
            c6 = math.sqrt(C6_TABLE[sym[i]] * C6_TABLE[sym[j]]) * 1e-54
            R0 = CHG_SR6 * (RVDW_TABLE[sym[i]] + RVDW_TABLE[sym[j]]) * 1e-10
            R = np.linalg.norm(r[i] - r[j]) * 1e-10
            f = 1.0 / (1.0 + CHG_A * (R0 / R) ** CHG_M)
            e += -CHG_S6 * c6 / R ** 6 * f
    return e / HA2JMOL


# --------------------------------------------------------------- structures
def build_xyz(symbols, coords):
    return "\n".join("%-2s %18.10f %18.10f %18.10f" % (s, x, y, z)
                     for s, (x, y, z) in zip(symbols, coords))


def water_geom(r=0.9584, ang=104.45):
    """experimental gas-phase r(O-H), angle(H-O-H)"""
    t = math.radians(ang / 2.0)
    return ['O', 'H', 'H'], [(0, 0, 0), (r * math.sin(t), 0, r * math.cos(t)),
                             (-r * math.sin(t), 0, r * math.cos(t))]


def ozone_geom(r=1.2717, ang=117.79):
    """experimental gas-phase r(O-O), angle(O-O-O); central O first"""
    t = math.radians(ang / 2.0)
    return ['O', 'O', 'O'], [(0, 0, 0), (r * math.sin(t), r * math.cos(t), 0),
                             (-r * math.sin(t), r * math.cos(t), 0)]


def oxygen_geom(r=1.2075):
    """experimental gas-phase r(O=O)"""
    return ['O', 'O'], [(0, 0, 0), (0, 0, r)]


SPECIES = {
    'h2o': dict(name='H2O (water)', charge=0, mult=1, geom=water_geom(),
                sigma=2, note='experimental gas-phase geometry'),
    'o3': dict(name='O3 (ozone)', charge=0, mult=1, geom=ozone_geom(),
               sigma=2, note='experimental gas-phase geometry'),
    'o2': dict(name='O2 (triplet dioxygen)', charge=0, mult=3, geom=oxygen_geom(),
               sigma=2, note='experimental gas-phase bond length'),
}

# ------------------------------------------------------------------ helpers
def make_mf(mol, solvent=None):
    base = dft.UKS(mol) if mol.spin else dft.RKS(mol)
    mf = smd.smd_for_scf(base, solvent_obj=smd.SMD(mol, solvent=solvent)) if solvent else base
    mf.xc = XC
    mf.conv_tol = SCF_CONV
    mf.max_cycle = 200
    mf.grids.level = GRID_LEVEL
    return mf


def xyz_of(mol):
    c = mol.atom_coords(unit='Angstrom')
    return [("%-2s" % mol.atom_symbol(i),
             [round(float(c[i, 0]), 8), round(float(c[i, 1]), 8), round(float(c[i, 2]), 8)])
            for i in range(mol.natm)]


def geom_string(mol):
    return "; ".join("%s %.8f %.8f %.8f" % (s, x, y, z) for s, (x, y, z) in xyz_of(mol))


def num_hess(mol, solvent, h=FD_STEP):
    """central-difference Hessian from analytic gradients (Bohr, Hartree)"""
    c0 = mol.atom_coords(unit='Bohr')
    n = mol.natm
    g = np.zeros((n, n, 3, 3))
    for i in range(n):
        for k in range(3):
            for s in (+1, -1):
                c = c0.copy()
                c[i, k] += s * h
                m = mol.copy()
                m.set_geom_(c, unit='Bohr')
                m.build()
                mf = make_mf(m, solvent=solvent)
                mf.kernel()
                g[i, :, k] += s * mf.nuc_grad_method().kernel().reshape(n, 3) / (2 * h)
    return g


def vib_freqs(mol, h):
    """mass-weighted normal-mode analysis; returns (freq_cm1[3N], modes_cm1)"""
    h = 0.5 * (h + h.transpose(1, 0, 3, 2))
    n = mol.natm
    H = h.transpose(0, 2, 1, 3).reshape(3 * n, 3 * n)
    mw = np.repeat(mol.atom_mass_list(isotope_avg=True), 3) ** -0.5
    w = np.linalg.eigvalsh(H * np.outer(mw, mw))
    # Eh/(amu*bohr^2) -> s^-2
    lam = w * data.nist.HARTREE2J / (AMU * data.nist.BOHR_SI ** 2)
    conv = 1.0 / (2 * np.pi * data.nist.LIGHT_SPEED_SI * 100.0)
    freq = np.sign(lam) * np.sqrt(np.abs(lam)) * conv
    freq = np.sort(freq)
    return freq


def thermochemistry(mol, modes_cm1, e_elec, mult, sigma, T=TEMP, P=PRESS):
    """rigid-rotor / harmonic-oscillator ideal-gas thermochemistry, 1 atm"""
    kB, hh, cc, NA = data.nist.BOLTZMANN, data.nist.PLANCK, data.nist.LIGHT_SPEED_SI, data.nist.AVOGADRO
    R = kB * NA
    mass = mol.atom_mass_list(isotope_avg=True)
    r = mol.atom_coords(unit='Angstrom')
    r -= (mass[:, None] * r).sum(0) / mass.sum()
    I = np.zeros((3, 3))
    for i in range(mol.natm):
        x, y, z = r[i]
        I += mass[i] * np.array([[y * y + z * z, -x * y, -x * z],
                                 [-x * y, x * x + z * z, -y * z],
                                 [-x * z, -y * z, x * x + y * y]])
    I *= AMU * 1e-20                                  # amu*A^2 -> kg*m^2
    w, _ = np.linalg.eigh(I)
    w = np.maximum(w, 1e-60)
    linear = bool(w[0] < 1e-47)
    if linear:
        theta_r = hh ** 2 / (8 * np.pi ** 2 * w[2] * kB)
        q_rot = T / (sigma * theta_r)
        S_rot = R * (1.0 + np.log(q_rot))
        E_rot = R * T
        rot_const = hh / (8 * np.pi ** 2 * cc * 100 * w[2])   # cm^-1
        B = [0.0, rot_const, rot_const]
    else:
        theta_r = hh ** 2 / (8 * np.pi ** 2 * w * kB)
        q_rot = np.sqrt(np.pi) / sigma * T ** 1.5 / np.sqrt(np.prod(theta_r))
        S_rot = R * (1.5 + np.log(q_rot))
        E_rot = 1.5 * R * T
        B = sorted(hh / (8 * np.pi ** 2 * cc * 100 * w))       # cm^-1
    Mt = mass.sum() * AMU
    q_tr = (2 * np.pi * Mt * kB * T / hh ** 2) ** 1.5 * kB * T / P
    S_tr = R * (2.5 + np.log(q_tr))
    E_tr = 1.5 * R * T
    th_v = hh * cc * 100.0 * np.abs(np.asarray(modes_cm1, dtype=float)) / kB
    x = th_v / T
    with np.errstate(divide='ignore', invalid='ignore'):
        S_vib = R * np.sum(x / (np.expm1(x)) - np.log1p(-np.exp(-x)))
        E_vib = R * np.sum(th_v * (0.5 + 1.0 / np.expm1(x)))
    ZPE = 0.5 * R * th_v.sum()
    S_el = R * math.log(mult)
    S_tot = S_tr + S_rot + S_vib + S_el
    E0 = e_elec * HA2JMOL + ZPE
    H298 = e_elec * HA2JMOL + E_tr + E_rot + E_vib + R * T
    G298 = H298 - T * S_tot
    return dict(
        T=T, P=P, sigma=sigma, linear=linear,
        rot_constants_cm1=[round(float(b), 6) for b in B],
        ZPE=E0 - e_elec * HA2JMOL, E_0K=E0, H_298=H298, G_298=G298,
        S_tot=S_tot, S_trans=S_tr, S_rot=S_rot, S_vib=S_vib, S_elec=S_el,
        thermal_H_corr=H298 - e_elec * HA2JMOL,
        thermal_G_corr=G298 - e_elec * HA2JMOL)


# ---------------------------------------------------------------- main loop
def run_one(key, spec, solvent):
    tag = "%s_%s" % (key, 'smd' if solvent else 'gas')
    t_start = time.time()
    rec = dict(species=key, name=spec['name'], phase=('smd_water' if solvent else 'gas'),
               charge=spec['charge'], multiplicity=spec['mult'],
               xc=XC, basis=BASIS, solvent_model=('SMD(%s)' % SOLVENT) if solvent else 'none',
               grid_level=GRID_LEVEL, scf_conv_tol=SCF_CONV, opt_conv=OPT_CONV,
               T=TEMP, P=PRESS, structure_source=spec['note'])

    syms, coords = spec['geom']
    geom0 = build_xyz(syms, coords)
    rec['initial_geometry_angstrom'] = geom0
    mol0 = gto.M(atom=geom0, basis=BASIS, charge=spec['charge'],
                 spin=spec['mult'] - 1, verbose=0)

    # ---- geometry optimisation
    t0 = time.time()
    mf_opt = make_mf(mol0.copy(), solvent=solvent)
    mol_opt = optimize(mf_opt, **OPT_CONV)
    rec['opt_seconds'] = round(time.time() - t0, 1)
    rec['optimised_geometry_angstrom'] = xyz_of(mol_opt)
    rec['geom_string'] = geom_string(mol_opt)

    # ---- final single point + gradient at the optimised geometry
    molf = gto.M(atom=geom_string(mol_opt), basis=BASIS, charge=spec['charge'],
                 spin=spec['mult'] - 1, verbose=0)
    mf = make_mf(molf, solvent=solvent)
    e_scf = float(mf.kernel())
    grad = mf.nuc_grad_method().kernel()
    gmax = float(np.abs(grad).max())
    grms = float(np.sqrt((grad ** 2).mean()))
    rec['scf_converged'] = bool(mf.converged)
    rec['e_dft_hartree'] = e_scf
    rec['max_gradient_hartree_bohr'] = gmax
    rec['rms_gradient_hartree_bohr'] = grms
    rec['optimisation_converged'] = bool(gmax < 5e-5)
    rec['n_grid_points'] = int(mf.grids.weights.size)
    rec['n_basis_functions'] = int(molf.nao_nr())
    if molf.spin:
        ss, sz = mf.spin_square()
        rec['spin_s2'] = float(ss)
    e_d2 = chg_d2(molf)
    rec['e_dispersion_d2_hartree'] = e_d2
    rec['e_total_wb97xd_hartree'] = e_scf + e_d2
    rec['e_dispersion_d2_kcal_mol'] = e_d2 * HARTREE2KCAL

    # ---- Hessian / frequencies
    t0 = time.time()
    hess_method = 'analytic'
    try:
        h = mf.Hessian().kernel()
        asym = float(np.abs(h - h.transpose(1, 0, 3, 2)).max())
        if not np.isfinite(asym) or asym > 1e-4:
            raise RuntimeError('analytic Hessian asymmetric: %.2e' % asym)
    except Exception as exc:
        sys.stderr.write('[%s] analytic Hessian failed (%s); using numerical\n'
                         % (tag, exc))
        h = num_hess(molf, solvent)
        hess_method = 'numerical (central difference of analytic gradients, h=%.3f Bohr)' % FD_STEP
        asym = float(np.abs(h - h.transpose(1, 0, 3, 2)).max())
    rec['hessian_seconds'] = round(time.time() - t0, 1)
    rec['hessian_method'] = hess_method
    rec['hessian_asymmetry'] = asym
    h = 0.5 * (h + h.transpose(1, 0, 3, 2))

    # primary: pyscf's projector (removes translation/rotation exactly)
    freq_raw = vib_freqs(molf, h)                   # all 3N, diagnostic only
    nfreq = 3 * molf.natm - (5 if _is_linear(molf) else 6)
    rec['raw_3n_frequencies_cm1'] = [round(float(f), 2) for f in freq_raw]
    rec['tr_rot_mode_max_abs_cm1'] = round(
        float(np.abs(freq_raw[:3 * molf.natm - nfreq]).max()), 2)
    modes = np.asarray(pyscf_thermo.harmonic_analysis(molf, h)['freq_wavenumber'],
                       dtype=float)
    modes = np.sort(modes)
    rec['n_imaginary'] = int(np.sum(modes < -1e-6))
    rec['frequencies_cm1'] = [round(float(f), 2) for f in modes]
    rec['lowest_frequency_cm1'] = round(float(modes[0]), 2)
    rec['n_vibrational_modes'] = int(len(modes))
    rec['n_expected_modes'] = 3 * molf.natm - (5 if _is_linear(molf) else 6)

    # ---- thermochemistry (own implementation; pyscf thermo() has a unit bug)
    th = thermochemistry(molf, modes, rec['e_total_wb97xd_hartree'],
                         spec['mult'], spec['sigma'])
    for k, v in th.items():
        if isinstance(v, float):
            th[k] = round(v, 10)
    rec['thermo'] = th
    rec['thermo_hartree'] = {k: round(th[k] / HA2JMOL, 10)
                             for k in ('ZPE', 'E_0K', 'H_298', 'G_298')}
    rec['is_minimum'] = bool(rec['n_imaginary'] == 0)
    rec['total_seconds'] = round(time.time() - t_start, 1)
    rec['pyscf_version'] = PYSCF_VERSION
    return rec


def _is_linear(mol):
    mass = mol.atom_mass_list(isotope_avg=True)
    r = mol.atom_coords(unit='Angstrom')
    r -= (mass[:, None] * r).sum(0) / mass.sum()
    I = np.zeros((3, 3))
    for i in range(mol.natm):
        x, y, z = r[i]
        I += mass[i] * np.array([[y * y + z * z, -x * y, -x * z],
                                 [-x * y, x * x + z * z, -y * z],
                                 [-x * z, -y * z, x * x + y * y]])
    w = np.linalg.eigvalsh(I * AMU * 1e-20)
    return bool(w[0] < 1e-47)


def main():
    print('python', sys.version.split()[0], '| pyscf', PYSCF_VERSION,
          '| numpy', np.__version__)
    all_rec = {}
    for key, spec in SPECIES.items():
        for solvent in (None, SOLVENT):
            tag = "%s_%s" % (key, 'smd' if solvent else 'gas')
            print('=' * 70)
            print('[%s] start  %s' % (tag, time.strftime('%H:%M:%S')))
            sys.stdout.flush()
            try:
                rec = run_one(key, spec, solvent)
            except Exception as exc:
                import traceback
                traceback.print_exc()
                rec = dict(species=key, phase=tag.split('_')[-1],
                           error='%s: %s' % (type(exc).__name__, exc))
            all_rec[tag] = rec
            with open(os.path.join(RES, '%s.json' % tag), 'w') as fh:
                json.dump(rec, fh, indent=2)
            print('[%s] done   %s  (%.0f s)' % (tag, time.strftime('%H:%M:%S'),
                                                rec.get('total_seconds', -1)))
            print('     E(ωB97X + D2) = %s  n_imag = %s  freqs = %s'
                  % (rec.get('e_total_wb97xd_hartree'),
                     rec.get('n_imaginary'), rec.get('frequencies_cm1')))
            sys.stdout.flush()
    with open(os.path.join(RES, 'baseline_all.json'), 'w') as fh:
        json.dump(all_rec, fh, indent=2)
    print('ALL DONE')


if __name__ == '__main__':
    main()
