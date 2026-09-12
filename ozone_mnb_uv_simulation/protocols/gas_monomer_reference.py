#!/usr/bin/env python
# -*- coding: utf-8 -*-
# DEPRECATED ENTRY (JOB-2026-0906-025)
#
# This v1 driver contained a defective pre-eval kernel patch on the object
# returned by mf.nuc_grad_method(): GradScanner copies __dict__ and never
# updates the bound mol, so the gradient was frozen at the INITIAL geometry
# (evidenced by JOB-024 run v1: constant max|g|=4.357e-2 while the energy
# converged).  The executed historical copy is preserved verbatim at
#   protocols/deprecated_history/gas_monomer_reference_v1_executed.py
# The ONLY supported entry for gas monomer reference optimization is
#   protocols/gas_monomer_reference_v2.py
# (BudgetedScanner proxy + per-category persistent budgets; regression
#  tests: protocols/test_gas_driver_v2.py, 5/5 PASS).
raise NotImplementedError(
    "gas_monomer_reference.py (v1) is DISABLED (frozen-gradient driver "
    "defect, JOB-2026-0906-025). Use gas_monomer_reference_v2.py. "
    "Historical copy: protocols/deprecated_history/"
    "gas_monomer_reference_v1_executed.py")
