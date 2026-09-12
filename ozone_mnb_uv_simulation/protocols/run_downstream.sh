#!/bin/bash
# Downstream driver for JOB-2026-0905-009 (Stages 2-5 -> report).
# Assumes Stage 1 gas_opt is fully converged (all 16 tasks max|grad|<5e-5).
# Each stage runs sequentially; a failure is logged but does NOT abort the
# rest (the report is generated from whatever artifacts exist, per the
# job brief: CCSD(T) stops-and-reports rather than downgrades).
set +e
cd /mnt/e/dft-service/ozone_mnb_uv_simulation
source /root/dftvenv/bin/activate
export OMP_NUM_THREADS=4
ART=run_artifacts/01_pure_water_o3_h2o

echo "===== DOWNSTREAM START $(date) ====="
python protocols/run_stage2_gas_freq.py          > $ART/stage2.log 2>&1; echo "STAGE2 rc=$?"
# Stage 2 output is the input of Stage 3/4/5: stop rather than waste hours
# of compute on garbage if the critical stage failed.
if [ ! -f $ART/stage2_minima.json ]; then
  echo "FATAL: stage2_minima.json missing -- stopping downstream"; exit 1
fi
python protocols/run_stage3_smd_freq.py --force  > $ART/stage3.log 2>&1; echo "STAGE3 rc=$?"
python protocols/run_stage4_binding.py           > $ART/stage4.log 2>&1; echo "STAGE4 rc=$?"
python protocols/run_stage5_ccsdt.py --force     > $ART/stage5.log 2>&1; echo "STAGE5 rc=$?"
python protocols/conformer_report.py            > $ART/report.log 2>&1; echo "REPORT rc=$?"
echo "===== DOWNSTREAM DONE $(date) ====="
