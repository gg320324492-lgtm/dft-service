import os, json, hashlib

BD = 'run_artifacts/02_nh3o3_reference/c1_nondiag_freq'
snap = {}
for f in sorted(os.listdir(BD)):
    p = os.path.join(BD, f)
    if os.path.isfile(p):
        snap[f] = dict(
            sha256=hashlib.sha256(open(p, 'rb').read()).hexdigest(),
            size=os.path.getsize(p))
for f in ['protocols/c1_nondiag_freq.py',
          'results/phase2_preparation/c1_nondiag_freq_report.md',
          'jobs/JOB-2026-0906-043_nondiag_freq_batch.md',
          'jobs/JOB-2026-0906-042_c1_cont042_batch.md']:
    if os.path.exists(f):
        snap[f] = dict(
            sha256=hashlib.sha256(open(f, 'rb').read()).hexdigest(),
            size=os.path.getsize(f))
json.dump(dict(job='JOB-2026-0906-044 evidence snapshot of JOB-043 '
                    '(preserved unmodified)', files=snap),
          open(BD + '/evidence_snapshot_hashes.json', 'w'), indent=2)
print('snapshot files:', len(snap))
for k, v in snap.items():
    print(' ', k, v['sha256'][:12])
