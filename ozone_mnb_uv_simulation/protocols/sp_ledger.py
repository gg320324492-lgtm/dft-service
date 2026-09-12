#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Persistent attempt ledger for single-point drivers (JOB-2026-0906-019).

Fixes the JOB-018 recording defect (computation performed before any record
was persisted; a post-computation read error lost the result and the script
restart silently regained the full budget).

Guarantees:
  * an attempt record is persisted BEFORE the computation starts;
  * SCF / gradient / result-save states are recorded separately;
  * restart loads the existing ledger -> the budget is NOT refreshed;
  * successful results are keyed by (tag, coords_sha, config) and reused
    after verification; error records are never overwritten (errors append).
No scientific-computation code here -- pure bookkeeping, unit-testable
without PySCF.
"""
import json
import os
import time


class SPLedger:
    def __init__(self, path, cap):
        self.path = path
        self.cap = int(cap)
        if os.path.exists(path):
            with open(path) as fh:
                self.data = json.load(fh)
        else:
            self.data = {'cap': self.cap, 'attempts': [], 'results': {}}
        # cap is fixed; a restarted script cannot regain budget
        self.data['cap'] = self.cap
        self.data.setdefault('attempts', [])
        self.data.setdefault('results', {})

    # ---- budget ----
    def attempts_used(self):
        return len(self.data['attempts'])

    def budget_left(self):
        return self.cap - self.attempts_used()

    # ---- recording ----
    def start_attempt(self, key, meta=None):
        """Persist the attempt BEFORE any computation starts."""
        rec = dict(key=key, status='started', meta=meta or {},
                   started_at=time.time())
        self.data['attempts'].append(rec)
        self._save()
        return len(self.data['attempts']) - 1

    def update(self, idx, **fields):
        rec = self.data['attempts'][idx]
        rec.update(fields)
        rec['updated_at'] = time.time()
        self._save()

    def mark_error(self, idx, error):
        """Append the error; never overwrite previous error records."""
        rec = self.data['attempts'][idx]
        rec['status'] = 'error'
        rec.setdefault('errors', []).append(
            dict(error=error, at=time.time()))
        self._save()

    def save_result(self, key, result):
        """Store the successful result under its identity key."""
        self.data['results'][key] = result
        self._save()

    def get_saved(self, key):
        return self.data['results'].get(key)

    def saved_keys(self):
        return sorted(self.data['results'].keys())

    def _save(self):
        tmp = self.path + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(self.data, fh, indent=2, default=str)
        os.replace(tmp, self.path)


def make_key(tag, coords_sha, config_tuple):
    return json.dumps([tag, coords_sha, list(config_tuple)],
                      sort_keys=True)
