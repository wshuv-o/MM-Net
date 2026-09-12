"""Reconstruct per-patient hypnograms for the project page.

predictions_seed42.npz holds y_true, y_pred, apnea_true and apnea_score pooled over
all 92,560 epochs, fold by fold and subject by subject within a fold, with no record
of where one patient ends and the next begins.

per_subject_seed42.json does carry each patient's own apnea score array. Those arrays
are effectively unique, so each one can be located in the pooled vector and the same
slice taken from y_true and y_pred. That recovers the predicted and reference
hypnograms for every patient without retraining anything.

Every slice is verified: the apnea scores must match element-wise at the offset found,
and no two patients may claim overlapping ranges.
"""
import json
import os

import numpy as np

R = r'D:\proc\mmnet-isleeps'
S = r'D:\proc\isleeps-sleep-staging'
F = os.path.join(S, 'MMNet_research', 'results', 'revision', 'runs', 'final')

npz = np.load(os.path.join(F, 'predictions_seed42.npz'))
y_true, y_pred = npz['y_true'], npz['y_pred']
ap_true, ap_score = npz['apnea_true'], npz['apnea_score']
ps = json.load(open(os.path.join(F, 'per_subject_seed42.json')))

# index the pooled apnea scores by their first value so the search is not quadratic
first = {}
for i, v in enumerate(ap_score):
    first.setdefault(round(float(v), 6), []).append(i)

nights, taken, failed = {}, np.zeros(len(ap_score), bool), []
for sid_s, rec in ps.items():
    a = np.asarray(rec['apnea'], dtype=np.float32)
    n = len(a)
    hit = None
    for i in first.get(round(float(a[0]), 6), []):
        if i + n <= len(ap_score) and np.allclose(ap_score[i:i + n], a, atol=1e-5):
            if not taken[i:i + n].any():
                hit = i
                break
    if hit is None:
        failed.append(sid_s)
        continue
    taken[hit:hit + n] = True
    nights[sid_s] = {
        'ref': y_true[hit:hit + n].astype(int).tolist(),
        'pred': y_pred[hit:hit + n].astype(int).tolist(),
        'apnea': [round(float(x), 3) for x in a],
        'apnea_true': ap_true[hit:hit + n].astype(int).tolist(),
        'acc': round(rec['acc'], 4),
        'kappa': round(rec['kappa'], 4),
    }

print('matched %d of %d patients' % (len(nights), len(ps)))
if failed:
    print('  unmatched:', failed[:8])
print('  epochs claimed: %d of %d' % (int(taken.sum()), len(ap_score)))

# independent check: recompute accuracy from the recovered slices
errs = []
for sid_s, v in nights.items():
    a = float(np.mean(np.array(v['ref']) == np.array(v['pred'])))
    errs.append(abs(a - v['acc']))
print('  max |recomputed acc - stored acc|: %.6f' % (max(errs) if errs else -1))

out = os.path.join(R, 'docs', 'data')
os.makedirs(out, exist_ok=True)
p = os.path.join(out, 'nights.json')
json.dump(nights, open(p, 'w'), separators=(',', ':'))
print('  nights.json: %.1f MB' % (os.path.getsize(p) / 1048576))
