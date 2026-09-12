"""Export the ONNX model and several patients, so the page can offer a choice.

Fold 0's checkpoint held ten patients out. Four of the lighter ones are shipped
here; each is loaded only when selected, so the page stays fast for anyone who
never presses the button.

The browser runs MM-Net. It does not run feature extraction or the frozen LaBraM
encoder that produce the 388-dimensional neural vector, so the inputs are
pre-extracted and the page says so.
"""
import json
import os
import sys

import numpy as np
import torch

REPO = r'D:\proc\isleeps-sleep-staging'
sys.path.insert(0, os.path.join(REPO, 'MMNet_research', 'model'))
sys.path.insert(0, os.path.join(REPO, 'MMNet_research', 'foundation'))
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import mmnet_core as C          # noqa: E402
import sweep                    # noqa: E402
import cardio_cnn               # noqa: E402

OUT = r'D:\proc\mmnet-isleeps\docs\static\demo'
CKPT = os.path.join(REPO, 'MMNet_research', 'results', 'checkpoints', 'mmnet_seed42_fold0.pt')
L = 20
N_SUBJECTS = 10          # every patient fold 0 held out

os.makedirs(OUT, exist_ok=True)
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
cfg = ck['config']

with sweep.config(C, cfg['arm'], hidden=cfg['hidden'], drop=cfg['drop'], lr=cfg['lr'],
                  wd=cfg['wd'], cardio=cfg['cardio'], cardio_mode=cfg['cardio_mode']) as h:
    model = C.MMFeatureNet(n_eeg=h.dim, n_card=h.n_card, hidden=cfg['hidden'],
                           drop=cfg['drop'], fusion='concat', bypass=True,
                           temporal=cfg['temporal'])
    model.card_enc = cardio_cnn.CardioCNN(d=64, drop=cfg['drop'])
    model.load_state_dict(ck['state_dict'])
    model.eval()
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_par == 2764774, 'rebuilt model does not match the paper: %d' % n_par
    print('rebuilt %s parameters, matching the paper' % format(n_par, ','))

    # ---- the ONNX graph, exported once ----------------------------------
    torch.onnx.export(
        model, (torch.zeros(1, L, h.dim), torch.zeros(1, L, h.n_card)),
        os.path.join(OUT, 'mmnet.onnx'),
        input_names=['feeg', 'fcard'], output_names=['stage_logits', 'apnea_logit'],
        dynamic_axes={'feeg': {0: 'batch'}, 'fcard': {0: 'batch'},
                      'stage_logits': {0: 'batch'}, 'apnea_logit': {0: 'batch'}},
        opset_version=17, dynamo=False)
    print('  mmnet.onnx %.1f MB' % (os.path.getsize(os.path.join(OUT, 'mmnet.onnx')) / 1048576))

    # ---- the fold's HMM parameters, built exactly as run_10fold builds them
    rng = np.random.RandomState(100 + ck['fold'])
    tr_all = [s for s in C.SUBS if s not in ck['test_subjects']]
    rng.shuffle(tr_all)
    tr = tr_all[max(10, len(tr_all) // 9):]
    Am = np.ones((C.NC, C.NC)); pi = np.ones(C.NC)
    for s in tr:
        yy = C.DATA[s][2]; pi[yy[0]] += 1
        for x, z in zip(yy[:-1], yy[1:]):
            Am[x, z] += 1
    A_log = np.log(Am / Am.sum(1, keepdims=True))
    pi_log = np.log(pi / pi.sum())

    def infer(fe, fc, n):
        pad = (-n) % L
        a = np.concatenate([fe, np.zeros((pad, h.dim), np.float32)]) if pad else fe
        b = np.concatenate([fc, np.zeros((pad, h.n_card), np.float32)]) if pad else fc
        at = torch.tensor(a.reshape(-1, L, h.dim)); bt = torch.tensor(b.reshape(-1, L, h.n_card))
        so, ao = [], []
        with torch.no_grad():
            for i in range(0, len(at), 32):
                s_o, a_o = model(at[i:i + 32], bt[i:i + 32])
                so.append(s_o.softmax(-1).reshape(-1, 5).numpy())
                ao.append(torch.sigmoid(a_o).reshape(-1).numpy())
        return np.concatenate(so)[:n], np.concatenate(ao)[:n]

    # ---- the lightest held-out patients ---------------------------------
    cand = sorted(ck['test_subjects'], key=lambda s: len(C.DATA[s][2]))[:N_SUBJECTS]
    print('  scoring every patient fold %d held out, so none was seen in training' % ck['fold'])
    index = {'fold': int(ck['fold']), 'window': L, 'n_eeg': int(h.dim), 'n_card': int(h.n_card),
             'A_log': A_log.tolist(), 'pi_log': pi_log.tolist(),
             'params': int(n_par), 'subjects': []}

    for sid in cand:
        fe, fc, y, apn = C.DATA[sid]
        fe = np.asarray(fe, np.float32); fc = np.asarray(fc, np.float32); n = len(y)
        stage, apnea = infer(fe, fc, n)
        pred = C.hmm(A_log, pi_log, np.log(stage + C.EPS))
        pad = (-n) % L
        fe_p = np.concatenate([fe, np.zeros((pad, h.dim), np.float32)]) if pad else fe
        fc_p = np.concatenate([fc, np.zeros((pad, h.n_card), np.float32)]) if pad else fc
        tag = 'SN%d' % sid
        fe_p.astype(np.float16).tofile(os.path.join(OUT, '%s_feeg.f16' % tag))
        fc_p.astype(np.float16).tofile(os.path.join(OUT, '%s_fcard.f16' % tag))
        json.dump({'subject': tag, 'n_epochs': int(n), 'pad': int(pad),
                   'reference_stage': y.astype(int).tolist(),
                   'reference_apnea': apn.astype(int).tolist(),
                   'expected_pred': pred.astype(int).tolist(),
                   'expected_apnea': [round(float(v), 4) for v in apnea],
                   'expected_acc': round(float((pred == y).mean()), 4)},
                  open(os.path.join(OUT, '%s.json' % tag), 'w'), separators=(',', ':'))
        mb = sum(os.path.getsize(os.path.join(OUT, '%s_%s.f16' % (tag, k)))
                 for k in ('feeg', 'fcard')) / 1048576
        index['subjects'].append({'id': tag, 'n_epochs': int(n),
                                  'acc': round(float((pred == y).mean()), 4),
                                  'events': int(apn.sum()), 'mb': round(mb, 1)})
        print('  %-6s %4d epochs  acc %.4f  %.1f MB' % (tag, n, (pred == y).mean(), mb))

    json.dump(index, open(os.path.join(OUT, 'index.json'), 'w'), separators=(',', ':'))
    total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    print('  total %.1f MB across %d files' % (total / 1048576, len(os.listdir(OUT))))
