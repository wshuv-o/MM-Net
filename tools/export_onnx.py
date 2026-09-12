"""Export MM-Net to ONNX and ship one patient's features, so the page can run the
published weights in a visitor's browser.

The browser runs MM-Net itself. It does not run the feature extraction or the frozen
LaBraM encoder: those produce the 388-dimensional neural vector, and LaBraM is far
too large to ship to a browser. So what is demonstrated is the released checkpoint
reproducing its own fold from pre-extracted inputs, which is the claim worth making
and is exactly how the page describes it.

SN76 is the lightest patient in fold 0's test set at 678 epochs. Features go out as
float16, which halves the download and costs nothing visible: the check below
reports the difference it makes to the predictions.
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

OUT = r'D:\proc\mmnet-isleeps\docs\data'
CKPT = os.path.join(REPO, 'MMNet_research', 'results', 'checkpoints', 'mmnet_seed42_fold0.pt')
SUBJECT = 76
L = 20

os.makedirs(OUT, exist_ok=True)
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
cfg = ck['config']
print('checkpoint fold %d, subjects %s' % (ck['fold'], ck['test_subjects']))
assert SUBJECT in ck['test_subjects'], 'demo subject must be held out for this fold'

with sweep.config(C, cfg['arm'], hidden=cfg['hidden'], drop=cfg['drop'], lr=cfg['lr'],
                  wd=cfg['wd'], cardio=cfg['cardio'], cardio_mode=cfg['cardio_mode']) as h:
    model = C.MMFeatureNet(n_eeg=h.dim, n_card=h.n_card, hidden=cfg['hidden'],
                           drop=cfg['drop'], fusion='concat', bypass=True,
                           temporal=cfg['temporal'])
    import cardio_cnn
    if cfg['cardio'] == 'raw_cnn':
        model.card_enc = cardio_cnn.CardioCNN(d=64, drop=cfg['drop'])
    model.load_state_dict(ck['state_dict'])
    model.eval()
    n_par = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('  rebuilt: %s trainable parameters' % format(n_par, ','))

    fe, fc, y, apn = C.DATA[SUBJECT]
    fe = np.asarray(fe, np.float32)
    fc = np.asarray(fc, np.float32)
    n = len(y)
    print('  SN%d: %d epochs, eeg %s, cardio %s' % (SUBJECT, n, fe.shape, fc.shape))

    # reference: full-precision PyTorch on CPU, same arithmetic as subj_infer
    def run(fe_arr, fc_arr):
        pad_ = (-n) % L
        a = np.concatenate([fe_arr, np.zeros((pad_, h.dim), np.float32)]) if pad_ else fe_arr
        b = np.concatenate([fc_arr, np.zeros((pad_, h.n_card), np.float32)]) if pad_ else fc_arr
        at = torch.tensor(a.reshape(-1, L, h.dim))
        bt = torch.tensor(b.reshape(-1, L, h.n_card))
        so_, ao_ = [], []
        with torch.no_grad():
            for i in range(0, len(at), 32):
                s_o, a_o = model(at[i:i + 32], bt[i:i + 32])
                so_.append(s_o.softmax(-1).reshape(-1, 5).numpy())
                ao_.append(torch.sigmoid(a_o).reshape(-1).numpy())
        return np.concatenate(so_)[:n], np.concatenate(ao_)[:n]

    ref_stage, ref_apnea = run(fe, fc)

    # ---- export ---------------------------------------------------------
    dummy_e = torch.zeros(1, L, h.dim)
    dummy_c = torch.zeros(1, L, h.n_card)
    onnx_path = os.path.join(OUT, 'mmnet.onnx')
    torch.onnx.export(
        model, (dummy_e, dummy_c), onnx_path,
        input_names=['feeg', 'fcard'], output_names=['stage_logits', 'apnea_logit'],
        dynamic_axes={'feeg': {0: 'batch'}, 'fcard': {0: 'batch'},
                      'stage_logits': {0: 'batch'}, 'apnea_logit': {0: 'batch'}},
        opset_version=17, dynamo=False)
    print('  mmnet.onnx: %.1f MB' % (os.path.getsize(onnx_path) / 1048576))

    # ---- the patient's inputs, padded to whole windows -------------------
    pad = (-n) % L
    fe_p = np.concatenate([fe, np.zeros((pad, h.dim), np.float32)]) if pad else fe
    fc_p = np.concatenate([fc, np.zeros((pad, h.n_card), np.float32)]) if pad else fc
    fe16, fc16 = fe_p.astype(np.float16), fc_p.astype(np.float16)
    fe16.tofile(os.path.join(OUT, 'demo_feeg.f16'))
    fc16.tofile(os.path.join(OUT, 'demo_fcard.f16'))

    # what float16 actually costs, measured rather than assumed
    h16_stage, h16_apnea = run(fe16.astype(np.float32)[:n], fc16.astype(np.float32)[:n])
    print('  float16 vs float32 inputs: max |Dstage| %.2e, max |Dapnea| %.2e, argmax agrees %.4f'
          % (np.abs(h16_stage - ref_stage).max(), np.abs(h16_apnea - ref_apnea).max(),
             (h16_stage.argmax(1) == ref_stage.argmax(1)).mean()))

    # ---- the fold's HMM parameters, exactly as run_10fold builds them ----
    rng = np.random.RandomState(100 + ck['fold'])
    tr_all = [s for s in C.SUBS if s not in ck['test_subjects']]
    tr_all = list(tr_all); rng.shuffle(tr_all)
    nv = max(10, len(tr_all) // 9)
    tr = tr_all[nv:]
    Am = np.ones((C.NC, C.NC)); pi = np.ones(C.NC)
    for s in tr:
        yy = C.DATA[s][2]; pi[yy[0]] += 1
        for x, z in zip(yy[:-1], yy[1:]):
            Am[x, z] += 1
    A_log = np.log(Am / Am.sum(1, keepdims=True))
    pi_log = np.log(pi / pi.sum())
    hmm_pred = C.hmm(A_log, pi_log, np.log(ref_stage + C.EPS))

    meta = {
        'subject': 'SN%d' % SUBJECT, 'n_epochs': int(n), 'window': L, 'pad': int(pad),
        'n_eeg': int(h.dim), 'n_card': int(h.n_card), 'fold': int(ck['fold']),
        'A_log': A_log.tolist(), 'pi_log': pi_log.tolist(),
        'reference_stage': y.astype(int).tolist(),
        'reference_apnea': apn.astype(int).tolist(),
        'expected_pred': hmm_pred.astype(int).tolist(),
        'expected_apnea': [round(float(v), 4) for v in ref_apnea],
        'expected_acc': round(float((hmm_pred == y).mean()), 4),
        'checkpoint_scores': ck['scores'],
        'params': int(n_par),
    }
    json.dump(meta, open(os.path.join(OUT, 'demo_meta.json'), 'w'), separators=(',', ':'))
    print('  SN%d accuracy from this checkpoint: %.4f' % (SUBJECT, meta['expected_acc']))
    for f in ('demo_feeg.f16', 'demo_fcard.f16', 'demo_meta.json'):
        print('  %-18s %.1f MB' % (f, os.path.getsize(os.path.join(OUT, f)) / 1048576))
