const $ = s => document.querySelector(s);
const fmt = (v, n = 3) => v === null || v === undefined ? '—' : Number(v).toFixed(n);
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();

function dpi(cv, h) {
  const r = window.devicePixelRatio || 1, w = cv.parentElement.clientWidth - 2;
  cv.width = w * r; cv.height = h * r; cv.style.height = h + 'px';
  const c = cv.getContext('2d'); c.setTransform(r, 0, 0, r, 0, 0); return [c, w, h];
}

let DATA = {}, PATIENTS = [], BENCH = [], CKPT = [];

/* ---------- headline ---------- */
function stats() {
  const h = DATA.headline;
  $('#stats').innerHTML = [
    ['Staging accuracy', h.acc.toFixed(3), '± 0.020'],
    ['Cohen’s κ', h.kappa.toFixed(3), '± 0.024'],
    ['Respiratory AUC', h.auc.toFixed(3), '± 0.038'],
    ['Parameters', (h.params / 1e6).toFixed(2) + ' M', 'trainable'],
  ].map(([k, v, s]) => `<div><p class="k">${k}</p><p class="v">${v}</p><p class="s">${s}</p></div>`).join('');
}

/* ---------- ablation ---------- */
const ABL_LABEL = {
  '-EEG': 'EEG (neural)', '-EOG': 'EOG (ocular)', '-EMG': 'EMG (muscle)', '-SpO2': 'SpO₂',
  '-effort': 'respiratory effort', '-pulse/HRV': 'pulse / HRV', '-ECG': 'ECG',
  '-airflow': 'airflow', '-all cardio': 'all cardiorespiratory'
};
const ABL_ORDER = ['-EEG', '-EOG', '-EMG', '-SpO2', '-effort', '-pulse/HRV', '-ECG', '-airflow', '-all cardio'];

function ablation() {
  const cv = $('#ablation'); const [c, W] = dpi(cv, 300);
  const rows = ABL_ORDER.map(k => DATA.ablation.find(r => r.condition === k)).filter(Boolean);
  const L = 160, R = 16, T = 26, gap = 8;
  const half = (W - L - R - gap) / 2;
  const rowH = (300 - T - 18) / rows.length;
  const maxA = 0.075, maxU = 0.14;
  c.clearRect(0, 0, W, 300);
  c.font = '600 10px "IBM Plex Sans",sans-serif'; c.fillStyle = css('--slate');
  c.textAlign = 'center';
  c.fillText('STAGING ACCURACY', L + half / 2, 12);
  c.fillText('RESPIRATORY AUC', L + half + gap + half / 2, 12);

  rows.forEach((r, i) => {
    const y = T + i * rowH, h = Math.min(rowH * .56, 16);
    c.textAlign = 'right'; c.font = '400 11.5px "IBM Plex Sans",sans-serif';
    c.fillStyle = css('--ink-2');
    c.fillText(ABL_LABEL[r.condition], L - 10, y + h * .8);
    [[r.d_acc, r.p_acc, maxA, L, css('--blue')],
     [r.d_auc, r.p_auc, maxU, L + half + gap, css('--red')]].forEach(([d, p, mx, x0, col]) => {
      const zero = x0 + half * .72;
      c.strokeStyle = css('--rule'); c.lineWidth = 1;
      c.beginPath(); c.moveTo(zero, y - 2); c.lineTo(zero, y + h + 2); c.stroke();
      const w = Math.max(1, Math.abs(d) / mx * (half * .68));
      const sig = p < 0.05;
      c.fillStyle = col; c.globalAlpha = sig ? 1 : .28;
      c.fillRect(d < 0 ? zero - w : zero, y, w, h);
      c.globalAlpha = 1;
      c.textAlign = 'left'; c.font = (sig ? '600 ' : '400 ') + '10.5px "IBM Plex Mono",monospace';
      c.fillStyle = sig ? col : css('--ink-3');
      c.fillText((d > 0 ? '+' : '') + d.toFixed(3) + (sig ? '  p ' + p.toFixed(3) : ''),
                 zero + 6 + (d < 0 ? 0 : w), y + h * .8);
    });
  });
}

/* ---------- one night ---------- */
let NIGHTS = {}, nightIds = [], nightAt = 0;
const STAGES = ['W', 'N1', 'N2', 'N3', 'R'];
const LADDER = { 0: 4, 1: 2, 2: 1, 3: 0, 4: 3 };          // W top, N3 bottom, R above N3
const ROWS = ['N3', 'N2', 'N1', 'R', 'W'];

function hypno() {
  const id = nightIds[nightAt];
  const n = NIGHTS[id];
  if (!n) return;
  const cv = $('#hypno'); const [c, W, H] = dpi(cv, 380);
  const L = 34, R = 12, T = 16, gap = 14;
  const nEp = n.ref.length, hours = nEp * 30 / 3600;
  const panelH = 96, apH = 74;
  const X = i => L + i / nEp * (W - L - R);
  const rowY = (top, s) => top + (4 - LADDER[s]) * (panelH - 16) / 4 + 8;

  c.clearRect(0, 0, W, H);
  c.font = '400 9.5px "IBM Plex Mono",monospace';

  function trace(top, seq, label, col, diffAgainst) {
    c.fillStyle = css('--slate'); c.textAlign = 'left';
    c.font = '600 10px "IBM Plex Sans",sans-serif';
    c.fillText(label, L, top - 3);
    c.font = '400 9.5px "IBM Plex Mono",monospace';
    c.textAlign = 'right'; c.fillStyle = css('--ink-3');
    ROWS.forEach((s, k) => {
      const y = top + k * (panelH - 16) / 4 + 8;
      c.fillText(s, L - 6, y + 3);
      c.strokeStyle = css('--rule-soft'); c.lineWidth = 1;
      c.beginPath(); c.moveTo(L, y); c.lineTo(W - R, y); c.stroke();
    });
    if (diffAgainst) {                                   // tint the disagreements
      c.fillStyle = css('--red'); c.globalAlpha = .16;
      for (let i = 0; i < nEp; i++) if (seq[i] !== diffAgainst[i]) c.fillRect(X(i), top + 4, Math.max(1, (W - L - R) / nEp), panelH - 8);
      c.globalAlpha = 1;
    }
    c.strokeStyle = col; c.lineWidth = 1.15; c.beginPath();
    for (let i = 0; i < nEp; i++) {
      const y = rowY(top, seq[i]), x = X(i), x2 = X(i + 1);
      if (i === 0) c.moveTo(x, y); else c.lineTo(x, y);
      c.lineTo(x2, y);
    }
    c.stroke();
    c.strokeStyle = css('--purple'); c.lineWidth = 2; c.beginPath();   // REM highlighted
    let open = false;
    for (let i = 0; i < nEp; i++) {
      if (seq[i] === 4) { const y = rowY(top, 4); if (!open) { c.moveTo(X(i), y); open = true; } c.lineTo(X(i + 1), y); }
      else open = false;
    }
    c.stroke();
  }

  trace(T, n.ref, 'REFERENCE  ·  technician', css('--amber'), null);
  trace(T + panelH + gap, n.pred, 'PREDICTED  ·  MM-Net', css('--blue'), n.ref);

  // respiratory probability
  const top = T + 2 * (panelH + gap);
  c.fillStyle = css('--slate'); c.textAlign = 'left';
  c.font = '600 10px "IBM Plex Sans",sans-serif';
  c.fillText('RESPIRATORY EVENT PROBABILITY', L, top - 3);
  c.strokeStyle = css('--rule-soft'); c.beginPath();
  c.moveTo(L, top + apH - 14); c.lineTo(W - R, top + apH - 14); c.stroke();
  c.fillStyle = css('--green'); c.globalAlpha = .5; c.beginPath();
  c.moveTo(L, top + apH - 14);
  for (let i = 0; i < nEp; i++) c.lineTo(X(i), top + apH - 14 - n.apnea[i] * (apH - 22));
  c.lineTo(W - R, top + apH - 14); c.closePath(); c.fill(); c.globalAlpha = 1;
  c.fillStyle = css('--red');                                  // scored events
  for (let i = 0; i < nEp; i++) if (n.apnea_true[i]) c.fillRect(X(i), top + apH - 11, Math.max(1, (W - L - R) / nEp), 6);
  c.fillStyle = css('--ink-3'); c.font = '400 9.5px "IBM Plex Mono",monospace';
  c.textAlign = 'right'; c.fillText('1.0', L - 6, top + 8);
  c.fillText('0', L - 6, top + apH - 11);
  c.textAlign = 'left'; c.fillText('scored events', L, top + apH + 10);

  // hours axis
  c.textAlign = 'center'; c.fillStyle = css('--ink-3');
  for (let h = 0; h <= Math.floor(hours); h += 2) c.fillText(h + 'h', X(h * 120), H - 4);

  const ev = n.apnea_true.reduce((a, b) => a + b, 0);
  $('#hypnoCap').textContent =
    `${id} · ${nEp} epochs (${hours.toFixed(1)} h) · staging accuracy ${n.acc.toFixed(3)}, ` +
    `κ ${n.kappa.toFixed(3)} · ${ev} epochs carry a scored respiratory event ` +
    `(${(ev / nEp * 100).toFixed(0)}% of the night).`;
  $('#nightMeta').textContent = `${nightAt + 1} of ${nightIds.length}`;
}

function nightSetup() {
  nightIds = Object.keys(NIGHTS).sort((a, b) => NIGHTS[b].acc - NIGHTS[a].acc);
  $('#night').innerHTML = nightIds.map((id, i) =>
    `<option value="${i}">${id} — accuracy ${NIGHTS[id].acc.toFixed(3)}</option>`).join('');
  const go = d => { nightAt = (nightAt + d + nightIds.length) % nightIds.length; $('#night').value = nightAt; hypno(); };
  $('#prevN').onclick = () => go(-1);
  $('#nextN').onclick = () => go(1);
  $('#night').onchange = e => { nightAt = +e.target.value; hypno(); };
  hypno();
}

/* ---------- per-patient scatter ---------- */
const SEV_COL = { Normal: '--green', Mild: '--blue', Moderate: '--amber', Severe: '--red' };
let showMontage = false;

function scatter() {
  const cv = $('#scatter'); const [c, W, H] = dpi(cv, 360);
  const key = $('#yaxis').value;
  const pts = PATIENTS.filter(p => p.ahi !== null && p.ahi !== undefined);
  const L = 52, R = 14, T = 14, B = 44;
  const xs = pts.map(p => p.ahi), ys = pts.map(p => p[key]);
  const x0 = 0, x1 = Math.max(...xs) * 1.05;
  const y0 = Math.min(...ys) * 0.97, y1 = Math.max(...ys) * 1.02;
  const X = v => L + (v - x0) / (x1 - x0) * (W - L - R);
  const Y = v => H - B - (v - y0) / (y1 - y0) * (H - T - B);
  c.clearRect(0, 0, W, H);
  c.strokeStyle = css('--rule-soft'); c.lineWidth = 1;
  c.font = '400 10.5px "IBM Plex Mono",monospace'; c.fillStyle = css('--ink-3');
  for (let i = 0; i <= 4; i++) {
    const v = y0 + (y1 - y0) * i / 4, y = Y(v);
    c.beginPath(); c.moveTo(L, y); c.lineTo(W - R, y); c.stroke();
    c.textAlign = 'right'; c.fillText(v.toFixed(2), L - 8, y + 3.5);
  }
  [0, 15, 30, 60, 90].filter(v => v <= x1).forEach(v => {
    c.textAlign = 'center'; c.fillText(v, X(v), H - B + 16);
  });
  c.font = '400 11px "IBM Plex Sans",sans-serif';
  c.fillText('clinical apnea–hypopnea index (events/h)', (L + W - R) / 2, H - 10);

  pts.forEach(p => {
    const col = css(SEV_COL[p.severity] || '--slate');
    c.beginPath(); c.arc(X(p.ahi), Y(p[key]), 4.6, 0, 7);
    c.fillStyle = col; c.globalAlpha = .78; c.fill(); c.globalAlpha = 1;
    if (showMontage && p.complete === false) {
      c.lineWidth = 2; c.strokeStyle = css('--ink'); c.stroke();
    }
  });
  // legend
  c.font = '400 10.5px "IBM Plex Sans",sans-serif'; c.textAlign = 'left';
  let lx = L + 4;
  Object.entries(SEV_COL).forEach(([k, v]) => {
    c.beginPath(); c.arc(lx, T + 4, 4, 0, 7); c.fillStyle = css(v); c.fill();
    c.fillStyle = css('--ink-3'); c.fillText(k, lx + 8, T + 7.5);
    lx += c.measureText(k).width + 26;
  });
  const n = pts.length, inc = pts.filter(p => p.complete === false).length;
  $('#scatterCap').textContent =
    `${n} patients with a clinical AHI on record. ` +
    (showMontage ? `${inc} with an incomplete cardiorespiratory montage are ringed. ` : '') +
    `Spearman ρ = ${DATA.ahi.rho.toFixed(3)} between predicted burden and AHI over all ${DATA.ahi.n}.`;
}

/* ---------- benchmark ---------- */
let sortK = null, sortDesc = true;
function bench() {
  const tb = $('#bench tbody');
  let rows = BENCH.slice();
  if (sortK) {
    rows.sort((a, b) => {
      const x = a[sortK], y = b[sortK];
      if (x === null || x === undefined) return 1;
      if (y === null || y === undefined) return -1;
      return (typeof x === 'number' ? y - x : String(x).localeCompare(String(y))) * (sortDesc ? 1 : -1);
    });
    tb.innerHTML = rows.map(r => row(r)).join('');
  } else {
    let out = '', blk = null;
    rows.forEach(r => {
      if (r.block !== blk) { blk = r.block; out += `<tr class="blk"><td colspan="7">${blk}</td></tr>`; }
      out += row(r);
    });
    tb.innerHTML = out;
  }
}
const row = r => `<tr class="${r.ours ? 'ours' : ''}">
  <td>${r.model}</td><td class="sub">${r.channels}</td><td class="num sub">${r.params}</td>
  <td class="num">${fmt(r.healthy)}</td><td class="num">${fmt(r.acc)}</td>
  <td class="num">${fmt(r.mf1)}</td><td class="num">${fmt(r.kappa)}</td></tr>`;

/* ---------- confusion ---------- */
function confusion() {
  const cv = $('#confusion'); const [c, W] = dpi(cv, 260);
  const S = ['W', 'N1', 'N2', 'N3', 'R'], M = DATA.confusion;
  const P = 34, cell = Math.min((W - P - 10) / 5, (260 - P - 22) / 5);
  c.clearRect(0, 0, W, 260);
  c.font = '500 11px "IBM Plex Sans",sans-serif';
  S.forEach((s, i) => {
    c.fillStyle = css('--slate'); c.textAlign = 'center';
    c.fillText(s, P + i * cell + cell / 2, P - 10);
    c.textAlign = 'right'; c.fillText(s, P - 8, P + i * cell + cell / 2 + 4);
  });
  M.forEach((r, i) => r.forEach((v, j) => {
    c.fillStyle = css('--blue'); c.globalAlpha = Math.pow(v, .62);
    c.fillRect(P + j * cell, P + i * cell, cell - 2, cell - 2); c.globalAlpha = 1;
    c.fillStyle = v > .45 ? '#fff' : css('--ink-2');
    c.font = '500 10.5px "IBM Plex Mono",monospace'; c.textAlign = 'center';
    c.fillText(v.toFixed(2), P + j * cell + cell / 2 - 1, P + i * cell + cell / 2 + 3.5);
  }));
  c.fillStyle = css('--ink-3'); c.font = '400 10.5px "IBM Plex Sans",sans-serif';
  c.textAlign = 'left'; c.fillText('predicted →', P, 260 - 6);
}

/* ---------- small bar lists ---------- */
function bars(el, items, col, fmtv) {
  const mx = Math.max(...items.map(i => i[1]));
  el.innerHTML = items.map(([k, v, sub]) => `
    <div style="display:grid;grid-template-columns:4.6rem 1fr 3.4rem;gap:.5rem;align-items:center;margin:.32rem 0">
      <span class="sub" style="color:var(--ink-2)">${k}</span>
      <span style="background:var(--rule-soft);height:8px;border-radius:2px;overflow:hidden">
        <span style="display:block;height:100%;width:${(v / mx * 100).toFixed(1)}%;background:var(--${col})"></span></span>
      <span class="mono num" style="font-size:.8rem;text-align:right">${fmtv(v)}${sub || ''}</span>
    </div>`).join('');
}

/* ---------- checkpoints ---------- */
function ckpt() {
  const tb = $('#ckpt tbody');
  if (!CKPT.length) {
    tb.innerHTML = `<tr><td colspan="5" class="sub">Checkpoint manifest not published yet.</td></tr>`;
    $('#ckptCap').textContent = 'Weights are attached to the GitHub release rather than committed, so cloning the repository stays light.';
    return;
  }
  tb.innerHTML = CKPT.map(f => `<tr>
    <td class="mono">${f.file}</td><td class="num">${fmt(f.acc)}</td>
    <td class="num">${fmt(f.kappa)}</td><td class="num">${fmt(f.auc)}</td>
    <td class="num sub">${f.mb} MB</td></tr>`).join('');
  const n = CKPT.length;
  $('#ckptCap').textContent = `${n} of 10 folds published, ${CKPT.reduce((a, b) => a + b.mb, 0).toFixed(0)} MB total. `
    + 'Weights are attached to the GitHub release rather than committed, so cloning the repository stays light.';
}

/* ---------- live inference ---------- */
const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';

function f16to32(buf) {                       // IEEE half -> float, no Float16Array needed
  const u = new Uint16Array(buf), out = new Float32Array(u.length);
  for (let i = 0; i < u.length; i++) {
    const h = u[i], s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x03ff;
    out[i] = e === 0 ? (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024)
      : e === 0x1f ? (f ? NaN : (s ? -Infinity : Infinity))
      : (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  return out;
}

function viterbi(logp, A, pi, K) {            // the same decode the paper applies
  const T = logp.length / K;
  const dp = new Float64Array(T * K), bp = new Int32Array(T * K);
  for (let k = 0; k < K; k++) dp[k] = pi[k] + logp[k];
  for (let t = 1; t < T; t++) for (let k = 0; k < K; k++) {
    let best = -Infinity, arg = 0;
    for (let j = 0; j < K; j++) { const v = dp[(t - 1) * K + j] + A[j][k]; if (v > best) { best = v; arg = j; } }
    dp[t * K + k] = best + logp[t * K + k]; bp[t * K + k] = arg;
  }
  const path = new Int32Array(T);
  let best = -Infinity;
  for (let k = 0; k < K; k++) if (dp[(T - 1) * K + k] > best) { best = dp[(T - 1) * K + k]; path[T - 1] = k; }
  for (let t = T - 2; t >= 0; t--) path[t] = bp[(t + 1) * K + path[t + 1]];
  return path;
}

const script = src => new Promise((ok, no) => {
  const s = document.createElement('script'); s.src = src; s.onload = ok; s.onerror = no;
  document.head.appendChild(s);
});

let LIVE = null;
async function runLive() {
  const btn = $('#runBtn'), st = $('#runStatus');
  btn.disabled = true;
  const say = m => { st.textContent = m; };
  try {
    say('loading runtime…');
    if (!window.ort) { await script(ORT + 'ort.min.js'); ort.env.wasm.wasmPaths = ORT; }
    say('loading weights and inputs…');
    const [meta, eBuf, cBuf] = await Promise.all([
      fetch('data/demo_meta.json').then(r => r.json()),
      fetch('data/demo_feeg.f16').then(r => r.arrayBuffer()),
      fetch('data/demo_fcard.f16').then(r => r.arrayBuffer()),
    ]);
    const sess = await ort.InferenceSession.create('data/mmnet.onnx', { executionProviders: ['wasm'] });
    const { window: L, n_eeg: NE, n_card: NC_, n_epochs: N } = meta;
    const fe = f16to32(eBuf), fc = f16to32(cBuf);
    const nWin = fe.length / (L * NE);

    const t0 = performance.now();
    const stage = new Float32Array(nWin * L * 5), apnea = new Float32Array(nWin * L);
    const B = 8;
    for (let w = 0; w < nWin; w += B) {
      const b = Math.min(B, nWin - w);
      say(`running MM-Net… ${Math.round(w / nWin * 100)}%`);
      const out = await sess.run({
        feeg: new ort.Tensor('float32', fe.subarray(w * L * NE, (w + b) * L * NE), [b, L, NE]),
        fcard: new ort.Tensor('float32', fc.subarray(w * L * NC_, (w + b) * L * NC_), [b, L, NC_]),
      });
      const sl = out.stage_logits.data, al = out.apnea_logit.data;
      for (let i = 0; i < b * L; i++) {
        let mx = -Infinity;
        for (let k = 0; k < 5; k++) mx = Math.max(mx, sl[i * 5 + k]);
        let sum = 0;
        for (let k = 0; k < 5; k++) { const e = Math.exp(sl[i * 5 + k] - mx); stage[(w * L + i) * 5 + k] = e; sum += e; }
        for (let k = 0; k < 5; k++) stage[(w * L + i) * 5 + k] /= sum;
        apnea[w * L + i] = 1 / (1 + Math.exp(-al[i]));
      }
      await new Promise(r => setTimeout(r));            // let the status paint
    }
    const secs = (performance.now() - t0) / 1000;

    const logp = new Float64Array(N * 5);
    for (let i = 0; i < N * 5; i++) logp[i] = Math.log(stage[i] + 1e-8);
    const pred = viterbi(logp, meta.A_log, meta.pi_log, 5);

    const ref = meta.reference_stage, exp = meta.expected_pred;
    let acc = 0, agree = 0;
    for (let i = 0; i < N; i++) { if (pred[i] === ref[i]) acc++; if (pred[i] === exp[i]) agree++; }
    acc /= N; agree /= N;
    let maxd = 0;
    for (let i = 0; i < N; i++) maxd = Math.max(maxd, Math.abs(apnea[i] - meta.expected_apnea[i]));

    LIVE = { meta, pred, apnea, N, acc, agree, maxd, secs };
    $('#livePanel').hidden = false;
    drawLive();
    say(`done in ${secs.toFixed(1)} s`);
    btn.textContent = 'Run again';
    btn.disabled = false;
  } catch (e) {
    say('could not run here: ' + (e && e.message ? e.message : e));
    btn.disabled = false;
  }
}

function drawLive() {
  if (!LIVE) return;
  const { meta, pred, apnea, N, acc, agree, maxd, secs } = LIVE;
  const cv = $('#live'); const [c, W, H] = dpi(cv, 250);
  const L = 34, R = 12, T = 16, panelH = 88, gap = 14;
  const X = i => L + i / N * (W - L - R);
  c.clearRect(0, 0, W, H);
  const trace = (top, seq, label, col, diff) => {
    c.fillStyle = css('--slate'); c.textAlign = 'left';
    c.font = '600 10px "IBM Plex Sans",sans-serif'; c.fillText(label, L, top - 3);
    c.font = '400 9.5px "IBM Plex Mono",monospace'; c.textAlign = 'right'; c.fillStyle = css('--ink-3');
    ROWS.forEach((s, k) => {
      const y = top + k * (panelH - 16) / 4 + 8;
      c.fillText(s, L - 6, y + 3);
      c.strokeStyle = css('--rule-soft'); c.beginPath(); c.moveTo(L, y); c.lineTo(W - R, y); c.stroke();
    });
    if (diff) {
      c.fillStyle = css('--red'); c.globalAlpha = .16;
      for (let i = 0; i < N; i++) if (seq[i] !== diff[i]) c.fillRect(X(i), top + 4, Math.max(1, (W - L - R) / N), panelH - 8);
      c.globalAlpha = 1;
    }
    c.strokeStyle = col; c.lineWidth = 1.15; c.beginPath();
    for (let i = 0; i < N; i++) {
      const y = top + (4 - LADDER[seq[i]]) * (panelH - 16) / 4 + 8;
      if (i === 0) c.moveTo(X(i), y); else c.lineTo(X(i), y);
      c.lineTo(X(i + 1), y);
    }
    c.stroke();
  };
  trace(T, meta.reference_stage, 'REFERENCE  ·  technician', css('--amber'), null);
  trace(T + panelH + gap, Array.from(pred), 'COMPUTED IN YOUR BROWSER', css('--blue'), meta.reference_stage);

  $('#liveStats').innerHTML = [
    ['Accuracy here', acc.toFixed(4), 'against the technician'],
    ['Checkpoint says', meta.expected_acc.toFixed(4), 'same patient, PyTorch'],
    ['Epochs identical', (agree * 100).toFixed(1) + '%', 'browser vs PyTorch'],
    ['Max Δ apnea', maxd.toExponential(1), 'probability drift'],
    ['Time', secs.toFixed(1) + ' s', `${N} epochs`],
  ].map(([k, v, s]) => `<div style="background:var(--surface);padding:.8rem .9rem">
    <p class="k" style="font-size:.68rem;letter-spacing:.05em;color:var(--slate);margin:0">${k}</p>
    <p style="font-family:'Source Serif 4',serif;font-size:1.35rem;font-weight:600;margin:.15rem 0 0;font-variant-numeric:tabular-nums">${v}</p>
    <p class="s" style="font-size:.72rem;color:var(--ink-3);margin:.1rem 0 0">${s}</p></div>`).join('');

  $('#liveCap').textContent =
    `${meta.subject}, held out of fold ${meta.fold}. The browser reproduces PyTorch on `
    + `${(agree * 100).toFixed(1)}% of epochs; the rest are ties broken differently in float32 `
    + `arithmetic. Inputs were shipped as float16, which the export measured at a maximum `
    + `probability difference of 1.9e-4 and no change to any predicted stage.`;
}

/* ---------- boot ---------- */
async function boot() {
  const get = async p => { try { const r = await fetch(p); return r.ok ? await r.json() : null; } catch { return null; } };
  DATA = await get('data/results.json') || {};
  PATIENTS = await get('data/patients.json') || [];
  BENCH = await get('data/benchmark.json') || [];
  NIGHTS = await get('data/nights.json') || {};
  const man = await get('data/checkpoints.json');
  CKPT = man && man.folds ? man.folds : [];

  if (!DATA.headline) { document.body.insertAdjacentHTML('afterbegin',
    '<div class="wrap"><div class="note"><p>Result files did not load. If you are viewing this file directly from disk, serve the folder over HTTP instead: <code>python -m http.server</code>.</p></div></div>'); return; }

  stats(); ablation(); scatter(); bench(); confusion(); ckpt();
  if (Object.keys(NIGHTS).length) nightSetup();
  bars($('#f1'), Object.entries(DATA.per_class_f1).map(([k, v]) => [k, v, ` (${(DATA.support[k] / 1000).toFixed(1)}k)`]), 'blue', v => v.toFixed(3));
  bars($('#sev'), Object.entries(DATA.severity).map(([k, v]) => [k, v.acc, ` (n=${v.n})`]), 'amber', v => v.toFixed(3));
  bars($('#etype'), Object.entries(DATA.event_type).filter(([, v]) => v.n_positive > 200)
       .map(([k, v]) => [k, v.auc, ` (${v.n_positive})`]), 'green', v => v.toFixed(3));
  $('#rho').textContent = DATA.ahi.rho.toFixed(3);
  $('#rhoSub').textContent = `Spearman, n = ${DATA.ahi.n}, p = ${DATA.ahi.p.toExponential(1)}`;
}

document.querySelectorAll('#bench th[data-k]').forEach(th => th.addEventListener('click', () => {
  const k = th.dataset.k;
  if (sortK === k) { sortDesc = !sortDesc; } else { sortK = k; sortDesc = true; }
  document.querySelectorAll('#bench th .ar').forEach(a => a.remove());
  th.insertAdjacentHTML('beforeend', ` <span class="ar">${sortDesc ? '▾' : '▴'}</span>`);
  bench();
}));
$('#runBtn').addEventListener('click', runLive);
$('#yaxis').addEventListener('change', scatter);
$('#togMontage').addEventListener('click', e => {
  showMontage = !showMontage; e.target.setAttribute('aria-pressed', showMontage); scatter();
});
let rt; addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(() => { ablation(); scatter(); confusion(); hypno(); drawLive(); }, 150); });
matchMedia('(prefers-color-scheme:dark)').addEventListener('change', () => { ablation(); scatter(); confusion(); hypno(); });
boot();