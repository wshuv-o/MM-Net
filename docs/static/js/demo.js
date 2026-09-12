/* Runs MM-Net in the browser on a selected held-out patient.
   The model is the released fold-0 checkpoint exported to ONNX. Feature
   extraction and the frozen LaBraM encoder are not run here; their outputs are
   shipped as inputs. */

(function () {
  const ORT = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
  const BASE = './static/demo/';
  const STAGES = ['W', 'N1', 'N2', 'N3', 'R'];
  const ROWS = ['N3', 'N2', 'N1', 'R', 'W'];
  const LADDER = { 0: 4, 1: 2, 2: 1, 3: 0, 4: 3 };
  const $ = s => document.querySelector(s);

  let INDEX = null, SESS = null, RESULT = null;

  const f16 = buf => {
    const u = new Uint16Array(buf), out = new Float32Array(u.length);
    for (let i = 0; i < u.length; i++) {
      const h = u[i], s = (h & 0x8000) >> 15, e = (h & 0x7c00) >> 10, f = h & 0x3ff;
      out[i] = e === 0 ? (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024)
        : e === 0x1f ? (f ? NaN : (s ? -Infinity : Infinity))
          : (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    return out;
  };

  function viterbi(logp, A, pi, K, T) {
    const dp = new Float64Array(T * K), bp = new Int32Array(T * K);
    for (let k = 0; k < K; k++) dp[k] = pi[k] + logp[k];
    for (let t = 1; t < T; t++) for (let k = 0; k < K; k++) {
      let best = -Infinity, arg = 0;
      for (let j = 0; j < K; j++) { const v = dp[(t - 1) * K + j] + A[j][k]; if (v > best) { best = v; arg = j; } }
      dp[t * K + k] = best + logp[t * K + k]; bp[t * K + k] = arg;
    }
    const path = new Int32Array(T); let best = -Infinity;
    for (let k = 0; k < K; k++) if (dp[(T - 1) * K + k] > best) { best = dp[(T - 1) * K + k]; path[T - 1] = k; }
    for (let t = T - 2; t >= 0; t--) path[t] = bp[(t + 1) * K + path[t + 1]];
    return path;
  }

  const load = src => new Promise((ok, no) => {
    const s = document.createElement('script'); s.src = src; s.onload = ok; s.onerror = no;
    document.head.appendChild(s);
  });

  function draw() {
    if (!RESULT) return;
    const { meta, pred, apnea, N } = RESULT;
    const cv = $('#demoCanvas'); if (!cv) return;
    const r = window.devicePixelRatio || 1, W = cv.parentElement.clientWidth, H = 300;
    cv.width = W * r; cv.height = H * r; cv.style.width = '100%'; cv.style.height = H + 'px';
    const c = cv.getContext('2d'); c.setTransform(r, 0, 0, r, 0, 0);
    const L = 38, R = 10, T = 18, panelH = 82, gap = 16, apH = 62;
    const X = i => L + i / N * (W - L - R);
    c.clearRect(0, 0, W, H);

    const trace = (top, seq, label, col, diff) => {
      c.fillStyle = '#4a4a4a'; c.textAlign = 'left';
      c.font = '600 11px "Google Sans","Noto Sans",sans-serif'; c.fillText(label, L, top - 4);
      c.font = '400 10px "Noto Sans",sans-serif'; c.textAlign = 'right'; c.fillStyle = '#8a8a8a';
      ROWS.forEach((s, k) => {
        const y = top + k * (panelH - 14) / 4 + 7;
        c.fillText(s, L - 6, y + 3);
        c.strokeStyle = '#ededed'; c.lineWidth = 1;
        c.beginPath(); c.moveTo(L, y); c.lineTo(W - R, y); c.stroke();
      });
      if (diff) {
        c.fillStyle = '#a6212d'; c.globalAlpha = .15;
        for (let i = 0; i < N; i++) if (seq[i] !== diff[i]) c.fillRect(X(i), top + 3, Math.max(1, (W - L - R) / N), panelH - 6);
        c.globalAlpha = 1;
      }
      c.strokeStyle = col; c.lineWidth = 1.2; c.beginPath();
      for (let i = 0; i < N; i++) {
        const y = top + (4 - LADDER[seq[i]]) * (panelH - 14) / 4 + 7;
        if (i === 0) c.moveTo(X(i), y); else c.lineTo(X(i), y);
        c.lineTo(X(i + 1), y);
      }
      c.stroke();
    };

    trace(T, meta.reference_stage, 'REFERENCE — technician', '#c46e0d', null);
    trace(T + panelH + gap, Array.from(pred), 'PREDICTED — computed in your browser', '#1f4e79', meta.reference_stage);

    const top = T + 2 * (panelH + gap);
    c.fillStyle = '#4a4a4a'; c.textAlign = 'left';
    c.font = '600 11px "Google Sans","Noto Sans",sans-serif';
    c.fillText('RESPIRATORY EVENT PROBABILITY', L, top - 4);
    c.strokeStyle = '#ededed'; c.beginPath(); c.moveTo(L, top + apH - 12); c.lineTo(W - R, top + apH - 12); c.stroke();
    c.fillStyle = '#2e7d32'; c.globalAlpha = .5; c.beginPath(); c.moveTo(L, top + apH - 12);
    for (let i = 0; i < N; i++) c.lineTo(X(i), top + apH - 12 - apnea[i] * (apH - 20));
    c.lineTo(W - R, top + apH - 12); c.closePath(); c.fill(); c.globalAlpha = 1;
    c.fillStyle = '#a6212d';
    for (let i = 0; i < N; i++) if (meta.reference_apnea[i]) c.fillRect(X(i), top + apH - 9, Math.max(1, (W - L - R) / N), 5);
    c.fillStyle = '#8a8a8a'; c.font = '400 10px "Noto Sans",sans-serif';
    c.textAlign = 'right'; c.fillText('1', L - 6, top + 7); c.fillText('0', L - 6, top + apH - 9);
    c.textAlign = 'left'; c.fillText('scored events', L, top + apH + 9);
  }

  async function run() {
    const btn = $('#demoRun'), st = $('#demoStatus'), sel = $('#demoSubject');
    const id = sel.value;
    btn.disabled = true; sel.disabled = true;
    const say = m => { st.textContent = m; };
    try {
      say('loading runtime…');
      if (!window.ort) { await load(ORT + 'ort.min.js'); ort.env.wasm.wasmPaths = ORT; }
      say('loading weights…');
      if (!SESS) SESS = await ort.InferenceSession.create(BASE + 'mmnet.onnx', { executionProviders: ['wasm'] });
      say(`loading ${id}…`);
      const [meta, eb, cb] = await Promise.all([
        fetch(`${BASE}${id}.json`).then(r => r.json()),
        fetch(`${BASE}${id}_feeg.f16`).then(r => r.arrayBuffer()),
        fetch(`${BASE}${id}_fcard.f16`).then(r => r.arrayBuffer()),
      ]);
      const { window: WIN, n_eeg: NE, n_card: NC } = INDEX;
      const N = meta.n_epochs, fe = f16(eb), fc = f16(cb), nWin = fe.length / (WIN * NE);
      const t0 = performance.now();
      const stage = new Float32Array(nWin * WIN * 5), apnea = new Float32Array(nWin * WIN);
      for (let w = 0; w < nWin; w += 8) {
        const b = Math.min(8, nWin - w);
        say(`running MM-Net… ${Math.round(w / nWin * 100)}%`);
        const out = await SESS.run({
          feeg: new ort.Tensor('float32', fe.subarray(w * WIN * NE, (w + b) * WIN * NE), [b, WIN, NE]),
          fcard: new ort.Tensor('float32', fc.subarray(w * WIN * NC, (w + b) * WIN * NC), [b, WIN, NC]),
        });
        const sl = out.stage_logits.data, al = out.apnea_logit.data;
        for (let i = 0; i < b * WIN; i++) {
          let mx = -Infinity;
          for (let k = 0; k < 5; k++) mx = Math.max(mx, sl[i * 5 + k]);
          let sum = 0;
          for (let k = 0; k < 5; k++) { const e = Math.exp(sl[i * 5 + k] - mx); stage[(w * WIN + i) * 5 + k] = e; sum += e; }
          for (let k = 0; k < 5; k++) stage[(w * WIN + i) * 5 + k] /= sum;
          apnea[w * WIN + i] = 1 / (1 + Math.exp(-al[i]));
        }
        await new Promise(r => setTimeout(r));
      }
      const secs = (performance.now() - t0) / 1000;
      const logp = new Float64Array(N * 5);
      for (let i = 0; i < N * 5; i++) logp[i] = Math.log(stage[i] + 1e-8);
      const pred = viterbi(logp, INDEX.A_log, INDEX.pi_log, 5, N);

      let acc = 0, agree = 0;
      for (let i = 0; i < N; i++) { if (pred[i] === meta.reference_stage[i]) acc++; if (pred[i] === meta.expected_pred[i]) agree++; }
      acc /= N; agree /= N;

      RESULT = { meta, pred, apnea, N };
      $('#demoFigure').hidden = false;
      draw();
      $('#demoStats').innerHTML = [
        ['Accuracy', acc.toFixed(4), 'against the technician'],
        ['PyTorch says', meta.expected_acc.toFixed(4), 'same patient, same weights'],
        ['Epochs identical', (agree * 100).toFixed(1) + '%', 'browser vs PyTorch'],
        ['Time', secs.toFixed(1) + ' s', N + ' epochs'],
      ].map(([k, v, s]) => `<div class="column has-text-centered">
        <p class="heading">${k}</p><p class="title is-4">${v}</p>
        <p class="is-size-7 has-text-grey">${s}</p></div>`).join('');
      say(`${id} scored in ${secs.toFixed(1)} s`);
      btn.textContent = 'Run again';
    } catch (e) {
      say('could not run here: ' + (e && e.message ? e.message : e));
    }
    btn.disabled = false; sel.disabled = false;
  }

  async function init() {
    try {
      INDEX = await fetch(BASE + 'index.json').then(r => r.json());
    } catch { return; }
    const sel = $('#demoSubject');
    sel.innerHTML = INDEX.subjects.map(s =>
      `<option value="${s.id}">${s.id} — ${s.n_epochs} epochs, ${s.events} with events (${s.mb} MB)</option>`).join('');
    $('#demoRun').addEventListener('click', run);
    let rt; addEventListener('resize', () => { clearTimeout(rt); rt = setTimeout(draw, 150); });
  }

  document.readyState === 'loading' ? addEventListener('DOMContentLoaded', init) : init();
})();
