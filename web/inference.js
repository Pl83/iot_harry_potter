// Forward-pass CNN (MovementCNN) en JS pur, partagé par le moniteur et le duel.
// Nécessite window.MODEL_WEIGHTS (chargé avant, via movement_cnn_weights.js).
(function () {
  const MODEL = window.MODEL_WEIGHTS || null;

  function fwdConv(x, L) {
    const W = L.weight, b = L.bias, pad = L.pad;
    const inC = x.length, len = x[0].length;
    const outC = W.length, k = W[0][0].length;
    const padded = x.map(ch => { const p = new Float64Array(len + 2 * pad); p.set(ch, pad); return p; });
    const lout = len + 2 * pad - k + 1;
    const out = [];
    for (let oc = 0; oc < outC; oc++) {
      const acc = new Float64Array(lout);
      for (let ic = 0; ic < inC; ic++) {
        const wk = W[oc][ic], pc = padded[ic];
        for (let kk = 0; kk < k; kk++) {
          const wv = wk[kk];
          for (let i = 0; i < lout; i++) acc[i] += wv * pc[i + kk];
        }
      }
      const bo = b[oc];
      for (let i = 0; i < lout; i++) acc[i] += bo;
      out.push(acc);
    }
    return out;
  }
  function fwdBN(x, L) {
    const w = L.weight, b = L.bias, m = L.mean, v = L.var, eps = L.eps;
    return x.map((ch, c) => {
      const scale = w[c] / Math.sqrt(v[c] + eps), shift = b[c] - m[c] * scale;
      const o = new Float64Array(ch.length);
      for (let i = 0; i < ch.length; i++) o[i] = ch[i] * scale + shift;
      return o;
    });
  }
  function fwdReLU(x) {
    return x.map(ch => { const o = new Float64Array(ch.length); for (let i = 0; i < ch.length; i++) o[i] = ch[i] > 0 ? ch[i] : 0; return o; });
  }
  function fwdMaxPool(x, L) {
    const s = L.size;
    return x.map(ch => {
      const lout = Math.floor(ch.length / s), o = new Float64Array(lout);
      for (let i = 0; i < lout; i++) { let mx = -Infinity; for (let j = 0; j < s; j++) { const val = ch[i * s + j]; if (val > mx) mx = val; } o[i] = mx; }
      return o;
    });
  }
  function fwdAvgPool(x) {
    return x.map(ch => { let s = 0; for (let i = 0; i < ch.length; i++) s += ch[i]; return new Float64Array([s / ch.length]); });
  }
  function fwdLinear(x, L) {
    const flat = []; for (const ch of x) for (const v of ch) flat.push(v);
    const W = L.weight, b = L.bias, out = new Float64Array(W.length);
    for (let o = 0; o < W.length; o++) { let s = b[o]; const wr = W[o]; for (let i = 0; i < flat.length; i++) s += wr[i] * flat[i]; out[o] = s; }
    return out;
  }
  function forward(x) {
    let a = x;
    for (const L of MODEL.layers) {
      switch (L.type) {
        case 'conv': a = fwdConv(a, L); break;
        case 'bn': a = fwdBN(a, L); break;
        case 'relu': a = fwdReLU(a); break;
        case 'maxpool': a = fwdMaxPool(a, L); break;
        case 'avgpool': a = fwdAvgPool(a); break;
        case 'linear': return fwdLinear(a, L);
      }
    }
    return a;
  }
  function softmax(z) {
    let mx = -Infinity; for (const v of z) if (v > mx) mx = v;
    const e = Array.from(z, v => Math.exp(v - mx)); const s = e.reduce((a, b) => a + b, 0);
    return e.map(v => v / s);
  }

  window.MovementModel = {
    ready: !!MODEL,
    classes: MODEL ? MODEL.classes : [],
    seqLen: MODEL ? MODEL.seq_len : 100,
    forward: forward,
    softmax: softmax,
  };
})();
