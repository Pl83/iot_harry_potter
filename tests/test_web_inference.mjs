import fs from 'fs';

// Shim navigateur : les fichiers font `window.X = ...`
global.window = {};
const load = p => (0, eval)(fs.readFileSync(p, 'utf8'));
load('web/movement_cnn_weights.js');
load('web/inference.js');
const M = global.window.MovementModel;

// Entrée déterministe : x[c][t] = sin(0.1*t + c)
const x = [0, 1, 2].map(c => {
  const a = new Float64Array(100);
  for (let t = 0; t < 100; t++) a[t] = Math.sin(0.1 * t + c);
  return a;
});
const logits = Array.from(M.forward(x));
const expected = [4.0416, 0.9332, -5.4571, -2.0294];
let maxDiff = 0;
for (let i = 0; i < 4; i++) maxDiff = Math.max(maxDiff, Math.abs(logits[i] - expected[i]));
const argmax = logits.indexOf(Math.max(...logits));

const probs = M.softmax(logits);
const sum = probs.reduce((a, b) => a + b, 0);

let ok = true;
function check(name, cond) { console.log((cond ? 'PASS' : 'FAIL') + ' ' + name); if (!cond) ok = false; }
check('module chargé', !!M && M.ready === true);
check('classes exposées', M.classes.length === 4 && M.classes[0] === 'circle');
check('parité logits (diff < 1e-2)', maxDiff < 1e-2);
check('argmax = 0 (circle)', argmax === 0);
check('softmax somme à 1', Math.abs(sum - 1) < 1e-9);
process.exit(ok ? 0 : 1);
