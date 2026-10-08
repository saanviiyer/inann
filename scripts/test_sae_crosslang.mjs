// The Lab encodes SAE features in JavaScript (saeCore.js); explain_lab.py encodes the same SAE in
// numpy to pick examples for Claude. They must agree, or explanations would describe the wrong
// features. Trains a small SAE here, then compares both encoders on the same vectors.
import fs from 'node:fs';
import { execFileSync } from 'node:child_process';
import { makeSae, encode, toJSON } from '../src/lib/lab/saeCore.js';

let r = 11;
const rnd = () => { r = (r * 16807) % 2147483647; return r / 2147483647 - 0.5; };
const D = 12, F = 48, k = 3, N = 50;
const sae = makeSae(D, F, k, 5);
for (let i = 0; i < sae.be.length; i++) sae.be[i] = rnd() * 0.1;
for (let j = 0; j < D; j++) { sae.mu[j] = rnd(); sae.bd[j] = rnd() * 0.2; }
sae.s = 1.7;
const X = Array.from({ length: N }, () => Array.from({ length: D }, rnd));
const js = X.map((x) => {
  const { idx, val } = encode(sae, Float32Array.from(x));
  const out = new Array(F).fill(0);
  idx.forEach((f, t) => (out[f] = val[t]));
  return out;
});
const dir = new URL('./.cache/crosslang/', import.meta.url);
fs.mkdirSync(dir, { recursive: true });
fs.writeFileSync(new URL('sae.json', dir), JSON.stringify(toJSON(sae)));
fs.writeFileSync(new URL('x.json', dir), JSON.stringify(X));
const py = JSON.parse(execFileSync('python3', ['-c', `
import json, sys, numpy as np
sys.path.insert(0, 'scripts')
from explain_lab import sae_encode
sae = json.load(open('scripts/.cache/crosslang/sae.json'))
X = np.array(json.load(open('scripts/.cache/crosslang/x.json')), dtype=np.float32)
print(json.dumps(sae_encode(sae, X).tolist()))
`], { cwd: new URL('..', import.meta.url), encoding: 'utf8' }));
let worst = 0, sameSupport = 0;
js.forEach((row, i) => {
  row.forEach((v, f) => (worst = Math.max(worst, Math.abs(v - py[i][f]))));
  const a = row.map((v, f) => (v > 0 ? f : -1)).filter((f) => f >= 0).join(','), b = py[i].map((v, f) => (v > 0 ? f : -1)).filter((f) => f >= 0).join(',');
  if (a === b) sameSupport++;
});
console.log(`JS vs numpy SAE encode: max |diff| ${worst.toExponential(2)}, same active features on ${sameSupport}/${N}`);
if (worst > 1e-4 || sameSupport < N) { console.error('FAIL'); process.exit(1); }
console.log('OK crosslang');
