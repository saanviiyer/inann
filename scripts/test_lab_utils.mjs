// Unit tests for the Lab's data utilities: the .npy reader, the character tokenizer, and the
// measured summaries and search used to find units.
import assert from 'node:assert/strict';
import { parseNpy, textToIds, idsToText, inputKind, datasetFromNpy } from '../src/lib/lab/data.js';
import { describeUnit, searchUnits } from '../src/lib/lab/summary.js';

// a hand-built .npy (v1.0, little-endian float32, shape (2, 3))
function npy(header, body) {
  const h = header.padEnd(118, ' ') + '\n';
  const out = new Uint8Array(10 + h.length + body.byteLength);
  out.set([0x93, ...Buffer.from('NUMPY'), 1, 0, h.length & 255, h.length >> 8]);
  out.set(Buffer.from(h), 10);
  out.set(new Uint8Array(body.buffer), 10 + h.length);
  return out.buffer;
}
const a = parseNpy(npy("{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }", new Float32Array([1, 2, 3, 4, 5, 6])));
assert.deepEqual(a.shape, [2, 3]);
assert.deepEqual(Array.from(a.data), [1, 2, 3, 4, 5, 6]);
const b = parseNpy(npy("{'descr': '<i8', 'fortran_order': False, 'shape': (3,), }", new BigInt64Array([7n, 8n, 9n])));
assert.deepEqual(Array.from(b.data), [7, 8, 9]);
assert.throws(() => parseNpy(npy("{'descr': '<f4', 'fortran_order': True, 'shape': (2, 3), }", new Float32Array(6))), /Fortran/);

// tokenizer: characters outside the vocab dropped, padded to length with vocab[0]
const k = inputKind([16], { input: { kind: 'tokens', vocab: ['(', ')'], tokenizer: 'char' } }, 'int64');
assert.equal(k.kind, 'tokens');
const t = textToIds('(a)()', k);
assert.equal(t.used, 4);
assert.equal(idsToText(t.ids, k), '()()' + '('.repeat(12));
assert.equal(inputKind([1, 28, 28]).kind, 'image');
assert.equal(inputKind([784]).layout, 'flat');
assert.equal(inputKind([10, 6]).kind, 'image'); // 2-D float input is treated as a one-channel image
assert.equal(inputKind([5]).kind, 'vector');

// summaries and search: a unit whose top items are all class 3, at the top of the map
const N = 40, labels = Array.from({ length: N }, (_, i) => i % 4);
const acts = Float32Array.from({ length: N }, (_, i) => (labels[i] === 3 ? 5 + i / 100 : i % 7 === 0 ? 0 : 0.5));
const L = { kind: 'spatial', C: 1, H: 4, W: 4, P: 16 };
const s = describeUnit(acts, () => 1, L, labels, ['zero', 'one', 'two', 'three']);
assert.match(s.text, /10 of top 12 are three/);
assert.match(s.text, /usually upper/);
const entries = [
  { key: 'a', summary: s, className: 'three' },
  { key: 'b', summary: { text: '7 of top 12 are two · fires on 3% of items', purity: { value: 0.58 }, freq: 0.03 }, className: 'two' },
];
assert.deepEqual(searchUnits('three', entries, ['zero', 'one', 'two', 'three']).map((e) => e.key), ['a']);
assert.deepEqual(searchUnits('rare', entries).map((e) => e.key), ['b']);
// a class word must not match the same digit appearing in a count ("7 of top 12")
assert.deepEqual(searchUnits('7', entries, ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']).map((e) => e.key), []);
// integer labels keep their value, so label k lines up with model output k (a 3/7 subset of
// MNIST must not be renumbered 0/1)
const ds = datasetFromNpy({ shape: [4, 2], data: new Float32Array(8), dtype: '<f4' }, { data: Float64Array.from([3, 7, 7, 3]) }, { kind: 'vector' });
assert.deepEqual(ds.labels, [3, 7, 7, 3]);
assert.equal(ds.classes[7], '7');
const ds2 = datasetFromNpy({ shape: [2, 2], data: new Float32Array(4), dtype: '<f4' }, { data: Float64Array.from([0.5, 1.5]) }, { kind: 'vector' });
assert.deepEqual(ds2.labels, [0, 1]); // non-integer labels are ranked
console.log('OK lab utils');
