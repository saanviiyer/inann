// The Lab's probe view: a linear probe on every analysed layer of an uploaded model
// (Alain & Bengio 2016), each with a shuffled-label control, drawn as thermometers; then a
// steering test that adds a probed direction to a layer and counts how many answers move.
import React, { useMemo, useState } from 'react';
import ProbeTower, { ProbeLegend } from '../ProbeTower.jsx';
import { Btn, Cite, NotShown, Term } from '../ui.jsx';
import { Field, selectCls } from './LabParts.jsx';
import { modelInputOf } from '../../lib/lab/data.js';
import { splitOf, majorityOf, massMeanDirection, randomLike } from '../../lib/probe.js';

const MAXD = 16384; // wider layers are probed on per-channel maxima instead of every unit

// What to probe: the input (unless it is token ids), then every room, as flat N x D arrays.
export function probeInputs(store, rooms, ds, kind, norm) {
  const out = [];
  if (kind?.kind !== 'tokens' && ds) {
    const first = modelInputOf(ds, store.dsIndex[0], kind, norm);
    if (first.length <= MAXD) {
      const D = first.length, X = new Float32Array(store.N * D);
      for (let i = 0; i < store.N; i++) X.set(modelInputOf(ds, store.dsIndex[i], kind, norm), i * D);
      out.push({ name: '__input', label: 'Input', X, D, note: 'every value' });
    }
  }
  for (const r of rooms) {
    const st = store.stats[r.name];
    if (!st) continue;
    if (st.full && st.layout.size <= MAXD) out.push({ name: r.name, label: r.label ?? r.name, X: st.full, D: st.layout.size, note: 'every unit', full: true });
    else out.push({ name: r.name, label: r.label ?? r.name, X: st.chMax, D: st.layout.C, note: 'channel maxima' });
  }
  return out;
}

export function labelsOk(labels) {
  return !!labels && labels.every((y) => Number.isInteger(y) && y >= 0) && new Set(labels).size >= 2;
}

export function startProbes(store, layers, setProbes) {
  const labels = Array.from(store.labels);
  const K = Math.max(...labels) + 1;
  const split = splitOf(labels);
  const meta = layers.map(({ name, label, D, note, full }) => ({ name, label, D, note, full }));
  setProbes({ store, split, K, layers: meta, results: {}, running: true, cur: 0, majority: majorityOf(labels, split) });
  const w = new Worker(new URL('../../lib/lab/probe.worker.js', import.meta.url), { type: 'module' });
  w.onmessage = (e) => {
    const m = e.data;
    if (m.type === 'start') setProbes((p) => (p?.store === store ? { ...p, cur: m.i } : p));
    if (m.type === 'layer') setProbes((p) => (p?.store === store ? { ...p, results: { ...p.results, [m.name]: m.result } } : p));
    if (m.type === 'done') { setProbes((p) => (p?.store === store ? { ...p, running: false } : p)); w.terminate(); }
  };
  w.onerror = (e) => { setProbes((p) => (p?.store === store ? { ...p, running: false, error: e.message } : p)); w.terminate(); };
  w.postMessage({ layers: layers.map(({ name, X, D }) => ({ name, X, D })), labels, K, split: { train: split.train, test: split.test } });
}

const pct = (v) => (v == null ? '...' : `${(100 * v).toFixed(1)}%`);

export default function LabProbes({ model, store, rooms, ds, kind, norm, classes, probes, setProbes }) {
  const [sel, setSel] = useState(null);
  if (!store) return <div className="flex h-full items-center justify-center p-6 text-sm text-dim">Run a dataset first; a probe trains on each layer's stored activations.</div>;
  if (!labelsOk(store.labels)) return <div className="flex h-full items-center justify-center p-6 text-sm text-dim">Probes need a dataset with integer class labels (0, 1, 2, ...).</div>;
  const live = probes?.store === store ? probes : null;
  const run = () => startProbes(store, probeInputs(store, rooms, ds, kind, norm), setProbes);
  const rows = (live?.layers ?? []).map((l, i) => ({
    key: l.name, label: l.label, D: l.D, test: live.results[l.name]?.test ?? null, ctrlTest: live.results[l.name]?.ctrlTest ?? null,
    busy: live.running && i === live.cur,
  }));
  const name = (k) => classes?.[k] ?? k;
  return (
    <div className="relative h-full w-full">
      {live ? (
        <ProbeTower rows={rows} chance={live.majority} selected={sel} onSelect={setSel} shift={2.6} />
      ) : (
        <div className="flex h-full items-center justify-center p-6">
          <div className="max-w-md space-y-2 text-center text-sm text-slate-300">
            <p>Hang a thermometer on every layer: train a straight-line reader on each layer's activations and see how much of the label it can read.<Cite id="alain2016" /></p>
            <Btn on onClick={run}>Train a probe on every layer ({store.N} items)</Btn>
          </div>
        </div>
      )}
      {live && (
        <div className="scroll-thin absolute right-3 top-14 z-20 max-h-[calc(100%-4.5rem)] w-80 space-y-2 overflow-y-auto rounded-lg border border-line bg-panel/95 p-3 text-xs">
          <div className="text-slate-100">Linear probes{live.running ? ` (layer ${live.cur + 1} of ${live.layers.length})` : ''}</div>
          <p className="text-[11px] leading-snug text-slate-400">
            Each <Term k="linear probe" /> is fitted on {live.split.train.length} items and scored on the other {live.split.test.length}. The model is never changed.
          </p>
          <div>
            <div className="flex text-[10px] text-dim">
              <span className="flex-1" />
              <span className="w-14 text-right">held out</span>
              <span className="w-14 text-right" title="training accuracy of the same probe on shuffled labels">fits noise</span>
            </div>
            {[...live.layers].reverse().map((l) => {
              const r = live.results[l.name];
              return (
                <button key={l.name} onClick={() => setSel(l.name)} className={`flex w-full items-center rounded px-0.5 text-left text-[11px] ${sel === l.name ? 'bg-glow/10' : ''}`} title={`${l.D} values (${l.note})`}>
                  <span className={`flex-1 truncate ${sel === l.name ? 'text-glow' : 'text-slate-300'}`}>{l.label}{l.note === 'channel maxima' && <span className="text-dim"> · max/ch</span>}</span>
                  <span className="w-14 text-right text-glow">{pct(r?.test)}</span>
                  <span className="w-14 text-right text-dim">{pct(r?.ctrlTrain)}</span>
                </button>
              );
            })}
            <div className="mt-1 text-[10px] text-dim">Majority-class floor {pct(live.majority)}. Rooms marked max/ch are too wide to keep, so their probe reads each channel's largest value.</div>
          </div>
          <ProbeLegend />
          <p className="text-[11px] leading-snug text-slate-300">
            Compare each layer with the one below, not with zero. If a probe fits much of the shuffled labels on its training items, it is big enough to learn
            the task by itself, and only its held-out score counts.<Cite id="hewitt2019" /> A probe shows what a layer makes readable, not what the model uses.<Cite id="belinkov2022" />
          </p>
          {!live.running && <Btn onClick={run}>Train again</Btn>}
          {!live.running && <SteerTest model={model} store={store} ds={ds} kind={kind} norm={norm} live={live} sel={sel} setSel={setSel} name={name} />}
          <NotShown>
            The same probes on a copy of the model with shuffled weights, which the Tour shows. Here it would take a second full dataset pass.
          </NotShown>
        </div>
      )}
    </div>
  );
}

// Add alpha times the class-k mass-mean direction to one layer and count answers that become k,
// against a random direction of the same length.
function SteerTest({ model, store, ds, kind, norm, live, sel, setSel, name }) {
  const options = live.layers.filter((l) => l.full && l.name !== model.outputName);
  const room = options.find((l) => l.name === sel)?.name ?? options[options.length - 1]?.name;
  const [k, setK] = useState(0);
  const [alpha, setAlpha] = useState(1);
  const [res, setRes] = useState(null);
  const [busy, setBusy] = useState(false);
  const layer = options.find((l) => l.name === room);
  const dir = useMemo(() => (layer ? massMeanDirection(store.stats[room].full, layer.D, store.labels, k, live.split.train) : null), [layer, room, k, store, live]);
  if (!options.length) return <div className="text-[11px] text-dim">No stored layer is small enough to steer.</div>;
  const run = async () => {
    setBusy(true);
    const rand = randomLike(dir, 41 + k);
    const items = Array.from(live.split.test).filter((i) => store.preds[i] !== k).slice(0, 60);
    let hit = 0, hitRand = 0;
    const argmax = (d) => { let b = 0; for (let j = 1; j < d.length; j++) if (d[j] > d[b]) b = j; return b; };
    for (const i of items) {
      const x = modelInputOf(ds, store.dsIndex[i], kind, norm);
      const a = await model.runWith(x, room, (d) => d.map((v, j) => v + alpha * dir[j]));
      if (argmax(a[model.outputName].data) === k) hit++;
      const b = await model.runWith(x, room, (d) => d.map((v, j) => v + alpha * rand[j]));
      if (argmax(b[model.outputName].data) === k) hitRand++;
    }
    setRes({ n: items.length, hit, hitRand, k, alpha, room: layer.label });
    setBusy(false);
  };
  return (
    <div className="space-y-1.5 rounded-md border border-line p-2">
      <div className="text-slate-200">Does the model use it? Steer along a probed direction</div>
      <p className="text-[11px] leading-snug text-slate-400">
        The <Term k="mass-mean direction" /> for a class is its items' mean activation minus everyone else's.<Cite id="marks2023" /> Adding it to a layer
        should push answers toward that class if the model reads it; a random direction of the same length is the control.<Cite id="li2023" />
      </p>
      <Field label="layer">
        <select value={room} onChange={(e) => setSel(e.target.value)} className={`${selectCls} max-w-44`}>
          {options.map((l) => <option key={l.name} value={l.name}>{l.label}</option>)}
        </select>
      </Field>
      <Field label="toward">
        <select value={k} onChange={(e) => setK(+e.target.value)} className={selectCls}>
          {Array.from({ length: live.K }, (_, c) => <option key={c} value={c}>{name(c)}</option>)}
        </select>
      </Field>
      <Field label={`strength ${alpha.toFixed(2)}`}>
        <input type="range" min={0} max={3} step={0.25} value={alpha} onChange={(e) => setAlpha(+e.target.value)} className="w-32 accent-[#ffb547]" />
      </Field>
      <Btn onClick={run} disabled={busy}>{busy ? 'Steering...' : 'Test on held-out items'}</Btn>
      {res && (
        <div className="text-[11px] text-slate-300">
          {res.room} at {res.alpha.toFixed(2)}: of {res.n} held-out items not already {name(res.k)}, <span className="text-warm">{res.hit}</span> became {name(res.k)} with the
          probed direction and <span className="text-[#3f6fd1]">{res.hitRand}</span> with a random one.
          {res.hit <= res.hitRand + 2 && <span className="text-dim"> No better than chance here.</span>}
        </div>
      )}
    </div>
  );
}
