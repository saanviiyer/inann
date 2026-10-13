// Chapter 9: linear probes. A probe per layer reads the digit off that layer's activations with a
// straight line; the controls say how much of that reading to believe, and steering along the
// probed direction asks whether the network itself uses it.
import React, { useEffect, useMemo, useState } from 'react';
import ProbeTower, { ProbeLegend } from './ProbeTower.jsx';
import { Term, Cite, NotShown, Btn } from './ui.jsx';
import { net, DIGITS, LABELS } from '../lib/model.js';
import { forward } from '../lib/net.js';
import { predictProbe, splitOf } from '../lib/probe.js';
import { PROBE_LAYERS, probeNetwork, scrambledNet, steerTest } from '../lib/probeTour.js';

// computed once per page load and shared by the scene and the panel
let job = null;
const listeners = new Set();
let state = { real: {}, rand: {}, done: false };
function start() {
  if (job) return;
  const split = splitOf(LABELS);
  const emit = () => listeners.forEach((f) => f(state));
  job = (async () => {
    await new Promise((r) => setTimeout(r, 50));
    const real = await probeNetwork(net, DIGITS, LABELS, {
      split, yieldEvery: true,
      onLayer: (k, r) => { state = { ...state, real: { ...state.real, [k]: r } }; emit(); },
    });
    state = { ...state, acts: real.acts, split, majority: real.majority };
    emit();
    await probeNetwork(scrambledNet(net), DIGITS, LABELS, {
      split, yieldEvery: true,
      onLayer: (k, r) => { state = { ...state, rand: { ...state.rand, [k]: r } }; emit(); },
    });
    let ok = 0;
    for (const i of split.test) if (forward(net, DIGITS[i]).pred === LABELS[i]) ok++;
    state = { ...state, netAcc: ok / split.test.length, done: true };
    emit();
  })();
}
function useProbes() {
  const [s, setS] = useState(state);
  useEffect(() => { start(); listeners.add(setS); setS(state); return () => listeners.delete(setS); }, []);
  return s;
}

export function ProbeScene({ layer, setLayer }) {
  const s = useProbes();
  const rows = PROBE_LAYERS.map((l) => ({
    key: l.key, label: l.label, D: s.real[l.key]?.D,
    test: s.real[l.key]?.test ?? null, ctrlTest: s.real[l.key]?.ctrlTest ?? null,
    rand: l.key === 'input' ? null : s.rand[l.key]?.test ?? null, busy: !s.real[l.key],
  }));
  return <ProbeTower rows={rows} chance={s.majority} selected={layer} onSelect={setLayer} />;
}

const pct = (v) => (v == null ? '...' : `${(100 * v).toFixed(1)}%`);

export function ProbePanel({ layer, setLayer, x }) {
  const s = useProbes();
  const [k, setK] = useState(3);
  const [alpha, setAlpha] = useState(1);
  const [res, setRes] = useState(null);
  useEffect(() => setRes(null), [layer, k, alpha]);
  const steerable = layer !== 'out';
  // what each layer's probe reads off the current input
  const reads = useMemo(() => {
    const f = forward(net, x);
    return Object.fromEntries(PROBE_LAYERS.map((l) => {
      const p = s.real[l.key]?.probe;
      return [l.key, p ? predictProbe(p, l.key === 'out' ? f.logits : f.acts[l.key], 0) : null];
    }));
  }, [x, s.real]);
  const steered = useMemo(() => {
    if (!s.acts || !steerable) return null;
    const { X, D } = s.acts[layer];
    // the same mass-mean direction the test uses, applied to your input
    const a = new Float64Array(D), o = new Float64Array(D);
    let na = 0, no = 0;
    for (const i of s.split.train) {
      const t = LABELS[i] === k ? a : o;
      for (let j = 0; j < D; j++) t[j] += X[i * D + j];
      LABELS[i] === k ? na++ : no++;
    }
    const vec = Float32Array.from({ length: D }, (_, j) => alpha * (a[j] / na - o[j] / no));
    return forward(net, x, { add: { layer, vec } });
  }, [s.acts, s.split, layer, k, alpha, x, steerable]);
  const run = () => setRes(steerTest(net, DIGITS, LABELS, s.acts, s.split, layer, k, alpha));
  const name = PROBE_LAYERS.find((l) => l.key === layer)?.label;

  return (
    <div className="space-y-3 rounded-lg border border-line p-3 text-xs">
      <div className="text-sm text-slate-100">A thermometer on every layer</div>
      <p className="leading-snug text-slate-400">
        A <Term k="linear probe" /> is a one-layer classifier trained on a frozen layer's activations to name the digit.<Cite id="alain2016" /> It is fitted on{' '}
        {s.split ? s.split.train.length : 315} digits and scored on the other {s.split ? s.split.test.length : 135}. The network is never changed.
      </p>

      <div>
        <div className="flex text-[10px] text-dim">
          <span className="w-24" />
          <span className="w-14 text-right">held out</span>
          <span className="w-16 text-right" title="training accuracy of the same probe on shuffled labels">fits noise</span>
          <span className="w-16 text-right" title="held-out accuracy of the same probe on a network with shuffled weights">random net</span>
          <span className="flex-1 text-right" title="what each probe reads off the digit you drew">yours</span>
        </div>
        {[...PROBE_LAYERS].reverse().map((l) => {
          const r = s.real[l.key], q = s.rand[l.key];
          return (
            <button key={l.key} onClick={() => setLayer(l.key)} className={`flex w-full items-center rounded px-0.5 text-left text-[11px] ${layer === l.key ? 'bg-glow/10' : ''}`}>
              <span className={`w-24 ${layer === l.key ? 'text-glow' : 'text-slate-300'}`}>{l.label}</span>
              <span className="w-14 text-right text-glow">{pct(r?.test)}</span>
              <span className="w-16 text-right text-dim">{pct(r?.ctrlTrain)}</span>
              <span className="w-16 text-right text-[#3f6fd1]">{l.key === 'input' ? 'same' : pct(q?.test)}</span>
              <span className="flex-1 text-right text-warm">{reads[l.key] ?? ''}</span>
            </button>
          );
        })}
        <div className="mt-1 text-[10px] text-dim">
          Majority-class floor {pct(s.majority)}. The network itself gets {pct(s.netAcc)} of the same held-out digits. On shuffled labels every probe scores near chance on held-out digits.
        </div>
      </div>
      <ProbeLegend hasRand />

      <div className="space-y-1.5 text-[11px] leading-snug text-slate-300">
        <p>
          <b className="text-slate-100">Read the controls before the bars.</b> Raw pixels already give {pct(s.real.input?.test)}, so most of each bar was there before any
          layer did work. A probe on Conv 1 can memorise {pct(s.real.conv1?.ctrlTrain)} of <i>random</i> labels on its training digits, so a big layer plus a probe
          can learn the task itself; trust held-out numbers only, and note how much a layer adds over the one below.<Cite id="hewitt2019" />
        </p>
        <p>
          A network whose weights were shuffled still gives {pct(s.rand.conv1?.test)} at Conv 1 and {pct(s.rand.dense?.test)} at Dense. Random features of the pixels
          carry the digit too, so a high probe score alone is weak evidence of learning.<Cite id="belinkov2022" />
        </p>
      </div>

      {steerable ? (
        <div className="space-y-1.5 rounded-md border border-line p-2">
          <div className="text-slate-200">Does the network use it? Steer {name} along a probed direction</div>
          <p className="text-[11px] leading-snug text-slate-400">
            Take the mean {name} activation of training digits labelled <i>k</i> minus the mean of all others, the <Term k="mass-mean direction" />,
            and add it to the layer.<Cite id="marks2023" /> If the network reads that direction, adding it should push answers to <i>k</i>; a random direction of
            the same length is the control.<Cite id="li2023" />
          </p>
          <div className="flex items-center gap-1.5">
            <span className="text-dim">toward</span>
            <select value={k} onChange={(e) => setK(+e.target.value)} className="rounded border border-line bg-panel px-1 text-slate-200">
              {Array.from({ length: 10 }, (_, d) => <option key={d} value={d}>{d}</option>)}
            </select>
            <span className="text-dim">strength</span>
            <input type="range" min={0} max={3} step={0.25} value={alpha} onChange={(e) => setAlpha(+e.target.value)} className="flex-1 accent-[#ffb547]" />
            <span className="w-8 text-right text-warm">{alpha.toFixed(2)}</span>
          </div>
          {steered && (
            <div className="text-slate-300">
              Your input with it added: the model says <span className="text-base font-semibold text-warm">{steered.pred}</span>{' '}
              <span className="text-dim">({(100 * steered.probs[k]).toFixed(0)}% on {k}; without it {forward(net, x).pred})</span>
            </div>
          )}
          <Btn onClick={run} disabled={!s.acts}>Test on held-out digits</Btn>
          {res && (
            <div className="text-slate-300">
              Of {res.n} held-out digits not already called {k}, <span className="text-warm">{res.hit}</span> became {k} with the probed direction and{' '}
              <span className="text-[#3f6fd1]">{res.hitRand}</span> with a random one.
              {res.hit <= res.hitRand + 2 && <span className="text-dim"> Here the direction does no better than chance.</span>}
            </div>
          )}
        </div>
      ) : (
        <div className="text-[11px] text-dim">Pick a layer below the output to steer it.</div>
      )}

      <NotShown>
        Concept probes for ideas other than the label, like the TCAV test of whether "stripes" matters to "zebra".<Cite id="kim2018" /> These digits carry no
        concept annotations beyond the digit itself.
      </NotShown>
    </div>
  );
}
