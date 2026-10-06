import React, { useMemo } from 'react';
import { PixelGrid, Term, NotShown, Btn } from './ui.jsx';
import { net, describe, SCALE } from '../lib/model.js';
import { incoming } from '../lib/net.js';
import { topImages } from '../lib/analysis.js';

const LAYER_SENTENCE = {
  input: <>An input pixel. It has no weights; its brightness is just the ink you drew there.</>,
  conv1: <>A <Term k="conv layer" /> neuron. Every neuron in its <Term k="channel" /> uses the same 9 weights, applied at a different spot.</>,
  conv2: <>A second conv layer neuron. It reads a 3x3 patch from all 6 channels of Conv 1, after <Term k="max pooling" />.</>,
  dense: <>A <Term k="dense layer" /> neuron. It reads all 48 pooled values from Conv 2.</>,
  out: <>An output pod. Its score is turned into a <Term k="probability" /> for one digit.</>,
};

function WeightView({ sel }) {
  const { layer, index } = sel;
  if (layer === 'input') return null;
  const inc = incoming(net, layer, index);
  const m = Math.max(...inc.map((s) => Math.abs(s.w))) || 1;
  if (layer === 'conv1') {
    const c = Math.floor(index / 64);
    const k = Array.from(net.c1w.slice(c * 9, c * 9 + 9), (v) => v / m);
    return <div className="flex items-center gap-2"><PixelGrid values={k} w={3} h={3} size={54} map="diverging" /><span className="text-[11px] text-dim">its 3x3 <Term k="filter" /></span></div>;
  }
  if (layer === 'conv2') {
    const o = Math.floor(index / 16);
    let mm = 0;
    for (let i = o * 54; i < o * 54 + 54; i++) mm = Math.max(mm, Math.abs(net.c2w[i]));
    return (
      <div>
        <div className="flex gap-1">
          {Array.from({ length: 6 }, (_, c) => (
            <div key={c} className="text-center">
              <PixelGrid values={Array.from(net.c2w.slice((o * 6 + c) * 9, (o * 6 + c) * 9 + 9), (v) => v / mm)} w={3} h={3} size={30} map="diverging" />
              <div className="text-[9px] text-dim">c{c}</div>
            </div>
          ))}
        </div>
        <div className="text-[11px] text-dim">one 3x3 filter per Conv 1 channel</div>
      </div>
    );
  }
  // dense and out: one bar per incoming weight
  return (
    <div>
      <div className="flex h-10 items-center gap-px">
        {inc.map((s, i) => (
          <div key={i} className="flex h-full flex-1 flex-col justify-center">
            <div
              style={{ height: `${Math.abs(s.w / m) * 50}%`, transform: s.w < 0 ? 'translateY(50%)' : 'translateY(-50%)' }}
              className={s.w >= 0 ? 'bg-warm' : 'bg-cool'}
            />
          </div>
        ))}
      </div>
      <div className="text-[11px] text-dim">{inc.length} incoming weights, orange positive, blue negative</div>
    </div>
  );
}

export default function Inspector({ sel, f, rfMask, onClear }) {
  const top = useMemo(() => (sel && sel.layer !== 'input' ? topImages(sel.layer, sel.index, 5) : null), [sel]);
  if (!sel)
    return (
      <div className="rounded-lg border border-line p-3 text-sm text-slate-300">
        Click any neuron in the space to inspect it. Drag to orbit, scroll to zoom, <span className="text-dim">W A S D Q E</span> to fly.
      </div>
    );
  const d = describe(sel.layer, sel.index);
  const a = f.acts[sel.layer][sel.index];
  const pre = f.pre[sel.layer][sel.index];
  const rel = Math.min(1, Math.max(0, a / SCALE[sel.layer]));
  const inc = sel.layer === 'input' ? [] : incoming(net, sel.layer, sel.index);
  return (
    <div className="space-y-3 rounded-lg border border-line p-3">
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="text-sm font-semibold text-slate-100">{d.title}</div>
          <p className="mt-1 text-xs leading-snug text-slate-400">{LAYER_SENTENCE[sel.layer]}</p>
        </div>
        <Btn onClick={onClear}>Deselect</Btn>
      </div>
      <div className="grid grid-cols-2 gap-2 text-xs">
        <div className="rounded border border-line p-2">
          <div className="text-dim">{sel.layer === 'out' ? <Term k="probability" /> : <Term k="activation" />}</div>
          <div className="text-lg text-glow">{sel.layer === 'out' ? `${(a * 100).toFixed(1)}%` : a.toFixed(3)}</div>
        </div>
        <div className="rounded border border-line p-2">
          <div className="text-dim">{sel.layer === 'out' ? <Term k="logit" /> : sel.layer === 'input' ? 'pixel value' : <Term k="pre-activation" />}</div>
          <div className="text-lg text-slate-200">{pre.toFixed(3)}</div>
        </div>
      </div>
      {sel.layer !== 'out' && sel.layer !== 'input' && (
        <div className="h-1.5 overflow-hidden rounded bg-line" title="brightness relative to this layer's 99th percentile on held-out digits">
          <div className="h-full bg-glow" style={{ width: `${rel * 100}%` }} />
        </div>
      )}
      <div className="flex gap-3">
        <div>
          <PixelGrid values={f.input} size={84} overlay={rfMask} />
          <div className="mt-1 w-[84px] text-[10px] leading-tight text-dim">
            <Term k="receptive field">receptive field</Term> outlined
          </div>
        </div>
        <div className="flex-1 space-y-1">
          <WeightView sel={sel} />
          {inc.length > 0 && <div className="text-[10px] text-dim">Glowing lines in the space are the same weights.</div>}
        </div>
      </div>
      {top && (
        <div>
          <div className="mb-1 text-xs text-slate-300">Held-out digits that excite it most</div>
          <div className="flex gap-1.5">
            {top.map((t) => (
              <div key={t.i} className="text-center">
                <PixelGrid values={t.img} size={40} overlay={rfMask} />
                <div className="text-[10px] text-dim">{t.label} ({t.act.toFixed(2)})</div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
