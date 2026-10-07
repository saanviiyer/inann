import React, { useMemo } from 'react';
import { Btn, Cite, Term, NotShown } from './ui.jsx';
import { PRESETS, roomName, DIGITS, LABELS, net, sae } from '../lib/model.js';
import { forward } from '../lib/net.js';
import { explanationOf } from '../lib/rooms.js';

// Presets come from scripts/find_presets.mjs: measured, ranked by how selective the effect is.
export default function SteerPanel({ onRun, clamp }) {
  const list = useMemo(() => PRESETS.presets.slice(0, 6), []);
  const run = (p) => {
    const c = { layer: p.layer, index: p.index, scope: p.scope, value: p.value };
    // a held-out digit of the source class that this clamp actually flips, so the demo shows the effect
    const i = LABELS.findIndex((y, j) => y === p.from && forward(net, DIGITS[j], { sae }).pred === p.from && forward(net, DIGITS[j], { sae, clamp: c }).pred === p.to);
    onRun(p, c, i);
  };
  return (
    <div className="space-y-2 rounded-lg border border-line p-3">
      <div className="text-sm text-slate-100">Measured steering demos</div>
      <p className="text-xs leading-snug text-slate-400">
        Each demo <Term k="clamp">clamps</Term> one unit at a value it reaches naturally (or switches it off) and runs every held-out digit again. These are the most{' '}
        <Term k="selectivity">selective</Term> effects found by a search over all {6 + 12 + 32} neurons and channels and every live SAE feature.
      </p>
      {list.map((p, k) => {
        const room = { layer: p.layer, index: p.index, channel: p.scope === 'channel' };
        const ex = explanationOf(room);
        const on = clamp && clamp.layer === p.layer && clamp.index === p.index && clamp.value === p.value;
        return (
          <div key={k} className={`rounded-md border p-2 ${on ? 'border-warm/60' : 'border-line'}`}>
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-semibold text-slate-100">{roomName(room)}</span>
              <Btn on={on} onClick={() => run(p)}>{on ? 'Running' : `Run on a ${p.from}`}</Btn>
            </div>
            {ex && <div className="mt-0.5 text-[11px] italic text-slate-400">"{ex.explanation}"</div>}
            <p className="mt-1 text-xs text-slate-300">
              {p.mode === 'off' ? 'Switched off' : `Clamped at its natural max (${p.value.toFixed(2)})`}: {p.flipped} of {p.of} held-out {p.from}s become {p.to}s, while{' '}
              {p.othersToTarget} of {p.othersN} other digits become {p.to}s.
            </p>
            <div className="text-[10px] text-dim">Any answer changed for {p.changedAll} of {p.nAll} correctly read digits.</div>
          </div>
        );
      })}
      <p className="text-xs leading-snug text-slate-200">
        Setting one unit by hand and watching the answer move is causal evidence that the unit drives that behavior, not just that it fires alongside it.
        <Cite id="olah2020" /><Cite id="lin2023" />
      </p>
      {(() => {
        // the demo the brief asked for ("1s become 0s"), shown with what the measurement says about it
        const b = PRESETS.bestPair?.['1>0'];
        if (!b) return <NotShown>A clamp that turns 1s into 0s. The search found none.</NotShown>;
        const room = { layer: b.layer, index: b.index, channel: b.layer.startsWith('conv') };
        const p = { ...b, scope: room.channel ? 'channel' : 'unit', from: 1, to: 0 };
        return (
          <div className="rounded-md border border-dashed border-line p-2">
            <div className="flex items-center justify-between gap-2">
              <span className="text-xs font-semibold text-slate-100">Can anything turn 1s into 0s? {roomName(room)}</span>
              <Btn onClick={() => run(p)}>Run on a 1</Btn>
            </div>
            <p className="mt-1 text-xs text-slate-300">
              The strongest 1-to-0 clamp found turns {b.flipped} of {b.of} held-out 1s into 0s, but also {b.othersToTarget} of {b.othersN} other digits.
              It pushes almost anything toward 0, so it is not evidence about 1s in particular.
            </p>
          </div>
        );
      })()}
    </div>
  );
}
