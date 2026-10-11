import React, { useEffect, useMemo, useRef, useState } from 'react';
import { PixelGrid, Term, Cite, NotShown, Btn } from './ui.jsx';
import Inspector from './Inspector.jsx';
import { roomOf, roomName, roomActOf, roomActivations, EXPLAINED } from '../lib/model.js';
import { topExamples, histogram, explanationOf, scoreBand, measuredSummary } from '../lib/rooms.js';
import { toDigit, textToDigit } from '../lib/digitize.js';
import { sceneHex } from '../lib/theme.js';

export function ScoreBadge({ ex }) {
  const b = scoreBand(ex.score);
  return (
    <span className={`inline-flex items-center gap-1 rounded border px-1.5 py-px text-[10px] ${b.cls}`} title="Pearson r between the activations Claude predicted from this sentence alone and the real activations, on 20 digits it had not seen">
      {b.label}, r = {ex.score.toFixed(2)}
    </span>
  );
}

export function Explanation({ room, compact }) {
  const ex = explanationOf(room);
  if (ex)
    return (
      <div className="space-y-1">
        <p className={`${compact ? 'text-xs' : 'text-sm'} leading-snug text-slate-100`}>"{ex.explanation}"</p>
        <div className="flex flex-wrap items-center gap-1.5">
          <ScoreBadge ex={ex} />
          {!compact && (
            <span className="text-[10px] text-dim">
              written from 12 examples, scored by <Term k="simulation score">simulation</Term> on 20 others<Cite id="bills2023" />
            </span>
          )}
        </div>
      </div>
    );
  if (room.layer === 'out') return <p className="text-xs text-slate-300">By definition, this pod's score is the network's vote for digit {room.index}.</p>;
  // no written sentence: describe the unit from what it measurably does on the held-out digits
  return (
    <div className="space-y-1">
      <p className={`${compact ? 'text-xs' : 'text-sm'} leading-snug text-slate-100`}>{measuredSummary(room)}.</p>
      {!compact && (
        <span className="text-[10px] text-dim">
          Measured from the 450 held-out digits.{' '}
          {EXPLAINED ? 'This unit is active on too few digits for a written, scored explanation.' : 'A written, simulation-scored sentence appears here once scripts/explain.py has run with an API key.'}
          <Cite id="bills2023" />
        </span>
      )}
    </div>
  );
}

function Histogram({ room, current }) {
  const ref = useRef(null);
  const h = useMemo(() => histogram(room), [room]);
  useEffect(() => {
    const cv = ref.current, ctx = cv.getContext('2d'), W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    const mx = Math.max(...h.counts, 1);
    const bw = W / h.counts.length;
    h.counts.forEach((c, i) => {
      const bh = (c / mx) * (H - 4);
      ctx.fillStyle = '#5598e7';
      ctx.beginPath();
      ctx.roundRect(i * bw + 1, H - bh, bw - 2, bh, [2, 2, 0, 0]);
      ctx.fill();
    });
    const x = Math.min(1, current / h.max) * W;
    ctx.fillStyle = '#ffb547';
    ctx.fillRect(Math.min(W - 2, x), 0, 2, H);
  }, [h, current]);
  return (
    <div>
      <canvas ref={ref} width={300} height={56} className="h-14 w-full" />
      <div className="flex justify-between text-[10px] text-dim">
        <span>0</span>
        <span>{h.max.toFixed(1)}</span>
      </div>
      <div className="text-[10px] text-dim">
        Active on {h.n - h.zeros} of {h.n} held-out digits (zeros not drawn). The orange line is your input.
      </div>
    </div>
  );
}

// Draw or type to test one unit. Feeds the main input, so the whole space updates too.
function TestPad({ setX }) {
  const cv = useRef(null);
  const drawing = useRef(false);
  const last = useRef(null);
  const [text, setText] = useState('');
  useEffect(() => {
    const ctx = cv.current.getContext('2d');
    ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.strokeStyle = '#e6ebf5'; ctx.lineWidth = 16;
  }, []);
  const pos = (e) => {
    const r = cv.current.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * 200, ((e.clientY - r.top) / r.height) * 200];
  };
  return (
    <div className="flex items-start gap-2">
      <canvas
        ref={cv}
        width={200}
        height={200}
        className="h-[96px] w-[96px] touch-none cursor-crosshair rounded border border-line bg-[var(--pad)]"
        onPointerDown={(e) => { drawing.current = true; last.current = pos(e); e.currentTarget.getContext('2d').strokeStyle = sceneHex().ink; e.currentTarget.setPointerCapture(e.pointerId); }}
        onPointerMove={(e) => {
          if (!drawing.current) return;
          const p = pos(e), ctx = cv.current.getContext('2d');
          ctx.beginPath(); ctx.moveTo(...last.current); ctx.lineTo(...p); ctx.stroke();
          last.current = p;
          setX(toDigit(cv.current), 'test');
        }}
        onPointerUp={() => { drawing.current = false; }}
      />
      <div className="space-y-1.5 text-xs">
        <input
          value={text}
          maxLength={1}
          placeholder="or type"
          onChange={(e) => { setText(e.target.value); if (e.target.value.trim()) setX(textToDigit(e.target.value), 'test'); }}
          className="w-20 rounded border border-line bg-panel px-2 py-1 text-slate-100 placeholder:text-dim"
        />
        <Btn onClick={() => { cv.current.getContext('2d').clearRect(0, 0, 200, 200); setText(''); setX(new Float32Array(64), 'test'); }}>Clear</Btn>
        <div className="text-[10px] leading-tight text-dim">The model only learned digits; other characters are out of its range.</div>
      </div>
    </div>
  );
}

function Clamp({ room, clamp, setClamp, fBase, f }) {
  const max = useMemo(() => Math.max(...roomActivations(room)), [room]);
  const mine = clamp && clamp.layer === room.layer && clamp.index === room.index;
  const value = mine ? clamp.value : +max.toFixed(2);
  const set = (v) => setClamp({ layer: room.layer, index: room.index, scope: room.channel ? 'channel' : 'unit', value: v });
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <div className="text-xs text-slate-200">
          <Term k="clamp">Clamp</Term> this {room.channel ? 'channel' : 'unit'}
        </div>
        <Btn on={mine} onClick={() => (mine ? setClamp(null) : set(value))}>{mine ? 'Clamped' : 'Clamp it'}</Btn>
      </div>
      <input type="range" min={0} max={+(1.5 * max).toFixed(2)} step={0.01} value={value} onChange={(e) => set(Number(e.target.value))} className="w-full accent-[#ffb547]" />
      <div className="flex justify-between text-[10px] text-dim">
        <span>0 (off)</span>
        <span>its natural max {max.toFixed(2)}</span>
        <span>{(1.5 * max).toFixed(1)}</span>
      </div>
      {mine && (
        <div className="text-xs text-slate-300">
          Fixed at <span className="text-warm">{clamp.value.toFixed(2)}</span>{room.channel ? ' at every position' : ''}. Answer without the clamp:{' '}
          <span className="text-glow">{fBase.pred}</span> ({(fBase.probs[fBase.pred] * 100).toFixed(0)}%), with it:{' '}
          <span className={f.pred !== fBase.pred ? 'text-warm' : 'text-glow'}>{f.pred}</span> ({(f.probs[f.pred] * 100).toFixed(0)}%).
        </div>
      )}
    </div>
  );
}

export default function FeatureRoom({ sel, f, fBase, setX, clamp, setClamp, rfMask, onClear }) {
  const room = useMemo(() => (sel ? roomOf(sel.layer, sel.index) : null), [sel]);
  const top = useMemo(() => (room && room.layer !== 'input' ? topExamples(room, 12) : []), [room]);
  if (!sel)
    return (
      <div className="rounded-lg border border-line p-3 text-sm text-slate-300">
        Click any neuron in the space, or search for one above, to open its room. Drag to orbit, scroll to zoom, <span className="text-dim">W A S D Q E</span> to fly.
      </div>
    );
  if (sel.layer === 'input') return <Inspector sel={sel} f={f} rfMask={rfMask} onClear={onClear} />;
  const act = roomActOf(f.acts, room);
  const max = Math.max(...roomActivations(room));
  return (
    <div className="space-y-3 rounded-lg border border-line p-3">
      <div className="flex items-start justify-between gap-2">
        <div>
          <div className="text-[10px] uppercase tracking-[0.18em] text-dim">room</div>
          <div className="text-sm font-semibold text-slate-100">{roomName(room)}</div>
        </div>
        <Btn onClick={onClear}>Close</Btn>
      </div>
      <Explanation room={room} />

      <div className="rounded-md border border-line p-2">
        <div className="mb-1 text-xs text-slate-200">Test this neuron</div>
        <TestPad setX={setX} />
        <div className="mt-2 flex items-baseline gap-2">
          <span className="text-4xl font-semibold text-glow">{room.layer === 'out' ? `${(act * 100).toFixed(0)}%` : act.toFixed(2)}</span>
          {room.layer !== 'out' && <span className="text-xs text-dim">{max ? Math.round((act / max) * 100) : 0}% of its largest held-out activation</span>}
        </div>
        <div className="text-[10px] text-dim">The bright current in the space shows the strongest route from your input to this unit.</div>
      </div>

      {room.layer !== 'out' && (
        <>
          <div>
            <div className="mb-1 text-xs text-slate-200">Activation over all held-out digits</div>
            <Histogram room={room} current={act} />
          </div>
          <div>
            <div className="mb-1 text-xs text-slate-200">Top 12 held-out digits {room.channel && <span className="text-dim">(channel max over positions)</span>}</div>
            <div className="grid grid-cols-6 gap-1">
              {top.map((t) => (
                <div key={t.i} className="text-center">
                  <PixelGrid values={t.img} size={42} />
                  <div className="text-[9px] text-dim">{t.label} · {t.act.toFixed(1)}</div>
                </div>
              ))}
            </div>
            <div className="mt-1 text-[10px] text-dim">The same cards float in a ring around the unit in the space.</div>
          </div>
          <div className="rounded-md border border-line p-2">
            <Clamp room={room} clamp={clamp} setClamp={setClamp} fBase={fBase} f={f} />
          </div>
        </>
      )}
      <details className="text-xs">
        <summary className="cursor-pointer text-dim hover:text-slate-200">Wiring: weights and receptive field</summary>
        <div className="mt-2">
          <Inspector sel={sel} f={f} rfMask={rfMask} onClear={onClear} embedded />
        </div>
      </details>
    </div>
  );
}
