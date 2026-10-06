import React, { useEffect, useMemo, useRef, useState } from 'react';
import { PixelGrid, Term, Cite, NotShown, Btn } from './ui.jsx';
import { net } from '../lib/model.js';
import { forward, inputGradient } from '../lib/net.js';

export function occlude(x, occ) {
  if (!occ.on) return x;
  const y = Float32Array.from(x);
  const n = 8 - occ.size + 1;
  const ox = occ.pos % n, oy = Math.floor(occ.pos / n);
  for (let r = oy; r < oy + occ.size; r++) for (let c = ox; c < ox + occ.size; c++) y[r * 8 + c] = 0;
  return y;
}

// ---------- adversarial nudge ----------
export function Adversarial({ x, f, setX, paused }) {
  const [orig, setOrig] = useState(() => Float32Array.from(x));
  const [startPred, setStartPred] = useState(f.pred);
  const [target, setTarget] = useState(() => (f.pred === 8 ? 3 : 8));
  const [running, setRunning] = useState(false);
  const xr = useRef(x);
  xr.current = x;

  const changed = useMemo(() => {
    let n = 0, l1 = 0;
    for (let i = 0; i < 64; i++) { const d = Math.abs(x[i] - orig[i]); if (d > 1e-6) n++; l1 += d; }
    return { n, l1 };
  }, [x, orig]);

  const sal = useMemo(() => {
    const g = inputGradient(net, f, 'out', f.pred);
    const m = Math.max(...g.map(Math.abs)) || 1;
    return Array.from(g, (v) => Math.abs(v) / m);
  }, [f]);

  // Greedy few-pixel step in the spirit of Papernot et al.'s saliency map attack:
  // move the single pixel that most raises (target score minus current top score) to 0 or 1.
  const nudge = () => {
    const cur = xr.current, fc = forward(net, cur);
    if (fc.pred === target) return false;
    const gt = inputGradient(net, fc, 'out', target), gp = inputGradient(net, fc, 'out', fc.pred);
    let best = -1, gain = 0, val = 0;
    for (let i = 0; i < 64; i++) {
      const g = gt[i] - gp[i];
      const to = g > 0 ? 1 : 0;
      const gi = g * (to - cur[i]);
      if (gi > gain) { gain = gi; best = i; val = to; }
    }
    if (best < 0) return false;
    const y = Float32Array.from(cur);
    y[best] = val;
    setX(y, 'adv');
    return true;
  };

  useEffect(() => {
    if (!running) return;
    const id = setInterval(() => {
      if (paused.current) return;
      const fc = forward(net, xr.current);
      let n = 0;
      for (let i = 0; i < 64; i++) if (Math.abs(xr.current[i] - orig[i]) > 1e-6) n++;
      if (fc.pred === target || n >= 16 || !nudge()) setRunning(false);
    }, 280);
    return () => clearInterval(id);
  }, [running, target, orig]); // eslint-disable-line

  const reset = () => { setRunning(false); setX(orig, 'adv'); };
  const rebase = () => { setOrig(Float32Array.from(x)); setStartPred(f.pred); };

  return (
    <div className="space-y-2 rounded-lg border border-line p-3">
      <div className="text-sm text-slate-100">Adversarial nudge</div>
      <p className="text-xs leading-snug text-slate-400">
        Click a pixel to turn it on, right-click or shift-click to turn it off. Or pick a wrong answer and let the page change one pixel at a time, choosing the pixel whose <Term k="gradient" /> helps that answer most.
        <Cite id="papernot2016" />
      </p>
      <div className="flex gap-3">
        <div>
          <PixelGrid
            values={x}
            size={104}
            onPaint={(i, e) => {
              const y = Float32Array.from(x);
              y[i] = e.button === 2 || e.buttons === 2 || e.shiftKey ? 0 : 1;
              setX(y, 'adv');
            }}
          />
          <div className="mt-1 text-[10px] text-dim">editable input</div>
        </div>
        <div>
          <PixelGrid values={x} size={104} alpha={sal} />
          <div className="mt-1 text-[10px] text-dim"><Term k="saliency" /> for "{f.pred}"<Cite id="simonyan2013" /></div>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-1.5 text-xs">
        <span className="text-dim">push toward</span>
        <select
          value={target}
          onChange={(e) => setTarget(Number(e.target.value))}
          className="rounded border border-line bg-panel px-1 py-0.5 text-slate-200"
        >
          {Array.from({ length: 10 }, (_, d) => <option key={d} value={d}>{d}</option>)}
        </select>
        <Btn onClick={nudge} disabled={f.pred === target}>One pixel</Btn>
        <Btn on={running} onClick={() => setRunning((r) => !r)} disabled={f.pred === target && !running}>{running ? 'Stop' : 'Until it flips'}</Btn>
        <Btn onClick={reset}>Undo all</Btn>
        <Btn onClick={rebase} title="treat the current image as the new starting point">Keep as start</Btn>
      </div>
      <div className="text-xs text-slate-300">
        Started as <span className="text-glow">{startPred}</span>, now <span className={f.pred !== startPred ? 'text-warm' : 'text-glow'}>{f.pred}</span> at{' '}
        {(f.probs[f.pred] * 100).toFixed(0)}%. Pixels changed: {changed.n} of 64 (total change {changed.l1.toFixed(2)} on a 0 to 1 scale).
        {running ? '' : changed.n >= 16 && f.pred !== target ? ' Stopped at 16 pixels without reaching the target.' : ''}
      </div>
      <p className="text-xs text-slate-200">Small input changes can redirect the whole internal flow.<Cite id="szegedy2013" /></p>
      <NotShown>
        Invisible attacks. On large photos the changes can be too small for people to see; on an 8x8 image every pixel is big, so these changes are visible.
      </NotShown>
    </div>
  );
}

// ---------- occlusion ----------
export function Occlusion({ x, occ, setOcc }) {
  const base = useMemo(() => forward(net, x), [x]);
  const n = 8 - occ.size + 1;
  const map = useMemo(() => {
    // drop in the unoccluded answer's probability when each patch is covered (Zeiler & Fergus 2014)
    const p0 = base.probs[base.pred];
    const m = new Float32Array(64), cnt = new Float32Array(64);
    for (let pos = 0; pos < n * n; pos++) {
      const y = occlude(x, { on: true, size: occ.size, pos });
      const drop = Math.max(0, p0 - forward(net, y).probs[base.pred]);
      const ox = pos % n, oy = Math.floor(pos / n);
      for (let r = oy; r < oy + occ.size; r++) for (let c = ox; c < ox + occ.size; c++) { m[r * 8 + c] += drop; cnt[r * 8 + c]++; }
    }
    for (let i = 0; i < 64; i++) m[i] = cnt[i] ? m[i] / cnt[i] : 0;
    const mx = Math.max(...m) || 1;
    return { vals: Array.from(m, (v) => v / mx), max: mx };
  }, [x, occ.size, n, base]);
  const cover = useMemo(() => {
    const c = new Uint8Array(64);
    const ox = occ.pos % n, oy = Math.floor(occ.pos / n);
    for (let r = oy; r < oy + occ.size; r++) for (let cc = ox; cc < ox + occ.size; cc++) c[r * 8 + cc] = 1;
    return c;
  }, [occ, n]);
  const now = useMemo(() => forward(net, occlude(x, occ)), [x, occ]);
  return (
    <div className="space-y-2 rounded-lg border border-line p-3">
      <div className="flex items-center justify-between">
        <div className="text-sm text-slate-100"><Term k="occlusion">Occlusion</Term></div>
        <Btn on={occ.on} onClick={() => setOcc({ ...occ, on: !occ.on })}>{occ.on ? 'Cover on' : 'Cover off'}</Btn>
      </div>
      <p className="text-xs leading-snug text-slate-400">
        Slide a black square across the digit. The space above dims where the covered pixels used to feed it.<Cite id="zeiler2014" />
      </p>
      <label className="block text-xs text-dim">
        position
        <input type="range" min={0} max={n * n - 1} value={Math.min(occ.pos, n * n - 1)} onChange={(e) => setOcc({ ...occ, on: true, pos: Number(e.target.value) })} className="w-full accent-[#7ee7ff]" />
      </label>
      <label className="block text-xs text-dim">
        size {occ.size}x{occ.size}
        <input type="range" min={1} max={4} value={occ.size} onChange={(e) => setOcc({ ...occ, size: Number(e.target.value), pos: 0 })} className="w-full accent-[#7ee7ff]" />
      </label>
      <div className="flex gap-3">
        <div>
          <PixelGrid values={occlude(x, occ)} size={96} overlay={occ.on ? cover : undefined} />
          <div className="mt-1 text-[10px] text-dim">
            reads as {now.pred} ({(now.probs[now.pred] * 100).toFixed(0)}%)
          </div>
        </div>
        <div>
          <PixelGrid values={x} size={96} alpha={map.vals} />
          <div className="mt-1 w-[96px] text-[10px] leading-tight text-dim">
            where covering hurts "{base.pred}" most (max drop {(map.max * 100).toFixed(0)} points)
          </div>
        </div>
      </div>
      <p className="text-xs text-slate-400">The bright regions on the right are the ones this decision depends on.</p>
    </div>
  );
}
