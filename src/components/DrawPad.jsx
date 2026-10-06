import React, { useEffect, useRef } from 'react';
import { PixelGrid, Btn } from './ui.jsx';
import { DIGITS, LABELS } from '../lib/model.js';

// Turn a free drawing into the 8x8 format the model was trained on, following the
// optdigits recipe: fit the ink into a 32x32 bitmap, then count ink in each 4x4 block (0..16).
function toDigit(src) {
  const S = src.width;
  const d = src.getContext('2d').getImageData(0, 0, S, S).data;
  let x0 = S, y0 = S, x1 = -1, y1 = -1;
  for (let y = 0; y < S; y++)
    for (let x = 0; x < S; x++)
      if (d[(y * S + x) * 4 + 3] > 40) {
        if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
  const out = new Float32Array(64);
  if (x1 < 0) return out;
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1, k = 32 / Math.max(bw, bh);
  const tw = Math.max(1, Math.round(bw * k)), th = Math.max(1, Math.round(bh * k));
  const c = document.createElement('canvas');
  c.width = 32; c.height = 32;
  const cx = c.getContext('2d');
  cx.drawImage(src, x0, y0, bw, bh, Math.floor((32 - tw) / 2), Math.floor((32 - th) / 2), tw, th);
  const b = cx.getImageData(0, 0, 32, 32).data;
  for (let y = 0; y < 32; y++)
    for (let x = 0; x < 32; x++)
      if (b[(y * 32 + x) * 4 + 3] > 90) out[(y >> 2) * 8 + (x >> 2)] += 1 / 16;
  return out;
}

const SAMPLES = Array.from({ length: 10 }, (_, d) => LABELS.indexOf(d));

export default function DrawPad({ x, onChange }) {
  const cv = useRef(null);
  const drawing = useRef(false);
  const last = useRef(null);
  const pending = useRef(false);

  const paint = (v) => {
    const ctx = cv.current.getContext('2d');
    ctx.clearRect(0, 0, 200, 200);
    for (let p = 0; p < 64; p++) {
      ctx.fillStyle = `rgba(230,235,245,${v[p]})`;
      ctx.fillRect((p % 8) * 25, Math.floor(p / 8) * 25, 25, 25);
    }
  };
  useEffect(() => {
    paint(x); // start with the bundled sample on the pad, so strokes add to it
    const ctx = cv.current.getContext('2d');
    ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.strokeStyle = '#e6ebf5'; ctx.lineWidth = 16;
  }, []);

  const push = () => {
    if (pending.current) return;
    pending.current = true;
    requestAnimationFrame(() => { pending.current = false; onChange(toDigit(cv.current), 'draw'); });
  };
  const pos = (e) => {
    const r = cv.current.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * cv.current.width, ((e.clientY - r.top) / r.height) * cv.current.height];
  };
  const clear = () => cv.current.getContext('2d').clearRect(0, 0, 200, 200);
  const loadSample = (i) => {
    paint(DIGITS[i]); // as blocks, so further strokes add to it
    onChange(DIGITS[i], 'sample');
  };

  return (
    <div className="space-y-2">
      <div className="flex items-start gap-3">
        <div>
          <canvas
            ref={cv}
            width={200}
            height={200}
            className="h-[150px] w-[150px] touch-none cursor-crosshair rounded-md border border-line bg-[#070a12]"
            onPointerDown={(e) => { drawing.current = true; last.current = pos(e); e.currentTarget.setPointerCapture(e.pointerId); }}
            onPointerMove={(e) => {
              if (!drawing.current) return;
              const p = pos(e), ctx = cv.current.getContext('2d');
              ctx.beginPath(); ctx.moveTo(...last.current); ctx.lineTo(...p); ctx.stroke();
              last.current = p; push();
            }}
            onPointerUp={() => { drawing.current = false; push(); }}
          />
          <div className="mt-1 text-[10px] text-dim">draw a digit here</div>
        </div>
        <div>
          <PixelGrid values={x} size={78} />
          <div className="mt-1 w-[78px] text-[10px] leading-tight text-dim">what the model sees, 8x8</div>
        </div>
      </div>
      <div className="flex flex-wrap items-center gap-1">
        <Btn onClick={() => { clear(); onChange(new Float32Array(64), 'draw'); }}>Clear</Btn>
        <span className="ml-1 text-[10px] text-dim">or load a held-out digit:</span>
        {SAMPLES.map((i, d) => (
          <button key={d} onClick={() => loadSample(i)} className="h-5 w-5 rounded border border-line text-[10px] text-slate-300 hover:border-glow/50">
            {d}
          </button>
        ))}
      </div>
    </div>
  );
}
