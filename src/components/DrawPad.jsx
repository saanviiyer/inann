import React, { useEffect, useRef } from 'react';
import { PixelGrid, Btn } from './ui.jsx';
import { DIGITS, LABELS } from '../lib/model.js';
import { toDigit } from '../lib/digitize.js';
import { sceneHex, useTheme } from '../lib/theme.js';

const SAMPLES = Array.from({ length: 10 }, (_, d) => LABELS.indexOf(d));

export default function DrawPad({ x, onChange }) {
  const theme = useTheme();
  const cv = useRef(null);
  const drawing = useRef(false);
  const last = useRef(null);
  const pending = useRef(false);

  const paint = (v) => {
    const ctx = cv.current.getContext('2d');
    ctx.clearRect(0, 0, 200, 200);
    for (let p = 0; p < 64; p++) {
      ctx.globalAlpha = v[p];
      ctx.fillStyle = sceneHex().ink;
      ctx.fillRect((p % 8) * 25, Math.floor(p / 8) * 25, 25, 25);
    }
    ctx.globalAlpha = 1;
  };
  useEffect(() => {
    paint(x); // start with the bundled sample on the pad, so strokes add to it
    const ctx = cv.current.getContext('2d');
    ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.strokeStyle = '#e6ebf5'; ctx.lineWidth = 16;
  }, []);

  useEffect(() => { paint(x); }, [theme]); // eslint-disable-line

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
            className="h-[150px] w-[150px] touch-none cursor-crosshair rounded-md border border-line bg-[var(--pad)]"
            onPointerDown={(e) => { drawing.current = true; last.current = pos(e); e.currentTarget.getContext('2d').strokeStyle = sceneHex().ink; e.currentTarget.setPointerCapture(e.pointerId); }}
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
