// Small building blocks for the Lab panels.
import React, { useEffect, useRef } from 'react';
import { useTheme, sceneHex } from '../../lib/theme.js';

// A thumbnail of one input in display form (C,H,W in 0..1), with an optional amber box marking
// where a unit fired (box in input pixel coordinates).
export function Thumb({ pixels, kind, size = 44, box, title }) {
  const ref = useRef(null);
  const theme = useTheme();
  const C = kind?.C ?? 1, H = kind?.H ?? 1, W = kind?.W ?? pixels?.length ?? 1;
  useEffect(() => {
    if (!pixels) return;
    const ctx = ref.current.getContext('2d');
    const img = ctx.createImageData(W, H);
    for (let p = 0; p < H * W; p++) {
      const v = (c) => Math.round(255 * Math.min(1, Math.max(0, pixels[c * H * W + p] ?? 0)));
      // one-channel pictures read as dark ink on paper in the light theme
      const ink = (t) => (theme === 'light' ? [251 - (208 * t) / 255, 247 - (211 * t) / 255, 239 - (212 * t) / 255].map(Math.round) : [t, t, t]);
      const [r, g, b] = C === 3 ? [v(0), v(1), v(2)] : ink(v(0));
      img.data.set([r, g, b, 255], p * 4);
    }
    ctx.putImageData(img, 0, 0);
  }, [pixels, C, H, W, theme]);
  return (
    <div className="relative" style={{ width: size, height: (size * H) / W }} title={title}>
      <canvas ref={ref} width={W} height={H} className="pix h-full w-full rounded-sm bg-panel" />
      {box && (
        <div
          className="pointer-events-none absolute border border-warm"
          style={{ left: `${(box.x / W) * 100}%`, top: `${(box.y / H) * 100}%`, width: `${(box.w / W) * 100}%`, height: `${(box.h / H) * 100}%` }}
        />
      )}
    </div>
  );
}

// A vector input shown as a strip of colored cells, for models that do not take images.
export function VectorStrip({ values, size = 160 }) {
  const n = values?.length ?? 0;
  let m = 0;
  for (const v of values ?? []) m = Math.max(m, Math.abs(v));
  return (
    <div className="flex h-4 overflow-hidden rounded" style={{ width: size }}>
      {Array.from(values ?? []).slice(0, 256).map((v, i) => (
        <div key={i} className="h-full flex-1" style={{ background: v >= 0 ? `rgba(255,181,71,${Math.abs(v) / (m || 1)})` : `rgba(91,140,255,${Math.abs(v) / (m || 1)})` }} />
      ))}
      {n > 256 && <span className="ml-1 text-[9px] text-dim">+{n - 256}</span>}
    </div>
  );
}

export function Hist({ hist, current, height = 56 }) {
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current, ctx = cv.getContext('2d'), W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    const mx = Math.max(...hist.counts, 1), bw = W / hist.counts.length;
    hist.counts.forEach((c, i) => {
      const bh = (c / mx) * (H - 4);
      ctx.fillStyle = '#5598e7';
      ctx.beginPath();
      ctx.roundRect(i * bw + 1, H - bh, Math.max(1, bw - 2), bh, [2, 2, 0, 0]);
      ctx.fill();
    });
    if (current != null) {
      ctx.fillStyle = '#ffb547';
      ctx.fillRect(Math.min(W - 2, Math.max(0, ((current - hist.lo) / (hist.hi - hist.lo || 1)) * W)), 0, 2, H);
    }
  }, [hist, current]);
  return (
    <div>
      <canvas ref={ref} width={300} height={height} className="w-full" style={{ height }} />
      <div className="flex justify-between text-[10px] text-dim">
        <span>{hist.lo.toFixed(2)}</span>
        <span>{hist.hi.toFixed(2)}</span>
      </div>
    </div>
  );
}

// A small line chart for training curves.
export function Curve({ points, height = 70, label }) {
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current, ctx = cv.getContext('2d'), W = cv.width, H = cv.height;
    ctx.clearRect(0, 0, W, H);
    ctx.strokeStyle = sceneHex().axis;
    ctx.strokeRect(0.5, 0.5, W - 1, H - 1);
    if (points.length < 2) return;
    const xs = points.map((p) => p[0]), ys = points.map((p) => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y1 = Math.max(...ys, 1e-9);
    ctx.strokeStyle = '#3987e5'; ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach(([x, y], i) => {
      const px = ((x - x0) / (x1 - x0 || 1)) * (W - 8) + 4, py = H - 4 - (y / y1) * (H - 8);
      i ? ctx.lineTo(px, py) : ctx.moveTo(px, py);
    });
    ctx.stroke();
  }, [points]);
  return (
    <div>
      <canvas ref={ref} width={300} height={height} className="w-full" style={{ height }} />
      {label && <div className="text-[10px] text-dim">{label}</div>}
    </div>
  );
}

export function Field({ label, children }) {
  return (
    <label className="flex items-center justify-between gap-2 text-xs text-slate-300">
      <span className="text-dim">{label}</span>
      {children}
    </label>
  );
}

export const selectCls = 'rounded border border-line bg-panel px-1.5 py-0.5 text-xs text-slate-200';
