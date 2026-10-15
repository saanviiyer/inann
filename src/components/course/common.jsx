// Shared pieces for the Course lessons: slide links, a seeded rng, small math, and an
// interactive 2D plane (drag points, click to add, shade decision regions, draw lines).
// Everything runs in the browser from live math; no numbers are precomputed or faked.
import React, { useRef, useState } from 'react';
import { useTheme, sceneHex } from '../../lib/theme.js';
import { Btn } from '../ui.jsx';

export const SLIDES_BASE = 'http://work.caltech.edu/slides';
export const TELECOURSE = 'http://work.caltech.edu/telecourse';
export const slidesUrl = (n) => `${SLIDES_BASE}/slides${String(n).padStart(2, '0')}.pdf`;

// a link row shown at the bottom of every lesson
export function Slides({ n }) {
  return (
    <div className="flex flex-wrap items-center gap-3 border-t border-line pt-3 text-xs">
      <a className="text-glow/90 hover:text-glow" href={slidesUrl(n)} target="_blank" rel="noreferrer">
        lecture {n} slides (pdf)
      </a>
      <a className="text-dim hover:text-glow" href={TELECOURSE} target="_blank" rel="noreferrer">
        the full course
      </a>
    </div>
  );
}

// a labelled block of plain explanation above an interactive
export function Lead({ children }) {
  return <div className="space-y-2 text-sm leading-relaxed text-slate-300">{children}</div>;
}

// a small labelled slider that stays keyboard accessible
export function Slider({ label, value, min, max, step = 1, onChange, fmt = (v) => v }) {
  return (
    <label className="block text-xs text-dim">
      <span className="flex items-baseline justify-between">
        <span>{label}</span>
        <span className="text-slate-200">{fmt(value)}</span>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="mt-1 w-full accent-[var(--color-glow)]"
      />
    </label>
  );
}

// a stat chip
export function Stat({ label, value, tone = 'slate' }) {
  const color = tone === 'warm' ? 'text-warm' : tone === 'glow' ? 'text-glow' : 'text-slate-100';
  return (
    <div className="rounded border border-line p-2 text-center text-xs text-dim">
      <div className={`text-base ${color}`}>{value}</div>
      {label}
    </div>
  );
}

// a deterministic rng so a lesson's data is stable until the user resamples
export function rng(seed = 1) {
  let s = seed >>> 0 || 1;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}
// standard-normal from a uniform rng
export function gauss(r) {
  let u = 0, v = 0;
  while (u === 0) u = r();
  while (v === 0) v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

// least-squares fit of y onto the columns of X (with no extra bias column added here).
// returns the weight vector, solved by Gaussian elimination on the normal equations.
export function lstsq(X, y, ridge = 0) {
  const d = X[0].length;
  const A = Array.from({ length: d }, () => new Float64Array(d));
  const b = new Float64Array(d);
  for (let r = 0; r < X.length; r++) {
    for (let i = 0; i < d; i++) {
      b[i] += X[r][i] * y[r];
      for (let j = 0; j < d; j++) A[i][j] += X[r][i] * X[r][j];
    }
  }
  for (let i = 0; i < d; i++) A[i][i] += ridge;
  return solve(A, b);
}
// solve A w = b for small dense A
export function solve(A, b) {
  const d = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < d; c++) {
    let piv = c;
    for (let r = c + 1; r < d; r++) if (Math.abs(M[r][c]) > Math.abs(M[piv][c])) piv = r;
    if (Math.abs(M[piv][c]) < 1e-12) continue;
    [M[c], M[piv]] = [M[piv], M[c]];
    const p = M[c][c];
    for (let j = c; j <= d; j++) M[c][j] /= p;
    for (let r = 0; r < d; r++) {
      if (r === c) continue;
      const f = M[r][c];
      for (let j = c; j <= d; j++) M[r][j] -= f * M[c][j];
    }
  }
  return M.map((row) => row[d]);
}

// An interactive [-R,R]^2 plane rendered as SVG.
//  points: [{x,y,label}]   setPoints: optional, enables drag + click-to-add + right-click remove
//  classify(x,y) -> -1|0|1 shades a coarse decision grid
//  lines: [{a:[x,y], b:[x,y], color, width, dash}]
//  marks: [{x,y,color}] drawn as hollow rings (support vectors, the current pick, centers)
export function Plane({
  points = [], setPoints, classify, lines = [], marks = [], addLabel = 1,
  R = 1.2, size = 340, interactive = true, grid = 22, onBg,
}) {
  const theme = useTheme();
  const hex = sceneHex(theme);
  const ref = useRef(null);
  const drag = useRef(null);
  const col = (l) => (l > 0 ? hex.accent : hex.cool);
  const X = (x) => ((x + R) / (2 * R)) * size;
  const Y = (y) => size - ((y + R) / (2 * R)) * size;
  const toData = (e) => {
    const r = ref.current.getBoundingClientRect();
    return [
      ((e.clientX - r.left) / r.width) * 2 * R - R,
      R - ((e.clientY - r.top) / r.height) * 2 * R,
    ];
  };

  const down = (e, i) => {
    if (!interactive || !setPoints) return;
    e.preventDefault();
    if (e.button === 2) { setPoints(points.filter((_, j) => j !== i)); return; }
    drag.current = i;
    ref.current.setPointerCapture?.(e.pointerId);
  };
  const move = (e) => {
    if (drag.current == null || !setPoints) return;
    const [x, y] = toData(e);
    const i = drag.current;
    setPoints(points.map((p, j) => (j === i ? { ...p, x, y } : p)));
  };
  const up = () => { drag.current = null; };
  const bg = (e) => {
    if (!interactive || (!setPoints && !onBg)) return;
    const [x, y] = toData(e);
    if (onBg) onBg(x, y);
    else setPoints([...points, { x, y, label: addLabel }]);
  };

  const cells = [];
  if (classify) {
    const n = grid, cw = size / n;
    for (let j = 0; j < n; j++)
      for (let i = 0; i < n; i++) {
        const x = (i + 0.5) / n * 2 * R - R, y = R - (j + 0.5) / n * 2 * R;
        const s = classify(x, y);
        if (s === 0) continue;
        cells.push(
          <rect key={`${i}-${j}`} x={i * cw} y={j * cw} width={cw + 0.5} height={cw + 0.5}
            fill={s > 0 ? hex.accent : hex.cool} opacity={0.12} />,
        );
      }
  }

  return (
    <svg
      ref={ref}
      viewBox={`0 0 ${size} ${size}`}
      className="aspect-square w-full max-w-[340px] touch-none select-none rounded-lg border border-line bg-[var(--pad)]"
      onPointerMove={move}
      onPointerUp={up}
      onPointerLeave={up}
      onClick={bg}
      onContextMenu={(e) => e.preventDefault()}
      role="img"
    >
      {cells}
      <line x1={X(0)} y1={0} x2={X(0)} y2={size} stroke={hex.axis} strokeWidth="1" />
      <line x1={0} y1={Y(0)} x2={size} y2={Y(0)} stroke={hex.axis} strokeWidth="1" />
      {lines.map((ln, k) => (
        <line key={k} x1={X(ln.a[0])} y1={Y(ln.a[1])} x2={X(ln.b[0])} y2={Y(ln.b[1])}
          stroke={ln.color || hex.ink} strokeWidth={ln.width || 2} strokeDasharray={ln.dash || undefined} />
      ))}
      {marks.map((m, k) => (
        <circle key={`m${k}`} cx={X(m.x)} cy={Y(m.y)} r={10} fill="none"
          stroke={m.color || hex.glowText} strokeWidth="2" />
      ))}
      {points.map((p, i) => (
        <circle
          key={i}
          cx={X(p.x)} cy={Y(p.y)} r={6}
          fill={col(p.label)} stroke={hex.bg} strokeWidth="1.5"
          style={{ cursor: setPoints ? 'grab' : 'default' }}
          onPointerDown={(e) => { e.stopPropagation(); down(e, i); }}
          onClick={(e) => e.stopPropagation()}
        />
      ))}
    </svg>
  );
}

// a tiny line plot for curves y=f(x) over an index, with optional log y-axis.
// series: [{pts:[[x,y]...], color, dash, label}]
export function LinePlot({ series, xlabel, ylabel, width = 360, height = 220, xDomain, yDomain, logy = false }) {
  const theme = useTheme();
  const hex = sceneHex(theme);
  const pad = 34;
  const allX = series.flatMap((s) => s.pts.map((p) => p[0]));
  const allY = series.flatMap((s) => s.pts.map((p) => p[1]));
  const xd = xDomain || [Math.min(...allX), Math.max(...allX)];
  let yd = yDomain || [Math.min(...allY), Math.max(...allY)];
  const ty = (v) => (logy ? Math.log10(Math.max(v, 1e-9)) : v);
  if (logy) yd = [ty(yd[0] || 1), ty(yd[1] || 1)];
  const X = (x) => pad + ((x - xd[0]) / (xd[1] - xd[0] || 1)) * (width - pad - 8);
  const Y = (y) => height - pad - ((ty(y) - yd[0]) / (yd[1] - yd[0] || 1)) * (height - pad - 10);
  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="w-full rounded-lg border border-line bg-[var(--pad)]" role="img">
      <line x1={pad} y1={height - pad} x2={width - 8} y2={height - pad} stroke={hex.axis} />
      <line x1={pad} y1={10} x2={pad} y2={height - pad} stroke={hex.axis} />
      {series.map((s, k) => (
        <polyline key={k} fill="none" stroke={s.color || hex.glowText} strokeWidth="2" strokeDasharray={s.dash || undefined}
          points={s.pts.map((p) => `${X(p[0])},${Y(p[1])}`).join(' ')} />
      ))}
      <text x={width / 2} y={height - 4} textAnchor="middle" fontSize="10" fill={hex.neutral}>{xlabel}</text>
      <text x={10} y={14} fontSize="10" fill={hex.neutral}>{ylabel}</text>
      {series.filter((s) => s.label).map((s, k) => (
        <text key={`l${k}`} x={width - 10} y={22 + k * 14} textAnchor="end" fontSize="10" fill={s.color || hex.glowText}>{s.label}</text>
      ))}
    </svg>
  );
}

// a small control row of buttons
export function Controls({ children }) {
  return <div className="flex flex-wrap items-center gap-2">{children}</div>;
}

// endpoints of the line w0*x + w1*y + b = 0 clipped to [-R,R], for drawing a linear boundary.
export function boundaryLine(w, R = 1.2) {
  const [a, b, c] = w;
  if (Math.abs(b) > 1e-6) return { a: [-R, (-c - a * -R) / b], b: [R, (-c - a * R) / b] };
  if (Math.abs(a) > 1e-6) return { a: [-c / a, -R], b: [-c / a, R] };
  return { a: [-R, R], b: [R, R] };
}
export const sign = (v) => (v >= 0 ? 1 : -1);

export { Btn };
export const useLocalState = useState;
