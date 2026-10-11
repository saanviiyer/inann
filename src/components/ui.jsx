import React, { useEffect, useRef, useState } from 'react';
import { GLOSSARY } from '../lib/glossary.js';
import { REFS, REF_ORDER } from '../lib/refs.js';

// Hover (or focus) a term to get its one-sentence definition.
export function Term({ k, children }) {
  const def = GLOSSARY[k] ?? GLOSSARY[k?.toLowerCase()];
  return (
    <span className="relative group inline" tabIndex={0}>
      <span className="border-b border-dotted border-glow/60 cursor-help">{children ?? k}</span>
      <span className="pointer-events-none absolute left-0 top-full z-50 mt-1 hidden w-64 rounded-md border border-line bg-panel px-3 py-2 text-xs leading-snug text-slate-200 shadow-xl group-hover:block group-focus:block">
        <span className="font-semibold text-glow">{k}.</span> {def}
      </span>
    </span>
  );
}

// Inline citation: superscript number, hover for the full reference, click to open it.
export function Cite({ id }) {
  const r = REFS[id];
  const n = REF_ORDER.indexOf(id) + 1;
  return (
    <span className="relative group inline">
      <a
        href={r.url ?? undefined}
        target="_blank"
        rel="noreferrer"
        className="ml-0.5 align-super text-[0.7em] text-glow/80 hover:text-glow"
      >
        [{r.short}]
      </a>
      <span className="pointer-events-none absolute right-0 top-full z-50 mt-1 hidden w-72 rounded-md border border-line bg-panel px-3 py-2 text-xs leading-snug text-slate-300 shadow-xl group-hover:block">
        {n}. {r.full}
      </span>
    </span>
  );
}

export function NotShown({ children }) {
  return (
    <div className="rounded border border-dashed border-line px-2 py-1 text-[11px] text-dim">
      <span className="mr-1 rounded bg-line px-1 py-px text-[10px] uppercase tracking-wide text-slate-300">not shown here</span>
      {children}
    </div>
  );
}

export function Btn({ on, children, className = '', ...p }) {
  return (
    <button
      {...p}
      className={`rounded-md border px-2.5 py-1 text-xs transition disabled:opacity-40 ${
        on ? 'border-glow/70 bg-glow/15 text-glow' : 'border-line bg-panel text-slate-200 hover:border-glow/40'
      } ${className}`}
    >
      {children}
    </button>
  );
}

export function Kbd({ children }) {
  return <kbd className="rounded border border-line bg-panel px-1 text-[10px] text-slate-300">{children}</kbd>;
}

// Colormaps. All return [r,g,b] 0..255.
export const cmap = {
  gray: (v) => {
    const c = Math.round(255 * Math.min(1, Math.max(0, v)));
    return [c, c, c];
  },
  glow: (v) => {
    v = Math.min(1, Math.max(0, v));
    return [Math.round(20 + 150 * v * v), Math.round(28 + 210 * v), Math.round(45 + 210 * v)];
  },
  heat: (v) => {
    v = Math.min(1, Math.max(0, v));
    return [Math.round(30 + 225 * Math.min(1, v * 1.6)), Math.round(20 + 180 * Math.max(0, v - 0.35) * 1.5), Math.round(40 + 40 * v)];
  },
  // v in -1..1
  diverging: (v) => {
    v = Math.max(-1, Math.min(1, v));
    return v >= 0
      ? [Math.round(18 + 237 * v), Math.round(22 + 159 * v), Math.round(32 + 39 * v)]
      : [Math.round(18 + 73 * -v), Math.round(22 + 118 * -v), Math.round(32 + 223 * -v)];
  },
};

// A small pixel image drawn on a canvas. values: array of w*h numbers, already scaled for the colormap.
// overlay: optional array of 0/1 mask drawn as an amber outline region. onPaint(i, event) for drawing.
export function PixelGrid({ values, w = 8, h = 8, size = 128, map = 'gray', overlay, alpha, onPaint, title, className = '' }) {
  const ref = useRef(null);
  const [drag, setDrag] = useState(false);
  useEffect(() => {
    const cv = ref.current;
    const ctx = cv.getContext('2d');
    const img = ctx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
      const [r, g, b] = cmap[map](values[i]);
      img.data[i * 4] = r; img.data[i * 4 + 1] = g; img.data[i * 4 + 2] = b;
      img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [values, w, h, map]);
  const cell = (e) => {
    const r = e.currentTarget.getBoundingClientRect();
    const x = Math.floor(((e.clientX - r.left) / r.width) * w);
    const y = Math.floor(((e.clientY - r.top) / r.height) * h);
    if (x < 0 || y < 0 || x >= w || y >= h) return -1;
    return y * w + x;
  };
  return (
    <div className={`relative ${className}`} style={{ width: size, height: (size * h) / w }} title={title}>
      <canvas ref={ref} width={w} height={h} className="pix h-full w-full rounded-sm" />
      {(overlay || alpha) && (
        <div className="pointer-events-none absolute inset-0 grid" style={{ gridTemplateColumns: `repeat(${w},1fr)` }}>
          {Array.from({ length: w * h }, (_, i) => (
            <div
              key={i}
              style={{
                background: alpha ? `rgba(255,181,71,${0.75 * Math.min(1, alpha[i])})` : undefined,
                boxShadow: overlay?.[i] ? 'inset 0 0 0 1px rgba(255,181,71,0.85)' : undefined,
              }}
            />
          ))}
        </div>
      )}
      {onPaint && (
        <div
          className="absolute inset-0 cursor-crosshair"
          onContextMenu={(e) => e.preventDefault()}
          onPointerDown={(e) => { setDrag(true); const i = cell(e); if (i >= 0) onPaint(i, e); }}
          onPointerMove={(e) => { if (drag) { const i = cell(e); if (i >= 0) onPaint(i, e); } }}
          onPointerUp={() => setDrag(false)}
          onPointerLeave={() => setDrag(false)}
        />
      )}
    </div>
  );
}

export function Sources({ onClose }) {
  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 p-4" onClick={onClose}>
      <div className="max-h-[80vh] w-full max-w-2xl overflow-y-auto scroll-thin rounded-lg border border-line bg-panel p-5" onClick={(e) => e.stopPropagation()}>
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-sm font-semibold tracking-wide text-glow">Sources</h2>
          <Btn onClick={onClose}>Close</Btn>
        </div>
        <ol className="space-y-2 text-xs text-slate-300">
          {REF_ORDER.map((k, i) => (
            <li key={k}>
              {i + 1}. {REFS[k].full}{' '}
              {REFS[k].url && (
                <a className="text-glow/80 hover:text-glow" href={REFS[k].url} target="_blank" rel="noreferrer">link</a>
              )}
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}

export class Boundary extends React.Component {
  state = { err: null };
  static getDerivedStateFromError(err) { return { err }; }
  render() {
    if (this.state.err)
      return <div className="rounded border border-red-900 p-2 text-xs text-red-300">This panel hit an error: {String(this.state.err.message)}</div>;
    return this.props.children;
  }
}

// Why this project exists, in the author's words. Shown on the opening page and in About.
export function AboutBlurb({ compact }) {
  return (
    <div className={compact ? 'space-y-2' : 'space-y-2 rounded-lg border border-line bg-panel p-4'}>
      <h1 className="text-lg font-semibold text-slate-100">
        <span className="tracking-[0.2em] text-glow">INANN</span>: in a neural network!
      </h1>
      <p className="text-sm leading-relaxed text-slate-300">
        I started inann after I learned the Perceptron Learning Algorithm (PLA) in my Learning Systems class. I could watch a perceptron learn. Each mistake
        nudges one line, and I could see every nudge. Real networks make the same kind of update across thousands of weights, and the steps disappear from
        view. Interpretability research brings those steps back into view, and I want to do that work. With inann you can walk through a trained network, see
        what each unit responds to, test what it causes, and split a layer into readable features, on a small model or on your own.
      </p>
    </div>
  );
}

export function About({ onClose }) {
  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center bg-black/70 p-4" onClick={onClose}>
      <div className="w-full max-w-xl rounded-lg border border-line bg-panel p-5" onClick={(e) => e.stopPropagation()}>
        <AboutBlurb compact />
        <div className="mt-3 text-right"><Btn onClick={onClose}>Close</Btn></div>
      </div>
    </div>
  );
}
