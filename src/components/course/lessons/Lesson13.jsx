// Lecture 13: validation. Split the data into k folds and use the held-out error to estimate E_out.
import React, { useMemo, useState } from 'react';
import { useTheme, sceneHex } from '../../../lib/theme.js';
import { Slides, Lead, Slider, Stat, LinePlot, lstsq, gauss, rng } from '../common.jsx';

const target = (x) => 0.5 * x * x * x - 0.8 * x;
const vander = (x, d) => Array.from({ length: d + 1 }, (_, k) => x ** k);
const PLOT = Array.from({ length: 61 }, (_, i) => -1 + (2 * i) / 60);
const FOLDCOL = ['#7ee7ff', '#ffb547', '#5b8cff', '#c56a12', '#8af0ff', '#d98a2b'];

export default function Lesson13() {
  const [k, setK] = useState(4);
  const [degree, setDegree] = useState(3);
  const hex = sceneHex(useTheme());

  const data = useMemo(() => {
    const r = rng(99);
    return Array.from({ length: 24 }, () => { const x = r() * 2 - 1; return { x, y: target(x) + 0.25 * gauss(r) }; });
  }, []);

  const fitPoly = (rows, d) => {
    const w = lstsq(rows.map((p) => vander(p.x, d)), rows.map((p) => p.y), 1e-9);
    return (x) => vander(x, d).reduce((s, z, i) => s + z * w[i], 0);
  };

  const cvError = (d, kk) => {
    let se = 0;
    for (let f = 0; f < kk; f++) {
      const train = data.filter((_, i) => i % kk !== f);
      const test = data.filter((_, i) => i % kk === f);
      const g = fitPoly(train, d);
      for (const p of test) se += (g(p.x) - p.y) ** 2;
    }
    return se / data.length;
  };
  const trueEout = (d) => { const g = fitPoly(data, d); return PLOT.reduce((s, x) => s + (g(x) - target(x)) ** 2, 0) / PLOT.length; };

  const cv = cvError(degree, k);
  const eout = trueEout(degree);
  const sweep = useMemo(() => Array.from({ length: 9 }, (_, d) => ({ d, cv: cvError(d, k), eout: trueEout(d) })), [k]);
  const best = sweep.reduce((a, b) => (b.cv < a.cv ? b : a), sweep[0]);

  const full = fitPoly(data, degree);
  const series = [
    { label: 'target', color: 'var(--color-warm)', pts: PLOT.map((x) => [x, target(x)]) },
    { label: `degree-${degree} fit`, color: 'var(--color-glow)', pts: PLOT.map((x) => [x, full(x)]) },
  ];

  return (
    <div className="space-y-4">
      <Lead>
        <p>we cannot see E_out, but we can hide part of the data, train on the rest, and test on what we hid. k-fold cross-validation does this k times so every point gets a turn as the held-out test, then averages the errors.</p>
        <p>change k and the model degree. the validation estimate tracks the true E_out without ever peeking at the target, which is why it can choose the degree for you. the sweep below picks the degree with the smallest validation error.</p>
      </Lead>
      <div className="flex flex-wrap gap-1 rounded-lg border border-line bg-[var(--pad)] p-2">
        {data.map((p, i) => (
          <span key={i} title={`fold ${i % k}`} className="h-4 w-4 rounded-sm" style={{ background: FOLDCOL[(i % k) % FOLDCOL.length] }} />
        ))}
        <span className="ml-2 self-center text-[11px] text-dim">each square is a point, coloured by its fold</span>
      </div>
      <LinePlot series={series} xlabel="x" ylabel="y" xDomain={[-1, 1]} yDomain={[-2, 2]} />
      <div className="space-y-2">
        <Slider label="k (folds)" value={k} min={2} max={12} step={1} onChange={setK} />
        <Slider label="polynomial degree" value={degree} min={0} max={8} step={1} onChange={setDegree} />
      </div>
      <div className="grid grid-cols-2 gap-2">
        <Stat label="validation estimate" value={cv.toFixed(3)} tone="glow" />
        <Stat label="true E_out" value={eout.toFixed(3)} tone="warm" />
      </div>
      <div>
        <p className="mb-1 text-xs text-dim">validation error by degree (k={k}). the bar it chooses is degree {best.d}.</p>
        <div className="flex items-end gap-1" style={{ height: 90 }}>
          {sweep.map((s) => {
            const max = Math.max(...sweep.map((z) => z.cv));
            return (
              <div key={s.d} className="flex flex-1 flex-col items-center justify-end">
                <div className="w-full rounded-t" style={{ height: `${(s.cv / max) * 70}px`, background: s.d === best.d ? hex.glowText : hex.cool, opacity: s.d === degree ? 1 : 0.6 }} />
                <span className="text-[10px] text-dim">{s.d}</span>
              </div>
            );
          })}
        </div>
      </div>
      <p className="text-xs leading-relaxed text-slate-400">
        more folds means more training data per fit and a steadier estimate, at more compute. leave-one-out is k = n.
      </p>
      <Slides n={13} />
    </div>
  );
}
