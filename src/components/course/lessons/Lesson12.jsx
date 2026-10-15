// Lecture 12: regularization. A weight-decay lambda dial on the overfitting fit; the curve stiffens.
import React, { useMemo, useState } from 'react';
import { Slides, Lead, Slider, Stat, Controls, Btn, LinePlot, lstsq, gauss, rng } from '../common.jsx';

const target = (x) => 0.5 * x * x * x - 0.8 * x;
const vander = (x, d) => Array.from({ length: d + 1 }, (_, k) => x ** k);
const PLOT = Array.from({ length: 61 }, (_, i) => -1 + (2 * i) / 60);
const DEG = 12; // deliberately too flexible, so regularization has something to tame

export default function Lesson12() {
  const [logLam, setLogLam] = useState(-4); // lambda = 10^logLam
  const [seed, setSeed] = useState(1);
  const lam = 10 ** logLam;

  const data = useMemo(() => {
    const r = rng(seed * 17 + 3);
    return Array.from({ length: 11 }, () => { const x = r() * 2 - 1; return { x, y: target(x) + 0.2 * gauss(r) }; });
  }, [seed]);

  const fitAt = (ridge) => {
    const w = lstsq(data.map((p) => vander(p.x, DEG)), data.map((p) => p.y), ridge + 1e-9);
    return (x) => vander(x, DEG).reduce((s, z, i) => s + z * w[i], 0);
  };
  const fit = useMemo(() => fitAt(lam), [data, lam]);
  const eOut = PLOT.reduce((s, x) => s + (fit(x) - target(x)) ** 2, 0) / PLOT.length;
  const eIn = data.reduce((s, p) => s + (fit(p.x) - p.y) ** 2, 0) / data.length;

  // E_out across lambda, to show the sweet spot
  const sweep = useMemo(() => {
    const out = [];
    for (let l = -6; l <= 2; l += 0.25) {
      const g = fitAt(10 ** l);
      out.push([l, PLOT.reduce((s, x) => s + (g(x) - target(x)) ** 2, 0) / PLOT.length]);
    }
    return out;
  }, [data]);

  const fitSeries = [
    { label: 'target', color: 'var(--color-warm)', pts: PLOT.map((x) => [x, target(x)]) },
    { label: 'regularized fit', color: 'var(--color-glow)', pts: PLOT.map((x) => [x, fit(x)]) },
  ];

  return (
    <div className="space-y-4">
      <Lead>
        <p>regularization fights overfitting by charging for complexity. weight decay adds lambda times the size of the weights to what we minimize, so big wiggly coefficients are no longer free.</p>
        <p>this is the same degree-12 fit from the last lesson. turn lambda up and watch the glow curve stop chasing every point and stiffen toward the smooth target. too much, though, and it goes flat and misses the shape, so there is a sweet spot.</p>
      </Lead>
      <LinePlot series={fitSeries} xlabel="x" ylabel="y" xDomain={[-1, 1]} yDomain={[-2, 2]} />
      <Slider label="log10 lambda" value={logLam} min={-6} max={2} step={0.25} onChange={setLogLam} fmt={(v) => `${v.toFixed(2)}  (lambda=${(10 ** v).toExponential(1)})`} />
      <Controls><Btn onClick={() => setSeed((s) => s + 1)}>resample</Btn></Controls>
      <div className="grid grid-cols-2 gap-2">
        <Stat label="E_in" value={eIn.toFixed(3)} tone="glow" />
        <Stat label="E_out" value={eOut.toFixed(3)} tone="warm" />
      </div>
      <LinePlot
        series={[{ label: 'E_out vs lambda', color: 'var(--color-cool)', pts: sweep }, { color: 'var(--color-glow)', pts: [[logLam, 0], [logLam, Math.max(...sweep.map((s) => s[1]))]], dash: '3 3' }]}
        xlabel="log10 lambda" ylabel="E_out" height={160}
      />
      <p className="text-xs leading-relaxed text-slate-400">
        the lower curve is E_out as lambda sweeps; the dashed marker is where you are. the minimum is neither zero nor huge: a little regularization helps, too much hurts.
      </p>
      <Slides n={12} />
    </div>
  );
}
