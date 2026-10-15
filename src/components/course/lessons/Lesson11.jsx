// Lecture 11: overfitting. Raise the polynomial degree and watch E_out blow up while E_in keeps falling.
import React, { useMemo, useState } from 'react';
import { Slides, Lead, Slider, Stat, Controls, Btn, LinePlot, lstsq, gauss, rng } from '../common.jsx';

const target = (x) => 0.5 * x * x * x - 0.8 * x; // the clean rule we are trying to recover
const vander = (x, d) => Array.from({ length: d + 1 }, (_, k) => x ** k);
const PLOT = Array.from({ length: 61 }, (_, i) => -1 + (2 * i) / 60);

export default function Lesson11() {
  const [degree, setDegree] = useState(3);
  const [N, setN] = useState(12);
  const [noise, setNoise] = useState(0.2);
  const [seed, setSeed] = useState(1);

  const data = useMemo(() => {
    const r = rng(seed * 31 + 5);
    return Array.from({ length: N }, () => { const x = r() * 2 - 1; return { x, y: target(x) + noise * gauss(r) }; });
  }, [N, noise, seed]);

  const w = useMemo(() => lstsq(data.map((p) => vander(p.x, degree)), data.map((p) => p.y), 1e-9), [data, degree]);
  const fit = (x) => vander(x, degree).reduce((s, z, i) => s + z * w[i], 0);

  const eIn = data.reduce((s, p) => s + (fit(p.x) - p.y) ** 2, 0) / data.length;
  const eOut = PLOT.reduce((s, x) => s + (fit(x) - target(x)) ** 2, 0) / PLOT.length;

  const series = [
    { label: 'target', color: 'var(--color-warm)', pts: PLOT.map((x) => [x, target(x)]) },
    { label: `degree-${degree} fit`, color: 'var(--color-glow)', pts: PLOT.map((x) => [x, fit(x)]) },
  ];

  return (
    <div className="space-y-4">
      <Lead>
        <p>overfitting is fitting the noise. a flexible model can thread every training point exactly, driving E_in to nothing, and still be badly wrong between them.</p>
        <p>raise the degree past what the data supports, or shrink the sample, or add noise, and watch the glow curve wiggle away from the amber target. E_in keeps dropping while E_out climbs. that growing gap is the whole danger.</p>
      </Lead>
      <LinePlot series={series} xlabel="x" ylabel="y" xDomain={[-1, 1]} yDomain={[-2, 2]} />
      <div className="space-y-2">
        <Slider label="polynomial degree" value={degree} min={1} max={14} step={1} onChange={setDegree} />
        <Slider label="training points" value={N} min={4} max={40} step={1} onChange={setN} />
        <Slider label="noise" value={noise} min={0} max={0.6} step={0.02} onChange={setNoise} fmt={(v) => v.toFixed(2)} />
        <Controls><Btn onClick={() => setSeed((s) => s + 1)}>resample</Btn></Controls>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <Stat label="E_in (on the sample)" value={eIn.toFixed(3)} tone="glow" />
        <Stat label="E_out (vs the target)" value={eOut.toFixed(3)} tone="warm" />
      </div>
      <p className="text-xs leading-relaxed text-slate-400">
        try degree 12 with only 6 points: E_in is tiny, E_out is enormous. the fix is next, two of them: regularization and validation.
      </p>
      <Slides n={11} />
    </div>
  );
}
