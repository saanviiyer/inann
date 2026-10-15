// Lecture 8: bias-variance. Resample tiny datasets, fit, accumulate the average hypothesis.
import React, { useEffect, useRef, useState } from 'react';
import { Slides, Lead, Stat, Controls, Btn, LinePlot, rng } from '../common.jsx';

const GRID = Array.from({ length: 41 }, (_, i) => -1 + (2 * i) / 40);
const f = (x) => Math.sin(Math.PI * x);

export default function Lesson08() {
  const [model, setModel] = useState('line'); // 'const' = h(x)=b, 'line' = h(x)=ax+b
  const [running, setRunning] = useState(false);
  const [snap, setSnap] = useState(null);
  const acc = useRef(null);
  const r = useRef(rng(42));
  const timer = useRef(null);

  const resetAcc = () => {
    acc.current = { sum: new Float64Array(41), sq: new Float64Array(41), n: 0, lastFit: null };
    r.current = rng(42 + Math.floor(Math.random() * 1000));
    setSnap(null);
  };
  useEffect(resetAcc, [model]);

  const addBatch = (count) => {
    const a = acc.current;
    for (let b = 0; b < count; b++) {
      const x1 = r.current() * 2 - 1, x2 = r.current() * 2 - 1;
      const y1 = f(x1), y2 = f(x2);
      let g;
      if (model === 'const') { const c = (y1 + y2) / 2; g = () => c; }
      else { const m = Math.abs(x2 - x1) < 1e-6 ? 0 : (y2 - y1) / (x2 - x1); const c = y1 - m * x1; g = (x) => m * x + c; }
      for (let i = 0; i < 41; i++) { const v = g(GRID[i]); a.sum[i] += v; a.sq[i] += v * v; }
      a.n++;
      a.lastFit = GRID.map((x) => g(x));
    }
    const gbar = GRID.map((_, i) => a.sum[i] / a.n);
    let bias = 0, varc = 0;
    for (let i = 0; i < 41; i++) {
      bias += (gbar[i] - f(GRID[i])) ** 2;
      varc += a.sq[i] / a.n - gbar[i] * gbar[i];
    }
    setSnap({ gbar, std: GRID.map((_, i) => Math.sqrt(Math.max(0, a.sq[i] / a.n - gbar[i] * gbar[i]))), lastFit: a.lastFit, n: a.n, bias: bias / 41, varc: varc / 41 });
  };

  useEffect(() => {
    if (!running) { clearInterval(timer.current); return; }
    timer.current = setInterval(() => addBatch(25), 120);
    return () => clearInterval(timer.current);
  }, [running, model]);

  const series = [];
  series.push({ label: 'target sin(pi x)', color: 'var(--color-warm)', pts: GRID.map((x) => [x, f(x)]) });
  if (snap) {
    series.push({ label: 'average hypothesis', color: 'var(--color-glow)', pts: GRID.map((x, i) => [x, snap.gbar[i]]) });
    series.push({ color: 'var(--color-cool)', dash: '3 3', pts: GRID.map((x, i) => [x, snap.gbar[i] + snap.std[i]]) });
    series.push({ color: 'var(--color-cool)', dash: '3 3', pts: GRID.map((x, i) => [x, snap.gbar[i] - snap.std[i]]) });
    if (snap.lastFit) series.push({ label: 'one fit', color: 'var(--ink)', pts: GRID.map((x, i) => [x, snap.lastFit[i]]) });
  }

  return (
    <div className="space-y-4">
      <Lead>
        <p>fit the same simple model to many tiny datasets (two points each, drawn from the target sin) and average all the fits. that average is gbar. two things pull E_out up: bias, how far gbar sits from the target, and variance, how much each individual fit jiggles around gbar.</p>
        <p>run it and watch gbar (glow) build up with its ±1 std band (blue). the constant model has high bias but low variance; the line has low bias but high variance from only two points. simpler is not always worse.</p>
      </Lead>
      <Controls>
        <Btn on={model === 'const'} onClick={() => setModel('const')}>constant h(x)=b</Btn>
        <Btn on={model === 'line'} onClick={() => setModel('line')}>line h(x)=ax+b</Btn>
      </Controls>
      <LinePlot series={series} xlabel="x" ylabel="y" xDomain={[-1, 1]} yDomain={[-2.5, 2.5]} />
      <Controls>
        <Btn on={running} onClick={() => setRunning((v) => !v)}>{running ? 'pause' : 'resample'}</Btn>
        <Btn onClick={() => { setRunning(false); addBatch(1); }}>one dataset</Btn>
        <Btn onClick={() => { setRunning(false); resetAcc(); }}>reset</Btn>
      </Controls>
      <div className="grid grid-cols-3 gap-2">
        <Stat label="datasets averaged" value={snap?.n ?? 0} />
        <Stat label="bias" value={snap ? snap.bias.toFixed(3) : '-'} tone="glow" />
        <Stat label="variance" value={snap ? snap.varc.toFixed(3) : '-'} tone="warm" />
      </div>
      <p className="text-xs leading-relaxed text-slate-400">
        E_out settles toward bias + variance. switch models and watch the trade: the line's bias is tiny but its variance band is wide, the constant's band is tight but it misses the shape.
      </p>
      <Slides n={8} />
    </div>
  );
}
