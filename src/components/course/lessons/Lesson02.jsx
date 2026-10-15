// Lecture 2: is learning feasible? Marbles in a bin, Hoeffding's bound against the real deviation.
import React, { useMemo, useState } from 'react';
import { Slides, Lead, Slider, Stat, Controls, Btn, rng } from '../common.jsx';

export default function Lesson02() {
  const [mu, setMu] = useState(0.5); // true fraction of red marbles in the bin
  const [N, setN] = useState(30); // sample size
  const [eps, setEps] = useState(0.2);
  const [seed, setSeed] = useState(1);

  // one visible sample
  const sample = useMemo(() => {
    const r = rng(seed * 7 + 3);
    return Array.from({ length: N }, () => r() < mu);
  }, [mu, N, seed]);
  const nu = sample.filter(Boolean).length / N;

  // estimate P(|nu - mu| > eps) by drawing many samples, and the Hoeffding bound
  const empirical = useMemo(() => {
    const r = rng(seed * 101 + 17);
    const T = 4000;
    let bad = 0;
    for (let t = 0; t < T; t++) {
      let red = 0;
      for (let i = 0; i < N; i++) if (r() < mu) red++;
      if (Math.abs(red / N - mu) > eps) bad++;
    }
    return bad / T;
  }, [mu, N, eps, seed]);
  const bound = Math.min(1, 2 * Math.exp(-2 * eps * eps * N));

  return (
    <div className="space-y-4">
      <Lead>
        <p>a bin holds red and green marbles; the red fraction is mu, which we cannot see. we grab a handful of n marbles and measure their red fraction nu. the worry: does nu say anything about mu?</p>
        <p>hoeffding says yes. the chance that nu is more than eps away from mu is at most 2 e^(-2 eps^2 n). it shrinks fast as the sample grows, and it does not depend on mu. that single inequality is why a training-set error can speak for the error everywhere.</p>
      </Lead>
      <div className="flex flex-col gap-4 lg:flex-row">
        <div className="w-full max-w-[340px]">
          <div className="grid grid-cols-10 gap-1 rounded-lg border border-line bg-[var(--pad)] p-2">
            {sample.map((red, i) => (
              <span key={i} className="aspect-square rounded-full" style={{ background: red ? 'var(--color-warm)' : 'var(--color-cool)' }} />
            ))}
          </div>
          <p className="mt-2 text-center text-xs text-dim">one handful of {N} marbles</p>
        </div>
        <div className="flex-1 space-y-3">
          <Slider label="mu (true red fraction, hidden in practice)" value={mu} min={0} max={1} step={0.01} onChange={setMu} fmt={(v) => v.toFixed(2)} />
          <Slider label="n (sample size)" value={N} min={5} max={200} step={1} onChange={setN} />
          <Slider label="eps (tolerance)" value={eps} min={0.02} max={0.5} step={0.01} onChange={setEps} fmt={(v) => v.toFixed(2)} />
          <Controls><Btn onClick={() => setSeed((s) => s + 1)}>draw a new handful</Btn></Controls>
          <div className="grid grid-cols-2 gap-2">
            <Stat label="nu in this handful" value={nu.toFixed(2)} tone="glow" />
            <Stat label="|nu - mu|" value={Math.abs(nu - mu).toFixed(2)} tone="warm" />
            <Stat label="real P(|nu-mu|>eps)" value={empirical.toFixed(3)} tone="glow" />
            <Stat label="hoeffding bound" value={bound.toFixed(3)} tone="warm" />
          </div>
          <p className="text-xs leading-relaxed text-slate-400">
            the real deviation probability (measured over 4000 handfuls) always sits under the bound. the bound is loose, but it holds for every mu, which is the point.
          </p>
        </div>
      </div>
      <Slides n={2} />
    </div>
  );
}
