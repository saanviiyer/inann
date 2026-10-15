// Lecture 6: theory of generalization. Growth function against 2^N, and where the break point bites.
import React, { useMemo, useState } from 'react';
import { Slides, Lead, Stat, Controls, Btn, LinePlot } from '../common.jsx';

// exact growth functions for a few hypothesis sets, plus their break point k
const SETS = {
  rays: { name: 'positive rays', k: 2, m: (N) => N + 1 },
  intervals: { name: 'positive intervals', k: 3, m: (N) => (N * (N + 1)) / 2 + 1 },
  convex: { name: 'convex sets', k: null, m: (N) => 2 ** N },
};
const choose = (n, r) => { if (r < 0 || r > n) return 0; let c = 1; for (let i = 0; i < r; i++) c = (c * (n - i)) / (i + 1); return c; };
// Sauer bound once a break point k exists
const sauer = (N, k) => { let s = 0; for (let i = 0; i < k; i++) s += choose(N, i); return s; };

export default function Lesson06() {
  const [set, setSet] = useState('intervals');
  const [Nmax, setNmax] = useState(12);
  const S = SETS[set];

  const series = useMemo(() => {
    const ns = Array.from({ length: Nmax }, (_, i) => i + 1);
    const out = [
      { label: '2^N', color: 'var(--color-warm)', pts: ns.map((N) => [N, 2 ** N]) },
      { label: 'growth function', color: 'var(--color-glow)', pts: ns.map((N) => [N, S.m(N)]) },
    ];
    if (S.k) out.push({ label: `Sauer bound (k=${S.k})`, color: 'var(--color-cool)', dash: '5 4', pts: ns.map((N) => [N, sauer(N, S.k)]) });
    return out;
  }, [set, Nmax]);

  return (
    <div className="space-y-4">
      <Lead>
        <p>a break point k is the first number of points the hypothesis set cannot shatter: at n=k it already fails to make some labeling. the surprising theorem of this lecture is that one break point forces the whole growth function to be polynomial, not exponential.</p>
        <p>pick a set and compare its growth function (glow) to 2^n (amber). once there is a break point, the growth function is trapped under the Sauer bound, a polynomial of degree k-1. convex sets have no break point, so they track 2^n exactly and cannot generalize.</p>
      </Lead>
      <Controls>
        {Object.entries(SETS).map(([k, v]) => (
          <Btn key={k} on={set === k} onClick={() => setSet(k)}>{v.name}</Btn>
        ))}
      </Controls>
      <LinePlot series={series} xlabel="N points" ylabel="patterns (log scale)" logy yDomain={[1, 2 ** Nmax]} />
      <div className="grid grid-cols-3 gap-2">
        <Stat label="break point k" value={S.k ?? 'none'} tone="glow" />
        <Stat label={`growth at N=${Nmax}`} value={S.m(Nmax).toLocaleString()} tone="warm" />
        <Stat label={`2^${Nmax}`} value={(2 ** Nmax).toLocaleString()} />
      </div>
      <p className="text-xs leading-relaxed text-slate-400">
        on the log axis, 2^n is a straight climb while a set with a break point bends flat. that flattening is the entire reason a finite sample can say anything about unseen data.
      </p>
      <Slides n={6} />
    </div>
  );
}
