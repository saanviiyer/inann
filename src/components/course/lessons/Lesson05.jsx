// Lecture 5: training versus testing. Dichotomies and the growth function, shown on positive rays.
import React, { useMemo, useState } from 'react';
import { useTheme, sceneHex } from '../../../lib/theme.js';
import { Slides, Lead, Slider, Stat } from '../common.jsx';

export default function Lesson05() {
  const [N, setN] = useState(4);
  const [thr, setThr] = useState(0.0); // threshold position of the positive ray
  const theme = useTheme();
  const hex = sceneHex(theme);

  const xs = useMemo(() => Array.from({ length: N }, (_, i) => -0.85 + (1.7 * (i + 0.5)) / N), [N]);
  const labels = xs.map((x) => (x > thr ? 1 : -1));

  // a positive ray h(x)=+1 for x>thr can only produce these N+1 patterns
  const patterns = useMemo(() => {
    const out = [];
    for (let k = 0; k <= N; k++) out.push(xs.map((_, i) => (i >= k ? 1 : -1)));
    return out;
  }, [xs, N]);
  const total = 2 ** N;

  const W = 340, H = 60, X = (x) => ((x + 1) / 2) * W;

  return (
    <div className="space-y-4">
      <Lead>
        <p>testing used one hypothesis. training picks one out of many, so the real question is how many distinct label patterns a hypothesis set can even produce on n points. that count is the growth function m_H(n).</p>
        <p>here the set is positive rays: pick a threshold, everything to its right is +1. slide the threshold and watch the pattern change. no matter where you put it, you can only ever make n+1 of the 2^n patterns. that gap is what makes generalization possible.</p>
      </Lead>
      <div className="space-y-3">
        <svg viewBox={`0 0 ${W} ${H}`} className="w-full rounded-lg border border-line bg-[var(--pad)]" role="img">
          <line x1={0} y1={H / 2} x2={W} y2={H / 2} stroke={hex.axis} strokeWidth="2" />
          <line x1={X(thr)} y1={8} x2={X(thr)} y2={H - 8} stroke={hex.glowText} strokeWidth="2" strokeDasharray="4 3" />
          {xs.map((x, i) => (
            <circle key={i} cx={X(x)} cy={H / 2} r={7} fill={labels[i] > 0 ? hex.accent : hex.cool} stroke={hex.bg} strokeWidth="1.5" />
          ))}
        </svg>
        <Slider label="threshold" value={thr} min={-1} max={1} step={0.01} onChange={setThr} fmt={(v) => v.toFixed(2)} />
        <Slider label="n (number of points)" value={N} min={2} max={8} step={1} onChange={setN} />
        <div className="grid grid-cols-3 gap-2">
          <Stat label="patterns possible, m_H(n)" value={N + 1} tone="glow" />
          <Stat label="all patterns, 2^n" value={total} tone="warm" />
          <Stat label="fraction reachable" value={`${(((N + 1) / total) * 100).toFixed(0)}%`} />
        </div>
        <div>
          <p className="mb-1 text-xs text-dim">the only {N + 1} dichotomies positive rays can make:</p>
          <div className="flex flex-wrap gap-2">
            {patterns.map((pat, k) => (
              <div key={k} className="flex gap-1 rounded border border-line p-1">
                {pat.map((l, i) => (
                  <span key={i} className="h-3 w-3 rounded-sm" style={{ background: l > 0 ? hex.accent : hex.cool }} />
                ))}
              </div>
            ))}
          </div>
        </div>
        <p className="text-xs leading-relaxed text-slate-400">
          2^n grows exponentially, but positive rays stay linear in n. a set that cannot reach every pattern is a set that can be trusted to generalize.
        </p>
      </div>
      <Slides n={5} />
    </div>
  );
}
