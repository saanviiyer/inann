// Lecture 14: support vector machines. Drag points, see the widest-margin line and its support vectors.
import React, { useMemo, useState } from 'react';
import { Plane, Slides, Lead, Slider, Stat, Controls, Btn, boundaryLine, sign } from '../common.jsx';

const DEFAULT = [
  { x: -0.6, y: 0.3, label: 1 }, { x: -0.4, y: 0.6, label: 1 }, { x: -0.7, y: -0.1, label: 1 }, { x: -0.2, y: 0.2, label: 1 },
  { x: 0.5, y: -0.3, label: -1 }, { x: 0.3, y: -0.6, label: -1 }, { x: 0.7, y: 0.0, label: -1 }, { x: 0.2, y: -0.2, label: -1 },
];

// Pegasos: subgradient descent on (lambda/2)||w||^2 + mean hinge. Smaller lambda = harder margin.
function svm(pts, lambda) {
  let w = [0, 0], b = 0;
  const N = pts.length || 1;
  for (let t = 1; t <= 3000; t++) {
    const eta = 1 / (lambda * t);
    let gw = [lambda * w[0], lambda * w[1]], gb = 0;
    for (const p of pts) {
      const m = p.label * (w[0] * p.x + w[1] * p.y + b);
      if (m < 1) { gw[0] -= (p.label * p.x) / N; gw[1] -= (p.label * p.y) / N; gb -= p.label / N; }
    }
    w = [w[0] - eta * gw[0], w[1] - eta * gw[1]];
    b -= eta * gb;
  }
  return [w[0], w[1], b];
}

export default function Lesson14() {
  const [pts, setPts] = useState(DEFAULT);
  const [logLam, setLogLam] = useState(-2.3);
  const lambda = 10 ** logLam;

  const w = useMemo(() => svm(pts, lambda), [pts, lambda]);
  const h = (x, y) => sign(w[0] * x + w[1] * y + w[2]);
  const norm = Math.hypot(w[0], w[1]) || 1;
  const margin = 1 / norm;
  const svs = pts.filter((p) => p.label * (w[0] * p.x + w[1] * p.y + w[2]) <= 1.05);

  return (
    <div className="space-y-4">
      <Lead>
        <p>many lines separate two classes. the support vector machine picks the one with the widest margin, the biggest gap to the nearest points on either side. a fat margin is a safer bet on unseen data.</p>
        <p>drag the points and watch the solid boundary and its two dashed margin lines re-solve. only a few points touch the margin; those are the support vectors (ringed). move a point far from the gap and nothing changes, move a support vector and everything does.</p>
      </Lead>
      <div className="flex flex-col gap-4 lg:flex-row">
        <Plane
          points={pts}
          setPoints={setPts}
          classify={h}
          lines={[
            { ...boundaryLine(w), color: 'var(--color-glow)', width: 2.5 },
            { ...boundaryLine([w[0], w[1], w[2] - 1]), color: 'var(--color-dim, #888)', width: 1, dash: '4 4' },
            { ...boundaryLine([w[0], w[1], w[2] + 1]), color: 'var(--color-dim, #888)', width: 1, dash: '4 4' },
          ]}
          marks={svs.map((p) => ({ x: p.x, y: p.y }))}
          addLabel={1}
        />
        <div className="flex-1 space-y-3">
          <Slider label="log10 lambda (smaller = harder margin)" value={logLam} min={-4} max={0} step={0.1} onChange={setLogLam} fmt={(v) => v.toFixed(1)} />
          <Controls>
            <Btn onClick={() => setPts(DEFAULT)}>reset points</Btn>
            <span className="text-[11px] text-dim">click to add +1, right-click to remove</span>
          </Controls>
          <div className="grid grid-cols-2 gap-2">
            <Stat label="margin width" value={margin.toFixed(2)} tone="glow" />
            <Stat label="support vectors" value={svs.length} tone="warm" />
          </div>
          <p className="text-xs leading-relaxed text-slate-400">
            the margin width is 1/||w||, so the solver really is maximizing that gap. with a softer margin (bigger lambda) it tolerates a point or two inside the gap, which helps when the classes overlap.
          </p>
          <p className="text-[11px] text-dim">solved in-browser by 3000 pegasos subgradient steps, so a very hard margin on overlapping points may wobble slightly.</p>
        </div>
      </div>
      <Slides n={14} />
    </div>
  );
}
