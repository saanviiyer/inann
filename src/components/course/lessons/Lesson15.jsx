// Lecture 15: kernel methods. Swap the kernel on non-separable data; the boundary changes shape.
import React, { useMemo, useState } from 'react';
import { Plane, Slides, Lead, Slider, Stat, Controls, Btn, sign, rng } from '../common.jsx';

function makeRing(seed) {
  const r = rng(seed * 5 + 1);
  return Array.from({ length: 60 }, () => {
    const x = (r() * 2 - 1) * 0.95, y = (r() * 2 - 1) * 0.95;
    let label = x * x + y * y < 0.4 ? 1 : -1;
    if (r() < 0.04) label = -label;
    return { x, y, label };
  });
}

const kernels = {
  linear: (a, b) => a.x * b.x + a.y * b.y,
  poly: (a, b) => (1 + a.x * b.x + a.y * b.y) ** 3,
  rbf: (a, b, g) => Math.exp(-g * ((a.x - b.x) ** 2 + (a.y - b.y) ** 2)),
};

export default function Lesson15() {
  const [kernel, setKernel] = useState('rbf');
  const [gamma, setGamma] = useState(4);
  const [seed, setSeed] = useState(1);
  const pts = useMemo(() => makeRing(seed), [seed]);

  // dual (kernel) perceptron: alpha_i counts mistakes on point i
  const alpha = useMemo(() => {
    const K = kernels[kernel];
    const a = new Float64Array(pts.length);
    for (let ep = 0; ep < 40; ep++) {
      let changed = false;
      for (let i = 0; i < pts.length; i++) {
        let s = 0;
        for (let j = 0; j < pts.length; j++) if (a[j]) s += a[j] * pts[j].label * K(pts[j], pts[i], gamma);
        if (sign(s) !== pts[i].label) { a[i] += 1; changed = true; }
      }
      if (!changed) break;
    }
    return a;
  }, [pts, kernel, gamma]);

  const score = (x, y) => {
    const K = kernels[kernel], q = { x, y };
    let s = 0;
    for (let j = 0; j < pts.length; j++) if (alpha[j]) s += alpha[j] * pts[j].label * K(pts[j], q, gamma);
    return s;
  };
  const h = (x, y) => sign(score(x, y));
  const eIn = pts.filter((p) => h(p.x, p.y) !== p.label).length / pts.length;

  return (
    <div className="space-y-4">
      <Lead>
        <p>a kernel is a similarity between two points that secretly equals a dot product in some high-dimensional feature space, without ever building that space. swap the kernel and you swap the shape of boundary the same algorithm can draw.</p>
        <p>the data is a disk inside a ring, which no line can split. the linear kernel fails on it; the polynomial and rbf kernels bend around it. with the rbf kernel, gamma sets how local each point's influence is.</p>
      </Lead>
      <div className="flex flex-col gap-4 lg:flex-row">
        <Plane points={pts} classify={h} interactive={false} grid={28} />
        <div className="flex-1 space-y-3">
          <Controls>
            {Object.keys(kernels).map((k) => (
              <Btn key={k} on={kernel === k} onClick={() => setKernel(k)}>{k}</Btn>
            ))}
          </Controls>
          {kernel === 'rbf' && <Slider label="gamma (rbf width)" value={gamma} min={0.5} max={20} step={0.5} onChange={setGamma} />}
          <Controls><Btn onClick={() => setSeed((s) => s + 1)}>new data</Btn></Controls>
          <div className="grid grid-cols-2 gap-2">
            <Stat label="kernel" value={kernel} />
            <Stat label="E_in" value={eIn.toFixed(2)} tone={eIn < 0.1 ? 'glow' : 'warm'} />
          </div>
          <p className="text-xs leading-relaxed text-slate-400">
            the linear kernel is stuck near E_in = 0.4 because a line cannot carve a ring. the rbf kernel drops it to near zero, but push gamma too high and each point only speaks for itself, which overfits.
          </p>
        </div>
      </div>
      <Slides n={15} />
    </div>
  );
}
