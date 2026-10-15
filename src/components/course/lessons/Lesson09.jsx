// Lecture 9: the linear model II. A nonlinear feature transform makes a line bend into a circle.
import React, { useMemo, useState } from 'react';
import { Plane, Slides, Lead, Stat, Controls, Btn, lstsq, sign, rng } from '../common.jsx';

export default function Lesson09() {
  const [transform, setTransform] = useState('circle'); // 'none' or 'circle'
  const [seed, setSeed] = useState(1);

  const pts = useMemo(() => {
    const r = rng(seed * 9 + 2);
    return Array.from({ length: 80 }, () => {
      const x = (r() * 2 - 1) * 0.95, y = (r() * 2 - 1) * 0.95;
      const inside = x * x + y * y < 0.4;
      let label = inside ? 1 : -1;
      if (r() < 0.05) label = -label;
      return { x, y, label };
    });
  }, [seed]);

  const phi = (x, y) => (transform === 'none' ? [x, y, 1] : [x, y, x * x + y * y, 1]);
  const w = useMemo(() => lstsq(pts.map((p) => phi(p.x, p.y)), pts.map((p) => p.label)), [pts, transform]);
  const h = (x, y) => sign(phi(x, y).reduce((s, z, i) => s + z * w[i], 0));
  const eIn = pts.filter((p) => h(p.x, p.y) !== p.label).length / pts.length;

  return (
    <div className="space-y-4">
      <Lead>
        <p>these dots cannot be split by a straight line: one colour sits in a disk inside the other. a linear model in x-space is stuck.</p>
        <p>but linear only has to mean linear in the features. map each point to (x1, x2, x1^2 + x2^2) and fit a line in that bigger space. back in the original plane that line is a circle. the model is still a linear model; we only changed what we feed it.</p>
      </Lead>
      <div className="flex flex-col gap-4 lg:flex-row">
        <Plane points={pts} classify={h} interactive={false} />
        <div className="flex-1 space-y-3">
          <Controls>
            <Btn on={transform === 'none'} onClick={() => setTransform('none')}>no transform</Btn>
            <Btn on={transform === 'circle'} onClick={() => setTransform('circle')}>add x1^2 + x2^2</Btn>
            <Btn onClick={() => setSeed((s) => s + 1)}>new data</Btn>
          </Controls>
          <div className="grid grid-cols-2 gap-2">
            <Stat label="features fed to the line" value={transform === 'none' ? '2' : '3'} />
            <Stat label="E_in" value={eIn.toFixed(2)} tone={eIn < 0.1 ? 'glow' : 'warm'} />
          </div>
          <p className="text-xs leading-relaxed text-slate-400">
            with no transform the best line gets about half the points wrong. add the squared-radius feature and the boundary becomes a circle that fits. the catch, saved for lecture 11: every feature you add costs generalization, so you pay for the bend.
          </p>
        </div>
      </div>
      <Slides n={9} />
    </div>
  );
}
