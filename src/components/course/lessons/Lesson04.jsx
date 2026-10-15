// Lecture 4: error and noise. Move the target and the noise level, watch E_in and E_out react.
import React, { useMemo, useState } from 'react';
import { Plane, Slides, Lead, Slider, Stat, Controls, Btn, boundaryLine, lstsq, sign, rng } from '../common.jsx';

export default function Lesson04() {
  const [angle, setAngle] = useState(40); // target line orientation, degrees
  const [offset, setOffset] = useState(0.0); // target line offset
  const [noise, setNoise] = useState(0.1); // label flip probability
  const [seed, setSeed] = useState(1);

  const a = (angle * Math.PI) / 180;
  const wf = [Math.cos(a), Math.sin(a), offset]; // target weights
  const target = (x, y) => sign(wf[0] * x + wf[1] * y + wf[2]);

  const pts = useMemo(() => {
    const r = rng(seed * 13 + 1);
    return Array.from({ length: 80 }, () => {
      const x = (r() * 2 - 1) * 0.95, y = (r() * 2 - 1) * 0.95;
      let label = target(x, y);
      if (r() < noise) label = -label;
      return { x, y, label };
    });
  }, [seed, angle, offset, noise]);

  // learner: least-squares linear fit on the noisy sample, then threshold
  const wh = useMemo(() => lstsq(pts.map((p) => [p.x, p.y, 1]), pts.map((p) => p.label)), [pts]);
  const h = (x, y) => sign(wh[0] * x + wh[1] * y + wh[2]);

  const eIn = pts.filter((p) => h(p.x, p.y) !== p.label).length / pts.length;
  // E_out from the disagreement area between h and f, lifted by the noise floor
  const eOut = useMemo(() => {
    const r = rng(777);
    let dis = 0; const M = 5000;
    for (let i = 0; i < M; i++) {
      const x = r() * 2 - 1, y = r() * 2 - 1;
      if (h(x, y) !== target(x, y)) dis++;
    }
    const d = dis / M;
    return d * (1 - noise) + (1 - d) * noise;
  }, [wh, angle, offset, noise]);

  return (
    <div className="space-y-4">
      <Lead>
        <p>the target f is the rule the world actually uses. noise means the labels we see are sometimes flipped, so even a perfect learner cannot be right every time.</p>
        <p>move the target and turn up the noise. E_in is how often the learned line (glow) disagrees with the sample it saw; E_out is how often it would disagree out in the world. notice E_out never drops below the noise level, because that error is baked into the labels.</p>
      </Lead>
      <div className="flex flex-col gap-4 lg:flex-row">
        <Plane
          points={pts}
          classify={h}
          lines={[
            { ...boundaryLine(wf), color: 'var(--ink)', width: 2.5 },
            { ...boundaryLine(wh), color: 'var(--color-glow)', width: 2, dash: '5 4' },
          ]}
          interactive={false}
        />
        <div className="flex-1 space-y-3">
          <Slider label="target orientation" value={angle} min={0} max={180} step={1} onChange={setAngle} fmt={(v) => `${v} deg`} />
          <Slider label="target offset" value={offset} min={-0.8} max={0.8} step={0.02} onChange={setOffset} fmt={(v) => v.toFixed(2)} />
          <Slider label="noise (label flip rate)" value={noise} min={0} max={0.5} step={0.01} onChange={setNoise} fmt={(v) => v.toFixed(2)} />
          <Controls><Btn onClick={() => setSeed((s) => s + 1)}>resample data</Btn></Controls>
          <div className="grid grid-cols-3 gap-2">
            <Stat label="E_in" value={eIn.toFixed(2)} tone="glow" />
            <Stat label="E_out" value={eOut.toFixed(2)} tone="warm" />
            <Stat label="noise floor" value={noise.toFixed(2)} />
          </div>
          <p className="text-xs leading-relaxed text-slate-400">
            the solid line is the target, the dashed line is what the learner recovered from noisy labels. E_out is estimated over 5000 fresh inputs, so it moves even when E_in happens to look good.
          </p>
        </div>
      </div>
      <Slides n={4} />
    </div>
  );
}
