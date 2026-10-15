// Lecture 7: the VC dimension. Try to shatter points with a 2D perceptron; it breaks at d_vc+1 = 4.
import React, { useMemo, useState } from 'react';
import { Plane, Slides, Lead, Stat, Controls, Btn, sign } from '../common.jsx';

// is a labeling of these 2D points linearly separable? iterative perceptron check, capped.
function separable(pts, labeling) {
  let w = [0, 0, 0];
  for (let it = 0; it < 4000; it++) {
    let err = 0;
    for (let i = 0; i < pts.length; i++) {
      const t = labeling[i] ? 1 : -1;
      if (sign(w[0] * pts[i].x + w[1] * pts[i].y + w[2]) !== t) {
        w = [w[0] + t * pts[i].x, w[1] + t * pts[i].y, w[2] + t];
        err++;
      }
    }
    if (err === 0) return true;
  }
  return false;
}

const DEFAULT = [
  { x: -0.5, y: -0.4, label: 1 },
  { x: 0.5, y: -0.4, label: 1 },
  { x: 0, y: 0.5, label: 1 },
];

export default function Lesson07() {
  const [pts, setPts] = useState(DEFAULT);
  const [tested, setTested] = useState(null);

  const result = useMemo(() => {
    const N = pts.length;
    if (N === 0 || N > 7) return null;
    let ok = 0;
    const total = 2 ** N;
    for (let m = 0; m < total; m++) {
      const labeling = pts.map((_, i) => (m >> i) & 1);
      if (separable(pts, labeling)) ok++;
    }
    return { ok, total, shattered: ok === total };
  }, [pts, tested]);

  return (
    <div className="space-y-4">
      <Lead>
        <p>the VC dimension d_vc is the most points a hypothesis set can shatter, meaning realize every one of their 2^n labelings. for a line in the plane it is 3.</p>
        <p>place points (click to add, drag to move, right-click to remove), then shatter them. the tool tries all 2^n labelings and, for each, searches for a separating line. with 3 points in general position every labeling works. add a 4th and some labeling (the xor-like one) can never be separated, so the line breaks at 4 = d_vc + 1.</p>
      </Lead>
      <div className="flex flex-col gap-4 lg:flex-row">
        <Plane points={pts} setPoints={setPts} addLabel={1} />
        <div className="flex-1 space-y-3">
          <Controls>
            <Btn onClick={() => setTested((t) => (t || 0) + 1)}>test shattering</Btn>
            <Btn onClick={() => setPts(DEFAULT)}>reset to 3 points</Btn>
            <Btn onClick={() => setPts([...pts, { x: 0, y: -0.5, label: 1 }])}>add a 4th point</Btn>
          </Controls>
          {result ? (
            <>
              <div className="grid grid-cols-2 gap-2">
                <Stat label="separable labelings" value={`${result.ok}/${result.total}`} tone="glow" />
                <Stat label="fully shattered?" value={result.shattered ? 'yes' : 'no'} tone={result.shattered ? 'glow' : 'warm'} />
              </div>
              <p className="text-xs leading-relaxed text-slate-400">
                {result.shattered
                  ? `all ${result.total} labelings of these ${pts.length} points are linearly separable. the line still has room.`
                  : `${result.total - result.ok} of ${result.total} labelings cannot be separated by any line. this set is not shattered, so d_vc is below ${pts.length}.`}
              </p>
            </>
          ) : (
            <p className="text-xs text-dim">place between 1 and 7 points, then test.</p>
          )}
          <p className="text-[11px] text-dim">the separability search is an iterative perceptron capped at 4000 passes, so a labeling it cannot separate within that budget is reported as not separable.</p>
        </div>
      </div>
      <Slides n={7} />
    </div>
  );
}
