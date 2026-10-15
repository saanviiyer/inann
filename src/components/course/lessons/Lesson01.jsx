// Lecture 1: the learning problem. Label points, then let the perceptron learn a boundary.
import React, { useEffect, useRef, useState } from 'react';
import { Plane, Slides, Lead, Controls, Btn, Stat, boundaryLine, sign, rng } from '../common.jsx';

function seedData() {
  const r = rng(11);
  const pts = [];
  while (pts.length < 10) {
    const x = (r() * 2 - 1) * 0.9, y = (r() * 2 - 1) * 0.9;
    pts.push({ x, y, label: 0.9 * x - 0.5 * y + 0.1 > 0 ? 1 : -1 });
  }
  return pts;
}

export default function Lesson01() {
  const [pts, setPts] = useState(seedData);
  const [label, setLabel] = useState(1);
  const [w, setW] = useState([0.1, -0.2, 0]);
  const [running, setRunning] = useState(false);
  const [info, setInfo] = useState({ updates: 0, wrong: pts.length });
  const timer = useRef(null);

  const wrongOf = (ww) => pts.filter((p) => sign(ww[0] * p.x + ww[1] * p.y + ww[2]) !== p.label);

  const step = () => {
    setW((ww) => {
      const bad = wrongOf(ww);
      setInfo((o) => ({ updates: o.updates + (bad.length ? 1 : 0), wrong: bad.length }));
      if (!bad.length) { setRunning(false); return ww; }
      const p = bad[0];
      return [ww[0] + p.label * p.x, ww[1] + p.label * p.y, ww[2] + p.label];
    });
  };

  useEffect(() => {
    if (!running) { clearInterval(timer.current); return; }
    timer.current = setInterval(step, 260);
    return () => clearInterval(timer.current);
  }, [running, pts]);

  const classify = (x, y) => sign(w[0] * x + w[1] * y + w[2]);
  const line = boundaryLine(w);
  const wrongNow = wrongOf(w);

  return (
    <div className="space-y-4">
      <Lead>
        <p>learning starts from data, not from a rule we wrote. here each point already carries a label (orange is +1, blue is -1), and the learner only sees the dots. its job: find a line that puts the two colours on opposite sides.</p>
        <p>the perceptron learning algorithm does this by nudging. it walks the points, and every time it gets one wrong it tilts its line toward that point. drag the dots or add your own, then press run and watch it settle.</p>
      </Lead>
      <div className="flex flex-col gap-4 lg:flex-row">
        <Plane points={pts} setPoints={setPts} classify={classify} lines={[{ ...line, color: 'var(--color-glow)', width: 2 }]} addLabel={label} />
        <div className="flex-1 space-y-3">
          <Controls>
            <Btn on={running} onClick={() => setRunning((v) => !v)}>{running ? 'pause' : 'run pla'}</Btn>
            <Btn onClick={() => { setRunning(false); step(); }}>step</Btn>
            <Btn onClick={() => { setRunning(false); setW([0.1, -0.2, 0]); setInfo({ updates: 0, wrong: pts.length }); }}>reset line</Btn>
            <Btn onClick={() => { setRunning(false); setPts(seedData()); setW([0.1, -0.2, 0]); setInfo({ updates: 0, wrong: pts.length }); }}>new data</Btn>
          </Controls>
          <Controls>
            <span className="text-xs text-dim">click empty space to add a</span>
            <Btn on={label === 1} onClick={() => setLabel(1)}>+1 point</Btn>
            <Btn on={label === -1} onClick={() => setLabel(-1)}>-1 point</Btn>
            <span className="text-[11px] text-dim">right-click a point to remove it</span>
          </Controls>
          <div className="grid grid-cols-2 gap-2">
            <Stat label="nudges so far" value={info.updates} tone="glow" />
            <Stat label="points wrong now" value={wrongNow.length} tone="warm" />
          </div>
          <p className="text-xs leading-relaxed text-slate-400">
            if the two colours can be separated by a line, pla always finds one. if you mix them so no line works, it never settles, which is exactly the problem lecture 3 fixes.
          </p>
        </div>
      </div>
      <Slides n={1} />
    </div>
  );
}
